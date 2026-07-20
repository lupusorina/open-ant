import os
import sys
import json
import copy
import argparse
import subprocess
import csv
import time
import math
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Callable, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist

from skrl.envs.wrappers.torch import wrap_env
from skrl.utils import set_seed
try:
    from .buffer_acme import NStepReplayBufferSamples, ReplayBuffer  # imported as package
except ImportError:
    from buffer_acme import NStepReplayBufferSamples, ReplayBuffer   # run standalone

import gymnasium as gym
from gymnasium.vector import AutoresetMode
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import make_ant_env, ForwardTask, BackAndForthTask
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from reward import RewardTracker

def arr_to_str(x):
    if isinstance(x, np.ndarray):
        return "[" + " ".join(map(str, x.tolist())) + "]"
    return x
# import crane
# from crane import tile_images, save_video

_MPO_FLOAT_EPSILON = 1e-8

class Actor(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 act_low: np.ndarray,
                 act_high: np.ndarray,
                 hidden_dims: List[int] = [256, 256, 256],
                 use_layer_norm: bool = False,
                 min_scale: float = 1e-6,
                 init_scale: float = 0.3):
        super().__init__()
        self.min_scale = min_scale
        self.init_scale = init_scale

        self.register_buffer("action_low",torch.as_tensor(act_low, dtype=torch.float32))
        self.register_buffer("action_high",torch.as_tensor(act_high, dtype=torch.float32))
        
        layers = []
        prev = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h
        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, act_dim)
        self.log_sigma_head = nn.Linear(prev, act_dim)

       # self._softplus_bias = float(np.log(np.exp(init_scale - min_scale) - 1.0))

        self.apply(self._init_weights)
        small_std = math.sqrt(1e-4 / prev)
        with torch.no_grad():
            nn.init.normal_(
                self.mu_head.weight,
                mean=0.0,
                std=small_std
            )
            nn.init.zeros_(self.mu_head.bias)
            nn.init.normal_(
                self.log_sigma_head.weight,
                mean=0.0,
                std=small_std,
            )
            nn.init.zeros_(self.log_sigma_head.bias)
            #self.log_sigma_head.bias.fill_(self._softplus_bias)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> dist.Independent:
        logits = self.net(x)
        mu = self.action_low + (self.action_high - self.action_low) * torch.sigmoid(self.mu_head(logits))
        #sigma = F.softplus(self.log_sigma_head(logits)) + self.min_scale
        
        raw_scale = self.log_sigma_head(logits)
        softplus_zero = F.softplus(
        torch.zeros((), dtype=raw_scale.dtype, device=raw_scale.device)
        )
        sigma = (
            F.softplus(raw_scale)
            * self.init_scale
            / softplus_zero
            + self.min_scale
        )
        return dist.Independent(dist.Normal(mu, sigma), 1)

# critic is only allowed to output probabilities on fixed grid of return vals
# Z = [logit(return =vmin), logit(return = vmax)], bins = num atoms 
def l2_project(Zp: torch.Tensor, P: torch.Tensor, Zq: torch.Tensor) -> torch.Tensor:
    """Project distribution (Zp, P) onto support Zq under the L2 metric over CDFs.

    Zp: (B, Kp) source support
    P:  (B, Kp) source probabilities
    Zq: (Kq,) target support
    returns: (B, Kq)
    """
    #Extracts vmin and vmax and construct helper tensors from Zq
    vmin, vmax = Zq[0], Zq[-1]
    d_pos = torch.cat([Zq, vmin[None]], dim=0)[1:]
    d_neg = torch.cat([vmax[None], Zq], dim=0)[:-1]

    # Clips Zp to be in new support range (vmin, vmax).
    clipped_zp = torch.clamp(Zp, vmin, vmax)[:, None, :]
    clipped_zq = Zq[None, :, None]

    # Gets the distance between atom values in support
    d_pos = (d_pos - Zq)[None, :, None]
    d_neg = (Zq - d_neg)[None, :, None]

    delta_qp = clipped_zp - clipped_zq
    d_sign = (delta_qp >= 0.0).to(P.dtype)

    delta_hat = (
        d_sign * delta_qp / d_pos
        - (1.0 - d_sign) * delta_qp / d_neg
    )
    P = P[:, None, :]
    # returns the L2 projection of (Zp, P) onto Zq. --> note it's Zq!! not Zp!!

    return torch.sum(torch.clamp(1.0 - delta_hat, 0.0, 1.0) * P, dim=2)

class DiscreteValuedDistribution:
    # categorical distribution. values = fixed return values / atoms ranging from vmin to vmax
    # logits = raw neural net output for each atom
    def __init__(self, values: torch.Tensor, logits: torch.Tensor):
        self.values = values
        self.logits = logits

# softmax convert logits into probabilities
    @property
    def probs(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)

    # mean of the distrib = the expected Q value. Q(s, a)= E[Z(s,a)]
    def mean(self) -> torch.Tensor:
        return (self.probs * self.values).sum(dim=-1, keepdim=True)


# computes categorical distributional loss
# q_t.values = the target atoms (the return bins, i.e. -150, -145, .. +150)
# target for current = Z_target_t = r_t + gamma * Z_target(S_{t+1}, a'), where a' is from target policy
# p_t is the probabilities for the logit values, which here, each logit val correspond to a target atom from q_t
def categorical(
    q_tm1: DiscreteValuedDistribution,
    r_t: torch.Tensor,
    d_t: torch.Tensor,
    q_t: DiscreteValuedDistribution,
) -> torch.Tensor:
    """Categorical distributional loss. Returns per-sample loss (B,)."""
    values = q_t.values

    # r_t.reshape change reward shape from (B,) to (B,1)
    # since values have shape (k,), where k = num atoms
    # this makes every atom get the reward and d_t bellman shift.
    z_t = r_t.reshape(-1, 1) + d_t.reshape(-1, 1) * values  # shape (B,K)
    # in order to do l2_project, the atoms need probability mass p_t attached
    # to each atom, not logits. 
    p_t = torch.softmax(q_t.logits, dim=-1)

    # need to project shifted distrib back onto atom values
    target = l2_project(z_t, p_t, values).detach()

    # pytorch equivalent of tf.nn.softmax_cross_entropy_with_logits
    # loss = (-labels * torch.nn.functional.log_softmax(logits, dim=-1)).sum(dim=-1)
    log_p_tm1 = torch.log_softmax(q_tm1.logits, dim=-1)
    
    # target = the projected (Zp, P) onto Zq - Z_w'(St, At)
    # q_tm1 logits = online critic's prediction.
    # cross entropy formula
    return -(target * log_p_tm1).sum(dim=-1)

def variance_scaling_init_(
    tensor: torch.Tensor,
    scale: float = 1.0,
    mode: str = "fan_in",
    distribution: str = "truncated_normal"
) -> torch.Tensor:
    distribution = distribution.lower()
    if distribution not in {"truncated_normal","untruncated_normal","uniform"}:
        raise ValueError("distribution must be 'truncated_normal','untruncated_normal', or 'uniform'")
    if mode not in {"fan_in", "fan_out", "fan_avg"}:
        raise ValueError("mode must be 'fan_in', 'fan_out', or 'fan_avg'")
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        n = fan_in
    elif mode == "fan_out":
        n = fan_out
    else:
        n = (fan_in + fan_out) / 2.0
    variance = scale / max(1.0, n)
    with torch.no_grad():
        if distribution == "truncated_normal":
            correction = 0.87962566103423978
            std = math.sqrt(variance) / correction
            return nn.init.trunc_normal_(tensor,mean=0.0,std=std,a=-2.0 * std,b=2.0 * std)
        if distribution == "untruncated_normal":
            std = math.sqrt(variance)
            return tensor.normal_(mean=0.0, std=std)
        if distribution == "uniform":
            limit = math.sqrt(3.0 * variance)
            return tensor.uniform_(-limit, limit)
class AcmeLayerNormMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        layer_sizes: Sequence[int],
        w_init: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        activation: type[nn.Module] = nn.ELU,
        activate_final: bool = False,
    ) -> None:
        super().__init__()

        if len(layer_sizes) == 0:
            raise ValueError("layer_sizes must contain at least one size")

        self._w_init = w_init
        # self._activate_final = activate_final KEEP OR DELETE THSI LINE?

        layers: list[nn.Module] = []
        first_linear = nn.Linear(in_features=input_size,out_features=layer_sizes[0])
        self._initialize_linear(first_linear)

        layers.extend([
            first_linear,
            nn.LayerNorm(
                # shape of final dimension being normalized is 512 - output dim of the 1st linear layer
                normalized_shape=layer_sizes[0],
                elementwise_affine=True,
            ),nn.Tanh()])
        previous_size = layer_sizes[0]

        for index, output_size in enumerate(layer_sizes[1:]):
            linear = nn.Linear(in_features=previous_size,out_features=output_size)
            self._initialize_linear(linear)
            layers.append(linear)
            is_final_layer = (index == len(layer_sizes[1:]) - 1)
            
            if not is_final_layer or activate_final:
                layers.append(activation())
            previous_size = output_size
        self._network = nn.Sequential(*layers)

    def _initialize_linear(self, linear: nn.Linear) -> None:
        if self._w_init is None:
            variance_scaling_init_(linear.weight,scale=0.333,mode="fan_out",distribution="uniform")
        else:
            self._w_init(linear.weight)
        nn.init.zeros_(linear.bias)
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self._network(observations)

class DiscreteValuedHead(nn.Module):
    ''' maps hidden features to logits over a fixed support of return atoms,
    then returns a discretevalueddistribution '''
    def __init__(
        self,
        input_size: int,
        vmin: float,
        vmax: float,
        num_atoms: int,
        w_init: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        b_init: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> None:
        super().__init__()

        if vmax <= vmin:
            raise ValueError(f"vmax must be greater than vmin, got vmin={vmin}, vmax={vmax}")

        self.num_atoms = num_atoms
        self.register_buffer("atom_values",torch.linspace(vmin,vmax,num_atoms,dtype=torch.float32))
        
        self._distributional_layer = nn.Linear(in_features=input_size,out_features=num_atoms)
        # edit the weights and biases to match snt.Linear's default: bias = 0. 
        # weight = truncated random normal values with stddev = 1 / sqrt(input-feature-size)
        # truncated at: no more than 2 stddevs from the mean.
        with torch.no_grad():
            if w_init is None:
                std = 1.0 / math.sqrt(input_size)
                nn.init.trunc_normal_(
                    self._distributional_layer.weight,
                    mean=0.0,
                    std=std,
                    a=-2.0 * std,
                    b=2.0 * std,
                )
            else:
                w_init(self._distributional_layer.weight)

            if self._distributional_layer.bias is not None:
                if b_init is None:
                    nn.init.zeros_(self._distributional_layer.bias)
                else:
                    b_init(self._distributional_layer.bias)

    def forward(self,inputs: torch.Tensor) -> DiscreteValuedDistribution:
        # Input:(B, input_size)
        # Output logits:(B, num_atoms)
        logits = self._distributional_layer(inputs)

        # Cast support to match logits, equivalent to:
        # tf.cast(self._values, logits.dtype)
        atom_values = self.atom_values.to(dtype=logits.dtype,device=logits.device)
        return DiscreteValuedDistribution(values=atom_values,logits=logits)

class MultivariateNormalDiagHead(nn.Module):
    """Module that produces a multivariate normal distribution."""

    def __init__(
        self,
        input_size: int,
        num_dimensions: int,
        init_scale: float = 0.3,
        min_scale: float = 1e-6,
        tanh_mean: bool = False,
        fixed_scale: bool = False,
        use_independent: bool = True,
    ) -> None:
        super().__init__()
        self._init_scale = float(init_scale)
        self._min_scale = float(min_scale)
        self._tanh_mean = tanh_mean
        self._fixed_scale = fixed_scale
        self._use_independent = use_independent

        self._mean_layer = nn.Linear(in_features=input_size, out_features=num_dimensions)
        self._initialize_linear(self._mean_layer)
        if not fixed_scale:
            self._scale_layer = nn.Linear(in_features=input_size, out_features=num_dimensions)
            self._initialize_linear(self._scale_layer)
    @staticmethod
    def _initialize_linear(linear: nn.Linear) -> None:
        variance_scaling_init_(linear.weight,scale=1e-4,mode="fan_in",distribution="truncated_normal")
        nn.init.zeros_(linear.bias)
    def forward(self, inputs: torch.Tensor) -> dist.Distribution:
        mean = self._mean_layer(inputs)
        if self._fixed_scale:
            scale = torch.full_like(mean, self._init_scale)
        else:
            raw_scale = self._scale_layer(inputs)
            softplus_zero = F.softplus(torch.zeros((), dtype=raw_scale.dtype, device=raw_scale.device))
            scale = (
                F.softplus(raw_scale)
                * self._init_scale
                / softplus_zero
                + self._min_scale
            )
        if self._tanh_mean:
            mean = torch.tanh(mean)
        if self._use_independent:
            return dist.Independent(dist.Normal(loc=mean, scale=scale),reinterpreted_batch_ndims=1)

        return dist.MultivariateNormal(loc=mean,scale_tril=torch.diag_embed(scale))
  

class AcmeActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_low: np.ndarray,
        act_high: np.ndarray,
        layer_sizes: Sequence[int] = (256, 256, 256),
        init_scale: float = 0.7,   
        min_scale: float = 1e-4,    # originally 1e-6
    ) -> None:
        super().__init__()
        if len(layer_sizes) == 0:
            raise ValueError("layer_sizes must contain at least one size")
        self.register_buffer("action_low",torch.as_tensor(act_low, dtype=torch.float32))
        self.register_buffer("action_high",torch.as_tensor(act_high, dtype=torch.float32))
        
        self.torso = AcmeLayerNormMLP(
            input_size=obs_dim,
            layer_sizes=layer_sizes,
            activate_final=True,
        )
        self.policy_head = MultivariateNormalDiagHead(
            input_size=layer_sizes[-1],
            num_dimensions=act_dim,
            init_scale=init_scale,
            min_scale=min_scale,
            tanh_mean=False,
            fixed_scale=False,
            use_independent=True,
        )
    def forward(self, observations: torch.Tensor) -> dist.Distribution:
        observations = observations.reshape(observations.shape[0],-1)
        features = self.torso(observations)
        return self.policy_head(features)

class AcmeCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        layer_sizes: Sequence[int] = (512, 512, 256),
        vmin: float = -500.0,
        vmax: float = 20.0,
        num_atoms: int = 101,
    ) -> None:
        super().__init__()
        self.register_buffer("action_low",torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high",torch.as_tensor(action_high, dtype=torch.float32))
        
        self.torso = AcmeLayerNormMLP(
            input_size=obs_dim + act_dim,
            layer_sizes=layer_sizes,
            activate_final=True)

        self.distributional_head = DiscreteValuedHead(
            input_size=layer_sizes[-1],
            vmin=vmin,
            vmax=vmax,
            num_atoms=num_atoms)
    
    @property
    def atom_values(self) -> torch.Tensor:
        return self.distributional_head.atom_values
    def forward(self,observation: torch.Tensor,action: torch.Tensor) -> DiscreteValuedDistribution:
        # Clip action
        observation = observation.reshape(observation.shape[0],-1)
        action = action.reshape(action.shape[0],-1)

        action = torch.clamp(action,self.action_low,self.action_high)
        action = action.to(dtype=observation.dtype,device=observation.device)
        inputs = torch.cat([observation, action],dim=-1)

        # LayerNormMLP and distribution head
        torso_output = self.torso(inputs)  # (B, 256)
        distribution = self.distributional_head(torso_output)
        return distribution

class MPO:
    def __init__(self, args, envs, disk_folder='', run_name=None, runs_directory='runs'):
        self.args = args
        self.envs = envs
        self.device = envs.device
        print(f"[√] Using device: {self.device}")
        self.dt = args.dt

        self.disk_folder = disk_folder
        self.run_name = run_name
        self.runs_directory = runs_directory
        self.run_dir = os.path.join(disk_folder,runs_directory,run_name)
        self.weights_folder = os.path.join(self.run_dir, "weights_and_args")
        self.video_dir = os.path.join(self.run_dir,"videos",)
        self.log_dir = os.path.join(self.run_dir, "tensorboard")
        for directory in (self.run_dir,self.weights_folder,self.video_dir,self.log_dir):
            os.makedirs(directory, exist_ok=True)
        with open(os.path.join(self.weights_folder, "args.json"), 'w') as f:
            json.dump(args.__dict__, f)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        #Nvidia cuDNN library uses deterministic convolution algo; if false can use FFT based or Winograd convolutions; slower but mathematically more sound
        torch.backends.cudnn.deterministic = args.torch_deterministic 
        torch.backends.cudnn.benchmark = not args.torch_deterministic #if not deterministic, find and use fastest one

        policy_layer_sizes = tuple(args.policy_layer_sizes)
        critic_layer_sizes = tuple(args.critic_layer_sizes)

        self.obs_dim = int(np.array(self.envs.single_observation_space.shape).prod())
        self.act_dim = int(np.prod(self.envs.single_action_space.shape))
        self.envs.single_observation_space.dtype = np.float32
        self.action_low = envs.single_action_space.low.astype(np.float32)
        self.action_high = envs.single_action_space.high.astype(np.float32)
        assert np.all(np.isfinite(envs.single_action_space.low)) and np.all(np.isfinite(envs.single_action_space.high))

        self.actor = AcmeActor(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            act_low=self.action_low,
            act_high=self.action_high,
            layer_sizes=policy_layer_sizes,
            init_scale=0.3,  # min_scale, 
            min_scale=1e-6,
        ).to(self.device)

        self.actor_target = AcmeActor(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            act_low=self.action_low,
            act_high=self.action_high,
            layer_sizes=policy_layer_sizes,
            init_scale=0.3,
            min_scale=1e-6,
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        for p in self.actor_target.parameters():
            p.requires_grad = False
        
        self.critic = AcmeCritic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            action_low=self.action_low,
            action_high=self.action_high,
            layer_sizes=critic_layer_sizes,
            vmin=args.vmin,
            vmax=args.vmax,
            num_atoms=args.num_atoms,
        ).to(self.device)

        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.policy_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.q_lr)

        self.policy_learning_starts = args.policy_learning_starts

        self._init_log_eta = args.init_log_temperature
        self._init_log_alpha_mean = args.init_log_alpha_mean
        self._init_log_alpha_stddev = args.init_log_alpha_stddev

        # Whether to penalize out-of-bound actions via MO-MPO and its corresponding
        # constraint threshold.
        self._action_penalization = args.action_penalization
        self._epsilon_penalty = args.epsilon_penalty

        self.log_eta = nn.Parameter(
            torch.full((1,), self._init_log_eta, dtype=torch.float32, device=self.device))
        self.log_alpha_mean = nn.Parameter(
            torch.full((self.act_dim,), self._init_log_alpha_mean, dtype=torch.float32, device=self.device))
        self.log_alpha_stddev = nn.Parameter(
            torch.full((self.act_dim,), self._init_log_alpha_stddev, dtype=torch.float32, device=self.device))
        self.log_penalty_eta = nn.Parameter(torch.full((1,),self._init_log_eta,dtype=torch.float32,device=self.device))
        self.dual_optimizer = optim.Adam(
            [
                self.log_eta,
                self.log_alpha_mean,
                self.log_alpha_stddev,
                self.log_penalty_eta,
            ],
            lr=args.dual_lr,
        )

        # self.dual_temp_optimizer = optim.Adam([self.log_eta], lr=args.dual_lr)

        # Scalar Lagrange multipliers for M-step KL constraints.

        #self._uniform_log_prob = -float(np.sum(np.log(self.action_high - self.action_low)))

        self.rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            self.device,
            n_envs=args.num_envs,
            handle_timeout_termination=False,
        )

        self.batch_size = self.args.batch_size
        self.trajectory_length = self.args.td_horizon

        self.samples_per_insert = self.args.samples_per_insert
        self.learner_update_budget = 0.0

        self.global_step = 0
        self.learner_step = 0
        self.target_policy_update_period = self.args.target_policy_update_period
        self.target_critic_update_period = self.args.target_critic_update_period

        self.obs = None
        self.last_actions = None
        #self.last_log_probs = None

        # Logging state.
        self.start_time = None
        self.reward_tracker = None
        self.csv_file_info = None
        self.csv_file_agent_vars = None
        self.writer_info = None
        self.writer_agent_vars = None
        self.keys_info = None
        self.keys_agent_vars = [
            "critic_loss","policy_loss","dual_alpha_mean","dual_alpha_stddev",
            "dual_temperature","loss_alpha","loss_temperature","loss_policy_cross_entropy",
            "loss_kl_penalty","kl_q_rel","kl_mean_rel","kl_stddev_rel","kl_mean_rel_min",
            "kl_mean_rel_max","kl_stddev_rel_min","kl_stddev_rel_max","q_min",
            "q_max","pi_stddev_min","pi_stddev_max","pi_stddev_cond","utd",
            "SPS","average_reward_per_second","reward"]

        for idx in range(self.act_dim):
            self.keys_agent_vars.extend(
                [
                    f"dual_alpha_mean_{idx}",
                    f"dual_alpha_stddev_{idx}",
                    f"kl_mean_rel_{idx}",
                    f"kl_stddev_rel_{idx}",
                    f"pi_stddev_{idx}",
                ]
            )
        self.info_log_buffer = []
        self.agent_vars_buffer = []

 
        self._epsilon = args.epsilon_eta
        self._epsilon_mean = args.epsilon_mu_kl
        self._epsilon_stddev = args.epsilon_sigma_kl
        
        # a variable that tracks when reset happens - the transition after done = true is
        # when transitioning from terminal state to reset state - this is when pending_autoreset=1,
        # to inform agent to not use this transition during training.
        self.pending_autoreset = torch.zeros(self.envs.num_envs,dtype=torch.bool,device=self.device)
        print("Agent initialized and compiled")
    
    def _random_action(self):
        low, high = self.actor.action_low, self.actor.action_high
        rand_actions = low + (high - low) * torch.rand(self.envs.num_envs, self.act_dim, device=self.device)
        # self.last_actions = rand_actions
        # self.last_log_probs = torch.full((self.envs.num_envs, 1), self._uniform_log_prob,dtype=torch.float32, device=self.device)
        return rand_actions
    
    def get_action(self, obs, evaluate=False):
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        if self.obs is None:
            self.obs = obs
            return self._random_action()

        if evaluate or self.global_step > self.args.learning_starts:
            with torch.no_grad():
                d = self.actor.forward(self.obs)
                action = d.mean if evaluate else d.sample()
                actions = action
                #actions = torch.clamp(action, self.actor.action_low, self.actor.action_high) #mjx clipos internally but overflowed action stored in buffer --> inconsistency
                actions = actions.squeeze(1)     # (n_envs, act_dim)
                
                log_probs = d.log_prob(actions).unsqueeze(-1)
                self.last_actions = actions
                self.last_log_probs = log_probs
                return actions
        return self._random_action()
    
    def agent_step(self, next_obs, actions, rewards, terminations, truncations, infos):
        terminations = terminations.bool()
        truncations = truncations.bool()

        # Episode boundary: true termination OR time-limit truncation
        dones = terminations | truncations

        # timeout/truncation, this should NOT kill bootstrapping
        timeouts = truncations & (~terminations)
        
        if "autoreset" in infos:
            autoreset_now = torch.as_tensor(infos["autoreset"],dtype=torch.bool,device=self.device)
        else:
            autoreset_now = self.pending_autoreset
        valid = ~autoreset_now

        num_inserts = int(valid.sum().item())

        self.rb.add(
            self.obs,
            next_obs,
            actions,
            rewards,
            dones,
            [{}] * self.envs.num_envs,
           # behavior_log_prob=self.last_log_probs,
            truncation=timeouts,
            valid=valid)
        #___________________________________________________________________________________________________
        self.global_step += 1
        self.obs = next_obs
        self.pending_autoreset = dones.detach().clone()
        metrics = None

        if self.global_step > self.args.learning_starts and self.rb.size() >= self.batch_size:
            # Count actual environment transitions inserted.
            # Reset-only rows have valid=False and should not count.
    
            self.learner_update_budget += (self.samples_per_insert * num_inserts / self.batch_size)
            num_updates = int(self.learner_update_budget)
            if num_updates > 0:
                self.learner_update_budget -= (num_updates)
                results = [self._learn() for _ in range(num_updates)]
                keys = results[0].keys()
                metrics = {k: sum(r[k] for r in results) / len(results) for k in keys}

                # Actual learner updates performed during this vector-environment step.
                metrics["utd"] = num_updates
        return metrics
    
    def agent_step_eval(self, next_obs):
        self.global_step += 1
        self.obs = next_obs

    def agent_step_no_learn(self, next_obs, actions, rewards, terminations, truncations, infos):
        """Buffer add + step count only. Call learn_step() from a separate timer."""
        #___________________________________________________________________________________________________
        terminations = terminations.bool()
        truncations = truncations.bool()

        dones = terminations | truncations
        timeouts = truncations & (~terminations)

        if "autoreset" in infos:
            autoreset_now = torch.as_tensor(infos["autoreset"],dtype=torch.bool,device=self.device)
        else:
            autoreset_now = self.pending_autoreset
        valid = ~autoreset_now

        self.rb.add(
            self.obs,
            next_obs,
            actions,
            rewards,
            dones,
            [{}] * self.envs.num_envs,
            #behavior_log_prob=self.last_log_probs,
            truncation=timeouts,
            valid=valid)
        self.pending_autoreset = dones.detach().clone()
        self.global_step += 1
        self.obs = next_obs
    
    def learn_step(self):
        """Run one round of UTD critic/actor updates. Returns metrics dict or None."""
        if self.global_step <= self.args.learning_starts:
            return None
        if self.rb.size() < self.batch_size:
            return None
        results = [self._learn() for _ in range(self.args.utd)]
        keys = results[0].keys()
        metrics = {k: sum(r[k] for r in results) / len(results) for k in keys}
        metrics['utd'] = self.args.utd
        return metrics

 

    def _learn(self):
        with torch.no_grad():
            if (self.learner_step % self.target_policy_update_period== 0):
                self.actor_target.load_state_dict(self.actor.state_dict())

            if (self.learner_step % self.target_critic_update_period == 0):
                self.target_critic.load_state_dict(self.critic.state_dict())
        self.learner_step += 1
        
        if self.global_step >= self.args.learning_starts + self.policy_learning_starts:
            self.args.decouple_q_learning = False
        with torch.no_grad():
            #1. sample 1 replay batch
            data = self.rb.sample_nstep(
                batch_size=self.batch_size,
                n_step=self.trajectory_length,
                gamma=self.args.gamma,
            )
            B = self.batch_size
            # the same samples are used to average the target categorical 
            # critic distirbutions & obtain scalar Q vals for MPO.
            N = self.args.sample_action_num
            D = self.act_dim

            # this is the current, s_t
            s_tm1 = data.observations
            a_tm1 = data.actions

            # here, the .next_obs being accessed is actually s_t+n
            s_t = data.next_observations

            #the collapsed n-step discounted return
            r_t = data.rewards
            # bootstrapping coefficient
            discount_t = self.args.gamma * data.discounts
            
            target_policy = self.actor_target.forward(s_t)
            # Shape: (N, B, D)
            sampled_actions = target_policy.sample((N,))

            # sampled_actions = torch.clamp(
            #     sampled_actions,
            #     self.actor_target.action_low,
            #     self.actor_target.action_high,
            # )
            tiled_states = (
                s_t.unsqueeze(0)
                .expand(N, -1, -1)
                .reshape(N * B, self.obs_dim)
            )

            flat_actions = sampled_actions.reshape(N * B, D)

            # Distributional target critic for every sampled action.
            sampled_q_t_distribution = self.target_critic(
                tiled_states,
                flat_actions,
            )
            num_atoms = sampled_q_t_distribution.logits.shape[-1]
            sampled_q_logits = sampled_q_t_distribution.logits.reshape(N,B,num_atoms,)
            sampled_q_logprobs = torch.log_softmax(sampled_q_logits,dim=-1)
            averaged_target_logits = torch.logsumexp(sampled_q_logprobs, dim=0)

            target_q_distribution = DiscreteValuedDistribution(
                values=self.target_critic.atom_values,
                logits=averaged_target_logits,
            )
            q_values = (sampled_q_t_distribution.mean().reshape(N, B))

        online_policy = self.actor.forward(s_t)
        online_q_distribution = self.critic(s_tm1, a_tm1)
        critic_loss = categorical(
            q_tm1=online_q_distribution,r_t=r_t,d_t=discount_t,q_t=target_q_distribution
        ).mean()

        # compute MPO policy-related losses
        scalar_dtype = q_values.dtype
        dual_variable_shape = D

        # Project dual variables to ensure they stay positive.
       
        with torch.no_grad():
            self.log_eta.clamp_(min=-18.0)
            self.log_alpha_mean.clamp_(min=-18.0)
            self.log_alpha_stddev.clamp_(min=-18.0)

        # Transform dual variables from log-space.
        # using softplus instead of exponential for numerical stability.
        temperature = F.softplus(self.log_eta) + _MPO_FLOAT_EPSILON
        alpha_mean = F.softplus(self.log_alpha_mean) + _MPO_FLOAT_EPSILON
        alpha_stddev = F.softplus(self.log_alpha_stddev) + _MPO_FLOAT_EPSILON

        # Get online and target means and stddevs in preparation for decomposition.
        target_mu = target_policy.base_dist.loc
        target_sigma = target_policy.base_dist.scale
        online_mu = online_policy.base_dist.loc
        online_sigma = online_policy.base_dist.scale

        # Compute normalized importance weights, used to compute expectations with
        # respect to the non-parametric policy; and the temperature loss, used to
        # adapt the tempering of Q-values.
        normalized_weights, loss_temperature = compute_weights_and_temperature_loss(
            q_values, self._epsilon, temperature)
        
        # Only needed for diagnostics: Compute estimated actualized KL between the
        # non-parametric and current target policies. 
        kl_nonparametric = compute_nonparametric_kl_from_normalized_weights(
            normalized_weights)
        
        if self._action_penalization:
            # Project and transform action penalization temperature.
            with torch.no_grad():
                self.log_penalty_eta.clamp_(min=-18.0)
            penalty_temperature = F.softplus(self.log_penalty_eta) + _MPO_FLOAT_EPSILON
           
            # Compute action penalization cost.
            # Note: the cost is zero in [-1, 1] and quadratic beyond.
            diff_out_of_bound = sampled_actions - torch.clamp(sampled_actions,self.actor_target.action_low,self.actor_target.action_high)
            
            cost_out_of_bound = -torch.linalg.vector_norm(diff_out_of_bound, dim=-1)

            penalty_normalized_weights, loss_penalty_temperature = compute_weights_and_temperature_loss(
                cost_out_of_bound, self._epsilon_penalty, penalty_temperature)

            # Only needed for diagnostics: Compute estimated actualized KL between the
            # non-parametric and current target policies.
            penalty_kl_nonparametric = compute_nonparametric_kl_from_normalized_weights(
                penalty_normalized_weights)

            # Combine normalized weights.
            normalized_weights += penalty_normalized_weights  # pyrefly: ignore[unsupported-operation]
            loss_temperature += loss_penalty_temperature  # pyrefly: ignore[unsupported-operation]
    
        # Decompose the online policy into fixed-mean & fixed-stddev distributions
        #  https://arxiv.org/pdf/1812.02256.pdf.
        fixed_stddev_distribution = dist.Independent(dist.Normal(online_mu, target_sigma), 1)
        fixed_mean_distribution = dist.Independent(dist.Normal(target_mu, online_sigma), 1)

            # Compute the decomposed policy losses.
        loss_policy_mean = compute_cross_entropy_loss(
            sampled_actions, normalized_weights, fixed_stddev_distribution)
        loss_policy_stddev = compute_cross_entropy_loss(
            sampled_actions, normalized_weights, fixed_mean_distribution)

        kl_mean = torch.distributions.kl_divergence(target_policy.base_dist,
            fixed_stddev_distribution.base_dist)  # Shape [B, D].
        kl_stddev = torch.distributions.kl_divergence(target_policy.base_dist,
            fixed_mean_distribution.base_dist)  # Shape [B, D]
        
        # Compute the alpha-weighted KL-penalty and dual losses to adapt the alphas.
        loss_kl_mean, loss_alpha_mean = compute_parametric_kl_penalty_and_dual_loss(
            kl_mean, alpha_mean, self._epsilon_mean)
        loss_kl_stddev, loss_alpha_stddev = compute_parametric_kl_penalty_and_dual_loss(
            kl_stddev, alpha_stddev, self._epsilon_stddev)
        
        # Combine losses.
        loss_policy = loss_policy_mean + loss_policy_stddev  # pyrefly: ignore[unsupported-operation]
        loss_kl_penalty = loss_kl_mean + loss_kl_stddev  # pyrefly: ignore[unsupported-operation]
        loss_dual = loss_alpha_mean + loss_alpha_stddev + loss_temperature  # pyrefly: ignore[unsupported-operation]
        total_mpo_loss = (loss_policy + loss_kl_penalty + loss_dual)
            # DO i need to do requires grad=True here or anywhere else..
        # critic_trainable_variables = self.__critic_network.trainable_variables

        # record stats
        mean_kl_mean_per_dim = kl_mean.mean(dim=0)
        mean_kl_stddev_per_dim = kl_stddev.mean(dim=0)

        pi_stddev = online_sigma
        pi_stddev_min_per_state = pi_stddev.min(dim=-1).values
        pi_stddev_max_per_state = pi_stddev.max(dim=-1).values

        policy_stats = {
            "dual_alpha_mean": alpha_mean.detach().mean().item(),
            "dual_alpha_stddev": alpha_stddev.detach().mean().item(),
            "dual_temperature": temperature.detach().mean().item(),

            # ACME's loss_policy statistic is the complete MPO loss.
            "policy_loss": total_mpo_loss.detach().mean().item(),
            "loss_alpha": (
                loss_alpha_mean.detach() + loss_alpha_stddev.detach()).mean().item(),
            "loss_temperature": loss_temperature.detach().mean().item(),
            "loss_policy_cross_entropy": loss_policy.detach().mean().item(),
            "loss_kl_penalty": loss_kl_penalty.detach().mean().item(),

            # Relative KL measurements.
            "kl_q_rel": (kl_nonparametric.detach().mean() / self._epsilon).item(),
            "kl_mean_rel": (kl_mean.detach().mean() / self._epsilon_mean).item(),
            "kl_stddev_rel": (kl_stddev.detach().mean() / self._epsilon_stddev).item(),

            # Per-dimension constraint range.
            "kl_mean_rel_min": (mean_kl_mean_per_dim.detach().min() / self._epsilon_mean).item(),
            "kl_mean_rel_max": (mean_kl_mean_per_dim.detach().max() / self._epsilon_mean).item(),
            "kl_stddev_rel_min": (mean_kl_stddev_per_dim.detach().min()/ self._epsilon_stddev).item(),
            "kl_stddev_rel_max": (mean_kl_stddev_per_dim.detach().max()/ self._epsilon_stddev).item(),

            # Q-values: min/max over actions for each state, then average states.
            "q_min": (q_values.detach().min(dim=0).values.mean()).item(),

            "q_max": (q_values.detach().max(dim=0).values.mean()).item(),

            # Policy exploration.
            "pi_stddev_min": pi_stddev_min_per_state.detach().mean().item(),
            "pi_stddev_max": pi_stddev_max_per_state.detach().mean().item(),

            "pi_stddev_cond": (pi_stddev_max_per_state/ pi_stddev_min_per_state.clamp_min(_MPO_FLOAT_EPSILON)).detach().mean().item(),
        }
        if self._action_penalization:
            policy_stats["penalty_kl_q_rel"] = (
                penalty_kl_nonparametric.detach().mean() / self._epsilon_penalty).item()  # pyrefly: ignore[unbound-name]

        # per dim vals
        for j in range(self.act_dim):
            policy_stats[f"dual_alpha_mean_{j}"] = (
                alpha_mean[j].detach().item()
            )
            policy_stats[f"dual_alpha_stddev_{j}"] = (
                alpha_stddev[j].detach().item()
            )
            policy_stats[f"kl_mean_rel_{j}"] = (
                mean_kl_mean_per_dim[j].detach() / self._epsilon_mean
            ).item()
            policy_stats[f"kl_stddev_rel_{j}"] = (
                mean_kl_stddev_per_dim[j].detach()
                / self._epsilon_stddev
            ).item()
            policy_stats[f"pi_stddev_{j}"] = (
                pi_stddev[:, j].detach().mean().item()
            )

        # Compute gradients 

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.dual_optimizer.zero_grad()

        critic_loss.backward()   # retain because actor loss shares obs/critic graph? 
        total_mpo_loss.backward()               # policy + KL penalty + dual, one backward

        # clip
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)

        self.critic_optimizer.step()
        self.actor_optimizer.step()
        self.dual_optimizer.step()

        # Losses to track.
        fetches = {
            'critic_loss': critic_loss.detach().item(),
            'policy_loss': total_mpo_loss.detach().item(),

            **policy_stats,
        }
        return fetches
    
    def initialize_logging(self, info):
        self.start_time = time.time()

        log_dir = os.path.join(self.disk_folder, self.runs_directory, self.run_name)
        self.csv_file_info = open(os.path.join(log_dir, "info_logs.csv"), "w", newline="")
        self.keys_info = [k for k in info.keys() if not (k.startswith("bodies") or k.startswith("_"))]
        self.writer_info = csv.DictWriter(self.csv_file_info, fieldnames=["step"] + self.keys_info)
        self.writer_info.writeheader()

        self.reward_tracker = RewardTracker(
            env_dt=self.args.dt,
            env_id=self.args.env_id,
            log_folder=log_dir,
            time_window=120.0,
        )

        self.csv_file_agent_vars = open(os.path.join(log_dir, "performance_variables.csv"), "w", newline="")
        self.writer_agent_vars = csv.DictWriter(self.csv_file_agent_vars, fieldnames=["step"] + self.keys_agent_vars)
        self.writer_agent_vars.writeheader()

        self.info_log_buffer = []
        self.agent_vars_buffer = []

    def log_step(self, global_step, infos, rewards, metrics=None):
        if self.writer_info is None:
            return

        self.reward_tracker.update(infos['original_reward'][0])
        self.reward_tracker.log()

        if global_step % self.args.log_every_n_steps != 0:
            return

        row = {"step": global_step}
        for k in self.keys_info:
            if k in infos:
                row[k] = arr_to_str(infos[k][0])
        self.info_log_buffer.append(row)


        if metrics is not None:
            elapsed_time = max(time.time() - self.start_time,1e-8)
            # global_step counts vector-environment steps, so multiply
            # by num_envs to obtain total environment transitions.
            sps = int(global_step * self.envs.num_envs / elapsed_time)
            reward_value = float(rewards.reshape(-1)[0].item())
            agent_vars_row = {
                "step": global_step,
                "critic_loss": metrics.get("critic_loss"),
                "policy_loss": metrics.get("policy_loss"),
                "dual_alpha_mean": metrics.get("dual_alpha_mean"),
                "dual_alpha_stddev": metrics.get("dual_alpha_stddev"),
                "dual_temperature": metrics.get("dual_temperature"),
                "loss_alpha": metrics.get("loss_alpha"),
                "loss_temperature": metrics.get("loss_temperature"),
                "loss_policy_cross_entropy": metrics.get("loss_policy_cross_entropy"),
                "loss_kl_penalty": metrics.get("loss_kl_penalty"),
                "kl_q_rel": metrics.get("kl_q_rel"),
                "kl_mean_rel": metrics.get("kl_mean_rel"),
                "kl_stddev_rel": metrics.get("kl_stddev_rel"),
                "kl_mean_rel_min": metrics.get("kl_mean_rel_min"),
                "kl_mean_rel_max": metrics.get("kl_mean_rel_max"),
                "kl_stddev_rel_min": metrics.get("kl_stddev_rel_min"),
                "kl_stddev_rel_max": metrics.get("kl_stddev_rel_max"),
                "q_min": metrics.get("q_min"),
                "q_max": metrics.get("q_max"),
                "pi_stddev_min": metrics.get("pi_stddev_min"),
                "pi_stddev_max": metrics.get("pi_stddev_max"),
                "pi_stddev_cond": metrics.get("pi_stddev_cond"),
                "utd": metrics.get("utd"),
                "SPS": sps,
                "average_reward_per_second": (self.reward_tracker.average_reward_per_second),
                "reward": reward_value,
            }
            for idx in range(self.act_dim):
                agent_vars_row[f"dual_alpha_mean_{idx}"] = metrics.get(f"dual_alpha_mean_{idx}")
                agent_vars_row[f"dual_alpha_stddev_{idx}"] = metrics.get(f"dual_alpha_stddev_{idx}")
                agent_vars_row[f"kl_mean_rel_{idx}"] = metrics.get(f"kl_mean_rel_{idx}")
                agent_vars_row[f"kl_stddev_rel_{idx}"] = metrics.get(f"kl_stddev_rel_{idx}")
                agent_vars_row[f"pi_stddev_{idx}"] = metrics.get(f"pi_stddev_{idx}")
            
            self.agent_vars_buffer.append(agent_vars_row)

        if global_step % self.args.save_every_n_steps == 0:
            for row in self.info_log_buffer:
                self.writer_info.writerow(row)
            self.csv_file_info.flush()
            self.info_log_buffer = []

            for row in self.agent_vars_buffer:
                self.writer_agent_vars.writerow(row)
            self.csv_file_agent_vars.flush()
            self.agent_vars_buffer = []
    def save_checkpoint(self):
        checkpoint_step = self.global_step
        checkpoint = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),

            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "dual_optimizer": self.dual_optimizer.state_dict(),

            "log_eta": self.log_eta.detach().cpu().clone(),
            "log_alpha_mean": (
                self.log_alpha_mean.detach().cpu().clone()
            ),
            "log_alpha_stddev": (
                self.log_alpha_stddev.detach().cpu().clone()
            ),
            "log_penalty_eta": (self.log_penalty_eta.detach().cpu().clone()),

            "global_step": self.global_step,
            "learner_step": self.learner_step,

            # Optional, but useful for checking continued-run consistency.
            "target_policy_update_period": (
                self.target_policy_update_period
            ),
            "target_critic_update_period": (
                self.target_critic_update_period
            ),
        }
        torch.save(checkpoint, os.path.join(self.weights_folder, f"checkpoint_{self.global_step}.pth"))
        # if global_step % self.args.save_every_n_steps == 0:
        self.rb.save(os.path.join(self.weights_folder, "replay_buffer.npz"))

    def load_checkpoint(self, weights_path, checkpoint_step=None):
        checkpoint_files = [f for f in os.listdir(weights_path) if f.endswith(".pth")]
        checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

        if checkpoint_step is None:
            checkpoint_file = checkpoint_files[-1]
        else:
            checkpoint_file = f"checkpoint_{checkpoint_step}.pth"
            if checkpoint_file not in checkpoint_files:
                available = ", ".join(checkpoint_files)
                raise FileNotFoundError(
                    f"Requested {checkpoint_file}, but it was not found in {weights_path}.\n"
                    f"Available checkpoints: {available}"
                )

        checkpoint_path = os.path.join(weights_path, checkpoint_file)
        print(f"[√] Loading checkpoint file: {checkpoint_path}", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.dual_optimizer.load_state_dict(checkpoint["dual_optimizer"])
      
        with torch.no_grad():
            if "log_eta" in checkpoint:
                self.log_eta.copy_(torch.as_tensor(checkpoint["log_eta"],dtype=self.log_eta.dtype,device=self.device).reshape_as(self.log_eta))
            if "log_alpha_mean" in checkpoint:
                self.log_alpha_mean.copy_(torch.as_tensor(checkpoint["log_alpha_mean"],dtype=self.log_alpha_mean.dtype,device=self.device).reshape_as(self.log_alpha_mean))
            if "log_alpha_stddev" in checkpoint:
                self.log_alpha_stddev.copy_(torch.as_tensor(checkpoint["log_alpha_stddev"],dtype=self.log_alpha_stddev.dtype,device=self.device).reshape_as(self.log_alpha_stddev))
            if "log_penalty_eta" in checkpoint:
                self.log_penalty_eta.copy_(torch.as_tensor(checkpoint["log_penalty_eta"],dtype=self.log_penalty_eta.dtype,device=self.device,).reshape_as(self.log_penalty_eta))
        self.global_step = int(checkpoint.get("global_step", 0))
        self.learner_step = int(checkpoint.get("learner_step", 0))
        print(f"[√] Loaded checkpoint from {weights_path}, global_step={self.global_step}")

        buffer_path = os.path.join(weights_path, "replay_buffer.npz")
        if os.path.exists(buffer_path):
            self.rb.load(buffer_path, self.device)
            print(f"[√] Loaded replay buffer.")
    
    def cleanup(self):
        if self.writer_info is not None and self.info_log_buffer:
            for row in self.info_log_buffer:
                self.writer_info.writerow(row)
            self.csv_file_info.flush()
        if self.writer_agent_vars is not None and self.agent_vars_buffer:
            for row in self.agent_vars_buffer:
                self.writer_agent_vars.writerow(row)
            self.csv_file_agent_vars.flush()
        if self.csv_file_info:
            self.csv_file_info.close()
        if self.csv_file_agent_vars:
            self.csv_file_agent_vars.close()
        if self.envs:
            self.envs.close()

def compute_weights_and_temperature_loss(q_values: torch.Tensor, epsilon: float, temperature: nn.Parameter,) -> Tuple[torch.Tensor, torch.Tensor]:
  """Computes normalized importance weights for the policy optimization.

  Args:
    q_values: Q-values associated with the actions sampled from the target
      policy; expected shape [N, B].
    epsilon: Desired constraint on the KL between the target and non-parametric
      policies.
    temperature: Scalar used to temper the Q-values before computing normalized
      importance weights from them. This is really the Lagrange dual variable
      in the constrained optimization problem, the solution of which is the
      non-parametric policy targeted by the policy loss.
  Returns:
    Normalized importance weights, used for policy optimization.
    Temperature loss, used to adapt the temperature.
  """

  # Temper the given Q-values using the current temperature.
  tempered_q_values = q_values.detach() / temperature

  # Compute the normalized importance weights used to compute expectations with
  # respect to the non-parametric policy.
  normalized_weights = F.softmax(tempered_q_values, dim=0).detach()

  # Compute the temperature loss (dual of the E-step optimization problem).
  q_logsumexp = torch.logsumexp(tempered_q_values, dim=0)
  log_num_actions = torch.log(
        torch.tensor(
            q_values.shape[0],
            dtype=q_values.dtype,
            device=q_values.device,
        )
    )
  loss_temperature = epsilon + q_logsumexp.mean() - log_num_actions
  loss_temperature = temperature * loss_temperature

  return normalized_weights, loss_temperature

def compute_cross_entropy_loss(
    sampled_actions: torch.Tensor,
    normalized_weights: torch.Tensor,
    online_policy: dist.Distribution,) -> torch.Tensor:
  """Compute cross-entropy online and the reweighted target policy.

  Args:
    sampled_actions: samples used in the Monte Carlo integration in the policy
      loss. Expected shape is [N, B, ...], where N is the number of sampled
      actions and B is the number of sampled states.
    normalized_weights: target policy multiplied by the exponentiated Q values
      and normalized; expected shape is [N, B].
    online_action_distribution: policy to be optimized.

  Returns:
    loss_policy_gradient: the cross-entropy loss that, when differentiated,
      produces the policy gradient.
  """

  # Compute the M-step loss.
  log_prob = online_policy.log_prob(sampled_actions)

  # Compute the weighted average log-prob using the normalized weights.
  loss_policy_gradient = -torch.sum(log_prob * normalized_weights, dim=0,) #(B,)

  # Return the mean loss over the batch of states.
  return loss_policy_gradient.mean(dim=0)

def compute_parametric_kl_penalty_and_dual_loss(
    kl: torch.Tensor,
    alpha: nn.Parameter,
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Computes the KL cost to be added to the Lagragian and its dual loss.

  The KL cost is simply the alpha-weighted KL divergence and it is added as a
  regularizer to the policy loss. The dual variable alpha itself has a loss that
  can be minimized to adapt the strength of the regularizer to keep the KL
  between consecutive updates at the desired target value of epsilon.

  Args:
    kl: KL divergence between the target and online policies.
    alpha: Lagrange multipliers (dual variables) for the KL constraints.
    epsilon: Desired value for the KL.

  Returns:
    loss_kl: alpha-weighted KL regularization to be added to the policy loss.
    loss_alpha: The Lagrange dual loss minimized to adapt alpha.
  """

  # Compute the mean KL over the batch.
  mean_kl = kl.mean(dim=0)  # (D,)

  # actor sees gradients through KL, not alpha
  loss_kl = torch.sum(alpha.detach() * mean_kl)

  # Compute the dual loss.
  loss_alpha = torch.sum(alpha * (epsilon - mean_kl.detach()))

  return loss_kl, loss_alpha

def compute_nonparametric_kl_from_normalized_weights(
    normalized_weights: torch.Tensor) -> torch.Tensor:
    """Returns estimated KL(q || target_policy), shape (B,)
    Estimate the actualized KL between the non-parametric and target policies."""
    num_actions = normalized_weights.shape[0]

    integrand = torch.log(
        num_actions * normalized_weights + _MPO_FLOAT_EPSILON)
    # Return the expectation with respect to the non-parametric policy.
    return torch.sum(normalized_weights * integrand,dim=0,)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="dmpo_ant")
    parser.add_argument("--runs_directory", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch_deterministic", type=bool, default=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--save_every_n_steps", type=int, default=4000)
    parser.add_argument("--log_every_n_steps", type=int, default=100)

    parser.add_argument("--env_id", type=str, default="EAnt")
    parser.add_argument("--total_timesteps", type=int, default=60_000)
    parser.add_argument("--num_envs", type=int, default=1)

    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    #parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=512)

    parser.add_argument("--vmin", type=float, default=-500,
                    help="Minimum atom value for distributional critic")
    parser.add_argument("--vmax", type=float, default=20,
                        help="Maximum atom value for distributional critic")
    parser.add_argument("--num_atoms", type=int, default=101,
                        help="Number of categorical atoms for distributional critic")
    # parser.add_argument("--critic_num_samples", type=int, default=32,
    #                     help="Number of next-action samples used to build target critic distribution")

    #Logging
    parser.add_argument("--log_interval", type=int, default=100,
                        help="env steps between TensorBoard scalar writes")
    #MPO
    parser.add_argument("--epsilon_eta", type=float, default=1e-1,
                        help="epsilon for E-step temperature dual")
    parser.add_argument("--epsilon_mu_kl", type=float, default=2.5e-3,
                        help="epsilon_mu for M-step mean KL")
    parser.add_argument("--epsilon_sigma_kl", type=float, default=1e-6,
                        help="epsilon_sigma for M-step covariance KL")
  
    parser.add_argument("--sample_action_num", type=int, default=64,
                        help="actions sampled per state in E-step")
    # parser.add_argument("--mstep_iteration_num", type=int, default=4,
    #                     help="actor gradient steps per learn() call")
    parser.add_argument("--dual_lr", type=float, default=1e-3)
   
    parser.add_argument("--max_grad_norm", type=float, default=40.0)

    parser.add_argument("--ensemble", type=int, default=1) #doesn't matter in this script
    # true learning start = learning start // num_envs (floor division)
    parser.add_argument("--learning_starts", type=int, default=200)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--q_lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    # layer norm arg is only effective if old Actor class is used rather than AcmeActo
    parser.add_argument("--use_layer_norm", action=argparse.BooleanOptionalAction, default=True) 

    parser.add_argument("--policy_layer_sizes",type=int,nargs="+",default=[256, 256, 256])
    parser.add_argument("--critic_layer_sizes",type=int,nargs="+",default=[512, 512, 256])
    parser.add_argument("--utd", type=int, default=32)
    parser.add_argument("--td_horizon", type=int, default=5,
                        help="number of steps collapsed into each replay transition")
    parser.add_argument("--decouple_q_learning", action=argparse.BooleanOptionalAction, default=True,
                        help="use --decouple_q_learning or --no-decouple_q_learning")
    parser.add_argument("--policy_learning_starts", type=int, default=500,
                        help="steps of Q-only warmup after learning_starts when --decouple_q_learning is set")
    
    parser.add_argument("--checkpoint_step",type=int,default=None,
                            help="Specific checkpoint step to load, e.g. 95000. If omitted, loads latest.")

    parser.add_argument("--target_policy_update_period",type=int,default=100,
                        help="Hard-copy online actor to target actor every N learner updates")
    parser.add_argument("--target_critic_update_period",type=int,default=100,
                        help="Hard-copy online critic to target critic every N learner updates",)
    parser.add_argument("--init_log_temperature", type=float, default=10.0)
    parser.add_argument("--init_log_alpha_mean", type=float, default=10.0)
    parser.add_argument("--init_log_alpha_stddev", type=float, default=1000.0)

    # Environment.
    parser.add_argument("--dt", type=float, default=0.15)
    parser.add_argument("--hw_config", type=str, default=None)
    parser.add_argument("--render_mode", type=str, default=None)
    parser.add_argument("--terminate_on_upside_down", type=bool, default=True)
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--task_type", type=str, default="back_and_forth",
                        choices=["forward", "back_and_forth"])
    parser.add_argument("--radius_back_and_forth", type=float, default=0.3)
    parser.add_argument("--origin_back_and_forth", type=float, nargs=2, default=[0.75, -0.3])
    parser.add_argument("--reward_scale", type=float, default=100.0)
    parser.add_argument("--model_path", type=str,
                        default="../../sim/assets/ant_with_camera_after_sys_id.xml")

    parser.add_argument("--samples_per_insert",type=float,default=512.0,
                        help="Replay samples consumed per valid environment transition")
    parser.add_argument("--action_penalization",action=argparse.BooleanOptionalAction,default=True,
                        help="MO-MPO action penalization for pi-target samples outside bounds")
    parser.add_argument("--epsilon_penalty",type=float,default=1e-3,
                        help="KL constraint for action penalization")

    return parser.parse_args()

def make_ant_envs(args, task, disk_folder, run_name, runs_directory='runs'):
    def make_env(seed, idx, capture_video, run_name):
        def _init():
            joint_config = {
                'hip_zero': 0,
                'knee_zero': -np.radians(50),
                'hip_range': np.radians(30),
                'knee_range': np.radians(20),
            }
            if args.hw_config is None:
                env = AntEnv(
                    control_dt=args.dt,
                    render_mode=args.render_mode,
                    terminate_on_upside_down=args.terminate_on_upside_down,
                    task=task,
                    joint_config=joint_config,
                    model_path=os.path.join(os.path.dirname(__file__), args.model_path),
                )
            else:
                with open(args.hw_config, 'r') as f:
                    cfg = json.load(f)
                env = make_ant_env(cfg, render_mode=args.render_mode,
                                   dt=args.dt, joint_config=joint_config, task=task)
            if capture_video and idx == 0:
                env = gym.wrappers.RecordVideo(
                    env,
                    os.path.join(disk_folder, runs_directory, run_name, "videos", run_name),
                    step_trigger=lambda x: x % args.save_every_n_steps == 0,
                    video_length=args.save_every_n_steps,
                )
            env.action_space.seed(seed)
            env = gym.wrappers.TransformReward(env, lambda r: r * args.reward_scale)
            return env
        return _init

    env_raw = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)],autoreset_mode=AutoresetMode.NEXT_STEP)
    assert isinstance(env_raw.single_action_space, gym.spaces.Box), "[!] Only continuous action space is supported."
    
    # wrap the env using skrl, gymnasium
    envs = wrap_env(env_raw, wrapper="gymnasium")
    print(f"[√] Created environment with {envs.num_envs} environments.")
    
    return env_raw, envs

def _try_run_git_command(args, cwd):
    try:
        result = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, check=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"
 

def save_git_info(output_dir, cwd):
    os.makedirs(output_dir, exist_ok=True)
    # Save git hash
    git_hash = _try_run_git_command(['git', 'rev-parse', 'HEAD'], cwd)
    with open(os.path.join(output_dir, "git_hash.txt"), "w") as f:
        f.write(git_hash + "\n")
    # Save branch name
    git_branch = _try_run_git_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], cwd)
    with open(os.path.join(output_dir, "git_branch.txt"), "w") as f:
        f.write(git_branch + "\n")
    # Save diffs
    git_diff = _try_run_git_command(['git', 'diff'], cwd)
    with open(os.path.join(output_dir, "git_diff.patch"), "w") as f:
        f.write(git_diff)

def scalar_value(value):
    if torch.is_tensor(value):
        return value.detach().reshape(-1)[0].item()
    array = np.asarray(value)
    if array.size == 0:
        return None
    return array.reshape(-1)[0].item()

def main():
    args = parse_args()
    set_seed(args.seed,deterministic=args.torch_deterministic)
    seed = args.seed
    args.learning_starts = args.learning_starts//args.num_envs #integer div takes floor
    args.policy_learning_starts = args.policy_learning_starts//args.num_envs

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    disk_folder = ''
    os.makedirs(args.runs_directory, exist_ok=True)
    run_name = f"{args.exp_name}_{date}_seed_{args.seed}"
    os.makedirs(os.path.join(args.runs_directory, run_name), exist_ok=True)

    if args.task_type == "forward":
        task = ForwardTask()
    elif args.task_type == "back_and_forth":
        RADIUS = args.radius_back_and_forth
        ORIGIN = np.array(args.origin_back_and_forth)
        task = BackAndForthTask(radius=RADIUS, origin=ORIGIN)
        print(f"BackAndForthTask: radius={RADIUS}, origin={ORIGIN}")
    else:
        raise ValueError(f"Invalid task type: {args.task_type}")

    raw_env, envs = make_ant_envs(args, task, disk_folder, run_name, runs_directory=args.runs_directory)
    agent = MPO(args=args,envs=envs,disk_folder=disk_folder,run_name=run_name,runs_directory=args.runs_directory)
    
    if args.weights_path is not None:
        agent.load_checkpoint(args.weights_path,
            checkpoint_step=args.checkpoint_step,
        )
        if args.eval:
            agent.global_step = 0
    print(f"obs space: {envs.observation_space}")
    print(f"act space: {envs.action_space}")
    print(f"device: {envs.device}")
    print(f"num_envs: {envs.num_envs}")

    obs, info = envs.reset()
    agent.initialize_logging(info)

    try:
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(agent.log_dir, exist_ok=True)
        writer = SummaryWriter(agent.log_dir)
    except ImportError:
        writer = None
        print("[!] tensorboard not available, printing only")

    #time_start_learning = time.time()
    step_times = []

    train_steps = int(np.round(args.total_timesteps/args.num_envs))
    try:
        for step in tqdm(range(agent.global_step, args.total_timesteps)):
            time_now = time.time()
            selected_actions = agent.get_action(obs, args.eval)
            next_obs, rewards, terminations, truncations, infos = envs.step(selected_actions)

            if args.eval:
                agent.agent_step_eval(next_obs)
                metrics = None
            else:
                metrics = agent.agent_step(next_obs, selected_actions, rewards, terminations, truncations, infos)
            
            current_step = agent.global_step
            agent.log_step(current_step, infos, rewards, metrics)

            if writer is not None and current_step % args.log_interval == 0:
                writer.add_scalar("reward/mean", float(rewards.mean()), current_step)
                if "benchmark_reward" in infos:
                    writer.add_scalar("reward/benchmark",
                                        float(np.asarray(infos["benchmark_reward"]).mean()), current_step)
                writer.add_scalar("perf/buffer_size", agent.rb.size(), current_step)
            if metrics is not None:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        writer.add_scalar(f"agent/{k}", v, agent.learner_step)
    
            # ---- checkpoints (<run_dir>/weights_and_args) ----
            if (not args.eval and current_step % args.save_every_n_steps == 0):
                agent.save_checkpoint()
                if writer is not None:
                    writer.flush()
            step_times.append(f"{current_step},{time.time() - time_now}\n")

        with open(os.path.join(args.runs_directory, run_name, "step_times.csv"), "w") as f:
            f.writelines(step_times)
    finally:
        with open(os.path.join(args.runs_directory,run_name,"step_times.csv"),"w") as f:
            f.writelines(step_times)

        if not args.eval:
            agent.save_checkpoint()

        if writer is not None:
            writer.close()
    
    agent.cleanup()
if __name__ == "__main__":
    main()
