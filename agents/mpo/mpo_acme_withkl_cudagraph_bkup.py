import os
import sys
import json
import copy
import shutil
import argparse
import subprocess
import csv
import time
import math
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime

from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist

from skrl.utils import set_seed
try:
    from .buffer_acme import ReplayBuffer  # imported as package
except ImportError:
    from buffer_acme import ReplayBuffer   # run standalone
try:
    from .envs.envs import is_gymnasium_env, make_envs
except ImportError:
    from envs.envs import is_gymnasium_env, make_envs
try:
    from .resume_state import get_rng_state, set_rng_state, snapshot_envs, restore_envs
except ImportError:
    from resume_state import get_rng_state, set_rng_state, snapshot_envs, restore_envs

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import ForwardTask, BackAndForthTask
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from reward import RewardTracker, PuckSolvedTracker
from early_stopping import StopTrainingOnNoImprovement, StopTrainingOnRewardThreshold
from sb3_eval import Evaluator

try:
    from .nn import (AcmeActor, AcmeCritic, DiscreteValuedDistribution,  # imported as package
                     Actor, Critic, ScalarAcmeCritic, QuantileCritic, categorical, td_learning,
                     quantile_huber_loss)
except ImportError:
    from nn import (AcmeActor, AcmeCritic, DiscreteValuedDistribution,   # run standalone
                    Actor, Critic, ScalarAcmeCritic, QuantileCritic, categorical, td_learning, 
                    quantile_huber_loss)

def arr_to_str(x):
    if isinstance(x, np.ndarray):
        return "[" + " ".join(map(str, x.tolist())) + "]"
    return x
# import crane
# from crane import tile_images, save_video
_MPO_FLOAT_EPSILON = 1e-8
# Distribution arg checks call .all() on GPU tensors, a hidden sync per distribution
dist.Distribution.set_default_validate_args(False)


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
        # NVIDIA cuDNN library uses deterministic convolution algo; if false can use FFT based or Winograd convolutions; slower but mathematically more sound
        torch.backends.cudnn.deterministic = args.torch_deterministic 
        torch.backends.cudnn.benchmark = not args.torch_deterministic #if not deterministic, find and use fastest one
        print(f"[√] Torch deterministic: {torch.backends.cudnn.deterministic}")
        print(f"[√] Torch benchmark: {torch.backends.cudnn.benchmark}")

        policy_layer_sizes = tuple(args.policy_layer_sizes)
        critic_layer_sizes = tuple(args.critic_layer_sizes)

        self.obs_dim = int(np.array(self.envs.single_observation_space.shape).prod())
        self.act_dim = int(np.prod(self.envs.single_action_space.shape))
        self.envs.single_observation_space.dtype = np.float32
        self.action_low = envs.single_action_space.low.astype(np.float32)
        self.action_high = envs.single_action_space.high.astype(np.float32)
        assert np.all(np.isfinite(envs.single_action_space.low)) and np.all(np.isfinite(envs.single_action_space.high))

        # 'acme': Acme LayerNormMLP torso + near-zero-init head.
        # 'default': ReLU (+ optional LayerNorm) torso, same mean/stddev head math, PyTorch default init.
        if args.actor_net == "acme":
            actor_cls, actor_kwargs = AcmeActor, {}
        else:
            actor_cls, actor_kwargs = Actor, dict(use_layer_norm=args.use_layer_norm)

        def make_actor():
            return actor_cls(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                act_low=self.action_low,
                act_high=self.action_high,
                layer_sizes=policy_layer_sizes,
                init_scale=args.policy_init_scale,
                min_scale=args.policy_min_scale,
                **actor_kwargs,
            ).to(self.device)

        self.actor = make_actor()
        self.actor_target = make_actor() # q(a|s) in the main MPO paper.
        self.actor_target.load_state_dict(self.actor.state_dict())
        for p in self.actor_target.parameters():
            p.requires_grad = False

        if self.args.critic_type == "categorical":
            critic_cls = AcmeCritic
            critic_kwargs = dict(
                vmin=args.vmin,
                vmax=args.vmax,
                num_atoms=args.num_atoms,
            )
        elif self.args.critic_type == "quantile":
            if self.args.ensemble != 1:
                raise ValueError("Quantile critic currently supports --ensemble 1 only.")
            critic_cls = QuantileCritic
            critic_kwargs = dict(num_quantiles=args.num_quantiles)
        elif self.args.critic_net == "default":
            critic_cls = Critic
            critic_kwargs = dict(use_layer_norm=args.use_layer_norm)
        else:
            critic_cls = ScalarAcmeCritic
            critic_kwargs = {}
        if self.args.critic_net == "default" and self.args.critic_type != "scalar":
            raise ValueError("--critic_net default only supports --critic_type scalar.")

        self.critics = nn.ModuleList([
            critic_cls(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                action_low=self.action_low,
                action_high=self.action_high,
                layer_sizes=critic_layer_sizes,
                clip_action=args.clip_action_critic,
                **critic_kwargs,
            ) for _ in range(self.args.ensemble)
        ]).to(self.device)

        self.target_critics = copy.deepcopy(self.critics)
        for p in self.target_critics.parameters():
            p.requires_grad = False

        # The whole learner update is captured once as a CUDA graph and replayed every step
        self._use_cuda_graph = args.cuda_graph and torch.device(self.device).type == "cuda"
        self._graph = None
        self._graph_warmup_left = 3
        self._static_inputs = None
        self._static_stats = None
        cap = self._use_cuda_graph
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.policy_lr, capturable=cap)
        self.critic_optimizers = [
            optim.Adam(critic.parameters(), lr=args.q_lr, capturable=cap)
            for critic in self.critics
        ]

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
        # Per-dim dual for the joint (full-policy) KL constraint, used only with --kl_constraint joint.
        self.log_alpha = nn.Parameter(
            torch.full((self.act_dim,), args.init_log_alpha, dtype=torch.float32, device=self.device))
        # Scalar dual for the L2 parameter-space constraint, used only with --kl_constraint l2.
        self.log_beta = nn.Parameter(
            torch.full((1,), args.init_log_beta, dtype=torch.float32, device=self.device))
        self.log_penalty_eta = nn.Parameter(torch.full((1,),self._init_log_eta,dtype=torch.float32,device=self.device))
        self.dual_optimizer = optim.Adam(
            [
                self.log_eta,
                self.log_alpha_mean,
                self.log_alpha_stddev,
                self.log_penalty_eta,
                # only when M-step KL is NOT decoupled between mean & std dev 
                *([self.log_alpha] if args.kl_constraint == "joint" else []),
            ],
            lr=args.dual_lr,
            capturable=cap,
        )
        # Separate optimizer for the L2 dual so dual_optimizer keeps the same layout across
        # modes (e.g. a decoupled sim1 checkpoint can be loaded into an l2 sim2 run).
        self.l2_dual_optimizer = (
            optim.Adam([self.log_beta], lr=args.dual_lr, capturable=cap) if args.kl_constraint == "l2" else None)

        # Independent per-dim E-step KL multipliers for parametric Gaussian q.
        # The existing nonparametric temperature and M-step dual optimizer are untouched.
        self._variational = args.variational_distribution
        if self._variational == "parametric":
            self.param_log_eta_mu = nn.Parameter(torch.full(
                (self.act_dim,), args.parametric_init_log_eta_mu, dtype=torch.float32, device=self.device))
            self.param_log_eta_sigma = nn.Parameter(torch.full(
                (self.act_dim,), args.parametric_init_log_eta_sigma, dtype=torch.float32, device=self.device))
            self.parametric_dual_optimizer = (optim.Adam(
                [
                    self.param_log_eta_mu, 
                    self.param_log_eta_sigma
                ],
                    lr=args.dual_lr, 
                    capturable=cap
            ))
        self.rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            self.device,
            n_envs=args.num_envs,
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
        self.csv_file_raw_actions = None
        self.writer_raw_actions = None
        self.raw_action_buffer = []
        self.writer_info = None
        self.writer_agent_vars = None
        self.keys_info = None
        self.keys_agent_vars = [
            "critic_loss",
            *([f"critic_loss_{idx}" for idx in range(self.args.ensemble)]
              if self.args.ensemble > 1 else []),
            "policy_loss","dual_alpha_mean","dual_alpha_stddev",
            "dual_temperature","loss_alpha","loss_temperature","loss_policy_cross_entropy",
            "loss_kl_penalty","kl_q_rel","kl_mean_rel","kl_stddev_rel","q_min",
            "q_max","pi_stddev_min","pi_stddev_max","pi_stddev_cond","utd",
            "SPS","average_reward_per_second","reward","mean_return"]

        for idx in range(self.act_dim):
            self.keys_agent_vars.extend(
                [
                    f"dual_alpha_mean_{idx}",
                    f"dual_alpha_stddev_{idx}",
                    f"pi_stddev_{idx}",
                ]
            )
        if args.kl_constraint == "joint":
            self.keys_agent_vars.extend(["dual_alpha", "kl_joint_rel"])
            self.keys_agent_vars.extend([f"dual_alpha_{idx}" for idx in range(self.act_dim)])
        if args.kl_constraint == "l2":
            self.keys_agent_vars.extend(["dual_beta", "l2_dist", "l2_rel"])
        self.info_log_buffer = []
        self.agent_vars_buffer = []

        self._epsilon = args.epsilon_eta
        self._epsilon_mean = args.epsilon_mu_kl
        self._epsilon_stddev = args.epsilon_sigma_kl
        
        self._kl_constraint = args.kl_constraint
        self._epsilon_kl = args.epsilon_kl
        self._epsilon_l2 = args.epsilon_l2
        self._ce_coupling = args.ce_coupling
        self._param_epsilon_mu = args.parametric_epsilon_mu
        self._param_epsilon_sigma = args.parametric_epsilon_sigma
        self._epsilon_clip = args.epsilon_clip

        # a variable that tracks when reset happens - the transition after boundary = true is
        # when transitioning from terminal state to reset state - this is when pending_autoreset=1,
        # to inform agent to not use this transition during training.
        self.pending_autoreset = torch.zeros(self.envs.num_envs,dtype=torch.bool,device=self.device)
        self._ep_returns = np.zeros(self.envs.num_envs, dtype=np.float64)
        print("Agent initialized and compiled")
    
    def _random_action(self):
        low, high = self.actor.action_low, self.actor.action_high
        rand_actions = low + (high - low) * torch.rand(self.envs.num_envs, self.act_dim, device=self.device)
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
                actions = actions.squeeze(1) # (n_envs, act_dim)

                log_probs = d.log_prob(actions).unsqueeze(-1)
                self.last_actions = actions
                self.last_log_probs = log_probs
                return actions
        return self._random_action()

    def _step_rewards_np(self, infos, rewards):
        if "original_reward" in infos:
            return np.asarray(infos["original_reward"], dtype=np.float64).reshape(-1)
        if isinstance(rewards, torch.Tensor):
            return rewards.detach().cpu().numpy().reshape(-1).astype(np.float64)
        return np.asarray(rewards, dtype=np.float64).reshape(-1)

    def _update_episode_returns(self, infos, rewards, boundaries, autoreset_now):
        if self.reward_tracker is None:
            return
        step_rewards = self._step_rewards_np(infos, rewards)
        done_np = boundaries.detach().cpu().numpy().reshape(-1).astype(bool)
        autoreset_np = autoreset_now.detach().cpu().numpy().reshape(-1).astype(bool)
        n = min(self.envs.num_envs, len(step_rewards))
        for i in range(n):
            if autoreset_np[i]:
                self._ep_returns[i] = step_rewards[i]
            else:
                self._ep_returns[i] += step_rewards[i]
                if done_np[i]:
                    self.reward_tracker.record_episode_return(self._ep_returns[i])

    def agent_step(self, next_obs, actions, rewards, terminations, truncations, infos):
        terminations = terminations.bool()
        truncations = truncations.bool()

        # either flag will end the sampled trajectory
        # only terminations disable bootstrapping in the replay buffer though
        boundaries = terminations | truncations

        if "autoreset" in infos:
            autoreset_now = torch.as_tensor(infos["autoreset"],dtype=torch.bool,device=self.device)
        else:
            autoreset_now = self.pending_autoreset
        valid = ~autoreset_now

        self._update_episode_returns(infos, rewards, boundaries, autoreset_now)

        num_inserts = int(valid.sum().item())
        self.rb.add(
            obs=self.obs,
            next_obs=next_obs,
            action=actions,
            reward=rewards,
            terminated=terminations,
            truncated=truncations,
            valid=valid,
        )

        self.global_step += 1
        self.obs = next_obs
        self.pending_autoreset = boundaries.detach().clone()
        metrics = None

        if self.global_step > self.args.learning_starts and self.rb.size() >= self.batch_size:
            # Count actual environment transitions inserted.
            # Reset-only rows have valid=False and should not count.
    
            self.learner_update_budget += (self.samples_per_insert * num_inserts / self.batch_size)
            num_updates = int(self.learner_update_budget)
            if num_updates > 0:
                self.learner_update_budget -= (num_updates)
                # Stats stay on the GPU; they are only copied to the CPU on logging steps
                need_stats = self.global_step % self.args.log_every_n_steps == 0
                results = []
                for _ in range(num_updates):
                    stats = self._learn()
                    if need_stats:
                        # clone: CUDA graph outputs are overwritten by the next replay
                        results.append({k: v.clone() for k, v in stats.items()})
                if not need_stats:
                    return None
                metrics = _stats_to_floats(
                    {k: sum(r[k] for r in results) / len(results) for k in results[0]})

                # Actual learner updates performed during this vector-environment step.
                metrics["utd"] = num_updates
        return metrics

    def agent_step_eval(self, next_obs, rewards, terminations, truncations, infos):
        terminations = terminations.bool()
        truncations = truncations.bool()
        boundaries = terminations | truncations

        if "autoreset" in infos:
            autoreset_now = torch.as_tensor(infos["autoreset"], dtype=torch.bool, device=self.device)
        else:
            autoreset_now = self.pending_autoreset

        self._update_episode_returns(infos, rewards, boundaries, autoreset_now)

        self.global_step += 1
        self.obs = next_obs
        self.pending_autoreset = boundaries.detach().clone()

    def _learn(self):
        with torch.no_grad():
            if (self.learner_step % self.target_policy_update_period== 0):
                self.actor_target.load_state_dict(self.actor.state_dict())

            if (self.learner_step % self.target_critic_update_period == 0):
                for critic, target_critic in zip(self.critics, self.target_critics):
                    target_critic.load_state_dict(critic.state_dict())
        self.learner_step += 1

        data = self.rb.sample_nstep(
            batch_size=self.batch_size,
            n_step=self.trajectory_length,
            gamma=self.args.gamma,
        )
        # Random draws happen outside the graph and are passed in as inputs
        noise = torch.randn(self.args.sample_action_num, data.observations.shape[0], self.act_dim,
                            device=data.observations.device)
        subset_idx = (torch.randperm(len(self.critics), device=self.device)[:2]
                      if len(self.critics) > 1 else torch.zeros(0, dtype=torch.long, device=self.device))
        inputs = (data.observations, data.actions, data.rewards,
                  data.discounts, data.next_observations, noise, subset_idx)
        if not self._use_cuda_graph:
            return self._update(*inputs)

        if self._graph is None:
            if self._graph_warmup_left > 0:
                # CUDA graphs need a few eager iterations on a side stream before capture
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    stats = self._update(*inputs)
                torch.cuda.current_stream().wait_stream(side)
                self._graph_warmup_left -= 1
                return stats
            self._static_inputs = [x.clone() for x in inputs]
            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._static_stats = self._update(*self._static_inputs)
            # capture only records the kernels; the replay below runs this step's update

        for static, x in zip(self._static_inputs, inputs):
            static.copy_(x)
        self._graph.replay()
        return self._static_stats

    def _update(self, s_tm1, a_tm1, rewards, discounts, s_t, noise, subset_idx):
        """One learner update on a sampled batch. No CPU syncs, so it can be captured as a CUDA graph."""
        with torch.no_grad():
            # Randomized critic subset in case of ensemble
            if len(self.critics) == 1:
                subset_idx = None

            B = s_tm1.shape[0]
            if B == 0:
                raise RuntimeError("Replay buffer returned an empty batch")
            # the same samples are used to average the target categorical 
            # critic distributions & obtain scalar Q vals for MPO.
            N = self.args.sample_action_num
            D = self.act_dim

            # the collapsed n-step discounted return
            r_t = rewards.squeeze(-1)
            
            target_policy = self.actor_target.forward(s_t)
            # Shape: (N, B, D)
            sampled_actions = target_policy.base_dist.loc + target_policy.base_dist.scale * noise

            tiled_states = (
                s_t.unsqueeze(0)
                .expand(N, -1, -1)
                .reshape(N * B, self.obs_dim)
            )

            flat_actions = sampled_actions.reshape(N * B, D)

        online_policy = self.actor.forward(s_t)

        # bootstrapping coefficient
        pcont_t = (self.args.gamma * discounts).squeeze(-1)

        if self.args.critic_type == "categorical":
            # Sample from the target critic(s) and compute the average target distribution
            with torch.no_grad():
                sampled_q_t_distributions = [
                    critic(tiled_states, flat_actions)
                    for critic in self.target_critics
                ]
                num_atoms = sampled_q_t_distributions[0].logits.shape[-1]
                sampled_q_logits = torch.stack([
                    d.logits.reshape(N, B, num_atoms)
                    for d in sampled_q_t_distributions], dim=0)

                # Average the target over the sampled distributions from a subset of critics
                if subset_idx is None:
                    subset_logits = sampled_q_logits
                    subset_atom_values = self.target_critics[0].atom_values
                else:
                    subset_logits = sampled_q_logits[subset_idx]
                    subset_atom_values = self.target_critics[0].atom_values
                sampled_q_logprobs = torch.log_softmax(subset_logits, dim=-1)
                averaged_target_logits = torch.logsumexp(sampled_q_logprobs, dim=(0, 1))

                target_q_distribution = DiscreteValuedDistribution(
                    values=subset_atom_values,
                    logits=averaged_target_logits,
                )
                q_values = torch.stack([
                    d.mean().reshape(N, B)
                    for d in sampled_q_t_distributions], dim=0).mean(dim=0)

            # Cross-entropy loss per critic
            critic_loss_per_critic = torch.stack([
                categorical(
                    q_tm1=critic(s_tm1, a_tm1),
                    r_t=r_t,
                    d_t=pcont_t,
                    q_t=target_q_distribution,
                ).mean()
                for critic in self.critics], dim=0)
        elif self.args.critic_type == "quantile":
            with torch.no_grad():
                num_quantiles = self.args.num_quantiles
                next_quantiles = self.target_critics[0](
                    tiled_states, flat_actions).reshape(N, B, num_quantiles)
                #  for nonparametric q(a|s) in E step of MPO. get q_value for each of N actions
                q_values = next_quantiles.mean(dim=-1) # [N, B]

                # find loss for each of M next actions separately, then average 
                # the losses by multiplying each with the corresponding action's
                # probability
                M = self.args.quantile_target_action_num
                target_quantiles = r_t.unsqueeze(0).unsqueeze(-1) + pcont_t.unsqueeze(0).unsqueeze(-1) * next_quantiles[:M]
            losses = []
            current_quantiles = self.critics[0](s_tm1, a_tm1)
            for n in range(M):                
                loss_n = quantile_huber_loss(
                    current_quantiles,
                    target_quantiles[n],
                    sum_over_quantiles=self.args.sum_over_quantiles,
                    huber_param=self.args.quantile_huber_param,
                )
                losses.append(loss_n)
                # just use one of the sampled a_t+1 from pi_target(s_t+1) to compute target?
                # [B, 201]  -> for 1 action, has 201 quantiles.
                # target_quantiles = r_t.unsqueeze(-1) + pcont_t.unsqueeze(-1) * best_target_quantiles #next_quantiles[0]
            critic_loss_per_critic = torch.stack(losses).mean()
            # critic_loss_per_critic = quantile_huber_loss(
            #     current_quantiles,
            #     target_quantiles,
            #     sum_over_quantiles=self.args.sum_over_quantiles,
            #     huber_param=self.args.quantile_huber_param,
            # )

        else:
            with torch.no_grad():
                sampled_q_t = torch.stack([
                    critic(tiled_states, flat_actions).squeeze(-1).reshape(N, B)
                    for critic in self.target_critics], dim=0)

                # Min over the sampled subset of critics
                if subset_idx is None:
                    q_values_target_calc = sampled_q_t.min(dim=0).values
                else:
                    q_values_target_calc = sampled_q_t[subset_idx].min(dim=0).values

                # Mean over all critics for the Q values in the E-step
                q_values = sampled_q_t.mean(dim=0)

                # Average Q value across the sampled actions
                averaged_q_t = q_values_target_calc.mean(dim=0)

            # Q(s_t, a_t) for EACH online critic in the ensemble
            online_q = torch.stack([
                critic(s_tm1, a_tm1).squeeze(-1) for critic in self.critics], dim=0)

            # TD loss: 0.5 * (target - online_q)^2, target = r_t + pcont_t * averaged_q_t
            critic_loss_per_critic, _target, _td_error = td_learning(online_q, r_t, pcont_t, averaged_q_t)
            critic_loss_per_critic = critic_loss_per_critic.mean(dim=1)

        # Scalar objective as a sum over the ensemble of critics
        critic_loss = critic_loss_per_critic.sum()

        with torch.no_grad():
            self.log_eta.clamp_(min=-18.0)
            self.log_alpha_mean.clamp_(min=-18.0)
            self.log_alpha_stddev.clamp_(min=-18.0)
            self.log_alpha.clamp_(min=-18.0)  # if M-step KL is coupled instead
            self.log_beta.clamp_(min=-18.0)  # if M-step uses the L2 constraint instead
            if self._variational == "parametric":
                self.param_log_eta_mu.clamp_(min=-18.0)
                self.param_log_eta_sigma.clamp_(min=-18.0)
        # Transform dual variables from log-space.
        # using softplus instead of exponential for numerical stability
        temperature = F.softplus(self.log_eta) + _MPO_FLOAT_EPSILON
        alpha_mean = F.softplus(self.log_alpha_mean) + _MPO_FLOAT_EPSILON
        alpha_stddev = F.softplus(self.log_alpha_stddev) + _MPO_FLOAT_EPSILON
        alpha = F.softplus(self.log_alpha) + _MPO_FLOAT_EPSILON
        beta = F.softplus(self.log_beta) + _MPO_FLOAT_EPSILON
        if self._variational == "parametric":
            param_eta_mu = F.softplus(self.param_log_eta_mu) + _MPO_FLOAT_EPSILON
            param_eta_sigma = F.softplus(self.param_log_eta_sigma) + _MPO_FLOAT_EPSILON

        # Get online and target means and stddevs in preparation for decomposition.
        target_mu = target_policy.base_dist.loc
        target_sigma = target_policy.base_dist.scale
        online_mu = online_policy.base_dist.loc
        online_sigma = online_policy.base_dist.scale

        if self._variational == "parametric":
            adv = q_values - q_values.mean(dim=0, keepdim=True)                # [N, B]
            log_q = online_policy.log_prob(sampled_actions)                    # [N, B], grad -> theta
            log_p = target_policy.log_prob(sampled_actions)                    # [N, B], no grad
            ratio = torch.exp(log_q - log_p)
            # PPO clipped surrogate (target policy plays the role of pi_old)
            clipped_ratio = torch.clamp(ratio, 1.0 - self._epsilon_clip, 1.0 + self._epsilon_clip)
            loss_policy = -torch.min(ratio * adv, clipped_ratio * adv).mean()
            clip_frac = ((ratio - 1.0).abs() > self._epsilon_clip).float().mean().detach()

            fixed_stddev_distribution = dist.Independent(dist.Normal(online_mu, target_sigma), 1)
            fixed_mean_distribution = dist.Independent(dist.Normal(target_mu, online_sigma), 1)
            
            kl_mean = torch.distributions.kl_divergence(target_policy.base_dist,
                fixed_stddev_distribution.base_dist)  # Shape [B, D].
            kl_stddev = torch.distributions.kl_divergence(target_policy.base_dist,
                fixed_mean_distribution.base_dist)  # Shape [B, D]
            # coupled per-dim KL
            kl_joint = torch.distributions.kl_divergence(target_policy.base_dist,
                online_policy.base_dist)  # Shape [B, D]

            loss_kl_mu, loss_eta_mu = compute_parametric_kl_penalty_and_dual_loss(
                kl_mean, param_eta_mu, self._param_epsilon_mu)
            loss_kl_sigma, loss_eta_sigma = compute_parametric_kl_penalty_and_dual_loss(
                kl_stddev, param_eta_sigma, self._param_epsilon_sigma)
            loss_kl_penalty = loss_kl_mu + loss_kl_sigma
            loss_dual = loss_eta_mu + loss_eta_sigma        # no MPO temperature / M-step alpha
            total_mpo_loss = loss_policy + loss_kl_penalty + loss_dual

        else:
            
            # Compute normalized importance weights, used to compute expectations with
            # respect to the non-parametric policy; and the temperature loss, used to
            # adapt the tempering of Q-values.
            normalized_weights, loss_temperature = compute_weights_and_temperature_loss(
                q_values, self._epsilon, temperature)
        
            # Only for diagnostics: Compute estimated actualized KL between the
            # non-parametric and current target policies. 
            kl_nonparametric = compute_nonparametric_kl_from_normalized_weights(
                normalized_weights)
        
            if self._action_penalization:
                # Project and transform action penalization temperature.
                with torch.no_grad():
                    self.log_penalty_eta.clamp_(min=-18.0)
                penalty_temperature = F.softplus(self.log_penalty_eta) + _MPO_FLOAT_EPSILON
           
                # Compute action penalization cost
                # the cost is zero in the specified action range (but NOT quadratic beyond)
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
            # https://arxiv.org/pdf/1812.02256.pdf.
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
            # Full, coupled per-dim KL between target and online policy
            kl_joint = torch.distributions.kl_divergence(target_policy.base_dist,
                online_policy.base_dist)  # Shape [B, D]

            # M-step cross-entropy, chosen independently of the KL / L2 regularizer below.
            if self._ce_coupling == "decoupled":
                loss_policy = loss_policy_mean + loss_policy_stddev
            else:
                loss_policy = compute_cross_entropy_loss(
                    sampled_actions, normalized_weights, online_policy)

            if self._kl_constraint == "decoupled":
                # Compute mean & stddev separately: alpha-weighted KL-penalty and dual losses
                loss_kl_mean, loss_alpha_mean = compute_parametric_kl_penalty_and_dual_loss(
                    kl_mean, alpha_mean, self._epsilon_mean)
                loss_kl_stddev, loss_alpha_stddev = compute_parametric_kl_penalty_and_dual_loss(
                    kl_stddev, alpha_stddev, self._epsilon_stddev)
                loss_kl_penalty = loss_kl_mean + loss_kl_stddev
                loss_alpha_total = loss_alpha_mean + loss_alpha_stddev
            elif self._kl_constraint == "soft":
                # Soft constraint: decoupled Cross entropy & fixed (non-dual) KL coefficients.
                # "dual_alpha_*"" logs show fixed values
                alpha_mean = F.softplus(torch.full_like(
                    self.log_alpha_mean, self.args.log_alpha_mean_fixed)) + _MPO_FLOAT_EPSILON
                alpha_stddev = F.softplus(torch.full_like(
                    self.log_alpha_stddev, self.args.log_alpha_stddev_fixed)) + _MPO_FLOAT_EPSILON
            
                # Only the KL penalty is used; the dual losses are dropped
                loss_kl_mean, _ = compute_parametric_kl_penalty_and_dual_loss(
                    kl_mean, alpha_mean, self._epsilon_mean)
                loss_kl_stddev, _ = compute_parametric_kl_penalty_and_dual_loss(
                    kl_stddev, alpha_stddev, self._epsilon_stddev)
                loss_kl_penalty = loss_kl_mean + loss_kl_stddev
                loss_alpha_total = torch.zeros((), device=self.device)
            elif self._kl_constraint == "l2":
                # L2 M-step: Gaussian prior N(theta_target, beta^-1 I) on the actor params, i.e. the
                # Fisher in KL ~= 0.5 dtheta^T F dtheta replaced by identity. Constraint
                # 0.5 ||theta - theta_target||^2 <= epsilon_l2 with dual beta.
                l2_dist = 0.5 * sum(
                    torch.sum((p - p_t.detach()) ** 2)
                    for p, p_t in zip(self.actor.parameters(), self.actor_target.parameters()))
                # no separate mean / std dev KL or alpha for l2 regularization
                loss_kl_penalty, loss_alpha_total = compute_parametric_kl_penalty_and_dual_loss(
                    l2_dist.reshape(1, 1), beta, self._epsilon_l2)
            elif self._kl_constraint == "none":
                # Unregularized M-step: cross-entropy only, no KL / L2 trust region and no M-step dual.
                loss_kl_penalty = torch.zeros((), device=self.device)
                loss_alpha_total = torch.zeros((), device=self.device)
            else: # COUPLED mean and stddev KL constraint. 1 dual alpha for each dim.
                # Joint M-step: one per-dim dual alpha constraining the full per-dim KL.
                loss_kl_penalty, loss_alpha_total = compute_parametric_kl_penalty_and_dual_loss(
                    kl_joint, alpha, self._epsilon_kl)

            # Combine losses.
            loss_dual = loss_alpha_total + loss_temperature
            total_mpo_loss = (loss_policy + loss_kl_penalty + loss_dual)
                # DO i i need to do requires grad=True here or anywhere else..
            # critic_trainable_variables = self.__critic_network.trainable_variables

        # record stats
        mean_kl_mean_per_dim = kl_mean.mean(dim=0)
        mean_kl_stddev_per_dim = kl_stddev.mean(dim=0)

        pi_stddev = online_sigma
        pi_stddev_min_per_state = pi_stddev.min(dim=-1).values
        pi_stddev_max_per_state = pi_stddev.max(dim=-1).values

        # Kept as GPU tensors: .item() would sync the GPU on every update.
        # Keys ending in "_" are per-action-dim vectors, expanded by _stats_to_floats.
        policy_stats = {
            # ACME's loss_policy statistic is the complete MPO loss
            "total_mpo_loss": total_mpo_loss.detach().mean(),
            "loss_kl_penalty": loss_kl_penalty.detach().mean(),

            # Q-values: min/max over actions for each state, then average states.
            "q_min": q_values.detach().min(dim=0).values.mean(),
            "q_max": q_values.detach().max(dim=0).values.mean(),

            # Policy exploration
            "pi_stddev_min": pi_stddev_min_per_state.detach().mean(),
            "pi_stddev_max": pi_stddev_max_per_state.detach().mean(),
            "pi_stddev_cond": (pi_stddev_max_per_state / pi_stddev_min_per_state.clamp_min(_MPO_FLOAT_EPSILON)).detach().mean(),
            "pi_stddev_": pi_stddev.detach().mean(dim=0),
        }
        if self._variational == "parametric":
            # Only the parametric E-step duals are used; MPO temperature / M-step alphas are not logged
            policy_stats.update({
                "loss_policy_surrogate": loss_policy.detach().mean(),
                "loss_param_dual": loss_dual.detach().mean(),
                "param_eta_mu": param_eta_mu.detach().mean(),  # pyrefly: ignore[unbound-name]
                "param_eta_sigma": param_eta_sigma.detach().mean(),  # pyrefly: ignore[unbound-name]
                "param_kl_mean_rel": kl_mean.detach().mean() / self._param_epsilon_mu,
                "param_kl_stddev_rel": kl_stddev.detach().mean() / self._param_epsilon_sigma,
                "kl_joint": kl_joint.detach().mean(),
                "ratio_clip_frac": clip_frac,  # pyrefly: ignore[unbound-name]
                "param_eta_mu_": param_eta_mu.detach(),  # pyrefly: ignore[unbound-name]
                "param_eta_sigma_": param_eta_sigma.detach(),  # pyrefly: ignore[unbound-name]
            })
        else:
            policy_stats.update({
                "dual_alpha_mean": alpha_mean.detach().mean(),
                "dual_alpha_stddev": alpha_stddev.detach().mean(),
                "dual_temperature": temperature.detach().mean(),
                "loss_alpha": loss_alpha_total.detach().mean(),  # pyrefly: ignore[unbound-name]
                "loss_temperature": loss_temperature.detach().mean(),  # pyrefly: ignore[unbound-name]
                "loss_policy_cross_entropy": loss_policy.detach().mean(),

                # Relative KL measurements
                "kl_q_rel": kl_nonparametric.detach().mean() / self._epsilon,  # pyrefly: ignore[unbound-name]
                "kl_mean_rel": kl_mean.detach().mean() / self._epsilon_mean,
                "kl_stddev_rel": kl_stddev.detach().mean() / self._epsilon_stddev,

                # per dim vals
                "dual_alpha_mean_": alpha_mean.detach(),
                "dual_alpha_stddev_": alpha_stddev.detach(),
            })
            if self._kl_constraint == "joint":
                policy_stats["dual_alpha"] = alpha.detach().mean()
                policy_stats["kl_joint_rel"] = kl_joint.detach().mean() / self._epsilon_kl
                policy_stats["dual_alpha_"] = alpha.detach()
            if self._kl_constraint == "l2":
                policy_stats["dual_beta"] = beta.detach().mean()
                policy_stats["l2_dist"] = l2_dist.detach()  # pyrefly: ignore[unbound-name]
                policy_stats["l2_rel"] = l2_dist.detach() / self._epsilon_l2  # pyrefly: ignore[unbound-name]
            if self._action_penalization:
                policy_stats["penalty_kl_q_rel"] = (
                    penalty_kl_nonparametric.detach().mean() / self._epsilon_penalty)  # pyrefly: ignore[unbound-name]

        # Compute gradients 
        for optimizer in self.critic_optimizers:
            optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.dual_optimizer.zero_grad()
        if self._variational == "parametric":
            self.parametric_dual_optimizer.zero_grad()

        if self.l2_dual_optimizer is not None:
            self.l2_dual_optimizer.zero_grad()

        critic_loss.backward()   # retain because actor loss shares obs/critic graph? 
        total_mpo_loss.backward()               # policy + KL penalty + dual, one backward
        
        # clip
        for critic, optimizer in zip(self.critics, self.critic_optimizers):
            if self.args.grad_clip:
                nn.utils.clip_grad_norm_(critic.parameters(), self.args.max_grad_norm)
            optimizer.step()

        if self.args.grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)

        self.actor_optimizer.step()
        self.dual_optimizer.step()
        if self._variational == "parametric":
            self.parametric_dual_optimizer.step()
        if self.l2_dual_optimizer is not None:
            self.l2_dual_optimizer.step()

        # Losses to track.
        fetches = {
            'critic_loss': critic_loss_per_critic.detach().mean(),
            'policy_loss': total_mpo_loss.detach().mean(),

            **policy_stats,
        }
        if self.args.ensemble > 1:
            fetches["critic_loss_"] = critic_loss_per_critic.detach()
        return fetches

    def initialize_logging(self, info, append=False):
        self.start_time = time.time()

        log_dir = os.path.join(self.disk_folder, self.runs_directory, self.run_name)
        
        mode = "a" if append else "w"

        def _check_existing_header(path, fieldnames):
            with open(path, "r", newline="") as f:
                existing = next(csv.reader(f), None)
            if existing is None or existing == list(fieldnames):
                return
            raise ValueError(
                f"Cannot append to {path}: the run is being resumed with different settings than it was started with"
            )

        info_path = os.path.join(log_dir, "info_logs.csv")
        info_has_header = (
            append
            and os.path.exists(info_path)
            and os.path.getsize(info_path) > 0
        )
        self.keys_info = [k for k in info.keys() if not (k.startswith("bodies") or k.startswith("_"))]
        if info_has_header:
            _check_existing_header(info_path, ["step"] + self.keys_info)
        self.csv_file_info = open(os.path.join(log_dir, "info_logs.csv"), mode, newline="")
        self.writer_info = csv.DictWriter(self.csv_file_info, fieldnames=["step"] + self.keys_info)
        if not info_has_header:
            self.writer_info.writeheader()
        
        raw_action_path = os.path.join(log_dir,"raw_actions.csv")
        raw_action_has_header = (
            append
            and os.path.exists(raw_action_path)
            and os.path.getsize(raw_action_path) > 0)
        if raw_action_has_header:
            _check_existing_header(
                raw_action_path,
                ["step"] + [f"action_{i}" for i in range(self.act_dim)],
            )
        self.csv_file_raw_actions = open(raw_action_path,mode,newline="")
        self.writer_raw_actions = csv.writer(self.csv_file_raw_actions)
        if not raw_action_has_header:
            self.writer_raw_actions.writerow(
                ["step"]
                + [f"action_{i}" for i in range(self.act_dim)]
            )
        self.raw_action_buffer = []
        reward_log_env_id = self.args.env_id.replace("/", "_")
        self.reward_tracker = RewardTracker(
            env_dt=self.args.dt,
            env_id=reward_log_env_id,
            log_folder=log_dir,
            time_window=120.0,
        )
        performance_path = os.path.join(log_dir,"performance_variables.csv")
        performance_has_header = (
            append
            and os.path.exists(performance_path)
            and os.path.getsize(performance_path) > 0
        )
        if performance_has_header:
            _check_existing_header(performance_path, ["step"] + self.keys_agent_vars)
        self.csv_file_agent_vars = open(performance_path, mode, newline="")
        self.writer_agent_vars = csv.DictWriter(self.csv_file_agent_vars, fieldnames=["step"] + self.keys_agent_vars)
        if not performance_has_header:
            self.writer_agent_vars.writeheader()

        self.info_log_buffer = []
        self.agent_vars_buffer = []

    def log_step(self, global_step, infos, rewards, metrics=None, raw_action=None):
        if self.writer_info is None:
            return

        if "original_reward" in infos:
            tracked_reward = infos["original_reward"][0]
        else:
            tracked_reward = float(rewards.reshape(-1)[0].item())
        self.reward_tracker.update(tracked_reward)
        if raw_action is not None:
            if torch.is_tensor(raw_action):
                raw_action_np = (raw_action.detach().cpu().numpy())
            else:
                raw_action_np = np.asarray(raw_action)
            action_0 = raw_action_np[0]
            self.raw_action_buffer.append([global_step, *action_0.tolist()])
        # Write the accumulated actions only once every save interval.
        if (global_step % self.args.save_every_n_steps == 0
            and self.raw_action_buffer
        ):
            self.writer_raw_actions.writerows(self.raw_action_buffer)
            self.csv_file_raw_actions.flush()
            self.raw_action_buffer.clear()
        if global_step % self.args.log_every_n_steps != 0:
            return

        row = {"step": global_step}
        
        for k in self.keys_info:
            if k in infos:
                row[k] = arr_to_str(infos[k][0])
        
        self.info_log_buffer.append(row)


        if metrics is not None or self.args.eval:
            m = metrics if metrics is not None else {}
            elapsed_time = max(time.time() - self.start_time,1e-8)
            # global_step counts vector-environment steps, so multiply
            # by num_envs to obtain total environment transitions.
            sps = int(global_step * self.envs.num_envs / elapsed_time)
            reward_value = float(rewards.reshape(-1)[0].item())
            agent_vars_row = {
                "step": global_step,
                "critic_loss": m.get("critic_loss"),
                "policy_loss": m.get("policy_loss"),
                "dual_alpha_mean": m.get("dual_alpha_mean"),
                "dual_alpha_stddev": m.get("dual_alpha_stddev"),
                "dual_temperature": m.get("dual_temperature"),
                "loss_alpha": m.get("loss_alpha"),
                "loss_temperature": m.get("loss_temperature"),
                "loss_policy_cross_entropy": m.get("loss_policy_cross_entropy"),
                "loss_kl_penalty": m.get("loss_kl_penalty"),
                "kl_q_rel": m.get("kl_q_rel"),
                "kl_mean_rel": m.get("kl_mean_rel"),
                "kl_stddev_rel": m.get("kl_stddev_rel"),
                "q_min": m.get("q_min"),
                "q_max": m.get("q_max"),
                "pi_stddev_min": m.get("pi_stddev_min"),
                "pi_stddev_max": m.get("pi_stddev_max"),
                "pi_stddev_cond": m.get("pi_stddev_cond"),
                "utd": m.get("utd"),
                "SPS": sps,
                "average_reward_per_second": (self.reward_tracker.average_reward_per_second),
                "reward": reward_value,
                "mean_return": self.reward_tracker.mean_return,
            }
            for idx in range(self.act_dim):
                agent_vars_row[f"dual_alpha_mean_{idx}"] = m.get(f"dual_alpha_mean_{idx}")
                agent_vars_row[f"dual_alpha_stddev_{idx}"] = m.get(f"dual_alpha_stddev_{idx}")
                agent_vars_row[f"pi_stddev_{idx}"] = m.get(f"pi_stddev_{idx}")
            if self._kl_constraint == "joint":
                agent_vars_row["dual_alpha"] = m.get("dual_alpha")
                agent_vars_row["kl_joint_rel"] = m.get("kl_joint_rel")
                for idx in range(self.act_dim):
                    agent_vars_row[f"dual_alpha_{idx}"] = m.get(f"dual_alpha_{idx}")
            if self._kl_constraint == "l2":
                agent_vars_row["dual_beta"] = m.get("dual_beta")
                agent_vars_row["l2_dist"] = m.get("l2_dist")
                agent_vars_row["l2_rel"] = m.get("l2_rel")
            
            self.agent_vars_buffer.append(agent_vars_row)

        if global_step % self.args.save_every_n_steps == 0:
            self.reward_tracker.log()
            for row in self.info_log_buffer:
                self.writer_info.writerow(row)
            self.csv_file_info.flush()
            self.info_log_buffer = []

            for row in self.agent_vars_buffer:
                self.writer_agent_vars.writerow(row)
            self.csv_file_agent_vars.flush()
            self.agent_vars_buffer = []
    
    def save_checkpoint(self):
        checkpoint = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critics": [critic.state_dict() for critic in self.critics],
            "target_critics": [critic.state_dict() for critic in self.target_critics],

            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizers": [optimizer.state_dict() for optimizer in self.critic_optimizers],
            "dual_optimizer": self.dual_optimizer.state_dict(),

            "log_eta": self.log_eta.detach().cpu().clone(),
            "log_alpha_mean": (self.log_alpha_mean.detach().cpu().clone()),
            "log_alpha_stddev": (self.log_alpha_stddev.detach().cpu().clone()),
            "log_alpha": (self.log_alpha.detach().cpu().clone()),
            "log_beta": (self.log_beta.detach().cpu().clone()),
            "log_penalty_eta": (self.log_penalty_eta.detach().cpu().clone()),
            "global_step": self.global_step,
            "learner_step": self.learner_step,
            "target_policy_update_period": (self.target_policy_update_period),
            "target_critic_update_period": (self.target_critic_update_period),
        }
        if self.l2_dual_optimizer is not None:
            checkpoint["l2_dual_optimizer"] = self.l2_dual_optimizer.state_dict()
        if self._variational == "parametric":
            checkpoint["parametric_dual_optimizer"] = self.parametric_dual_optimizer.state_dict()
            checkpoint["param_log_eta_mu"] = self.param_log_eta_mu.detach().cpu().clone()
            checkpoint["param_log_eta_sigma"] = self.param_log_eta_sigma.detach().cpu().clone()
        if len(self.critics) == 1:
            checkpoint["critic"] = checkpoint["critics"][0]
            checkpoint["target_critic"] = checkpoint["target_critics"][0]
            checkpoint["critic_optimizer"] = checkpoint["critic_optimizers"][0]
        # Random generators, simulator/task state and the in-flight episode, so a
        # resume continues exactly where this step left off.
        checkpoint["rng"] = get_rng_state()
        checkpoint["envs"] = snapshot_envs(self.envs._env)
        checkpoint["obs"] = None if self.obs is None else self.obs.detach().cpu()
        checkpoint["pending_autoreset"] = self.pending_autoreset.detach().cpu()
        checkpoint["ep_returns"] = self._ep_returns.copy()
        checkpoint["learner_update_budget"] = self.learner_update_budget
        torch.save(checkpoint, os.path.join(self.weights_folder, f"checkpoint_{self.global_step}.pth"))
        self.rb.save(os.path.join(self.weights_folder, "replay_buffer.npz"))
        if self.args.save_rb_ckpt:
            self.rb.save(os.path.join(self.weights_folder, f"replay_buffer_{self.global_step}.npz"))

    def load_checkpoint(self, weights_path, checkpoint_step=None):
        weights_path = os.path.abspath(weights_path.rstrip("/"))
        if os.path.basename(weights_path) != "weights_and_args":
            nested = os.path.join(weights_path, "weights_and_args")
            if os.path.isdir(nested):
                weights_path = nested
        checkpoint_files = [f for f in os.listdir(weights_path) if f.endswith(".pth")]
        checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

        if not checkpoint_files:
            raise FileNotFoundError(
                f"No .pth checkpoints in {weights_path}. "
                "--weights_path must be a weights_and_args directory."
            )

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
        # weights_only=False: the checkpoint also holds numpy RNG and simulator state.
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])

        def _checked_list(plural, singular, live, what):
            saved = checkpoint[plural] if plural in checkpoint else [checkpoint[singular]]
            if len(saved) != len(live):
                raise ValueError(
                    f"Ensemble size mismatch for {what}: the checkpoint at {checkpoint_path} stores {len(saved)} but this run was saved with --ensemble {len(live)}."
                )
            return saved

        for critic, state_dict in zip(
            self.critics, _checked_list("critics", "critic", self.critics, "critics")
        ):
            critic.load_state_dict(state_dict)
        for target_critic, state_dict in zip(
            self.target_critics,
            _checked_list("target_critics", "target_critic", self.target_critics, "target critics"),
        ):
            target_critic.load_state_dict(state_dict)
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        for optimizer, state_dict in zip(
            self.critic_optimizers,
            _checked_list(
                "critic_optimizers", "critic_optimizer", self.critic_optimizers, "critic optimizers"
            ),
        ):
            optimizer.load_state_dict(state_dict)
        self.dual_optimizer.load_state_dict(checkpoint["dual_optimizer"])
        if self.l2_dual_optimizer is not None and "l2_dual_optimizer" in checkpoint:
            self.l2_dual_optimizer.load_state_dict(checkpoint["l2_dual_optimizer"])
        if self._variational == "parametric" and "parametric_dual_optimizer" in checkpoint:
            self.parametric_dual_optimizer.load_state_dict(checkpoint["parametric_dual_optimizer"])

        with torch.no_grad():
            if "log_eta" in checkpoint:
                self.log_eta.copy_(torch.as_tensor(checkpoint["log_eta"],dtype=self.log_eta.dtype,device=self.device).reshape_as(self.log_eta))
            if "log_alpha_mean" in checkpoint:
                self.log_alpha_mean.copy_(torch.as_tensor(checkpoint["log_alpha_mean"],dtype=self.log_alpha_mean.dtype,device=self.device).reshape_as(self.log_alpha_mean))
            if "log_alpha_stddev" in checkpoint:
                self.log_alpha_stddev.copy_(torch.as_tensor(checkpoint["log_alpha_stddev"],dtype=self.log_alpha_stddev.dtype,device=self.device).reshape_as(self.log_alpha_stddev))
            if "log_alpha" in checkpoint:
                self.log_alpha.copy_(torch.as_tensor(checkpoint["log_alpha"],dtype=self.log_alpha.dtype,device=self.device).reshape_as(self.log_alpha))
            if "log_beta" in checkpoint:
                self.log_beta.copy_(torch.as_tensor(checkpoint["log_beta"],dtype=self.log_beta.dtype,device=self.device).reshape_as(self.log_beta))
            if "log_penalty_eta" in checkpoint:
                self.log_penalty_eta.copy_(torch.as_tensor(checkpoint["log_penalty_eta"],dtype=self.log_penalty_eta.dtype,device=self.device,).reshape_as(self.log_penalty_eta))
            if self._variational == "parametric":
                if "param_log_eta_mu" in checkpoint:
                    self.param_log_eta_mu.copy_(torch.as_tensor(checkpoint["param_log_eta_mu"],dtype=self.param_log_eta_mu.dtype,device=self.device).reshape_as(self.param_log_eta_mu))
                if "param_log_eta_sigma" in checkpoint:
                    self.param_log_eta_sigma.copy_(torch.as_tensor(checkpoint["param_log_eta_sigma"],dtype=self.param_log_eta_sigma.dtype,device=self.device).reshape_as(self.param_log_eta_sigma))
        self.global_step = int(checkpoint.get("global_step", 0))
        self.learner_step = int(checkpoint.get("learner_step", 0))
        print(f"[√] Loaded checkpoint from {weights_path}, global_step={self.global_step}")

        step_buffer_path = os.path.join(weights_path, f"replay_buffer_{self.global_step}.npz")
        buffer_path = (step_buffer_path if os.path.exists(step_buffer_path)
                       else os.path.join(weights_path, "replay_buffer.npz"))
        if os.path.exists(buffer_path):
            self.rb.load(buffer_path, self.device)
            print(f"[√] Loaded replay buffer {os.path.basename(buffer_path)}")

        # Continue the saved episode. Runs after envs.reset(), so the reset is overwritten.
        if checkpoint.get("envs") is not None and restore_envs(self.envs._env, checkpoint["envs"]):
            self.obs = checkpoint["obs"].to(self.device)
            self.pending_autoreset = checkpoint["pending_autoreset"].to(self.device)
            self._ep_returns = checkpoint["ep_returns"]
            print("[√] Restored simulator and task state")
        if "rng" in checkpoint:
            self.learner_update_budget = checkpoint["learner_update_budget"]
            set_rng_state(checkpoint["rng"])
            print("[√] Restored Python, NumPy, torch and CUDA random states")

    def cleanup(self):
        if self.writer_raw_actions is not None and self.raw_action_buffer:
            self.writer_raw_actions.writerows(self.raw_action_buffer)
            self.csv_file_raw_actions.flush()
            self.raw_action_buffer.clear()
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
        if self.csv_file_raw_actions is not None:
            self.csv_file_raw_actions.close()
        if self.csv_file_agent_vars:
            self.csv_file_agent_vars.close()
        if self.envs:
            self.envs.close()

def _stats_to_floats(stats):
    """GPU stat tensors -> floats; keys ending in "_" become one entry per action dim."""
    out = {}
    for k, v in stats.items():
        v = v.detach().float().cpu()
        if k.endswith("_"):
            for j, x in enumerate(v.reshape(-1).tolist()):
                out[f"{k}{j}"] = x
        else:
            out[k] = v.item()
    return out

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

  # divide Q-values by temp
  tempered_q_values = q_values.detach() / temperature

  # Compute the normalized importance weights (weights of actions that online policy should update toward)
  normalized_weights = F.softmax(tempered_q_values, dim=0).detach()

  # Compute the temperature loss (dual of the E-step optimization problem).
  q_logsumexp = torch.logsumexp(tempered_q_values, dim=0)
  log_num_actions = math.log(q_values.shape[0])
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

  # return the mean loss over batch of states = b
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

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="mpo_ant")
    parser.add_argument("--runs_directory", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch_deterministic", action="store_true", default=False)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--cuda_graph", action=argparse.BooleanOptionalAction, default=True,
                        help="Capture the learner update as a CUDA graph (GPU only)")
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--capture_video_steps", type=int, default=None,
                        help="If set, record capture_video_steps-long clips (see --capture_video_every) "
                             "instead of the recurring save_every_n_steps clips.")
    parser.add_argument("--capture_video_every", type=int, default=None,
                        help="Start a capture_video_steps-long clip every N env steps (default: only at step 0)")
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--save_every_n_steps", type=int, default=4000)
    parser.add_argument("--save_rb_ckpt", action="store_true", default=False,
                        help="Also save replay_buffer_{step}.npz alongside every checkpoint_{step}.pth")
    parser.add_argument("--log_every_n_steps", type=int, default=4000)
    parser.add_argument("--critic_type", type=str, default="scalar", choices=["scalar", "categorical", "quantile"], help="Critic type: scalar MPO, categorical/C51 DMPO, or quantile-regression distributional critic")
    parser.add_argument("--ensemble", type=int, default=1, help="Number of critics in the ensemble, where 1 reproduces the single-critic agent")

    parser.add_argument(
        "--env_id",
        type=str,
        default="EAnt",
        help="Embodied Ant id (EAnt / SimEmbodiedAnt / HwEmbodiedAnt) or Gymnasium "
             "MuJoCo id (Hopper-v5, Walker2d-v5, Humanoid-v5, Ant-v5, ...)",
    )
    parser.add_argument("--total_timesteps", type=int, default=120_000)
    parser.add_argument("--num_envs", type=int, default=1)

    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    parser.add_argument("--batch_size", type=int, default=512)

    # Categorical distributional critic
    parser.add_argument("--vmin", type=float, default=-500.0,
                        help="Minimum atom value for distributional critic")
    parser.add_argument("--vmax", type=float, default=50.0,
                        help="Maximum atom value for distributional critic")
    parser.add_argument("--num_atoms", type=int, default=101,
                        help="Number of categorical atoms for distributional critic")

    # Quantile-regression distributional critic
    parser.add_argument("--num_quantiles", type=int, default=201,
                        help="Number of learned return quantiles for quantile critic")
    parser.add_argument("--quantile_target_action_num", type=int, default=4,
                        help="Number of sampled next-actions (M) averaged over when computing the quantile critic TD-target loss")
    parser.add_argument("--quantile_huber_param", type=float, default=1.0,
                        help="Huber threshold for quantile regression; 0 uses absolute loss")
    parser.add_argument("--sum_over_quantiles", action=argparse.BooleanOptionalAction, default=True,
                        help="Sum the quantile-huber loss over quantiles before averaging over batch, vs. averaging over both")

    parser.add_argument("--log_interval", type=int, default=100,
                        help="env steps between TensorBoard scalar writes")
    # MPO
    parser.add_argument("--epsilon_eta", type=float, default=1e-1,
                        help="epsilon for E-step temperature dual")
    parser.add_argument("--epsilon_mu_kl", type=float, default=2.5e-3,
                        help="epsilon_mu for M-step mean KL")
    parser.add_argument("--epsilon_sigma_kl", type=float, default=1e-6,
                        help="epsilon_sigma for M-step covariance KL")
    parser.add_argument("--kl_constraint", type=str, default="decoupled", choices=["decoupled", "joint", "soft", "l2", "none"],
                        help="M-step regularizer (CE is set separately by --ce_coupling). 'decoupled': separate per-dim alpha_mu/alpha_sigma; "
                             "'joint': one per-dim alpha on the full per-dim KL; "
                             "'soft': fixed alpha_mu/alpha_sigma (no dual update); "
                             "'l2': one dual beta on 0.5*||theta - theta_target||^2 over actor params; "
                             "'none': CE only, no M-step trust region")
    parser.add_argument("--ce_coupling", type=str, default="decoupled", choices=["decoupled", "coupled"],
                        help="M-step cross-entropy, independent of --kl_constraint: 'decoupled' (mean fit with target stddev, "
                             "stddev fit with target mean) or 'coupled' (full online policy)")

    # if solving soft KL constraint in M-step, dual variables log alpha are fixed. 
    parser.add_argument("--log_alpha_mean_fixed", type=float, default=-17.0,
                        help="fixed log KL coefficient on the mean (only with --kl_constraint soft)")
    parser.add_argument("--log_alpha_stddev_fixed", type=float, default=935.0,
                        help="fixed log KL coefficient on the stddev (only with --kl_constraint soft)")
    # if solving coupled mean & stddev KL constraint in M-step:
    parser.add_argument("--epsilon_kl", type=float, default=2.5e-3,
                        help="per-dim epsilon for the joint M-step KL (only with --kl_constraint joint)")

    # E-step variational distribution: nonparametric (sample weights) or parametric Gaussian q.
    parser.add_argument("--variational_distribution", type=str, default="nonparametric",
                        choices=["nonparametric", "parametric"],
                        help="'nonparametric': softmax(Q/eta) sample weights + M-step fit; "
                             "'parametric': online actor is q, trained with KL(pi_target || q) duals")
    parser.add_argument("--parametric_init_log_eta_mu", type=float, default=10.0,
                        help="init per-dim log eta_mu for the parametric E-step mean KL")
    parser.add_argument("--parametric_init_log_eta_sigma", type=float, default=1000.0,
                        help="init per-dim log eta_sigma for the parametric E-step stddev KL")
    parser.add_argument("--parametric_epsilon_mu", type=float, default=2.5e-3,
                        help="per-dim epsilon_mu for the parametric E-step mean KL")
    parser.add_argument("--parametric_epsilon_sigma", type=float, default=1e-6,
                        help="per-dim epsilon_sigma for the parametric E-step stddev KL")
    parser.add_argument("--epsilon_clip", type=float, default=0.2,
                        help="PPO-style clip on the parametric E-step ratio q/pi_target to [1-eps, 1+eps]")
  
    parser.add_argument("--sample_action_num", type=int, default=64,
                        help="actions sampled per state in E-step")
    # parser.add_argument("--mstep_iteration_num", type=int, default=4,
    #                     help="actor gradient steps per learn() call")
    parser.add_argument("--dual_lr", type=float, default=1e-3)
   
    parser.add_argument("--grad_clip", action=argparse.BooleanOptionalAction, default=True,
                        help="Clip actor and critic gradient norms to --max_grad_norm (duals are never clipped)")
    parser.add_argument("--max_grad_norm", type=float, default=40.0)

    # true learning start = learning start // num_envs (floor division)
    parser.add_argument("--learning_starts", type=int, default=200)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--q_lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--actor_net", type=str, default="acme", choices=["acme", "default"],
                        help="'acme': AcmeActor; 'default': Actor (ReLU torso, PyTorch default init, same mean/stddev head math)")
    parser.add_argument("--critic_net", type=str, default="acme", choices=["acme", "default"],
                        help="'acme': Acme critics; 'default': Critic (ReLU torso, PyTorch default init). 'default' needs --critic_type scalar")
    # layer norm arg is only effective with --actor_net default / --critic_net default
    parser.add_argument("--use_layer_norm", action=argparse.BooleanOptionalAction, default=True) 
    parser.add_argument("--clip_action_critic", action=argparse.BooleanOptionalAction, default=True,
                        help="Clamp actions to the action bounds before feeding them to the critic")

    parser.add_argument("--policy_layer_sizes",type=int,nargs="+",default=[256, 256, 256])
    parser.add_argument("--critic_layer_sizes",type=int,nargs="+",default=[512, 512, 256])
    parser.add_argument("--td_horizon", type=int, default=5,
                        help="number of steps collapsed into each replay transition")
    
    parser.add_argument("--checkpoint_step",type=int,default=None,
                            help="Specific checkpoint step to load, e.g. 95000. If omitted, loads latest.")

    parser.add_argument("--target_policy_update_period",type=int,default=100,
                        help="Hard-copy online actor to target actor every N learner updates")
    parser.add_argument("--target_critic_update_period",type=int,default=100,
                        help="Hard-copy online critic to target critic every N learner updates",)
    parser.add_argument("--init_log_temperature", type=float, default=10.0)
    parser.add_argument("--init_log_alpha_mean", type=float, default=10.0)
    parser.add_argument("--init_log_alpha_stddev", type=float, default=1000.0)
    parser.add_argument("--init_log_alpha", type=float, default=10.0,
                        help="init log dual for the coupled mean stddev KL (only with --kl_constraint joint)")
    # if solving the L2 (parameter-space) constraint in M-step:
    parser.add_argument("--epsilon_l2", type=float, default=5e-2,
                        help="bound on 0.5*||theta - theta_target||^2 over actor params (only with --kl_constraint l2)")
    parser.add_argument("--init_log_beta", type=float, default=0.0,
                        help="init log dual for the L2 M-step constraint (only with --kl_constraint l2)")

    # Environment.
    parser.add_argument("--dt", type=float, default=0.12)
    parser.add_argument("--hw_config", type=str, default=None)
    parser.add_argument("--render_mode", type=str, default=None)
    parser.add_argument("--terminate_on_upside_down", type=bool, default=True)
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--resume_in_place", action="store_true", default=False)
    parser.add_argument("--task_type", type=str, default="back_and_forth",
                        choices=["forward", "back_and_forth"])
    parser.add_argument("--radius_back_and_forth", type=float, default=0.3)
    parser.add_argument("--origin_back_and_forth", type=float, nargs=2, default=[0.7, -0.25])
    parser.add_argument(
        "--reward_scale",
        type=float,
        default=None,
        help="Reward multiplier (default: 100 for embodied Ant, 1 for Gymnasium envs)",
    )
    parser.add_argument("--model_path", type=str,
                        default="../../sim/assets/open-ant/ant_sim_1.xml")

    parser.add_argument("--samples_per_insert",type=float,default=1536.0,
                        help="Replay samples consumed per valid environment transition")

    parser.add_argument("--action_penalization",action=argparse.BooleanOptionalAction,default=True,
                        help="MO-MPO action penalization for pi-target samples outside bounds")
    parser.add_argument("--epsilon_penalty",type=float,default=1e-3,
                        help="KL constraint for action penalization")

    # actor and critic network initializations
    parser.add_argument("--policy_init_scale", type=float, default=0.7)
    parser.add_argument("--policy_min_scale", type=float, default=1e-4)

    args = parser.parse_args(argv)

    assert args.ensemble >= 1

    return args

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

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    disk_folder = ""
    if args.resume_in_place:
        if args.weights_path is None:
            raise ValueError("--resume_in_place requires --weights_path")

        weights_path = os.path.abspath(args.weights_path.rstrip("/"))
        if os.path.basename(weights_path) != "weights_and_args":
            raise ValueError("--weights_path must point to the weights_and_args directory ")
        run_dir = os.path.dirname(weights_path)

        args.runs_directory = os.path.dirname(run_dir)
        run_name = os.path.basename(run_dir)

        print(f"[√] Resuming in existing run directory: {run_dir}")
    else:
        os.makedirs(args.runs_directory, exist_ok=True)
        run_name = f"{args.exp_name}_{date}_seed_{args.seed}"
        os.makedirs(os.path.join(args.runs_directory, run_name), exist_ok=True)

    task = None
    print(f"args.env_id: {args.env_id}")
    if args.task_type == "forward":
        task = ForwardTask()
    elif args.task_type == "back_and_forth":
        RADIUS = args.radius_back_and_forth
        ORIGIN = np.array(args.origin_back_and_forth)
        task = BackAndForthTask(radius=RADIUS, origin=ORIGIN)
        print(f"BackAndForthTask: radius={RADIUS}, origin={ORIGIN}")
    else:
        raise ValueError(f"Invalid task type: {args.task_type}")

    raw_env, envs = make_envs(args, task, disk_folder, run_name, runs_directory=args.runs_directory)
    
    print("\n========== LOADED ENVIRONMENT ==========")
    print("env_id:", args.env_id)
    print("observation space:", raw_env.single_observation_space)
    print("action space:", raw_env.single_action_space)
    print("dt:", args.dt)
    agent = MPO(args=args,envs=envs,disk_folder=disk_folder,run_name=run_name,runs_directory=args.runs_directory)
    
    save_git_info(
        os.path.join(args.runs_directory, run_name),
        os.path.dirname(__file__),
    )
    # Reset before loading, so load_checkpoint can restore the saved episode over it.
    obs, info = envs.reset()
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

    agent.initialize_logging(info, append=args.resume_in_place)
    solved_tracker = None
    if args.env_id.startswith("Pusher"):
        solved_tracker = PuckSolvedTracker(agent.envs._env, agent.reward_tracker.log_folder)
    
    if args.resume_in_place:
        # RewardTracker has its own counter. Continue it from the last row already
        # written to its CSV (not from agent.global_step) because global_step can
        # include an offset inherited from an earlier phase (e.g. sim1 weights
        # loaded before a continual-learning run), while the CSV's own "step"
        # column is always local to this run.
        reward_csv_path = agent.reward_tracker.csv_path
        last_local_step = 0.0
        if os.path.exists(reward_csv_path):
            with open(reward_csv_path, "r", newline="") as f:
                rows = list(csv.reader(f))
            if len(rows) > 1:
                last_local_step = float(rows[-1][0])
        agent.reward_tracker.step = last_local_step

    # try:
    #     from torch.utils.tensorboard import SummaryWriter
    #     os.makedirs(agent.log_dir, exist_ok=True)
    #     writer = SummaryWriter(agent.log_dir)
    # except ImportError:
    #     writer = None
    #     print("[!] tensorboard not available, printing only")
    writer = None

    #time_start_learning = time.time()
    step_times = []

    try:
        for step in tqdm(range(agent.global_step, args.total_timesteps)):
            time_now = time.time()
            selected_actions = agent.get_action(obs, args.eval)
            next_obs, rewards, terminations, truncations, infos = envs.step(selected_actions)

            if args.eval:
                agent.agent_step_eval(next_obs, rewards, terminations, truncations, infos)
                metrics = None
            else:
                metrics = agent.agent_step(next_obs, selected_actions, rewards, terminations, truncations, infos)
            
            current_step = agent.global_step
            agent.log_step(current_step, infos, rewards, metrics, raw_action=selected_actions)
            if solved_tracker is not None and (terminations[0] or truncations[0]):
                solved_tracker.end_episode(current_step)

            # TENSORBOARD THINGS
            # if writer is not None and current_step % args.log_interval == 0:
            #     writer.add_scalar("reward/mean", float(rewards.mean()), current_step)
            #     if "benchmark_reward" in infos:
            #         writer.add_scalar("reward/benchmark",
            #                             float(np.asarray(infos["benchmark_reward"]).mean()), current_step)
            #     writer.add_scalar("perf/buffer_size", agent.rb.size(), current_step)
            # if metrics is not None:
            #     for k, v in metrics.items():
            #         if isinstance(v, (int, float)):
            #             writer.add_scalar(f"agent/{k}", v, agent.learner_step)
    
            # checkpoints (<run_dir>/weights_and_args) 
            if (not args.eval and current_step % args.save_every_n_steps == 0):
                agent.save_checkpoint()
                if writer is not None:
                    writer.flush()
            step_times.append(f"{current_step},{time.time() - time_now}\n")

        # with open(os.path.join(args.runs_directory, run_name, "step_times.csv"), "w") as f:
        #     f.writelines(step_times)
    finally:
        step_times_mode = "a" if args.resume_in_place else "w"

        with open(os.path.join(args.runs_directory,run_name,"step_times.csv"),step_times_mode) as f:
            f.writelines(step_times)

        if not args.eval:
            agent.save_checkpoint()

        if writer is not None:
            writer.close()
    
    agent.cleanup()
if __name__ == "__main__":
    main()
