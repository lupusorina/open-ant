import os
import csv
import sys
import copy
import json
import math
import time
import random
import argparse
import numpy as np
from collections import deque
try:
    import wandb
except ImportError:
    wandb = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
except ImportError:
    pynvml = None

from tqdm import tqdm
import gymnasium as gym
from datetime import datetime
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import make_ant_env, ForwardTask, BackAndForthTask
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from reward import RewardTracker
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.buffers import ReplayBuffer, NStepReplayBufferSamples


def arr_to_str(x):
    if isinstance(x, np.ndarray):
        return "[" + " ".join(map(str, x.tolist())) + "]"
    return x

class Actor(nn.Module):
    def __init__(self,
                 env,
                 hidden_dims: List[int] = [256, 256],
                 use_layer_norm: bool = False,
                 min_scale: float = 1e-3,
                 init_scale: float = 0.5,):
        super().__init__()
        obs_dim = int(np.array(env.single_observation_space.shape).prod())
        act_dim = int(np.prod(env.single_action_space.shape))
        self.min_scale = min_scale

        self.register_buffer("action_low",torch.tensor(env.single_action_space.low, dtype=torch.float32))
        self.register_buffer("action_high",torch.tensor(env.single_action_space.high, dtype=torch.float32))
        
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

        self._softplus_bias = float(np.log(np.exp(init_scale - min_scale) - 1.0))

        self.apply(self._init_weights)
        with torch.no_grad():
            self.log_sigma_head.bias.fill_(self._softplus_bias)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> dist.Independent:
        logits = self.net(x)
        mu = self.action_low + (self.action_high - self.action_low) * torch.sigmoid(self.mu_head(logits))
        sigma = F.softplus(self.log_sigma_head(logits)) + self.min_scale
        return dist.Independent(dist.Normal(mu, sigma), 1)

    def get_action(self, obs: torch.Tensor, n_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.forward(obs)
        actions = d.rsample((n_samples,))
        actions = torch.clamp(actions, self.action_low, self.action_high)
        log_probs = d.log_prob(actions) #TODO: verify log prob consistency
        actions = actions.permute(1, 0, 2)  # (batch, n_samples, act_dim)
        log_probs = log_probs.transpose(0, 1).unsqueeze(-1)
        return actions, log_probs, d.mean

    def get_log_probs(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        d = self.forward(obs)
        return d.log_prob(action).unsqueeze(-1)


class Critic(nn.Module):
    def __init__(
        self,
        env,
        hidden_dims: List[int] = [256, 256],
        use_layer_norm: bool = False,
        use_gru: bool = False,
        history_len: int = 1,
        gru_hidden_dim: int = 128,
        shortcut_dim: int = 128,
    ):
        super().__init__()

        obs_dim = int(np.array(env.single_observation_space.shape).prod())
        act_dim = int(np.prod(env.single_action_space.shape))

        self.use_gru = use_gru
        self.history_len = history_len
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.shortcut_dim = shortcut_dim

        if self.use_gru:
            assert history_len > 1, "use_gru=True requires history_len > 1"
            assert obs_dim % history_len == 0, (
                f"obs_dim={obs_dim} must be divisible by history_len={history_len}"
            )
            self.raw_obs_dim = obs_dim // history_len

            # batch_first=False, GRU expects: (history_len, B, raw_obs_dim)
            self.gru = nn.GRU(
                input_size=self.raw_obs_dim,
                hidden_size=gru_hidden_dim,
                num_layers=1,
                batch_first=False,
            )
            # Shortcut for current obs + current action. first concatenate O_t, A_t
            shortcut_layers = [nn.Linear(self.raw_obs_dim + act_dim, shortcut_dim),]
            if use_layer_norm:
                shortcut_layers.append(nn.LayerNorm(shortcut_dim))
            shortcut_layers.append(nn.ReLU())

            self.current_shortcut_embedder = nn.Sequential(*shortcut_layers)

            # Final Q MLP input = GRU hidden feature + shortcut(O_t, A_t) 
            prev = gru_hidden_dim + shortcut_dim
        else:
            self.raw_obs_dim = obs_dim
            self.gru = None
            self.current_shortcut_embedder = None
            prev = obs_dim + act_dim

        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h

        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

        self.apply(self._init_weights)
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0.0)

    # Convert observ. sequence into GRU tensor. 
    # input = state = (B, history_len * raw obs dim). Output = (history_len, B, raw_obs_dim)
    # i.e. state = [o_{t-H+1}, ..., o_t], then seq[0] = o_{t-H+1} 
    def _as_sequence(self, state: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]

        # reshape to time-first sequence: (history_len, B, raw_obs_dim)
        return state.reshape(B, self.history_len, self.raw_obs_dim).permute(1, 0, 2)

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """ Encode stacked history. state = (B, history_len * raw_obs_dim)
            seq = (history_len, B, raw_obs_dim). h_n = (num_layers, B, gru_hidden_dim)
            history_feat: (B, gru_hidden_dim)
        """
        if not self.use_gru:
            return state

        seq = self._as_sequence(state)

        # h_n = final hidden state features @ last timestep of GRU for every GRU layer
        # Since num_layers=1, h_n[-1] has shape (B, gru_hidden_dim) 
        _, h_n = self.gru(seq)

        return h_n[-1]

    def get_current_obs(self, state: torch.Tensor) -> torch.Tensor:
        """ Get the latest/current raw observation o_t from flat stacked history.
        Input: state: (B, history_len * raw_obs_dim)  
        Output: current_obs: (B, raw_obs_dim)
        """
        if not self.use_gru:
            return state
        B = state.shape[0]

        # (B, history_len * raw_obs_dim) --> (B, history_len, raw_obs_dim)
        seq_batch_first = state.reshape(B, self.history_len, self.raw_obs_dim)

        # latest observation o_t
        current_obs = seq_batch_first[:, -1, :]
        return current_obs

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # history_feat = GRU([o_{t-H+1}, ..., o_t]), current_feat = MLP([o_t, action])
        # Q input = [history_feat, current_feat] 
        if not self.use_gru:
            x = torch.cat([state, action], dim=-1)
            return self.net(x)

        history_feat = self.encode_state(state)       # (B, gru_hidden_dim)
        current_obs = self.get_current_obs(state)     # (B, raw_obs_dim)

        shortcut_input = torch.cat([current_obs, action], dim=-1)
        current_feat = self.current_shortcut_embedder(shortcut_input)

        x = torch.cat([history_feat, current_feat], dim=-1)
        return self.net(x)


class FlattenObsHistory(gym.Wrapper):
    def __init__(self, env, history_len: int):
        super().__init__(env)
        self.history_len = history_len
        self.hist = deque(maxlen=history_len)

        old_space = env.observation_space
        assert isinstance(old_space, gym.spaces.Box)

        low = np.tile(old_space.low, history_len).astype(np.float32)
        high = np.tile(old_space.high, history_len).astype(np.float32)

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def _get_stacked_obs(self):
        return np.concatenate(list(self.hist), axis=-1).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = np.asarray(obs, dtype=np.float32)

        self.hist.clear()
        for _ in range(self.history_len):
            self.hist.append(obs.copy())

        return self._get_stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.asarray(obs, dtype=np.float32)

        self.hist.append(obs.copy())

        return self._get_stacked_obs(), reward, terminated, truncated, info

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

            if args.history_len > 1:
                env = FlattenObsHistory(env, args.history_len)
            return env
        return _init

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "[!] Only continuous action space is supported."
    print(f"[√] Created environment with {envs.num_envs} environments.")
    return envs

def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_system_metrics(device):
    metrics = {}

    if psutil is not None:
        metrics["system/cpu_percent"] = psutil.cpu_percent(interval=None)
        metrics["system/ram_used_mb"] = psutil.virtual_memory().used / (1024 ** 2)
        metrics["system/ram_percent"] = psutil.virtual_memory().percent

    if device.type == "cuda" and torch.cuda.is_available():
        metrics["system/gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024 ** 2)
        metrics["system/gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024 ** 2)
        metrics["system/gpu_max_memory_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

                metrics["system/gpu_util_percent"] = util.gpu
                metrics["system/gpu_mem_util_percent"] = util.memory
                metrics["system/gpu_memory_used_mb"] = mem.used / (1024 ** 2)
                metrics["system/gpu_memory_total_mb"] = mem.total / (1024 ** 2)

                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    metrics["system/gpu_power_watts"] = power_mw / 1000.0
                except Exception:
                    pass

            except Exception:
                pass

    return metrics


class MPO:
    def __init__(self, args, envs, disk_folder='', run_name=None, runs_directory='runs'):
        self.args = args
        self.envs = envs
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        print(f"[√] Using device: {self.device}")
        self.dt = self.args.dt

        self.disk_folder = disk_folder
        self.run_name = run_name
        self.runs_directory = runs_directory
        self.weights_folder = os.path.join(disk_folder, runs_directory, run_name, "weights_and_args")
        os.makedirs(self.weights_folder, exist_ok=True)
        with open(os.path.join(self.weights_folder, "args.json"), 'w') as f:
            json.dump(args.__dict__, f)
        
        self.use_wandb = bool(args.track_wandb)

        if self.use_wandb:
            if wandb is None:
                raise ImportError(
                    "You passed --track_wandb, but wandb is not installed. "
                    "Install it with: pip install wandb"
                )

            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name or run_name,
                group=args.wandb_group,
                config=vars(args),
                mode=args.wandb_mode,
                dir=os.path.join(disk_folder, runs_directory, run_name),
            )
        else:
            self.use_wandb = False

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        torch.backends.cudnn.benchmark = not args.torch_deterministic

        hidden_dims = [args.hidden_dim] * args.n_hidden_layers

        self.actor = Actor(envs, hidden_dims=hidden_dims, use_layer_norm=args.use_layer_norm).to(self.device)
        self.actor_target = Actor(envs, hidden_dims=hidden_dims, use_layer_norm=args.use_layer_norm).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        for p in self.actor_target.parameters():
            p.requires_grad = False
        
        self.critics = nn.ModuleList([
            Critic(
                envs,
                hidden_dims=hidden_dims,
                use_layer_norm=args.use_layer_norm,
                use_gru=args.critic_use_gru,
                history_len=args.history_len,
                gru_hidden_dim=args.critic_gru_hidden_dim,
            )
            for _ in range(self.args.ensemble)
        ]).to(self.device)
        self.target_critics = copy.deepcopy(self.critics)
        for p in self.target_critics.parameters():
            p.requires_grad = False
            
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.policy_lr)
        self.critic_optimizers = [optim.Adam(c.parameters(), lr=args.q_lr) for c in self.critics]

        self.actor_param_count = count_params(self.actor)
        self.critic_param_count = sum(count_params(c) for c in self.critics)
        self.total_train_param_count = self.actor_param_count + self.critic_param_count

        print(f"[model] actor params: {self.actor_param_count}")
        print(f"[model] critic params: {self.critic_param_count}")
        print(f"[model] total train params: {self.total_train_param_count}")

        if self.use_wandb:
            wandb.log({
                "model/actor_params": self.actor_param_count,
                "model/critic_params": self.critic_param_count,
                "model/total_train_params": self.total_train_param_count,
                "model/ensemble": self.args.ensemble,
            }, step=0)

            if self.args.wandb_watch:
                wandb.watch(self.actor, log="gradients", log_freq=max(100, self.args.log_every_n_steps))

        self.policy_learning_starts = args.policy_learning_starts

        # Dual variable for E-step temperature (Adam state stored as raw tensors, no autograd).
        self.log_eta = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.dual_m = torch.zeros((), dtype=torch.float32, device=self.device)
        self.dual_v = torch.zeros((), dtype=torch.float32, device=self.device)
        self.dual_beta1_pow = torch.ones((), dtype=torch.float32, device=self.device)
        self.dual_beta2_pow = torch.ones((), dtype=torch.float32, device=self.device)

        # Scalar Lagrange multipliers for M-step KL constraints.
        self.alpha_mu = 1e-3 #if 0.0 first policy imporvement steps are unconstrained
        self.alpha_sigma = 1e-3

        self.obs_dim = int(np.array(self.envs.single_observation_space.shape).prod())
        self.act_dim = int(np.prod(self.envs.single_action_space.shape))
        self.envs.single_observation_space.dtype = np.float32
        self.action_low = envs.single_action_space.low.astype(np.float32)
        self.action_high = envs.single_action_space.high.astype(np.float32)
        assert np.all(np.isfinite(envs.single_action_space.low)) and np.all(np.isfinite(envs.single_action_space.high))
        self._uniform_log_prob = -float(np.sum(np.log(self.action_high - self.action_low)))

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

        self.global_step = 0
        self.obs = None
        self.last_actions = None
        self.last_log_probs = None

        # Logging state.
        self.start_time = None
        self.reward_tracker = None
        self.csv_file_info = None
        self.csv_file_agent_vars = None
        self.writer_info = None
        self.writer_agent_vars = None
        self.keys_info = None
        self.keys_agent_vars = [
            'loss_q', 'loss_p', 'mean_q', 'eta',
            'kl_mu', 'kl_sigma', 'alpha_mu', 'alpha_sigma',
            'utd_ratio', 'SPS', 'average_reward_per_second', 'reward',
            't_critic', 't_estep', 't_mstep', 't_learn',
            'actor_params', 'critic_params', 'total_train_params',
            'gpu_memory_allocated_mb', 'gpu_memory_reserved_mb',
            'gpu_max_memory_allocated_mb', 'gpu_util_percent',
            'gpu_memory_used_mb', 'gpu_memory_total_mb', 'gpu_power_watts',
            'cpu_percent', 'ram_used_mb', 'ram_percent',
        ]
        self.info_log_buffer = []
        self.agent_vars_buffer = []

        #compile modules
        # self.actor = torch.compile(self.actor, mode='reduce-overhead')
        # self.actor_target = torch.compile(self.actor_target, mode='reduce-overhead')
        # self.critics = nn.ModuleList([torch.compile(c,mode='reduce-overhead') for c in self.critics])
        # self.target_critics = nn.ModuleList([torch.compile(c,mode='reduce-overhead') for c in self.target_critics])
        


    def get_action(self, obs, evaluate=False):
        act_dim = self.envs.single_action_space.shape[0]

        if self.obs is None:
            self.obs = obs
            rand_actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)]) #Gymnasium uses uniform sampling for box --> if other than change logprob formula
            self.last_actions = rand_actions.copy()
            self.last_log_probs = np.full((self.envs.num_envs, 1), self._uniform_log_prob, dtype=np.float32)
            return rand_actions

        if evaluate or self.global_step > self.args.learning_starts:
            with torch.no_grad():
                obs_t = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device)
                d = self.actor.forward(obs_t)
                action = d.mean if evaluate else d.rsample()
                actions = torch.clamp(action, self.actor.action_low, self.actor.action_high) #mjx clipos internally but overflowed action stored in buffer --> inconsistency
                actions = actions.squeeze(1)     # (n_envs, act_dim)
                
                log_probs = d.log_prob(actions).unsqueeze(-1)
                self.last_actions = actions.cpu().numpy()
                self.last_log_probs = log_probs.cpu().numpy()
                return actions.cpu().numpy()

        rand_actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
        self.last_actions = rand_actions.copy()
        self.last_log_probs = np.full((self.envs.num_envs, 1), self._uniform_log_prob, dtype=np.float32)
        return rand_actions

    def agent_step(self, next_obs, actions, rewards, terminations, truncations, infos):
        # SyncVectorEnv auto-resets and stores the true final obs; hw env does not.
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx in range(self.envs.num_envs):
                if terminations[idx] or truncations[idx]:
                    real_next_obs[idx] = infos["final_observation"][idx]

        self.rb.add(self.obs, real_next_obs, actions, rewards, terminations,
                    [{}] * self.envs.num_envs,
                    behavior_log_prob=self.last_log_probs)
        self.global_step += 1
        self.obs = next_obs

        metrics = None
        if self.global_step > self.args.learning_starts and self.rb.size() >= self.batch_size:
            results = [self._learn() for _ in range(self.args.utd_ratio)]
            keys = results[0].keys()
            metrics = {k: sum(r[k] for r in results) / len(results) for k in keys}
            metrics['utd_ratio'] = self.args.utd_ratio
        return metrics

    def agent_step_eval(self, next_obs):
        self.global_step += 1
        self.obs = next_obs

    def _learn(self):
        if self.global_step >= self.args.learning_starts + self.policy_learning_starts:
            self.args.decouple_q_learning = False

        data = self.rb.sample_nstep(self.batch_size, self.trajectory_length)

        t0 = time.time()
        loss_q, mean_q = self._update_critic(data)
        t_critic = time.time() - t0

        loss_p = kl_mu = kl_sigma = 0.0
        eta = self.log_eta.exp()
        t_estep = t_mstep = 0.0

        # Always run E-step so eta stays calibrated during Q warmup.
        t0 = time.time()
        action_samples, weights, eta, b_mu, b_sigma = self._e_step(data)
        t_estep = time.time() - t0

        if not self.args.decouple_q_learning:
            t0 = time.time()
            loss_p, kl_mu, kl_sigma = self._m_step(data.observations, action_samples, weights, b_mu, b_sigma)
            t_mstep = time.time() - t0

        self._update_targets()

        return {
            'loss_q': loss_q,
            'loss_p': loss_p,
            'mean_q': mean_q,
            'eta': eta.item(),
            'kl_mu': kl_mu,
            'kl_sigma': kl_sigma,
            'alpha_mu': self.alpha_mu,
            'alpha_sigma': self.alpha_sigma,
            't_critic': t_critic,
            't_estep': t_estep,
            't_mstep': t_mstep,
            't_learn': t_critic + t_estep + t_mstep,
        }

    def _update_critic(self, data: NStepReplayBufferSamples) -> Tuple[float, float]:
        n = self.trajectory_length
        B = self.batch_size
        gamma_dt = self.args.gamma ** self.dt

        with torch.no_grad():
            # Fixed critic subset for the entire Retrace computation — consistent targets.
            subset_size = min(2, len(self.critics))
            subset_idx = torch.randperm(len(self.critics), device=self.device)[:subset_size]

            s_k_flat = data.all_observations.reshape(B*n, self.obs_dim)
            a_k_flat = data.all_actions.reshape(B*n,self.act_dim)
            q_traj_stack = self.aggregation_operator(state=s_k_flat, action=a_k_flat, critics=self.target_critics,
                                                     mode='min_subset', subset_size=subset_size, subset_idx=subset_idx)
            q_traj = q_traj_stack.reshape(B,n)
             
            s_kp1_flat = data.all_next_observations.reshape(B*n, self.obs_dim)
            # a_kp1_flat = self.actor_target.get_action(s_kp1_flat)[0].squeeze(1) #squeeze samples to (B*n;act_dim) discard mean and log probs
            d_kp1 = self.actor_target.forward(s_kp1_flat)
            a_kp1_flat = torch.clamp(d_kp1.rsample(),
                                    self.actor_target.action_low,
                                    self.actor_target.action_high)
            q_kp1_stack = self.aggregation_operator(state=s_kp1_flat, action=a_kp1_flat, critics=self.target_critics,
                                                    mode='min_subset', subset_size=subset_size, subset_idx=subset_idx)
            q_kp1_traj = q_kp1_stack.reshape(B,n)
            if n > 1: #importance sampling coef only needed if td horizon larger than 1
                s_middle_flat = data.all_observations[:,1:,:].reshape(B*(n-1), self.obs_dim)
                a_middle_flat = data.all_actions[:,1:,:].reshape(B*(n-1), self.act_dim)
                logp_pi = self.actor_target.get_log_probs(s_middle_flat, a_middle_flat).reshape(B,n-1) #log proba of obtaining a given s under target policy (even if samples are from traj folowing behavior pol)
                logp_mu = data.behavior_log_probs[:,1:,:].squeeze(-1)
                c_all = self.importance_sampling_coef(log_pi=logp_pi, log_mu=logp_mu)

            # Bootstrap from Q_target(s_0, a_0).
            y = q_traj[:,0:1].clone()

            c_prod = torch.ones_like(y)   # running IS product (B, 1)
            alive = torch.ones_like(y)   # episode-still-alive mask (B, 1)
            prev_done = None

            for k in range(n):
                q_kp1 = q_kp1_traj[:,k:k+1]
                q_k = q_traj[:,k:k+1]
                r_k = data.rewards[:, k, :]                # (B, 1)
                done_k = data.dones[:, k, :]                  # (B, 1)
                # s_kp1 = data.all_next_observations[:, k, :]  # (B, obs_dim)

                # a_kp1, _, _ = self.actor_target.get_action(s_kp1)
                # a_kp1 = a_kp1.squeeze(1)
                # q_kp1 = self.aggregation_operator(
                #     s_kp1, a_kp1, self.target_critics,
                #     mode='min_subset', subset_size=2, subset_idx=subset_idx,
                # )
                # # k=0: reuse the initial y to avoid an extra forward pass.
                # q_k = y if k == 0 else self.aggregation_operator(
                #     s_k, a_k, self.target_critics,
                #     mode='min_subset', subset_size=2, subset_idx=subset_idx,
                # )

                delta_k = r_k * self.dt + gamma_dt * (1.0 - done_k) * q_kp1 - q_k

                if k > 0:
                    alive  = alive * (1.0 - prev_done)
                    c_prod = c_prod * c_all[:,k-1:k]

                y = y + (gamma_dt ** k) * c_prod * alive * delta_k
                prev_done = done_k

        losses = []
        q_preds = torch.stack([c(data.observations, data.actions) for c in self.critics], dim=0)
        joint_loss = F.mse_loss(q_preds, y.unsqueeze(0).expand_as(q_preds), reduction='sum') / B

        for opt in self.critic_optimizers:
            opt.zero_grad()
        joint_loss.backward()
        for i, (critic, opt) in enumerate(zip(self.critics, self.critic_optimizers)):
            if self.args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(critic.parameters(), self.args.max_grad_norm)
            opt.step()
            losses.append(F.mse_loss(q_preds[i].detach(), y).item())

        with torch.no_grad():
            mean_q = q_preds.detach().mean().item()

        return float(np.mean(losses)), mean_q

    def aggregation_operator(self, state, action, critics, mode='mean', beta=1.0, subset_size=2, subset_idx=None):
        q_values = torch.stack([c(state, action) for c in critics], dim=0)
        if mode == 'mean':
            return q_values.mean(dim=0)
        elif mode == 'min_subset':
            subset_size = min(subset_size, q_values.shape[0])
            if subset_idx is None:
                subset_idx = torch.randperm(q_values.shape[0], device=self.device)[:subset_size]
            return q_values[subset_idx].min(dim=0).values
        elif mode == 'LCB':
            return q_values.mean(dim=0) - beta * (q_values.std(dim=0) + 1e-6)
        elif mode == 'UCB':
            return q_values.mean(dim=0) + beta * (q_values.std(dim=0) + 1e-6)
        elif mode == 'median':
            return q_values.median(dim=0).values
        else:
            raise ValueError(f"Unknown aggregation mode: {mode}")

    def importance_sampling_coef(self, log_pi: torch.Tensor, log_mu: torch.Tensor) -> torch.Tensor:
        return (log_pi - log_mu).exp().clamp(max=1.0)

    def _solve_temp_dual(self, q_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q_values.detach()  # (B, N)
        q = (q - q.mean()) / q.std().clamp_min(1e-6)

        self.log_eta, self.dual_m, self.dual_v, self.dual_beta1_pow, self.dual_beta2_pow = \
            _run_dual_optim(
                self.log_eta, self.dual_m, self.dual_v,
                self.dual_beta1_pow, self.dual_beta2_pow,
                q, self.args.dual_constraint, math.log(q.shape[-1]),
                self.args.dual_lr, self.args.dual_steps,
            )

        eta_star = self.log_eta.exp()
        weights = torch.softmax(q / eta_star, dim=-1)  # (B, N)
        return eta_star, weights

    def _e_step(self, data):
        obs = data.observations
        N = self.args.sample_action_num
        B, ds = obs.shape[0], obs.shape[-1]
        da = self.envs.single_action_space.shape[0]

        with torch.no_grad():
            d = self.actor_target.forward(obs)
            b_mu, b_sigma = d.base_dist.loc, d.base_dist.scale                            # (B, da)
            action_samples = d.rsample((N,)).permute(1, 0, 2)            # (B, N, da)
            action_samples = torch.clamp(action_samples, self.actor_target.action_low, self.actor_target.action_high) #mjx clipos internally but overflowed action stored in buffer --> inconsistency
            obs_exp = obs.unsqueeze(1).expand(-1, N, -1).reshape(-1, ds)
            acts_flat = action_samples.reshape(-1, da)
            q_values = self.aggregation_operator(
                obs_exp, acts_flat, self.target_critics, mode='mean'
            ).reshape(B, N)

        eta, weights = self._solve_temp_dual(q_values)
        return action_samples, weights, eta, b_mu, b_sigma

    def _m_step(self, obs, action_samples, weights, b_mu, b_sigma) -> Tuple[float, float, float]:
        loss_p_val = kl_mu_val = kl_sigma_val = 0.0

        for _ in range(self.args.mstep_iteration_num):
            curr_d = self.actor.forward(obs)

            # Weighted log-likelihood under current policy.
            log_probs = dist.Normal(curr_d.base_dist.loc.unsqueeze(1), curr_d.base_dist.scale.unsqueeze(1)).log_prob(action_samples)
            nll = -(weights.detach() * log_probs.sum(-1)).sum(-1).mean()

            # Decoupled KL: stop-gradient on sigma for mean term, on mu for covariance term.
            kl_mu = dist.kl_divergence(
                dist.Normal(b_mu, b_sigma),
                dist.Normal(curr_d.base_dist.loc, b_sigma),     # σ fixed at old
            ).sum(-1).mean()

            kl_sigma = dist.kl_divergence(
                dist.Normal(b_mu, b_sigma),
                dist.Normal(b_mu, curr_d.base_dist.scale),      # μ fixed at old
            ).sum(-1).mean()

            loss = (nll
                    + self.alpha_mu * (kl_mu - self.args.kl_mean_constraint)
                    + self.alpha_sigma * (kl_sigma - self.args.kl_var_constraint))

            self.actor_optimizer.zero_grad()
            loss.backward()
            if self.args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
            self.actor_optimizer.step()

            loss_p_val = nll.item()
            kl_mu_val = kl_mu.item()
            kl_sigma_val = kl_sigma.item()

        # Dual-ascent update once per _learn() call, not per inner gradient step.
        # With mstep_iteration_num=5 and utd_ratio=3, updating inside the loop
        # applies 15 alpha updates per env step, causing KL constraint oscillation.
        self.alpha_mu += self.args.alpha_mean_scale * (kl_mu_val - self.args.kl_mean_constraint)
        self.alpha_mu = float(np.clip(self.alpha_mu, 0.0, self.args.alpha_mean_max))
        self.alpha_sigma += self.args.alpha_var_scale * (kl_sigma_val - self.args.kl_var_constraint)
        self.alpha_sigma = float(np.clip(self.alpha_sigma, 0.0, self.args.alpha_var_max))

        return loss_p_val, kl_mu_val, kl_sigma_val

    def _update_targets(self):
        if not self.args.decouple_q_learning:
            for p, p_tgt in zip(self.actor.parameters(), self.actor_target.parameters()):
                p_tgt.data.lerp_(p.data, self.args.tau)
        
        for critic, target_critic in zip(self.critics, self.target_critics):
            for p, p_tgt in zip(critic.parameters(), target_critic.parameters()):
                p_tgt.data.lerp_(p.data, self.args.tau)

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
            system_metrics = get_system_metrics(self.device)

            row_agent = {
                "step": global_step,
                "loss_q": metrics.get('loss_q'),
                "loss_p": metrics.get('loss_p'),
                "mean_q": metrics.get('mean_q'),
                "eta": metrics.get('eta'),
                "kl_mu": metrics.get('kl_mu'),
                "kl_sigma": metrics.get('kl_sigma'),
                "alpha_mu": metrics.get('alpha_mu'),
                "alpha_sigma": metrics.get('alpha_sigma'),
                "utd_ratio": metrics.get('utd_ratio'),
                "SPS": int(global_step / (time.time() - self.start_time)) if self.start_time else 0,
                "average_reward_per_second": self.reward_tracker.average_reward_per_second,
                "reward": rewards[0] if hasattr(rewards, '__len__') else float(rewards),

                "t_critic": metrics.get('t_critic'),
                "t_estep": metrics.get('t_estep'),
                "t_mstep": metrics.get('t_mstep'),
                "t_learn": metrics.get('t_learn'),

                "actor_params": self.actor_param_count,
                "critic_params": self.critic_param_count,
                "total_train_params": self.total_train_param_count,

                "gpu_memory_allocated_mb": system_metrics.get("system/gpu_memory_allocated_mb"),
                "gpu_memory_reserved_mb": system_metrics.get("system/gpu_memory_reserved_mb"),
                "gpu_max_memory_allocated_mb": system_metrics.get("system/gpu_max_memory_allocated_mb"),
                "gpu_util_percent": system_metrics.get("system/gpu_util_percent"),
                "gpu_memory_used_mb": system_metrics.get("system/gpu_memory_used_mb"),
                "gpu_memory_total_mb": system_metrics.get("system/gpu_memory_total_mb"),
                "gpu_power_watts": system_metrics.get("system/gpu_power_watts"),
                "cpu_percent": system_metrics.get("system/cpu_percent"),
                "ram_used_mb": system_metrics.get("system/ram_used_mb"),
                "ram_percent": system_metrics.get("system/ram_percent"),
            }

            self.agent_vars_buffer.append(row_agent)

            if self.use_wandb:
                wandb_log = {
                    "train/loss_q": metrics.get('loss_q'),
                    "train/loss_p": metrics.get('loss_p'),
                    "train/mean_q": metrics.get('mean_q'),
                    "train/eta": metrics.get('eta'),
                    "train/kl_mu": metrics.get('kl_mu'),
                    "train/kl_sigma": metrics.get('kl_sigma'),
                    "train/alpha_mu": metrics.get('alpha_mu'),
                    "train/alpha_sigma": metrics.get('alpha_sigma'),
                    "train/utd_ratio": metrics.get('utd_ratio'),
                    "train/SPS": row_agent["SPS"],
                    "train/average_reward_per_second": self.reward_tracker.average_reward_per_second,
                    "train/reward": row_agent["reward"],

                    "time/t_critic": metrics.get('t_critic'),
                    "time/t_estep": metrics.get('t_estep'),
                    "time/t_mstep": metrics.get('t_mstep'),
                    "time/t_learn": metrics.get('t_learn'),

                    "model/actor_params": self.actor_param_count,
                    "model/critic_params": self.critic_param_count,
                    "model/total_train_params": self.total_train_param_count,
                    "model/ensemble": self.args.ensemble,
                }

                wandb_log.update(system_metrics)
                wandb.log(wandb_log, step=global_step)

        if global_step % self.args.save_every_n_steps == 0:
            for row in self.info_log_buffer:
                self.writer_info.writerow(row)
            self.csv_file_info.flush()
            self.info_log_buffer = []

            for row in self.agent_vars_buffer:
                self.writer_agent_vars.writerow(row)
            self.csv_file_agent_vars.flush()
            self.agent_vars_buffer = []

    def save_checkpoint(self, global_step):
        checkpoint = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critics": [c.state_dict() for c in self.critics],
            "target_critics": [c.state_dict() for c in self.target_critics],
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizers": [opt.state_dict() for opt in self.critic_optimizers],
            "log_eta": self.log_eta.item(),
            "dual_m": self.dual_m.item(),
            "dual_v": self.dual_v.item(),
            "dual_beta1_pow": self.dual_beta1_pow.item(),
            "dual_beta2_pow": self.dual_beta2_pow.item(),
            "alpha_mu": self.alpha_mu,
            "alpha_sigma": self.alpha_sigma,
            "global_step": global_step,
        }
        torch.save(checkpoint, os.path.join(self.weights_folder, f"checkpoint_{global_step}.pth"))
        if global_step % (10 * self.args.save_every_n_steps) == 0:
            self.rb.save(os.path.join(self.weights_folder, "replay_buffer.npz"))

    def load_checkpoint(self, weights_path):
        checkpoint_files = [f for f in os.listdir(weights_path) if f.endswith(".pth")]
        checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        checkpoint = torch.load(os.path.join(weights_path, checkpoint_files[-1]), map_location=self.device)

        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        for c, sd in zip(self.critics, checkpoint["critics"]):
            c.load_state_dict(sd)
        for c, sd in zip(self.target_critics, checkpoint["target_critics"]):
            c.load_state_dict(sd)
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        for opt, sd in zip(self.critic_optimizers, checkpoint["critic_optimizers"]):
            opt.load_state_dict(sd)
        if "log_eta" in checkpoint:
            self.log_eta.fill_(checkpoint["log_eta"])
        if "dual_m" in checkpoint:
            self.dual_m.fill_(checkpoint["dual_m"])
            self.dual_v.fill_(checkpoint["dual_v"])
            self.dual_beta1_pow.fill_(checkpoint["dual_beta1_pow"])
            self.dual_beta2_pow.fill_(checkpoint["dual_beta2_pow"])
        self.alpha_mu = float(checkpoint.get("alpha_mu", 0.0))
        self.alpha_sigma = float(checkpoint.get("alpha_sigma", 0.0))
        self.global_step = checkpoint.get("global_step", 0)
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
        if getattr(self, "use_wandb", False):
            wandb.finish()

@torch.compile
def _run_dual_optim(log_eta, m, v, beta1_pow, beta2_pow, q, dual_constraint, log_n, lr, n_steps):
    """
    Runs n_steps of Adam on the temperature dual variable using the analytical gradient,
    avoiding autograd entirely. The gradient of the dual loss w.r.t. log_eta is:
        dL/d(log_eta) = L(eta) - mean((softmax(q/eta, dim=-1) * q).sum(-1))
    where L(eta) = eta*(eps + mean(logsumexp(q/eta)) - log_n).
    """
    beta1 = 0.9
    beta2 = 0.999
    adam_eps = 1e-8
    for _ in range(n_steps):
        eta = log_eta.exp()
        scaled_q = q / eta
        w = torch.softmax(scaled_q, dim=-1)
        L = eta * (dual_constraint + torch.logsumexp(scaled_q, dim=-1).mean() - log_n)
        grad = L - (w * q).sum(-1).mean()

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad * grad
        beta1_pow = beta1_pow * beta1
        beta2_pow = beta2_pow * beta2
        m_hat = m / (1.0 - beta1_pow)
        v_hat = v / (1.0 - beta2_pow)
        log_eta = (log_eta - lr * m_hat / (v_hat.sqrt() + adam_eps)).clamp(-4.0, 4.0)

    return log_eta, m, v, beta1_pow, beta2_pow


def parse_args():
    parser = argparse.ArgumentParser()

    # General.
    parser.add_argument("--exp_name", type=str, default="mpo_ant")
    parser.add_argument("--runs_directory", type=str, default="runs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch_deterministic", type=bool, default=True)
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--save_every_n_steps", type=int, default=4000)
    parser.add_argument("--log_every_n_steps", type=int, default=100)

    # Algorithm.
    parser.add_argument("--env_id", type=str, default="EAnt")
    parser.add_argument("--total_timesteps", type=int, default=60_000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_starts", type=int, default=2000)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--q_lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.92)
    parser.add_argument("--use_layer_norm", type=bool, default=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_hidden_layers", type=int, default=2)
    parser.add_argument("--utd_ratio", type=int, default=1)
    parser.add_argument("--td_horizon", type=int, default=1,
                        help="n-step TD horizon for Q learning")
    parser.add_argument("--decouple_q_learning", action='store_true', default=False)
    parser.add_argument("--policy_learning_starts", type=int, default=0,
                        help="steps of Q-only warmup after learning_starts when --decouple_q_learning is set")

    # MPO specific.
    parser.add_argument("--dual_constraint", type=float, default=0.1,
                        help="epsilon for E-step temperature dual")
    parser.add_argument("--kl_mean_constraint", type=float, default=0.01,
                        help="epsilon_mu for M-step mean KL")
    parser.add_argument("--kl_var_constraint", type=float, default=1e-4,
                        help="epsilon_sigma for M-step covariance KL")
    parser.add_argument("--alpha_mean_scale", type=float, default=1.0)
    parser.add_argument("--alpha_var_scale", type=float, default=20.0)
    parser.add_argument("--alpha_mean_max", type=float, default=0.1)
    parser.add_argument("--alpha_var_max", type=float, default=10.0)
    parser.add_argument("--sample_action_num", type=int, default=64,
                        help="actions sampled per state in E-step")
    parser.add_argument("--mstep_iteration_num", type=int, default=5,
                        help="actor gradient steps per learn() call")
    parser.add_argument("--dual_lr", type=float, default=1e-2)
    parser.add_argument("--dual_steps", type=int, default=30)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--ensemble", type=int, default=1)

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
    
    # Weights & Biases logging.
    parser.add_argument("--track_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="mpo-ant")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_watch", action="store_true", default=False)
    parser.add_argument("--history_len", type=int, default=1,
                    help="Number of past observations stacked into each observation")

    parser.add_argument("--critic_use_gru", action="store_true", default=False,
                        help="Use GRU encoder inside critic over stacked observation history")

    parser.add_argument("--critic_gru_hidden_dim", type=int, default=128,
                        help="Hidden size of critic GRU")
        

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

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

    envs = make_ant_envs(args, task, disk_folder, run_name, runs_directory=args.runs_directory)

    agent = MPO(args, envs, disk_folder=disk_folder, run_name=run_name, runs_directory=args.runs_directory)

    if args.weights_path is not None:
        agent.load_checkpoint(args.weights_path)
        if args.eval:
            agent.global_step = 0

    obs, info = envs.reset(seed=args.seed)
    agent.initialize_logging(info)

    time_start_learning = time.time()
    step_times = []

    for step in tqdm(range(agent.global_step, args.total_timesteps)):
        time_now = time.time()

        selected_actions = agent.get_action(obs, args.eval)
        next_obs, rewards, terminations, truncations, infos = envs.step(selected_actions)

        if args.eval:
            agent.agent_step_eval(next_obs)
            metrics = None
        else:
            metrics = agent.agent_step(next_obs, selected_actions, rewards, terminations, truncations, infos)

        agent.log_step(step, infos, rewards, metrics)

        if step % args.save_every_n_steps == 0:
            agent.save_checkpoint(step)
        step_times.append(f"{step},{time.time() - time_now}\n")

    with open(os.path.join(args.runs_directory, run_name, "step_times.csv"), "w") as f:
        f.writelines(step_times)

    agent.cleanup()
