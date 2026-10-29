# This file is adapted from CleanRL (https://github.com/vwxyzjn/cleanrl)
# Copyright (c) 2019 CleanRL developers
# Licensed under the MIT License (see LICENSE file)

from typing import Optional, Tuple
import os

import copy
import csv
import sys
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import gymnasium as gym
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Import custom modules.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import make_ant_env, ForwardTask, BackAndForthTask
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from reward import RewardTracker
# Same [time, env] buffer as MPO, for n-step (td_horizon) targets.
from mpo.buffer_acme import ReplayBuffer as NStepReplayBuffer
from mpo.resume_state import get_rng_state, set_rng_state, snapshot_envs, restore_envs

# Distribution arg checks call .all() on GPU tensors, a hidden sync per distribution
torch.distributions.Distribution.set_default_validate_args(False)

class SoftQNetwork(nn.Module):
    def __init__(self, env, use_layer_norm=False):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

        if use_layer_norm:
            self.ln1 = nn.LayerNorm(256)
            self.ln2 = nn.LayerNorm(256)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.fc1(x)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, env, use_layer_norm=False):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))

        if use_layer_norm:
            self.ln1 = nn.LayerNorm(256)
            self.ln2 = nn.LayerNorm(256)

        # Action rescaling.
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = self.fc1(x)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        x = F.relu(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() # for reparameterization trick (mean + std * N(0,1)).
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound.
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class OriginalRewardWrapper(gym.Wrapper):
    """Store the pre-scaling reward in info['original_reward'].

    The embodied Ant Task objects set this key themselves (see
    embodied_ant_env.py); plain Gymnasium envs (e.g. Walker2d-v5) never do,
    so this wrapper fills it in for those.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["original_reward"] = np.asarray(reward, dtype=np.float64)
        return obs, reward, terminated, truncated, info


# Embodied / custom Ant IDs used by this repo (not Gymnasium registry entries).
EMBODIED_ANT_ENV_IDS = {
    "EAnt",
    "SimEmbodiedAnt",
    "HwEmbodiedAnt",
    "CustomAnt-v0",
}


def is_gymnasium_env(env_id: str) -> bool:
    """True for registered Gymnasium envs (Hopper-v5, Walker2d-v5, Humanoid-v5, ...)."""
    if env_id in EMBODIED_ANT_ENV_IDS:
        return False
    return env_id in gym.envs.registry


def effective_reward_scale(args) -> float:
    if args.reward_scale is not None:
        return float(args.reward_scale)
    return 1.0 if is_gymnasium_env(args.env_id) else 100.0


def _unwrap_base_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def _maybe_record_video(env, args, idx, disk_folder, run_name, runs_directory):
    if args.capture_video and idx == 0:
        if args.capture_video_steps is not None:
            # Clips of capture_video_steps each, starting every capture_video_every
            # steps (default: one clip at step 0 only).
            every = args.capture_video_every
            step_trigger = (lambda x: x == 0) if every is None else (lambda x: x % every == 0)
            video_length = args.capture_video_steps
        else:
            step_trigger = lambda x: x % args.save_every_n_steps == 0
            video_length = args.save_every_n_steps
        env = gym.wrappers.RecordVideo(
            env,
            # On --resume_in_place, video_subdir points at a fresh folder so the
            # restarted step counter doesn't overwrite earlier clips.
            os.path.join(disk_folder, runs_directory, run_name, "videos",
                         getattr(args, "video_subdir", None) or run_name),
            step_trigger=step_trigger,
            video_length=video_length,
        )
    return env


def _make_gymnasium_env(args, seed, idx, disk_folder, run_name, runs_directory, reward_scale):
    render_mode = args.render_mode
    if args.capture_video and idx == 0 and render_mode is None:
        render_mode = "rgb_array"

    env = gym.make(args.env_id, render_mode=render_mode, xml_file=args.model_path)
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    # Gymnasium MuJoCo often returns float64 obs while declaring float32 spaces.
    float32_obs_space = gym.spaces.Box(
        low=env.observation_space.low.astype(np.float32),
        high=env.observation_space.high.astype(np.float32),
        dtype=np.float32,
    )
    env = gym.wrappers.TransformObservation(
        env,
        lambda obs: np.asarray(obs, dtype=np.float32),
        float32_obs_space,
    )
    env = OriginalRewardWrapper(env)
    env = _maybe_record_video(env, args, idx, disk_folder, run_name, runs_directory)
    env.action_space.seed(seed)
    env = gym.wrappers.TransformReward(env, lambda reward, scale=reward_scale: reward * scale)
    return env


def _make_embodied_ant_env(args, task, seed, idx, disk_folder, run_name, runs_directory, reward_scale):
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
                           dt=args.dt,
                           joint_config=joint_config,
                           task=task,
                           )
        # env.metadata['render_fps'] = 1/args.dt

    env = _maybe_record_video(env, args, idx, disk_folder, run_name, runs_directory)
    env.action_space.seed(seed)
    env = gym.wrappers.TransformReward(env, lambda reward, scale=reward_scale: reward * scale)
    return env


def make_ant_envs(args, task, disk_folder, run_name, runs_directory='runs'):
    """Create the vectorized environment outside the SAC class.

    Builds either the custom embodied AntEnv, or (when args.env_id names a
    registered Gymnasium env, e.g. Walker2d-v5) a real Gymnasium MuJoCo env
    loaded with args.model_path as its XML.
    """
    reward_scale = effective_reward_scale(args)
    args.reward_scale = reward_scale
    use_gym = is_gymnasium_env(args.env_id)

    if use_gym:
        print(f"[√] Using Gymnasium env_id={args.env_id} (reward_scale={reward_scale})")
        if task is not None:
            print("[!] Ignoring --task_type for Gymnasium environments")
    else:
        print(f"[√] Using embodied Ant env_id={args.env_id} (reward_scale={reward_scale})")

    def make_env(seed, idx, capture_video, run_name):
        def _init():
            if use_gym:
                return _make_gymnasium_env(args, seed, idx, disk_folder, run_name, runs_directory, reward_scale)
            return _make_embodied_ant_env(args, task, seed, idx, disk_folder, run_name, runs_directory, reward_scale)
        return _init

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "[!] Only continuous action space is supported."

    if use_gym:
        base = _unwrap_base_env(envs.envs[0])
        if hasattr(base, "dt"):
            args.dt = float(base.dt)
            print(f"[√] Synced args.dt to Gymnasium env dt={args.dt}")

    print(f"[√] Created environment with {envs.num_envs} environments.")
    return envs


class SAC:
    def __init__(self,
                 envs: gym.vector.SyncVectorEnv,
                 device: torch.device,
                 seed: int,
                 q_lr: float,
                 alpha_lr: float,
                 policy_lr: float,
                 autotune: bool,
                 alpha: float,
                 buffer_size: int,
                 batch_size: int,
                 learning_starts: int,
                 policy_frequency: int,
                 target_network_frequency: int,
                 tau: float,
                 gamma: float,
                 use_layer_norm: bool,
                 dt: float,
                 torch_deterministic: bool = True,
                 record_infos_sac = True,
                 td_horizon: int = 5,
                 cuda_graph: bool = True,
                 ):

        # Environment.
        self.envs = envs

        self.device = device
        print(f"[√] Using device: {self.device}")

        # Set seeds for reproducibility.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = torch_deterministic
        torch.backends.cudnn.benchmark = not torch_deterministic

        # Networks.
        self.actor = Actor(self.envs, use_layer_norm=use_layer_norm).to(self.device)
        self.qf1 = SoftQNetwork(self.envs, use_layer_norm=use_layer_norm).to(self.device)
        self.qf2 = SoftQNetwork(self.envs, use_layer_norm=use_layer_norm).to(self.device)
        self.qf1_target = SoftQNetwork(self.envs, use_layer_norm=use_layer_norm).to(self.device)
        self.qf2_target = SoftQNetwork(self.envs, use_layer_norm=use_layer_norm).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        # The whole learner update is captured as a CUDA graph and replayed every step. The update differs
        # by step (actor/alpha every policy_frequency, targets every target_network_frequency), so there is
        # one graph per (actor step, target step) combination.
        self._use_cuda_graph = cuda_graph and torch.device(self.device).type == "cuda"
        self._graphs = {}
        self._graph_warmup_left = {}
        self._static_inputs = None
        self._static_stats = {}
        cap = self._use_cuda_graph
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=q_lr,
                                      capturable=cap)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=policy_lr, capturable=cap)

        self.learning_starts = learning_starts
        self.autotune = autotune
        self.batch_size = batch_size
        self.gamma = gamma
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.tau = tau
        self.dt = dt
        self.td_horizon = td_horizon

        self.record_infos_sac = record_infos_sac

        # Alpha (entropy coefficient).
        if self.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.envs.single_action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.a_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr, capturable=cap)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
            self.log_alpha = None
            self.a_optimizer = None

        self.envs.single_observation_space.dtype = np.float32

        # Replay buffer.
        self.rb = NStepReplayBuffer(
                buffer_size,
                envs.single_observation_space,
                envs.single_action_space,
                device,
                n_envs=envs.num_envs,
            )
        # Envs that hit a boundary last step; their next row is a reset row (valid=False).
        self.pending_autoreset = torch.zeros(envs.num_envs, dtype=torch.bool, device=device)

        self.global_step = 0
        self.obs = None
        
    def get_action(self, obs, evaluate=False):
        if self.obs is None:
            self.obs = obs
            actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
            # log prob of random action = -log (action-high-bound - action-low-bound) summed over all dim
            self._last_behavior_log_probs = torch.full(
                (self.envs.num_envs, 1),
                -float(np.log(self.envs.single_action_space.high - self.envs.single_action_space.low).sum()),
                dtype=torch.float32, device=self.device,
            )
            return actions

           # get action from online actor and take action.
        if evaluate == True or self.global_step > self.learning_starts:
            actions, log_probs, _ = self.actor.get_action(torch.Tensor(self.obs).to(self.device))
            self._last_behavior_log_probs = log_probs.detach()
            actions = actions.detach().cpu().numpy()
            return actions

        # Otherwise, pick a random action.
        actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])
        self._last_behavior_log_probs = torch.full(
            (self.envs.num_envs, 1),
            -float(np.log(self.envs.single_action_space.high - self.envs.single_action_space.low).sum()),
            dtype=torch.float32, device=self.device,
        )
        return actions
 
    def agent_step(self, next_obs, actions, rewards, terminations, truncations, infos):
        terminations = torch.as_tensor(terminations, dtype=torch.bool, device=self.device)
        truncations = torch.as_tensor(truncations, dtype=torch.bool, device=self.device)
        boundaries = terminations | truncations
        if "autoreset" in infos:
            autoreset_now = torch.as_tensor(infos["autoreset"], dtype=torch.bool, device=self.device)
        else:
            autoreset_now = self.pending_autoreset
        valid = ~autoreset_now

        num_inserts = int(valid.sum().item())
        self.rb.add(
            obs=torch.as_tensor(self.obs, dtype=torch.float32, device=self.device),
            next_obs=torch.as_tensor(next_obs, dtype=torch.float32, device=self.device),
            action=torch.as_tensor(actions, dtype=torch.float32, device=self.device),
            reward=torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            terminated=terminations,
            truncated=truncations,
            valid=valid,
            behavior_log_prob=self._last_behavior_log_probs,
        )

        self.global_step += 1
        self.obs = next_obs
        self.pending_autoreset = boundaries.detach().clone()

        # Learning.
        stats = {}
        if self.global_step > self.learning_starts:
            stats = self.learn()

        info_sac = None
        if self.record_infos_sac == True:
            info_sac = {
                'global_step': self.global_step,
                'qf1_loss': stats.get('qf1_loss'),
                'qf2_loss': stats.get('qf2_loss'),
                'qf1_values_mean': stats.get('qf1_values_mean'),
                'qf2_values_mean': stats.get('qf2_values_mean'),
                'actor_loss': stats.get('actor_loss'),
                'alpha': self.alpha,
                'alpha_loss': stats.get('alpha_loss'),
                'rewards': rewards.flatten().tolist(),
                'original_rewards': infos['original_reward'].flatten().tolist(),
            }
        return info_sac

    def learn(self):
        """Samples a batch and runs one learner update. Returns the logged stats as floats (one GPU sync)."""
        data = self.rb.sample_nstep(
            batch_size=self.batch_size,
            n_step=self.td_horizon,
            gamma=self.gamma,
        )
        inputs = (data.observations, data.actions, data.rewards,
                  data.all_observations, data.all_next_observations,
                  data.all_actions, data.behavior_log_probs,
                  data.terminations, data.truncations)
        key = (self.global_step % self.policy_frequency == 0,
               self.global_step % self.target_network_frequency == 0)
        if not self._use_cuda_graph:
            stats = self._update(*inputs, *key)
        else:
            stats = self._graph_update(inputs, key)

        stats = dict(zip(stats, torch.stack(list(stats.values())).tolist()))
        if "alpha" in stats:
            self.alpha = stats.pop("alpha")
        return stats

    def _graph_update(self, inputs, key):
        """Replays the CUDA graph of this step's update, capturing it on first use."""
        if key not in self._graphs:
            if self._graph_warmup_left.setdefault(key, 3) > 0:
                # CUDA graphs need a few eager iterations on a side stream before capture
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    stats = self._update(*inputs, *key)
                torch.cuda.current_stream().wait_stream(side)
                self._graph_warmup_left[key] -= 1
                return stats
            if self._static_inputs is None:
                self._static_inputs = [x.clone() for x in inputs]
            self._graphs[key] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graphs[key]):
                self._static_stats[key] = self._update(*self._static_inputs, *key)
            # capture only records the kernels; the replay below runs this step's update

        for static, x in zip(self._static_inputs, inputs):
            static.copy_(x)
        self._graphs[key].replay()
        return self._static_stats[key]

    def _update(self, observations, actions, rewards, all_observations,
                all_next_observations, all_actions, behavior_log_probs,
                terminations, truncations, actor_step, target_step):
        """ TD(N) recursion is within this update"""
        alpha = self.log_alpha.detach().exp() if self.autotune else self.alpha
        with torch.no_grad():
            # N = td_horizon step count.
            B, N = rewards.shape
            flat_next_obs = all_next_observations.reshape(B * N, -1)
            next_actions, next_log_pi, _ = self.actor.get_action(flat_next_obs)
            v_next = (torch.min(
                self.qf1_target(flat_next_obs, next_actions),
                self.qf2_target(flat_next_obs, next_actions)
            ) - alpha * next_log_pi).reshape(B, N)

            flat_obs = all_observations.reshape(B * N, -1)
            flat_actions = all_actions.reshape(B * N, -1)
            # build the gaussian distribution of the current actor.
            mean, log_std = self.actor(flat_obs)
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)

            # undo action scaling
            scaled = (flat_actions - self.actor.action_bias) / self.actor.action_scale
            scaled = scaled.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            pre_tanh = torch.atanh(scaled)
        
            # transform from prob in gaussiand istribution to prob in actor distribution
            log_pi_taken = (
                normal.log_prob(pre_tanh)
                - torch.log(self.actor.action_scale * (1 - scaled.square()) + 1e-6)
            ).sum(dim=-1).reshape(B, N)
            log_ratio = log_pi_taken - behavior_log_probs
            rho = torch.exp(log_ratio) # torch.ones_like(log_ratio)  #TODO: does IS ratio need clamping

            # final value at t+n is like TD(1) 
            target = rewards[:, -1] + self.gamma*(~terminations[:, -1]).float() * v_next[:, -1]
            for k in range(N - 2, -1, -1):
                boundary = terminations[:, k] | truncations[:, k]
                # if at boundary, then no need to go beyond and simply use v_next. 
                continuation = torch.where(
                    boundary,
                    v_next[:, k],
                    rho[:, k + 1] * (target - alpha * log_pi_taken[:, k + 1]),
                )
                # don't bootstrap if termination. do bootstrap if truncation.
                target = rewards[:, k] + self.gamma * (~terminations[:, k]).float() * continuation
            next_q_value = target
        qf1_a_values = self.qf1(observations, actions).view(-1)
        qf2_a_values = self.qf2(observations, actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        stats = {
            'qf1_loss': qf1_loss.detach(),
            'qf2_loss': qf2_loss.detach(),
            'qf1_values_mean': qf1_a_values.detach().mean(),
            'qf2_values_mean': qf2_a_values.detach().mean(),
        }

        # Optimize the Action-Value networks.
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if actor_step:
            for _ in range(
                    self.policy_frequency
            ):  # Compensate for the delay by doing 'actor_update_interval' instead of 1.
                pi, log_pi, _ = self.actor.get_action(observations)
                qf1_pi = self.qf1(observations, pi)
                qf2_pi = self.qf2(observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                # Optimize the Actor network.
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = self.actor.get_action(observations)
                    alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    alpha = self.log_alpha.detach().exp()
            stats['actor_loss'] = actor_loss.detach()
            if self.autotune:
                stats['alpha_loss'] = alpha_loss.detach()
                stats['alpha'] = alpha.squeeze()

        # Update the target networks.
        if target_step:
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)
        return stats

    def agent_step_eval(self, next_obs):
        self.global_step += 1
        self.obs = next_obs

    def set_global_step(self, global_step: int):
        self.global_step = global_step

    def get_replay_buffer(self):
        return self.rb

    def load_replay_buffer(self, replay_buffer_path):
        self.rb.load(replay_buffer_path, device=self.device)

    def get_state(self):
        """Returns the full state of the agent including all network weights and optimizers."""
        state = {
            "actor": self.actor.state_dict(),
            "qf1": self.qf1.state_dict(),
            "qf2": self.qf2.state_dict(),
            "qf1_target": self.qf1_target.state_dict(),
            "qf2_target": self.qf2_target.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "global_step": self.global_step
        }
        if self.autotune:
            state["log_alpha"] = self.log_alpha.cpu().detach()
            state["alpha"] = self.alpha
            state["a_optimizer"] = self.a_optimizer.state_dict()
        # Random generators, simulator/task state and the in-flight episode, so a
        # resume continues exactly where this step left off (same as MPO).
        state["rng"] = get_rng_state()
        state["envs"] = snapshot_envs(self.envs)
        state["obs"] = None if self.obs is None else np.array(self.obs, copy=True)
        state["pending_autoreset"] = self.pending_autoreset.detach().cpu()
        return state

    def load_state(self, state_dict):
        """Loads the full state of the agent from a state dictionary."""
        self.actor.load_state_dict(state_dict["actor"])
        self.qf1.load_state_dict(state_dict["qf1"])
        self.qf2.load_state_dict(state_dict["qf2"])
        self.qf1_target.load_state_dict(state_dict["qf1_target"])
        self.qf2_target.load_state_dict(state_dict["qf2_target"])
        self._load_optimizer(self.q_optimizer, state_dict["q_optimizer"])
        self._load_optimizer(self.actor_optimizer, state_dict["actor_optimizer"])
        self.global_step = state_dict.get("global_step", 0)
        if self.autotune and "log_alpha" in state_dict:
            log_alpha_tensor = state_dict["log_alpha"].to(self.device)
            if not log_alpha_tensor.requires_grad:
                log_alpha_tensor.requires_grad_(True)
            self.log_alpha.data = log_alpha_tensor
            if "alpha" in state_dict:
                self.alpha = state_dict["alpha"]
            else:
                self.alpha = self.log_alpha.exp().item()
            if "a_optimizer" in state_dict and self.a_optimizer is not None:
                self._load_optimizer(self.a_optimizer, state_dict["a_optimizer"])
        # Continue the saved episode. Runs after envs.reset(), so the reset is overwritten.
        if state_dict.get("envs") is not None and restore_envs(self.envs, state_dict["envs"]):
            self.obs = state_dict["obs"]
            self.pending_autoreset = state_dict["pending_autoreset"].to(self.device)
            print("[√] Restored simulator and task state")
        if "rng" in state_dict:
            set_rng_state(state_dict["rng"])
            print("[√] Restored Python, NumPy, torch and CUDA random states")

    def _load_optimizer(self, optimizer, state):
        """Loads an optimizer state saved with or without --cuda_graph. load_state_dict restores the saved
        capturable flag, so this run's flag is put back and Adam's step counters moved to match it
        (capturable needs them on the GPU, non-capturable on the CPU)."""
        optimizer.load_state_dict(state)
        for group in optimizer.param_groups:
            group["capturable"] = self._use_cuda_graph
        for param_state in optimizer.state.values():
            if "step" in param_state:
                param_state["step"] = param_state["step"].to(self.device if self._use_cuda_graph else "cpu")


# For logging.
def arr_to_str(x):
    if isinstance(x, np.ndarray):
        return "[" + " ".join(map(str, x.tolist())) + "]"
    return x

def parse_args():
    parser = argparse.ArgumentParser()

    # General.
    parser.add_argument("--exp_name", type=str, default="sac_ant",
                        help="the name of this experiment")
    parser.add_argument("--runs_directory", type=str, default="runs",
                        help="the directory to save the runs in")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--no-torch_deterministic", action="store_false",
                        dest="torch_deterministic",
                        help="disable torch deterministic mode")
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--cuda_graph", action=argparse.BooleanOptionalAction, default=True,
                        help="Capture the learner update as a CUDA graph (GPU only)")
    parser.add_argument("--capture_video", action="store_true",
                        help="capture video of agent performances")
    parser.add_argument("--eval", action="store_true", default=False,
                        help="evaluate the agent")
    parser.add_argument("--save_every_n_steps", type=int, default=4000,
                        help="save every n steps")

    # Algorithm.
    parser.add_argument("--env_id", type=str, default="EAnt",
                        help="environment ID")
    parser.add_argument("--total_timesteps", type=int, default=60_000,
                        help="total training timesteps")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="number of parallel envs")
    parser.add_argument("--buffer_size", type=int, default=int(1e6),
                        help="replay buffer size")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="target smoothing coefficient")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="batch size")
    parser.add_argument("--learning_starts", type=int, default=2000,
                        help="timestep to start learning")
    parser.add_argument("--policy_lr", type=float, default=3e-4,
                        help="policy learning rate")
    parser.add_argument("--q_lr", type=float, default=1e-3,
                        help="Q-network learning rate")
    parser.add_argument("--alpha_lr", type=float, default=1e-3,
                        help="alpha learning rate")
    parser.add_argument("--policy_frequency", type=int, default=2,
                        help="policy update frequency")
    parser.add_argument("--target_network_frequency", type=int, default=1,
                        help="target network update frequency")
    parser.add_argument("--alpha", type=float, default=0.2,
                        help="entropy regularization coefficient")
    parser.add_argument("--no-autotune", action="store_false",
                        dest="autotune",
                        help="disable automatic entropy tuning")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="discount factor")
    parser.add_argument("--td_horizon", type=int, default=5,
                        help="n-step TD horizon for the critic target (1 = standard SAC)")
    parser.add_argument("--no-use_layer_norm", action="store_false",
                        dest="use_layer_norm",
                        help="disable layer normalization in networks")

    # Environment.
    parser.add_argument("--dt", type=float, default=0.12,
                        help="environment timestep")
    parser.add_argument("--hw_config", type=str, default=None,
                        help="hardware config file")
    parser.add_argument("--render_mode", type=str, default="rgb_array",
                        help="render mode")
    parser.add_argument("--no-terminate_on_upside_down", action="store_false",
                        dest="terminate_on_upside_down",
                        help="do not terminate episode when upside down")
    parser.add_argument("--weights_path", type=str, default=None,
                        help="load previous weights")
    parser.add_argument("--task_type", type=str, default="back_and_forth",
                        choices=["forward", "back_and_forth"],
                        help="type of task")
    parser.add_argument("--radius_back_and_forth", type=float, default=0.3,
                        help="radius of the back and forth task")
    parser.add_argument("--origin_back_and_forth", type=float, nargs=2, default=[0.75, -0.3],
                        help="origin of the back and forth task")
    parser.add_argument("--reward_scale", type=float, default=100.0,
                        help="reward scale factor")
    parser.add_argument("--model_path", type=str, default="../../sim/assets/ant_with_camera_after_sys_id.xml",
                        help="XML file to use for the environment")

    parser.set_defaults(
        torch_deterministic=True,
        autotune=True,
        use_layer_norm=True,
        terminate_on_upside_down=True,
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    # Set up folders for environment creation.
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    disk_folder = ''
    os.makedirs(args.runs_directory, exist_ok=True)
    run_name = f"{args.exp_name}_{date}_seed_{args.seed}"
    os.makedirs(os.path.join(args.runs_directory, run_name), exist_ok=True)

    # Save the args.
    with open(os.path.join(args.runs_directory, run_name, "args.json"), "w") as f:
        json.dump(args.__dict__, f)

    # Create task.
    if args.task_type == "forward":
        task = ForwardTask()
    elif args.task_type == "back_and_forth":
        RADIUS = args.radius_back_and_forth
        ORIGIN = np.array(args.origin_back_and_forth) # Measured in the environment.
        task = BackAndForthTask(
            radius=RADIUS,
            origin=ORIGIN,
        )
        print(f"BackAndForthTask initialized for radius: {RADIUS}, origin: {ORIGIN}")
    else:
        raise ValueError(f"Invalid task type: {args.task_type}")

    # Create environment.
    envs = make_ant_envs(args=args,
                         task=task,
                         disk_folder=disk_folder,
                         run_name=run_name,
                         runs_directory=args.runs_directory)
    # Setup device.
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create SAC agent.
    agent = SAC(envs=envs,
                device=device,
                q_lr=args.q_lr,
                alpha_lr=args.alpha_lr,
                policy_lr=args.policy_lr,
                autotune=args.autotune,
                alpha=args.alpha,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                learning_starts=args.learning_starts,
                policy_frequency=args.policy_frequency,
                target_network_frequency=args.target_network_frequency,
                tau=args.tau,
                gamma=args.gamma,
                use_layer_norm=args.use_layer_norm,
                seed=args.seed,
                dt=args.dt,
                td_horizon=args.td_horizon,
                cuda_graph=args.cuda_graph)

    # Reset before loading, so load_state can restore the saved episode over it.
    obs, info = envs.reset(seed=args.seed)

    step = 0
    # Load the model.
    if args.weights_path is not None:
        ckpt_path = os.path.abspath(os.path.join(args.weights_path, "weights.pth"))
        buffer_path = os.path.abspath(os.path.join(args.weights_path, "replay_buffer.npz"))
        # weights_only=False: the checkpoint also holds numpy RNG and simulator state.
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        print("==========================================")
        print(f"[i] loading checkpoint: {ckpt_path}")
        print(f"[i] loading replay buffer: {buffer_path}")
        print(f"[i] resuming from global_step {state['global_step']}, ")
            #   f"target actor in checkpoint: {'actor_target' in state}, "
            #   f"KL state in checkpoint: {'log_beta_mean' in state}")
        print("==========================================")
        agent.load_state(state)
        step = state["global_step"]
        agent.load_replay_buffer(buffer_path)

    if args.eval:
        step = 0 # Reset the step to 0 when eval.

    agent.set_global_step(step)
    print(f"Step: {step}")

    # Reward tracker.
    env_dt = args.dt
    env_id = args.env_id
    reward_tracker = RewardTracker(env_dt=env_dt,
                                   env_id=env_id,
                                   time_window=120.0,
                                   log_folder=os.path.join(args.runs_directory, run_name),
                                   )

    info_sac_logs = []
    info_sac = None

    for step in tqdm(range(step, args.total_timesteps), initial=step):

        # Get action.
        selected_actions = agent.get_action(obs, args.eval)

        # Step env.
        next_obs, rewards, terminations, truncations, infos = envs.step(selected_actions)
        if args.eval == True:
            agent.agent_step_eval(next_obs)
        else:
            # Learn.
            info_sac = agent.agent_step(next_obs, selected_actions, rewards, terminations, truncations, infos)

        reward_tracker.update(infos['original_reward'][0])

        if info_sac is not None:
            info_sac_logs.append(info_sac)

        # Save the model.
        if step % args.save_every_n_steps == 0 or step == args.total_timesteps - 1:
            state = agent.get_state()
            torch.save(state, os.path.join(args.runs_directory, run_name, f"weights.pth"))

            replay_buffer = agent.get_replay_buffer()
            replay_buffer.save(os.path.join(args.runs_directory, run_name, f"replay_buffer.npz"))
            # Reward tracker.
            reward_tracker.log()

            # Save to csv.
            df_info_sac_logs = pd.DataFrame(info_sac_logs)
            df_info_sac_logs.to_csv(os.path.join(args.runs_directory, run_name, "info_sac_logs.csv"), index=False)

