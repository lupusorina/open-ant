# This file is adapted from CleanRL (https://github.com/vwxyzjn/cleanrl)
# Copyright (c) 2019 CleanRL developers
# Licensed under the MIT License (see LICENSE file)

from typing import Tuple
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
from torchrl.data import ReplayBuffer, LazyTensorStorage, RandomSampler
from tensordict import TensorDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
from ant_mujoco import AntEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import make_ant_env, ForwardTask, BackAndForthTask
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from reward import RewardTracker

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
        """Returns (action, log_prob, squashed mean, pre-tanh mean, pre-tanh log_std)."""
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
        squashed_mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, squashed_mean, mean, log_std


def compute_parametric_kl_penalty_and_dual_loss(
    kl: torch.Tensor,
    beta: torch.Tensor,
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """beta-weighted KL penalty for the actor and the dual loss that adapts beta (as in mpo_acme.py)."""
    mean_kl = kl.mean(dim=0)  # (D,)
    loss_kl = torch.sum(beta.detach() * mean_kl)  # actor sees gradients through KL, not beta
    loss_beta = torch.sum(beta * (epsilon - mean_kl.detach()))
    return loss_kl, loss_beta


def make_ant_envs(args, task, disk_folder, run_name, runs_directory='runs'):
    """Create the vectorized environment outside the SAC class."""
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
                                   dt=args.dt,
                                   joint_config=joint_config,
                                   task=task,
                                   )
                # env.metadata['render_fps'] = 1/args.dt

            if capture_video and idx == 0:
                print('RecordVideo')
                env = gym.wrappers.RecordVideo(env, os.path.join(disk_folder, runs_directory, run_name, "videos", run_name),
                                               step_trigger=lambda x: x % args.save_every_n_steps == 0, video_length=args.save_every_n_steps)
            env.action_space.seed(seed)
            # Reward scaling.
            env = gym.wrappers.TransformReward(env, lambda reward: reward * args.reward_scale)
            return env
        return _init

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "[!] Only continuous action space is supported."
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
                 use_kl_trust_region: bool = False,
                 epsilon_mean: float = 2.5e-3,
                 epsilon_stddev: float = 1e-6,
                 init_log_beta_mean: float = 10.0,
                 init_log_beta_stddev: float = 1000.0,
                 dual_lr: float = 1e-3,
                 max_grad_norm: float = 40.0,
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
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=policy_lr)

        self.learning_starts = learning_starts
        self.autotune = autotune
        self.batch_size = batch_size
        self.gamma = gamma
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.tau = tau
        self.dt = dt
        self.max_grad_norm = max_grad_norm

        self.record_infos_sac = record_infos_sac

        # MPO-style decoupled KL trust region on the actor: KL(pi_target || pi_online),
        # split into a mean term and a stddev term, each with its own Lagrange multiplier
        # (beta_mean, beta_stddev) learned via a dual loss. The target actor is Polyak-averaged
        # like the target critics.
        self.use_kl_trust_region = use_kl_trust_region
        if self.use_kl_trust_region:
            act_dim = int(np.prod(self.envs.single_action_space.shape))
            self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
            self.epsilon_mean = epsilon_mean
            self.epsilon_stddev = epsilon_stddev
            self.log_beta_mean = torch.full((act_dim,), init_log_beta_mean, requires_grad=True, device=self.device)
            self.log_beta_stddev = torch.full((act_dim,), init_log_beta_stddev, requires_grad=True, device=self.device)
            self.beta_optimizer = optim.Adam([self.log_beta_mean, self.log_beta_stddev], lr=dual_lr)

        # Alpha (entropy coefficient).
        if self.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.envs.single_action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.a_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
            self.log_alpha = None
            self.a_optimizer = None

        self.envs.single_observation_space.dtype = np.float32

        # Replay buffer.
        self.rb = ReplayBuffer(
                storage=LazyTensorStorage(buffer_size, device=device),
                sampler=RandomSampler(),
                batch_size=batch_size,
            )

        self.global_step = 0
        self.obs = None
        
    def get_action(self, obs, evaluate=False):
        if self.obs is None:
            self.obs = obs
            return np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])

        # If we are in evaluation mode or we have reached the learning starts, get the action from the actor.
        if evaluate == True or self.global_step > self.learning_starts:
            actions, _, _, _, _ = self.actor.get_action(torch.Tensor(self.obs).to(self.device))
            actions = actions.detach().cpu().numpy()
            return actions

        # Otherwise, pick a random action.
        actions = np.array([self.envs.single_action_space.sample() for _ in range(self.envs.num_envs)])

        return actions
 
    def agent_step(self, next_obs, actions, rewards, terminations, truncations, infos):

        tensor_dict = TensorDict({
            "observations": torch.as_tensor(self.obs, dtype=torch.float32, device=self.device),
            "next_observations": torch.as_tensor(next_obs, dtype=torch.float32, device=self.device),
            "actions": torch.as_tensor(actions, dtype=torch.float32, device=self.device),
            "rewards": torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1),
            "terminations": torch.as_tensor(terminations, dtype=torch.float32, device=self.device).unsqueeze(-1),                     # or use "terminated" + "truncated" separately if you prefer
        }, batch_size=[self.envs.num_envs])

        # Use .extend() instead of .add() when adding a batch of transitions.
        self.rb.extend(tensor_dict)
 
        self.global_step += 1
        self.obs = next_obs

        qf1_loss = None
        qf2_loss = None
        alpha_loss = None
        actor_loss = None
        pi_stddev_min = None
        pi_stddev_min_squashed = None
        qf1_values_mean = None
        qf2_values_mean = None
        kl_stats = {}

        # Learning.
        if self.global_step > self.learning_starts:
            data, info_buffer = self.rb.sample(self.batch_size, return_info=True)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _, _, _ = self.actor.get_action(data["next_observations"])
                qf1_next_target = self.qf1_target(data["next_observations"], next_state_actions)
                qf2_next_target = self.qf2_target(data["next_observations"], next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = data["rewards"].flatten() * self.dt + \
                                (1 - data["terminations"].flatten()) * (self.gamma ** self.dt) * (min_qf_next_target).view(-1)
                                # See K. De Asis, R. Sutton, "An Idiosyncrasy of Time-discretization in Reinforcement Learning"
            qf1_a_values = self.qf1(data["observations"], data["actions"]).view(-1)
            qf2_a_values = self.qf2(data["observations"], data["actions"]).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            qf1_values_mean = qf1_a_values.mean().item()
            qf2_values_mean = qf2_a_values.mean().item()

            # Optimize the Action-Value networks.
            self.q_optimizer.zero_grad()
            qf_loss.backward()
            if self.max_grad_norm is not None:
                # Per-critic like MPO, not over the joint qf1+qf2 parameter set.
                nn.utils.clip_grad_norm_(self.qf1.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.qf2.parameters(), self.max_grad_norm)
            self.q_optimizer.step()

            if self.global_step % self.policy_frequency == 0:
                for _ in range(
                        self.policy_frequency
                ):  # Compensate for the delay by doing 'actor_update_interval' instead of 1.
                    pi, log_pi, _, online_mu, online_log_std = self.actor.get_action(data["observations"])
                    qf1_pi = self.qf1(data["observations"], pi)
                    qf2_pi = self.qf2(data["observations"], pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    sac_actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
                    actor_loss = sac_actor_loss
                    loss_beta = None

                    if self.use_kl_trust_region:
                        with torch.no_grad():
                            self.log_beta_mean.clamp_(min=-18.0)
                            self.log_beta_stddev.clamp_(min=-18.0)
                            # Target only needs the Gaussian params, no sample.
                            target_mu, target_log_std = self.actor_target(data["observations"])

                        beta_mean = F.softplus(self.log_beta_mean) + 1e-8
                        beta_stddev = F.softplus(self.log_beta_stddev) + 1e-8

                        # Decoupled KL(pi_target || pi_online) on the pre-tanh Gaussians, as in MPO.
                        target_sigma = target_log_std.exp()
                        online_sigma = online_log_std.exp()
                        
                        fixed_stddev_distribution = torch.distributions.Normal(online_mu, target_sigma)
                        fixed_mean_distribution = torch.distributions.Normal(target_mu, online_sigma)
                        target_distribution = torch.distributions.Normal(target_mu, target_sigma)
                        kl_mean = torch.distributions.kl_divergence(target_distribution, fixed_stddev_distribution)

                        kl_stddev = torch.distributions.kl_divergence(target_distribution, fixed_mean_distribution)
                        
                        loss_kl_mean, loss_beta_mean = compute_parametric_kl_penalty_and_dual_loss(
                            kl_mean, beta_mean, self.epsilon_mean)
                        loss_kl_stddev, loss_beta_stddev = compute_parametric_kl_penalty_and_dual_loss(
                            kl_stddev, beta_stddev, self.epsilon_stddev)

                        loss_kl_penalty = loss_kl_mean + loss_kl_stddev
                        loss_beta = loss_beta_mean + loss_beta_stddev
                        actor_loss = sac_actor_loss + loss_kl_penalty
                        kl_stats = {
                            'kl_mean': kl_mean.detach().mean().item(),
                            'kl_stddev': kl_stddev.detach().mean().item(),
                            'kl_mean_rel': kl_mean.detach().mean().item() / self.epsilon_mean,
                            'kl_stddev_rel': kl_stddev.detach().mean().item() / self.epsilon_stddev,
                            'dual_beta_mean': beta_mean.detach().mean().item(),
                            'dual_beta_stddev': beta_stddev.detach().mean().item(),
                            'sac_actor_loss': sac_actor_loss.item(),
                            'loss_kl_penalty': loss_kl_penalty.item(),
                            'loss_beta': loss_beta.item(),
                        }

                    # Actor sees gradients through the KL only, betas through the dual only,
                    # so one backward on the sum is enough (same as MPO).
                    self.actor_optimizer.zero_grad()
                    if loss_beta is None:
                        actor_loss.backward()
                    else:
                        self.beta_optimizer.zero_grad()
                        (actor_loss + loss_beta).backward()
                        self.beta_optimizer.step()
                    if self.max_grad_norm is not None:
                        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    self.actor_optimizer.step()


                    # Same as MPO: narrowest action dim's std per state, averaged over the batch.
                    # pi_stddev_min is the pre-tanh Gaussian std; pi_stddev_min_squashed is the
                    # std of the executed action after tanh (action units, comparable to MPO),
                    # estimated from 16 samples per state.
                    with torch.no_grad():
                        mean, log_std = self.actor(data["observations"])
                        std = log_std.exp()
                        pi_stddev_min = std.min(dim=-1).values.mean().item()
                        x = mean + std * torch.randn((16, *mean.shape), device=mean.device)
                        squashed_std = (torch.tanh(x) * self.actor.action_scale).std(dim=0)  # [B, act_dim]
                        pi_stddev_min_squashed = squashed_std.min(dim=-1).values.mean().item()

                    if self.autotune:
                        with torch.no_grad():
                            _, log_pi, _, _, _ = self.actor.get_action(data["observations"])
                        alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                        self.a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.a_optimizer.step()
                        self.alpha = self.log_alpha.exp().item()

            # Update the target networks.
            if self.global_step % self.target_network_frequency == 0:
                for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)
                if self.use_kl_trust_region:
                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(
                            self.tau * param.data + (1 - self.tau) * target_param.data)

        info_sac = None
        if self.record_infos_sac == True:
            info_sac = {
                'global_step': self.global_step,
                'qf1_loss': qf1_loss.item() if qf1_loss is not None else qf1_loss,
                'qf2_loss': qf2_loss.item() if qf2_loss is not None else qf2_loss,
                'qf1_values_mean': qf1_values_mean if qf1_values_mean is not None else qf1_values_mean,
                'qf2_values_mean': qf2_values_mean if qf2_values_mean is not None else qf2_values_mean,
                'actor_loss': actor_loss.item() if actor_loss is not None else actor_loss,
                'alpha': self.alpha,
                'alpha_loss': alpha_loss.item() if alpha_loss is not None else alpha_loss,
                'pi_stddev_min': pi_stddev_min,
                'pi_stddev_min_squashed': pi_stddev_min_squashed,
                'rewards': rewards.flatten().tolist(),
                'original_rewards': infos['original_reward'].flatten().tolist(),
            }
            info_sac.update(kl_stats)
        return info_sac

    def agent_step_eval(self, next_obs):
        self.global_step += 1
        self.obs = next_obs

    def set_global_step(self, global_step: int):
        self.global_step = global_step

    def get_replay_buffer(self):
        return self.rb

    def load_replay_buffer(self, replay_buffer_path):
        self.rb.loads(replay_buffer_path)

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
        if self.use_kl_trust_region:
            state["actor_target"] = self.actor_target.state_dict()
            state["log_beta_mean"] = self.log_beta_mean.cpu().detach()
            state["log_beta_stddev"] = self.log_beta_stddev.cpu().detach()
            state["beta_optimizer"] = self.beta_optimizer.state_dict()
        return state

    def load_state(self, state_dict):
        """Loads the full state of the agent from a state dictionary."""
        self.actor.load_state_dict(state_dict["actor"])
        self.qf1.load_state_dict(state_dict["qf1"])
        self.qf2.load_state_dict(state_dict["qf2"])
        self.qf1_target.load_state_dict(state_dict["qf1_target"])
        self.qf2_target.load_state_dict(state_dict["qf2_target"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
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
                self.a_optimizer.load_state_dict(state_dict["a_optimizer"])
        if self.use_kl_trust_region:
            required = ["actor_target", "log_beta_mean", "log_beta_stddev", "beta_optimizer"]
            missing = [k for k in required if k not in state_dict]
            if not missing:
                self.actor_target.load_state_dict(state_dict["actor_target"])
                self.log_beta_mean.data = state_dict["log_beta_mean"].to(self.device)
                self.log_beta_stddev.data = state_dict["log_beta_stddev"].to(self.device)
                self.beta_optimizer.load_state_dict(state_dict["beta_optimizer"])
            elif len(missing) == len(required):
                # Checkpoint from a run without the KL (e.g. plain-SAC Sim1): turn the trust
                # region on from here, anchored at the loaded actor, with fresh betas
                # (their __init__ values) and a fresh beta optimizer.
                print("[i] checkpoint has no KL state; starting the KL trust region from the loaded actor")
                self.actor_target.load_state_dict(self.actor.state_dict())
            else:
                raise KeyError(f"checkpoint has partial KL state (missing: {missing})")


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
    parser.add_argument("--gamma", type=float, default=0.92,
                        help="discount factor")
    parser.add_argument("--no-use_layer_norm", action="store_false",
                        dest="use_layer_norm",
                        help="disable layer normalization in networks")

    # MPO-style KL trust region on the actor.
    parser.add_argument("--use_kl_trust_region", action="store_true", default=False,
                        help="add decoupled KL(pi_target || pi) mean/stddev constraints to the actor loss")
    parser.add_argument("--epsilon_mean", type=float, default=2.5e-3,
                        help="KL bound for the mean (same default as MPO epsilon_mu_kl)")
    parser.add_argument("--epsilon_stddev", type=float, default=1e-6,
                        help="KL bound for the stddev (same default as MPO epsilon_sigma_kl)")
    parser.add_argument("--init_log_beta_mean", type=float, default=10.0,
                        help="initial pre-softplus Lagrange multiplier for the mean KL")
    parser.add_argument("--init_log_beta_stddev", type=float, default=1000.0,
                        help="initial pre-softplus Lagrange multiplier for the stddev KL")
    parser.add_argument("--dual_lr", type=float, default=1e-3,
                        help="learning rate for beta_mean, beta_stddev")
    parser.add_argument("--max_grad_norm", type=float, default=40.0,
                        help="clip actor/critic grad norms (same default as MPO)")

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
                use_kl_trust_region=args.use_kl_trust_region,
                epsilon_mean=args.epsilon_mean,
                epsilon_stddev=args.epsilon_stddev,
                init_log_beta_mean=args.init_log_beta_mean,
                init_log_beta_stddev=args.init_log_beta_stddev,
                dual_lr=args.dual_lr,
                max_grad_norm=args.max_grad_norm)

    step = 0
    # Load the model.
    if args.weights_path is not None:
        ckpt_path = os.path.abspath(os.path.join(args.weights_path, "weights.pth"))
        buffer_path = os.path.abspath(os.path.join(args.weights_path, "replay_buffer"))
        state = torch.load(ckpt_path)
        print("==========================================")
        print(f"[i] loading checkpoint: {ckpt_path}")
        print(f"[i] loading replay buffer: {buffer_path}")
        print(f"[i] resuming from global_step {state['global_step']}, "
              f"KL state in checkpoint: {'actor_target' in state}")
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

    obs, info = envs.reset(seed=args.seed)

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

        if any(truncations) or any(terminations):
            envs.reset()

        # Save the model.
        if step % args.save_every_n_steps == 0 or step == args.total_timesteps - 1:
            state = agent.get_state()
            torch.save(state, os.path.join(args.runs_directory, run_name, f"weights.pth"))

            replay_buffer = agent.get_replay_buffer()
            replay_buffer.dumps(os.path.join(args.runs_directory, run_name, f"replay_buffer"))
            # Reward tracker.
            reward_tracker.log()

            # Save to csv.
            df_info_sac_logs = pd.DataFrame(info_sac_logs)
            df_info_sac_logs.to_csv(os.path.join(args.runs_directory, run_name, "info_sac_logs.csv"), index=False)
