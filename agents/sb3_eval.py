"""Periodic deterministic evaluation on a separate env, using Stable-Baselines3's
evaluate_policy (the function SB3's EvalCallback calls internally).

SB3's EvalCallback itself can't be used directly: it is driven by
BaseAlgorithm.learn() and reads model.get_env(), model.num_timesteps,
model.logger and model.save(). Our SAC/MPO agents are not SB3 algorithms, so
we reuse the part that does the work (evaluate_policy + Monitor + DummyVecEnv)
and do the EvalCallback bookkeeping (evaluations log, best-model save,
after-eval early-stopping hook) here.

evaluate_policy only needs an object with SB3's predict() signature, which
PolicyPredictor provides around a deterministic act_fn.

Each evaluation runs n_eval_episodes envs in parallel (stepped in lockstep in one
DummyVecEnv, policy batched across them), one episode per env. The envs are
re-seeded with seed, seed+1, ... before every evaluation, so every evaluation
starts from the same n_eval_episodes initial states and differences between
evaluations come from the policy, not from luckier/unluckier starts.

Usage:
    evaluator = Evaluator([gym_env_0, ..., gym_env_4], act_fn, action_space, run_dir,
                          n_eval_episodes=5, reward_scale=args.reward_scale, seed=args.seed)
    mean_reward, is_new_best = evaluator.evaluate(step)
"""
import os
import csv

import numpy as np
import gymnasium as gym
import torch
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


class PolicyPredictor:
    """Adapts act_fn(obs_tensor) -> deterministic action tensor to SB3's predict() API."""

    def __init__(self, act_fn, action_space, device):
        self.act_fn = act_fn
        self.action_space = action_space
        self.device = device

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions = self.act_fn(obs)
        actions = actions.detach().cpu().numpy().reshape(obs.shape[0], -1)
        # SB3's own predict() clips Box actions to the action space too.
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return actions, state


class Evaluator:
    def __init__(self, eval_envs, act_fn, action_space, run_dir, device,
                 n_eval_episodes=5, reward_scale=1.0, max_episode_steps=None, seed=None):
        """eval_envs: n_eval_episodes single (non-vector) gym.Envs built like the
        training env; each plays one episode per evaluation.
        reward_scale: the env's TransformReward scale; returns are divided by it
        so they're in the same units as RewardTracker (info['original_reward']).
        seed: env i is reset with seed + i at the start of every evaluation
        (None: no re-seeding, every evaluation gets new random starts)."""
        if len(eval_envs) != n_eval_episodes:
            raise ValueError(f"need one eval env per episode: got {len(eval_envs)} envs, "
                             f"n_eval_episodes={n_eval_episodes}")
        if max_episode_steps is not None:
            eval_envs = [gym.wrappers.TimeLimit(e, max_episode_steps=max_episode_steps) for e in eval_envs]
        self.eval_env = DummyVecEnv([lambda e=e: Monitor(e) for e in eval_envs])
        self.seed = seed
        self.policy = PolicyPredictor(act_fn, action_space, device)
        self.n_eval_episodes = n_eval_episodes
        self.reward_scale = reward_scale
        self.best_mean_reward = -np.inf

        self.log_path = os.path.join(run_dir, "evaluations.csv")
        if os.path.exists(self.log_path):  # resume: keep appending, restore best
            with open(self.log_path, newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                self.best_mean_reward = max(float(r["mean_reward"]) for r in rows)

    def evaluate(self, step):
        """Run n_eval_episodes deterministic episodes. Returns (mean_reward, is_new_best)."""
        if self.seed is not None:
            # Applied at evaluate_policy's reset (then cleared), so every evaluation
            # starts env i from the same seed + i initial state.
            self.eval_env.seed(self.seed)
        episode_rewards, episode_lengths = evaluate_policy(
            self.policy, self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
            return_episode_rewards=True,
            warn=False,
        )
        episode_rewards = np.asarray(episode_rewards, dtype=np.float64) / self.reward_scale
        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        mean_ep_length = float(np.mean(episode_lengths))

        is_new_best = mean_reward > self.best_mean_reward
        if is_new_best:
            self.best_mean_reward = mean_reward

        write_header = not os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["step", "mean_reward", "std_reward", "mean_ep_length", "episode_rewards"])
            writer.writerow([step, mean_reward, std_reward, mean_ep_length,
                             " ".join(f"{r:.4f}" for r in episode_rewards)])

        print(f"[eval] step {step}: mean_reward {mean_reward:.2f} +/- {std_reward:.2f}, "
              f"ep_length {mean_ep_length:.0f}" + ("  (new best)" if is_new_best else ""))
        return mean_reward, is_new_best

    def close(self):
        self.eval_env.close()
