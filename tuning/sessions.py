"""One training session per agent family.

Each session owns the environments, the agent and the reward tracker for a
single trial, and exposes the same handful of operations to the driver. The
agents themselves differ enough in how they are constructed and stepped that
the differences live here rather than in the driver.
"""

import os
from typing import Optional

import numpy as np
import torch


def _make_task(args):
    from embodied_ant_env import BackAndForthTask, ForwardTask

    if args.task_type == "forward":
        return ForwardTask()
    if args.task_type == "back_and_forth":
        return BackAndForthTask(
            radius=args.radius_back_and_forth,
            origin=np.array(args.origin_back_and_forth),
        )
    raise ValueError(f"Invalid task type: {args.task_type}")


def _reward_rate(tracker) -> Optional[float]:
    if tracker is None:
        return None
    average = tracker.average_reward_per_second
    return None if average is None else float(average)


class MpoSession:
    """The unified MPO agent, covering both critic types and any ensemble size."""

    def __init__(self, module, args, run_name):
        self.args = args
        self.raw_env, self.envs = module.make_envs(
            args, _make_task(args), "", run_name, runs_directory=args.runs_directory
        )
        self.agent = module.MPO(
            args=args,
            envs=self.envs,
            disk_folder="",
            run_name=run_name,
            runs_directory=args.runs_directory,
        )

    def reset(self):
        obs, info = self.envs.reset()
        self.agent.initialize_logging(info)
        return obs

    def step(self, obs):
        actions = self.agent.get_action(obs, False)
        next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)
        metrics = self.agent.agent_step(
            next_obs, actions, rewards, terminations, truncations, infos
        )
        self.agent.log_step(self.agent.global_step, infos, rewards, metrics)
        return next_obs

    @property
    def global_step(self) -> int:
        return self.agent.global_step

    @property
    def reward_rate(self) -> Optional[float]:
        return _reward_rate(getattr(self.agent, "reward_tracker", None))

    def close(self):
        try:
            self.agent.cleanup()
        finally:
            try:
                self.envs.close()
            finally:
                if self.raw_env is not None:
                    self.raw_env.close()


class SacSession:
    """SAC, which builds its agent from explicit arguments and tracks rewards outside it."""

    def __init__(self, module, args, run_name):
        from agents.reward import RewardTracker

        self.args = args
        self.envs = module.make_ant_envs(
            args=args,
            task=_make_task(args),
            disk_folder="",
            run_name=run_name,
            runs_directory=args.runs_directory,
        )
        self.agent = module.SAC(
            envs=self.envs,
            device=torch.device(
                "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
            ),
            seed=args.seed,
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
            dt=args.dt,
            torch_deterministic=args.torch_deterministic,
        )
        self.tracker = RewardTracker(
            env_dt=args.dt,
            env_id=args.env_id,
            time_window=120.0,
            log_folder=os.path.join(args.runs_directory, run_name),
        )

    def reset(self):
        obs, _info = self.envs.reset(seed=self.args.seed)
        return obs

    def step(self, obs):
        actions = self.agent.get_action(obs, False)
        next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)
        self.agent.agent_step(
            next_obs, actions, rewards, terminations, truncations, infos
        )
        self.tracker.update(infos["original_reward"][0])
        if any(truncations) or any(terminations):
            self.envs.reset()
        return next_obs

    @property
    def global_step(self) -> int:
        return self.agent.global_step

    @property
    def reward_rate(self) -> Optional[float]:
        return _reward_rate(self.tracker)

    def close(self):
        self.tracker.log()
        self.envs.close()
