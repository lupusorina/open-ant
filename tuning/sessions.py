"""One training session per agent family.

Each session owns the environments, the agent and the reward tracker for a
single trial, and exposes the same handful of operations to the driver. The
agents themselves differ enough in how they are constructed and stepped that
the differences live here rather than in the driver.
"""

import copy
import os
from typing import Optional

import numpy as np
import torch

_DUAL_PARAM_NAMES = ("log_eta", "log_alpha_mean", "log_alpha_stddev", "log_penalty_eta")

_SAC_NETWORK_NAMES = ("actor", "qf1", "qf2", "qf1_target", "qf2_target")


def _cpu(state_dict):
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def snapshot_weights(agent) -> dict:
    weights = {
        "actor": _cpu(agent.actor.state_dict()),
        "actor_target": _cpu(agent.actor_target.state_dict()),
        "critics": [_cpu(critic.state_dict()) for critic in agent.critics],
        "target_critics": [_cpu(critic.state_dict()) for critic in agent.target_critics],
    }
    for name in _DUAL_PARAM_NAMES:
        weights[name] = getattr(agent, name).detach().cpu().clone()
    return weights


def snapshot_sac_weights(agent) -> dict:
    weights = {name: _cpu(getattr(agent, name).state_dict()) for name in _SAC_NETWORK_NAMES}
    if getattr(agent, "autotune", False) and agent.log_alpha is not None:
        weights["log_alpha"] = agent.log_alpha.detach().cpu().clone()
    return weights


def _load_sac_weights(agent, weights) -> None:
    for name in _SAC_NETWORK_NAMES:
        getattr(agent, name).load_state_dict(weights[name])
    if "log_alpha" in weights and agent.log_alpha is not None:
        with torch.no_grad():
            agent.log_alpha.copy_(
                torch.as_tensor(
                    weights["log_alpha"],
                    dtype=agent.log_alpha.dtype,
                    device=agent.log_alpha.device,
                ).reshape_as(agent.log_alpha)
            )
        agent.alpha = agent.log_alpha.exp().item()


def weights_from_checkpoint(weights_path: str) -> dict:
    checkpoint_files = [f for f in os.listdir(weights_path) if f.endswith(".pth")]
    checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    checkpoint = torch.load(
        os.path.join(weights_path, checkpoint_files[-1]), map_location="cpu"
    )

    if "qf1" in checkpoint:
        weights = {name: checkpoint[name] for name in _SAC_NETWORK_NAMES}
        if "log_alpha" in checkpoint:
            weights["log_alpha"] = checkpoint["log_alpha"]
        return weights

    def _list(plural, singular):
        return checkpoint[plural] if plural in checkpoint else [checkpoint[singular]]

    weights = {
        "actor": checkpoint["actor"],
        "actor_target": checkpoint["actor_target"],
        "critics": _list("critics", "critic"),
        "target_critics": _list("target_critics", "target_critic"),
    }
    for name in _DUAL_PARAM_NAMES:
        weights[name] = checkpoint[name]
    return weights


def _load_weights(agent, weights) -> None:
    agent.actor.load_state_dict(weights["actor"])
    agent.actor_target.load_state_dict(weights["actor_target"])
    for critic, state_dict in zip(agent.critics, weights["critics"]):
        critic.load_state_dict(state_dict)
    for target_critic, state_dict in zip(agent.target_critics, weights["target_critics"]):
        target_critic.load_state_dict(state_dict)
    with torch.no_grad():
        for name in _DUAL_PARAM_NAMES:
            param = getattr(agent, name)
            param.copy_(
                torch.as_tensor(
                    weights[name], dtype=param.dtype, device=param.device
                ).reshape_as(param)
            )


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


def _evaluate_rollouts(eval_envs, act, dt, n_episodes, eval_seed_start, max_steps) -> dict:
    total_reward = 0.0
    total_steps = 0
    episodes_terminated = 0
    try:
        for i in range(n_episodes):
            obs, _ = eval_envs.reset(seed=eval_seed_start + i)
            for _ in range(max_steps):
                obs, rewards, terminations, truncations, _ = eval_envs.step(act(obs))
                total_reward += float(np.asarray(rewards).sum())
                total_steps += 1
                if np.any(terminations) or np.any(truncations):
                    if np.any(terminations):
                        episodes_terminated += 1
                    break
    finally:
        eval_envs.close()
    return {
        "reward_per_second": (
            float("nan") if total_steps == 0 else total_reward / (total_steps * dt)
        ),
        "episodes_terminated": episodes_terminated,
        "total_steps": total_steps,
    }


def _eval_args(args, env_id: str):
    eval_args = copy.deepcopy(args)
    eval_args.env_id = env_id
    eval_args.num_envs = 1
    eval_args.capture_video = False
    return eval_args


def _reward_rate(tracker, args) -> Optional[float]:
    if tracker is None:
        return None
    average = tracker.average_reward_per_second
    if average is None:
        return None
    scale = getattr(args, "reward_scale", None)
    return float(average) * (1.0 if scale is None else float(scale))


class MpoSession:
    """The unified MPO agent, covering both critic types and any ensemble size."""

    def __init__(self, module, args, run_name, *, weights=None):
        self.module = module
        self.args = args
        self.run_name = run_name
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
        if weights is not None:
            _load_weights(self.agent, weights)

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

    def evaluate(
        self,
        n_episodes: int,
        eval_seed_start: int,
        max_steps: int,
        env_id: str = "SimEmbodiedAnt",
    ) -> dict:
        eval_args = _eval_args(self.args, env_id)
        _, eval_envs = self.module.make_envs(
            eval_args,
            _make_task(eval_args),
            "",
            f"{self.run_name}_eval",
            runs_directory=eval_args.runs_directory,
            wrap_skrl=False,
        )
        device = self.agent.device

        def act(obs):
            obs_tensor = torch.as_tensor(
                np.asarray(obs), dtype=torch.float32, device=device
            )
            with torch.no_grad():
                return self.agent.actor.forward(obs_tensor).mean.squeeze(1).cpu().numpy()

        return _evaluate_rollouts(
            eval_envs, act, self.args.dt, n_episodes, eval_seed_start, max_steps
        )

    @property
    def global_step(self) -> int:
        return self.agent.global_step

    @property
    def reward_rate(self) -> Optional[float]:
        return _reward_rate(getattr(self.agent, "reward_tracker", None), self.args)

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

    def __init__(self, module, args, run_name, *, weights=None):
        from agents.reward import RewardTracker

        self.module = module
        self.args = args
        self.run_name = run_name
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
            policy_layer_sizes=args.policy_layer_sizes,
            critic_layer_sizes=args.critic_layer_sizes,
        )
        if weights is not None:
            _load_sac_weights(self.agent, weights)
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

    def evaluate(
        self,
        n_episodes: int,
        eval_seed_start: int,
        max_steps: int,
        env_id: str = "SimEmbodiedAnt",
    ) -> dict:
        eval_args = _eval_args(self.args, env_id)
        eval_envs = self.module.make_ant_envs(
            args=eval_args,
            task=_make_task(eval_args),
            disk_folder="",
            run_name=f"{self.run_name}_eval",
            runs_directory=eval_args.runs_directory,
        )
        device = self.agent.device

        def act(obs):
            obs_tensor = torch.as_tensor(
                np.asarray(obs), dtype=torch.float32, device=device
            )
            with torch.no_grad():
                _action, _log_prob, mean = self.agent.actor.get_action(obs_tensor)
            return mean.cpu().numpy()

        return _evaluate_rollouts(
            eval_envs, act, self.args.dt, n_episodes, eval_seed_start, max_steps
        )

    @property
    def global_step(self) -> int:
        return self.agent.global_step

    @property
    def reward_rate(self) -> Optional[float]:
        return _reward_rate(self.tracker, self.args)

    def close(self):
        self.tracker.log()
        self.envs.close()
