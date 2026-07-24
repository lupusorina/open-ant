"""Shared environment factory for embodied Ant and Gymnasium MuJoCo envs."""

from __future__ import annotations

import json
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode
from skrl.envs.wrappers.torch import wrap_env

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../sim")))
from ant_mujoco import AntEnv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../embodied_ant_env")))
from embodied_ant_env import make_ant_env

# Embodied / custom Ant IDs used by this repo (not Gymnasium registry entries).
EMBODIED_ANT_ENV_IDS = {
    "EAnt",
    "SimEmbodiedAnt",
    "HwEmbodiedAnt",
    "CustomAnt-v0",
}


class OriginalRewardWrapper(gym.Wrapper):
    """Store the pre-scaling reward in info['original_reward'] for logging."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["original_reward"] = np.asarray(reward, dtype=np.float64)
        return obs, reward, terminated, truncated, info


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
        env = gym.wrappers.RecordVideo(
            env,
            os.path.join(disk_folder, runs_directory, run_name, "videos", run_name),
            step_trigger=lambda x: x % args.save_every_n_steps == 0,
            video_length=args.save_every_n_steps,
        )
    return env


def _make_gymnasium_env(args, seed, idx, disk_folder, run_name, runs_directory, reward_scale):
    render_mode = args.render_mode
    if args.capture_video and idx == 0 and render_mode is None:
        render_mode = "rgb_array"

    env = gym.make(args.env_id, render_mode=render_mode)
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
    env = gym.wrappers.TransformReward(env, lambda r, scale=reward_scale: r * scale)
    return env


def _make_embodied_ant_env(args, task, seed, idx, disk_folder, run_name, runs_directory, reward_scale):
    joint_config = {
        "hip_zero": 0,
        "knee_zero": -np.radians(50),
        "hip_range": np.radians(30),
        "knee_range": np.radians(20),
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
        with open(args.hw_config, "r") as f:
            cfg = json.load(f)
        env = make_ant_env(
            cfg,
            render_mode=args.render_mode,
            dt=args.dt,
            joint_config=joint_config,
            task=task,
        )
    env = _maybe_record_video(env, args, idx, disk_folder, run_name, runs_directory)
    env.action_space.seed(seed)
    env = gym.wrappers.TransformReward(env, lambda r, scale=reward_scale: r * scale)
    return env


def make_envs(
    args,
    task,
    disk_folder,
    run_name,
    runs_directory="runs",
    *,
    wrap_skrl=True,
    autoreset_mode=AutoresetMode.NEXT_STEP,
):
    """
    Build vectorized envs for either embodied Ant or a Gymnasium MuJoCo env_id.

    Returns (env_raw, envs). When wrap_skrl is False, both are the SyncVectorEnv.
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
        if task is None:
            raise ValueError("Embodied Ant requires a task (forward / back_and_forth)")

    def make_env(seed, idx):
        def _init():
            if use_gym:
                return _make_gymnasium_env(
                    args, seed, idx, disk_folder, run_name, runs_directory, reward_scale
                )
            return _make_embodied_ant_env(
                args, task, seed, idx, disk_folder, run_name, runs_directory, reward_scale
            )

        return _init

    vec_kwargs = {}
    if autoreset_mode is not None:
        vec_kwargs["autoreset_mode"] = autoreset_mode

    env_raw = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i, i) for i in range(args.num_envs)],
        **vec_kwargs,
    )
    assert isinstance(env_raw.single_action_space, gym.spaces.Box), (
        "[!] Only continuous action space is supported."
    )

    if use_gym:
        base = _unwrap_base_env(env_raw.envs[0])
        if hasattr(base, "dt"):
            args.dt = float(base.dt)
            print(f"[√] Synced args.dt to Gymnasium env dt={args.dt}")

    if wrap_skrl:
        envs = wrap_env(env_raw, wrapper="gymnasium")
    else:
        envs = env_raw

    print(f"[√] Created environment with {envs.num_envs} environments.")
    return env_raw, envs


# Backwards-compatible alias.
make_ant_envs = make_envs
