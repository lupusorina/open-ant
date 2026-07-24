"""Roll out MPO checkpoints and write evaluation videos for training runs.

Examples:
  # One video per env (seed 0), latest checkpoint
  python eval_videos.py --one_per_env

  # All hopper seeds
  python eval_videos.py --env hopper

  # Specific run / checkpoint / episode count
  python eval_videos.py --run_dir runs/mpo_hopper_20260723-172305_seed_0 \\
      --checkpoint_step 1200000 --num_episodes 2

Headless rendering uses MUJOCO_GL=egl by default (override with MUJOCO_GL=osmesa
or MUJOCO_GL=glfw if needed).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from types import SimpleNamespace

# Must run before importing mujoco / gymnasium MuJoCo envs.
# Offline video capture works best with EGL; override via MUJOCO_GL if needed.
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
import torch

try:
    from .envs import is_gymnasium_env
    from .nn import AcmeActor
    from .plot_runs import env_from_dir, find_run_dirs, seed_from_dir
except ImportError:
    from envs import is_gymnasium_env
    from nn import AcmeActor
    from plot_runs import env_from_dir, find_run_dirs, seed_from_dir

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../sim")))
from ant_mujoco import AntEnv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../embodied_ant_env")))
from embodied_ant_env import BackAndForthTask, ForwardTask

import gymnasium as gym


def load_run_args(run_dir: str) -> SimpleNamespace:
    args_path = os.path.join(run_dir, "weights_and_args", "args.json")
    with open(args_path, "r") as f:
        raw = json.load(f)
    return SimpleNamespace(**raw)


def list_checkpoints(weights_dir: str) -> list[tuple[int, str]]:
    files = []
    for path in glob.glob(os.path.join(weights_dir, "checkpoint_*.pth")):
        step = int(os.path.basename(path).split("_")[-1].split(".")[0])
        files.append((step, path))
    return sorted(files, key=lambda x: x[0])


def resolve_checkpoint(weights_dir: str, checkpoint_step: int | None) -> tuple[int, str]:
    checkpoints = list_checkpoints(weights_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {weights_dir}")
    if checkpoint_step is None:
        return checkpoints[-1]
    for step, path in checkpoints:
        if step == checkpoint_step:
            return step, path
    available = ", ".join(str(s) for s, _ in checkpoints[-5:])
    raise FileNotFoundError(
        f"checkpoint_{checkpoint_step}.pth not found in {weights_dir}. "
        f"Latest available: {available}"
    )


def make_eval_env(args: SimpleNamespace):
    """Single non-vectorized env with rgb_array rendering for video capture."""
    if is_gymnasium_env(args.env_id):
        env = gym.make(args.env_id, render_mode="rgb_array")
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
        return env

    task_type = getattr(args, "task_type", "back_and_forth")
    if task_type == "forward":
        task = ForwardTask()
    elif task_type == "back_and_forth":
        task = BackAndForthTask(
            radius=getattr(args, "radius_back_and_forth", 0.3),
            origin=np.array(getattr(args, "origin_back_and_forth", [0.75, -0.3])),
        )
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    joint_config = {
        "hip_zero": 0,
        "knee_zero": -np.radians(50),
        "hip_range": np.radians(30),
        "knee_range": np.radians(20),
    }
    model_path = getattr(
        args,
        "model_path",
        "../../sim/assets/ant_with_camera_after_sys_id.xml",
    )
    if not os.path.isabs(model_path):
        model_path = os.path.join(os.path.dirname(__file__), model_path)

    return AntEnv(
        control_dt=getattr(args, "dt", 0.12),
        render_mode="rgb_array",
        terminate_on_upside_down=getattr(args, "terminate_on_upside_down", True),
        task=task,
        joint_config=joint_config,
        model_path=model_path,
    )


def build_actor(env, args: SimpleNamespace, device: torch.device) -> AcmeActor:
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    actor = AcmeActor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_low=env.action_space.low.astype(np.float32),
        act_high=env.action_space.high.astype(np.float32),
        layer_sizes=tuple(getattr(args, "policy_layer_sizes", [256, 256, 256])),
        init_scale=float(getattr(args, "policy_init_scale", 0.5)),
        min_scale=float(getattr(args, "policy_min_scale", 1e-4)),
    ).to(device)
    actor.eval()
    return actor


def resolve_fps(env, args: SimpleNamespace, fps: float | None) -> float:
    """Playback FPS for the written mp4 (not the physics rate).

    Gymnasium MuJoCo envs advertise render_fps≈125. Embodied Ant uses a large
    control dt (0.12s → ~8 fps), which looks sluggish if used as playback rate,
    so we floor playback at 30 fps unless the user overrides --fps.
    """
    if fps is not None:
        return float(fps)
    meta_fps = (getattr(env, "metadata", {}) or {}).get("render_fps")
    if meta_fps:
        return float(meta_fps)
    dt = float(getattr(env.unwrapped, "dt", getattr(args, "dt", 0.05)) or 0.05)
    return max(30.0, 1.0 / dt)


@torch.no_grad()
def select_action(actor: AcmeActor, obs: np.ndarray, device: torch.device, deterministic: bool):
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    dist = actor(obs_t)
    action = dist.mean if deterministic else dist.sample()
    return action.squeeze(0).cpu().numpy().astype(np.float32)


def rollout_episode(
    env,
    actor: AcmeActor,
    device: torch.device,
    *,
    seed: int,
    target_frames: int,
    deterministic: bool,
) -> tuple[list[np.ndarray], list[float], int]:
    """Collect frames for a fixed-length video, auto-resetting if episodes end early."""
    obs, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    returns: list[float] = []
    ep_return = 0.0
    steps = 0
    episode_idx = 0

    frame = env.render()
    if frame is not None:
        frames.append(np.asarray(frame, dtype=np.uint8))

    while len(frames) < target_frames:
        action = select_action(actor, obs, device, deterministic=deterministic)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_return += float(reward)
        steps += 1
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame, dtype=np.uint8))
        if terminated or truncated:
            returns.append(ep_return)
            episode_idx += 1
            obs, _ = env.reset(seed=seed + episode_idx)
            ep_return = 0.0
            frame = env.render()
            if frame is not None and len(frames) < target_frames:
                frames.append(np.asarray(frame, dtype=np.uint8))

    if ep_return != 0.0 or not returns:
        returns.append(ep_return)
    return frames[:target_frames], returns, steps


def write_video(path: str, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise RuntimeError(f"No frames to write for {path}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
    duration = len(frames) / fps
    print(f"[√] Wrote {path} ({len(frames)} frames @ {fps:.1f} fps, {duration:.1f}s)")


def select_runs(
    runs_dir: str,
    *,
    run_dir: str | None,
    env_name: str | None,
    seed: int | None,
    prefix: str | None,
    one_per_env: bool,
) -> list[str]:
    if run_dir is not None:
        path = run_dir if os.path.isabs(run_dir) else os.path.join(os.path.dirname(__file__), run_dir)
        if not os.path.isdir(path):
            path = os.path.join(runs_dir, os.path.basename(run_dir))
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return [path]

    run_dirs = find_run_dirs(runs_dir, prefix=prefix)
    if env_name is not None:
        env_name = env_name.lower()
        run_dirs = [d for d in run_dirs if env_from_dir(d) == env_name]
    if seed is not None:
        run_dirs = [d for d in run_dirs if seed_from_dir(d) == seed]

    if one_per_env:
        chosen = {}
        for d in sorted(run_dirs, key=seed_from_dir):
            env = env_from_dir(d)
            if env is not None and env not in chosen:
                chosen[env] = d
        run_dirs = [chosen[k] for k in sorted(chosen)]

    return run_dirs


def evaluate_run(
    run_dir: str,
    *,
    checkpoint_step: int | None,
    num_episodes: int,
    duration: float,
    fps: float | None,
    deterministic: bool,
    device: torch.device,
    output_dir: str | None,
) -> None:
    args = load_run_args(run_dir)
    weights_dir = os.path.join(run_dir, "weights_and_args")
    step, ckpt_path = resolve_checkpoint(weights_dir, checkpoint_step)
    print(f"\n=== {os.path.basename(run_dir)} @ step {step} ===")

    env = make_eval_env(args)
    actor = build_actor(env, args, device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint["actor"])
    print(f"[√] Loaded actor from {ckpt_path}")

    fps = resolve_fps(env, args, fps)
    target_frames = max(1, int(round(duration * fps)))
    print(f"[√] Target video length: {duration:.1f}s ({target_frames} frames @ {fps:.1f} fps)")

    video_root = output_dir or os.path.join(run_dir, "videos")
    os.makedirs(video_root, exist_ok=True)
    base_seed = int(getattr(args, "seed", 0))

    returns = []
    for ep in range(num_episodes):
        frames, ep_returns, ep_steps = rollout_episode(
            env,
            actor,
            device,
            seed=base_seed + 10_000 + ep * 1000,
            target_frames=target_frames,
            deterministic=deterministic,
        )
        returns.extend(ep_returns)
        out_path = os.path.join(
            video_root,
            f"eval_{os.path.basename(run_dir)}_step_{step}_ep{ep}.mp4",
        )
        write_video(out_path, frames, fps=fps)
        joined = ", ".join(f"{r:.2f}" for r in ep_returns)
        print(f"    video {ep}: steps={ep_steps}, episode returns=[{joined}]")

    env.close()
    print(f"[√] mean episode return: {float(np.mean(returns)):.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--run_dir", type=str, default=None, help="Evaluate a single run directory")
    parser.add_argument("--env", type=str, default=None, choices=["ant", "hopper", "walker", "humanoid"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prefix", type=str, default=None, help="Only runs whose folder name starts with this")
    parser.add_argument("--one_per_env", action="store_true", help="Keep the lowest seed for each env")
    parser.add_argument("--checkpoint_step", type=int, default=None, help="Defaults to latest checkpoint")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of videos to write per run")
    parser.add_argument("--duration", type=float, default=10.0, help="Video length in seconds")
    parser.add_argument("--fps", type=float, default=None, help="Defaults to env render_fps / 1/dt")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of mean")
    parser.add_argument("--output_dir", type=str, default=None, help="Override per-run videos/ folder")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    cli = parse_args()
    runs_dir = cli.runs_dir
    if not os.path.isabs(runs_dir):
        runs_dir = os.path.join(os.path.dirname(__file__), runs_dir)

    device = torch.device("cpu" if cli.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[√] Device: {device}")

    run_dirs = select_runs(
        runs_dir,
        run_dir=cli.run_dir,
        env_name=cli.env,
        seed=cli.seed,
        prefix=cli.prefix,
        one_per_env=cli.one_per_env,
    )
    if not run_dirs:
        raise SystemExit(f"No matching runs under {runs_dir}")

    print(f"[√] Evaluating {len(run_dirs)} run(s)")
    for run_dir in run_dirs:
        evaluate_run(
            run_dir,
            checkpoint_step=cli.checkpoint_step,
            num_episodes=cli.num_episodes,
            duration=cli.duration,
            fps=cli.fps,
            deterministic=not cli.stochastic,
            device=device,
            output_dir=cli.output_dir,
        )


if __name__ == "__main__":
    main()
