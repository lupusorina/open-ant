"""Compare SAC and MPO: one row per environment.

Columns are the environment, average reward per second, and mean episodic return.
"""

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

AGENTS = Path(__file__).resolve().parent
REPO = AGENTS.parent
sys.path.insert(0, str(REPO / "sim"))

ENV_ORDER = ("MiniPiWalk-v0", "Pusher-v5", "Reacher-v5", "Swimmer-v5")
ENV_NAMES = {
    "MiniPiWalk-v0": "Mini Pi",
    "Pusher-v5": "Pusher",
    "Reacher-v5": "Reacher",
    "Swimmer-v5": "Swimmer",
}
# Free-camera overrides. None keeps Gymnasium's default (the track camera, when
# the model defines one).
CAMERAS = {
    "Pusher-v5": {
        "distance": 3.15,
        "azimuth": 128.0,
        "elevation": -36.0,
        "lookat": (0.22, 0.42, -0.02),
    },
    "Reacher-v5": {
        "distance": 0.95,
        "azimuth": 90.0,
        "elevation": -58.0,
        "lookat": (0.0, 0.0, 0.0),
    },
}

SAC_COLOR = "#2F6FE0"
MPO_COLOR = "#E36A2C"
INK = "#1C1C1C"
MUTED = "#8A8478"
GRID = "#E6E1D8"
ZERO = "#D5D0C6"
PANEL = "#FBF9F6"

OUTPUT = AGENTS / "runs_comparison.png"


def discover():
    """Latest reward log for each (environment, algorithm) pair."""
    found = {}
    candidates = [("SAC", path.parent, path) for path in (AGENTS / "sac" / "runs").glob("*/args.json")]
    candidates += [
        ("MPO", path.parents[1], path)
        for path in (AGENTS / "mpo" / "runs").glob("*/*/weights_and_args/args.json")
    ]
    for algo, run_dir, args_path in candidates:
        with open(args_path) as handle:
            env_id = json.load(handle).get("env_id")
        if not env_id:
            continue
        csv_path = run_dir / f"{env_id}_average_rewards.csv"
        if not csv_path.is_file() or csv_path.stat().st_size == 0:
            found.setdefault((env_id, algo), None)
            continue
        previous = found.get((env_id, algo))
        if previous is None or csv_path.stat().st_mtime >= Path(previous).stat().st_mtime:
            found[(env_id, algo)] = csv_path
    envs = [env_id for env_id in ENV_ORDER if any(key[0] == env_id for key in found)]
    envs += sorted({key[0] for key in found if key[0] not in envs})
    return envs, found


def load_curve(path):
    steps, rewards, returns = [], [], []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "step" not in reader.fieldnames:
            return None
        for row in reader:
            if not row.get("step"):
                continue
            steps.append(float(row["step"]))
            rewards.append(float(row["reward"]))
            value = (row.get("mean_return") or "").strip()
            returns.append(np.nan if value in ("", "None", "nan") else float(value))
    if not steps:
        return None
    return np.asarray(steps), np.asarray(rewards), np.asarray(returns)


def smooth(values, fraction=0.008):
    window = int(len(values) * fraction)
    if window < 5:
        return values
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def decimate(steps, values, max_points=900):
    if len(steps) <= max_points:
        return steps, values
    index = np.unique(np.linspace(0, len(steps) - 1, max_points).astype(int))
    return steps[index], values[index]


def return_series(steps, values):
    finite = np.isfinite(values)
    steps, values = steps[finite], values[finite]
    if len(steps) == 0:
        return steps, values
    changed = np.ones(len(values), dtype=bool)
    changed[1:] = values[1:] != values[:-1]
    changed[-1] = True
    return steps[changed], values[changed]


def step_formatter(value, _pos):
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1000:
        return f"{value / 1000:.0f}k"
    return f"{value:.0f}"


def render_env(env_id):
    import gymnasium as gym
    import mini_pi_walk_env  # noqa: F401  registers MiniPiWalk-v0

    env = gym.make(env_id, render_mode="rgb_array", width=640, height=480)
    try:
        env.reset(seed=0)
        camera = CAMERAS.get(env_id)
        if camera is not None:
            renderer = env.unwrapped.mujoco_renderer
            renderer._get_viewer("rgb_array")
            cam = renderer.viewer.cam
            cam.distance = camera["distance"]
            cam.azimuth = camera["azimuth"]
            cam.elevation = camera["elevation"]
            cam.lookat[:] = camera["lookat"]
        frame = env.render()
    finally:
        env.close()
    return trim_void(frame)


def trim_void(frame):
    """Crop a pure-black surround. Gray robots and checkered floors stay intact."""
    ink = frame.max(axis=2) > 8
    if ink.mean() > 0.9 or int(ink.sum()) < 100:
        return frame
    rows = np.flatnonzero(ink.any(axis=1))
    cols = np.flatnonzero(ink.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    # Already filling the frame.
    if y0 < 4 and x0 < 4 and y1 > frame.shape[0] - 4 and x1 > frame.shape[1] - 4:
        return frame
    pad_y = int(0.07 * (y1 - y0))
    pad_x = int(0.07 * (x1 - x0))
    y0 = max(0, y0 - pad_y)
    x0 = max(0, x0 - pad_x)
    y1 = min(frame.shape[0], y1 + pad_y)
    x1 = min(frame.shape[1], x1 + pad_x)
    return np.ascontiguousarray(frame[y0:y1, x0:x1])


def style_curve_axis(ax, ylabel, show_xlabel):
    ax.set_facecolor(PANEL)
    ax.set_ylabel(ylabel, color=INK)
    if show_xlabel:
        ax.set_xlabel("Steps", color=INK)
    ax.axhline(0.0, color=ZERO, linewidth=0.8, zorder=0)
    ax.grid(True, axis="both", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#5E5A54", length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_color("#E3DED4")
        spine.set_linewidth(0.8)
    ax.xaxis.set_major_formatter(FuncFormatter(step_formatter))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.margins(x=0.02, y=0.12)


def blank_axis(ax, ylabel, show_xlabel):
    ax.set_facecolor(PANEL)
    ax.set_ylabel(ylabel, color=INK)
    if show_xlabel:
        ax.set_xlabel("Steps", color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#E3DED4")
        spine.set_linewidth(0.8)
    ax.text(
        0.5,
        0.5,
        "No logs yet",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=11,
    )


def plot(envs, logs):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelsize": 10,
            "axes.titlesize": 13,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
        }
    )
    n = len(envs)
    fig, axes = plt.subplots(
        n,
        3,
        figsize=(12.8, 2.72 * n + 0.85),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.02, 1.62, 1.62]},
    )
    axes = np.atleast_2d(axes)

    for row, env_id in enumerate(envs):
        name = ENV_NAMES.get(env_id, env_id)
        image_ax, reward_ax, return_ax = axes[row]
        show_xlabel = row == n - 1

        frame = render_env(env_id)
        image_ax.imshow(frame, interpolation="bilinear")
        image_ax.set_title(name, pad=8, fontweight="medium", loc="left")
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        for spine in image_ax.spines.values():
            spine.set_visible(False)

        if row == 0:
            reward_ax.set_title("Average reward", pad=8, fontweight="medium", loc="left")
            return_ax.set_title("Episodic return", pad=8, fontweight="medium", loc="left")

        xmax = 0.0
        plotted = False
        for algo, color in (("SAC", SAC_COLOR), ("MPO", MPO_COLOR)):
            path = logs.get((env_id, algo))
            if path is None:
                continue
            loaded = load_curve(path)
            if loaded is None:
                continue
            steps, rewards, returns = loaded
            xmax = max(xmax, float(steps[-1]))
            reward_steps, reward_values = decimate(steps, smooth(rewards))
            reward_ax.plot(
                reward_steps,
                reward_values,
                color=color,
                linewidth=1.9,
                solid_capstyle="round",
                label=algo,
                zorder=3,
            )
            return_steps, return_values = return_series(steps, returns)
            if len(return_steps):
                return_ax.plot(
                    return_steps,
                    return_values,
                    color=color,
                    linewidth=1.9,
                    drawstyle="steps-post",
                    solid_capstyle="round",
                    zorder=3,
                )
            plotted = True

        if plotted:
            style_curve_axis(reward_ax, "Average reward / s", show_xlabel)
            style_curve_axis(return_ax, "Mean return", show_xlabel)
            reward_ax.set_xlim(0, xmax)
            return_ax.set_xlim(0, xmax)
        else:
            blank_axis(reward_ax, "Average reward / s", show_xlabel)
            blank_axis(return_ax, "Mean return", show_xlabel)

    handles = [
        Line2D([0], [0], color=SAC_COLOR, linewidth=2.2, label="SAC"),
        Line2D([0], [0], color=MPO_COLOR, linewidth=2.2, label="MPO"),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 1.0),
        frameon=False,
        ncol=2,
        borderaxespad=0.2,
        handlelength=1.8,
        columnspacing=1.2,
    )
    fig.suptitle("SAC and MPO", fontsize=16, fontweight="medium", x=0.012, ha="left")
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    print(f"Saved to {OUTPUT}")


def main():
    envs, logs = discover()
    if not envs:
        raise SystemExit(f"No runs with args.json under {AGENTS}")
    plot(envs, logs)


if __name__ == "__main__":
    main()
