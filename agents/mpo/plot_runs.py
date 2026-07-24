"""Plot MPO seed runs as a paper-style 2x4 learning-curve figure.

Row 0: average reward. Row 1: episodic return (mean_return).
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ENV_ORDER = ("ant", "hopper", "walker", "humanoid")
ENV_TITLES = {
    "ant": "Ant",
    "hopper": "Hopper",
    "walker": "Walker2d",
    "humanoid": "Humanoid",
}
METRIC_SPECS = (
    ("reward", "Average reward"),
    ("mean_return", "Episodic return"),
)


def find_run_dirs(runs_dir: str, prefix: str | None = None) -> list[str]:
    pattern = os.path.join(runs_dir, "*_seed_*")
    run_dirs = [
        d for d in glob.glob(pattern)
        if os.path.isdir(d) and re.search(r"_seed_\d+$", os.path.basename(d))
    ]
    if prefix:
        run_dirs = [d for d in run_dirs if os.path.basename(d).startswith(prefix)]
    return sorted(run_dirs, key=lambda d: int(re.search(r"_seed_(\d+)$", d).group(1)))


def env_from_dir(run_dir: str) -> str | None:
    """Extract env short name from e.g. mpo_ant_20260723-164017_seed_0."""
    name = os.path.basename(run_dir)
    m = re.match(r"mpo_([a-zA-Z0-9]+)_", name)
    return m.group(1).lower() if m else None


def load_reward_csv(run_dir: str) -> pd.DataFrame | None:
    """Load reward CSV keyed by environment steps."""
    matches = sorted(glob.glob(os.path.join(run_dir, "*_average_rewards.csv")))
    if not matches:
        return None
    df = pd.read_csv(matches[0])
    if df.empty or "step" not in df.columns or "reward" not in df.columns:
        return None
    cols = ["step", "reward"]
    if "mean_return" in df.columns:
        cols.append("mean_return")
    df = df[cols].dropna(subset=["step", "reward"]).sort_values("step").copy()
    return df


def seed_from_dir(run_dir: str) -> int:
    return int(re.search(r"_seed_(\d+)$", os.path.basename(run_dir)).group(1))


def interpolate_to_grid(
    dfs: list[pd.DataFrame], steps: np.ndarray, value_col: str,
) -> np.ndarray:
    """Return (n_runs, n_points) values interpolated onto a shared step grid."""
    curves = []
    for df in dfs:
        curves.append(
            np.interp(steps, df["step"].to_numpy(), df[value_col].to_numpy())
        )
    return np.stack(curves, axis=0)


def style_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="out", length=3.5, width=0.8, labelsize=9)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, color="0.5")
    ax.set_axisbelow(True)


def plot_env_on_ax(
    ax,
    env: str,
    loaded: list[tuple[int, pd.DataFrame]],
    value_col: str,
    ylabel: str,
    show_title: bool,
    show_xlabel: bool,
    show_ylabel: bool,
) -> tuple:
    """Draw one env panel. Returns (mean_line, std_patch) for a shared legend."""
    title = ENV_TITLES.get(env, env.capitalize())
    color = "#2c7bb6"
    mean_line = std_patch = None

    usable = [(s, df) for s, df in loaded if value_col in df.columns]
    usable = [
        (s, df.dropna(subset=[value_col]))
        for s, df in usable
    ]
    usable = [(s, df) for s, df in usable if len(df) > 0]

    if not usable:
        ax.text(
            0.5, 0.5, "no data yet",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color="0.45",
        )
        if show_title:
            ax.set_title(title, fontsize=12, pad=8)
        if show_xlabel:
            ax.set_xlabel("Steps", fontsize=10)
        if show_ylabel:
            ax.set_ylabel(ylabel, fontsize=10)
        style_ax(ax)
        return mean_line, std_patch

    for _, df in usable:
        ax.plot(
            df["step"],
            df[value_col],
            color=color,
            alpha=0.18,
            linewidth=0.9,
            zorder=1,
        )

    if len(usable) >= 2:
        min_max_t = min(df["step"].max() for _, df in usable)
        max_min_t = max(df["step"].min() for _, df in usable)
        if min_max_t > max_min_t:
            grid = np.linspace(max_min_t, min_max_t, num=500)
            stack = interpolate_to_grid([df for _, df in usable], grid, value_col)
            mean = stack.mean(axis=0)
            se = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0])
            std_patch = ax.fill_between(
                grid, mean - 3 * se, mean + 3 * se,
                color=color, alpha=0.22, linewidth=0, zorder=2, label="±3 SE",
            )
            mean_line, = ax.plot(
                grid, mean, color=color, linewidth=2.0, zorder=3, label="mean",
            )
    else:
        _, df = usable[0]
        mean_line, = ax.plot(
            df["step"], df[value_col],
            color=color, linewidth=2.0, zorder=3, label="mean",
        )

    if show_title:
        ax.set_title(f"{title} ({len(usable)} seeds)", fontsize=12, pad=8)
    if show_xlabel:
        ax.set_xlabel("Steps", fontsize=10)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    style_ax(ax)
    return mean_line, std_patch


def load_env_runs(
    env_dirs: list[str],
    downsample: int,
) -> tuple[list[tuple[int, pd.DataFrame]], list[str]]:
    loaded: list[tuple[int, pd.DataFrame]] = []
    missing = []
    for run_dir in env_dirs:
        seed = seed_from_dir(run_dir)
        df = load_reward_csv(run_dir)
        if df is None or len(df) == 0:
            missing.append(os.path.basename(run_dir))
            continue
        if downsample > 1:
            df = df.iloc[::downsample].reset_index(drop=True)
        loaded.append((seed, df))
    return loaded, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs_dir",
        default=os.path.join(os.path.dirname(__file__), "runs"),
        help="Directory containing seed run folders",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional run-name prefix filter (e.g. mpo_ant_20260723-161131)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output figure path (default: <runs_dir>/mpo_runs.png)",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=50,
        help="Keep every Nth reward sample for lighter plots (default: 50)",
    )
    args = parser.parse_args()

    run_dirs = find_run_dirs(args.runs_dir, prefix=args.prefix)
    if not run_dirs:
        raise SystemExit(f"No seed run directories found in {args.runs_dir}")

    by_env: dict[str, list[str]] = defaultdict(list)
    for run_dir in run_dirs:
        env = env_from_dir(run_dir)
        if env is None:
            print(f"Skipping unrecognized run dir: {os.path.basename(run_dir)}")
            continue
        by_env[env].append(run_dir)

    if not by_env:
        raise SystemExit("No recognizable mpo_<env>_* run directories found")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 4, figsize=(14, 5.8), constrained_layout=True)
    any_data = False
    legend_handles = []

    for col, env in enumerate(ENV_ORDER):
        env_dirs = by_env.get(env, [])
        loaded, missing = load_env_runs(env_dirs, args.downsample)
        print(
            f"[{env}] {len(loaded)}/{len(env_dirs)} runs"
            + (f" (missing data: {missing})" if missing else "")
        )
        if loaded:
            any_data = True

        for row, (value_col, ylabel) in enumerate(METRIC_SPECS):
            mean_line, std_patch = plot_env_on_ax(
                axes[row, col],
                env,
                loaded,
                value_col=value_col,
                ylabel=ylabel,
                show_title=(row == 0),
                show_xlabel=(row == 1),
                show_ylabel=(col == 0),
            )
            if not legend_handles and mean_line is not None:
                legend_handles = [h for h in (mean_line, std_patch) if h is not None]

    if not any_data:
        plt.close(fig)
        raise SystemExit("No plots created; no reward CSVs found yet.")

    if legend_handles:
        fig.legend(
            legend_handles,
            [h.get_label() for h in legend_handles],
            loc="upper center",
            ncol=len(legend_handles),
            frameon=False,
            fontsize=9,
            bbox_to_anchor=(0.5, 1.05),
        )

    out_path = args.output or os.path.join(args.runs_dir, "mpo_runs.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
