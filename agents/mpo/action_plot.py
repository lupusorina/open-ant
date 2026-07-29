from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

PATH = "/home/serenaliu/caltech_linc_home/open-ant/agents/mpo/runs/long_ensemble_spi1536_gamma92"

# "sim1" or "sim2"
SIM = "sim1"

# Plot only actions executed during this step interval.
# Use None for no lower/upper limit.
STEP_START = 25_000
STEP_END = 30_000

X_MIN = -8.0
X_MAX = 8.0
X_TICK_STEP = 1

Y_MIN = 0.0
Y_MAX = 0.1
Y_TICK_STEP = 0.05
NUM_BINS = 80

OUTPUT_PATH = (
    f"{PATH}/{SIM}_actions_"
    f"{STEP_START}_to_{STEP_END}.png"
)


# ============================================================
# FIND RUNS
# ============================================================

parent = Path(PATH)

if SIM == "sim1":
    run_dirs = [
        path
        for path in parent.glob("mpo_*_seed_*")
        if path.is_dir()
        and (path / "raw_actions.csv").exists()
    ]

elif SIM == "sim2":
    run_dirs = [
        path
        for pattern in (
            "continual_*_seed_*",
            "continuous_*_seed_*",
        )
        for path in parent.glob(pattern)
        if path.is_dir()
        and (path / "raw_actions.csv").exists()
    ]

else:
    raise ValueError("SIM must be 'sim1' or 'sim2'.")

if not run_dirs:
    raise FileNotFoundError(
        f"No matching runs with raw_actions.csv found in {PATH}"
    )


# ============================================================
# LOAD AND FILTER ACTIONS
# ============================================================

action_columns = [
    f"action_{i}"
    for i in range(8)
]

seed_actions = []

for run_dir in run_dirs:
    csv_path = run_dir / "raw_actions.csv"

    # Normally this is "step", but this also handles "tep".
    columns = pd.read_csv(csv_path, nrows=0).columns
    step_column = "step" if "step" in columns else "tep"

    df = pd.read_csv(
        csv_path,
        usecols=[step_column] + action_columns,
    )

    if STEP_START is not None:
        df = df[df[step_column] >= STEP_START]

    if STEP_END is not None:
        df = df[df[step_column] <= STEP_END]

    if df.empty:
        print(
            f"Skipping {run_dir.name}: "
            "no rows in requested step range"
        )
        continue

    seed_actions.append(
        df[action_columns].to_numpy(dtype=float)
    )

    print(
        f"{run_dir.name}: using {len(df)} rows, "
        f"steps {df[step_column].min()}–{df[step_column].max()}"
    )

if not seed_actions:
    raise ValueError(
        "No runs contained actions in the requested step range."
    )


# ============================================================
# PLOT DISTRIBUTIONS
# ============================================================

bin_edges = np.linspace(
    X_MIN,
    X_MAX,
    NUM_BINS + 1,
)

bin_centers = (
    bin_edges[:-1] + bin_edges[1:]
) / 2

bin_width = bin_edges[1] - bin_edges[0]

fig, axes = plt.subplots(
    4,
    2,
    figsize=(14, 16),
    sharey=True
)

axes = axes.flatten()

for action_dim in range(8):
    seed_histograms = []
    outside_fractions = []

    for actions in seed_actions:
        values = actions[:, action_dim]
        values = values[np.isfinite(values)]

        # Calculated using all selected values, even those beyond [-2, 2].
        outside_fractions.append(
            np.mean(
                (values < -1.0)
                | (values > 1.0)
            )
        )

        counts, _ = np.histogram(
            values,
            bins=bin_edges,
        )

        if counts.sum() > 0:
            # Normalize each seed separately, then average seeds equally.
            seed_histograms.append(
                counts / counts.sum()
            )

    mean_histogram = np.mean(
        seed_histograms,
        axis=0,
    )

    mean_outside_fraction = np.mean(
        outside_fractions
    )

    ax = axes[action_dim]

    ax.bar(
        bin_centers,
        mean_histogram,
        width=bin_width,
    )

    ax.axvline(-1.0, linestyle="--", linewidth=1.5)
    ax.axvline(1.0, linestyle="--", linewidth=1.5)

    ax.set_xlim(X_MIN, X_MAX)

    ax.set_xticks(
        np.arange(
            X_MIN,
            X_MAX + X_TICK_STEP,
            X_TICK_STEP,
        )
    )
    ax.set_ylim(Y_MIN, Y_MAX)

    ax.set_yticks(
        np.arange(
            Y_MIN,
            Y_MAX + Y_TICK_STEP / 2,
            Y_TICK_STEP,
        )
    )

    ax.set_title(
        f"Action {action_dim}\n"
        f"Outside [-1, 1]: "
        f"{100 * mean_outside_fraction:.2f}%"
    )

    ax.set_xlabel("Raw policy action")
    ax.set_ylabel("Probability")
    ax.grid(True, alpha=0.3)


fig.suptitle(
    f"{SIM.upper()} Raw Action Distributions\n"
    f"Steps {STEP_START} to {STEP_END}, "
    f"{len(seed_actions)} seed(s)",
    fontsize=16,
)

fig.tight_layout(rect=(0, 0, 1, 0.95))

fig.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(f"\nSaved plot to:\n{OUTPUT_PATH}")