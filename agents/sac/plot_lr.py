import os
import re

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

RUN_DIR = (
    "/home/serenaliu/caltech_linc_home/open-ant/agents/sac/"
    "runs/no_adam/idbd_largerweight_layernorm/"
    "sac_sim1_20260831-145506_seed_3"
)


# ============================================================
# HELPERS
# ============================================================

def get_step(path):
    """Extract step number from lr_step_XXXXX.npz."""
    match = re.search(
        r"lr_step_(\d+)\.npz$",
        os.path.basename(path),
    )

    if match is None:
        raise ValueError(
            f"Could not parse step from: {path}"
        )

    return int(match.group(1))


def main():

    lr_dir = os.path.join(
        RUN_DIR,
        "learning_rates",
    )

    if not os.path.isdir(lr_dir):
        raise FileNotFoundError(
            f"Could not find learning_rates directory:\n{lr_dir}"
        )

    lr_files = [
        os.path.join(lr_dir, filename)
        for filename in os.listdir(lr_dir)
        if filename.startswith("lr_step_")
        and filename.endswith(".npz")
    ]

    # Sort numerically by training step.
    lr_files.sort(key=get_step)

    if len(lr_files) == 0:
        raise RuntimeError(
            f"No lr_step_*.npz files found in {lr_dir}"
        )

    steps = []
    mean_lrs = []
    median_lrs = []
    max_lrs = []

    for path in lr_files:

        step = get_step(path)

        with np.load(path) as lr_data:

            all_lrs = np.concatenate([
                np.asarray(lr_data[key]).reshape(-1)
                for key in lr_data.files
            ])

        mean_lr = np.mean(all_lrs)
        median_lr = np.median(all_lrs)
        max_lr = np.max(all_lrs)

        steps.append(step)
        mean_lrs.append(mean_lr)
        median_lrs.append(median_lr)
        max_lrs.append(max_lr)

        print(
            f"Step {step:>8}: "
            f"mean={mean_lr:.6g}, "
            f"median={median_lr:.6g}, "
            f"max={max_lr:.6g}"
        )

    steps = np.asarray(steps)
    mean_lrs = np.asarray(mean_lrs)
    median_lrs = np.asarray(median_lrs)
    max_lrs = np.asarray(max_lrs)

    # ============================================================
    # PLOT
    # ============================================================

    plt.figure(figsize=(10, 6))

    plt.plot(
        steps,
        mean_lrs,
        linewidth=2,
        label="Mean LR",
    )

    plt.plot(
        steps,
        median_lrs,
        linewidth=2,
        label="Median LR",
    )

    plt.plot(
        steps,
        max_lrs,
        linewidth=2,
        label="Max LR",
    )

    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.title("Per-Weight Learning Rate Progression")

    plt.yscale("log")

    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        RUN_DIR,
        "learning_rate_progression.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nSaved plot to:\n{output_path}"
    )


if __name__ == "__main__":
    main()
