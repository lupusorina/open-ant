import os
import glob

import matplotlib.pyplot as plt
import pandas as pd


RUNS_DIR = "/home/serenaliu/caltech_linc_home/open-ant/agents/sac/runs/sac_pernetwork_neworder/"



# Only seed 1 currently exists, but missing seeds will be skipped.
SEEDS_TO_PLOT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

plt.figure(figsize=(14, 6))

num_plotted = 0

for seed in SEEDS_TO_PLOT:
    sim1_dirs = sorted(
        glob.glob(
            os.path.join(
                RUNS_DIR,
                f"sac_sim1_*_seed_{seed}",
            )
        ),
        reverse=True,
    )

    sim2_dirs = sorted(
        glob.glob(
            os.path.join(
                RUNS_DIR,
                f"continuous_sac_*_seed_{seed}",
            )
        ),
        reverse=True,
    )

    # Sim1 is required.
    if not sim1_dirs:
        print(f"Seed {seed}: Sim1 folder not found, skipping.")
        continue

    sim1_csv = os.path.join(
        sim1_dirs[0],
        "SimEmbodiedAnt_average_rewards.csv",
    )

    if not os.path.exists(sim1_csv):
        print(f"Seed {seed}: Sim1 reward CSV not found, skipping.")
        continue

    df1 = pd.read_csv(sim1_csv)

    if "step" not in df1.columns or "reward" not in df1.columns:
        raise ValueError(
            f"Unexpected columns in {sim1_csv}: "
            f"{df1.columns.tolist()}"
        )

    df_all = df1.copy()
    last_step_1 = df1["step"].max()
    has_sim2 = False

    # Append Sim2 only when it exists.
    if sim2_dirs:
        sim2_csv = os.path.join(
            sim2_dirs[0],
            "SimEmbodiedAnt_average_rewards.csv",
        )

        if os.path.exists(sim2_csv):
            df2 = pd.read_csv(sim2_csv)

            if "step" not in df2.columns or "reward" not in df2.columns:
                raise ValueError(
                    f"Unexpected columns in {sim2_csv}: "
                    f"{df2.columns.tolist()}"
                )

            df2 = df2.copy()
            df2["step"] = df2["step"] + last_step_1

            df_all = pd.concat(
                [df1, df2],
                ignore_index=True,
            )

            has_sim2 = True
        else:
            print(
                f"Seed {seed}: Sim2 folder exists, "
                "but reward CSV is missing. Plotting Sim1 only."
            )
    else:
        print(f"Seed {seed}: no Sim2 run; plotting Sim1 only.")

    line, = plt.plot(
        df_all["step"],
        df_all["reward"],
        linewidth=1,
        label=f"seed {seed}",
    )

    if has_sim2:
        plt.axvline(
            last_step_1,
            linestyle="--",
            linewidth=0.8,
            color=line.get_color(),
            label="_nolegend_",
        )

    num_plotted += 1


if num_plotted == 0:
    raise RuntimeError(
        f"No reward curves were found under:\n{RUNS_DIR}"
    )

plt.xlabel("Total Step")
plt.ylabel("Reward")
# plt.xlim(0, 120000)
plt.ylim(-0.025, 0.2)
plt.yticks(
    [
        0,
        0.025,
        0.05,
        0.075,
        0.10,
        0.125,
        0.15,
        0.175,
        0.20,
    ]
)

plt.title("Reward vs Step: All Available Seeds Overlaid")
plt.legend(fontsize=8)
plt.grid(True)

output_path = os.path.join(
    RUNS_DIR,
    "rewardallseed.png",
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(f"Saved plot to:\n{output_path}")