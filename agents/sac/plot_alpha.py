import os
import glob

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

RUNS_DIR = (
    "/home/seliu/open-ant/agents/sac/"
    "runs/idbd_newweight_nolayernorm"
)

SEEDS_TO_PLOT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

OUTPUT_PATH = os.path.join(
    RUNS_DIR,
    "entropy_alpha_all_seeds.png",
)


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(14, 6))

num_plotted = 0
transition_steps = []


for seed in SEEDS_TO_PLOT:

    # --------------------------------------------------------
    # Find latest Sim1 / Sim2 run for this seed
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Sim1
    # --------------------------------------------------------

    if not sim1_dirs:
        print(
            f"Seed {seed}: Sim1 folder not found, skipping."
        )
        continue

    sim1_csv = os.path.join(
        sim1_dirs[0],
        "info_sac_logs.csv",
    )

    if not os.path.exists(sim1_csv):
        print(
            f"Seed {seed}: Sim1 info_sac_logs.csv "
            "not found, skipping."
        )
        continue

    df1 = pd.read_csv(sim1_csv)

    required_columns = {
        "global_step",
        "alpha",
    }

    if not required_columns.issubset(df1.columns):
        raise ValueError(
            f"Unexpected columns in {sim1_csv}:\n"
            f"{df1.columns.tolist()}"
        )

    # Keep only the values we need.
    df1 = df1[
        ["global_step", "alpha"]
    ].copy()

    # Sort just in case.
    df1 = df1.sort_values(
        "global_step"
    ).reset_index(drop=True)

    last_step_1 = df1["global_step"].max()

    df_all = df1.copy()

    has_sim2 = False


    # --------------------------------------------------------
    # Sim2
    # --------------------------------------------------------

    if sim2_dirs:

        sim2_csv = os.path.join(
            sim2_dirs[0],
            "info_sac_logs.csv",
        )

        if os.path.exists(sim2_csv):

            df2 = pd.read_csv(sim2_csv)

            if not required_columns.issubset(df2.columns):
                raise ValueError(
                    f"Unexpected columns in {sim2_csv}:\n"
                    f"{df2.columns.tolist()}"
                )

            df2 = df2[
                ["global_step", "alpha"]
            ].copy()

            df2 = df2.sort_values(
                "global_step"
            ).reset_index(drop=True)


            # ==================================================
            # IMPORTANT:
            #
            # Normally Sim2 already resumes global_step from
            # Sim1, e.g.
            #
            # Sim1: 2001 ... 40000
            # Sim2: 40001 ... 120000
            #
            # In that case, DO NOT shift anything.
            #
            # But if Sim2 somehow starts again near zero,
            # shift it so that it follows Sim1.
            # ==================================================

            first_step_2 = df2["global_step"].min()

            if first_step_2 < last_step_1:

                print(
                    f"Seed {seed}: Sim2 global_step appears "
                    "to restart; shifting Sim2 steps."
                )

                df2["global_step"] = (
                    df2["global_step"]
                    - first_step_2
                    + last_step_1
                )

            else:

                print(
                    f"Seed {seed}: Sim2 already continues "
                    f"global_step "
                    f"({first_step_2} after {last_step_1})."
                )


            # Concatenate Sim1 + Sim2.
            df_all = pd.concat(
                [df1, df2],
                ignore_index=True,
            )

            # Ensure final order.
            df_all = df_all.sort_values(
                "global_step"
            ).reset_index(drop=True)

            has_sim2 = True

        else:
            print(
                f"Seed {seed}: Sim2 folder exists, "
                "but info_sac_logs.csv is missing. "
                "Plotting Sim1 only."
            )

    else:
        print(
            f"Seed {seed}: no Sim2 run; "
            "plotting Sim1 only."
        )


    # --------------------------------------------------------
    # Plot this seed
    # --------------------------------------------------------

    line, = plt.plot(
        df_all["global_step"],
        df_all["alpha"],
        linewidth=1.5,
        label=f"Seed {seed}",
    )


    # Draw Sim1 -> Sim2 transition in same seed color.
    if has_sim2:

        plt.axvline(
            last_step_1,
            linestyle="--",
            linewidth=0.8,
            color=line.get_color(),
            alpha=0.5,
            label="_nolegend_",
        )

        transition_steps.append(
            last_step_1
        )


    num_plotted += 1


# ============================================================
# FINAL FORMATTING
# ============================================================

if num_plotted == 0:
    raise RuntimeError(
        f"No entropy-alpha curves were found under:\n"
        f"{RUNS_DIR}"
    )


plt.xlabel(
    "Global Step",
    fontsize=14,
)

plt.ylabel(
    "Entropy Temperature α",
    fontsize=14,
)

plt.title(
    "SAC Entropy Temperature: Sim1 → Sim2",
    fontsize=16,
)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.legend(
    fontsize=9,
)

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    f"\nSaved plot to:\n{OUTPUT_PATH}"
)