import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

RUNS1_DIR = "/data2/sorina_data/runs_July_24"
RUNS2_DIR = "/data2/serenaliu_data/continual_mpo_hopper"
NAME = "hopper"
OUTPUTNAME = "hopper"
ENV = "Hopper-v5"
SEEDS_TO_PLOT = [0, 1,2,3,4,5,6,7,8,9,10]
#SEEDS_TO_PLOT = [0]

AVG_REWARD_OUTPUT = (
    "/home/serenaliu/caltech_linc_home/open-ant/agents/mpo/"
    f"{OUTPUTNAME}_average_reward_per_second.png"
)

EPISODE_RETURN_OUTPUT = (
    "/home/serenaliu/caltech_linc_home/open-ant/agents/mpo/"
    f"{OUTPUTNAME}_mean_episode_return.png"
)


def load_continual_data(seed):
    """Load Sim1 and Sim2 CSVs and join their step axes."""

    sim1_dirs = sorted(
        glob.glob(f"{RUNS1_DIR}/mpo_{NAME}_*_seed_{seed}"),
        reverse=True,
    )

    sim2_dirs = sorted(
        glob.glob(f"{RUNS2_DIR}/continual_mpo_{NAME}_*_seed_{seed}"),
        reverse=True,
    )

    if not sim1_dirs or not sim2_dirs:
        print(f"Seed {seed}: run folders not found, skipping.")
        return None, None

    sim1_csv = os.path.join(
        sim1_dirs[0],
        f"{ENV}_average_rewards.csv",
    )

    sim2_csv = os.path.join(
        sim2_dirs[0],
        f"{ENV}_average_rewards.csv",
    )

    if not os.path.exists(sim1_csv) or not os.path.exists(sim2_csv):
        print(f"Seed {seed}: CSV not found, skipping.")
        return None, None

    df1 = pd.read_csv(sim1_csv)
    df2 = pd.read_csv(sim2_csv)

    last_step_1 = df1["step"].max()

    # RewardTracker begins Sim2's step counter from zero,
    # so shift Sim2 to follow Sim1 on the x-axis.
    df2 = df2.copy()
    df2["step"] += last_step_1

    df_all = pd.concat([df1, df2], ignore_index=True)

    return df_all, last_step_1


# ============================================================
# Plot 1: rolling average reward per second
# ============================================================

plt.figure(figsize=(14, 6))

for seed in SEEDS_TO_PLOT:
    df_all, continual_start = load_continual_data(seed)

    if df_all is None:
        continue

    line, = plt.plot(
        df_all["step"],
        df_all["reward"],
        linewidth=1,
        label=f"seed {seed}",
    )

    plt.axvline(
        continual_start,
        linestyle="--",
        linewidth=0.8,
        color=line.get_color(),
        label="_nolegend_",
    )

plt.xlabel("Total environment steps")
plt.ylabel("Average reward per second")
plt.title(f"{OUTPUTNAME}: Average Reward per Second")
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig(AVG_REWARD_OUTPUT, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved average-reward plot to: {AVG_REWARD_OUTPUT}")


# ============================================================
# Plot 2: rolling mean episodic return
# ============================================================

plt.figure(figsize=(14, 6))

for seed in SEEDS_TO_PLOT:
    df_all, continual_start = load_continual_data(seed)

    if df_all is None:
        continue

    # mean_return is None until at least one episode has completed,
    # so remove those initial missing values.
    return_df = df_all.dropna(subset=["mean_return"])

    line, = plt.plot(
        return_df["step"],
        return_df["mean_return"],
        linewidth=1,
        label=f"seed {seed}",
    )

    plt.axvline(
        continual_start,
        linestyle="--",
        linewidth=0.8,
        color=line.get_color(),
        label="_nolegend_",
    )

plt.xlabel("Total environment steps")
plt.ylabel("Mean episodic return")
plt.title(f"{OUTPUTNAME}: Mean Episodic Return over Last 100 Episodes")
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig(EPISODE_RETURN_OUTPUT, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved episodic-return plot to: {EPISODE_RETURN_OUTPUT}")