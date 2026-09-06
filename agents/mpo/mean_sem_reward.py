import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUNS_DIR = "/data2/serenaliu_data/2dmpo_evar0008"

SEEDS_TO_PLOT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                  21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

STEP_GRID = np.linspace(0, 150000, 500)

curves = []

for seed in SEEDS_TO_PLOT:
    sim1_dirs = sorted(glob.glob(f"{RUNS_DIR}/mpo_*_seed_{seed}"), reverse=True)
    sim2_dirs = sorted(glob.glob(f"{RUNS_DIR}/continuous_mpo_*_seed_{seed}"), reverse=True)

    if not sim1_dirs or not sim2_dirs:
        print(f"Seed {seed}: folders not found, skipping.")
        continue

    sim1_csv = os.path.join(sim1_dirs[0], "SimEmbodiedAnt_average_rewards.csv")
    sim2_csv = os.path.join(sim2_dirs[0], "SimEmbodiedAnt_average_rewards.csv")

    if not os.path.exists(sim1_csv) or not os.path.exists(sim2_csv):
        print(f"Seed {seed}: CSV not found, skipping.")
        continue

    df1 = pd.read_csv(sim1_csv)
    df2 = pd.read_csv(sim2_csv)

    last_step_1 = df1["step"].max()
    df2["step"] = df2["step"] + last_step_1
    df_all = pd.concat([df1, df2], ignore_index=True)

    # interpolate this seed's curve onto the common step grid
    interp_reward = np.interp(STEP_GRID, df_all["step"], df_all["reward"])
    curves.append(interp_reward)

curves = np.array(curves)  # shape: (num_seeds, len(STEP_GRID))
mean_reward = curves.mean(axis=0)
sem_reward = curves.std(axis=0, ddof=1) / np.sqrt(curves.shape[0])

plt.figure(figsize=(14, 6))
plt.plot(STEP_GRID, mean_reward, label=f"mean (n={curves.shape[0]} seeds)")
plt.fill_between(STEP_GRID, mean_reward - 3 * sem_reward, mean_reward + 3 * sem_reward,
                  alpha=0.3, label="±3 SEM")

plt.xlabel("Total Step")
plt.ylabel("Reward")
plt.xlim(0, 150000)
plt.ylim(0, 0.2)
plt.yticks([0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20])
plt.title("Reward vs Step: Mean ± SEM Across Seeds")
plt.legend(fontsize=8)
plt.grid(True)
output_path = os.path.join(RUNS_DIR, "reward_mean_sem.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved")
