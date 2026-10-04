import os
import glob
import csv
import pandas as pd
import matplotlib.pyplot as plt


def read_step_reward(csv_path):
    """Read only the 'step' and 'reward' columns, tolerating rows with
    extra/trailing fields (some logs gained a mean_return column mid-run).

    Some eval logs were restarted mid-run and the logger reopened the CSV in
    append mode, re-writing the header and restarting the step count from 1
    partway through the file. Any repeated header row is skipped, and step
    numbers are made monotonically increasing by offsetting each restarted
    segment to continue from the last step seen before it."""
    steps, rewards = [], []
    step_offset = 0.0
    last_step = 0.0
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        step_idx = header.index("step")
        reward_idx = header.index("reward")
        for row in reader:
            if len(row) <= max(step_idx, reward_idx):
                continue
            if row[step_idx] == "step":
                step_offset = last_step
                continue
            step = float(row[step_idx]) + step_offset
            last_step = step
            steps.append(step)
            rewards.append(float(row[reward_idx]))
    return pd.DataFrame({"step": steps, "reward": rewards})

RUNS_DIR = "/home/serenaliu/caltech_linc_home/open-ant/agents/mpo/runs/qrdqn_avg4actions"

# ← Edit this list to include whichever seeds you want
SEEDS_TO_PLOT = [1, 2, 3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
# SEEDS_TO_PLOT = [1, 2, 8, 12, 15]
XLIM = 148000  # maximum step to plot
SIM1_XLIM = 40000  # expected final step of sim1
short_seeds = []  # seeds whose final step never reaches XLIM
short_sim1_seeds = []  # seeds whose sim1 final step never reaches SIM1_XLIM

PLOT_EVAL = True  # ← set False to skip overlaying eval curves
EVAL_GLOB = "eval_sim1_150k_*_seed_{seed}"  # folder pattern under RUNS_DIR

# Fixed, unique color per seed number (1-30) so colors never repeat/collide,
# regardless of which seeds actually have data in a given run.
_palette = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)[:10]
_palette = _palette[::-1]  # reversed so the vibrant tab20 colors land on the
                           # higher seed numbers, which plot last and sit on top
SEED_COLORS = {seed: _palette[seed - 1] for seed in range(1, 31)}

plt.figure(figsize=(14, 6))

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

    df1 = read_step_reward(sim1_csv)
    df2 = read_step_reward(sim2_csv)

    last_step_1 = df1["step"].max()
    if last_step_1 < SIM1_XLIM:
        short_sim1_seeds.append((seed, last_step_1))

    df2["step"] = df2["step"] + last_step_1
    df_all = pd.concat([df1, df2], ignore_index=True)

    final_step = df_all["step"].max()
    if final_step < XLIM:
        short_seeds.append((seed, final_step))

    # Plot the reward curve using this seed's fixed, unique color
    color = SEED_COLORS.get(seed, "black")
    plt.plot(df_all["step"], df_all["reward"], linewidth=1, label=f"seed {seed}", color=color)

    # Draw vertical line in same color, only label first one to avoid legend clutter
    plt.axvline(last_step_1, linestyle="--", linewidth=0.8, color=color,
                label="_nolegend_")

    # Optionally overlay the eval curve for this seed, offset so it visually
    # starts right after sim1 (eval CSV steps start at 1, so shift by last_step_1)
    if PLOT_EVAL:
        eval_dirs = sorted(glob.glob(f"{RUNS_DIR}/{EVAL_GLOB.format(seed=seed)}"), reverse=True)
        if not eval_dirs:
            print(f"Seed {seed}: eval folder not found, skipping eval plot.")
        else:
            eval_csv = os.path.join(eval_dirs[0], "SimEmbodiedAnt_average_rewards.csv")
            if not os.path.exists(eval_csv):
                print(f"Seed {seed}: eval CSV not found, skipping eval plot.")
            else:
                df_eval = read_step_reward(eval_csv)
                df_eval["step"] = df_eval["step"] + last_step_1
                plt.plot(df_eval["step"], df_eval["reward"], linewidth=1, linestyle=":",
                          color=color, label="_nolegend_")

# Single legend entry for the vertical lines
plt.axvline(-1, linestyle="--", linewidth=0.8, color="gray", label="continual learning starts")
if PLOT_EVAL:
    plt.plot([], [], linestyle=":", linewidth=1, color="gray", label="eval (sim1 policy)")

plt.xlabel("Total Step")
plt.ylabel("Reward")
plt.xlim(0, 148000)
plt.ylim(-0.02, 0.22)
#plt.yticks([0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20])
plt.title("Reward vs Step: All Seeds Overlaid")
plt.legend(fontsize=8)
plt.grid(False)
output_path = os.path.join(RUNS_DIR, "rewardallseed.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved")

if short_sim1_seeds:
    print(f"\nSeeds whose sim1 did not reach {SIM1_XLIM} steps:")
    for seed, final_step in short_sim1_seeds:
        print(f"  seed {seed}: sim1 final step = {final_step}")
else:
    print(f"\nAll plotted seeds' sim1 reached {SIM1_XLIM} steps.")

if short_seeds:
    print(f"\nSeeds that did not reach {XLIM} steps:")
    for seed, final_step in short_seeds:
        print(f"  seed {seed}: final step = {final_step}")
else:
    print(f"\nAll plotted seeds reached {XLIM} steps.")
