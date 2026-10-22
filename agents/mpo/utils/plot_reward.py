import os

import matplotlib.pyplot as plt
import pandas as pd

run_dir = 'runs/mpo_ant_20260723-155539_seed_1'
csv_path = os.path.join(run_dir, "EAnt_average_rewards.csv")
out_path = os.path.join(run_dir, "average_reward.png")

df = pd.read_csv(csv_path)
dt = 0.12  # seconds per step (from args.json)
time_min = df["step"] * dt / 60.0
reward_cm_s = df["reward"] * 100.0  # m/s -> cm/s

plt.figure(figsize=(10, 5))
plt.plot(time_min, reward_cm_s, linewidth=1)
plt.xlabel("Time (min)")
plt.ylabel("Average Reward (cm/s)")
plt.title("EAnt Average Reward")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_path, dpi=200)
plt.close()

print(f"Saved to {out_path}")
