import os
import pandas as pd
import matplotlib.pyplot as plt
import re

csv_path_1 = "/home/serenaliu/caltech_linc_home/open-ant/agents/mpo/runs/spi1536_gamma92/mpo_20260726-150707_seed_10/SimEmbodiedAnt_average_rewards.csv"
csv_path_2 = "/home/serenaliu/caltech_linc_home/open-ant/agents/mpo/runs/ensemble1/continuous_mpo_20260726-153715_seed_10/SimEmbodiedAnt_average_rewards.csv"

output_dir = "/home/serenaliu/caltech_linc_home/open-ant/agents/mpo/runs/spi1536_gamma92/mpo_20260726-150707_seed_10"
match = re.search(r"_seed_(\d+)", csv_path_1)

if match is None:
    raise ValueError(f"Could not detect seed number from path: {csv_path_1}")

seed = int(match.group(1))
figure_name = f"seed{seed}_mpo.png"


df1 = pd.read_csv(csv_path_1)
df2 = pd.read_csv(csv_path_2)

# Make second CSV continue after first CSV
last_step_1 = df1["step"].max()
df2["step"] = df2["step"] + last_step_1

# Combine them
df_all = pd.concat([df1, df2], ignore_index=True)

plt.figure(figsize=(12, 6))
plt.plot(df_all["step"], df_all["reward"], linewidth=1)

# Optional vertical line showing where second run starts
plt.axvline(last_step_1, linestyle="--", linewidth=1)
#plt.text(last_step_1, df_all["reward"].max(), " second sim starts", va="top")

plt.xlabel("Total Step")
plt.ylabel("Reward")
plt.ylim(0, 0.2)
plt.yticks([0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20])
plt.title(f"Reward vs Step: Sim + Continual Learning, seed {seed}")
plt.grid(True)

plt.savefig(os.path.join(output_dir, figure_name), dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved to {os.path.join(output_dir, figure_name)}")
