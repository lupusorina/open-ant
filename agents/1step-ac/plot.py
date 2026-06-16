

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CSV_PATH = "/home/seliu/open-ant/agents/1step-ac/runs_cartpole/experiment17/one_step_ac_20260615-180421_seed_0/episode_log.csv"

ENV_DT = 0.02

YLIMS = {
    "ep_return": (0, 600),
    "avg_reward_per_step": (0, 1.2),
    "avg_delta": (-2, 2),
    "avg_v_s": (0, 300),
    "avg_critic_loss": (-1, 40),
}


def load_csv(path):
    episodes = []
    global_steps = []
    ep_return = []
    avg_delta = []
    avg_v_s = []
    avg_critic_loss = []

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            global_steps.append(int(row["global_step"]))
            ep_return.append(float(row["ep_return"]))
            avg_delta.append(float(row["avg_delta"]))
            avg_v_s.append(float(row["avg_v_s"]))
            avg_critic_loss.append(float(row["avg_critic_loss"]))

    return episodes, global_steps, ep_return, avg_delta, avg_v_s, avg_critic_loss


def compute_episode_steps(global_steps):
    episode_steps = []
    prev_step = 0

    for step in global_steps:
        episode_steps.append(step - prev_step)
        prev_step = step

    return episode_steps


def main():
    csv_path = CSV_PATH
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    out_path = os.path.join(out_dir, "episode_log_plots.png")

    episodes, global_steps, ep_return, avg_delta, avg_v_s, avg_critic_loss = load_csv(csv_path)

    episode_steps = compute_episode_steps(global_steps)

    avg_reward_per_step = [
        ret / steps if steps > 0 else 0.0
        for ret, steps in zip(ep_return, episode_steps)
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # episode return vs episode number
    ax = axes[0, 0]
    ax.plot(episodes, ep_return, color="tab:blue")
    ax.set_title("Episode Return vs Episode Number")
    ax.set_xlabel("Episode number")
    ax.set_ylabel("Return")
    ax.set_ylim(*YLIMS["ep_return"])
    ax.grid(True, alpha=0.3)

    # avg reward per step vs global step
    ax = axes[0, 1]
    ax.plot(global_steps, avg_reward_per_step, color="tab:purple")
    ax.set_title("Average Reward per Step vs Global Step")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Avg reward / step")
    ax.set_ylim(*YLIMS["avg_reward_per_step"])
    ax.grid(True, alpha=0.3)

    # avg delta (TD error) vs global step
    ax = axes[1, 0]
    ax.plot(global_steps, avg_delta, color="tab:orange")
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("Avg TD Error vs Global Step")
    ax.set_xlabel("Global step")
    ax.set_ylabel("avg delta")
    ax.set_ylim(*YLIMS["avg_delta"])
    ax.grid(True, alpha=0.3)

    # avg v_s vs global step
    ax = axes[1, 1]
    ax.plot(global_steps, avg_v_s, color="tab:green")
    ax.set_title("Avg Critic Value Estimate vs Global Step")
    ax.set_xlabel("Global step")
    ax.set_ylabel("avg v_s")
    ax.set_ylim(*YLIMS["avg_v_s"])
    ax.grid(True, alpha=0.3)

    # avg critic loss vs global step
    ax = axes[2, 0]
    ax.plot(global_steps, avg_critic_loss, color="tab:red")
    ax.set_title("Avg Critic Loss vs Global Step")
    ax.set_xlabel("Global step")
    ax.set_ylabel("avg critic_loss")
    ax.set_ylim(*YLIMS["avg_critic_loss"])
    ax.grid(True, alpha=0.3)

    axes[2, 1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()