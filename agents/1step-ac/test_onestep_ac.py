

import os
import csv
import argparse
import random
import gymnasium as gym
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128, use_layer_norm=False):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_logits = nn.Linear(hidden, n_actions)

        if use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden)
            self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x):
        x = self.fc1(x)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        x = F.relu(x)
        logits = self.fc_logits(x)
        return Categorical(logits=logits)


class Critic(nn.Module):
    # I've been fidgeting with hidden size since i read online about the network might need to be
    # smaller to avoid overfitting in a simple env.
    def __init__(self, obs_dim, hidden=64, use_layer_norm=False):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

        if use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden)
            self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x):
        x = self.fc1(x)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x.squeeze(-1)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--run_dir", type=str, default="runs")
    parser.add_argument("--exp_name", type=str, default="one_step_ac")
    parser.add_argument("--actor_step_size", type=float, default=6.85e-05)
    parser.add_argument("--critic_step_size", type=float, default=0.0117)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--no-use_layer_norm", action="store_false",
                        dest="use_layer_norm",
                        help="disable layer normalization in networks")
    # parser.add_argument("--solved_threshold", type=float, default=195.0,
    #                     help="rolling-average return where the experiment will be considered 'solved'")
    # parser.add_argument("--solved_window", type=int, default=100,
    #                     help="number of episodes to average over for the solved check")
    parser.add_argument("--capture_video", action="store_true",
                        help="whether to capture videos")
    parser.add_argument("--save_video_every_n_episodes", type=int, default=1,
                        help="save video every n episodes")
    parser.set_defaults(
        use_layer_norm=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()

   # set up the seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.exp_name}_{date_time}_seed_{args.seed}"
    out_dir = os.path.join(args.run_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    render_mode = "rgb_array" if args.capture_video else None
    env = gym.make("CartPole-v1", render_mode=render_mode)
    if args.capture_video:
        print('RecordVideo')
        env = gym.wrappers.RecordVideo(env, os.path.join(out_dir, "videos", run_name),
                                       episode_trigger=lambda ep: ep % args.save_video_every_n_episodes == 0,
                                       video_length=0)
    env.action_space.seed(args.seed)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    actor = Actor(obs_dim, n_actions, use_layer_norm=args.use_layer_norm).to(DEVICE)
    critic = Critic(obs_dim, use_layer_norm=args.use_layer_norm).to(DEVICE)

    actor_opt = optim.SGD(actor.parameters(), lr=args.actor_step_size)
    critic_opt = optim.SGD(critic.parameters(), lr=args.critic_step_size)

    episode_returns = []

    log_rows = []
    global_step = 0

    for episode in range(args.num_episodes):
        obs, _ = env.reset(seed=args.seed)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)

        I = 1.0
        ep_return = 0.0
        done = False

        # some metrics we keep track for an episode
        delta_sum = 0.0
        abs_delta_sum = 0.0
        v_s_sum = 0.0
        critic_loss_sum = 0.0
        n_steps = 0

        while not done:
            dist = actor(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            ep_return += reward
            global_step += 1
            n_steps += 1

            next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=DEVICE)

            v_s = critic(obs.unsqueeze(0)).squeeze(0)
            with torch.no_grad():
                v_s_next = critic(next_obs_t.unsqueeze(0)).squeeze(0) if not terminated else torch.tensor(0.0, device=DEVICE)

            td_target = reward + args.gamma * v_s_next
            delta = td_target - v_s

            delta_sum += delta.item()
            abs_delta_sum += abs(delta.item())
            v_s_sum += v_s.item()
            critic_loss_sum += 0.5 * (delta.item() ** 2)

            critic_opt.zero_grad()
            critic_loss = -delta.detach() * v_s
            critic_loss.backward()
            critic_opt.step()

            actor_opt.zero_grad()
            actor_loss = -delta.detach() * log_prob
            actor_loss.backward()
            actor_opt.step()

            obs = next_obs_t

        episode_returns.append(ep_return)

        # average metrics for this episode
        avg_delta = delta_sum / n_steps
        avg_abs_delta = abs_delta_sum / n_steps
        avg_v_s = v_s_sum / n_steps
        avg_critic_loss = critic_loss_sum / n_steps

        log_rows.append({
            "episode": episode + 1,
            "global_step": global_step,
            "ep_return": ep_return,
            "avg_delta": avg_delta,
            "avg_abs_delta": avg_abs_delta,
            "avg_v_s": avg_v_s,
            "avg_critic_loss": avg_critic_loss,
        })

        if (episode + 1) % 20 == 0:
            avg_return = np.mean(episode_returns[-20:])
            recent = log_rows[-20:]
            recent_avg_delta = np.mean([r["avg_delta"] for r in recent])
            recent_avg_abs_delta = np.mean([r["avg_abs_delta"] for r in recent])
            recent_avg_v_s = np.mean([r["avg_v_s"] for r in recent])
            recent_avg_critic_loss = np.mean([r["avg_critic_loss"] for r in recent])
            print(
                f"[seed {args.seed}] Episode {episode+1}, "
                f"avg return (last 20): {avg_return:.2f}, "
                f"avg|delta|: {recent_avg_abs_delta:.3f}, "
                f"avg delta: {recent_avg_delta:.3f}, "
                f"avg v_s: {recent_avg_v_s:.3f}, "
                f"avg critic_loss: {recent_avg_critic_loss:.4f}"
            )

        # check if the solved condition for rewards is met.
        # if len(episode_returns) >= args.solved_window:
        #     rolling_avg = np.mean(episode_returns[-args.solved_window:])
        #     if rolling_avg >= args.solved_threshold:
        #         print(
        #             f"[seed {args.seed}] Solved at episode {episode+1}! "
        #             f"Rolling avg over last {args.solved_window} episodes: {rolling_avg:.2f} "
        #             f">= threshold {args.solved_threshold}"
        #         )
        #         break

    env.close()


    csv_path = os.path.join(out_dir, "episode_log.csv")
    fieldnames = ["episode", "global_step", "ep_return", "avg_delta", "avg_abs_delta", "avg_v_s", "avg_critic_loss"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in log_rows:
            writer.writerow(row)
    print(f"[seed {args.seed}] Saved log to {csv_path}")

    
    weights_path = os.path.join(out_dir, "weights.pth")
    torch.save({
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "args": vars(args),
    }, weights_path)
    print(f"[seed {args.seed}] Saved weights to {weights_path}")


if __name__ == "__main__":
    main()
