

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
    parser.add_argument("--total_timesteps", type=int, default=40000)
    parser.add_argument("--run_dir", type=str, default="runs")
    parser.add_argument("--exp_name", type=str, default="one_step_ac")
    parser.add_argument("--actor_lr", type=float, default=6.85e-05)
    parser.add_argument("--critic_lr", type=float, default=0.0117)
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
    parser.add_argument("--flush_log_every_n_episodes", type=int, default=1,
                        help="flush episode_log.csv every n episodes")
    parser.add_argument("--dt", type=float, default=None)
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
    
    env_dt = args.dt if args.dt is not None else env.unwrapped.tau
    print(f"Using env_dt = {env_dt}", flush=True)
    
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

    # creates a SGD optimizer for the actor, and one for the critic. 
    # actor_opt updates all actor parameters using the actor_lr
    #actor_opt = optim.SGD(actor.parameters(), lr=args.actor_lr)
    #critic_opt = optim.SGD(critic.parameters(), lr=args.critic_lr)

    episode_returns = []

    episode_avg_delta = []
    episode_avg_abs_delta = []
    episode_avg_v_s = []
    episode_avg_critic_loss = []

    global_step = 0

    csv_path = os.path.join(out_dir, "episode_log.csv")
    fieldnames = [
        "episode",
        "global_step",
        "ep_return",
        "avg_delta",
        "avg_abs_delta",
        "avg_v_s",
        "avg_critic_loss",
    ]

    # create CSV to write to 
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_file.flush()
    os.fsync(csv_file.fileno())

    try:
        obs, _ = env.reset(seed=args.seed)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)

        episode = 0
        ep_return = 0.0
        n_steps = 0

        delta_sum = 0.0
        abs_delta_sum = 0.0
        v_s_sum = 0.0
        critic_loss_sum = 0.0

        for global_step in range(args.total_timesteps):
            dist = actor(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=DEVICE)

            ep_return += reward
            n_steps += 1

            v_s = critic(obs.unsqueeze(0)).squeeze(0)

            with torch.no_grad():
                if not terminated:
                    v_s_next = critic(next_obs_t.unsqueeze(0)).squeeze(0)
                else:
                    v_s_next = torch.tensor(0.0, device=DEVICE)
        
           
         
            td_target = reward * env_dt + (args.gamma ** env_dt) * v_s_next
            delta = td_target - v_s

            delta_sum += delta.item()
            abs_delta_sum += abs(delta.item())
            v_s_sum += v_s.item()
            critic_loss_sum += 0.5 * (delta.item() ** 2)

            # zero old gradients
            critic.zero_grad()
            #take gradient of Vw(S)
            v_s.backward(retain_graph=True)
                
            #Update critic weight
            with torch.no_grad():
                for p in critic.parameters():
                    if p.grad is not None:
                        p += args.critic_lr * delta.detach() * p.grad

            # update actor via manual gradient ascent, maximize the objective
            actor.zero_grad()
            log_prob.backward()
            with torch.no_grad():
                for p in actor.parameters():
                    if p.grad is not None:
                        p += args.actor_lr * delta.detach() * p.grad
            
            if global_step % 100 == 0:
                print(
                    f"[step {global_step}] ep={episode+1} n_steps={n_steps} "
                    f"reward={reward:.2f} delta={delta.item():.4f} "
                    f"terminated={terminated} truncated={truncated}",
                    flush=True,
                )

            if terminated or truncated:
                avg_delta = delta_sum / n_steps
                avg_abs_delta = abs_delta_sum / n_steps
                avg_v_s = v_s_sum / n_steps
                avg_critic_loss = critic_loss_sum / n_steps

                episode_returns.append(ep_return)
                episode_avg_delta.append(avg_delta)
                episode_avg_abs_delta.append(avg_abs_delta)
                episode_avg_v_s.append(avg_v_s)
                episode_avg_critic_loss.append(avg_critic_loss)

                row = {
                    "episode": episode + 1,
                    "global_step": global_step + 1,
                    "ep_return": ep_return,
                    "avg_delta": avg_delta,
                    "avg_abs_delta": avg_abs_delta,
                    "avg_v_s": avg_v_s,
                    "avg_critic_loss": avg_critic_loss,
                }

                csv_writer.writerow(row)

                if (episode + 1) % 20 == 0:
                    avg_return = np.mean(episode_returns[-20:])
                    recent_avg_delta = np.mean(episode_avg_delta[-20:])
                    recent_avg_abs_delta = np.mean(episode_avg_abs_delta[-20:])
                    recent_avg_v_s = np.mean(episode_avg_v_s[-20:])
                    recent_avg_critic_loss = np.mean(episode_avg_critic_loss[-20:])

                    print(
                        f"[seed {args.seed}] Episode {episode+1}, "
                        f"global_step: {global_step+1}, "
                        f"avg return last 20: {avg_return:.2f}, "
                        f"avg|delta|: {recent_avg_abs_delta:.3f}, "
                        f"avg delta: {recent_avg_delta:.3f}, "
                        f"avg v_s: {recent_avg_v_s:.3f}, "
                        f"avg critic_loss: {recent_avg_critic_loss:.4f}",
                        flush=True,
                    )

                if (episode + 1) % args.flush_log_every_n_episodes == 0:
                    csv_file.flush()
                    os.fsync(csv_file.fileno())

                episode += 1

                obs, _ = env.reset(seed=args.seed + episode)
                obs = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)

                ep_return = 0.0
                n_steps = 0
                delta_sum = 0.0
                abs_delta_sum = 0.0
                v_s_sum = 0.0
                critic_loss_sum = 0.0

            else:
                obs = next_obs_t

    finally:
        csv_file.flush()
        os.fsync(csv_file.fileno())
        csv_file.close()
        print(f"[seed {args.seed}] Finished writing log to {csv_path}")
        
        env.close()

    
    weights_path = os.path.join(out_dir, "weights.pth")
    torch.save({
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "args": vars(args),
    }, weights_path)
    print(f"[seed {args.seed}] Saved weights to {weights_path}")


if __name__ == "__main__":
    main()
