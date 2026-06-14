

import os
import csv
import argparse
import random
from datetime import datetime
import gymnasium as gym
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
    # trying hidden=64, slightly smaller network for critic. performed well in one-step ac without idbd.
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
    parser.add_argument("--num_episodes", type=int, default=5000)
    parser.add_argument("--run_dir", type=str, default="runs")
    parser.add_argument("--exp_name", type=str, default="one_step_ac_idbd")
    parser.add_argument("--actor_lr", type=float, default=6.85e-05,
                        help="initial actor learning rate; IDBD adapts from this value")
    parser.add_argument("--critic_lr", type=float, default=0.0117,
                        help="initial critic learning rate; IDBD adapts from this value")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--no-use_layer_norm", action="store_false",
                        dest="use_layer_norm",
                        help="disable layer normalization in networks")
    parser.add_argument("--capture_video", action="store_true",
                        help="whether to capture videos")
    parser.add_argument("--save_video_every_n_episodes", type=int, default=1,
                        help="save video every n episodes")
    parser.add_argument("--flush_log_every_n_episodes", type=int, default=1,
                        help="flush episode_log.csv every n episodes")

    # IDBD specific args
    parser.add_argument("--actor_meta_lr", type=float, default=1e-5)
    parser.add_argument("--critic_meta_lr", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--max_lr", type=float, default=1e-2)

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
        print("RecordVideo")
        env = gym.wrappers.RecordVideo(
            env,
            os.path.join(out_dir, "videos", run_name),
            episode_trigger=lambda ep: ep % args.save_video_every_n_episodes == 0,
            video_length=0,
        )

    env.action_space.seed(args.seed)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    actor = Actor(obs_dim, n_actions, use_layer_norm=args.use_layer_norm).to(DEVICE)
    critic = Critic(obs_dim, use_layer_norm=args.use_layer_norm).to(DEVICE)

    # Create learning rate scalars for actor & critic. 
    # h_actor and h_critic = sensitivty of each parameter in actor, and each parameter in critic, 
    # to the corresponding beta. h = vector. 
    beta_actor = torch.tensor(np.log(args.actor_lr), dtype=torch.float32, device=DEVICE)
    beta_critic = torch.tensor(np.log(args.critic_lr), dtype=torch.float32, device=DEVICE)

    h_actor = [
        torch.zeros_like(p, device=DEVICE)
        for p in actor.parameters()
    ]

    h_critic = [
        torch.zeros_like(p, device=DEVICE)
        for p in critic.parameters()
    ]

    min_beta = np.log(args.min_lr)
    max_beta = np.log(args.max_lr)


    episode_returns = []
    episode_avg_delta = []
    episode_avg_abs_delta = []
    episode_avg_v_s = []
    episode_avg_critic_loss = []
    episode_avg_actor_lr = []
    episode_avg_critic_lr = []

    
    global_step = 0

    csv_path = os.path.join(out_dir, "episode_log.csv")
    fieldnames = [
        "episode",
        "video_episode_index",
        "episode_start_step",
        "global_step",
        "n_steps",
        "ep_return",
        "terminated",
        "truncated",
        "avg_delta",
        "avg_abs_delta",
        "avg_v_s",
        "avg_critic_loss",
        "avg_actor_lr",
        "avg_critic_lr",
    ]

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_file.flush()
    os.fsync(csv_file.fileno())
    print(f"[seed {args.seed}] Logging episodes live to {csv_path}")

    try:
        for episode in range(args.num_episodes):
            episode_start_step = global_step
            obs, _ = env.reset(seed=args.seed + episode)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)

            ep_return = 0.0
            done = False

            # Metrics tracked over this episode.
            delta_sum = 0.0
            abs_delta_sum = 0.0
            v_s_sum = 0.0
            critic_loss_sum = 0.0
            actor_lr_sum = 0.0
            critic_lr_sum = 0.0
            n_steps = 0
            terminated = False
            truncated = False

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
                    v_s_next = (
                        critic(next_obs_t.unsqueeze(0)).squeeze(0)
                        if not terminated
                        else torch.tensor(0.0, device=DEVICE)
                    )

                td_target = reward + args.gamma * v_s_next
                delta = td_target - v_s

                delta_sum += delta.item()
                abs_delta_sum += abs(delta.item())
                v_s_sum += v_s.item()
                critic_loss_sum += 0.5 * (delta.item() ** 2)

    # IDBD critic Update
            # clear old gradients
                critic.zero_grad()
            
            # compute ∇w Vw(s_t) for every w critic parameter. Now, every critic parameter
            # has a p.grad attached that = ∂Vw(s_t) / ∂ that parameter
                v_s.backward(retain_graph=True)

            # save the gradients into a list. 
                critic_grads = [
                    p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                    for p in critic.parameters()
                ]

                with torch.no_grad():
                    # dot_critic = (∇_w V_w(s_t))^T h_w = dot product of the transpose of (grad V w.r.t each weight) and vector h.
                    dot_critic = sum(
                        (g * h).sum()
                        for g, h in zip(critic_grads, h_critic)
                    )

                    # update beta_critic first:
                    # beta_{t+1}^w = beta_t^w + eta_w * delta_t * (g_w^T h_w)
                    beta_critic += args.critic_meta_lr * delta.detach() * dot_critic
                    beta_critic.clamp_(min_beta, max_beta)

                    # Use alpha_{t+1}^V = exp(beta_{t+1}^w)
                    alpha_critic = torch.exp(beta_critic)

                    # h of critic update
                    for h, g in zip(h_critic, critic_grads):
                        h += alpha_critic * (delta.detach() - dot_critic) * g

                    # critic weight update: w += alpha(t+1) * delta * grad V
                    for p, g in zip(critic.parameters(), critic_grads):
                        p += alpha_critic * delta.detach() * g
                    critic_lr_sum += alpha_critic.item()


        # IDBD actor Update
                # actor_signal is the scalar multiplying ∇log pi(a_t|s_t) in actor loss J.
                actor_signal = delta.detach()

                # clear old actor gradients
                actor.zero_grad()

                # Compute g_theta = ∇_theta log pi_theta(a_t|s_t)
                # After this line, each actor parameter p has p.grad = ∂log_prob/∂p.
                log_prob.backward()

                # Save actor gradients into a list
                actor_grads = [
                    p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                    for p in actor.parameters()
                ]

                with torch.no_grad():
                    # dot_actor = (∇_theta log pi)^T h_theta
                    dot_actor = sum(
                        (g * h).sum()
                        for g, h in zip(actor_grads, h_actor)
                    )
                    # update beta_actor first:
                    # beta_{t+1}^theta = beta_t^theta + eta_theta * I_t * delta_t * (g_theta^T h_theta)
                    beta_actor += args.actor_meta_lr * actor_signal * dot_actor
                    beta_actor.clamp_(min_beta, max_beta)

                    # Use alpha_{t+1}^pi = exp(beta_{t+1}^theta)
                    alpha_actor = torch.exp(beta_actor)

                    # update h_actor, first-order approx 
                    # h_{t+1}^theta = h_t^theta + alpha_{t+1}^pi * delta_t * g_theta
                    # however, this means there's no decay term for h_actor, since we assume 
                    # gradient of delta_t w.r.t beta_actor = 0. But in h_critic, there is a decay term
                    for h, g in zip(h_actor, actor_grads):
                        h += alpha_actor * actor_signal * g
        

                    # Update actor weights:
                    # theta_{t+1} = theta_t + alpha_{t+1}^pi * delta_t * ∇_theta log pi(a_t|s_t)
                    for p, g in zip(actor.parameters(), actor_grads):
                        p += alpha_actor * actor_signal * g

                    actor_lr_sum += alpha_actor.item()

           # I = 1.0
                obs = next_obs_t

            episode_returns.append(ep_return)
            # average metrics for this episode
            avg_delta = delta_sum / n_steps
            avg_abs_delta = abs_delta_sum / n_steps
            avg_v_s = v_s_sum / n_steps
            avg_critic_loss = critic_loss_sum / n_steps
            avg_actor_lr = actor_lr_sum / n_steps
            avg_critic_lr = critic_lr_sum / n_steps
            
            episode_avg_delta.append(avg_delta)
            episode_avg_abs_delta.append(avg_abs_delta)
            episode_avg_v_s.append(avg_v_s)
            episode_avg_critic_loss.append(avg_critic_loss)
            episode_avg_actor_lr.append(avg_actor_lr)
            episode_avg_critic_lr.append(avg_critic_lr)

            row = {
                "episode": episode + 1,
                "video_episode_index": episode,
                "episode_start_step": episode_start_step,
                "global_step": global_step,
                "n_steps": n_steps,
                "ep_return": ep_return,
                "terminated": int(terminated),
                "truncated": int(truncated),
                "avg_delta": avg_delta,
                "avg_abs_delta": avg_abs_delta,
                "avg_v_s": avg_v_s,
                "avg_critic_loss": avg_critic_loss,
                "avg_actor_lr": avg_actor_lr,
                "avg_critic_lr": avg_critic_lr,
            }
            csv_writer.writerow(row)

            if (episode + 1) % 20 == 0:
                    avg_return = np.mean(episode_returns[-20:])
                    recent_avg_delta = np.mean(episode_avg_delta[-20:])
                    recent_avg_abs_delta = np.mean(episode_avg_abs_delta[-20:])
                    recent_avg_v_s = np.mean(episode_avg_v_s[-20:])
                    recent_avg_critic_loss = np.mean(episode_avg_critic_loss[-20:])
                    recent_avg_actor_lr = np.mean(episode_avg_actor_lr[-20:])
                    recent_avg_critic_lr = np.mean(episode_avg_critic_lr[-20:])

                    print(
                        f"[seed {args.seed}] Episode {episode+1}, "
                        f"avg return (last 20): {avg_return:.2f}, "
                        f"avg|delta|: {recent_avg_abs_delta:.3f}, "
                        f"avg delta: {recent_avg_delta:.3f}, "
                        f"avg v_s: {recent_avg_v_s:.3f}, "
                        f"avg critic_loss: {recent_avg_critic_loss:.4f}, "
                        f"actor_lr: {recent_avg_actor_lr:.2e}, "
                        f"critic_lr: {recent_avg_critic_lr:.2e}"
                    )
            if (episode + 1) % args.flush_log_every_n_episodes == 0:
                    csv_file.flush()
                    os.fsync(csv_file.fileno())
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
        "beta_actor": beta_actor.item(),
        "beta_critic": beta_critic.item(),
        "actor_lr_final": torch.exp(beta_actor).item(),
        "critic_lr_final": torch.exp(beta_critic).item(),
        "args": vars(args),
    }, weights_path)
    print(f"[seed {args.seed}] Saved weights to {weights_path}")


if __name__ == "__main__":
    main()
