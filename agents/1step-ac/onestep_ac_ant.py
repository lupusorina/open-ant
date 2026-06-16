import os
import sys
import csv
import json
import argparse
import random
import gymnasium as gym
from datetime import datetime
from torch.distributions import Normal
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchrl.data import ReplayBuffer, LazyTensorStorage, RandomSampler
from tensordict import TensorDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../sim')))
from ant_mujoco import AntEnv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../embodied_ant_env')))
from embodied_ant_env import make_ant_env, ForwardTask, BackAndForthTask
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from reward import RewardTracker

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# clamp vals so standard deviation won't explode
LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    # n_actions mean # of discrete action choices - for cartpole. 
    # aciton_dim means # of continuous joint values. 
    def __init__(self, obs_dim, action_dim, hidden=256, use_layer_norm=False):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        # two output heads, one for mean and one for log std, represent a gaussian distribution of action output
        self.fc_mean   = nn.Linear(hidden, action_dim)
        self.fc_logstd = nn.Linear(hidden, action_dim)

        # math wrappers to covnert RL output into physical torque vals
        self.register_buffer("action_scale", torch.ones(action_dim))
        self.register_buffer("action_bias",  torch.zeros(action_dim))

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

        # gives a mean action vector with same dim as action_dim
        mean    = self.fc_mean(x)

        # Since tanh(z) is always between [-1, 1], applying tanh on raw logstd ensures log_std always [-1, 1]
        # for every action dimension.
        log_std = torch.tanh(self.fc_logstd(x))

        # map a value from [-1, 1] to the new min max range of [-5, 2]
        # squashes raw network output into [log_std_min, log_std_max]
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std     = log_std.exp()
        return Normal(mean, std)


class Critic(nn.Module):
    # 
    def __init__(self, obs_dim, hidden=256, use_layer_norm=False):
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
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
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
    # parser.add_argument("--save_video_every_n_episodes", type=int, default=1,
    #                     help="save video every n episodes")
    parser.add_argument("--dt", type=float, default=0.12)
    parser.add_argument("--hw_config", type=str, default=None)
    parser.add_argument("--render_mode", type=str, default="rgb_array")
    parser.add_argument("--task_type", type=str, default="back_and_forth",
                        choices=["forward", "back_and_forth"])
    parser.add_argument("--radius_back_and_forth", type=float, default=0.3)
    parser.add_argument("--origin_back_and_forth", type=float, nargs=2, default=[0.75, -0.3])
    parser.add_argument("--reward_scale", type=float, default=1)
    parser.add_argument("--model_path", type=str,
                        default="../../sim/assets/ant_with_camera_after_sys_id.xml")
    parser.add_argument("--no-terminate_on_upside_down", action="store_false",
                        dest="terminate_on_upside_down",
                        help="do not terminate episode when upside down")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--save_video_every_n_steps", type=int, default=4000)
    parser.add_argument("--total_timesteps", type=int, default=40000)
    parser.add_argument("--log_every_n_steps", type=int, default=4000)
        
    parser.set_defaults(
        use_layer_norm=True,
        terminate_on_upside_down=True,
    )
    return parser.parse_args()

def make_ant_envs(args, task, disk_folder, run_name, runs_directory='runs'):
    """Create the vectorized environment outside the SAC class."""
    def make_env(seed, idx, capture_video, run_name):
        def _init():
            joint_config = {
                'hip_zero': 0,
                'knee_zero': -np.radians(50),
                'hip_range': np.radians(30),
                'knee_range': np.radians(20),
            }
            if args.hw_config is None:
                env = AntEnv(
                    control_dt=args.dt,
                    render_mode=args.render_mode,
                    terminate_on_upside_down=args.terminate_on_upside_down,
                    task=task,
                    joint_config=joint_config,
                    model_path=os.path.join(os.path.dirname(__file__), args.model_path),
                )
            else:
                with open(args.hw_config, 'r') as f:
                    cfg = json.load(f)
                env = make_ant_env(cfg, render_mode=args.render_mode,
                                   dt=args.dt,
                                   joint_config=joint_config,
                                   task=task,
                                   )
                # env.metadata['render_fps'] = 1/args.dt

            if capture_video and idx == 0:
                print('RecordVideo')
                env = gym.wrappers.RecordVideo(env, os.path.join(disk_folder, runs_directory, run_name, "videos", run_name),
                                               step_trigger=lambda x: x % args.save_video_every_n_steps == 0, video_length=args.save_video_every_n_steps)
            env.action_space.seed(seed)
            # Reward scaling.
            env = gym.wrappers.TransformReward(env, lambda reward: reward * args.reward_scale)
            return env
        return _init

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "[!] Only continuous action space is supported."
    print(f"[√] Created environment with {envs.num_envs} environments.")
    return envs


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
    
    if args.task_type == "forward":
        task = ForwardTask()
    else:
        task = BackAndForthTask(
            radius=args.radius_back_and_forth,
            origin=np.array(args.origin_back_and_forth),
        )

    envs = make_ant_envs(args=args, task=task, disk_folder='', run_name=run_name, runs_directory=args.run_dir)

    reward_tracker = RewardTracker(
        env_dt=args.dt,
        env_id="unused",
        time_window=120.0,
        log_folder=out_dir,
    )
    
    obs_dim    = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]


    actor  = Actor(obs_dim, action_dim, use_layer_norm=args.use_layer_norm).to(DEVICE)
    critic = Critic(obs_dim, use_layer_norm=args.use_layer_norm).to(DEVICE)

    action_scale = torch.tensor(
        (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
        dtype=torch.float32, device=DEVICE
    )
    action_bias = torch.tensor(
        (envs.single_action_space.high + envs.single_action_space.low) / 2.0,
        dtype=torch.float32, device=DEVICE
    )
    actor.action_scale.copy_(action_scale)
    actor.action_bias.copy_(action_bias)


    episode_returns = []

    csv_path = os.path.join(out_dir, "episode_log.csv")
    fieldnames = [
        "episode",
        "global_step",
        "ep_return",
        "avg_delta",
        "avg_abs_delta",
        "avg_v_s",
        "avg_critic_loss",
        "reward",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_file.flush()
    global_step = 0

    try:
    # reset once before the global timestep loop
        obs_np, _ = envs.reset(seed=args.seed)
        obs = torch.as_tensor(obs_np[0], dtype=torch.float32, device=DEVICE)

        episode = 0
        ep_return = 0.0
        n_steps = 0

        # per-episode metric accumulators
        delta_sum = 0.0
        abs_delta_sum = 0.0
        v_s_sum = 0.0
        critic_loss_sum = 0.0

        def write_log_row():
            if n_steps == 0:
                return

            row = {
                "episode": episode + 1,
                "global_step": global_step + 1,
                "ep_return": ep_return,
                "avg_delta": delta_sum / n_steps,
                "avg_abs_delta": abs_delta_sum / n_steps,
                "avg_v_s": v_s_sum / n_steps,
                "avg_critic_loss": critic_loss_sum / n_steps,
                "reward": reward_tracker.average_reward_per_second,
            }

            csv_writer.writerow(row)

        for global_step in range(args.total_timesteps):
             # making the obs from (111) to (1, 111) - expected input shape for actor network
            dist = actor(obs.unsqueeze(0))

            # sample raw Gaussian value, shape (1, action_dim)         
            raw = dist.sample()

            #use tanh to squash raw unbounded gaussian val to [-1, 1]              
            action = torch.tanh(raw) * actor.action_scale + actor.action_bias

            # dist.log_prob gives a log-prob PER action dimension. sum them to get total log-prob of entire action vector
            log_prob = dist.log_prob(raw).sum(-1)

            # tanh correction term
            log_prob -= torch.log(
                actor.action_scale * (1 - torch.tanh(raw).pow(2)) + 1e-6
            ).sum(-1)

            next_obs_np, reward_np, terminated_np, truncated_np, infos = envs.step(
                action.detach().cpu().numpy()
            )
            reward = float(reward_np[0])
            terminated = bool(terminated_np[0])
            truncated = bool(truncated_np[0])
            next_obs_t = torch.as_tensor(next_obs_np[0], dtype=torch.float32, device=DEVICE)
            reward_tracker.update(infos["original_reward"][0])
         
            ep_return += reward
     
            n_steps += 1

            v_s = critic(obs.unsqueeze(0)).squeeze(0)
            with torch.no_grad():
                if not terminated:
                    v_s_next = critic(next_obs_t.unsqueeze(0)).squeeze(0)
                else:
                    v_s_next = torch.tensor(0.0, device=DEVICE)
            
            
            #td_target = reward * args.dt + (args.gamma ** args.dt) * v_s_next
            td_target = reward + args.gamma * v_s_next
            delta = td_target - v_s

            delta_sum += delta.item()
            abs_delta_sum += abs(delta.item())
            v_s_sum += v_s.item()
            critic_loss_sum += 0.5 * (delta.item() ** 2)

            # critic update, zero old gradients
            critic.zero_grad()
            #take gradient of Vw(S)
            v_s.backward()

            # update critic weight
            with torch.no_grad():
                for p in critic.parameters():
                    if p.grad is not None:
                        #need to do .data to actually modify weight in place??
                        p += args.critic_lr * delta.detach() * p.grad
            
            # update actor via manual gradient ascent, maximize the objective
            actor.zero_grad()
            log_prob.backward()
            
            with torch.no_grad():
                for p in actor.parameters():
                    if p.grad is not None:
                        p += args.actor_lr * delta.detach() * p.grad
            
            # step level prints to debug
            # step level prints to debug
            if global_step % 100 == 0:
                print(
                    f"[step {global_step}] ep={episode+1} n_steps={n_steps} "
                    f"reward={reward:.4f} term={terminated} trunc={truncated} "
                    f"delta={delta.item():.4f} v_s={v_s.item():.4f}",
                    flush=True,
                )

            # CSV logging every N global training steps
            write_log_row()

            # Only flush to disk every N steps.
            if (global_step + 1) % args.log_every_n_steps == 0:
                csv_file.flush()
                os.fsync(csv_file.fileno())

                print(
                    f"[seed {args.seed}] CSV flushed | "
                    f"episode={episode+1} | global_step={global_step+1} | "
                    f"return_so_far={ep_return:.2f}",
                    flush=True,
                )
            
            
            
            # episode ending logic - ends only if env says terminated or truncated
            if terminated or truncated:
                avg_abs_delta = abs_delta_sum / n_steps
                avg_v_s = v_s_sum / n_steps

                episode_returns.append(ep_return)

                print(
                    f"[seed {args.seed}] Episode {episode+1} | "
                    f"global_step: {global_step+1} | steps: {n_steps} | "
                    f"return: {ep_return:.2f} | avg|delta|: {avg_abs_delta:.3f} | "
                    f"avg_v_s: {avg_v_s:.3f}",
                    flush=True,
                )

                if (episode + 1) % 20 == 0:
                    avg_return = np.mean(episode_returns[-20:])
                    print(
                        f"[seed {args.seed}] Episode {episode+1}, "
                        f"avg return last 20: {avg_return:.2f}",
                        flush=True,
                    )
                # Reset env and episode counters
                episode += 1
                obs_np, _ = envs.reset()
                obs = torch.as_tensor(obs_np[0], dtype=torch.float32, device=DEVICE)

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
        print(f"[seed {args.seed}] Saved log to {csv_path}", flush=True)

        weights_path = os.path.join(out_dir, "weights.pth")
        torch.save({
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "args": vars(args),
        }, weights_path)
        print(f"[seed {args.seed}] Saved weights to {weights_path}", flush=True)

        envs.close()

if __name__ == "__main__":
    main()
