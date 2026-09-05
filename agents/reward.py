import os
import csv
from collections import deque


class RewardTracker:
    def __init__(self, env_dt, env_id, time_window=10.0, log_folder=".", episode_window=100):
        self.env_dt = env_dt
        self.env_id = env_id

        self.window_size = int(time_window / env_dt)
        self.queue = deque(maxlen=self.window_size)
        self.buffer = []
        self._queue_sum = 0.0 # Running sum for efficient average calculation.

        self.episode_returns = deque(maxlen=episode_window)
        self._mean_return = None

        self.log_folder = log_folder
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        self.step = 0.0
        self._average_reward_per_second = None

        self.csv_path = os.path.join(self.log_folder, f"{self.env_id}_average_rewards.csv")
        print(f"CSV path: {self.csv_path}")
        self._csv_file_exists = os.path.exists(self.csv_path) # Cache file existence check.

    def update(self, reward):
        self.step += 1

        reward_per_second = reward / self.env_dt
        if len(self.queue) == self.window_size:
            # Queue is full, remove oldest value
            self._queue_sum -= self.queue[0]
        self.queue.append(reward_per_second)
        self._queue_sum += reward_per_second
        # Average over however many samples we have so far, so the curve
        # isn't blank for the first `window_size` steps of every run.
        self._average_reward_per_second = self._queue_sum / len(self.queue)

        self.buffer.append(
            [self.step, reward, self._average_reward_per_second, self._mean_return]
        )

    def record_episode_return(self, episode_return):
        """Record a completed episode return and refresh the running mean."""
        self.episode_returns.append(float(episode_return))
        self._mean_return = sum(self.episode_returns) / len(self.episode_returns)

    @property
    def average_reward_per_second(self):
        return self._average_reward_per_second

    @property
    def mean_return(self):
        return self._mean_return

    def log(self):
        if self.buffer:
            with open(self.csv_path, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not self._csv_file_exists:
                    writer.writerow(["step", "raw_reward", "reward", "mean_return"])
                    self._csv_file_exists = True
                writer.writerows(self.buffer)
            self.buffer.clear()
