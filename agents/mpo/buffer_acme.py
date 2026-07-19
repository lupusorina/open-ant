# Copyright notice
#
# This file contains code adapted from stable-baselines3
# (https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py)
# licensed under the MIT License.
#
# Copyright (c) 2019-2023 Antonin Raffin, Ashley Hill, Anssi Kanervisto,
# Maximilian Ernestus, Rinu Boney, Pavan Goli, and other contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


__all__ = [
    "BaseBuffer",
    "ReplayBuffer",
    "ReplayBufferSamples",
    "NStepReplayBufferSamples",
]

class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor

class NStepReplayBufferSamples(NamedTuple):
    # observations: th.Tensor           # (B, obs_dim) 
    # actions: th.Tensor                
    # next_observations: th.Tensor      
    # dones: th.Tensor                  # (B, n_step, 1)

    # # episode boundary done: termination OR truncation
    # # Used to stop retrace/n-step continuation across resets
    # episode_dones: th.Tensor          # (B, n_step, 1)

    # rewards: th.Tensor                
    # all_observations: th.Tensor       # (B, n_step, obs_dim)
    # all_next_observations: th.Tensor 
    # all_actions: th.Tensor            
    # behavior_log_probs: th.Tensor     # (B, n_step, 1)
    observations: th.Tensor       # (B, obs_dim)
    actions: th.Tensor            # (B, act_dim)
    rewards: th.Tensor            # (B, 1), collapsed n-step reward
    discounts: th.Tensor          # (B, 1), collapsed ACME discount
    next_observations: th.Tensor  # (B, obs_dim), final arrival state


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_device(device: th.device | str = "auto") -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")

    return device


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        """
        :param batch_inds:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = th.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=th.float32, device=self.device)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = th.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=th.float32, device=self.device)

        self.actions = th.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=th.float32, device=self.device)
        #self.behavior_log_probs = th.zeros((self.buffer_size, self.n_envs, 1), dtype=th.float32, device=self.device)

        self.rewards = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=self.device)
        self.dones = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=self.device)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = th.zeros((self.buffer_size, self.n_envs), dtype=th.float32, device=self.device)

        # Per-row validity: 0 marks a phantom autoreset-boundary transition that
        # must never be a sampling start index. Defaults to 1 (usable).
        self.valid = th.ones((self.buffer_size, self.n_envs), dtype=th.float32, device=self.device)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: th.Tensor,
        next_obs: th.Tensor,
        action: th.Tensor,
        reward: th.Tensor,
        done: th.Tensor,
        infos: list[dict[str, Any]],
        #behavior_log_prob: th.Tensor | None = None,
        truncation: th.Tensor | None = None,
        valid: th.Tensor | None=None,
    ) -> None:

        # Copy to avoid modification by reference
        self.observations[self.pos].copy_(obs)

        self.next_observations[self.pos].copy_(next_obs)
        self.actions[self.pos].copy_(action.view(self.n_envs, self.action_dim))
        # if behavior_log_prob is not None:
        #     self.behavior_log_probs[self.pos].copy_(behavior_log_prob.view(self.n_envs, 1))
        self.rewards[self.pos].copy_(reward.view(self.n_envs))
        self.dones[self.pos].copy_(done.view(self.n_envs))
        if valid is not None:
            self.valid[self.pos].copy_(valid.float().view(self.n_envs))
        else:
            self.valid[self.pos].fill_(1.0)
        #_______________________________________________________________________________
        if self.handle_timeout_termination:
            if truncation is not None:
                self.timeouts[self.pos].copy_(
                    truncation.to(self.device).float().view(self.n_envs)
                )
            else:
                self.timeouts[self.pos].copy_(
                    th.as_tensor(
                        [info.get("TimeLimit.truncated", False) for info in infos],
                        dtype=th.float32,
                        device=self.device,
                    ).view(self.n_envs)
                )
        #____________________________________________________________________________________
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :]
        else:
            next_obs = self.next_observations[batch_inds, env_indices, :]

        data = (
            self.observations[batch_inds, env_indices, :],
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # set "returned_dones"=0 for endings that are due to timeouts
            # this makes truncations (endings due to timeouts) bootstrapable in target = r + y*(1-dones)* V(S_t+1)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def sample_nstep(self, batch_size: int, n_step: int, gamma: float) -> "NStepReplayBufferSamples":
        if n_step < 1:
            raise ValueError("n_step must be at least 1")

        if not self.full:
            max_start = self.pos - n_step + 1
            if max_start <= 0:
                raise ValueError(f"Not enough data for {n_step}-step sampling ({self.pos} steps stored)")
        else:
            max_start = self.buffer_size - n_step + 1
            if max_start <= 0:
                raise ValueError(f"Buffer has {self.buffer_size} rows but n_step={n_step}")

        accepted_batch_inds = []
        accepted_env_inds = []
        num_accepted = 0

        while num_accepted < batch_size:
            num_missing = batch_size - num_accepted
            num_candidates = max(num_missing * 2, 32)

            if not self.full:
                candidate_batch_inds = np.random.randint(0, max_start, size=num_candidates)
            else:
                linear_inds = np.random.randint(0, max_start, size=num_candidates)
                candidate_batch_inds = (linear_inds + self.pos) % self.buffer_size

            candidate_env_inds = np.random.randint(0, self.n_envs, size=num_candidates)

            keep = self.valid[candidate_batch_inds, candidate_env_inds].bool().cpu().numpy()
            kept_batch_inds = candidate_batch_inds[keep]
            kept_env_inds = candidate_env_inds[keep]

            if kept_batch_inds.size > 0:
                accepted_batch_inds.append(kept_batch_inds)
                accepted_env_inds.append(kept_env_inds)
                num_accepted += kept_batch_inds.size

        batch_inds = np.concatenate(accepted_batch_inds)[:batch_size]
        env_inds = np.concatenate(accepted_env_inds)[:batch_size]

        return self._get_nstep_samples(batch_inds, env_inds, n_step, gamma)
        
    def _get_nstep_samples(self, batch_inds, env_inds, n_step: int, gamma: float,) -> "NStepReplayBufferSamples":
        batch_inds = th.as_tensor(batch_inds, dtype=th.long, device=self.device)
        env_inds = th.as_tensor(env_inds, dtype=th.long, device=self.device)
        offsets = th.arange(n_step, dtype=th.long, device=self.device).unsqueeze(0)
        all_inds = (batch_inds.unsqueeze(1) + offsets) % self.buffer_size
            # shape of rewards, dones, timeouts = [buffer-rows, num-envs]
            # unsqueeze(1) turns a row of env ids being sampled into 1 column
        rewards = self.rewards[all_inds, env_inds.unsqueeze(1)]
        
        # constructing each of these requires all_inds[b,k] - which supplies the time-row coordinate
        # and env_inds[b] - which supplies the environment coordinate.
        # Episode boundary = termination or truncation
        boundaries = self.dones[all_inds, env_inds.unsqueeze(1)]
        
        # whether episode boundary is a time-limit truncation
        if self.handle_timeout_termination:
            timeouts = self.timeouts[all_inds, env_inds.unsqueeze(1)]
        else:
            timeouts = th.zeros_like(boundaries)
        
        #  environment discount: termination = 0, truncation -> 1, ordinary step -> 1
        terminals = boundaries * (1.0 - timeouts)
        env_discounts = 1.0 - terminals

        # only include transition at index k if no episode boundary 
        # occurred before k.
        # i.e. boundary = [0, 0, 1, ...], then valid = [1, 1, 1, 0, ...]
        prior_boundaries = th.cat(
            [th.zeros_like(boundaries[:, :1]), boundaries[:, :-1]], dim=1)

        valid = th.cumprod(
            1.0 - prior_boundaries,
            dim=1,
        )

        # Product of environment discounts before each reward:
        # reward 0 multiplier has no preceding discount
        preceding_env_discount = th.cumprod(
            th.cat(
                [
                    th.ones_like(env_discounts[:, :1]),
                    env_discounts[:, :-1],
                ],
                dim=1,
            ),
            dim=1,
        )

        gamma_tensor = th.as_tensor(
            gamma,dtype=rewards.dtype,device=self.device)

        gamma_powers = th.pow(
            gamma_tensor,
            th.arange(
                n_step,
                dtype=rewards.dtype,
                device=self.device,
            ),
        ).unsqueeze(0)

        # n-step reward: r_0 + gamma*d_0*r_1 + ...
        collapsed_rewards = (
            valid
            * preceding_env_discount
            * gamma_powers
            * rewards
        ).sum(
            dim=1,
            keepdim=True)

        # Effective trajectory length:
        # n_step unless a termination/truncation boundary occurs first.
        horizons = valid.sum(dim=1).to(th.long)  # (B,)

        # Product of environment discounts over included transitions.
        included_env_discounts = th.where(
            valid.bool(),
            env_discounts,
            th.ones_like(env_discounts))

        env_discount_product = included_env_discounts.prod(
            dim=1,
            keepdim=True)

        # ACME stores gamma^(horizon - 1) * product(environment discounts).
        # The learner applies the final gamma.
        collapsed_discounts = (
            th.pow(
                gamma_tensor,
                horizons.to(rewards.dtype) - 1.0,
            ).unsqueeze(-1)
            * env_discount_product)

        # Select s_{t+horizon}
        last_inds = (batch_inds + horizons - 1) % self.buffer_size

        observations = self.observations[batch_inds,env_inds]

        actions = self.actions[batch_inds,env_inds]
        # self.next_obs[t+n-1] = s_t+n
        next_observations = self.next_observations[last_inds,env_inds]

        return NStepReplayBufferSamples(
            observations=observations,
            actions=actions,
            rewards=collapsed_rewards,
            discounts=collapsed_discounts,
            next_observations=next_observations,
        )

    def save(self, filepath: str) -> None:
        """
        Save the replay buffer to a file.

        :param filepath: Path to the file where the replay buffer will be saved.
        """
        #TODO:alternate save a .pt file with torch.save() --> no need for the gpu cpu overhead
        np.savez(
            filepath,
            observations=self.observations.cpu().numpy(),
            next_observations=self.next_observations.cpu().numpy(),
            actions=self.actions.cpu().numpy(),
           # behavior_log_probs=self.behavior_log_probs.cpu().numpy(),
            rewards=self.rewards.cpu().numpy(),
            dones=self.dones.cpu().numpy(),
            timeouts=self.timeouts.cpu().numpy(),
            pos=self.pos,
            full=self.full,
            valid=self.valid.cpu().numpy(),
        )

    def load(self, filepath: str, device: th.device | str = "auto") -> None:
        data = np.load(filepath)

        # Keep internal storage as numpy arrays.
        self.observations = th.from_numpy(data['observations']).to(self.device)
        self.next_observations = th.from_numpy(data['next_observations']).to(self.device)
        self.actions = th.from_numpy(data['actions']).to(self.device)
       # self.behavior_log_probs = th.from_numpy(data['behavior_log_probs']).to(self.device) if 'behavior_log_probs' in data else th.zeros_like(self.behavior_log_probs)
        self.rewards = th.from_numpy(data['rewards']).to(self.device)
        self.dones = th.from_numpy(data['dones']).to(self.device)
        self.timeouts = th.from_numpy(data['timeouts']).to(self.device)
        self.pos = int(data['pos'])
        self.full = bool(data['full'])
        self.valid = th.from_numpy(data['valid']).to(self.device) if 'valid' in data else th.ones_like(self.dones)


        # Only store device for later (used in to_torch)
        self.device = th.device(device)

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype
