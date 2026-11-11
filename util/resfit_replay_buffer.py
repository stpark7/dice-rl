"""Single-control-step replay and queued BC actions for ResFiT.

Observations in replay are frozen BC visual features plus proprioception. The
actual queued base actions travel with transitions, preserving the action-chunk
phase when a random replay batch is used to train the residual controller.
"""

from collections import deque

import numpy as np
import torch


class BaseActionQueue:
    """Keep a separate frozen-policy action chunk for each vector environment."""

    def __init__(self, model, n_envs, chunk_steps, action_dim, device):
        if min(n_envs, chunk_steps, action_dim) < 1:
            raise ValueError("Queue dimensions must be positive")
        self.model = model
        self.n_envs = n_envs
        self.chunk_steps = chunk_steps
        self.action_dim = action_dim
        self.device = torch.device(device)
        self._chunks = torch.empty(n_envs, chunk_steps, action_dim,
                                   dtype=torch.float32, device=self.device)
        self._positions = np.full(n_envs, chunk_steps, dtype=np.int64)

    def reset(self, indices=None):
        """Discard completed episodes' chunks without disturbing other workers."""
        if indices is None:
            self._positions.fill(self.chunk_steps)
        else:
            self._positions[np.asarray(indices, dtype=np.int64)] = self.chunk_steps

    @torch.no_grad()
    def next(self, obs, indices=None):
        """Pop selected workers' actions from a full vector-observation batch.

        Selection is useful after auto-reset: a fresh first action can be
        fetched for ended episodes without advancing ongoing episodes twice.
        """
        active = (np.arange(self.n_envs) if indices is None else
                  np.asarray(indices, dtype=np.int64))
        exhausted = active[self._positions[active] >= self.chunk_steps]
        if len(exhausted):
            selected = {}
            for key, value in obs.items():
                if isinstance(value, torch.Tensor):
                    selected[key] = value[torch.as_tensor(exhausted, device=value.device)]
                else:
                    selected[key] = np.asarray(value)[exhausted]
            chunks = torch.as_tensor(self.model.base_actions(selected),
                                     dtype=torch.float32, device=self.device)
            if (chunks.ndim != 3 or chunks.shape[0] != len(exhausted)
                    or chunks.shape[1] < self.chunk_steps
                    or chunks.shape[2] != self.action_dim):
                raise ValueError("BC output must have shape (batch, horizon >= chunk_steps, action_dim)")
            self._chunks[torch.as_tensor(exhausted, device=self.device)] = chunks[:, :self.chunk_steps]
            self._positions[exhausted] = 0
        env_indices = torch.as_tensor(active, device=self.device)
        positions = torch.as_tensor(self._positions[active], device=self.device)
        action = self._chunks[env_indices, positions].clone()
        self._positions[active] += 1
        return action


def _numpy(value, dtype=np.float32):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


class ResFiTReplayBuffer:
    """Bounded CPU ring with per-worker n-step queues and timeout bootstrapping."""

    def __init__(self, max_size, n_envs, n_step=3, gamma=.99, seed=42):
        if min(max_size, n_envs, n_step) < 1:
            raise ValueError("Replay capacity, environment count and n_step must be positive")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be between zero and one")
        self.max_size = int(max_size)
        self.n_envs = int(n_envs)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self._pending = [deque() for _ in range(self.n_envs)]
        self._storage = None
        self._position = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self):
        return self._size

    def add(self, obs, base_action, action, reward, next_obs,
            next_base_action, terminated, truncated):
        """Add one step per worker; next_obs must precede any auto-reset.

        Episode endings flush shorter n-step windows. Only actual termination
        zeros the bootstrap discount; a time limit uses its true final state.
        """
        values = dict(obs=_numpy(obs), base_action=_numpy(base_action),
                      action=_numpy(action), reward=_numpy(reward).reshape(-1, 1),
                      next_obs=_numpy(next_obs),
                      next_base_action=_numpy(next_base_action))
        terminals = _numpy(terminated, np.bool_).reshape(-1)
        timeouts = _numpy(truncated, np.bool_).reshape(-1)
        if any(value.shape[0] != self.n_envs for value in values.values()):
            raise ValueError("Replay inputs must have n_envs rows")
        if len(terminals) != self.n_envs or len(timeouts) != self.n_envs:
            raise ValueError("Episode flags must have n_envs entries")
        for worker, pending in enumerate(self._pending):
            item = {key: value[worker].copy() for key, value in values.items()}
            item["terminated"] = bool(terminals[worker])
            pending.append(item)
            if len(pending) >= self.n_step:
                self._store(pending)
                pending.popleft()
            if terminals[worker] or timeouts[worker]:
                while pending:
                    self._store(pending)
                    pending.popleft()

    def _store(self, pending):
        first, last = pending[0], pending[-1]
        reward = sum(self.gamma ** step * transition["reward"]
                     for step, transition in enumerate(pending))
        item = dict(obs=first["obs"], base_action=first["base_action"],
                    action=first["action"], reward=reward,
                    next_obs=last["next_obs"],
                    next_base_action=last["next_base_action"],
                    discount=np.array([0. if last["terminated"] else
                                       self.gamma ** len(pending)], dtype=np.float32))
        if self._storage is None:
            self._storage = {key: np.empty((self.max_size,) + value.shape,
                                          dtype=np.float32)
                             for key, value in item.items()}
        for key, value in item.items():
            self._storage[key][self._position] = value
        self._position = (self._position + 1) % self.max_size
        self._size = min(self._size + 1, self.max_size)

    def sample(self, batch_size, device="cpu"):
        """Return detached float32 tensors, retaining scalar columns as (B, 1)."""
        if self._size == 0:
            raise ValueError("Cannot sample an empty ResFiT replay buffer")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        indices = self._rng.choice(self._size, size=batch_size,
                                   replace=self._size < batch_size)
        return {key: torch.as_tensor(values[indices], device=device)
                for key, values in self._storage.items()}
