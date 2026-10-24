"""Raw DexJoCo chunks and episode-separated splits for hand chord training."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _split_episode_ids(n_episodes, val_fraction, seed):
    n_validation = max(1, min(n_episodes - 1, int(round(n_episodes * val_fraction))))
    permutation = np.random.RandomState(seed).permutation(n_episodes)
    train_ids = np.sort(permutation[n_validation:]).astype(np.int64)
    validation_ids = np.sort(permutation[:n_validation]).astype(np.int64)
    return train_ids, validation_ids


class ActionChunkDataset(Dataset):
    """
    Return full ``[H, raw_action_dim]`` actions and their measured start state.
    """

    def __init__(self, actions, states, lengths, episode_ids, horizon_steps,
                 raw_action_dim=22, state_dim=23):
        """Index full windows from prepared arrays and valid episode indices."""
        if not isinstance(horizon_steps, (int, np.integer)) or horizon_steps <= 0:
            raise ValueError("horizon_steps must be a positive integer")
        # Input data is checked once in load_data; this class only indexes windows.
        self.horizon_steps = horizon_steps
        self.raw_action_dim = raw_action_dim
        self.state_dim = state_dim
        self.traj_lengths = np.asarray(lengths)
        self.episode_ids = np.asarray(episode_ids)
        self.starts = self.make_starts(self.traj_lengths, self.horizon_steps, self.episode_ids)
        if not len(self.starts):
            raise ValueError("Episode partition has no full-horizon windows")

        # Reuse float32 array storage; convert only when called directly with another dtype.
        self.actions = torch.as_tensor(actions, dtype=torch.float32)
        self.states = torch.as_tensor(states, dtype=torch.float32)

    @staticmethod
    def make_starts(lengths, horizon_steps, episode_ids):
        """Return global frame indices of full windows, in the supplied episode order."""
        episode_ends = np.cumsum(lengths)
        episode_starts = np.concatenate(([0], episode_ends[:-1]))

        starts = []
        for episode_id in episode_ids:
            episode_start = int(episode_starts[episode_id])
            episode_length = int(lengths[episode_id])
            last_valid_start = episode_start + episode_length - horizon_steps

            # Include the last start whose full chunk fits inside the episode.
            starts.extend(range(episode_start, last_valid_start + 1))

        return np.asarray(starts, dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        return self.actions[start:start + self.horizon_steps], self.states[start]


def load_data(dataset_path, horizon_steps, max_n_episodes=None, val_fraction=0.1, seed=0,
              raw_action_dim=22, state_dim=23):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    if max_n_episodes is not None and (
        not isinstance(max_n_episodes, (int, np.integer)) or max_n_episodes < 2
    ):
        raise ValueError("max_n_episodes must be an integer of at least two")

    # Read only required NPZ members; image and object arrays remain unopened.
    path = Path(dataset_path)
    with np.load(path, allow_pickle=False) as archive:
        lengths = archive["traj_lengths"]
        actions = archive["actions"].astype(np.float32)
        states = archive["states"].astype(np.float32)

    # Check lengths before casting so fractional or missing frames cannot be hidden.
    if (lengths.ndim != 1 or not np.isfinite(lengths).all()
            or np.any(lengths <= 0) or np.any(lengths != np.floor(lengths))):
        raise ValueError("Trajectory lengths must be a one-dimensional array of positive integers")
    lengths = lengths.astype(np.int64)
    total_frames = int(lengths.sum())
    if actions.shape != (total_frames, raw_action_dim):
        raise ValueError(f"actions must have shape [{total_frames}, {raw_action_dim}]")
    if states.shape != (total_frames, state_dim):
        raise ValueError(f"states must have shape [{total_frames}, {state_dim}]")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError("Actions and states must be finite")

    # Apply the episode limit before splitting and trim the matching frame arrays.
    lengths = lengths[:max_n_episodes]
    if len(lengths) < 2:
        raise ValueError("At least two episodes are required for training and validation")
    total_frames = sum(length for length in lengths)
    actions, states = actions[:total_frames], states[:total_frames]

    # Split whole episodes so training and validation windows cannot overlap.
    train_ids, validation_ids = _split_episode_ids(len(lengths), val_fraction, seed)
    train = ActionChunkDataset(actions, states, lengths, train_ids, horizon_steps,
                               raw_action_dim=raw_action_dim, state_dim=state_dim)
    validation = ActionChunkDataset(actions, states, lengths, validation_ids, horizon_steps,
                                    raw_action_dim=raw_action_dim, state_dim=state_dim)

    return train, validation
