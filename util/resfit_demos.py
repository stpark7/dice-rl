"""Prepare single-step demonstrations for residual off-policy finetuning."""

import logging

import hydra
import numpy as np
import torch

from util.resfit_replay_buffer import ResFiTReplayBuffer


log = logging.getLogger(__name__)


def _observations(transitions):
    return {key: torch.stack([torch.as_tensor(item.conditions[key])
                              for item in transitions])
            for key in ("state", "rgb")}


def _cpu_array(value):
    return torch.as_tensor(value).detach().cpu().numpy()


@torch.no_grad()
def prepare_resfit_demos(model, cfg):
    """Cache frozen BC features and queued base actions without changing demos.

    Dataset horizon is one control step, independently of the BC prediction
    horizon. BC replanning follows each episode's chunk cadence, while critic
    actions always come from the actual demonstration. Terminal observations
    absent from the archive use zeros and a zero bootstrap discount.
    """
    dataset = hydra.utils.instantiate(cfg.expert_dataset)
    if dataset.horizon_steps != 1 or getattr(dataset, "use_n_step", False):
        raise ValueError("ResFiT demonstrations require horizon_steps=1 and use_n_step=false")
    lengths = np.asarray(dataset.traj_lengths, dtype=np.int64)
    total = int(lengths.sum())
    if total == 0 or len(dataset) != total:
        raise ValueError("ResFiT demonstrations must contain every frame of nonempty episodes")
    if not np.array_equal(np.asarray(dataset.indices)[:, 0], np.arange(total)):
        raise ValueError("ResFiT demonstrations must be ordered consecutively by episode")
    chunk_steps = int(cfg.bc_action_steps)
    batch_size = int(cfg.train.get("preprocess_batch_size", 64))
    if min(chunk_steps, batch_size) < 1:
        raise ValueError("bc_action_steps and preprocess_batch_size must be positive")

    features = None
    actions = None
    rewards = np.empty(total, dtype=np.float32)
    terminals = np.empty(total, dtype=np.bool_)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        transitions = [dataset[index] for index in range(start, stop)]
        encoded = _cpu_array(model.encode(_observations(transitions)))
        demo_actions = np.stack([_cpu_array(item.actions[0]) for item in transitions])
        if features is None:
            features = np.empty((total,) + encoded.shape[1:], dtype=np.float32)
            actions = np.empty((total,) + demo_actions.shape[1:], dtype=np.float32)
        features[start:stop] = encoded
        actions[start:stop] = demo_actions
        rewards[start:stop] = [float(_cpu_array(item.rewards).reshape(-1)[0])
                               for item in transitions]
        terminals[start:stop] = [bool(_cpu_array(item.dones).reshape(-1)[0])
                                 for item in transitions]

    # Generate BC chunks only at episode-local replanning boundaries. Batching
    # chunk starts avoids a separate frozen-policy inference for every frame.
    chunk_starts = []
    chunk_stops = []
    episode_start = 0
    for length in lengths:
        episode_stop = episode_start + int(length)
        for start in range(episode_start, episode_stop, chunk_steps):
            chunk_starts.append(start)
            chunk_stops.append(min(start + chunk_steps, episode_stop))
        episode_start = episode_stop
    base_actions = np.empty_like(actions)
    for batch_start in range(0, len(chunk_starts), batch_size):
        starts = chunk_starts[batch_start:batch_start + batch_size]
        chunks = _cpu_array(model.base_actions(_observations([dataset[index] for index in starts])))
        if chunks.ndim != 3 or chunks.shape[1] < chunk_steps:
            raise ValueError("BC prediction horizon must cover bc_action_steps")
        for offset, start in enumerate(starts):
            stop = chunk_stops[batch_start + offset]
            base_actions[start:stop] = chunks[offset, :stop - start]

    replay = ResFiTReplayBuffer(
        max_size=total, n_envs=1, n_step=cfg.train.n_step,
        gamma=cfg.train.gamma, seed=cfg.get("seed", 42) + 1,
    )
    episode_start = 0
    for length in lengths:
        episode_stop = episode_start + int(length)
        for index in range(episode_start, episode_stop):
            terminal = bool(terminals[index]) or index == episode_stop - 1
            next_features = (np.zeros_like(features[index]) if terminal else
                             features[index + 1])
            next_base = (np.zeros_like(base_actions[index]) if terminal else
                         base_actions[index + 1])
            replay.add(features[index:index + 1], base_actions[index:index + 1],
                       actions[index:index + 1], rewards[index:index + 1],
                       next_features[None], next_base[None], [terminal], [False])
        episode_start = episode_stop
    log.info("Prepared %d ResFiT demonstration transitions from %d episodes",
             len(replay), len(lengths))
    return replay
