"""
Process the DexJoCo `pinch_tongs` raw demonstrations (zarr) into the custom
.npz format consumed by DICE-RL, mirroring `process_pusht_dataset.py`.

Outputs (under `--save_dir`, default data_dir/dexjoco/pinch_tongs/):

  ph_pretrain/{train.npz, val.npz, normalization.npz}   # BC flow-matching prior
  ph_finetune/{train.npz, normalization.npz}            # residual-RL expert / RLPD

NPZ keys (already normalized to [-1, 1], the datasets do NOT normalize):
  states       (sum_T, obs_dim)   float32   obs_dim = 31 (or 23 with --proprio_only)
  actions      (sum_T, 22)        float32   action_rotvec = [xyz(3), rotvec(3), allegro(16)]
  traj_lengths (E,)               int64
  rewards      (sum_T,)           float32   sparse: 1.0 at the success step, else 0  (finetune)
  terminals    (sum_T,)           float32   1.0 at the success step, else 0          (finetune)

normalization.npz: obs_min, obs_max (obs_dim,), action_min, action_max (22,).

-----------------------------------------------------------------------------
STATE LAYOUT (must match the live env `DexjocoObsAdapter`)
-----------------------------------------------------------------------------
The live env flattens obs["state"] to 31-dim in this order:

    [ tcp_pose(7), allegro_qpos(16), tongs_ori_pose(7), table_delta_height(1) ]

The raw demo `data/state` stores proprio `[tcp_pose(7), allegro_qpos(16)]` in its
first 23 columns; `tongs_ori_pose(7)` and `table_delta_height(1)` live somewhere
in the privileged tail.  The slices below encode where -- VERIFY THESE INDICES
against a live `env.reset()` observation on a machine that has DexJoCo installed
(see Phase 1 verification).  Override at the CLI with --tongs_slice / --table_slice,
or drop the privileged dims entirely with --proprio_only (obs_dim becomes 23).
"""

import argparse
import datetime
import logging
import os

import numpy as np
import zarr
from tqdm import tqdm

# --- demo `data/state` column layout (0-indexed). CONFIRM AGAINST LIVE ENV. ---
PROPRIO_SLICE = slice(0, 23)        # [tcp_pose(7), allegro_qpos(16)]
TONGS_ORI_POSE_SLICE = slice(23, 30)  # tongs_ori_pose(7)   <-- best-effort default
TABLE_DELTA_HEIGHT_SLICE = slice(30, 31)  # table_delta_height(1) <-- best-effort default

ACTION_DIM = 22  # action_rotvec: [xyz(3), rotvec(3), allegro(16)]


def setup_logging(save_dir, save_name_prefix=""):
    """Log to file and console (matches the pusht/robomimic scripts)."""
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(
        save_dir,
        save_name_prefix
        + f"process_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log",
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
    )
    return log_path


def _parse_slice(spec):
    """Parse 'start:stop' into a slice."""
    start, stop = spec.split(":")
    return slice(int(start), int(stop))


def load_raw_episodes(raw_path):
    """
    Load raw demonstrations as (states, actions_rotvec, episode_ends).

    Supports two layouts:
      1. A single combined zarr ReplayBuffer (like pusht): `data/state`,
         `data/action_rotvec`, `meta/episode_ends`.
      2. A directory containing many per-episode zarr stores (each a `*.zarr`
         or `*/replay.zarr` with `data/state` + `data/action_rotvec`); episodes
         are concatenated and `episode_ends` is built from their lengths.
    """
    # Case 1: combined buffer with meta/episode_ends.
    try:
        root = zarr.open(raw_path, "r")
        if "meta" in root and "episode_ends" in root["meta"]:
            states = root["data"]["state"][:]
            actions = root["data"]["action_rotvec"][:]
            episode_ends = root["meta"]["episode_ends"][:]
            return states, actions, np.asarray(episode_ends, dtype=np.int64)
    except Exception:
        pass

    # Case 2: directory of per-episode zarr stores.
    if os.path.isdir(raw_path):
        candidates = []
        for name in sorted(os.listdir(raw_path)):
            p = os.path.join(raw_path, name)
            if name.endswith(".zarr"):
                candidates.append(p)
            elif os.path.isdir(p) and os.path.exists(os.path.join(p, "replay.zarr")):
                candidates.append(os.path.join(p, "replay.zarr"))
        if not candidates:
            raise FileNotFoundError(
                f"No combined zarr (meta/episode_ends) and no per-episode "
                f"*.zarr / */replay.zarr found under {raw_path}"
            )
        states_list, actions_list, ends, running = [], [], [], 0
        for c in candidates:
            r = zarr.open(c, "r")
            s = r["data"]["state"][:]
            a = r["data"]["action_rotvec"][:]
            states_list.append(s)
            actions_list.append(a)
            running += len(s)
            ends.append(running)
        return (
            np.concatenate(states_list, axis=0),
            np.concatenate(actions_list, axis=0),
            np.asarray(ends, dtype=np.int64),
        )

    raise FileNotFoundError(f"Could not interpret raw_path: {raw_path}")


def build_state(raw_state, proprio_only):
    """Reconstruct the 31-dim (or 23-dim) state in DexjocoObsAdapter order."""
    proprio = raw_state[:, PROPRIO_SLICE]
    if proprio_only:
        return proprio.astype(np.float32)
    tongs = raw_state[:, TONGS_ORI_POSE_SLICE]
    table = raw_state[:, TABLE_DELTA_HEIGHT_SLICE]
    return np.concatenate([proprio, tongs, table], axis=1).astype(np.float32)


def process_dexjoco_dataset(
    raw_path,
    save_root,
    max_episodes=-1,
    val_split=0.1,
    proprio_only=False,
    normalize=True,
):
    log_path = setup_logging(save_root)
    logging.info(f"Log file: {log_path}")
    logging.info(f"Processing DexJoCo pinch_tongs from: {raw_path}")
    logging.info(f"Saving under: {save_root}")
    logging.info(
        f"proprio_only={proprio_only}  "
        f"tongs_slice={TONGS_ORI_POSE_SLICE}  table_slice={TABLE_DELTA_HEIGHT_SLICE}"
    )

    states_data, actions_data, episode_ends = load_raw_episodes(raw_path)
    logging.info(
        f"Raw shapes - state: {states_data.shape}, action_rotvec: {actions_data.shape}, "
        f"episodes: {len(episode_ends)}"
    )
    assert actions_data.shape[1] == ACTION_DIM, (
        f"expected action_rotvec dim {ACTION_DIM}, got {actions_data.shape[1]}"
    )

    if max_episodes is not None and max_episodes > 0:
        episode_ends = episode_ends[:max_episodes]

    # ----- per-trajectory extraction (sparse reward at the final/success step) -----
    all_states, all_actions, all_rewards, all_terminals, traj_lengths = [], [], [], [], []
    prev_end = 0
    for ep_idx, ep_end in enumerate(tqdm(episode_ends, desc="Episodes")):
        raw_state = states_data[prev_end:ep_end]
        raw_action = actions_data[prev_end:ep_end]
        prev_end = ep_end
        T = len(raw_state)
        if T == 0:
            continue

        state = build_state(raw_state, proprio_only)
        action = raw_action.astype(np.float32)

        # Curated human demos succeed at the episode end: place the single
        # sparse reward (and the terminal) on the last step. The QLearning
        # dataset additionally enforces done=1 at each trajectory end.
        rewards = np.zeros(T, dtype=np.float32)
        terminals = np.zeros(T, dtype=np.float32)
        rewards[-1] = 1.0
        terminals[-1] = 1.0

        all_states.append(state)
        all_actions.append(action)
        all_rewards.append(rewards)
        all_terminals.append(terminals)
        traj_lengths.append(T)

    obs_dim = all_states[0].shape[1]
    expected_dim = 23 if proprio_only else 31
    assert obs_dim == expected_dim, f"obs_dim {obs_dim} != expected {expected_dim}"

    # ----- normalization stats over ALL data (before split), like pusht -----
    states_concat = np.concatenate(all_states, axis=0)
    actions_concat = np.concatenate(all_actions, axis=0)
    obs_min, obs_max = states_concat.min(axis=0), states_concat.max(axis=0)
    action_min, action_max = actions_concat.min(axis=0), actions_concat.max(axis=0)

    logging.info("===== Basic stats =====")
    logging.info(f"Total transitions: {sum(traj_lengths)}")
    logging.info(f"Total trajectories: {len(traj_lengths)}")
    logging.info(
        f"Traj length mean/std/min/max: {np.mean(traj_lengths):.1f} / "
        f"{np.std(traj_lengths):.1f} / {np.min(traj_lengths)} / {np.max(traj_lengths)}"
    )
    logging.info(f"obs_dim={obs_dim}  action_dim={ACTION_DIM}")
    logging.info(f"obs min/max range: [{obs_min.min():.3f}, {obs_max.max():.3f}]")
    logging.info(f"action min/max range: [{action_min.min():.3f}, {action_max.max():.3f}]")

    if normalize:
        logging.info("Normalizing states and actions to [-1, 1]")
        for i in range(len(all_states)):
            all_states[i] = (
                2 * (all_states[i] - obs_min) / (obs_max - obs_min + 1e-6) - 1
            ).astype(np.float32)
            all_actions[i] = (
                2 * (all_actions[i] - action_min) / (action_max - action_min + 1e-6) - 1
            ).astype(np.float32)

    normalization = dict(
        obs_min=obs_min, obs_max=obs_max, action_min=action_min, action_max=action_max
    )

    def assemble(indices, include_rewards):
        data = dict(
            states=np.concatenate([all_states[i] for i in indices], axis=0),
            actions=np.concatenate([all_actions[i] for i in indices], axis=0),
            traj_lengths=np.array([traj_lengths[i] for i in indices], dtype=np.int64),
        )
        if include_rewards:
            data["rewards"] = np.concatenate([all_rewards[i] for i in indices], axis=0)
            data["terminals"] = np.concatenate(
                [all_terminals[i] for i in indices], axis=0
            )
        return data

    def save(out_dir, data, with_val=False, val_indices=None):
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(os.path.join(out_dir, "train.npz"), **data)
        if with_val and val_indices:
            val = assemble(val_indices, include_rewards="rewards" in data)
            np.savez_compressed(os.path.join(out_dir, "val.npz"), **val)
        np.savez_compressed(os.path.join(out_dir, "normalization.npz"), **normalization)
        logging.info(
            f"Saved {out_dir}: train traj={len(data['traj_lengths'])} "
            f"transitions={int(np.sum(data['traj_lengths']))}"
        )

    # ----- deterministic split (last val_split episodes for validation) -----
    n_eps = len(traj_lengths)
    n_val = int(n_eps * val_split)
    train_idx = list(range(n_eps - n_val))
    val_idx = list(range(n_eps - n_val, n_eps))

    # ph_pretrain: BC prior (with optional val split).
    save(
        os.path.join(save_root, "ph_pretrain"),
        assemble(train_idx, include_rewards=False),
        with_val=n_val > 0,
        val_indices=val_idx,
    )
    # ph_finetune: residual-RL expert / RLPD, all episodes, with rewards+terminals.
    save(
        os.path.join(save_root, "ph_finetune"),
        assemble(list(range(n_eps)), include_rewards=True),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_path",
        type=str,
        required=True,
        help="Combined zarr (with meta/episode_ends) OR directory of per-episode zarr stores.",
    )
    parser.add_argument(
        "--save_dir", type=str, default="data_dir/dexjoco/pinch_tongs",
        help="Root dir; ph_pretrain/ and ph_finetune/ are created underneath.",
    )
    parser.add_argument("--max_episodes", type=int, default=100)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument(
        "--proprio_only", action="store_true",
        help="Fallback: drop privileged tongs/table dims -> obs_dim=23.",
    )
    parser.add_argument(
        "--tongs_slice", type=str, default=None,
        help="Override tongs_ori_pose columns as 'start:stop' (length 7).",
    )
    parser.add_argument(
        "--table_slice", type=str, default=None,
        help="Override table_delta_height columns as 'start:stop' (length 1).",
    )
    parser.add_argument("--no_normalize", action="store_true")
    args = parser.parse_args()

    if args.tongs_slice is not None:
        TONGS_ORI_POSE_SLICE = _parse_slice(args.tongs_slice)
    if args.table_slice is not None:
        TABLE_DELTA_HEIGHT_SLICE = _parse_slice(args.table_slice)

    process_dexjoco_dataset(
        raw_path=args.raw_path,
        save_root=args.save_dir,
        max_episodes=args.max_episodes,
        val_split=args.val_split,
        proprio_only=args.proprio_only,
        normalize=not args.no_normalize,
    )
