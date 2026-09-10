"""Convert single-arm DexJoCo Zarr demos to BC and residual-RL NPZ datasets.

Default targets are current-observation deltas: world xyz displacement, world
relative SO(3) rotation vector, and absolute Allegro joint targets (22 values).
Raw state columns retain the live environment's ordering. Leading zero-command
markers and long initial held-target prefixes are removed before subset selection.
Timestamps are seconds; held commands do not imply physical stationarity.

Outputs under data_dir/dexjoco/<task>_delta_n<N> are ph_pretrain and ph_finetune
train.npz / normalization.npz, plus episodes.json with selection and cut provenance.
Use --action_mode absolute for legacy absolute xyz/rotvec/hand targets.
"""

import argparse
import datetime
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import zarr
from tqdm import tqdm

# Also support direct invocation as `python script/dataset/process_...py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from util.dexjoco_actions import encode_actions, normalize as normalize_values

ACTION_DIM = 22
EXPECTED_OBS_DIM = {"pinch_tongs": 31, "pick_bucket": 38, "hammer_nail": 38, "click_mouse": 31}
DATASET_FOLDERS = ("ph_pretrain", "ph_finetune")
OBSERVATION_VERSIONS = ('initial_object_pose_v1', 'current_object_pose_v1')


@dataclass
class _Episode:
    states: np.ndarray
    actions: np.ndarray
    details: dict


def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(
        save_dir, f"process_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler()],
        force=True,
    )
    return log_path


def _demo_sort_key(name):
    """Sort raw demo dirs by (demo index, timestamp), as before."""
    match = re.search(r"demo_(\d+)_", name)
    return (int(match.group(1)) if match else 10**9, name)


def list_raw_episodes(raw_path):
    candidates = []
    for name in sorted(os.listdir(raw_path), key=_demo_sort_key):
        path = os.path.join(raw_path, name)
        if name.endswith(".zarr"):
            candidates.append(path)
        elif os.path.isdir(path) and os.path.exists(os.path.join(path, "replay.zarr")):
            candidates.append(os.path.join(path, "replay.zarr"))
    if not candidates:
        raise FileNotFoundError(f"No per-episode *.zarr / */replay.zarr under {raw_path}")
    return candidates


def load_episode(zarr_path, action_mode="delta", *, motion_position_threshold=0.001,
                 motion_rotation_threshold_deg=0.5, motion_hand_threshold=0.01,
                 motion_consecutive_samples=3, hold_min_duration=0.5,
                 hold_context_duration=0.1, expected_obs_dim=None):
    """Validate a complete episode, remove only its prefix, and encode targets."""
    states, targets, marker_actions, timestamps = _read_episode(zarr_path, expected_obs_dim)
    # Validate and encode all targets, including frames later cut from the prefix.
    actions = encode_actions(states, targets, action_mode=action_mode)
    details = _prefix_details(
        targets, marker_actions, timestamps,
        motion_position_threshold=motion_position_threshold,
        motion_rotation_threshold_deg=motion_rotation_threshold_deg,
        motion_hand_threshold=motion_hand_threshold,
        motion_consecutive_samples=motion_consecutive_samples,
        hold_min_duration=hold_min_duration,
        hold_context_duration=hold_context_duration,
    )
    start = details["retained_start_index"]
    return states[start:].astype(np.float32), actions[start:].astype(np.float32), details


def _read_episode(zarr_path, expected_obs_dim):
    """Read and validate the full recording before any prefix is discarded."""
    store = zarr.open(zarr_path, mode="r")
    data = store["data"]
    states = np.asarray(data["state"][:])
    targets = np.asarray(data["action"][:])
    marker_actions = np.asarray(data["action_rotvec"][:])
    timestamps = np.asarray(data["timestamp"][:], dtype=np.float64)
    if states.ndim == 3 and states.shape[1] == 1:
        states = states[:, 0, :]
    if states.ndim != 2 or states.shape[1] < 23:
        raise ValueError(f"invalid state shape {states.shape}; expected (T, D>=23)")
    length = len(states)
    if not length:
        raise ValueError("empty episode")
    if expected_obs_dim is not None and states.shape[1] != expected_obs_dim:
        raise ValueError(f"state dimension {states.shape[1]} != expected {expected_obs_dim}")
    # Observations are training inputs even when absolute target encoding does
    # not consume them. Validate before cutting so bad prefix data is reported.
    if not np.isfinite(states).all():
        raise ValueError("states contains nonfinite values")
    if np.any(np.linalg.norm(states[:, 3:7].astype(np.float64), axis=1) < 1e-8):
        raise ValueError("Observed pose quaternion has zero norm")
    if targets.shape != (length, 23):
        raise ValueError(f"invalid raw action shape {targets.shape}; expected ({length}, 23)")
    if marker_actions.shape != (length, ACTION_DIM):
        raise ValueError(f"invalid action_rotvec shape {marker_actions.shape}")
    if timestamps.shape != (length,) or not np.isfinite(timestamps).all():
        raise ValueError("timestamps must be finite and have shape (T,)")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if not np.isfinite(marker_actions).all():
        raise ValueError("action_rotvec contains nonfinite values")
    if "meta" in store and "episode_ends" in store["meta"]:
        ends = np.asarray(store["meta/episode_ends"][:])
        if ends.shape != (1,) or ends[0] != length:
            raise ValueError(f"expected one episode ending at {length}, got {ends}")
    return states, targets, marker_actions, timestamps


def _leading_zero_prefix(marker_actions):
    """Return the first usable target index, rejecting interior zero markers."""
    length = len(marker_actions)
    # Keep the historical t=0 or stale t=0 + t=1 zero-marker cleanup. These
    # markers are only used for cleanup; training targets always use raw quats.
    zero = np.all(marker_actions == 0, axis=1)
    zero_cut = 0
    if zero.any():
        first = int(np.argmax(zero))
        if first <= 1:
            zero_cut = first
            while zero_cut < length and zero[zero_cut]:
                zero_cut += 1
    if zero_cut == length:
        raise ValueError("only leading zero-action markers; no valid target remains")
    if zero[zero_cut:].any():
        raise ValueError("all-zero action_rotvec markers outside the leading prefix")
    return zero_cut


def _motion_onset(targets, position_threshold, rotation_threshold_deg,
                  hand_threshold, consecutive_samples):
    """Find sustained target departure relative to the first usable target."""
    xyz_departed = (
        np.linalg.norm(targets[:, :3] - targets[0, :3], axis=1) > position_threshold
    )
    rotations = Rotation.from_quat(targets[:, [4, 5, 6, 3]])
    angles = (rotations * rotations[0].inv()).magnitude()
    rot_departed = angles > np.deg2rad(rotation_threshold_deg)
    hand_departed = (
        np.max(np.abs(targets[:, 7:] - targets[0, 7:]), axis=1) > hand_threshold
    )
    departed = xyz_departed | rot_departed | hand_departed
    run = 0
    for index, moved in enumerate(departed):
        run = run + 1 if moved else 0
        if run >= consecutive_samples:
            return index - consecutive_samples + 1
    raise ValueError("no sustained target motion after the leading zero prefix")


def _prefix_details(targets, marker_actions, timestamps, *,
                    motion_position_threshold, motion_rotation_threshold_deg,
                    motion_hand_threshold, motion_consecutive_samples,
                    hold_min_duration, hold_context_duration):
    """Choose a timestamp-based prefix cut and record its provenance."""
    length = len(targets)
    zero_cut = _leading_zero_prefix(marker_actions)
    onset = zero_cut + _motion_onset(
        targets[zero_cut:], motion_position_threshold, motion_rotation_threshold_deg,
        motion_hand_threshold, motion_consecutive_samples,
    )

    held_duration = float(timestamps[onset] - timestamps[zero_cut])
    start = zero_cut
    cut_reasons = ["leading_zero_action_markers"] if zero_cut else []
    if held_duration + 1e-12 >= hold_min_duration:
        # Select the last sample at/before the requested context boundary, so
        # irregular sampling still retains at least the requested context.
        boundary = timestamps[onset] - hold_context_duration
        start = max(zero_cut, int(np.searchsorted(timestamps, boundary + 1e-12, side="right")) - 1)
        if start > zero_cut:
            cut_reasons.append("initial_held_target_prefix")
    return dict(
        original_length=length,
        zero_prefix_steps=zero_cut,
        held_prefix_steps=start - zero_cut,
        retained_start_index=start,
        retained_start_time=float(timestamps[start]),
        retained_length=length - start,
        motion_onset_index=onset,
        motion_onset_time=float(timestamps[onset]),
        held_prefix_duration=held_duration,
        cut_reasons=cut_reasons,
        prefix_decision=("trimmed" if start > zero_cut else "held_prefix_short_or_context_preserved"),
    )


def process(task, raw_path, save_root, max_episodes=50, subset_seed=0, normalize=True,
            action_mode="delta", *, motion_position_threshold=0.001,
            motion_rotation_threshold_deg=0.5, motion_hand_threshold=0.01,
            motion_consecutive_samples=3, hold_min_duration=0.5,
            hold_context_duration=0.1, overwrite=False,
            observation_version='initial_object_pose_v1'):
    """Validate recordings, select a reproducible subset, and save both datasets."""
    if action_mode not in ("delta", "absolute"):
        raise ValueError(f"unsupported action_mode: {action_mode}")
    if observation_version not in OBSERVATION_VERSIONS:
        raise ValueError(f'Unsupported observation version: {observation_version}')
    thresholds = dict(
        motion_position_threshold=motion_position_threshold,
        motion_rotation_threshold_deg=motion_rotation_threshold_deg,
        motion_hand_threshold=motion_hand_threshold,
        motion_consecutive_samples=motion_consecutive_samples,
        hold_min_duration=hold_min_duration,
        hold_context_duration=hold_context_duration,
    )
    _validate_thresholds(thresholds)
    _check_output(raw_path, save_root, action_mode, overwrite, observation_version)
    log_path = setup_logging(save_root)
    logging.info(f"Log file: {log_path}")
    logging.info(f"task={task} action_mode={action_mode} raw_path={raw_path} save_root={save_root}")

    all_paths = list_raw_episodes(raw_path)
    valid, rejected = _collect_episodes(all_paths, raw_path, task, action_mode, thresholds, observation_version)
    selected = _select_episodes(valid, max_episodes, subset_seed)
    report = _build_report(
        task, raw_path, action_mode, normalize, subset_seed, len(all_paths),
        valid, selected, rejected, thresholds,
    )
    report['observation_version'] = observation_version
    with (Path(save_root) / "episodes.json").open("w") as file:
        json.dump(report, file, indent=2)
    if not selected:
        raise ValueError(f"No valid episodes; rejection reasons saved in {save_root}/episodes.json")

    _save_datasets(save_root, selected, action_mode, normalize, observation_version)
    logging.info(
        f"Saved {len(selected)}/{len(valid)} valid episodes ({len(rejected)} rejected), "
        f"{sum(report['traj_lengths'])} transitions"
    )
    return report


def _validate_thresholds(thresholds):
    for name, value in thresholds.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    consecutive_samples = thresholds["motion_consecutive_samples"]
    if consecutive_samples < 1 or int(consecutive_samples) != consecutive_samples:
        raise ValueError("motion_consecutive_samples must be a positive integer")


def _check_output(raw_path, save_root, action_mode, overwrite, observation_version):
    """Prevent source/output overlap and incompatible output reuse before writing."""
    output = Path(save_root).resolve()
    source = Path(raw_path).resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("save_root must be separate from the raw dataset")
    existing = [output / name for name in ("episodes.json", *DATASET_FOLDERS)]
    if not any(path.exists() for path in existing):
        return
    if not overwrite:
        raise FileExistsError(f"Output exists: {save_root}; use --overwrite for an explicit rerun")
    report_path = output / "episodes.json"
    if report_path.exists():
        with report_path.open() as file:
            previous = json.load(file)
            _check_action_mode(previous.get("action_mode", "absolute"), action_mode)
            if previous.get('observation_version', OBSERVATION_VERSIONS[0]) != observation_version:
                raise ValueError('Cannot overwrite output with a different observation version')
    for folder in DATASET_FOLDERS:
        archive = output / folder / "normalization.npz"
        if archive.exists():
            with np.load(archive) as saved:
                previous_mode = str(saved["action_mode"].item()) if "action_mode" in saved else "absolute"
                previous_version = str(saved['observation_version'].item()) if 'observation_version' in saved else OBSERVATION_VERSIONS[0]
                if previous_version != observation_version:
                    raise ValueError('Cannot overwrite output with a different observation version')
            _check_action_mode(previous_mode, action_mode)


def _check_action_mode(previous_mode, action_mode):
    if previous_mode != action_mode:
        raise ValueError(
            f"Cannot overwrite output with a different action mode ({previous_mode} -> {action_mode})"
        )


def _collect_episodes(all_paths, raw_path, task, action_mode, thresholds, observation_version):
    """Clean every recording so invalid episodes never enter subset selection."""
    valid, rejected = [], []
    for path in tqdm(all_paths, desc="Validate and clean episodes"):
        relative_path = os.path.relpath(path, raw_path)
        try:
            store = zarr.open(path, mode='r')
            version = store.attrs.get('observation_version', OBSERVATION_VERSIONS[0])
            if version != observation_version:
                raise ValueError(f'Observation version {version} != expected {observation_version}')
            if version == 'current_object_pose_v1' and store.attrs.get('replay_success') is not True:
                raise ValueError('Replay did not succeed; excluded from successful BC demonstrations')
            state, action, details = load_episode(
                path, action_mode=action_mode, expected_obs_dim=EXPECTED_OBS_DIM.get(task), **thresholds
            )
            details["path"] = relative_path
            if version == 'current_object_pose_v1':
                details['source_path'] = store.attrs.get('source_path')
                details['replay_seed'] = store.attrs.get('replay_seed')
            valid.append(_Episode(state, action, details))
        except (ValueError, KeyError, IndexError, TypeError, OSError) as error:
            rejected.append(dict(path=relative_path, reason=str(error)))
            logging.warning(f"Rejected {relative_path}: {error}")
    return valid, rejected


def _select_episodes(valid, max_episodes, subset_seed):
    """Sample without replacement while preserving the raw episode order."""
    if max_episodes is not None and 0 < max_episodes < len(valid):
        chosen = np.sort(np.random.RandomState(subset_seed).choice(len(valid), size=max_episodes, replace=False))
        return [valid[index] for index in chosen]
    return valid


def _build_report(task, raw_path, action_mode, normalize, subset_seed, n_available,
                  valid, selected, rejected, thresholds):
    details = [episode.details for episode in selected]
    return dict(
        task=task, raw_path=os.path.abspath(raw_path), action_mode=action_mode,
        normalized=bool(normalize), subset_seed=subset_seed,
        n_available=n_available, n_valid=len(valid), n_selected=len(selected),
        episodes=[item["path"] for item in details],
        traj_lengths=[item["retained_length"] for item in details],
        original_lengths=[item["original_length"] for item in details],
        retained_start_indices=[item["retained_start_index"] for item in details],
        retained_start_times=[item["retained_start_time"] for item in details],
        trimmed_leading_idle_steps={item["path"]: item["zero_prefix_steps"] for item in details if item["zero_prefix_steps"]},
        selected_episode_details=details,
        valid_episode_details=[episode.details for episode in valid],
        rejected_episodes=rejected, thresholds=thresholds,
        timestamp_source="data/timestamp", timestamp_units="seconds",
    )


def _save_datasets(save_root, selected, action_mode, normalize, observation_version):
    """Share normalization and trajectories across BC and residual-RL outputs."""
    states = np.concatenate([episode.states for episode in selected])
    actions = np.concatenate([episode.actions for episode in selected])
    obs_min, obs_max = states.min(axis=0), states.max(axis=0)
    action_min, action_max = actions.min(axis=0), actions.max(axis=0)
    if normalize:
        states = normalize_values(states, obs_min, obs_max).astype(np.float32)
        actions = normalize_values(actions, action_min, action_max).astype(np.float32)
    normalization = dict(obs_min=obs_min, obs_max=obs_max, action_min=action_min,
                         action_max=action_max, action_mode=np.asarray(action_mode),
                         normalized=np.asarray(bool(normalize)),
                         observation_version=np.asarray(observation_version))
    lengths = np.asarray([episode.details["retained_length"] for episode in selected], dtype=np.int64)
    shared = dict(states=states, actions=actions, traj_lengths=lengths)
    rewards = np.zeros(len(states), dtype=np.float32)
    rewards[np.cumsum(lengths) - 1] = 1.0
    for folder, arrays in (
        ("ph_pretrain", shared),
        ("ph_finetune", dict(shared, rewards=rewards, terminals=rewards.copy())),
    ):
        directory = os.path.join(save_root, folder)
        os.makedirs(directory, exist_ok=True)
        np.savez_compressed(os.path.join(directory, "train.npz"), **arrays)
        np.savez_compressed(os.path.join(directory, "normalization.npz"), **normalization)
    logging.info(f"Trajectory length mean/std/min/max: {lengths.mean():.1f}/{lengths.std():.1f}/{lengths.min()}/{lengths.max()}")
    logging.info(f"obs_dim={states.shape[1]} action_dim={actions.shape[1]} normalized={normalize}")


def main(argv=None):
    """Parse CLI options and resolve the historical default dataset paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=list(EXPECTED_OBS_DIM))
    parser.add_argument("--raw_path", default=None,
                        help="Raw demo directory (default: ../dexjoco/datasets/raw/dexjoco_raw_datasets/<task>)")
    parser.add_argument("--save_dir", default=None,
                        help="Output root (default: data_dir/dexjoco/<task>_delta_n<N>; absolute mode omits _delta)")
    parser.add_argument("--max_episodes", type=int, default=50, help="Subset size; <=0 selects all valid episodes")
    parser.add_argument("--subset_seed", type=int, default=0)
    parser.add_argument("--action_mode", choices=["delta", "absolute"], default="delta")
    parser.add_argument('--observation_version', choices=OBSERVATION_VERSIONS,
                        default=OBSERVATION_VERSIONS[0])
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly rerun conversion into an existing output root")
    parser.add_argument("--motion_position_threshold", type=float, default=0.001, help="Target xyz departure in meters")
    parser.add_argument("--motion_rotation_threshold_deg", type=float, default=0.5)
    parser.add_argument("--motion_hand_threshold", type=float, default=0.01, help="Any hand target departure in radians")
    parser.add_argument("--motion_consecutive_samples", type=int, default=3)
    parser.add_argument("--hold_min_duration", type=float, default=0.5, help="Minimum held prefix to trim, in seconds")
    parser.add_argument("--hold_context_duration", type=float, default=0.1, help="Context retained before motion, in seconds")
    args = vars(parser.parse_args(argv))
    task = args["task"]
    args["raw_path"] = args["raw_path"] or os.path.join(
        "..", "dexjoco", "datasets", "raw", "dexjoco_raw_datasets", task
    )
    suffix = "_delta" if args["action_mode"] == "delta" else ""
    if args['observation_version'] == 'current_object_pose_v1':
        suffix += '_current_pose'
    args["save_root"] = args.pop("save_dir") or os.path.join(
        "data_dir", "dexjoco", f"{task}{suffix}_n{args['max_episodes']}"
    )
    args["normalize"] = not args.pop("no_normalize")
    process(**args)


if __name__ == "__main__":
    main()
