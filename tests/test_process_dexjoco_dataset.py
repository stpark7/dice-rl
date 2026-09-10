"""End-to-end conversion checks against small raw DexJoCo Zarr recordings."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import zarr

from script.dataset.process_dexjoco_dataset import process


class ConverterTests(unittest.TestCase):
    def test_current_pose_metadata_and_failed_replay_rejection(self):
        store, state, _ = self.episode()
        store.attrs.update(observation_version='current_object_pose_v1', replay_success=True)
        state[:, 23] = np.linspace(0, 1, len(state))
        store['data/state'][:] = state[:, None, :]
        report, bc, _, norm = self.convert(observation_version='current_object_pose_v1', normalize=False)
        self.assertEqual(report['observation_version'], 'current_object_pose_v1')
        self.assertEqual(norm['observation_version'].item(), 'current_object_pose_v1')
        np.testing.assert_allclose(bc['states'][:, 23], state[14:, 23])
        store.attrs['replay_success'] = False
        with self.assertRaisesRegex(ValueError, 'No valid episodes'):
            self.convert(observation_version='current_object_pose_v1', overwrite=True)

    def test_legacy_cannot_be_converted_as_current_pose(self):
        self.episode()
        with self.assertRaisesRegex(ValueError, 'No valid episodes'):
            self.convert(observation_version='current_object_pose_v1')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.out = self.root / "delta"

    def episode(self, name="demo_0_test", onset=16, zeros=0, channel="xyz"):
        n = 32
        state = np.zeros((n, 31), dtype=np.float32)
        state[:, 0] = np.linspace(0.3, 0.4, n)
        state[:, 3] = 1
        action = np.zeros((n, 23))
        action[:, 0] = 0.5
        action[:, 3] = 1
        action[:, 7:] = 0.2
        if onset is not None:
            if channel == "xyz":
                action[onset:, 0] += 0.02
            elif channel == "rotation":
                q = Rotation.from_euler("z", 1, degrees=True).as_quat()
                action[onset:, 3:7] = q[[3, 0, 1, 2]]
            elif channel == "hand":
                action[onset:, 7] += 0.02
        rotvec = np.ones((n, 22)) * 99  # intentionally stale: training must use action
        if zeros:
            rotvec[1:zeros] = 0  # stale first command, then leading zero run
        path = self.raw / name / "replay.zarr"
        store = zarr.open(str(path), mode="w")
        data = store.create_group("data")
        for key, value in dict(state=state[:, None, :], action=action,
                               action_rotvec=rotvec, timestamp=np.arange(n) * 0.05).items():
            data.array(key, value)
        store.create_group("meta").array("episode_ends", np.array([n]))
        return store, state, action

    def convert(self, **kwargs):
        process("click_mouse", str(self.raw), str(self.out), max_episodes=kwargs.pop("max_episodes", 0),
                subset_seed=0, **kwargs)
        report = json.loads((self.out / "episodes.json").read_text())
        with np.load(self.out / "ph_pretrain/train.npz") as f:
            bc = dict(f)
        with np.load(self.out / "ph_finetune/train.npz") as f:
            rl = dict(f)
        with np.load(self.out / "ph_pretrain/normalization.npz") as f:
            norm = dict(f)
        return report, bc, rl, norm

    def test_default_delta_normalization_alignment_metadata_and_interior_hold(self):
        _, state, targets = self.episode()
        report, bc, rl, norm = self.convert()
        self.assertEqual(report["retained_start_indices"], [14])
        self.assertEqual(report["original_lengths"], [32])
        self.assertEqual(report["traj_lengths"], [18])
        self.assertEqual(norm["action_mode"].shape, ())
        self.assertEqual(norm["action_mode"].item(), "delta")
        self.assertIs(norm["normalized"].item(), True)
        raw_action = (bc["actions"] + 1) / 2 * (norm["action_max"] - norm["action_min"] + 1e-6) + norm["action_min"]
        raw_state = (bc["states"] + 1) / 2 * (norm["obs_max"] - norm["obs_min"] + 1e-6) + norm["obs_min"]
        np.testing.assert_allclose(raw_state, state[14:], atol=1e-7)
        np.testing.assert_allclose(raw_action[:, :3], targets[14:, :3] - state[14:, :3], atol=1e-7)
        np.testing.assert_allclose(raw_action[:, 3:6], 0, atol=1e-7)
        np.testing.assert_allclose(raw_action[:, 6:], targets[14:, 7:], atol=1e-7)
        for key in bc:
            np.testing.assert_array_equal(bc[key], rl[key])
        self.assertEqual(float(rl["rewards"].sum()), 1)
        self.assertEqual(float(rl["terminals"][-1]), 1)
        self.assertEqual(bc["actions"].dtype, np.float32)

    def test_stale_then_zeros_jitter_and_short_excursions_do_not_trigger_onset(self):
        store, _, _ = self.episode(onset=20, zeros=4)
        a = store["data/action"][:]
        a[5:7, 0] += 0.0005
        a[9:11, 0] += 0.003  # two frames is not sustained
        a[12, 3:7] *= -1  # q and -q encode the same rotation
        store["data/action"][:] = a
        report, bc, _, _ = self.convert()
        self.assertEqual(report["retained_start_indices"], [18])
        self.assertEqual(report["selected_episode_details"][0]["zero_prefix_steps"], 4)
        self.assertEqual(report["selected_episode_details"][0]["held_prefix_steps"], 14)
        self.assertEqual(len(bc["states"]), 14)

    def test_rotation_and_hand_departure_each_trigger_prefix_cut(self):
        self.episode("demo_0_test", channel="rotation")
        self.episode("demo_1_test", channel="hand")
        report, _, _, _ = self.convert()
        self.assertEqual(report["retained_start_indices"], [14, 14])

    def test_short_prefix_is_retained_and_explicit_absolute_unnormalized_works(self):
        _, state, action = self.episode(onset=4)
        report, bc, _, norm = self.convert(normalize=False, action_mode="absolute")
        self.assertEqual(report["retained_start_indices"], [0])
        self.assertEqual(norm["action_mode"].item(), "absolute")
        self.assertIs(norm["normalized"].item(), False)
        np.testing.assert_array_equal(bc["states"], state)
        np.testing.assert_allclose(bc["actions"][:, :3], action[:, :3])
        np.testing.assert_allclose(bc["actions"][:, 6:], action[:, 7:])

    def test_reject_invalid_and_all_held_before_subset_selection(self):
        self.episode("demo_0_test", onset=None)
        bad, _, _ = self.episode("demo_1_test")
        a = bad["data/action"][:]
        a[30, 3:7] = 0
        bad["data/action"][:] = a
        bad, _, _ = self.episode("demo_2_test")
        bad["data/timestamp"][20] = float("nan")
        bad, _, _ = self.episode("demo_3_test")
        bad["data/state"][0, 0, 5] = float("nan")
        self.episode("demo_4_test")
        report, _, _, _ = self.convert(max_episodes=1)
        self.assertEqual(report["episodes"], ["demo_4_test/replay.zarr"])
        self.assertEqual(report["n_valid"], 1)
        self.assertEqual(len(report["rejected_episodes"]), 4)
        self.assertTrue(all(item["reason"] for item in report["rejected_episodes"]))

    def test_absolute_mode_rejects_nonfinite_observation_before_prefix_cut(self):
        store, _, _ = self.episode()
        store["data/state"][0, 0, 25] = float("nan")
        with self.assertRaisesRegex(ValueError, "No valid episodes"):
            self.convert(action_mode="absolute")
        report = json.loads((self.out / "episodes.json").read_text())
        self.assertIn("nonfinite", report["rejected_episodes"][0]["reason"])

    def test_absolute_mode_rejects_zero_observed_quaternion_before_prefix_cut(self):
        store, _, _ = self.episode()
        store["data/state"][0, 0, 3:7] = 0
        with self.assertRaisesRegex(ValueError, "No valid episodes"):
            self.convert(action_mode="absolute")
        report = json.loads((self.out / "episodes.json").read_text())
        self.assertIn("quaternion", report["rejected_episodes"][0]["reason"])

    def test_all_rejected_still_writes_report(self):
        self.episode(onset=None)
        with self.assertRaisesRegex(ValueError, "No valid episodes"):
            self.convert()
        report = json.loads((self.out / "episodes.json").read_text())
        self.assertEqual(report["n_selected"], 0)
        self.assertIn("sustained", report["rejected_episodes"][0]["reason"])

    def test_timestamps_control_duration_and_leading_zeros_can_start_at_zero(self):
        store, _, _ = self.episode(onset=20, zeros=4)
        store["data/action_rotvec"][0] = 0
        # 20 samples are only 0.2 seconds here: the nonzero prefix stays.
        store["data/timestamp"][:] = np.arange(32) * 0.01
        report, _, _, _ = self.convert()
        self.assertEqual(report["retained_start_indices"], [4])
        self.assertEqual(report["selected_episode_details"][0]["zero_prefix_steps"], 4)

    def test_existing_outputs_require_explicit_rerun_and_cannot_change_mode(self):
        self.episode()
        self.convert()
        original = (self.out / "ph_pretrain/train.npz").read_bytes()
        with self.assertRaises(FileExistsError):
            self.convert()
        with self.assertRaisesRegex(ValueError, "action mode"):
            self.convert(overwrite=True, action_mode="absolute")
        self.assertEqual((self.out / "ph_pretrain/train.npz").read_bytes(), original)
        report, _, _, _ = self.convert(overwrite=True)
        self.assertEqual(report["action_mode"], "delta")

    def test_configurable_motion_thresholds(self):
        self.episode()
        report, _, _, _ = self.convert(hold_min_duration=1.0, motion_position_threshold=0.005)
        self.assertEqual(report["retained_start_indices"], [0])
        self.assertEqual(report["thresholds"]["hold_min_duration"], 1.0)


if __name__ == "__main__":
    unittest.main()
