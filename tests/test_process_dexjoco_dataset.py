"""End-to-end conversion checks against small raw DexJoCo Zarr recordings."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation
import zarr
import av

from script.dataset.process_dexjoco_dataset import process


class ConverterTests(unittest.TestCase):
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
        process("click_mouse", str(self.raw), str(self.out), **kwargs)
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
        self.assertEqual(report["episode_details"][0]["zero_prefix_steps"], 4)
        self.assertEqual(report["episode_details"][0]["held_prefix_steps"], 14)
        self.assertEqual(len(bc["states"]), 14)

    def test_rotation_and_hand_departure_each_trigger_prefix_cut(self):
        self.episode("demo_0_test", channel="rotation")
        self.episode("demo_1_test", channel="hand")
        report, bc, _, _ = self.convert()
        self.assertEqual(report["retained_start_indices"], [14, 14])
        self.assertEqual(report["episodes"],
                         ["demo_0_test/replay.zarr", "demo_1_test/replay.zarr"])
        self.assertEqual(len(bc["states"]), 36)

    def test_short_prefix_is_retained_and_unnormalized_output_stays_raw(self):
        _, state, action = self.episode(onset=4)
        report, bc, _, norm = self.convert(normalize=False)
        self.assertEqual(report["retained_start_indices"], [0])
        self.assertIs(norm["normalized"].item(), False)
        np.testing.assert_array_equal(bc["states"], state)
        np.testing.assert_allclose(
            bc["actions"][:, :3], action[:, :3] - state[:, :3], atol=1e-6
        )
        np.testing.assert_allclose(bc["actions"][:, 6:], action[:, 7:])

    def test_invalid_episodes_are_rejected_with_reasons(self):
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
        report, _, _, _ = self.convert()
        self.assertEqual(report["episodes"], ["demo_4_test/replay.zarr"])
        self.assertEqual(report["n_valid"], 1)
        self.assertEqual(len(report["rejected_episodes"]), 4)
        self.assertTrue(all(item["reason"] for item in report["rejected_episodes"]))

    def test_rejects_nonfinite_observation_before_prefix_cut(self):
        store, _, _ = self.episode()
        store["data/state"][0, 0, 25] = float("nan")
        with self.assertRaisesRegex(ValueError, "No valid episodes"):
            self.convert()
        report = json.loads((self.out / "episodes.json").read_text())
        self.assertIn("nonfinite", report["rejected_episodes"][0]["reason"])

    def test_rejects_zero_observed_quaternion_before_prefix_cut(self):
        store, _, _ = self.episode()
        store["data/state"][0, 0, 3:7] = 0
        with self.assertRaisesRegex(ValueError, "No valid episodes"):
            self.convert()
        report = json.loads((self.out / "episodes.json").read_text())
        self.assertIn("quaternion", report["rejected_episodes"][0]["reason"])

    def test_all_rejected_still_writes_report(self):
        self.episode(onset=None)
        with self.assertRaisesRegex(ValueError, "No valid episodes"):
            self.convert()
        report = json.loads((self.out / "episodes.json").read_text())
        self.assertEqual(report["n_valid"], 0)
        self.assertIn("sustained", report["rejected_episodes"][0]["reason"])

    def test_timestamps_control_duration_and_leading_zeros_can_start_at_zero(self):
        store, _, _ = self.episode(onset=20, zeros=4)
        store["data/action_rotvec"][0] = 0
        # 20 samples are only 0.2 seconds here: the nonzero prefix stays.
        store["data/timestamp"][:] = np.arange(32) * 0.01
        report, _, _, _ = self.convert()
        self.assertEqual(report["retained_start_indices"], [4])
        self.assertEqual(report["episode_details"][0]["zero_prefix_steps"], 4)

    def test_rerun_replaces_the_previous_output_in_place(self):
        self.episode()
        _, bc, _, _ = self.convert()
        self.assertEqual(len(bc["states"]), 18)
        report, bc, _, _ = self.convert(hold_min_duration=1.0)
        self.assertEqual(report["retained_start_indices"], [0])
        self.assertEqual(len(bc["states"]), 32)

    def test_configurable_motion_thresholds(self):
        self.episode()
        report, _, _, _ = self.convert(hold_min_duration=1.0, motion_position_threshold=0.005)
        self.assertEqual(report["retained_start_indices"], [0])
        self.assertEqual(report["thresholds"]["hold_min_duration"], 1.0)

    def videos(self, name="demo_0_test", cameras=None, length=32, blue=200):
        cameras = cameras or ["front", "wrist", "ego_left", "ego_right"]
        directory = self.raw / name / "videos"
        directory.mkdir(exist_ok=True)
        for camera_index, camera in enumerate(cameras):
            with av.open(str(directory / (camera + ".mp4")), mode="w") as container:
                stream = container.add_stream("libx264rgb", rate=20)
                stream.width, stream.height = 128, 128
                stream.pix_fmt = "rgb24"
                stream.options = {"crf": "0", "preset": "ultrafast"}
                for index in range(length):
                    frame = np.zeros((128, 128, 3), dtype=np.uint8)
                    frame[:] = [index * 4, camera_index * 50, blue]
                    for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)

    def test_images_preserve_all_cameras_trim_and_select_in_loader(self):
        from agent.dataset.sequence import StitchedSequenceDataset
        self.episode()
        self.videos()
        report, bc, _, norm = self.convert(with_images=True)
        self.assertEqual(report["image_size"], [96, 96])
        self.assertEqual(report["camera_names"], ["ego_left", "ego_right", "front", "wrist"])
        self.assertEqual(bc["states"].shape, (18, 23))
        self.assertEqual(norm["obs_min"].shape, (23,))
        images = zarr.open(str(self.out / "images.zarr"), mode="r")
        self.assertEqual(set(images.array_keys()), set(report["camera_names"]))
        self.assertEqual(images["front"].shape, (18, 3, 96, 96))
        self.assertEqual(images["front"].dtype, np.uint8)
        np.testing.assert_allclose(images["front"][0, :, 0, 0], [56, 0, 200], atol=1)
        dataset = StitchedSequenceDataset(
            str(self.out / "ph_pretrain/train.npz"), device="cpu", use_img=True,
            image_keys=["wrist", "front"], horizon_steps=4, cond_steps=2, img_cond_steps=2,
        )
        rgb = dataset[0].conditions["rgb"].numpy()
        self.assertEqual(rgb.shape, (2, 6, 96, 96))
        np.testing.assert_array_equal(rgb[0], rgb[1])
        np.testing.assert_allclose(rgb[0, :, 0, 0], [56, 50, 200, 56, 0, 200], atol=1)
        with self.assertRaisesRegex(ValueError, "camera"):
            StitchedSequenceDataset(str(self.out / "ph_pretrain/train.npz"),
                                    device="cpu", use_img=True, image_keys=["missing"])

    def test_frame_mismatch_fails_conversion(self):
        self.episode()
        self.videos(length=31)
        with self.assertRaisesRegex(ValueError, "frame"):
            self.convert(with_images=True)
        self.assertFalse((self.out / "ph_pretrain/train.npz").exists())

    def test_camera_set_mismatch_fails_conversion(self):
        self.episode()
        self.episode("demo_1_test")
        self.videos()
        self.videos("demo_1_test", cameras=["front"])
        with self.assertRaisesRegex(ValueError, "camera set"):
            self.convert(with_images=True)
        self.assertFalse((self.out / "ph_pretrain/train.npz").exists())

    def test_image_history_does_not_cross_episodes_and_legacy_images_work(self):
        from agent.dataset.sequence import StitchedSequenceDataset
        self.episode()
        self.episode("demo_1_test")
        self.videos()
        self.videos("demo_1_test", blue=100)
        self.convert(with_images=True, image_size=32)
        path = str(self.out / "ph_pretrain/train.npz")
        dataset = StitchedSequenceDataset(path, device="cpu", use_img=True,
                                          horizon_steps=4, cond_steps=3, img_cond_steps=2)
        # 18 retained samples - horizon 4 + 1 = 15 samples per episode.
        second = dataset[15].conditions["rgb"].numpy()
        np.testing.assert_array_equal(second[0], second[1])
        np.testing.assert_allclose(second[:, 2::3], 100, atol=1)
        np.testing.assert_allclose(dataset[0].conditions["rgb"][:, 2::3], 200, atol=1)
        self.assertEqual(tuple(dataset[0].conditions["rgb"].shape), (2, 12, 32, 32))
        with self.assertRaises(FileExistsError):
            self.convert(with_images=True)
        with np.load(path) as archive:
            arrays = {k: archive[k] for k in ("states", "actions", "traj_lengths")}
        images = np.arange(36 * 3 * 4 * 4, dtype=np.uint8).reshape(36, 3, 4, 4)
        legacy = str(self.out / "legacy.npz")
        np.savez(legacy, **arrays, images=images)
        old = StitchedSequenceDataset(legacy, device="cpu", use_img=True,
                                      horizon_steps=4, cond_steps=3, img_cond_steps=2)
        np.testing.assert_array_equal(old[2].conditions["rgb"], images[1:3])
        np.testing.assert_array_equal(old[15].conditions["rgb"], images[[18, 18]])

    def test_qlearning_reads_only_requested_image_history(self):
        from agent.dataset.sequence import StitchedSequenceQLearningDataset
        self.episode()
        self.videos()
        self.convert(with_images=True)
        dataset = StitchedSequenceQLearningDataset(
            str(self.out / "ph_finetune/train.npz"), device="cpu", use_img=True,
            image_keys=["front"], horizon_steps=2, cond_steps=2, img_cond_steps=2,
            get_mc_return=False,
        )

        class ReadRecorder:
            def __init__(self, images):
                self.images, self.slices = images, []

            def __getitem__(self, index):
                self.slices.append(index)
                return self.images[index]

        dataset.images = ReadRecorder(dataset.images)
        sample = dataset[8]
        self.assertTrue(all(s.stop - s.start <= 2 for s in dataset.images.slices))
        np.testing.assert_allclose(sample.conditions["rgb"][:, 0, 0, 0], [84, 88], atol=1)
        np.testing.assert_allclose(sample.conditions["next_rgb"][:, 0, 0, 0], [92, 96], atol=1)

    def test_bimanual_videos_and_left_arm_motion(self):
        store, _, _ = self.episode(onset=None)
        state = np.zeros((32, 61), dtype=np.float32)
        state[:, 3] = state[:, 10] = 1
        state[:, 7] = 0.2
        targets = np.zeros((32, 46))
        targets[:, 3] = targets[:, 26] = 1
        targets[:, 23] = 0.2
        targets[16:, 23] += 0.02  # only the left arm moves
        for key, value in dict(state=state[:, None], action=targets,
                               action_rotvec=np.ones((32, 44))).items():
            del store["data"][key]
            store["data"].array(key, value)
        self.videos(cameras=["ego", "wrist_left", "wrist_right"])
        report = process("bimanual_assembly", str(self.raw), str(self.out),
                         with_images=True, normalize=False)
        self.assertEqual(report["retained_start_indices"], [14])
        self.assertEqual(report["camera_names"], ["ego", "wrist_left", "wrist_right"])
        with np.load(self.out / "ph_pretrain/train.npz") as dataset:
            self.assertEqual(dataset["states"].shape, (18, 46))
            self.assertEqual(dataset["actions"].shape, (18, 44))
            np.testing.assert_allclose(dataset["actions"][2:, 22], 0.02, atol=1e-7)
            from util.dexjoco_actions import decode_bimanual_actions
            restored = decode_bimanual_actions(dataset["states"], dataset["actions"])
            # Recordings use [right23, left23]; policy_mode expects poses first.
            retained = targets[14:]
            expected = np.concatenate(
                [retained[:, :7], retained[:, 23:30], retained[:, 7:23], retained[:, 30:]],
                axis=1,
            )
            np.testing.assert_allclose(restored, expected, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
