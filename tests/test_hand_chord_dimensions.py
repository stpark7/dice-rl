"""Configured action/state layouts survive training and checkpoint round trips."""

from pathlib import Path
import tempfile
import unittest

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from agent.hand_chord.train_chunk_hand_chord_agent import TrainChunkHandChordAgent
from model.hand_chord.chunk_hand_chord import ChunkHandActionChord


ROOT = Path(__file__).resolve().parents[1]


class HandChordDimensionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.layout = dict(raw_action_dim=9, hand_dim=4, state_dim=10, hand_state_start=3)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.actions = torch.randn(2, 3, 9) * 0.1
        self.state = torch.randn(2, 10) * 0.1

    def write_normalization(self, raw_action_dim=9, state_dim=10):
        path = Path(self.temp.name) / f"normalization_{raw_action_dim}_{state_dim}.npz"
        np.savez(path, normalized=False,
                 action_min=np.full(raw_action_dim, -1.), action_max=np.full(raw_action_dim, 1.),
                 obs_min=np.full(state_dim, -1.), obs_max=np.full(state_dim, 1.))
        return str(path)

    def chord(self, **kwargs):
        kwargs.setdefault("normalization_path", self.write_normalization())
        return ChunkHandActionChord(
            horizon_steps=3, latent_dim=2, width=16, layers=1, heads=2,
            **self.layout, **kwargs,
        )

    def test_configured_layout_drives_physical_coordinates_packing_and_loss(self):
        chord = self.chord().eval()
        torch.testing.assert_close(chord.get_hand_targets(self.actions), self.actions[..., 5:9])
        torch.testing.assert_close(chord.get_initial_hand_positions(self.state), self.state[:, 3:7])
        torch.testing.assert_close(chord.joint_weights, torch.ones(4))
        packed = chord.encode(self.actions, self.state)
        self.assertEqual(tuple(packed.shape), (2, 17))
        decoded = chord.decode(packed, self.state)
        self.assertEqual(decoded.shape, self.actions.shape)
        torch.testing.assert_close(decoded[..., :5], self.actions[..., :5])
        history = torch.stack([self.state + 1, self.state], dim=1)
        reconstructed, mean, logvar = chord.reconstruct(self.actions, history)
        torch.testing.assert_close(reconstructed, decoded)
        expected_mean, expected_logvar = chord.posterior(self.actions, self.state)
        torch.testing.assert_close(mean, expected_mean)
        torch.testing.assert_close(logvar, expected_logvar)
        terms = chord.loss(self.actions, self.state)
        self.assertTrue(torch.isfinite(terms["loss"]))
        terms["loss"].backward()
        self.assertIsNotNone(chord.q0_embedding.weight.grad)
        metadata = chord.metadata()
        self.assertEqual(metadata["action_slice"], [5, 9])
        self.assertEqual(metadata["state_slice"], [3, 7])
        self.assertEqual(len(metadata["joint_names"]), 4)
        for name, value in self.layout.items():
            self.assertEqual(metadata["config"][name], value)

    def test_custom_normalizers_and_dimensions_survive_self_contained_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            normalization = Path(temp) / "normalization.npz"
            np.savez(normalization, normalized=True,
                     action_min=np.full(9, -4.), action_max=np.full(9, 4.),
                     obs_min=np.full(10, -2.), obs_max=np.full(10, 2.))
            chord = self.chord(normalization_path=normalization).eval()
            hand = (self.actions[..., 5:9] + 1) / 2 * (8 + 1e-6) - 4
            q0 = (self.state[:, 3:7] + 1) / 2 * (4 + 1e-6) - 2
            torch.testing.assert_close(chord.get_relative_hand_targets(self.actions, self.state),
                                       hand - q0[:, None])
            torch.testing.assert_close(chord.normalize_hand(hand), self.actions[..., 5:9])
            chord.set_statistics(torch.arange(4) * 0.1, torch.ones(4) * 0.4)
            packed = chord.encode(self.actions, self.state)
            decoded = chord.decode(packed, self.state)
            checkpoint = Path(temp) / "chord.pt"
            chord.save(checkpoint)
            normalization.unlink()
            restored = ChunkHandActionChord.load(checkpoint)
            self.assertEqual(restored.metadata(), chord.metadata())
            torch.testing.assert_close(restored.encode(self.actions, self.state), packed)
            torch.testing.assert_close(restored.decode(packed, self.state), decoded)

    def test_invalid_layout_and_joint_weight_dimensions_are_rejected(self):
        for changes in (dict(raw_action_dim=3), dict(hand_dim=0), dict(state_dim=6),
                        dict(hand_state_start=-1), dict(joint_weights=[1., 1.])):
            layout = dict(self.layout, **changes)
            normalization = self.write_normalization(layout["raw_action_dim"], layout["state_dim"])
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ChunkHandActionChord(normalization_path=normalization, **layout)

    def test_existing_checkpoints_without_dimension_config_still_load(self):
        with tempfile.TemporaryDirectory() as temp:
            chord = ChunkHandActionChord(
                normalization_path=self.write_normalization(22, 23),
                horizon_steps=3, latent_dim=2, width=16, layers=1, heads=2,
            ).eval()
            path = Path(temp) / "legacy.pt"
            chord.save(path)
            checkpoint = torch.load(path, weights_only=True)
            for name in self.layout:
                checkpoint["config"].pop(name, None)
            torch.save(checkpoint, path)
            restored = ChunkHandActionChord.load(path)
            actions, state = torch.randn(2, 3, 22), torch.randn(2, 23)
            torch.testing.assert_close(restored.encode(actions, state), chord.encode(actions, state))

    def test_task_yaml_explicitly_configures_model_dimensions(self):
        for task in ("pick_bucket", "hammer_nail"):
            with self.subTest(task=task):
                cfg = OmegaConf.load(ROOT / "cfg/dexjoco/hand_chord/pretrain" / task / "pre_chunk_hand_chord.yaml")
                self.assertEqual(cfg.model.raw_action_dim, 22)
                self.assertEqual(cfg.model.hand_dim, 16)
                self.assertEqual(cfg.model.state_dim, 23)
                self.assertEqual(cfg.model.hand_state_start, 7)
                cfg.model.normalization_path = self.write_normalization(22, 23)
                model = hydra.utils.instantiate(cfg.model)
                self.assertEqual(model.arm_dim, 6)

    def test_hydra_training_accepts_custom_dimensions_and_exports_matching_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            rng = np.random.RandomState(42)
            np.savez(directory / "train.npz",
                     actions=rng.randn(16, 9).astype(np.float32) * 0.1,
                     states=rng.randn(16, 10).astype(np.float32) * 0.1,
                     traj_lengths=np.array([4] * 4))
            cfg = OmegaConf.create(dict(
                seed=42, device="cpu", wandb=None, logdir=str(directory / "output"),
                model=dict(_target_="model.hand_chord.chunk_hand_chord.ChunkHandActionChord",
                           normalization_path=self.write_normalization(),
                           horizon_steps=3, latent_dim=2, width=16, layers=1, heads=2,
                           **self.layout),
                train_dataset=dict(dataset_path=str(directory / "train.npz"), max_n_episodes=4,
                                   val_fraction=0.25, split_seed=42),
                train=dict(n_steps=2, batch_size=2, learning_rate=3e-4, val_freq=1,
                           save_model_freq=1, log_freq=1),
            ))
            summary = TrainChunkHandChordAgent(cfg).run()
            self.assertEqual(set(summary["validation"]), {"mae_deg"})
            self.assertTrue(np.isfinite(summary["validation"]["mae_deg"]))
            model = ChunkHandActionChord.load(summary["checkpoint"])
            self.assertEqual(model.raw_action_dim, 9)
            self.assertEqual(model.hand_dim, 4)
            self.assertEqual(model.hand_action_slice, slice(5, 9))
            self.assertEqual(model.hand_state_slice, slice(3, 7))


if __name__ == "__main__":
    unittest.main()
