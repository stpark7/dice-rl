"""Physical coordinates and statistics for the chunk hand chord."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from model.hand_chord.chunk_hand_chord import ChunkHandActionChord


class ChunkHandChordCoordinateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.normalization_path = Path(self.temp.name) / "normalization.npz"
        self.action_min = np.linspace(-0.7, 0.2, 22, dtype=np.float32)
        self.action_max = self.action_min + np.linspace(0.5, 1.8, 22, dtype=np.float32)
        self.obs_min = np.linspace(-1.6, -0.3, 23, dtype=np.float32)
        self.obs_max = self.obs_min + np.linspace(2.7, 1.2, 23, dtype=np.float32)
        np.savez(
            self.normalization_path,
            action_min=self.action_min, action_max=self.action_max,
            obs_min=self.obs_min, obs_max=self.obs_max,
            normalized=np.asarray(True),
        )
        self.raw_q0 = torch.linspace(-0.1, 0.7, 16).repeat(2, 1)
        raw_state = torch.zeros(2, 23)
        raw_state[:, 7:23] = self.raw_q0
        self.state = self.normalize(raw_state, self.obs_min, self.obs_max)
        self.raw_actions = torch.zeros(2, 16, 22)
        self.raw_actions[:, :, :6] = torch.linspace(-0.2, 0.2, 6)
        self.raw_actions[:, :, 6:22] = (
            self.raw_q0[:, None] + 0.35 + torch.arange(16)[None, :, None] * 0.01
        )
        self.actions = self.normalize(
            self.raw_actions, self.action_min, self.action_max
        )

    @staticmethod
    def normalize(values, minimum, maximum):
        minimum, maximum = torch.as_tensor(minimum), torch.as_tensor(maximum)
        return 2 * (values - minimum) / (maximum - minimum + 1e-6) - 1

    def make_chord(self, **kwargs):
        return ChunkHandActionChord(
            normalization_path=self.normalization_path,
            width=32, layers=1, heads=4, **kwargs,
        )

    def test_physical_relative_uses_separate_action_and_state_statistics(self):
        chord = self.make_chord()
        torch.testing.assert_close(chord.get_hand_targets(self.actions), self.raw_actions[:, :, 6:22])
        torch.testing.assert_close(chord.get_initial_hand_positions(self.state), self.raw_q0)
        expected = self.raw_actions[:, :, 6:22] - self.raw_q0[:, None]
        torch.testing.assert_close(chord.get_relative_hand_targets(self.actions, self.state), expected)
        torch.testing.assert_close(
            chord.normalize_hand(chord.get_hand_targets(self.actions)),
            self.actions[:, :, 6:22],
        )
        self.assertFalse(torch.allclose(
            expected, self.actions[:, :, 6:22] - self.state[:, None, 7:23]
        ))

    def test_checkpoint_is_self_contained_and_preserves_statistics(self):
        chord = self.make_chord().eval()
        chord.set_statistics(torch.linspace(-0.1, 0.3, 16), torch.linspace(0.2, 1.0, 16))
        chord.set_latent_statistics(torch.linspace(-0.3, 0.4, 8), torch.linspace(0.7, 1.1, 8))
        packed = chord.encode(self.actions, self.state)
        decoded = chord.decode(packed, self.state)
        path = Path(self.temp.name) / "chord.pt"
        chord.save(path)
        self.normalization_path.unlink()
        restored = ChunkHandActionChord.load(path).eval()
        json.dumps(restored.metadata())
        torch.testing.assert_close(restored.encode(self.actions, self.state), packed)
        torch.testing.assert_close(restored.decode(packed, self.state), decoded)
        for name in ("relative_mean", "relative_std", "latent_mean", "latent_std"):
            torch.testing.assert_close(getattr(restored, name), getattr(chord, name))

    def test_weighted_loss_backpropagates_through_encoder_and_decoder(self):
        chord = self.make_chord(reconstruction_weight=0.7)
        chord.set_statistics(torch.zeros(16), torch.full((16,), 0.4))
        terms = chord.loss(self.actions, self.state, beta=0.01)
        self.assertEqual(set(terms), {"loss", "reconstruction", "kl"})
        for value in terms.values():
            self.assertEqual(value.ndim, 0)
            self.assertTrue(torch.isfinite(value))
        torch.testing.assert_close(
            terms["loss"], 0.7 * terms["reconstruction"] + 0.01 * terms["kl"],
        )
        terms["loss"].backward()
        for name in ("encoder", "decoder", "posterior_mean", "posterior_logvar"):
            gradients = [parameter.grad for parameter in getattr(chord, name).parameters()]
            self.assertTrue(any(g is not None and bool(g.abs().sum() > 0) for g in gradients), name)

    def test_default_ring_weights_and_invalid_statistics(self):
        chord = self.make_chord()
        expected = torch.ones(16)
        expected[8:12] = 0.5
        torch.testing.assert_close(chord.joint_weights, expected)
        with self.assertRaises(ValueError):
            chord.set_statistics(torch.zeros(16), torch.zeros(16))
        with self.assertRaises(ValueError):
            chord.set_latent_statistics(torch.zeros(8), torch.full((8,), float("nan")))

    def test_missing_normalization_path_raises_error(self):
        for kwargs in ({}, {"normalization_path": None}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "normalization_path"):
                ChunkHandActionChord(width=32, layers=1, heads=4, **kwargs)

    def test_missing_normalization_file_raises_error(self):
        self.normalization_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.make_chord()

    def test_explicit_unnormalized_file_uses_physical_coordinates(self):
        np.savez(
            self.normalization_path,
            action_min=self.action_min, action_max=self.action_max,
            obs_min=self.obs_min, obs_max=self.obs_max, normalized=False,
        )
        chord = self.make_chord()
        self.assertTrue(torch.equal(chord.get_hand_targets(self.raw_actions), self.raw_actions[:, :, 6:22]))
        raw_state = torch.zeros(2, 23)
        raw_state[:, 7:23] = self.raw_q0
        self.assertTrue(torch.equal(chord.get_initial_hand_positions(raw_state), self.raw_q0))


if __name__ == "__main__":
    unittest.main()
