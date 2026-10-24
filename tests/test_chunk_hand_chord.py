"""Global hand-latent contracts for the chunk chord."""

import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


class ChunkHandChordTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        try:
            self.cls = importlib.import_module("model.hand_chord.chunk_hand_chord").ChunkHandActionChord
        except ModuleNotFoundError as error:
            self.fail("Global chunk chord is missing: {}".format(error))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.normalization_path = Path(self.temp.name) / "normalization.npz"
        np.savez(self.normalization_path, normalized=False,
                 action_min=np.full(22, -1.), action_max=np.full(22, 1.),
                 obs_min=np.full(23, -1.), obs_max=np.full(23, 1.))
        self.actions = torch.randn(3, 16, 22) * 0.2
        self.state = torch.randn(3, 23) * 0.1

    def chord(self, **kwargs):
        kwargs.setdefault("normalization_path", self.normalization_path)
        return self.cls(width=16, layers=1, heads=2, **kwargs)

    def test_one_global_code_arm_identity_and_no_arm_encoder_input(self):
        for dim in (8, 16, 20):
            chord = self.chord(latent_dim=dim).eval()
            mu, logvar = chord.posterior(self.actions, self.state)
            self.assertEqual(tuple(mu.shape), (3, dim))
            self.assertEqual(tuple(logvar.shape), (3, dim))
            changed = self.actions.clone()
            changed[..., :6] += 10
            torch.testing.assert_close(chord.posterior(changed, self.state)[0], mu)
            chord.set_latent_statistics(torch.full((dim,), 0.3), torch.full((dim,), 0.7))
            packed = chord.encode(self.actions, self.state)
            self.assertEqual(tuple(packed.shape), (3, 96 + dim))
            torch.testing.assert_close(packed[:, 96:], (mu - 0.3) / 0.7)
            self.assertTrue(torch.equal(packed[:, :96], self.actions[..., :6].reshape(3, 96)))
            decoded = chord.decode(packed, self.state)
            self.assertEqual(decoded.shape, self.actions.shape)
            self.assertTrue(torch.equal(decoded[..., :6], self.actions[..., :6]))
            with self.assertRaises(ValueError):
                chord.decode(packed[:, :-1], self.state)
            with self.assertRaises(ValueError):
                chord.posterior(self.actions[:, :-1], self.state)

    def test_q0_conditioning_physical_scales_and_latest_state(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "normalization.npz"
            np.savez(path, action_min=np.full(22, -4.), action_max=np.full(22, 4.),
                     obs_min=np.full(23, -2.), obs_max=np.full(23, 2.), normalized=True)
            chord = self.chord(normalization_path=path).eval()
            expected = ((self.actions[..., 6:] + 1) / 2 * (8 + 1e-6) - 4
                        - ((self.state[:, None, 7:] + 1) / 2 * (4 + 1e-6) - 2))
            torch.testing.assert_close(chord.get_relative_hand_targets(self.actions, self.state), expected)
            mu, _ = chord.posterior(self.actions, initial_hand_state=self.state)
            history = torch.stack([self.state + 1, self.state], dim=1)
            torch.testing.assert_close(chord.encode(self.actions, history),
                                       chord.encode(self.actions, self.state))
            self.assertFalse(torch.allclose(chord.posterior(self.actions, self.state + 0.2)[0], mu))

    def test_packed_layout_for_different_horizons_and_noncontiguous_actions(self):
        for horizon in (1, 4, 16):
            with self.subTest(horizon=horizon):
                chord = self.chord(horizon_steps=horizon, latent_dim=3).eval()
                actions = torch.randn(3, horizon, 44)[..., ::2]
                self.assertFalse(actions.is_contiguous())
                packed = chord.encode(actions, self.state)
                arm_count = horizon * 6
                self.assertEqual(packed.shape, (3, arm_count + 3))
                self.assertTrue(torch.equal(packed[:, :arm_count], actions[..., :6].flatten(1)))
                decoded = chord.decode(packed, self.state)
                self.assertEqual(decoded.shape, actions.shape)
                self.assertTrue(torch.equal(decoded[..., :6], actions[..., :6]))
                metadata = chord.metadata()
                self.assertEqual(metadata["policy_dim"], arm_count + 3)
                self.assertEqual(metadata["packed_arm_slice"], [0, arm_count])
                self.assertEqual(metadata["packed_hand_slice"], [arm_count, arm_count + 3])

    def test_decode_keeps_hand_independent_of_arm_and_uses_latest_state(self):
        chord = self.chord().eval()
        packed = chord.encode(self.actions, self.state)
        decoded = chord.decode(packed, self.state)
        changed = packed.clone()
        changed[:, :96] += 10
        torch.testing.assert_close(chord.decode(changed, self.state)[..., 6:], decoded[..., 6:])
        changed = packed.clone()
        changed[:, 96:] += 1
        self.assertFalse(torch.allclose(chord.decode(changed, self.state)[..., 6:], decoded[..., 6:]))
        history = torch.stack([self.state + 1, self.state], dim=1)
        torch.testing.assert_close(chord.decode(packed, history), decoded)
        self.assertFalse(torch.allclose(chord.decode(packed, self.state + 0.2)[..., 6:], decoded[..., 6:]))

    def test_kl_position_only_loss_stochastic_training_and_deterministic_evaluation(self):
        chord = self.chord()
        with patch("torch.diff", side_effect=AssertionError("temporal loss must not be computed")):
            first = chord.loss(self.actions, self.state, beta=0.01)
            second = chord.loss(self.actions, self.state, beta=0.01)
        self.assertEqual(set(first), {"loss", "reconstruction", "kl"})
        self.assertFalse(torch.equal(first["loss"], second["loss"]))
        torch.testing.assert_close(first["loss"], first["reconstruction"] + 0.01 * first["kl"])
        first["loss"].backward()
        for name in ("encoder", "decoder", "posterior_mean", "posterior_logvar"):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in getattr(chord, name).parameters()), name)
        chord.eval()
        torch.testing.assert_close(chord.loss(self.actions, self.state)["loss"],
                                   chord.loss(self.actions, self.state)["loss"])

    def test_checkpoint_omits_temporal_weights_and_loads_legacy_configuration(self):
        chord = self.chord().eval()
        packed = chord.encode(self.actions, self.state)
        decoded = chord.decode(packed, self.state)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "chord.pt"
            chord.save(path)
            checkpoint = torch.load(path, weights_only=True)
            for config in (checkpoint["config"], checkpoint["metadata"]["config"]):
                self.assertNotIn("velocity_weight", config)
                self.assertNotIn("acceleration_weight", config)
                config.update(velocity_weight=0.0, acceleration_weight=0.0)
            torch.save(checkpoint, path)
            loaded = self.cls.load(path)
            torch.testing.assert_close(loaded.encode(self.actions, self.state), packed)
            torch.testing.assert_close(loaded.decode(packed, self.state), decoded)
            self.assertNotIn("velocity_weight", loaded.metadata()["config"])
            self.assertNotIn("acceleration_weight", loaded.metadata()["config"])

    def test_self_contained_checkpoint_explicit_layout_and_unsupported_layout_rejected(self):
        chord = self.chord().eval()
        chord.set_statistics(torch.arange(16) * 0.01, torch.ones(16) * 0.4)
        chord.set_latent_statistics(torch.arange(8) * 0.1, torch.ones(8) * 0.7)
        packed = chord.encode(self.actions, self.state)
        decoded = chord.decode(packed, self.state)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "chord.pt"
            chord.save(path)
            loaded = self.cls.load(path)
            self.assertFalse(loaded.training)
            torch.testing.assert_close(loaded.encode(self.actions, self.state), packed)
            torch.testing.assert_close(loaded.decode(packed, self.state), decoded)
            metadata = loaded.metadata()
            self.assertEqual(metadata, chord.metadata())
            self.assertEqual(metadata["latent_layout"], "B,K")
            self.assertEqual(metadata["latent_scalars_per_chunk"], 8)
            self.assertEqual(metadata["packed_arm_slice"], [0, 96])
            self.assertEqual(metadata["packed_hand_slice"], [96, 104])
            checkpoint = torch.load(path, weights_only=True)
            checkpoint["metadata"]["latent_layout"] = "B,H,K"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "chunk"):
                self.cls.load(path)


if __name__ == "__main__":
    unittest.main()
