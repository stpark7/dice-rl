"""Temporal hand codec packing, checkpoint compatibility, and gradient tests."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model.hand_chord.chunk_hand_chord import ChunkHandActionChord
from model.hand_chord.codec import load_hand_chord
from model.hand_chord.temporal_hand_chord import TemporalHandActionChord
from util.hand_chord_training import fit_statistics


class TemporalHandActionChordTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.normalizer = self.root / "normalization.npz"
        np.savez(
            self.normalizer,
            action_min=np.full(22, -1.), action_max=np.full(22, 1.),
            obs_min=np.full(23, -2.), obs_max=np.full(23, 2.),
        )
        torch.manual_seed(17)
        self.config = dict(
            normalization_path=self.normalizer, horizon_steps=3,
            latent_dim=2, width=8, layers=1, heads=2,
        )
        self.chord = TemporalHandActionChord(**self.config).eval()
        self.actions = torch.randn(2, 3, 22)
        self.state = torch.randn(2, 23)

    def test_posterior_and_packed_layout_preserve_exact_arm_values(self):
        mean, logvar = self.chord.posterior(self.actions, self.state)
        self.assertEqual(mean.shape, (2, 3, 2))
        self.assertEqual(logvar.shape, mean.shape)
        self.chord.set_latent_statistics([1., -1.], [2., 3.])
        packed = self.chord.encode(self.actions, self.state[:, None])
        self.assertEqual(self.chord.policy_dim, 3 * 6 + 3 * 2)
        self.assertEqual(packed.shape, (2, 24))
        torch.testing.assert_close(packed[:, :18], self.actions[..., :6].flatten(1), rtol=0, atol=0)
        torch.testing.assert_close(packed[:, 18:], ((mean - self.chord.latent_mean) / self.chord.latent_std).flatten(1))
        decoded = self.chord.decode(packed, self.state[:, None])
        reconstructed, _, _ = self.chord.reconstruct(self.actions, self.state)
        self.assertEqual(decoded.shape, self.actions.shape)
        torch.testing.assert_close(decoded[..., :6], self.actions[..., :6], rtol=0, atol=0)
        torch.testing.assert_close(decoded, reconstructed)

    def test_decoder_consumes_each_frame_latent_and_measured_q0(self):
        # A decoder with no temporal mixing makes repeated latents observable.
        self.chord.decoder = torch.nn.Identity()
        self.chord.decoder_position.data.zero_()
        latent = torch.zeros(2, 3, 2)
        baseline = self.chord._decode_relative_hand_action(latent, self.state)
        latent[:, 1, 0] = 1
        changed = self.chord._decode_relative_hand_action(latent, self.state)
        torch.testing.assert_close(changed[:, (0, 2)], baseline[:, (0, 2)], rtol=0, atol=0)
        self.assertGreater((changed[:, 1] - baseline[:, 1]).abs().max().item(), 0)
        # The actual attention stack must also receive the measured pose token.
        chord = TemporalHandActionChord(**self.config).eval()
        original = chord._decode_relative_hand_action(latent, self.state)
        shifted_state = self.state.clone()
        shifted_state[:, 7:23] += 1
        changed = chord._decode_relative_hand_action(latent, shifted_state)
        self.assertGreater((changed - original).abs().max().item(), 0)
        original_mean = chord.posterior(self.actions, self.state)[0]
        changed_mean = chord.posterior(self.actions, shifted_state)[0]
        self.assertGreater((changed_mean - original_mean).abs().max().item(), 0)

    def test_self_contained_checkpoint_roundtrip_and_metadata(self):
        self.chord.set_statistics(torch.arange(16) / 10, torch.arange(16) / 10 + 1)
        self.chord.set_latent_statistics([1., -1.], [2., 3.])
        expected = self.chord.decode(self.chord.encode(self.actions, self.state), self.state)
        checkpoint_path = self.root / "temporal.pt"
        self.chord.save(checkpoint_path)
        self.normalizer.unlink()
        loaded = load_hand_chord(checkpoint_path)
        direct = TemporalHandActionChord.load(checkpoint_path)
        self.assertIsInstance(loaded, TemporalHandActionChord)
        self.assertFalse(loaded.training)
        for chord in (loaded, direct):
            torch.testing.assert_close(chord.decode(chord.encode(self.actions, self.state), self.state), expected)
            torch.testing.assert_close(chord.relative_mean, self.chord.relative_mean)
            torch.testing.assert_close(chord.latent_std, self.chord.latent_std)
        metadata = loaded.metadata()
        self.assertEqual(metadata["format_version"], 2)
        self.assertEqual(metadata["latent_layout"], "B,H,K")
        self.assertEqual(metadata["latent_scalars_per_chunk"], 6)
        self.assertEqual(metadata["packed_arm_slice"], [0, 18])
        self.assertEqual(metadata["packed_hand_slice"], [18, 24])
        self.assertEqual(metadata["packed_hand_order"], "frame_major")
        self.assertEqual(metadata["policy_dim"], 24)
        with self.assertRaises(ValueError):
            ChunkHandActionChord.load(checkpoint_path)

    def test_shared_loader_preserves_v1_global_checkpoints(self):
        chord = ChunkHandActionChord(**self.config).eval()
        path = self.root / "global.pt"
        chord.save(path)
        # Historical global checkpoints stored unused derivative loss weights.
        checkpoint = torch.load(path, weights_only=True)
        checkpoint["config"].update(velocity_weight=0., acceleration_weight=0.)
        torch.save(checkpoint, path)
        self.normalizer.unlink()
        loaded = load_hand_chord(path)
        self.assertIs(type(loaded), ChunkHandActionChord)
        self.assertEqual(loaded.policy_dim, 20)
        torch.testing.assert_close(loaded.encode(self.actions, self.state), chord.encode(self.actions, self.state))
        with self.assertRaises(ValueError):
            TemporalHandActionChord.load(path)

    def test_loader_rejects_inconsistent_or_unknown_layouts(self):
        path = self.root / "invalid.pt"
        self.chord.save(path)
        checkpoint = torch.load(path, weights_only=True)
        for version, layout in ((1, "B,H,K"), (2, "B,K"), (3, "B,H,K")):
            with self.subTest(version=version, layout=layout):
                checkpoint["format_version"] = version
                checkpoint["metadata"]["latent_layout"] = layout
                torch.save(checkpoint, path)
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    load_hand_chord(path)

    def test_temporal_kl_uses_same_per_k_normalization_as_global(self):
        mean = torch.tensor([[1., 2.], [-1., 0.5]])
        logvar = torch.tensor([[0.1, -0.2], [0.3, -0.4]])
        global_chord = ChunkHandActionChord(**self.config).eval()
        with patch.object(global_chord, "posterior", return_value=(mean, logvar)):
            expected = global_chord.loss(self.actions, self.state)["kl"]
        temporal_moments = tuple(values[:, None].expand(-1, 3, -1) for values in (mean, logvar))
        with patch.object(self.chord, "posterior", return_value=temporal_moments):
            result = self.chord.loss(self.actions, self.state, beta=0.2)
        torch.testing.assert_close(result["kl"], expected)
        torch.testing.assert_close(result["loss"], result["reconstruction"] + 0.2 * expected)

    def test_frozen_decoder_preserves_gradients_to_temporal_residual(self):
        self.chord.requires_grad_(False)
        packed = self.chord.encode(self.actions, self.state).detach()
        residual = torch.zeros_like(packed, requires_grad=True)
        decoded = self.chord.decode(packed + residual, self.state)
        decoded[..., 6:].square().mean().backward()
        self.assertGreater(residual.grad[:, 18:].abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.chord.parameters()))

    def test_shared_statistics_pool_temporal_latents_to_k(self):
        dataset = torch.utils.data.TensorDataset(self.actions, self.state)
        expected = self.chord.posterior(self.actions, self.state)[0].detach().reshape(-1, 2)
        mean, std = fit_statistics(self.chord, dataset, batch_size=2, device="cpu", latent=True)
        self.assertEqual(mean.shape, (2,))
        torch.testing.assert_close(mean, expected.mean(0))
        torch.testing.assert_close(std, expected.std(0, unbiased=False))
        self.chord.set_latent_statistics(mean, std)

    def test_invalid_temporal_shapes_raise_clear_errors(self):
        for latent in (torch.zeros(2, 2), torch.zeros(2, 2, 2), torch.zeros(2, 3, 3)):
            with self.subTest(shape=latent.shape), self.assertRaises(ValueError):
                self.chord._decode_relative_hand_action(latent, self.state)
        with self.assertRaises(ValueError):
            self.chord.decode(torch.zeros(2, 20), self.state)
        with self.assertRaises(ValueError):
            self.chord.decode(torch.zeros(2, 24), self.state[:1])


if __name__ == "__main__":
    unittest.main()
