"""Regression tests for latent residual gradients and checkpoint evaluation."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from model.hand_chord.chunk_hand_chord import ChunkHandActionChord
from model.rl.hand_chord_residual_rl_img import HandChordResidualRLImgModel


class RawPolicy(nn.Module):
    def forward_from_features(self, features, init_noise):
        return SimpleNamespace(trajectories=init_noise * 0.1)


class HandChordResidualRLTests(unittest.TestCase):
    def test_valid_model_keeps_decoder_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normalizer = root / "normalization.npz"
            np.savez(normalizer, action_min=np.full(22, -1.),
                     action_max=np.full(22, 1.), obs_min=np.full(23, -2.),
                     obs_max=np.full(23, 2.))
            chord = ChunkHandActionChord(
                normalization_path=normalizer, horizon_steps=2,
                latent_dim=3, width=8, layers=1, heads=2,
            )
            chord_path = root / "chord.pt"
            chord.save(chord_path)
            with patch.object(HandChordResidualRLImgModel, "_load_pretrained_policy",
                              return_value=RawPolicy()):
                model = HandChordResidualRLImgModel(
                    obs_dim=27, action_dim=22, horizon_steps=2, cond_steps=1,
                    pretrained_flow_policy_path="unused.pt", chord_path=str(chord_path),
                    actor_hidden_dims=[8], critic_hidden_dims=[8], device="cpu",
                )
            actions = model.get_action(torch.zeros(2, 1, 27), torch.zeros(2, 2, 22))
            self.assertEqual(actions.shape, (2, 2, 22))
            actions.square().mean().backward()
            self.assertTrue(any(p.grad is not None for p in model.actor.parameters()))
            self.assertTrue(all(p.grad is None for p in model.chord.parameters()))

    def test_checkpoint_evaluation_augments_saved_observation_dimension_once(self):
        from script.eval_rl_checkpoint import _prepare_image_model_config

        pretrained_cfg = OmegaConf.create({"model": {"network": {"visual_feature_dim": 128}}})
        for saved_obs_dim in (23, 151):
            with self.subTest(saved_obs_dim=saved_obs_dim):
                cfg = OmegaConf.create({
                    "obs_dim": saved_obs_dim,
                    "shape_meta": {"obs": {"state": {"shape": [23]}}},
                })
                raw_dim, visual_dim = _prepare_image_model_config(cfg, pretrained_cfg)
                self.assertEqual((raw_dim, visual_dim), (23, 128))
                self.assertEqual(cfg.obs_dim, 151)


if __name__ == "__main__":
    unittest.main()
