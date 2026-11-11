"""Exercise offline codec exports, Hydra sweeps, and raw-action residual RL."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from agent.hand_chord.train_chunk_hand_chord_agent import TrainChunkHandChordAgent
from model.rl.hand_chord_residual_rl_img import HandChordResidualRLImgModel


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "cfg/dexjoco/hand_chord/pretrain"
CODECS = ("global_chunk", "temporal_chunk")


def make_data(root):
    normalizer = root / "normalization.npz"
    dataset = root / "train.npz"
    np.savez(normalizer, action_min=np.full(22, -1.), action_max=np.full(22, 1.),
             obs_min=np.full(23, -2.), obs_max=np.full(23, 2.))
    random = np.random.RandomState(7)
    np.savez(dataset, actions=random.uniform(-.5, .5, (20, 22)).astype(np.float32),
             states=random.uniform(-.5, .5, (20, 23)).astype(np.float32),
             traj_lengths=np.full(4, 5))
    return normalizer, dataset


def training_config(root, codec):
    normalizer, dataset = make_data(root)
    target = ("chunk_hand_chord.ChunkHandActionChord" if codec == "global_chunk"
              else "temporal_hand_chord.TemporalHandActionChord")
    return OmegaConf.create({
        "codec": codec, "seed": 42, "device": "cpu", "logdir": str(root / codec),
        "wandb": None,
        "model": {"_target_": "model.hand_chord." + target,
                  "normalization_path": str(normalizer), "horizon_steps": 2,
                  "latent_dim": 3, "width": 8, "layers": 1, "heads": 2},
        "train": {"n_steps": 2, "batch_size": 4, "learning_rate": 1e-3,
                  "val_freq": 1, "save_model_freq": 1, "log_freq": 1},
        "train_dataset": {"dataset_path": str(dataset), "max_n_episodes": 4,
                          "val_fraction": .25, "split_seed": 0},
    })


class RawPolicy(nn.Module):
    def forward_from_features(self, features, init_noise):
        return SimpleNamespace(trajectories=init_noise * .1)


class HandChordIntegrationTests(unittest.TestCase):
    def test_short_training_exports_best_codec_and_reproducible_split(self):
        from model.hand_chord.codec import load_hand_chord

        for codec in CODECS:
            with self.subTest(codec=codec), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                agent = TrainChunkHandChordAgent(training_config(root, codec))
                summary = agent.run()
                self.assertEqual(summary["codec"], codec)
                self.assertEqual(summary["policy_dim"], 15 if codec == "global_chunk" else 18)
                self.assertEqual(summary["split_seed"], 0)
                self.assertEqual(summary["train_episode_ids"], [0, 1, 3])
                self.assertEqual(summary["validation_episode_ids"], [2])
                history = [json.loads(line) for line in
                           (agent.output_dir / "history.jsonl").read_text().splitlines()]
                best_record = min(history, key=lambda record: record["mae_deg"])
                self.assertEqual(summary["best_step"], best_record["step"])
                self.assertEqual(summary["validation"]["mae_deg"], best_record["mae_deg"])
                loaded = load_hand_chord(summary["checkpoint"])
                best = load_hand_chord(agent.output_dir / "best.pt")
                self.assertEqual(type(loaded), type(agent.model))
                for name, value in loaded.state_dict().items():
                    torch.testing.assert_close(value, best.state_dict()[name])
                self.assertTrue(torch.isfinite(loaded.latent_std).all())
                self.assertEqual(loaded.latent_mean.shape, (3,))
                self.assertEqual(len((agent.output_dir / "history.jsonl").read_text().splitlines()), 2)
                self.assertTrue((agent.output_dir / "latest.pt").is_file())

    def test_hydra_runtime_directory_overrides_programmatic_logdir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = training_config(root, "global_chunk")
            runtime = root / "hydra_output"
            with patch.object(HydraConfig, "initialized", return_value=True), patch.object(
                HydraConfig, "get", return_value=SimpleNamespace(
                    runtime=SimpleNamespace(output_dir=str(runtime)))):
                agent = TrainChunkHandChordAgent(cfg)
            self.assertEqual(agent.output_dir, runtime)
            self.assertTrue((runtime / "config.yaml").is_file())
            self.assertFalse(Path(cfg.logdir).exists())

    def test_task_selectors_and_default_budgets_compose(self):
        for task, steps, latent in (("pick_bucket", 20000, 8), ("hammer_nail", 30000, 8),
                                    ("fold_glasses", 30000, 32)):
            with self.subTest(task=task), hydra.initialize_config_dir(
                config_dir=str(CONFIG_ROOT / task), version_base=None
            ):
                cfg = hydra.compose(config_name="pre_chunk_hand_chord")
                self.assertEqual(cfg.codec, "global_chunk")
                self.assertEqual(cfg.train.n_steps, steps)
                self.assertEqual(cfg.latent_dim, latent)
                self.assertEqual(cfg.train_dataset.split_seed, 0)
                for codec in CODECS:
                    configured = hydra.compose(config_name="pre_chunk_hand_chord", overrides=[
                        "codec=" + codec, "latent_dim=9"])
                    self.assertEqual(configured.model._target_, configured.codec_targets[codec])
                    self.assertIn(codec, configured.name)
                    self.assertIn("k9", configured.name)
                    if configured.wandb is not None:
                        self.assertIn(codec, configured.wandb.run)
                        self.assertIn("k9", configured.wandb.run)

    def test_real_hydra_multirun_isolates_resolved_configs_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normalizer, dataset = make_data(root)
            sweep = root / "sweep"
            command = [sys.executable, "script/run.py", "--config-path",
                       str(CONFIG_ROOT / "pick_bucket"), "--config-name", "pre_chunk_hand_chord", "-m",
                       "codec=global_chunk,temporal_chunk", "latent_dim=3", "seed=42",
                       "device=cpu", "wandb=null", "horizon_steps=2", "train.n_steps=1",
                       "train.batch_size=4", "model.width=8", "model.layers=1", "model.heads=2",
                       "train_dataset.max_n_episodes=4", "train_dataset.val_fraction=0.25",
                       "normalization_path=" + str(normalizer), "train_dataset_path=" + str(dataset),
                       "logdir=" + str(root / "fallback"), "hydra.sweep.dir=" + str(sweep)]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            summaries = sorted(sweep.glob("*/summary.json"))
            self.assertEqual(len(summaries), 2, completed.stdout + completed.stderr)
            self.assertEqual({json.loads(path.read_text())["codec"] for path in summaries}, set(CODECS))
            for path in summaries:
                self.assertTrue((path.parent / "chord.pt").is_file())
                saved_cfg = OmegaConf.load(path.parent / "config.yaml")
                self.assertEqual(Path(saved_cfg.logdir), path.parent)
                self.assertEqual(saved_cfg.model._target_, saved_cfg.codec_targets[saved_cfg.codec])
                self.assertTrue((path.parent / ".hydra/config.yaml").is_file())
                self.assertEqual(Path(json.loads(path.read_text())["checkpoint"]).parent, path.parent)
            self.assertFalse((root / "fallback").exists())

    def test_temporal_rl_preserves_raw_interfaces_and_decoder_gradient(self):
        from model.hand_chord.temporal_hand_chord import TemporalHandActionChord

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            normalizer, _ = make_data(root)
            chord = TemporalHandActionChord(normalization_path=normalizer, horizon_steps=16,
                                            latent_dim=9, width=8, layers=1, heads=2)
            chord_path = root / "chord.pt"
            chord.save(chord_path)
            for condition_on_base in (False, True):
                with self.subTest(condition_on_base=condition_on_base), patch.object(
                    HandChordResidualRLImgModel, "_load_pretrained_policy", return_value=RawPolicy()
                ):
                    model = HandChordResidualRLImgModel(
                        obs_dim=27, action_dim=22, horizon_steps=16, cond_steps=1,
                        pretrained_flow_policy_path="unused.pt", chord_path=str(chord_path),
                        actor_hidden_dims=[8], critic_hidden_dims=[8], device="cpu",
                        condition_residual_on_base_action=condition_on_base)
                    self.assertEqual(model.actor.chord_dim, 240)
                    self.assertEqual(model.actor.input_action_dim, 240 if condition_on_base else 352)
                    self.assertEqual(model.action_dim, 22)
                    state, noise = torch.zeros(2, 1, 27), torch.randn(2, 16, 22)
                    actions, projected = model.get_action(state, noise, return_pretrained_actions=True)
                    self.assertEqual(actions.shape, (2, 16, 22))
                    self.assertEqual(projected.shape, actions.shape)
                    actions[..., 6:].square().mean().backward()
                    gradients = [p.grad for p in model.actor.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                    self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)
                    self.assertTrue(all(p.grad is None for p in model.chord.parameters()))
                    self.assertTrue(all(not p.requires_grad for p in model.chord.parameters()))
                    model.train()
                    self.assertFalse(model.chord.training)
                    self.assertFalse(model.pretrained_flow_policy.training)
                    self.assertEqual(model.critic(state, noise, actions.detach()).shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
