"""Offline chord agent contracts, including the public Hydra launcher."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from omegaconf import OmegaConf
import torch

from util.hand_chord_training import evaluate


ROOT = Path(__file__).resolve().parents[1]


class TrainChunkHandChordAgentTests(unittest.TestCase):
    def setUp(self):
        try:
            self.agent_class = importlib.import_module(
                "agent.hand_chord.train_chunk_hand_chord_agent"
            ).TrainChunkHandChordAgent
        except ModuleNotFoundError as error:
            self.fail("Chord training agent is missing: {}".format(error))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        rng = np.random.RandomState(5)
        self.actions = rng.uniform(-0.2, 0.2, (40, 22)).astype(np.float32)
        self.states = rng.uniform(-0.2, 0.2, (40, 23)).astype(np.float32)
        np.savez(self.directory / "train.npz", actions=self.actions, states=self.states,
                 traj_lengths=np.array([10] * 4))
        np.savez(self.directory / "normalization.npz", normalized=True,
                 action_min=np.full(22, -4.0), action_max=np.full(22, 4.0),
                 obs_min=np.full(23, -2.0), obs_max=np.full(23, 2.0))

    def config(self, name="result"):
        return OmegaConf.create({
            "seed": 42, "device": "cpu", "wandb": None,
            "logdir": str(self.directory / name),
            "train_dataset": {
                "dataset_path": str(self.directory / "train.npz"),
                "max_n_episodes": 4, "val_fraction": 0.25, "split_seed": 42,
            },
            "model": {
                "_target_": "model.hand_chord.chunk_hand_chord.ChunkHandActionChord",
                "normalization_path": str(self.directory / "normalization.npz"),
                "horizon_steps": 4, "latent_dim": 3, "width": 16,
                "layers": 1, "heads": 2,
            },
            "train": {
                "n_steps": 3, "batch_size": 4, "learning_rate": 3e-4,
                "weight_decay": 1e-4, "val_freq": 2, "save_model_freq": 2,
                "log_freq": 2, "max_grad_norm": 1.0,
                "kl_max": 3e-4, "kl_zero_steps": 0, "kl_ramp_steps": 2,
            },
        })

    def test_training_validates_at_interval_and_final_step_and_exports_best(self):
        agent = self.agent_class(self.config())
        initial_weights = {k: v.detach().clone() for k, v in agent.model.named_parameters()}
        with patch("agent.hand_chord.train_chunk_hand_chord_agent.evaluate", wraps=evaluate) as validation:
            summary = agent.run()
        self.assertEqual(validation.call_count, 2)
        output = self.directory / "result"
        history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
        self.assertEqual([entry["step"] for entry in history], [2, 3])
        self.assertTrue(all(set(entry) == {"step", "mae_deg"} for entry in history))
        best = min(history, key=lambda entry: entry["mae_deg"])
        self.assertEqual(summary["best_step"], best["step"])
        self.assertEqual(summary["validation"], {"mae_deg": best["mae_deg"]})
        self.assertEqual(set(summary), {
            "steps", "latent_dim", "best_step", "validation", "checkpoint",
            "normalization_path", "normalization_sha256", "config",
        })
        self.assertFalse(set(agent.dataset_train.episode_ids) & set(agent.dataset_val.episode_ids))
        self.assertTrue(np.isfinite(summary["validation"]["mae_deg"]))
        self.assertTrue(any(not torch.equal(initial_weights[k], v)
                            for k, v in agent.model.named_parameters()))
        self.assertEqual({path.name for path in output.iterdir()}, {
            "config.yaml", "summary.json", "history.jsonl", "latest.pt", "best.pt", "chord.pt",
        })
        self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
        self.assertEqual(summary["config"]["seed"], 42)
        from model.hand_chord.chunk_hand_chord import ChunkHandActionChord
        chord = ChunkHandActionChord.load(output / "chord.pt")
        actions, state = torch.from_numpy(self.actions[:4])[None], torch.from_numpy(self.states[:1])
        decoded = chord.decode(chord.encode(actions, state), state)
        self.assertEqual(decoded.shape, actions.shape)
        self.assertTrue(torch.isfinite(decoded).all())
        self.assertTrue(torch.equal(decoded[..., :6], actions[..., :6]))
        starts = agent.dataset_train.starts
        relative = np.stack([
            (self.actions[start:start + 4, 6:] + 1) / 2 * (8 + 1e-6) - 4
            - ((self.states[start, 7:] + 1) / 2 * (4 + 1e-6) - 2)
            for start in starts
        ])
        np.testing.assert_allclose(chord.relative_mean, relative.mean((0, 1)), atol=1e-6)
        means = torch.cat([chord.posterior(a, s)[0].detach()
                           for a, s in torch.utils.data.DataLoader(agent.dataset_train, batch_size=4)])
        torch.testing.assert_close(chord.latent_mean, means.mean(0), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(chord.latent_std, means.std(0, correction=0).clamp_min(1e-4),
                                   atol=1e-6, rtol=1e-5)

    def test_seed_reproduces_training_and_exports(self):
        self.agent_class(self.config("first")).run()
        self.agent_class(self.config("second")).run()
        from model.hand_chord.chunk_hand_chord import ChunkHandActionChord
        first = ChunkHandActionChord.load(self.directory / "first" / "chord.pt")
        second = ChunkHandActionChord.load(self.directory / "second" / "chord.pt")
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]), key)

    def test_validation_frequency_does_not_change_training_across_epochs(self):
        weights = []
        for frequency in (1, 3):
            cfg = self.config("frequency_{}".format(frequency))
            cfg.train.n_steps = 8
            cfg.train.batch_size = 8
            cfg.train.val_freq = frequency
            agent = self.agent_class(cfg)
            batch_sizes = []
            original_loss = agent.model.loss

            def record_loss(actions, state, beta):
                batch_sizes.append(len(actions))
                return original_loss(actions, state, beta=beta)

            # Inspect final training weights before best-checkpoint selection.
            with patch.object(agent.model, "loss", side_effect=record_loss), \
                    patch.object(agent, "_export_best", return_value=None):
                agent._train(None)
            self.assertEqual(agent.step, 8)
            self.assertEqual(agent.lr_scheduler.last_epoch, 8)
            self.assertEqual(batch_sizes, [8, 8, 5, 8, 8, 5, 8, 8])
            for state in agent.optimizer.state.values():
                self.assertEqual(state["step"].item(), 8)
            weights.append({key: value.detach().clone() for key, value in agent.model.named_parameters()})
        for key in weights[0]:
            self.assertTrue(torch.equal(weights[0][key], weights[1][key]), key)

    def test_logs_current_batch_losses_without_optimizer_diagnostics(self):
        cfg = self.config()
        cfg.train.n_steps = 5
        cfg.train.batch_size = 8
        cfg.train.log_freq = 3
        cfg.train.val_freq = 2
        agent = self.agent_class(cfg)
        batches = []
        original_loss = agent.model.loss

        def record_loss(actions, state, beta):
            losses = original_loss(actions, state, beta=beta)
            batches.append({
                "beta": beta,
                **{key: value.detach().item() for key, value in losses.items()},
            })
            return losses

        wandb_run = Mock()
        with patch.object(agent.model, "loss", side_effect=record_loss):
            agent._train(wandb_run)
        reports = [(call.kwargs["step"], call.args[0]) for call in wandb_run.log.call_args_list
                   if "train/loss" in call.args[0]]
        self.assertEqual([step for step, metrics in reports], [3, 5])
        for step, metrics in reports:
            self.assertEqual(set(metrics), {"train/loss", "train/reconstruction", "train/kl", "train/beta"})
            for key in ("loss", "reconstruction", "kl"):
                self.assertEqual(metrics["train/" + key], batches[step - 1][key])
            self.assertEqual(metrics["train/beta"], batches[step - 1]["beta"])
        validations = [(call.kwargs["step"], call.args[0]) for call in wandb_run.log.call_args_list
                       if "val/mae_deg" in call.args[0]]
        self.assertEqual([step for step, metrics in validations], [2, 4, 5])
        self.assertTrue(all(set(metrics) == {"val/mae_deg"} for step, metrics in validations))

    def test_validation_rejects_zero_steps(self):
        cfg = self.config()
        cfg.train.n_steps = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            self.agent_class(cfg)

    def test_training_rejects_nonpositive_batch_size_and_cadence(self):
        for field in ("batch_size", "val_freq", "save_model_freq", "log_freq"):
            for value in (0, -1):
                cfg = self.config()
                cfg.train[field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "positive"):
                    self.agent_class(cfg)

    def test_existing_run_launcher_accepts_task_config_and_overrides(self):
        output = self.directory / "launcher"
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        result = subprocess.run([
            sys.executable, "script/run.py", "--config-name=pre_chunk_hand_chord",
            "--config-dir=cfg/dexjoco/hand_chord/pretrain/pick_bucket/", "device=cpu",
            "wandb=null",
            "seed=7", "horizon_steps=4", "latent_dim=3", "model.width=16",
            "model.layers=1", "model.heads=2", "train.n_steps=2", "train.batch_size=4",
            "train.val_freq=1", "train.save_model_freq=1", "train.log_freq=1",
            "train_dataset.max_n_episodes=4", "train_dataset.val_fraction=0.25",
            "train_dataset_path=" + str(self.directory / "train.npz"),
            "normalization_path=" + str(self.directory / "normalization.npz"),
            "logdir=" + str(output),
        ], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["config"]["seed"], 7)
        self.assertEqual(summary["steps"], 2)
        self.assertEqual(summary["latent_dim"], 3)
        for key in ("threads", "source_hz", "control_hz"):
            self.assertNotIn(key, summary["config"])


if __name__ == "__main__":
    unittest.main()
