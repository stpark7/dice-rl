"""Hydra-managed offline training of global and temporal hand-action chords."""

import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random

import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from agent.dataset.hand_chunks import load_data
from model.hand_chord.chunk_hand_chord import ChunkHandActionChord
from model.hand_chord.codec import load_hand_chord
from util.hand_chord_training import evaluate, fit_statistics, kl_weight


log = logging.getLogger(__name__)


class TrainChunkHandChordAgent:
    """Own data, optimization, validation, and exports without a simulator."""

    def __init__(self, cfg):

        # Resolve the configuration and validate training cadence.
        self.cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg = self.cfg
        self.seed = int(cfg.get("seed", 42))
        self.device = torch.device(cfg.get("device", "cuda:0"))

        # Training and validation frequencies, and model saving cadence.
        self.val_freq = int(cfg.train.get("val_freq", 1000))
        self.save_model_freq = int(cfg.train.get("save_model_freq", self.val_freq))
        self.log_freq = int(cfg.train.get("log_freq", 100))

        # Model Hyperparameters
        self.n_steps = int(cfg.train.n_steps)
        self.batch_size = int(cfg.train.batch_size)
        if min(self.n_steps, self.batch_size, self.val_freq, self.save_model_freq, self.log_freq) <= 0:
            raise ValueError("Training steps, batch size, and frequencies must be positive")
        # Configure KL warmup and gradient clipping, and check device availability.
        self.kl_max = float(cfg.train.get("kl_max", 3e-4))
        self.kl_zero_steps = int(cfg.train.get("kl_zero_steps", 4000))
        self.kl_ramp_steps = int(cfg.train.get("kl_ramp_steps", 4000))
        self.max_grad_norm = float(cfg.train.get("max_grad_norm", 1.0))

        # Seed random generators and instantiate the configured chord model.
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.model = hydra.utils.instantiate(cfg.model).to(self.device)
        if not isinstance(self.model, ChunkHandActionChord):
            raise ValueError("TrainChunkHandChordAgent requires a ChunkHandActionChord model")

        # Split data by episode so training and validation chunks stay separate.
        data = cfg.train_dataset
        self.dataset_train, self.dataset_val = load_data(
            data.dataset_path, self.model.horizon_steps, data.get("max_n_episodes", 50),
            data.get("val_fraction", 0.1), data.get("split_seed", self.seed),
            raw_action_dim=self.model.raw_action_dim, state_dim=self.model.state_dim,
        )

        # Fit relative-action statistics on training chunks and prepare seeded shuffling.
        mean, std = fit_statistics(self.model, self.dataset_train, self.batch_size, self.device)
        self.model.set_statistics(mean, std)
        self.dataloader_train = DataLoader(
            self.dataset_train, batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        # Pair AdamW updates with cosine learning-rate decay over the full run.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.train.learning_rate,
            weight_decay=cfg.train.get("weight_decay", 1e-4),
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_steps)

        # Save the resolved configuration and initialize checkpoint-selection state.
        self.output_dir = Path(
            HydraConfig.get().runtime.output_dir if HydraConfig.initialized() else cfg.logdir
        )
        cfg.logdir = str(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, self.output_dir / "config.yaml", resolve=True)
        self.step = 0
        self.best_step = 0
        self.best_mae = float("inf")

    def save_model(self):
        """Export current weights in the existing chord format."""
        self.model.save(self.output_dir / "latest.pt")

    def run(self):
        # Start optional experiment tracking and always close it when training exits.
        wandb_run = None
        if self.cfg.get("wandb") is not None:
            import wandb

            options = OmegaConf.to_container(self.cfg.wandb, resolve=True)
            name = options.pop("run", self.cfg.get("name", "chunk_hand_chord"))
            wandb_run = wandb.init(name=name, config=OmegaConf.to_container(self.cfg), **options)
        try:
            return self._train(wandb_run)
        finally:
            if wandb_run is not None:
                wandb_run.finish()

    def _train(self, wandb_run):
        batches = iter(self.dataloader_train)
        (self.output_dir / "history.jsonl").write_text("")

        for step in range(1, self.n_steps + 1):
            self.step = step
            # Count optimizer steps, reshuffling whenever the loader finishes an epoch.
            try:
                actions, state = next(batches)
            except StopIteration:
                batches = iter(self.dataloader_train)
                actions, state = next(batches)

            beta = kl_weight(
                self.step - 1,
                self.kl_max,
                self.kl_zero_steps,
                self.kl_ramp_steps,
            )
            losses = self._train_step(actions, state, beta)

            # Log and validate at their own intervals, always including the final step.
            is_last = self.step == self.n_steps
            if is_last or self.step % self.log_freq == 0:
                train_metrics = {key: value.item() for key, value in losses.items()}
                train_metrics["beta"] = beta
                log.info("step %d/%d: loss %.6f | KL %.6f | beta %.6g",
                         self.step, self.n_steps, train_metrics["loss"], train_metrics["kl"], beta)
                if wandb_run is not None:
                    wandb_run.log({"train/" + key: value for key, value in train_metrics.items()},
                                  step=self.step)

            if is_last or self.step % self.val_freq == 0:
                self._validate(wandb_run)

            # Export periodic and final-step weights independently of best-model selection.
            if is_last or self.step % self.save_model_freq == 0:
                self.save_model()
        return self._export_best()

    def _train_step(self, actions, state, beta):
        """Apply one optimizer update and return detached losses."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.model.loss(actions.to(self.device), state.to(self.device), beta=beta)
        if not torch.isfinite(losses["loss"]):
            raise RuntimeError("Nonfinite chord training loss at step {}".format(self.step))
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.max_grad_norm, error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.lr_scheduler.step()
        return {key: value.detach() for key, value in losses.items()}

    def _validate(self, wandb_run):
        """Record validation MAE and retain the checkpoint with the lowest error."""
        metrics = evaluate(self.model, self.dataset_val, self.batch_size, self.device)
        validation_mae = metrics["mae_deg"]
        if not math.isfinite(validation_mae):
            raise RuntimeError("Nonfinite validation MAE at step {}".format(self.step))

        with (self.output_dir / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(step=self.step, **metrics)) + "\n")
        log.info("step %d: validation hand MAE %.6f deg", self.step, validation_mae)

        if validation_mae < self.best_mae:
            self.best_mae = validation_mae
            self.best_step = self.step
            self.model.save(self.output_dir / "best.pt")

        if wandb_run is not None:
            wandb_run.log({"val/mae_deg": validation_mae}, step=self.step)

    def _export_best(self):
        """Fit latent statistics on the best model and export it for downstream RL."""
        best_checkpoint_path = self.output_dir / "best.pt"
        checkpoint_path = self.output_dir / "chord.pt"

        # Reload the best weights and fit latent statistics on training chunks.
        self.model = load_hand_chord(best_checkpoint_path, device=str(self.device))
        latent_mean, latent_std = fit_statistics(
            self.model, self.dataset_train, self.batch_size, self.device, latent=True,
        )
        self.model.set_latent_statistics(latent_mean, latent_std)

        # Save both checkpoints with the fitted latent statistics.
        self.model.save(best_checkpoint_path)
        self.model.save(checkpoint_path)

        # Record the optional normalization file's location and fingerprint.
        normalization_path = self.cfg.model.get("normalization_path")
        resolved_normalization_path = None
        normalization_sha256 = None
        if normalization_path:
            normalization_file = Path(normalization_path)
            resolved_normalization_path = str(normalization_file.resolve())
            normalization_sha256 = hashlib.sha256(normalization_file.read_bytes()).hexdigest()

        summary = dict(
            steps=self.n_steps,
            codec=("temporal_chunk" if self.model.metadata()["latent_layout"] == "B,H,K"
                   else "global_chunk"),
            latent_dim=self.model.latent_dim,
            policy_dim=self.model.policy_dim,
            split_seed=int(self.cfg.train_dataset.get("split_seed", self.seed)),
            train_episode_ids=self.dataset_train.episode_ids.tolist(),
            validation_episode_ids=self.dataset_val.episode_ids.tolist(),
            best_step=self.best_step,
            validation={"mae_deg": self.best_mae},
            checkpoint=str(checkpoint_path.resolve()),
            normalization_path=resolved_normalization_path,
            normalization_sha256=normalization_sha256,
            config=OmegaConf.to_container(self.cfg, resolve=True),
        )

        # Persist the summary alongside the exported checkpoint.
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        log.info("Saved best chord from step %d to %s", self.best_step, summary["checkpoint"])
        return summary
