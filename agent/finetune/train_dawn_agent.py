"""Per-control-step DexJoCo adaptation of DAWN residual SAC.

Based on DAWN (Ma et al., ICML 2026), online_dawn/
pi_dec_diffusion_maniskill2_dawn.py. Shared rollout/evaluation machinery follows
the native ResFiT baseline; learning uses online replay only. See docs/DAWN.md.
"""

import logging
import math

import numpy as np
import torch

from agent.finetune.train_resfit_agent import TrainResFiTAgent
from util.resfit_replay_buffer import ResFiTReplayBuffer

log = logging.getLogger(__name__)


class TrainDAWNAgent(TrainResFiTAgent):
    algorithm = 'DAWN'
    model_config_fields = (
        'residual_scale', 'actor_hidden_dim', 'critic_hidden_dim',
        'critic_num_blocks', 'critic_expansion', 'log_std_min', 'log_std_max',
        'sac_alpha', 'autotune', 'target_entropy',
    )

    def _validate_training_config(self):
        cfg = self.cfg
        for key in ('batch_size', 'policy_frequency', 'target_network_frequency', 'training_freq'):
            value = cfg.train[key]
            if not math.isfinite(value) or value < 1 or int(value) != value:
                raise ValueError(f'train.{key} must be a positive integer')
        for key in ('buffer_warmup_steps', 'critic_warmup_steps'):
            value = cfg.train[key]
            if not math.isfinite(value) or value < 0 or int(value) != value:
                raise ValueError(f'train.{key} must be a nonnegative integer')
        if not math.isfinite(cfg.train.utd_ratio) or cfg.train.utd_ratio <= 0:
            raise ValueError('train.utd_ratio must be positive and finite')
        if cfg.train.n_step != 1:
            raise ValueError('DAWN SAC requires train.n_step=1')
        if cfg.replay_buffer.max_size < max(1, cfg.train.buffer_warmup_steps):
            raise ValueError('replay_buffer.max_size must hold the buffer warmup transitions')
        for key in ('actor_lr', 'critic_lr', 'alpha_lr'):
            if not math.isfinite(cfg.train[key]) or cfg.train[key] <= 0:
                raise ValueError(f'train.{key} must be positive and finite')
        if not math.isfinite(cfg.train.reward_offset):
            raise ValueError('train.reward_offset must be finite')
        if not 0 <= cfg.train.tau <= 1:
            raise ValueError('train.tau must be between zero and one')

    def _configure_optimizers(self):
        cfg = self.cfg
        self.actor_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=cfg.train.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.model.critics.parameters(), lr=cfg.train.critic_lr)
        self.alpha_optimizer = (torch.optim.Adam([self.model.log_alpha], lr=cfg.train.alpha_lr)
                                if self.model.autotune else None)

    def _prepare_replay(self):
        cfg = self.cfg
        self.replay = ResFiTReplayBuffer(
            cfg.replay_buffer.max_size, cfg.env.n_envs,
            n_step=1, gamma=cfg.train.gamma, seed=cfg.seed,
        )
        self._critic_warmed = False
        self._online_updates = 0
        self._pending_train_steps = 0
        self._update_credit = 0.

    def _training_action(self, features, base):
        if self.env_steps < self.cfg.train.buffer_warmup_steps:
            return base
        # The upstream actor is stochastic after data-anchored collection.
        return self.model.act(features, base, sample=True)

    def _replay_reward(self, reward):
        return np.asarray(reward) + self.cfg.train.reward_offset

    def _batch(self):
        return self.replay.sample(self.cfg.train.batch_size, self.device)

    def _update(self, actor=False, update_target=True, include_entropy=True):
        batch = self._batch()
        # Upstream clips each critic's gradient norm separately.
        metrics = {'critic_loss': self._step(
            self.critic_optimizer, self.model.critic_loss(batch, include_entropy=include_entropy),
            *(head.parameters() for head in self.model.critics.heads))}
        if actor:
            metrics['actor_loss'] = self._step(
                self.actor_optimizer, self.model.actor_loss(batch), self.model.actor.parameters())
            if self.alpha_optimizer is not None:
                metrics['alpha_loss'] = self._step(
                    self.alpha_optimizer, self.model.temperature_loss(batch))
        if update_target:
            self.model.update_targets(self.cfg.train.tau)
        self.update_steps += 1
        metrics['alpha'] = float(self.model.alpha.detach())
        return metrics

    def _train_on_replay(self):
        cfg = self.cfg
        if self.env_steps < cfg.train.buffer_warmup_steps or not len(self.replay):
            return {}
        metrics = {}
        if not self._critic_warmed:
            if cfg.train.critic_warmup_steps:
                log.info('Running %d DAWN critic-only warmup updates', cfg.train.critic_warmup_steps)
            # Upstream's default warmup target is hard: no entropy term.
            for index in range(cfg.train.critic_warmup_steps):
                metrics = self._update(
                    include_entropy=False,
                    update_target=(index + 1) % cfg.train.target_network_frequency == 0)
            self._critic_warmed = True
        self._pending_train_steps += cfg.env.n_envs
        if self._pending_train_steps < cfg.train.training_freq:
            return metrics
        transitions = (self._pending_train_steps // cfg.train.training_freq) * cfg.train.training_freq
        self._pending_train_steps -= transitions
        self._update_credit += transitions * cfg.train.utd_ratio
        updates = math.floor(self._update_credit)
        self._update_credit -= updates
        for _ in range(updates):
            self._online_updates += 1
            metrics = self._update(
                actor=self._online_updates % cfg.train.policy_frequency == 0,
                update_target=self._online_updates % cfg.train.target_network_frequency == 0)
        return metrics

    def _extra_checkpoint_state(self):
        return {'alpha_optimizer': (self.alpha_optimizer.state_dict()
                                    if self.alpha_optimizer is not None else None)}
