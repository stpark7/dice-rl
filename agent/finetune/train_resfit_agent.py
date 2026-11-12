"""ResFiT post-training of frozen, action-chunked Dexjoco BC policies.

All rollout and scheduling units are individual environment transitions, summed
across vector workers. The native BC visual encoder is frozen; replay stores
its features instead of images. TrainDAWNAgent reuses this rollout, evaluation
and checkpoint machinery through the underscore-prefixed hooks.
"""

import csv
import hashlib
import json
import logging
from pathlib import Path
import random

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from util.resfit_replay_buffer import BaseActionQueue, ResFiTReplayBuffer

log = logging.getLogger(__name__)


def replay_next_observation(next_obs, truncated, infos):
    """Recover pre-reset timeout observations for bootstrapping."""
    result = {key: np.array(value, copy=True) for key, value in next_obs.items()}
    for index in np.flatnonzero(truncated):
        if 'final_obs' not in infos[index]:
            raise ValueError('Timeout replay requires info["final_obs"] before auto-reset')
        for key in result:
            result[key][index] = infos[index]['final_obs'][key]
    return result


class TrainResFiTAgent:
    algorithm = 'ResFiT'
    model_config_fields = (
        'residual_scale', 'exploration_std', 'target_noise_clip',
        'actor_hidden_dims', 'critic_hidden_dims', 'critic_ensemble_size',
        'target_subset_size',
    )

    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = cfg.get('mode', 'train')
        if self.mode not in ('train', 'eval'):
            raise ValueError('mode must be train or eval')
        if cfg.act_steps != 1 or cfg.env.wrappers.multi_step.n_action_steps != 1:
            raise ValueError(f'{self.algorithm} requires act_steps=1 and multi_step.n_action_steps=1')
        if not cfg.env.wrappers.multi_step.reset_within_step:
            raise ValueError(f'{self.algorithm} requires multi_step.reset_within_step=true')
        if not 1 <= cfg.bc_action_steps <= cfg.horizon_steps:
            raise ValueError('bc_action_steps must be between 1 and horizon_steps')
        self._validate_training_config()
        if min(cfg.env.n_envs, cfg.eval_n_envs, cfg.num_eval_episodes,
               cfg.env.max_episode_steps) < 1:
            raise ValueError('Environment and evaluation counts must be positive')
        if min(cfg.train.buffer_warmup_steps, cfg.train.critic_warmup_steps) < 0:
            raise ValueError('Warmup lengths cannot be negative')
        if self.mode == 'eval' and not cfg.get('checkpoint_path'):
            raise ValueError('mode=eval requires checkpoint_path')

        self.device = torch.device(cfg.device)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.logdir = Path(cfg.logdir)
        self.checkpoint_dir = self.logdir / 'checkpoint'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model = hydra.utils.instantiate(cfg.model).to(self.device)
        self._configure_optimizers()
        self.env_steps = 0
        self.update_steps = 0
        if cfg.get('checkpoint_path'):
            self._load_checkpoint(cfg.checkpoint_path)
        self.wandb_run = None
        if cfg.get('wandb') is not None:
            import wandb
            self.wandb_run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.get('entity'),
                name=self.logdir.name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

    def _validate_training_config(self):
        if self.cfg.train.batch_size < 2 or self.cfg.train.batch_size % 2:
            raise ValueError('ResFiT requires an even batch_size for equal demo/online sampling')
        if self.cfg.train.utd_ratio < 1 or int(self.cfg.train.utd_ratio) != self.cfg.train.utd_ratio:
            raise ValueError('train.utd_ratio must be a positive integer')

    def _configure_optimizers(self):
        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor.parameters(), lr=self.cfg.train.actor_lr)
        self.critic_optimizer = torch.optim.AdamW(
            self.model.critics.parameters(), lr=self.cfg.train.critic_lr)

    def _make_env(self, n_envs):
        from env.gym_utils import make_async
        cfg = self.cfg
        return make_async(
            cfg.env.name, env_type=cfg.env.env_type, num_envs=n_envs,
            asynchronous=True, wrappers=cfg.env.wrappers,
            max_episode_steps=cfg.env.max_episode_steps,
            shape_meta=cfg.get('shape_meta'), obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            use_image_obs=cfg.env.get('use_image_obs', True),
            render_offscreen=cfg.env.get('save_video', False),
        )

    def _queue(self, n_envs):
        return BaseActionQueue(self.model, n_envs, self.cfg.bc_action_steps,
                               self.cfg.action_dim, self.device)

    @staticmethod
    def _reset(env, n_envs, seeds=None, videos=None):
        options = [{} for _ in range(n_envs)]
        if seeds is not None:
            for option, seed in zip(options, seeds):
                option['seed'] = int(seed)
        if videos:
            for index, path in videos.items():
                options[index]['video_path'] = str(path)
        obs = env.reset_arg(options_list=options)
        if isinstance(obs, list):
            obs = {key: np.stack([row[key] for row in obs]) for key in obs[0]}
        return obs

    def _prepare_demos(self):
        from util.resfit_demos import prepare_resfit_demos
        return prepare_resfit_demos(self.model, self.cfg)

    def _prepare_replay(self):
        cfg = self.cfg
        self.demos = self._prepare_demos()
        if not len(self.demos):
            raise ValueError('ResFiT requires a nonempty expert dataset')
        self.replay = ResFiTReplayBuffer(
            cfg.replay_buffer.max_size, cfg.env.n_envs,
            n_step=cfg.train.n_step, gamma=cfg.train.gamma, seed=cfg.seed,
        )
        self._critic_warmed = False
        self._online_updates = 0

    def _training_action(self, features, base):
        if self.env_steps < self.cfg.train.buffer_warmup_steps:
            scale = self.cfg.train.warmup_noise_scale
            noise = torch.empty_like(base).uniform_(-scale, scale)
            return (base + noise).clamp(-1, 1)
        return self.model.act(features, base, std=self.model.exploration_std)

    def _replay_reward(self, reward):
        return reward

    def _train_on_replay(self):
        cfg = self.cfg
        metrics = {}
        if self.env_steps >= cfg.train.buffer_warmup_steps and len(self.replay):
            if not self._critic_warmed:
                log.info('Running %d critic-only warmup updates', cfg.train.critic_warmup_steps)
                for _ in range(cfg.train.critic_warmup_steps):
                    metrics = self._update(actor=False)
                self._critic_warmed = True
            for _ in range(cfg.train.utd_ratio * cfg.env.n_envs):
                self._online_updates += 1
                metrics = self._update(actor=self._online_updates % cfg.train.utd_ratio == 0)
        return metrics

    def _batch(self):
        half = self.cfg.train.batch_size // 2
        online = self.replay.sample(half, self.device)
        demos = self.demos.sample(half, self.device)
        return {key: torch.cat([online[key], demos[key]], dim=0) for key in online}

    def _step(self, optimizer, loss, *clipped):
        """Apply one optimizer step, clipping each given parameter group's norm."""
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        maximum = self.cfg.train.get('max_grad_norm')
        if maximum is not None:
            for parameters in clipped:
                torch.nn.utils.clip_grad_norm_(parameters, maximum)
        optimizer.step()
        return float(loss.detach())

    def _update(self, actor=False):
        batch = self._batch()
        metrics = {'critic_loss': self._step(
            self.critic_optimizer, self.model.critic_loss(batch), self.model.critics.parameters())}
        if actor:
            metrics['actor_loss'] = self._step(
                self.actor_optimizer, self.model.actor_loss(batch), self.model.actor.parameters())
        self.model.update_targets(self.cfg.train.tau, update_actor=actor)
        self.update_steps += 1
        return metrics

    def _extra_checkpoint_state(self):
        return {}

    def _save_checkpoint(self):
        path = self.checkpoint_dir / f'state_{self.env_steps}.pt'
        torch.save({
            'model': self.model.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'env_steps': self.env_steps, 'update_steps': self.update_steps,
            'cfg': OmegaConf.to_container(self.cfg, resolve=True),
            'observation_settings': self._observation_settings(),
            'base_policy_settings': self._base_policy_settings(),
            **self._extra_checkpoint_state(),
        }, path)
        log.info('Saved %s checkpoint to %s', self.algorithm, path)
        return path

    def _observation_settings(self):
        wrapper = self.cfg.env.wrappers.get('dexjoco_image', {})
        settings = {}
        for key in ('image_keys', 'low_dim_keys'):
            value = wrapper.get(key)
            settings[key] = None if value is None else list(value)
        shape_meta = self.cfg.get('shape_meta')
        settings['shape_meta'] = OmegaConf.to_container(shape_meta) if shape_meta is not None else None
        normalizer = wrapper.get('normalization_path')
        settings['normalization'] = None
        if normalizer:
            digest = hashlib.sha256()
            with np.load(normalizer, allow_pickle=False) as arrays:
                for key in sorted(arrays.files):
                    array = arrays[key]
                    digest.update(str((key, array.dtype.str, array.shape)).encode())
                    digest.update(array.tobytes())
            settings['normalization'] = digest.hexdigest()
        return settings

    def _base_policy_settings(self):
        policy = getattr(self.model, 'base_policy', None)
        fields = ('flow_steps', 'integration_method', 'schedule', 'init_scale',
                  'denoising_steps', 'use_ddim', 'ddim_steps', 'predict_epsilon')
        return {key: getattr(policy, key) for key in fields if hasattr(policy, key)}

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        saved_cfg = checkpoint.get('cfg', {})
        for key in ('obs_dim', 'action_dim', 'horizon_steps', 'cond_steps', 'bc_action_steps'):
            if key in saved_cfg and saved_cfg[key] != self.cfg[key]:
                raise ValueError(f'Checkpoint {key}={saved_cfg[key]} differs from Hydra config {self.cfg[key]}')
        if saved_cfg.get('env', {}).get('name', self.cfg.env.name) != self.cfg.env.name:
            raise ValueError('Checkpoint environment differs from Hydra env.name')
        observation_settings = self._observation_settings()
        if checkpoint.get('observation_settings', observation_settings) != observation_settings:
            raise ValueError('Checkpoint observation settings differ: check cameras, state keys, shape_meta and normalization')
        base_policy_settings = self._base_policy_settings()
        if checkpoint.get('base_policy_settings', base_policy_settings) != base_policy_settings:
            raise ValueError('Checkpoint BC inference settings differ: check the saved base policy Hydra config')
        saved_model = saved_cfg.get('model', {})
        if saved_model.get('_target_', self.cfg.model.get('_target_')) != self.cfg.model.get('_target_'):
            raise ValueError('Checkpoint model target differs from Hydra config')
        for key in self.model_config_fields:
            if key in saved_model and saved_model[key] != self.cfg.model.get(key):
                raise ValueError(f'Checkpoint model.{key} differs from Hydra config; use the training value')
        self.model.load_state_dict(checkpoint['model'])
        if self.mode == 'eval':
            self.env_steps = checkpoint.get('env_steps', 0)
            self.update_steps = checkpoint.get('update_steps', 0)
        else:
            log.info('Warm-started %s weights; using fresh replay, optimizers and warmup', self.algorithm)

    def _record_evaluation(self, metrics):
        path = self.logdir / 'eval.csv'
        fields = ['env_steps', 'policy', 'num_episodes', 'mean_reward', 'std_reward',
                  'success_rate', 'mean_length']
        write_header = not path.exists()
        with path.open('a', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            if write_header:
                writer.writeheader()
            for policy, result in metrics.items():
                writer.writerow(dict(env_steps=self.env_steps, policy=policy, **result))
        with (self.logdir / 'eval_metrics.json').open('w') as output:
            json.dump(dict(env_steps=self.env_steps, **metrics), output, indent=2)
        for policy, result in metrics.items():
            log.info('%s at %d transitions: success %.3f, return %.3f, length %.1f (%d episodes)',
                     policy, self.env_steps, result['success_rate'], result['mean_reward'],
                     result['mean_length'], result['num_episodes'])
            if self.wandb_run is not None:
                self.wandb_run.log({f'eval/{policy}/{key}': value for key, value in result.items()},
                                   step=self.env_steps)

    @torch.no_grad()
    def evaluate(self, env, use_residual=True):
        """Count exactly num_eval_episodes, using the same seeds for BC and RL."""
        cfg = self.cfg
        rewards, lengths = [], []
        was_training = self.model.training
        self.model.eval()
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(cfg.eval_seed)
                queue = self._queue(cfg.eval_n_envs)
                for start in range(0, cfg.num_eval_episodes, cfg.eval_n_envs):
                    count = min(cfg.eval_n_envs, cfg.num_eval_episodes - start)
                    seeds = range(cfg.eval_seed + start, cfg.eval_seed + start + cfg.eval_n_envs)
                    videos = {}
                    if start == 0 and cfg.env.get('save_video', False):
                        directory = self.logdir / 'render'
                        directory.mkdir(exist_ok=True)
                        for index in range(min(cfg.train.render.num, count)):
                            policy = 'rl' if use_residual else 'bc'
                            videos[index] = directory / f'{policy}_step-{self.env_steps}_env-{index}.mp4'
                    obs = self._reset(env, cfg.eval_n_envs, seeds, videos)
                    queue.reset()
                    returns = np.zeros(cfg.eval_n_envs)
                    durations = np.zeros(cfg.eval_n_envs, dtype=int)
                    finished = np.arange(cfg.eval_n_envs) >= count
                    for _ in range(cfg.env.max_episode_steps):
                        base = queue.next(obs)
                        action = self.model.act(self.model.encode(obs), base) if use_residual else base
                        obs, reward, terminated, truncated, _ = env.step(
                            action.clamp(-1, 1)[:, None].cpu().numpy())
                        active = ~finished
                        returns[active] += np.asarray(reward).reshape(-1)[active]
                        durations[active] += 1
                        done = np.asarray(terminated) | np.asarray(truncated)
                        finished |= done
                        queue.reset(np.flatnonzero(done))
                        if finished.all():
                            break
                    rewards.extend(returns[:count])
                    lengths.extend(durations[:count])
        finally:
            self.model.train(was_training)
        return dict(num_episodes=len(rewards), mean_reward=float(np.mean(rewards)),
                    std_reward=float(np.std(rewards)), mean_length=float(np.mean(lengths)),
                    success_rate=float(np.mean(np.asarray(rewards) >= cfg.env.best_reward_threshold_for_success)))

    def run(self):
        cfg = self.cfg
        training_env = None
        evaluation_env = None
        try:
            if self.mode == 'eval' or cfg.get('run_eval', True):
                evaluation_env = self._make_env(cfg.eval_n_envs)
                evaluation_env.seed([cfg.eval_seed + index for index in range(cfg.eval_n_envs)])
            if self.mode == 'eval':
                results = {'rl': self.evaluate(evaluation_env)}
                if cfg.get('eval_base_policy', True):
                    results['bc'] = self.evaluate(evaluation_env, use_residual=False)
                self._record_evaluation(results)
                return results

            self._prepare_replay()
            last_eval_steps = None
            if evaluation_env is not None:
                results = {'rl': self.evaluate(evaluation_env)}
                if cfg.get('eval_base_policy', True):
                    results['bc'] = self.evaluate(evaluation_env, use_residual=False)
                self._record_evaluation(results)
                last_eval_steps = self.env_steps
            training_env = self._make_env(cfg.env.n_envs)
            training_env.seed([cfg.seed + index for index in range(cfg.env.n_envs)])
            obs = self._reset(training_env, cfg.env.n_envs)
            queue = self._queue(cfg.env.n_envs)
            base = queue.next(obs)
            metrics = {}
            while self.env_steps < cfg.train.num_train_steps:
                previous_steps = self.env_steps
                with torch.no_grad():
                    features = self.model.encode(obs)
                    action = self._training_action(features, base)
                    next_obs, reward, terminated, truncated, infos = training_env.step(
                        action.clamp(-1, 1)[:, None].cpu().numpy())
                    transition_obs = replay_next_observation(next_obs, truncated, infos)
                    next_features = self.model.encode(transition_obs)
                    next_base = queue.next(transition_obs)
                    self.replay.add(features, base, action, self._replay_reward(reward), next_features, next_base,
                                    terminated, truncated)
                    done_indices = np.flatnonzero(np.asarray(terminated) | np.asarray(truncated))
                    queue.reset(done_indices)
                    if len(done_indices):
                        next_base[done_indices] = queue.next(next_obs, indices=done_indices)
                    obs, base = next_obs, next_base
                self.env_steps += cfg.env.n_envs
                updates = self._train_on_replay()
                if updates:
                    metrics = updates
                for name in ('eval', 'save', 'log'):
                    frequency = cfg.train.get(f'{name}_freq', 0)
                    if frequency <= 0 or self.env_steps // frequency == previous_steps // frequency:
                        continue
                    if name == 'eval' and evaluation_env is not None:
                        self._record_evaluation({'rl': self.evaluate(evaluation_env)})
                        last_eval_steps = self.env_steps
                    elif name == 'save':
                        self._save_checkpoint()
                    elif name == 'log':
                        log.info('%s transitions=%d updates=%d replay=%d %s',
                                 self.algorithm, self.env_steps, self.update_steps, len(self.replay), metrics)
                        if self.wandb_run is not None:
                            self.wandb_run.log(dict(env_steps=self.env_steps, **metrics), step=self.env_steps)
            self._save_checkpoint()
            if evaluation_env is not None and last_eval_steps != self.env_steps:
                self._record_evaluation({'rl': self.evaluate(evaluation_env)})
        finally:
            if training_env is not None:
                training_env.close()
            if evaluation_env is not None:
                evaluation_env.close()
            if self.wandb_run is not None:
                self.wandb_run.finish()
