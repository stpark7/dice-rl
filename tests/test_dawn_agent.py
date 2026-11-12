"""DAWN rollout/update regression tests without simulator or BC assets."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from agent.finetune.train_dawn_agent import TrainDAWNAgent
from model.rl.dawn import DAWNModel


class ToyBC(nn.Module):
    action_dim, horizon_steps, cond_steps = 2, 2, 1

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.95))

    def forward(self, cond, deterministic=True):
        return SimpleNamespace(trajectories=self.weight.expand(len(cond['state']), 2, 2))


class ToyVectorEnv:
    def __init__(self, n_envs=2):
        self.n_envs = n_envs
        self.actions = []
        self.closed = False
        self.steps = np.zeros(n_envs, dtype=int)

    def seed(self, seeds):
        self.seeds = seeds

    def reset_arg(self, options_list):
        self.steps.fill(0)
        return {'state': np.zeros((self.n_envs, 1, 3), dtype=np.float32)}

    def step(self, actions):
        self.actions.append(np.array(actions, copy=True))
        self.steps += 1
        terminated = self.steps == 3
        truncated = (self.steps == 2) & (np.arange(self.n_envs) % 2 == 1)
        state = np.broadcast_to(self.steps[:, None, None], (self.n_envs, 1, 3)).astype(np.float32).copy()
        infos = [{'final_obs': {'state': state[index].copy()}} for index in range(self.n_envs)]
        done = terminated | truncated
        self.steps[done] = 0
        state[done] = 0
        return {'state': state}, terminated.astype(np.float32), terminated, truncated, infos

    def close(self):
        self.closed = True


class DAWNAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def config(self, **train_overrides):
        train = dict(num_train_steps=12, batch_size=3, utd_ratio=.25,
                     buffer_warmup_steps=4, critic_warmup_steps=0,
                     actor_lr=1e-3, critic_lr=1e-3, alpha_lr=1e-3,
                     gamma=.97, tau=.01, n_step=1, policy_frequency=1,
                     target_network_frequency=1, training_freq=4,
                     reward_offset=-1., max_grad_norm=50.,
                     eval_freq=0, save_freq=0, log_freq=0, render={'num': 0})
        train.update(train_overrides)
        return OmegaConf.create(dict(
            mode='train', checkpoint_path=None, device='cpu', seed=42,
            logdir=self.directory.name, wandb=None, run_eval=False,
            eval_base_policy=True, obs_dim=3, action_dim=2,
            horizon_steps=2, cond_steps=1, act_steps=1, bc_action_steps=2,
            env=dict(name='toy', env_type='toy', n_envs=2,
                     max_episode_steps=3, best_reward_threshold_for_success=1,
                     wrappers={'multi_step': {'n_action_steps': 1, 'reset_within_step': True}}),
            eval_n_envs=2, num_eval_episodes=3, eval_seed=10000,
            model=dict(_target_='model.rl.dawn.DAWNModel', obs_dim=3,
                       action_dim=2, horizon_steps=2, cond_steps=1,
                       residual_scale=.1, actor_hidden_dim=8,
                       critic_hidden_dim=8, critic_num_blocks=1,
                       critic_expansion=2, log_std_min=-20., log_std_max=2.,
                       sac_alpha=.01, autotune=True, target_entropy=None),
            replay_buffer={'max_size': 32}, train=train))

    def make_agent(self, cfg=None):
        cfg = cfg if cfg is not None else self.config()
        kwargs = OmegaConf.to_container(cfg.model)
        kwargs.pop('_target_')
        model = DAWNModel(pretrained_policy=ToyBC(), device='cpu', **kwargs)
        with patch('agent.finetune.train_resfit_agent.hydra.utils.instantiate', return_value=model):
            return TrainDAWNAgent(cfg)

    def test_online_only_warmup_fractional_utd_and_raw_replay(self):
        agent = self.make_agent()
        env = ToyVectorEnv()
        # Force an out-of-range residual sum, testing the execution boundary.
        def action(features, base, sample=False):
            return base + .1
        initial_bc = agent.model.base_policy.weight.detach().clone()
        with patch.object(agent, '_make_env', return_value=env), \
                patch.object(agent, '_prepare_demos', side_effect=AssertionError('demo access')), \
                patch.object(agent.model, 'act', side_effect=action):
            agent.run()
        self.assertEqual(agent.env_steps, 12)
        self.assertEqual(agent.update_steps, 2)
        self.assertEqual(len(agent.replay), 12)
        self.assertTrue(env.closed)
        for actions in env.actions[:2]:
            np.testing.assert_allclose(actions, .95)
        for actions in env.actions[2:]:
            np.testing.assert_allclose(actions, 1.)
        self.assertGreater(agent.replay._storage['action'][:12].max(), 1.)
        self.assertLessEqual(agent.replay._storage['reward'][:12].max(), 0.)
        self.assertTrue(agent.actor_optimizer.state)
        self.assertTrue(agent.alpha_optimizer.state)
        torch.testing.assert_close(agent.model.base_policy.weight, initial_bc)
        self.assertIsNone(agent.model.base_policy.weight.grad)
        self.assertTrue((Path(self.directory.name) / 'checkpoint/state_12.pt').is_file())

    def test_no_updates_if_run_ends_during_warmup(self):
        agent = self.make_agent(self.config(num_train_steps=2))
        env = ToyVectorEnv()
        with patch.object(agent, '_make_env', return_value=env), \
                patch.object(agent.model.actor, 'forward', side_effect=AssertionError('actor called')):
            agent.run()
        self.assertEqual(agent.update_steps, 0)
        self.assertFalse(agent.actor_optimizer.state)
        self.assertFalse(agent.critic_optimizer.state)
        self.assertFalse(agent.alpha_optimizer.state)

    def test_delay_and_target_update_frequencies(self):
        agent = self.make_agent(self.config(buffer_warmup_steps=0,
                                           training_freq=2, utd_ratio=.5,
                                           policy_frequency=2, target_network_frequency=3))
        env = ToyVectorEnv()
        with patch.object(agent, '_make_env', return_value=env), \
                patch.object(agent.model, 'actor_loss', wraps=agent.model.actor_loss) as actor, \
                patch.object(agent.model, 'update_targets', wraps=agent.model.update_targets) as target:
            agent.run()
        self.assertEqual(agent.update_steps, 6)
        self.assertEqual(actor.call_count, 3)
        self.assertEqual(target.call_count, 2)

    def test_critic_only_warmup_uses_hard_targets_once(self):
        agent = self.make_agent(self.config(critic_warmup_steps=3))
        env = ToyVectorEnv()
        with patch.object(agent, '_make_env', return_value=env), \
                patch.object(agent.model, 'critic_loss', wraps=agent.model.critic_loss) as critic, \
                patch.object(agent.model, 'actor_loss', wraps=agent.model.actor_loss) as actor:
            agent.run()
        self.assertEqual(agent.update_steps, 5)
        self.assertEqual(actor.call_count, 2)
        self.assertEqual(sum(call.kwargs.get('include_entropy') is False
                             for call in critic.call_args_list), 3)

    def test_gradient_clipping_is_per_critic_and_skips_temperature(self):
        agent = self.make_agent(self.config(max_grad_norm=1e-3))
        agent._prepare_replay()
        agent.replay.add(*(torch.randn(2, size) for size in (3, 2, 2, 1, 3, 2)),
                         [False, True], [False, False])
        clipped = []
        clip_grad_norm = torch.nn.utils.clip_grad_norm_

        def clip(parameters, maximum):
            clipped.append(list(parameters))
            return clip_grad_norm(clipped[-1], maximum)

        with patch.object(torch.nn.utils, 'clip_grad_norm_', side_effect=clip):
            agent._update(actor=True)
        self.assertEqual(len(clipped), 3)
        groups = [list(head.parameters()) for head in agent.model.critics.heads]
        groups.append(list(agent.model.actor.parameters()))
        for actual, expected in zip(clipped, groups):
            self.assertEqual(len(actual), len(expected))
            self.assertTrue(all(a is b for a, b in zip(actual, expected)))
            norm = torch.cat([p.grad.flatten() for p in expected]).norm()
            self.assertLessEqual(float(norm), 1e-3 + 1e-9)
        self.assertTrue(agent.alpha_optimizer.state)

    def test_checkpoint_reload_and_exact_evaluation_episode_count(self):
        cfg = self.config()
        agent = self.make_agent(cfg)
        with torch.no_grad():
            agent.model.log_alpha.fill_(-2.)
        agent.env_steps, agent.update_steps = 12, 2
        path = agent._save_checkpoint()
        checkpoint = torch.load(path, weights_only=False)
        self.assertIn('alpha_optimizer', checkpoint)
        cfg.mode, cfg.checkpoint_path = 'eval', str(path)
        loaded = self.make_agent(cfg)
        torch.testing.assert_close(loaded.model.log_alpha, torch.tensor(-2.).reshape_as(loaded.model.log_alpha))
        self.assertEqual(loaded.env_steps, 12)
        env = ToyVectorEnv()
        with patch.object(loaded, '_make_env', return_value=env):
            results = loaded.run()
        self.assertEqual(results['rl']['num_episodes'], 3)
        self.assertEqual(results['bc']['num_episodes'], 3)
        self.assertGreater(results['bc']['mean_reward'], 0.)
        self.assertTrue(env.closed)
        cfg.mode = 'train'
        warmstarted = self.make_agent(cfg)
        self.assertEqual(warmstarted.env_steps, 0)
        self.assertEqual(warmstarted.update_steps, 0)
        self.assertFalse(warmstarted.actor_optimizer.state)

    def test_checkpoint_rejects_incompatible_dawn_settings(self):
        cfg = self.config()
        path = self.make_agent(cfg)._save_checkpoint()
        cfg.mode, cfg.checkpoint_path = 'eval', str(path)
        cfg.model.log_std_min = -10.
        with self.assertRaisesRegex(ValueError, 'log_std_min'):
            self.make_agent(cfg)

    def test_invalid_training_settings_fail_before_model_loading(self):
        for setting in ({'utd_ratio': 0}, {'utd_ratio': float('nan')},
                        {'batch_size': 0}, {'policy_frequency': 0},
                        {'target_network_frequency': 1.5}, {'training_freq': 0},
                        {'n_step': 2}, {'buffer_warmup_steps': 33},
                        {'critic_warmup_steps': 1.5}):
            with self.subTest(setting=setting), \
                    patch('agent.finetune.train_resfit_agent.hydra.utils.instantiate',
                          side_effect=AssertionError('model loaded')):
                with self.assertRaises(ValueError):
                    TrainDAWNAgent(self.config(**setting))


if __name__ == '__main__':
    unittest.main()
