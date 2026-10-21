"""Regression tests for shared pretraining resets and image rollouts."""

from collections import namedtuple
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from omegaconf import OmegaConf
import torch

from agent.pretrain.train_agent import PreTrainAgent
from agent.pretrain.train_flow_matching_img_agent import TrainFlowMatchingImgAgent


class PretrainSeedingTests(unittest.TestCase):
    def agent(self):
        agent = PreTrainAgent.__new__(PreTrainAgent)
        cfg = OmegaConf.create({
            'device': 'cpu', 'seed': 42, 'wandb': None,
            'env': {'name': 'pick_bucket', 'env_type': 'dexjoco',
                    'n_envs': 2, 'max_episode_steps': 10},
            'obs_dim': 23, 'action_dim': 22, 'cond_steps': 1,
            'act_steps': 1, 'horizon_steps': 1, 'model': {},
        })
        env = Mock()
        env.reset_arg.return_value = {'state': np.zeros((2, 23))}
        with patch('agent.pretrain.train_agent.make_async', return_value=env), \
                patch('agent.pretrain.train_agent.hydra.utils.instantiate',
                      side_effect=RuntimeError('stop before model')), \
                patch('agent.pretrain.train_agent.torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'stop before model'):
                agent.__init__(cfg)
        return agent

    def test_first_reset_seeds_each_worker_only_once(self):
        agent = self.agent()
        agent.venv.seed.assert_not_called()
        agent.reset_env_all()
        self.assertEqual(agent.venv.reset_arg.call_args.kwargs['options_list'],
                         [{'seed': 42}, {'seed': 43}])
        agent.reset_env_all()
        self.assertEqual(agent.venv.reset_arg.call_args.kwargs['options_list'], [{}, {}])

    def test_explicit_options_are_preserved_without_mutation(self):
        agent = self.agent()
        options = [{'seed': 9, 'video_path': '/tmp/episode.mp4'}, {}]
        agent.reset_env_all(options_venv=options)
        self.assertEqual(agent.venv.reset_arg.call_args.kwargs['options_list'],
                         [{'seed': 9, 'video_path': '/tmp/episode.mp4'}, {'seed': 43}])
        self.assertEqual(options, [{'seed': 9, 'video_path': '/tmp/episode.mp4'}, {}])

    def test_single_reset_consumes_only_that_workers_initial_seed(self):
        agent = self.agent()
        agent.reset_env(1)
        self.assertEqual(agent.venv.reset_one_arg.call_args.kwargs['options'], {'seed': 43})
        agent.reset_env_all()
        self.assertEqual(agent.venv.reset_arg.call_args.kwargs['options_list'],
                         [{'seed': 42}, {}])


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def loss(self):
        return self.weight.square()

    def forward(self, cond):
        return SimpleNamespace(trajectories=torch.zeros(2, 16, 44))


class RolloutEnvironment:
    def __init__(self, results):
        self.results = results
        self.steps = 0
        self.calls = []
        self.action_lengths = []

    def step(self, actions):
        self.action_lengths.append(actions.shape[1])
        result = self.results[self.steps]
        self.steps += 1
        return {}, np.array(result[0]), np.array(result[1]), np.zeros(2, dtype=bool), {}

    def call(self, name):
        self.calls.append(name)


class ImageEvaluationTests(unittest.TestCase):
    def agent(self, results, max_steps):
        agent = TrainFlowMatchingImgAgent.__new__(TrainFlowMatchingImgAgent)
        agent.cfg = OmegaConf.create({'env': {'env_type': 'dexjoco'}})
        agent.n_epochs = 1
        agent.dataloader_train = [namedtuple('Batch', [])()]
        agent.dataset_train = SimpleNamespace(device='cpu')
        agent.dataloader_val = None
        agent.val_freq = 1
        agent.n_envs = 2
        agent.n_render = 1
        agent.render_dir = '/tmp'
        agent.reset_env_all = lambda **kwargs: {}
        agent.model = Policy()
        agent.optimizer = torch.optim.SGD(agent.model.parameters(), lr=0.01)
        agent.device = 'cpu'
        agent.obs_dims = {}
        agent.act_steps = 8
        agent.horizon_steps = 16
        agent.max_episode_steps = max_steps
        agent.venv = RolloutEnvironment(results)
        agent.best_reward_threshold_for_success = 1
        agent.use_wandb = False
        agent.lr_scheduler = SimpleNamespace(step=lambda: None)
        agent.step_ema = lambda: None
        agent.save_model_freq = 1
        agent.save_model = lambda: None
        agent.log_freq = 1
        return agent

    def run_agent(self, agent):
        with patch('agent.pretrain.train_flow_matching_img_agent.wandb.finish'), \
                patch('agent.pretrain.train_flow_matching_img_agent.log.info') as info:
            agent.run()
        return [str(call.args[0]) for call in info.call_args_list]

    def test_image_agent_honors_configured_evaluation_frequency(self):
        cfg = OmegaConf.create({
            'train': {'val_freq': 7},
            'shape_meta': {'obs': {'state': {'shape': [46]}}},
        })

        def initialize_parent(agent, config):
            agent.val_freq = config.train.val_freq

        with patch('agent.pretrain.train_flow_matching_img_agent.TrainFlowMatchingAgent.__init__',
                   initialize_parent):
            agent = TrainFlowMatchingImgAgent(cfg)
        self.assertEqual(agent.val_freq, 7)

    def test_rollout_uses_action_steps_and_finalizes_video_at_cap(self):
        agent = self.agent([([0, 0], [False, False])] * 3, max_steps=17)
        self.run_agent(agent)
        self.assertEqual(agent.venv.action_lengths, [8, 8, 1])
        self.assertIn('stop_video', agent.venv.calls)

    def test_success_counts_only_first_episode_after_auto_reset(self):
        agent = self.agent([
            ([0, 0], [True, False]),
            ([1, 0], [False, True]),
        ], max_steps=32)
        logs = self.run_agent(agent)
        self.assertIn('eval: success rate   0.0000 ', logs)
        self.assertEqual(agent.venv.steps, 2)


if __name__ == '__main__':
    unittest.main()
