"""Raw-base and zero-residual codec baselines share evaluation episodes."""

import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

from agent.finetune.train_distill_residual_flow_agent import TrainDistillResidualFlowAgent
from agent.finetune.train_distill_residual_flow_img_agent import TrainDistillResidualFlowImgAgent
from model.hand_chord.temporal_hand_chord import TemporalHandActionChord


class FixedBase(nn.Module):
    def __init__(self, actions):
        super().__init__()
        self.actions = actions
        self.conditions = []

    def forward(self, cond):
        self.conditions.append(cond)
        return SimpleNamespace(trajectories=self.actions.clone())


class TinyEvalEnv:
    def step(self, actions):
        return ({'state': np.zeros((2, 1, 23), dtype=np.float32)},
                np.array([1., 0.]), np.ones(2, dtype=bool), np.zeros(2, dtype=bool), {})


class TrainingStarts(Exception):
    pass


class HandChordProjectionEvaluationTests(unittest.TestCase):
    def test_helpers_project_full_horizon_with_measured_q0_before_execution_slice(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            normalizer = Path(directory) / 'normalization.npz'
            np.savez(normalizer, action_min=np.full(22, -1.), action_max=np.full(22, 1.),
                     obs_min=np.full(23, -2.), obs_max=np.full(23, 2.))
            chord = TemporalHandActionChord(normalization_path=normalizer, horizon_steps=4,
                                            latent_dim=3, width=8, layers=1, heads=2).eval()
            raw = torch.linspace(-.5, .5, 2 * 4 * 22).reshape(2, 4, 22)
            state = np.linspace(-.4, .4, 2 * 2 * 23, dtype=np.float32).reshape(2, 2, 23)
            with torch.no_grad():
                expected = chord.decode(chord.encode(raw, torch.from_numpy(state)), torch.from_numpy(state))
            for agent_class in (TrainDistillResidualFlowAgent, TrainDistillResidualFlowImgAgent):
                with self.subTest(agent=agent_class.__name__):
                    agent = object.__new__(agent_class)
                    base = FixedBase(raw)
                    agent.model = SimpleNamespace(pretrained_flow_policy=base, chord=chord,
                                                  actor=Mock(side_effect=AssertionError('actor must not run')))
                    agent.device, agent.act_steps = 'cpu', 2
                    obs = {'state': state, 'rgb': np.zeros((2, 1, 8, 8, 6), dtype=np.float32)}
                    with patch.object(chord, 'encode', wraps=chord.encode) as encode:
                        projected = agent._get_flow_action_for_eval(obs, project_with_chord=True)
                        args, _ = encode.call_args
                        self.assertEqual(args[0].shape, (2, 4, 22))
                        torch.testing.assert_close(args[1], torch.from_numpy(state))
                    np.testing.assert_allclose(projected, expected[:, :2].numpy(), rtol=1e-5, atol=1e-6)
                    np.testing.assert_array_equal(agent._get_flow_action_for_eval(obs), raw[:, :2].numpy())
                    agent.model.actor.assert_not_called()

    def test_projection_without_codec_has_clear_error_and_raw_helper_stays_compatible(self):
        for agent_class in (TrainDistillResidualFlowAgent, TrainDistillResidualFlowImgAgent):
            with self.subTest(agent=agent_class.__name__):
                agent = object.__new__(agent_class)
                raw = torch.zeros(2, 4, 22)
                agent.model = SimpleNamespace(pretrained_flow_policy=FixedBase(raw))
                agent.device, agent.act_steps = 'cpu', 2
                obs = {'state': np.zeros((2, 23), dtype=np.float32),
                       'rgb': np.zeros((2, 8, 8, 6), dtype=np.float32)}
                np.testing.assert_array_equal(agent._get_flow_action_for_eval(obs), raw[:, :2].numpy())
                with self.assertRaisesRegex(ValueError, 'chord'):
                    agent._get_flow_action_for_eval(obs, project_with_chord=True)
                with self.assertRaisesRegex(ValueError, 'chord'):
                    agent.evaluate_pretrained_flow_policy(project_with_chord=True)

    def test_evaluator_labels_baselines_and_reuses_fixed_episode_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = object.__new__(TrainDistillResidualFlowAgent)
            agent.model = SimpleNamespace(eval=Mock(), chord=object())
            agent.num_eval_episodes, agent.eval_n_envs = 2, 2
            agent.eval_seeds = [10000, 10001, 10002, 10003]
            agent.max_episode_steps, agent.act_steps = 2, 2
            agent.eval_venv = TinyEvalEnv()
            agent.eval_csv_path = str(Path(directory) / 'evaluation_results.csv')
            agent._init_eval_csv()
            agent.reset_env_all = Mock(return_value={'state': np.zeros((2, 1, 23), dtype=np.float32)})
            agent._get_flow_action_for_eval = Mock(return_value=np.zeros((2, 2, 22)))
            raw_result = agent.evaluate_pretrained_flow_policy()
            raw_resets = agent.reset_env_all.call_args_list.copy()
            agent.reset_env_all.reset_mock()
            projected_result = agent.evaluate_pretrained_flow_policy(project_with_chord=True)
            self.assertEqual(agent.reset_env_all.call_args_list, raw_resets)
            self.assertEqual([call.kwargs['options_venv'] for call in raw_resets],
                             [[{'seed': 10000}, {'seed': 10001}], [{'seed': 10002}, {'seed': 10003}]])
            for call in agent._get_flow_action_for_eval.call_args_list[-2:]:
                self.assertTrue(call.kwargs['project_with_chord'])
            self.assertEqual(raw_result[1:], projected_result[1:])
            with open(agent.eval_csv_path) as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual([row['eval_type'] for row in records], ['pretrained', 'codec_projection'])
            self.assertEqual([row['step'] for row in records], ['0', '0'])
            self.assertEqual([row['num_episodes'] for row in records], ['4', '4'])

    def test_projection_reuses_base_noise_without_advancing_training_rng(self):
        agent = object.__new__(TrainDistillResidualFlowAgent)
        agent.device = 'cpu'
        agent.model = SimpleNamespace(train=Mock(), chord=object(),
                                      pretrained_flow_policy=SimpleNamespace(eval=Mock()))
        agent.cfg = {'run_eval': True}
        agent.eval_venv = object()
        agent.visualize_trajectories = False
        samples = []

        def evaluate(project_with_chord=False):
            samples.append(torch.rand(3))
            return [], .0, .0

        agent.evaluate_pretrained_flow_policy = evaluate
        agent.reset_env_all = Mock(side_effect=TrainingStarts)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            expected_baseline = torch.rand(3)
            expected_training = torch.rand(3)
            torch.manual_seed(123)
            with self.assertRaises(TrainingStarts):
                agent.run()
            torch.testing.assert_close(samples[0], expected_baseline)
            torch.testing.assert_close(samples[1], expected_baseline)
            torch.testing.assert_close(torch.rand(3), expected_training)

    def test_run_evaluates_projection_only_for_chord_before_training_starts(self):
        for has_chord in (False, True):
            with self.subTest(has_chord=has_chord):
                agent = object.__new__(TrainDistillResidualFlowAgent)
                agent.model = SimpleNamespace(train=Mock(), pretrained_flow_policy=SimpleNamespace(eval=Mock()))
                if has_chord:
                    agent.model.chord = object()
                agent.device = 'cpu'
                agent.cfg = {'run_eval': True}
                agent.eval_venv = object()
                agent.visualize_trajectories = False
                agent.evaluate_pretrained_flow_policy = Mock(return_value=([], .0, .0))
                agent.reset_env_all = Mock(side_effect=TrainingStarts)
                with self.assertRaises(TrainingStarts):
                    agent.run()
                calls = agent.evaluate_pretrained_flow_policy.call_args_list
                self.assertEqual(len(calls), 2 if has_chord else 1)
                self.assertEqual(calls[0].kwargs, {})
                if has_chord:
                    self.assertEqual(calls[1].kwargs, {'project_with_chord': True})


if __name__ == '__main__':
    unittest.main()
