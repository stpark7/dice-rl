import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import gym

from agent.dataset.sequence import StitchedSequenceQLearningDataset
from env.gym_utils import make_async
from agent.finetune.train_distill_residual_flow_agent import TrainDistillResidualFlowAgent
from env.gym_utils.wrapper.multi_step_full import MultiStepFull


class DexjocoRLSpaceTests(unittest.TestCase):
    def test_full_trajectory_dummy_space_matches_worker_history(self):
        # Async shared-memory spaces must include the same history axis as
        # MultiStepFull workers, even with augmented model obs_dim=151.
        with patch('env.gym_utils.async_vector_env.AsyncVectorEnv',
                   side_effect=lambda env_fns, dummy_env_fn: dummy_env_fn()):
            env = make_async(
                'pick_bucket', env_type='dexjoco', obs_dim=151, action_dim=22,
                shape_meta={'obs': {'rgb': {'shape': [96, 96, 6]},
                                    'state': {'shape': [23]}}},
                wrappers={'multi_step_full': {'n_obs_steps': 1, 'n_action_steps': 8}},
            )
        self.assertEqual(env.observation_space['state'].shape, (1, 23))
        self.assertEqual(env.observation_space['rgb'].shape, (1, 96, 96, 6))
        self.assertEqual(env.action_space.shape, (8, 22))

    def test_false_environment_truncation_flag_preserves_wrapper_time_limit(self):
        class CountingEnv(gym.Env):
            observation_space = gym.spaces.Box(-100, 100, shape=(1,))
            action_space = gym.spaces.Box(-1, 1, shape=(1,))

            def reset(self, **kwargs):
                self.count = 0
                return np.zeros(1, dtype=np.float32)

            def step(self, action):
                self.count += 1
                return (np.array([self.count], dtype=np.float32), 0., False,
                        {'TimeLimit.truncated': False})

        env = MultiStepFull(CountingEnv(), n_action_steps=8,
                            max_episode_steps=3, reset_within_step=True)
        env.reset()
        for _ in range(2):
            obs, _, terminated, truncated, info = env.step(np.zeros((8, 1)))
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info['full_trajectory']['dones'], [False, False, True])
            self.assertTrue(info['full_trajectory']['include_initial'])
            self.assertEqual(obs.item(), 0.)  # reset observation


class ExpertReturnTests(unittest.TestCase):
    def dataset(self, directory):
        # Two episodes make crossing a terminal boundary observable.
        path = Path(directory) / 'train.npz'
        values = np.arange(60, dtype=np.float32)
        rewards = np.zeros(60, dtype=np.float32)
        rewards[[29, 59]] = 1
        np.savez(path, states=values[:, None], actions=values[:, None],
                 images=values[:, None, None, None], traj_lengths=[30, 30],
                 rewards=rewards, terminals=rewards)
        return StitchedSequenceQLearningDataset(
            str(path), horizon_steps=8, cond_steps=1, img_cond_steps=1,
            device='cpu', use_img=True, get_mc_return=False,
            use_n_step=True, n_step=3, gamma=.9,
            success_steps_for_termination=1,
        )

    def test_n_step_bootstrap_advances_three_chunks_for_state_and_image(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = self.dataset(directory)[0]
        self.assertEqual(sample.conditions['next_state'].item(), 24)
        self.assertEqual(sample.conditions['next_rgb'].item(), 24)
        self.assertEqual(sample.dones.item(), 0)

    def test_partial_final_chunk_keeps_success_without_crossing_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.dataset(directory)
            for start in (10, 40):
                index = next(i for i, pair in enumerate(dataset.indices) if pair[0] == start)
                sample = dataset[index]
                self.assertAlmostEqual(sample.rewards.item(), .9 ** 2, places=6)
                self.assertEqual(sample.dones.item(), 1)
                self.assertEqual(sample.conditions['next_state'].item(), 0)
                self.assertEqual(sample.conditions['next_rgb'].item(), 0)


class BaselineEvaluationTests(unittest.TestCase):
    def test_auto_reset_success_does_not_count_for_finished_seed(self):
        agent = object.__new__(TrainDistillResidualFlowAgent)
        agent.model = Mock()
        agent.num_eval_episodes = agent.eval_n_envs = 1
        agent.eval_seeds = [10000]
        agent.act_steps = 8
        agent.max_episode_steps = 16
        obs = {'state': np.zeros((1, 1, 23))}
        agent.reset_env_all = Mock(return_value=obs)
        agent._get_flow_action_for_eval = Mock(return_value=np.zeros((1, 8, 22)))
        agent.eval_venv = Mock()
        agent.eval_venv.step.side_effect = [
            (obs, np.array([0.]), np.array([True]), np.array([False]), {}),
            (obs, np.array([1.]), np.array([True]), np.array([False]), {}),
        ]
        agent._log_eval_to_csv = Mock()
        _, reward, success = agent.evaluate_pretrained_flow_policy()
        self.assertEqual(reward, 0.)
        self.assertEqual(success, 0.)


if __name__ == '__main__':
    unittest.main()
