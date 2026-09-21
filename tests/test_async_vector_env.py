import gc
import unittest

import gym
import numpy as np

from env.gym_utils.async_vector_env import AsyncVectorEnv


class CountingEnv(gym.Env):
    observation_space = gym.spaces.Box(-100, 100, shape=(1,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def reset(self, **kwargs):
        self.count = 0
        return np.array([self.count], dtype=np.float32)

    def step(self, action):
        self.count += 1
        return np.array([self.count], dtype=np.float32), 0., False, False, {}


class SpawnSharedMemoryTests(unittest.TestCase):
    def test_spawn_workers_reset_and_step_after_constructor_returns(self):
        env = AsyncVectorEnv([CountingEnv, CountingEnv], context='spawn')
        try:
            gc.collect()
            env.reset_async()
            np.testing.assert_array_equal(env.reset_wait(timeout=15), [[0.], [0.]])
            env.step_async(np.zeros((2, 1), dtype=np.float32))
            obs, _, terminated, truncated, _ = env.step_wait(timeout=15)
            np.testing.assert_array_equal(obs, [[1.], [1.]])
            self.assertFalse(terminated.any())
            self.assertFalse(truncated.any())
        finally:
            env.close(terminate=True)


if __name__ == '__main__':
    unittest.main()
