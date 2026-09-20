import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from gym import spaces


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    'dexjoco_image', ROOT / 'env/gym_utils/wrapper/dexjoco_image.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DexjocoImageWrapper = module.DexjocoImageWrapper


def shape_meta(state_dim=31, cameras=2, size=4):
    return {
        'obs': {
            'rgb': {'shape': [size, size, 3 * cameras]},
            'state': {'shape': [state_dim]},
        },
        'action': {'shape': [22]},
    }


class MeasuredEnvironment:
    """Commands and measured poses intentionally differ (controller lag)."""

    proprio_keys = ['tcp_pose', 'gripper_pose', 'object_pose']
    proprio_space = {
        'tcp_pose': spaces.Box(-1, 1, shape=(7,)),
        'gripper_pose': spaces.Box(-1, 1, shape=(16,)),
        'object_pose': spaces.Box(-1, 1, shape=(8,)),
    }
    front_key = 'front'
    image_size = 8

    @property
    def unwrapped(self):
        return self

    def _frame(self, fill):
        return np.full((self.image_size, self.image_size, 3), fill, dtype=np.uint8)

    def observation(self):
        return {
            'state': np.r_[self.x, 0, 0, 1, 0, 0, 0, np.zeros(24)],
            self.front_key: self._frame(10),
            'wrist': self._frame(200),
        }

    def reset(self, seed=None):
        self.x = 8.
        return self.observation(), {}

    def step(self, action):
        self.last_target = action.copy()
        self.x += 1
        return self.observation(), 0., False, False, {}


class RandomizedEnvironment(MeasuredEnvironment):
    """With randomization on, DexJoCo renames the third-person camera."""

    front_key = 'random_camera'


def wrap(env=None, state_dim=31, cameras=2, **kwargs):
    kwargs.setdefault('low_dim_keys', None)
    return DexjocoImageWrapper(
        env=env if env is not None else MeasuredEnvironment(),
        shape_meta=shape_meta(state_dim=state_dim, cameras=cameras),
        **kwargs,
    )


class ActionTests(unittest.TestCase):
    def test_each_step_and_reset_use_current_measurement(self):
        env = MeasuredEnvironment()
        wrapper = wrap(env=env)
        wrapper.reset()
        command = np.r_[3., 0, 0, 0, 0, 0, np.full(16, .7)]
        wrapper.step(command)
        self.assertEqual(env.last_target[0], 11)
        np.testing.assert_allclose(env.last_target[7:], .7)
        wrapper.step(command)
        self.assertEqual(env.last_target[0], 12)
        wrapper.reset()
        wrapper.step(command)
        self.assertEqual(env.last_target[0], 11)

    def test_delta_requires_reset(self):
        wrapper = wrap()
        with self.assertRaises(RuntimeError):
            wrapper.step(np.zeros(22))

    def test_unnormalized_archive_is_passed_through(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'normalization.npz'
            np.savez(path, obs_min=np.zeros(31), obs_max=np.ones(31),
                     action_min=np.zeros(22), action_max=np.ones(22),
                     normalized=False)
            env = MeasuredEnvironment()
            wrapper = wrap(env=env, normalization_path=str(path))
            self.assertEqual(wrapper.reset()['state'][0], 8.)
            wrapper.step(np.r_[3., np.zeros(21)])
            self.assertEqual(env.last_target[0], 11.)

    def test_normalized_delta_is_inverted_before_target_reconstruction(self):
        from util.dexjoco_actions import normalize
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'normalization.npz'
            lo, hi = np.zeros(22), np.full(22, 4.)
            np.savez(path, obs_min=np.zeros(31), obs_max=np.full(31, 10.),
                     action_min=lo, action_max=hi, normalized=True)
            env = MeasuredEnvironment()
            wrapper = wrap(env=env, normalization_path=str(path))
            wrapper.reset()
            command = np.r_[3., 0, 0, 0, 0, 0, np.full(16, .7)]
            wrapper.step(normalize(command, lo, hi))
            self.assertAlmostEqual(env.last_target[0], 11.)
            np.testing.assert_allclose(env.last_target[7:], .7, atol=1e-7)

    def test_dataset_state_width_must_match_shape_meta(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'normalization.npz'
            np.savez(path, obs_min=np.zeros(23), obs_max=np.ones(23),
                     action_min=np.zeros(22), action_max=np.ones(22))
            with self.assertRaisesRegex(ValueError, 'shape_meta'):
                wrap(normalization_path=str(path))


class ObservationTests(unittest.TestCase):
    def test_low_dim_keys_select_the_policy_state(self):
        env = MeasuredEnvironment()
        wrapper = DexjocoImageWrapper(
            env=env,
            shape_meta=shape_meta(state_dim=23),
            low_dim_keys=['tcp_pose', 'gripper_pose'],
        )
        obs = wrapper.reset()
        self.assertEqual(obs['state'].shape, (23,))
        np.testing.assert_allclose(obs['state'], env.observation()['state'][:23])
        self.assertEqual(wrapper.observation_space['state'].shape, (23,))

    def test_low_dim_keys_must_match_shape_meta(self):
        with self.assertRaisesRegex(ValueError, 'shape_meta declares'):
            DexjocoImageWrapper(
                env=MeasuredEnvironment(),
                shape_meta=shape_meta(state_dim=31),
                low_dim_keys=['tcp_pose', 'gripper_pose'],
            )

    def test_unknown_low_dim_key_is_reported(self):
        with self.assertRaisesRegex(ValueError, 'Unknown proprio keys'):
            DexjocoImageWrapper(
                env=MeasuredEnvironment(),
                shape_meta=shape_meta(state_dim=7),
                low_dim_keys=['nose_pose'],
            )

    def test_cameras_are_resized_and_stacked_in_order(self):
        wrapper = wrap()
        obs = wrapper.reset()
        self.assertEqual(obs['rgb'].shape, (4, 4, 6))
        self.assertEqual(obs['rgb'].dtype, np.uint8)
        np.testing.assert_array_equal(obs['rgb'][..., :3], 10)
        np.testing.assert_array_equal(obs['rgb'][..., 3:], 200)
        self.assertEqual(wrapper.observation_space['rgb'].dtype, np.uint8)

    def test_randomized_front_camera_is_resolved(self):
        wrapper = wrap(env=RandomizedEnvironment())
        np.testing.assert_array_equal(wrapper.reset()['rgb'][..., :3], 10)

    def test_missing_camera_names_the_available_ones(self):
        wrapper = wrap(image_keys=['front', 'ego_left'])
        with self.assertRaisesRegex(KeyError, 'ego_left'):
            wrapper.reset()

    def test_channel_count_must_match_the_camera_count(self):
        with self.assertRaisesRegex(ValueError, 'channels'):
            DexjocoImageWrapper(
                env=MeasuredEnvironment(),
                shape_meta=shape_meta(cameras=4),
                low_dim_keys=None,
                image_keys=['front', 'wrist'],
            )

    def test_shape_meta_is_required(self):
        with self.assertRaisesRegex(ValueError, 'shape_meta'):
            DexjocoImageWrapper(env=MeasuredEnvironment())


if __name__ == '__main__':
    unittest.main()
