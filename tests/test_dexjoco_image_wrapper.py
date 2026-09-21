import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from gym import spaces
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    'dexjoco_image', ROOT / 'env/gym_utils/wrapper/dexjoco_image.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DexjocoImageWrapper = module.DexjocoImageWrapper


def shape_meta(state_dim=31, cameras=2, size=4, action_dim=22):
    return {
        'obs': {
            'rgb': {'shape': [size, size, 3 * cameras]},
            'state': {'shape': [state_dim]},
        },
        'action': {'shape': [action_dim]},
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


class BimanualEnvironment(MeasuredEnvironment):
    # Object state precedes the robot, so actions must locate raw proprioception
    # by key instead of assuming it starts at column zero.
    proprio_keys = ['object_pose', 'tcp_pose', 'gripper_pose']
    proprio_space = {
        'object_pose': spaces.Box(-1, 1, shape=(8,)),
        'tcp_pose': spaces.Box(-1, 1, shape=(14,)),
        'gripper_pose': spaces.Box(-1, 1, shape=(32,)),
    }
    front_key = 'ego'

    def observation(self):
        right_q = Rotation.from_euler('x', self.x / 10).as_quat()[[3, 0, 1, 2]]
        left_q = Rotation.from_euler('y', -self.x / 10).as_quat()[[3, 0, 1, 2]]
        return {
            'state': np.r_[np.full(8, 99.), self.x, 1., 2., right_q,
                           -2 * self.x, 3., 4., left_q, np.full(16, .2), np.full(16, .3)],
            self.front_key: self._frame(10),
            'wrist_left': self._frame(100),
            'wrist_right': self._frame(200),
        }


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


class BimanualTests(unittest.TestCase):
    def wrap_bimanual(self, env, **kwargs):
        return DexjocoImageWrapper(
            env=env,
            shape_meta=shape_meta(state_dim=46, cameras=3, action_dim=44),
            image_keys=['ego', 'wrist_left', 'wrist_right'],
            **kwargs,
        )

    def test_normalized_actions_use_each_arms_latest_pose_and_absolute_hands(self):
        from util.dexjoco_actions import normalize
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'normalization.npz'
            lo = np.full(44, -2.)
            hi = np.full(44, 4.)
            np.savez(path, obs_min=np.full(46, -30.), obs_max=np.full(46, 30.),
                     action_min=lo, action_max=hi, normalized=True)
            env = BimanualEnvironment()
            wrapper = self.wrap_bimanual(env, normalization_path=str(path))
            obs = wrapper.reset()
            np.testing.assert_allclose(
                obs['state'], normalize(env.observation()['state'][8:], -30., 30.), atol=1e-7
            )
            command = np.r_[3., 0., 0., 0., 0., .4, np.full(16, .7),
                            0., -2., 0., .3, 0., 0., np.full(16, -.8)]
            for x in (8., 9., 8.):
                if x == 8. and env.x != 8.:
                    wrapper.reset()
                wrapper.step(normalize(command, lo, hi))
                target = env.last_target
                self.assertEqual(target.shape, (46,))
                self.assertEqual(target.dtype, np.float32)
                np.testing.assert_allclose(target[:3], [x + 3., 1., 2.], atol=1e-6)
                np.testing.assert_allclose(target[7:10], [-2 * x, 1., 4.], atol=1e-6)
                expected_right = Rotation.from_rotvec([0., 0., .4]) * Rotation.from_euler('x', x / 10)
                expected_left = Rotation.from_rotvec([.3, 0., 0.]) * Rotation.from_euler('y', -x / 10)
                for start, expected in ((3, expected_right), (10, expected_left)):
                    q = target[start:start + 4]
                    actual = Rotation.from_quat(q[[1, 2, 3, 0]])
                    self.assertAlmostEqual((actual * expected.inv()).magnitude(), 0., places=6)
                np.testing.assert_allclose(target[14:30], .7, atol=1e-7)
                np.testing.assert_allclose(target[30:46], -.8, atol=1e-7)

    def test_proprioception_and_three_camera_order_without_normalization(self):
        for randomized in (False, True):
            with self.subTest(randomized=randomized):
                env = BimanualEnvironment()
                if randomized:
                    env.front_key = 'random_camera'
                wrapper = self.wrap_bimanual(env)
                obs = wrapper.reset()
                self.assertEqual(wrapper.action_space.shape, (44,))
                np.testing.assert_allclose(obs['state'], env.observation()['state'][8:])
                self.assertEqual(obs['rgb'].shape, (4, 4, 9))
                self.assertEqual(obs['rgb'].dtype, np.uint8)
                for i, value in enumerate((10, 100, 200)):
                    np.testing.assert_array_equal(obs['rgb'][..., 3 * i:3 * (i + 1)], value)

    def test_action_width_must_match_environment_proprioception(self):
        with self.assertRaisesRegex(ValueError, '44-dim actions require tcp_pose'):
            DexjocoImageWrapper(
                env=MeasuredEnvironment(), shape_meta=shape_meta(state_dim=23, action_dim=44)
            )
        with self.assertRaisesRegex(ValueError, '22-dim actions require tcp_pose'):
            DexjocoImageWrapper(
                env=BimanualEnvironment(), shape_meta=shape_meta(state_dim=46)
            )

    def test_normalization_action_width_must_match_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'normalization.npz'
            np.savez(path, obs_min=np.zeros(46), obs_max=np.ones(46),
                     action_min=np.zeros(22), action_max=np.ones(22))
            with self.assertRaisesRegex(ValueError, 'shape_meta action shape'):
                self.wrap_bimanual(BimanualEnvironment(), normalization_path=str(path))

    def test_invalid_action_shape_is_rejected_before_broadcasting(self):
        wrapper = self.wrap_bimanual(BimanualEnvironment())
        wrapper.reset()
        for action in (0., np.zeros(1), np.zeros(22), np.zeros((1, 44))):
            with self.assertRaisesRegex(ValueError, 'Expected action shape'):
                wrapper.step(action)


if __name__ == '__main__':
    unittest.main()
