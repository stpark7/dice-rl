import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from util import dexjoco_actions as actions


def wxyz(rotation):
    return rotation.as_quat()[..., [3, 0, 1, 2]]


class ActionContractTests(unittest.TestCase):
    def test_current_measured_pose_and_absolute_hand_targets(self):
        state = np.r_[8., 0., 0., 1., 0., 0., 0., np.full(16, .2)]
        target = np.r_[11., 0., 0., 1., 0., 0., 0., np.full(16, .7)]
        encoded = actions.encode_actions(state, target)
        np.testing.assert_allclose(encoded[:6], [3, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(encoded[6:], target[7:])

    def test_relative_rotation_crosses_pi_without_absolute_rotvec_jump(self):
        state = np.r_[np.zeros(3), wxyz(Rotation.from_euler('z', 179, degrees=True)), np.zeros(16)]
        target = np.r_[np.zeros(3), wxyz(Rotation.from_euler('z', -179, degrees=True)), np.ones(16)]
        delta = actions.encode_actions(state, target)
        np.testing.assert_allclose(delta[3:6], [0, 0, np.deg2rad(2)], atol=1e-12)

    def test_noncommuting_rotation_roundtrip_and_world_frame_composition(self):
        observed = Rotation.from_euler('xyz', [[.7, -.8, .4], [1.1, .3, -.5]])
        increment = Rotation.from_rotvec([[.3, .2, -.1], [-.2, .4, .1]])
        targets = increment * observed
        states = np.c_[np.array([[1, 2, 3], [3, 4, 5]]), wxyz(observed), np.zeros((2, 31))]
        raw = np.c_[states[:, :3] + .05, wxyz(targets), np.full((2, 16), .8)]
        encoded = actions.encode_actions(states, raw)
        np.testing.assert_allclose(encoded[:, 3:6], increment.as_rotvec(), atol=1e-12)
        restored = actions.decode_actions(states, encoded)
        np.testing.assert_allclose(restored[:, :3], raw[:, :3], atol=1e-12)
        np.testing.assert_array_equal(restored[:, 7:], raw[:, 7:])
        recovered_rotation = Rotation.from_quat(restored[:, [4, 5, 6, 3]])
        np.testing.assert_allclose((recovered_rotation * targets.inv()).magnitude(), 0, atol=1e-12)

    def test_normalization_inverse_including_constant_dimensions(self):
        data = np.array([[1., 2., .000001], [1., 5., .000002]])
        lo, hi = data.min(axis=0), data.max(axis=0)
        normalized = actions.normalize(data, lo, hi)
        np.testing.assert_allclose(actions.unnormalize(normalized, lo, hi), data, atol=1e-15)
        np.testing.assert_array_equal(normalized[:, 0], [-1, -1])

    def test_rejects_invalid_data_and_shapes(self):
        state = np.r_[np.zeros(3), 1., 0., 0., 0., np.zeros(16)]
        for invalid in [np.zeros(23), np.full(23, np.nan), np.zeros(22)]:
            with self.subTest(target=invalid):
                with self.assertRaises(ValueError):
                    actions.encode_actions(state, invalid)
        with self.assertRaises(ValueError):
            actions.encode_actions(np.zeros(23), state)
        with self.assertRaises(ValueError):
            actions.decode_actions(state, np.zeros(21))


if __name__ == '__main__':
    unittest.main()
