"""DexJoCo actions in world coordinates; all pose quaternions use wxyz.

Actions carry the measured-to-target translation and rotation, followed by 16
absolute hand targets. Rotation increments left-multiply the measured pose, so
both encoding and decoding need the state the command was issued against.
"""

import numpy as np
from scipy.spatial.transform import Rotation


NORMALIZATION_EPSILON = 1e-6


def _array(value, dimension, name, minimum=False):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim < 1 or (
        value.shape[-1] < dimension if minimum else value.shape[-1] != dimension
    ):
        raise ValueError(f"{name} must have {'at least ' if minimum else ''}{dimension} columns")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")
    return value


def _rotation(pose):
    quaternion = pose[..., 3:7].reshape(-1, 4)
    if np.any(np.linalg.norm(quaternion, axis=-1) < 1e-8):
        raise ValueError("Pose quaternion has zero norm")
    return Rotation.from_quat(quaternion[:, [1, 2, 3, 0]])


def _observed(states, shape):
    states = _array(states, 23, "states", minimum=True)
    if states.shape[:-1] != shape:
        raise ValueError("States and actions must have the same leading dimensions")
    return states


def encode_actions(states, targets):
    """Encode (..., 23) absolute quaternion targets as (..., 22) delta actions."""
    targets = _array(targets, 23, "targets")
    shape = targets.shape[:-1]
    states = _observed(states, shape)
    position = targets[..., :3] - states[..., :3]
    rotation = _rotation(targets) * _rotation(states).inv()
    rotvec = rotation.as_rotvec().reshape(shape + (3,))
    return np.concatenate([position, rotvec, targets[..., 7:]], axis=-1)


def decode_actions(states, actions):
    """Recover (..., 23) absolute targets using the latest measured state."""
    actions = _array(actions, 22, "actions")
    shape = actions.shape[:-1]
    states = _observed(states, shape)
    position = states[..., :3] + actions[..., :3]
    # SciPy also needs a writable buffer when callers supply broadcast views.
    rotation = Rotation.from_rotvec(actions[..., 3:6].reshape(-1, 3).copy()) * _rotation(states)
    quaternion = rotation.as_quat()[:, [3, 0, 1, 2]].reshape(shape + (4,))
    return np.concatenate([position, quaternion, actions[..., 6:]], axis=-1)


def decode_bimanual_actions(states, actions):
    """Decode (..., 44) actions into DexJoCo's (..., 46) policy targets.

    States and returned targets are [right_pose7, left_pose7, right_hand16,
    left_hand16]. Actions instead contain [right_action22, left_action22].
    Each arm's world-frame delta uses its own latest measured pose.
    """
    states = _array(states, 46, "states", minimum=True)
    actions = _array(actions, 44, "actions")
    right_state = np.concatenate([states[..., :7], states[..., 14:30]], axis=-1)
    left_state = np.concatenate([states[..., 7:14], states[..., 30:46]], axis=-1)
    right = decode_actions(right_state, actions[..., :22])
    left = decode_actions(left_state, actions[..., 22:])
    return np.concatenate(
        [right[..., :7], left[..., :7], right[..., 7:], left[..., 7:]], axis=-1
    )


def normalize(values, minimum, maximum):
    return 2 * (np.asarray(values) - minimum) / (
        maximum - minimum + NORMALIZATION_EPSILON
    ) - 1


def unnormalize(values, minimum, maximum):
    return (np.asarray(values) + 1) / 2 * (
        maximum - minimum + NORMALIZATION_EPSILON
    ) + minimum
