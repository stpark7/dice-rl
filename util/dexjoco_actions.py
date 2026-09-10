"""DexJoCo actions in world coordinates; all pose quaternions use wxyz.

Delta actions contain measured-to-target translation and rotation, followed by
16 absolute hand targets. Rotation increments left-multiply the measured pose.
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


def _mode(action_mode):
    if action_mode not in ("delta", "absolute"):
        raise ValueError(f"Unsupported action mode: {action_mode}")


def _observed(states, shape):
    states = _array(states, 23, "states", minimum=True)
    if states.shape[:-1] != shape:
        raise ValueError("States and actions must have the same leading dimensions")
    return states


def encode_actions(states, targets, action_mode="delta"):
    """Encode (..., 23) absolute quaternion targets as (..., 22) actions."""
    _mode(action_mode)
    targets = _array(targets, 23, "targets")
    shape = targets.shape[:-1]
    rotation = _rotation(targets)
    position = targets[..., :3]
    if action_mode == "delta":
        states = _observed(states, shape)
        position = position - states[..., :3]
        rotation = rotation * _rotation(states).inv()
    rotvec = rotation.as_rotvec().reshape(shape + (3,))
    return np.concatenate([position, rotvec, targets[..., 7:]], axis=-1)


def decode_actions(states, actions, action_mode="delta"):
    """Recover (..., 23) absolute targets using the latest measured state.

    states may be None in absolute mode, which requires no reference pose.
    """
    _mode(action_mode)
    actions = _array(actions, 22, "actions")
    shape = actions.shape[:-1]
    position = actions[..., :3]
    rotation = Rotation.from_rotvec(actions[..., 3:6].reshape(-1, 3))
    if action_mode == "delta":
        states = _observed(states, shape)
        position = states[..., :3] + position
        rotation = rotation * _rotation(states)
    quaternion = rotation.as_quat()[:, [3, 0, 1, 2]].reshape(shape + (4,))
    return np.concatenate([position, quaternion, actions[..., 6:]], axis=-1)


def normalize(values, minimum, maximum):
    return 2 * (np.asarray(values) - minimum) / (
        maximum - minimum + NORMALIZATION_EPSILON
    ) - 1


def unnormalize(values, minimum, maximum):
    return (np.asarray(values) + 1) / 2 * (
        maximum - minimum + NORMALIZATION_EPSILON
    ) + minimum
