"""
Environment wrapper for DexJoCo tasks with image observations.

DexJoCo demonstrations carry no object state, so policies are trained from
camera images plus proprioception rather than from the privileged low-dim
state.  This exposes a DexJoCo (pure MuJoCo + Gymnasium) task to the DICE-RL
stack, which is built around the old `gym` (0.22) 4-tuple `step` API and a
`{"rgb": ..., "state": ...}` observation dict.  Three representation gaps are
bridged here:

  1. Image: `DexjocoObsAdapter` hands back one uint8 frame per camera at the
     model's offscreen resolution.  The cameras named by `image_keys` are
     resized to the `shape_meta` resolution and concatenated channel-wise into
     (H, W, 3 * n_cameras), the layout RobomimicImageWrapper also produces.

  2. State: the adapter flattens every proprio key the task config lists,
     object poses included.  `low_dim_keys` picks the slice the policy sees and
     defaults to proprioception only (`tcp_pose` + `gripper_pose`), which is
     what the demonstrations actually contain.  Values are normalized to
     [-1, 1] with stats from `normalization.npz`.

  3. Action: the policy emits 22D per-arm measured-pose delta actions; the 16
     hand targets stay absolute. Delta translation is in world coordinates and
     delta rotation left-multiplies each arm's latest measured orientation.
     Single-arm actions decode to 23D targets. Bimanual 44D actions decode to
     [right_pose7, left_pose7, right_hand16, left_hand16] for DualArmPolicyWrapper.

`dexjoco` itself is imported lazily inside `__init__` so that importing this
module (and the wrapper registry) never requires DexJoCo to be installed.
"""

import numpy as np
import gym
import imageio
from gym import spaces
from PIL import Image

from util.dexjoco_actions import decode_actions, decode_bimanual_actions, unnormalize


# DexJoCo renames the third-person camera when domain randomization is on.
CAMERA_ALIASES = {"front": "random_camera", "ego": "random_camera"}

DEFAULT_IMAGE_KEYS = ("front", "wrist")
DEFAULT_LOW_DIM_KEYS = ("tcp_pose", "gripper_pose")


class DexjocoImageWrapper(gym.Env):
    def __init__(
        self,
        env=None,
        shape_meta=None,
        normalization_path=None,
        low_dim_keys=DEFAULT_LOW_DIM_KEYS,
        image_keys=DEFAULT_IMAGE_KEYS,
        clamp_obs=False,
        task_name="pick_bucket",
        policy_mode=True,
        randomize=False,
        render_mode="rgb_array",
        max_episode_steps=1000,
        success_steps_before_termination=1,
        realtime=False,
        **kwargs,
    ):
        # shape_meta fixes the camera resolution and the state width the policy
        # was trained on, so it is resolved before anything touches MuJoCo.
        if shape_meta is None:
            raise ValueError(
                "DexjocoImageWrapper needs shape_meta with obs.rgb and obs.state shapes"
            )
        image_height, image_width, image_channels = (
            int(value) for value in shape_meta["obs"]["rgb"]["shape"]
        )
        self.image_height = image_height
        self.image_width = image_width
        self.image_keys = [str(key) for key in image_keys]
        if image_channels != 3 * len(self.image_keys):
            raise ValueError(
                f"shape_meta declares {image_channels} channels but {len(self.image_keys)} "
                f"cameras were requested ({self.image_keys})"
            )
        self.state_dim = int(shape_meta["obs"]["state"]["shape"][0])
        self.low_dim_keys = (
            None if low_dim_keys is None else [str(key) for key in low_dim_keys]
        )

        # Lazily create the DexJoCo environment (keeps the import optional).
        if env is None:
            from dexjoco.tasks.mappings import CONFIG_MAPPING

            # Hammer Nail always emits camera observations and does not accept
            # image_obs. Other tasks need it explicitly enabled. Missing camera
            # frames are still rejected when observations are processed.
            image_kwargs = {} if task_name == "hammer_nail" else {"image_obs": True}
            try:
                self.env = CONFIG_MAPPING[task_name]().get_environment(
                    policy_mode=policy_mode,
                    render_mode=render_mode,
                    randomize=randomize,
                    **image_kwargs,
                )
            except TypeError as error:
                raise TypeError(
                    f"The installed DexJoCo does not support image observations for "
                    f"task {task_name!r}."
                ) from error
        else:
            self.env = env

        # DexJoCo sleeps to hold its nominal control rate, which only wastes
        # wall-clock in batched rollouts. An infinite rate makes that sleep a
        # no-op without touching the physics timestep.
        if not realtime:
            base = getattr(self.env, "unwrapped", self.env)
            if hasattr(base, "hz"):
                base.hz = float("inf")

        self.task_name = task_name
        self.clamp_obs = clamp_obs
        self._max_episode_steps = max_episode_steps
        # Terminate the episode this many consecutive success steps (reward>=1)
        # after the first success, mirroring RobomimicImageWrapper. This keeps
        # online RL tractable even if the DexJoCo env runs to its truncation
        # limit without self-terminating on success.
        self.success_steps_before_termination = success_steps_before_termination
        self.success_count = 0
        self.ever_succeeded = False
        self.video_writer = None
        self._seed = None
        self._raw_state = None
        self._raw_frames = []

        # Normalization stats (also fix the action dim from them).
        self.normalize = normalization_path is not None
        if normalization_path is not None:
            with np.load(normalization_path, allow_pickle=False) as normalization:
                self.obs_min = normalization["obs_min"]
                self.obs_max = normalization["obs_max"]
                self.action_min = normalization["action_min"]
                self.action_max = normalization["action_max"]
                self.normalize = (
                    bool(normalization["normalized"].item())
                    if "normalized" in normalization
                    else True
                )
            if int(self.obs_min.shape[0]) != self.state_dim:
                raise ValueError(
                    f"Dataset state is {int(self.obs_min.shape[0])}-dim but shape_meta "
                    f"declares {self.state_dim}; check low_dim_keys against the dataset"
                )
            action_dim = int(self.action_min.shape[0])
        else:
            action_dim = int(shape_meta.get("action", {"shape": [22]})["shape"][0])

        if action_dim not in (22, 44):
            raise ValueError(f"DexJoCo actions must have 22 or 44 dimensions, got {action_dim}")
        if "action" in shape_meta and tuple(shape_meta["action"]["shape"]) != (action_dim,):
            raise ValueError(
                f"shape_meta action shape {shape_meta['action']['shape']} does not "
                f"match the {action_dim}-dim dataset actions"
            )
        self.action_dim = action_dim

        # Which columns of the flattened state the policy actually sees.
        self._state_index = self._resolve_state_index()
        # Decode against raw proprioception, independently of the policy's
        # selected/normalized state and of object fields in the observation.
        self._action_state_index = self._resolve_action_state_index()

        # Action space: normalized rotvec action in [-1, 1].
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

        # Observation space: stacked camera frames plus the normalized state.
        # The frames stay uint8 so the shared-memory buffers the vector env
        # allocates from this space match what the workers write into them.
        self.observation_space = spaces.Dict()
        self.observation_space["rgb"] = spaces.Box(
            low=0,
            high=255,
            shape=(self.image_height, self.image_width, image_channels),
            dtype=np.uint8,
        )
        self.observation_space["state"] = spaces.Box(
            low=-1.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )

    # ------------------------------------------------------------------ #
    # Observation assembly.
    # ------------------------------------------------------------------ #
    def _proprio_indices(self, selected_keys):
        """Locate named fields in DexJoCo's flattened observation."""
        keys = getattr(self.env, "proprio_keys", None)
        space = getattr(self.env, "proprio_space", None)
        if keys is None or space is None:
            raise ValueError(
                "low_dim_keys needs DexJoCo's observation adapter to locate each "
                "key inside the flattened state; pass low_dim_keys=null to take "
                "the state whole"
            )
        spans, start = {}, 0
        for key in keys:
            size = int(np.prod(space[key].shape))
            spans[key] = (start, start + size)
            start += size
        unknown = [key for key in selected_keys if key not in spans]
        if unknown:
            raise ValueError(
                f"Unknown proprio keys {unknown}; task {self.task_name!r} exposes {list(spans)}"
            )
        return np.concatenate([np.arange(*spans[key]) for key in selected_keys])

    def _resolve_state_index(self):
        """Indices of the selected proprio keys inside the flattened state."""
        if self.low_dim_keys is None:
            return None
        index = self._proprio_indices(self.low_dim_keys)
        if index.shape[0] != self.state_dim:
            raise ValueError(
                f"low_dim_keys {self.low_dim_keys} give a {index.shape[0]}-dim state "
                f"but shape_meta declares {self.state_dim}"
            )
        return index

    def _resolve_action_state_index(self):
        space = getattr(self.env, "proprio_space", None)
        if space is None or getattr(self.env, "proprio_keys", None) is None:
            # With no adapter metadata, require the raw state to start with
            # the canonical single-arm/bimanual proprioception layout.
            return None
        index = self._proprio_indices(DEFAULT_LOW_DIM_KEYS)
        n_arms = self.action_dim // 22
        for key, width in zip(DEFAULT_LOW_DIM_KEYS, (7 * n_arms, 16 * n_arms)):
            if int(np.prod(space[key].shape)) != width:
                raise ValueError(
                    f"{self.action_dim}-dim actions require {key} with {width} "
                    f"values, got shape {space[key].shape}"
                )
        return index

    def _camera_frame(self, raw_obs, key):
        for name in (key, CAMERA_ALIASES.get(key)):
            if name is not None and name in raw_obs:
                return np.asarray(raw_obs[name], dtype=np.uint8)
        available = [name for name in raw_obs if name != "state"]
        raise KeyError(
            f"Camera {key!r} is missing from the DexJoCo observation; it renders {available}"
        )

    def _resize(self, frame):
        if frame.shape[:2] == (self.image_height, self.image_width):
            return frame
        resized = Image.fromarray(frame).resize(
            (self.image_width, self.image_height), Image.BILINEAR
        )
        return np.asarray(resized, dtype=np.uint8)

    def normalize_obs(self, obs):
        """Normalize observation to [-1, 1]."""
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        """Un-normalize action from [-1, 1] back to raw rotvec-action units."""
        return unnormalize(action, self.action_min, self.action_max)

    def _rotvec_action_to_env_action(self, action):
        """Decode the delta action while preserving absolute hand targets."""
        if self._raw_state is None:
            raise RuntimeError("Reset the environment before applying delta actions")
        state = (
            self._raw_state
            if self._action_state_index is None
            else self._raw_state[self._action_state_index]
        )
        decode = decode_bimanual_actions if self.action_dim == 44 else decode_actions
        return decode(state, action).astype(np.float32)

    def get_observation(self, raw_obs):
        """Stack the configured cameras and slice out the policy's state."""
        # Delta actions decode against the full measured state, so the raw
        # vector is kept even when the policy only sees part of it.
        self._raw_state = np.asarray(raw_obs["state"], dtype=np.float64).reshape(-1).copy()
        state = (
            self._raw_state
            if self._state_index is None
            else self._raw_state[self._state_index]
        ).astype(np.float32)
        if self.normalize:
            state = self.normalize_obs(state)

        self._raw_frames = [self._camera_frame(raw_obs, key) for key in self.image_keys]
        rgb = np.concatenate([self._resize(frame) for frame in self._raw_frames], axis=-1)
        return {"rgb": rgb.astype(np.uint8), "state": state.astype(np.float32)}

    # ------------------------------------------------------------------ #
    # Gym API.
    # ------------------------------------------------------------------ #
    def seed(self, seed=None):
        """Store the seed; DexJoCo (gymnasium) is seeded via reset(seed=...)."""
        if seed is not None:
            np.random.seed(seed=seed)
            self._seed = int(seed)
        else:
            np.random.seed()
            self._seed = None
        # Best-effort: some envs still expose a .seed().
        if hasattr(self.env, "seed"):
            try:
                self.env.seed(seed)
            except Exception:
                pass

    def reset(self, options={}, **kwargs):
        """Reset the environment and return an image+state observation."""
        self.success_count = 0
        self.ever_succeeded = False

        self.stop_video()
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        # Seed handling mirrors RobomimicImageWrapper:
        #   - explicit seed in options  -> deterministic (evaluation)
        #   - otherwise                 -> fresh random seed (training)
        new_seed = options.get("seed", None)
        if new_seed is None:
            new_seed = np.random.randint(0, 2**31 - 1)

        raw_obs, _info = self.env.reset(seed=int(new_seed))
        return self.get_observation(raw_obs)

    def step(self, action):
        """Step with normalized 22D single-arm or 44D bimanual actions."""
        action = np.asarray(action)
        if action.shape != (self.action_dim,):
            raise ValueError(f"Expected action shape ({self.action_dim},), got {action.shape}")
        if self.normalize:
            action = self.unnormalize_action(action)
        env_action = self._rotvec_action_to_env_action(action)

        raw_obs, reward, terminated, truncated, info = self.env.step(env_action)
        obs = self.get_observation(raw_obs)
        reward = float(reward)

        # The observation already carries this step's frames, so recording
        # costs no extra renders.
        if self.video_writer is not None:
            self.video_writer.append_data(self._tile(self._raw_frames))

        # Success-based termination (reward>=1.0), mirroring RobomimicImageWrapper.
        success_terminated = False
        if reward >= 1.0:
            self.success_count += 1
            self.ever_succeeded = True
            if self.success_count >= self.success_steps_before_termination:
                success_terminated = True
        elif not self.ever_succeeded:
            self.success_count = 0

        terminated = bool(terminated) or success_terminated
        truncated = bool(truncated)

        # Collapse the gymnasium 5-tuple to the old-gym 4-tuple expected by
        # MultiStep, while preserving the terminated/truncated distinction so
        # downstream bootstrapping stays correct (truncation != termination).
        done = terminated
        info = dict(info) if info is not None else {}
        info["TimeLimit.truncated"] = truncated and not terminated

        if done or truncated:
            self.stop_video()
        return obs, reward, done, info

    @staticmethod
    def _tile(frames):
        """Four views go into a 2x2 grid, anything else into a strip."""
        if len(frames) == 1:
            return frames[0]
        if len(frames) == 4:
            return np.concatenate(
                [
                    np.concatenate(frames[:2], axis=1),
                    np.concatenate(frames[2:], axis=1),
                ],
                axis=0,
            )
        return np.concatenate(frames, axis=1)

    def render(self, mode="rgb_array"):
        # The latest observation already holds freshly rendered frames.
        if self._raw_frames:
            return self._tile(self._raw_frames)
        try:
            frames = self.env.render()
        except TypeError:
            # Older gym-style render that still takes a mode argument.
            frames = self.env.render(mode=mode)
        if isinstance(frames, (list, tuple)):
            return self._tile(list(frames))
        return frames

    def stop_video(self):
        """Finalize an optional recording, including an evaluator-imposed cap."""
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

    def close(self):
        self.stop_video()
        self.env.close()

    @property
    def max_episode_steps(self):
        return self._max_episode_steps
