"""
Environment wrapper for the DexJoCo `pinch_tongs` task with low-dimensional
(state) observations, mirroring `pusht_state.PushTStateWrapper`.

This exposes a DexJoCo (pure MuJoCo + Gymnasium) task to the DICE-RL stack,
which is built around the old `gym` (0.22) 4-tuple `step` API and a
`{"state": ...}` observation dict.  Two representation gaps are bridged here:

  1. Observation: DexJoCo's `DexjocoObsAdapter` flattens the privileged state to
     31-dim `[tcp_pose(7), allegro_qpos(16), tongs_ori_pose(7), table_delta(1)]`.
     We normalize it to [-1, 1] with stats from `normalization.npz`.

  2. Action: the policy/residual operates in the 22-dim `action_rotvec` space
     `[xyz(3), rotvec(3), allegro(16)]` (additive residuals are well-defined in
     rotvec space).  Just before stepping the env we un-normalize to raw units
     and convert rotvec -> quaternion to obtain the env's native 23-dim
     `[xyz(3), quat_wxyz(4), allegro(16)]` action.

NOTE: the rotvec -> quaternion convention (and the wxyz ordering) must match
DexJoCo's `SingleArmPolicyWrapper`; verify against the live env on the machine
that actually has DexJoCo installed.

`dexjoco` itself is imported lazily inside `__init__` so that importing this
module (and the wrapper registry) never requires DexJoCo to be installed.
"""

import numpy as np
import gym
import imageio
from gym import spaces
from scipy.spatial.transform import Rotation


class DexjocoLowdimWrapper(gym.Env):
    def __init__(
        self,
        env=None,
        normalization_path=None,
        clamp_obs=False,
        task_name="pinch_tongs",
        policy_mode=True,
        randomize=False,
        render_mode="rgb_array",
        max_episode_steps=1000,
        success_steps_before_termination=1,
        **kwargs,
    ):
        # Lazily create the DexJoCo environment (keeps the import optional).
        if env is None:
            from dexjoco.tasks.mappings import CONFIG_MAPPING

            self.env = CONFIG_MAPPING[task_name]().get_environment(
                policy_mode=policy_mode,
                render_mode=render_mode,
                randomize=randomize,
            )
        else:
            self.env = env

        self.task_name = task_name
        self.clamp_obs = clamp_obs
        self._max_episode_steps = max_episode_steps
        # Terminate the episode this many consecutive success steps (reward>=1)
        # after the first success, mirroring RobomimicLowdimWrapper. This keeps
        # online RL tractable even if the DexJoCo env runs to its 1000-step
        # truncation without self-terminating on success.
        self.success_steps_before_termination = success_steps_before_termination
        self.success_count = 0
        self.ever_succeeded = False
        self.video_writer = None
        self._seed = None

        # Normalization stats (also fix obs/action dims from them).
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]
            obs_dim = int(self.obs_min.shape[0])
            action_dim = int(self.action_min.shape[0])
        else:
            # Defaults for the 31-dim state / 22-dim rotvec action layout.
            obs_dim = 31
            action_dim = 22

        # Action space: normalized rotvec action in [-1, 1].
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

        # Observation space: single normalized "state" key.
        self.observation_space = spaces.Dict()
        self.observation_space["state"] = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    # ------------------------------------------------------------------ #
    # Normalization helpers (identical formulas to PushTStateWrapper).
    # ------------------------------------------------------------------ #
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
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def _rotvec_action_to_env_action(self, action):
        """Convert 22-dim [xyz, rotvec, allegro] -> 23-dim [xyz, quat_wxyz, allegro]."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        xyz = action[0:3]
        rotvec = action[3:6]
        allegro = action[6:]
        quat_xyzw = Rotation.from_rotvec(rotvec).as_quat()  # [x, y, z, w]
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        )
        return np.concatenate([xyz, quat_wxyz, allegro]).astype(np.float32)

    def get_observation(self, raw_obs):
        """Extract the flattened state vector and (optionally) normalize it."""
        state = np.asarray(raw_obs["state"], dtype=np.float32)
        if self.normalize:
            state = self.normalize_obs(state)
        return {"state": state.astype(np.float32)}

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
        """Reset the environment and return a normalized observation dict."""
        self.success_count = 0
        self.ever_succeeded = False

        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        # Seed handling mirrors PushTStateWrapper:
        #   - explicit seed in options  -> deterministic (evaluation)
        #   - otherwise                 -> fresh random seed (training)
        new_seed = options.get("seed", None)
        if new_seed is None:
            new_seed = np.random.randint(0, 2**31 - 1)

        raw_obs, _info = self.env.reset(seed=int(new_seed))
        return self.get_observation(raw_obs)

    def step(self, action):
        """Step the environment with a normalized 22-dim rotvec action."""
        if self.normalize:
            action = self.unnormalize_action(action)
        env_action = self._rotvec_action_to_env_action(action)

        raw_obs, reward, terminated, truncated, info = self.env.step(env_action)
        obs = self.get_observation(raw_obs)
        reward = float(reward)

        if self.video_writer is not None:
            self.video_writer.append_data(self.render(mode="rgb_array"))

        # Success-based termination (reward>=1.0), mirroring RobomimicLowdimWrapper.
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

        if done and self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        try:
            return self.env.render()
        except TypeError:
            # Older gym-style render that still takes a mode argument.
            return self.env.render(mode=mode)

    def close(self):
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        self.env.close()

    @property
    def max_episode_steps(self):
        return self._max_episode_steps
