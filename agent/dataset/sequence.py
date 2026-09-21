"""
Pre-training data loader. Modified from https://github.com/jannerm/diffuser/blob/main/diffuser/datasets/sequence.py

No normalization is applied here --- we always normalize the data when pre-processing it with a different script, and the normalization info is also used in RL fine-tuning.

"""

from collections import namedtuple
from pathlib import Path
import numpy as np
import torch
import logging
import pickle
import random
from tqdm import tqdm

log = logging.getLogger(__name__)

Batch = namedtuple("Batch", "actions conditions")
Transition = namedtuple("Transition", "actions conditions rewards dones mc_return")
TransitionWithReturn = namedtuple(
    "Transition", "actions conditions rewards dones reward_to_gos"
)


class StitchedSequenceDataset(torch.utils.data.Dataset):
    """
    Load stitched trajectories of states/actions/images, and 1-D array of traj_lengths, from npz or pkl file.

    Use the first max_n_episodes episodes (instead of random sampling)

    Example:
        states: [----------traj 1----------][---------traj 2----------] ... [---------traj N----------]
        Episode IDs (determined based on traj_lengths):  [----------   1  ----------][----------   2  ---------] ... [----------   N  ---------]

    Each sample is a namedtuple of (1) chunked actions and (2) a list (obs timesteps) of dictionary with keys states and images.

    """

    def __init__(
        self,
        dataset_path,
        horizon_steps=64,
        cond_steps=1,
        img_cond_steps=1,
        max_n_episodes=10000,
        use_img=False,
        device="cuda:0",
        use_6d_rot=False,  # New parameter for 6D rotation
        abs_action=False,  # Whether to use absolute action mode
        image_keys=None,  # Ordered camera names for named DexJoCo image archives
    ):
        assert (
            img_cond_steps <= cond_steps
        ), "consider using more cond_steps than img_cond_steps"
        
        # Enforce that 6D rotation requires absolute actions
        assert not (use_6d_rot and not abs_action), (
            "6D rotation representation requires absolute actions. "
            "Please set abs_action=True when use_6d_rot=True, "
            "and ensure you're using an absolute action dataset."
        )
        self.horizon_steps = horizon_steps
        self.cond_steps = cond_steps  # states (proprio, etc.)
        self.img_cond_steps = img_cond_steps
        self.device = device
        self.use_img = use_img
        self.max_n_episodes = max_n_episodes
        self.dataset_path = dataset_path
        self.use_6d_rot = use_6d_rot
        self.abs_action = abs_action

        # Load dataset to device specified
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)  # only np arrays
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]  # 1-D array
        print(f"Loaded {len(traj_lengths)} trajectories from {dataset_path}")
        total_num_steps = np.sum(traj_lengths)

        # Set up indices for sampling
        self.indices = self.make_indices(traj_lengths, horizon_steps)

        # Extract states and actions up to max_n_episodes
        states_data = dataset["states"][:total_num_steps]
        
        self.states = torch.from_numpy(states_data).float().to(device)
        # (total_num_steps, obs_dim)
        
        # Load actions - they should already be in the correct format from preprocessing
        actions = dataset["actions"][:total_num_steps]
        
        # Validate action dimensions match expectations
        if use_6d_rot:
            if actions.shape[-1] != 10:
                raise ValueError(
                    f"Expected 10D actions when use_6d_rot=True, but got {actions.shape[-1]}D. "
                    f"Please ensure the dataset was preprocessed with --use_6d_rot flag."
                )
            log.info(f"Loaded 10D actions with 6D rotation representation")
        else:
            if abs_action and actions.shape[-1] != 7:
                log.warning(f"Expected 7D actions for absolute actions, got {actions.shape[-1]}D")
                
        self.actions = torch.from_numpy(actions).float().to(device)
        print(f"States shape/type: {self.states.shape, self.states.dtype}")
        print(f"Actions shape/type: {self.actions.shape, self.actions.dtype}")
        log.info(f"Loaded dataset from {dataset_path}")
        log.info(f"Number of episodes: {min(max_n_episodes, len(traj_lengths))}")
        log.info(f"States shape/type: {self.states.shape, self.states.dtype}")
        log.info(f"Actions shape/type: {self.actions.shape, self.actions.dtype}")
        if self.use_img:
            if "image_store" in dataset:
                from agent.dataset.camera_images import CameraImages

                self.images = CameraImages(
                    Path(dataset_path).parent / str(dataset["image_store"].item()),
                    image_keys, total_num_steps, device,
                )
            else:
                if image_keys is not None:
                    raise ValueError("image_keys requires a named camera archive")
                self.images = torch.from_numpy(dataset["images"][:total_num_steps]).to(device)
            log.info(f"Images shape/type: {self.images.shape, self.images.dtype}")
        if isinstance(dataset, np.lib.npyio.NpzFile):
            dataset.close()

    def __getitem__(self, idx):
        """
        repeat states/images if using history observation at the beginning of the episode
        """
        start, num_before_start = self.indices[idx]
        end = start + self.horizon_steps
        states = self.states[(start - num_before_start) : (start + 1)]
        actions = self.actions[start:end]
        states = torch.stack(
            [
                states[max(num_before_start - t, 0)]
                for t in reversed(range(self.cond_steps))
            ]
        )  # more recent is at the end
        conditions = {"state": states}
        if self.use_img:
            image_start = max(start - num_before_start, start - self.img_cond_steps + 1)
            images = self.images[image_start : start + 1]
            images = torch.stack(
                [
                    images[max(start - image_start - t, 0)]
                    for t in reversed(range(self.img_cond_steps))
                ]
            )
            conditions["rgb"] = images
        batch = Batch(actions, conditions)
        return batch

    def make_indices(self, traj_lengths, horizon_steps):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint, also save the number of steps before it within the same trajectory
        """
        indices = []
        cur_traj_index = 0
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps
            indices += [
                (i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)
            ]
            cur_traj_index += traj_length
        return indices

    def set_train_val_split(self, train_split):
        """
        Not doing validation right now
        """
        num_train = int(len(self.indices) * train_split)
        train_indices = random.sample(self.indices, num_train)
        val_indices = [i for i in range(len(self.indices)) if i not in train_indices]
        self.indices = train_indices
        return val_indices

    def __len__(self):
        return len(self.indices)


class StitchedSequenceQLearningDataset(StitchedSequenceDataset):
    """
    Extends StitchedSequenceDataset to include rewards and dones for Q learning
    
    Pads offline expert trajectories to match online environment format:
    - Online episodes end with 15 consecutive success steps before termination
    - Offline episodes are padded with 15 success steps to maintain consistency
    - Supports n-step returns for proper sampling alignment
    """

    def __init__(
        self,
        dataset_path,
        max_n_episodes=10000,
        discount_factor=1.0,
        device="cuda:0",
        use_6d_rot=False,  # Add 6D rotation support
        get_mc_return=True,
        gamma=0.99,
        success_steps_for_termination=15,  # Must match online env setting
        use_n_step=False,  # Whether to support n-step returns  
        n_step=1,  # Number of steps for n-step returns
        **kwargs,
    ):
        if dataset_path.endswith(".npz"):
            dataset = np.load(dataset_path, allow_pickle=False)
        elif dataset_path.endswith(".pkl"):
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
        
        # Store parameters
        self.gamma = gamma
        self.success_steps_for_termination = success_steps_for_termination
        self.use_n_step = use_n_step
        self.n_step = n_step
        # Extract horizon_steps from kwargs for MC return computation
        self.horizon_steps = kwargs.get('horizon_steps', 8)
        traj_lengths = dataset["traj_lengths"][:max_n_episodes]
        self.traj_lengths = traj_lengths
        self.episode_ends = np.cumsum(traj_lengths)
        total_num_steps = np.sum(traj_lengths)

        # discount factor
        self.discount_factor = discount_factor

        # rewards and dones(terminals)
        self.rewards = (
            torch.from_numpy(dataset["rewards"][:total_num_steps]).float().to(device)
        )
        log.info(f"Rewards shape/type: {self.rewards.shape, self.rewards.dtype}")
        self.dones = (
            torch.from_numpy(dataset["terminals"][:total_num_steps]).to(device).float()
        )
        log.info(f"Dones shape/type: {self.dones.shape, self.dones.dtype}")
        
        # Fix done signals: set last step of each trajectory to done=True
        cumulative_traj_length = np.cumsum(traj_lengths)
        for traj_end_idx in cumulative_traj_length:
            self.dones[traj_end_idx - 1] = 1.0  # Last step of each trajectory
        log.info(f"Fixed done signals for {len(traj_lengths)} trajectories")

        super().__init__(
            dataset_path=dataset_path,
            max_n_episodes=max_n_episodes,
            device=device,
            use_6d_rot=use_6d_rot,  # Pass 6D rotation flag
            **kwargs,
        )
        log.info(f"Total number of transitions using: {len(self)}")

        # compute discounted reward-to-go for each trajectory
        self.get_mc_return = get_mc_return
        if get_mc_return:
            self.reward_to_go = torch.zeros_like(self.rewards)
            cumulative_traj_length = np.cumsum(traj_lengths)
            prev_traj_length = 0
            
            # Chunked MC return computation
            for i, traj_length in tqdm(
                enumerate(cumulative_traj_length), desc="Computing chunked reward-to-go"
            ):
                traj_rewards = self.rewards[prev_traj_length:traj_length]
                returns = torch.zeros_like(traj_rewards)
                traj_len = len(traj_rewards)
                
                # Process trajectory in chunks of horizon_steps
                for start_idx in range(traj_len):
                    # Compute chunked MC return for position start_idx
                    mc_return = 0.0
                    chunk_idx = 0
                    
                    # Process complete chunks
                    t = start_idx
                    while t < traj_len:
                        # Sum rewards within this chunk (simple sum, no within-chunk discounting)
                        chunk_end = min(t + self.horizon_steps, traj_len)
                        chunk_reward = traj_rewards[t:chunk_end].sum().item()
                        
                        # Add this chunk's contribution with gamma^chunk_idx discounting
                        mc_return += (self.discount_factor ** chunk_idx) * chunk_reward
                        
                        # Move to next chunk
                        t = chunk_end
                        chunk_idx += 1
                    
                    returns[start_idx] = mc_return
                
                self.reward_to_go[prev_traj_length:traj_length] = returns
                prev_traj_length = traj_length
            log.info(f"Computed chunked reward-to-go for each trajectory (horizon_steps={self.horizon_steps}).")

    def make_indices(self, traj_lengths, horizon_steps):
        """
        skip last step of truncated episodes
        """
        num_skip = 0
        indices = []
        cur_traj_index = 0
        self.processed_traj_lengths = []  # Track trajectory lengths after processing
        for traj_length in traj_lengths:
            max_start = cur_traj_index + traj_length - horizon_steps
            if not self.dones[cur_traj_index + traj_length - 1]:  # truncation
                max_start -= 1
                num_skip += 1
            prev_len = len(indices)
            indices += [
                (i, i - cur_traj_index) for i in range(cur_traj_index, max_start + 1)
            ]
            self.processed_traj_lengths.append(len(indices) - prev_len)
            cur_traj_index += traj_length
        log.info(f"Number of transitions skipped due to truncation: {num_skip}")
        return indices

    def __getitem__(self, idx):
        start, num_before_start = self.indices[idx]
        episode_start = start - num_before_start
        episode_index = np.searchsorted(self.episode_ends, start, side="right")
        episode_end = int(self.episode_ends[episode_index])
        actions = self.actions[start : start + self.horizon_steps]

        # Discount once per action chunk, matching online replay. Stop at the
        # episode boundary even when the final n-step chunk is shorter than H.
        num_chunks = self.n_step if self.use_n_step else 1
        rewards = torch.zeros(1, device=self.device)
        dones = torch.zeros(1, device=self.device)
        next_start = start
        for step in range(num_chunks):
            chunk_start = start + step * self.horizon_steps
            chunk_end = min(chunk_start + self.horizon_steps, episode_end)
            terminal = torch.nonzero(self.dones[chunk_start:chunk_end]).flatten()
            if len(terminal):
                chunk_end = chunk_start + int(terminal[0]) + 1
                dones.fill_(1)
            rewards += self.gamma ** step * self.rewards[chunk_start:chunk_end].sum()
            next_start = chunk_end
            if dones.item() or chunk_end == episode_end:
                break

        def history(values, current, count):
            first = max(episode_start, current - count + 1)
            frames = values[first : current + 1]
            return torch.stack([
                frames[max(current - first - t, 0)]
                for t in reversed(range(count))
            ])

        states = history(self.states, start, self.cond_steps)
        # Terminal observations are not in this archive; zero placeholders are
        # safe because done masks bootstrapping. Never read the next episode.
        next_states = (torch.zeros_like(states) if dones.item() else
                       history(self.states, next_start, self.cond_steps))
        conditions = {"state": states, "next_state": next_states}
        if self.use_img:
            images = history(self.images, start, self.img_cond_steps)
            next_images = (torch.zeros_like(images) if dones.item() else
                           history(self.images, next_start, self.img_cond_steps))
            conditions["rgb"] = images
            conditions["next_rgb"] = next_images
        if self.get_mc_return:
            mc_return = self.reward_to_go[start : (start + 1)]
            batch = Transition(
                actions,
                conditions,
                rewards,
                dones,
                mc_return,
            )
        else:
            batch = Transition(
                actions,
                conditions,
                rewards,
                dones,
                rewards,
            )
        return batch
