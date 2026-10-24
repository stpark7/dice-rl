"""Regression tests for raw chord windows, episode splits and training moments."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from agent.dataset.hand_chunks import ActionChunkDataset, load_data
from util.hand_chord_training import evaluate, fit_statistics, kl_weight


class HandChordDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "demonstrations.npz"
        self.lengths = np.array([5, 4, 6, 3], dtype=np.int64)
        frames = np.arange(self.lengths.sum(), dtype=np.float32)[:, None]
        self.actions = np.broadcast_to(frames, (len(frames), 22)).copy()
        self.states = np.broadcast_to(frames + 100, (len(frames), 23)).copy()

    def write(self, **changes):
        arrays = dict(actions=self.actions, states=self.states, traj_lengths=self.lengths,
                      images=np.array([object()], dtype=object))
        arrays.update(changes)
        np.savez(self.path, **arrays)

    def load(self, **changes):
        settings = dict(dataset_path=self.path, horizon_steps=3, max_n_episodes=None,
                        val_fraction=0.25, seed=7)
        settings.update(changes)
        return load_data(**settings)

    def test_full_actions_and_measured_chunk_start_state_never_cross_episodes(self):
        self.write()
        train, validation = self.load()
        self.assertFalse(set(train.episode_ids) & set(validation.episode_ids))
        self.assertEqual(set(train.episode_ids) | set(validation.episode_ids), set(range(4)))
        beginnings = np.r_[0, np.cumsum(self.lengths)[:-1]]
        for dataset in (train, validation):
            expected_starts = [start for episode in dataset.episode_ids
                               for start in range(beginnings[episode],
                                                  beginnings[episode] + self.lengths[episode] - 2)]
            np.testing.assert_array_equal(dataset.starts, expected_starts)
            for index, start in enumerate(dataset.starts):
                chunk, state = dataset[index]
                self.assertEqual(tuple(chunk.shape), (3, 22))
                self.assertEqual(tuple(state.shape), (23,))
                np.testing.assert_array_equal(chunk.numpy(), self.actions[start:start + 3])
                np.testing.assert_array_equal(state.numpy(), self.states[start])

    def test_episode_limit_is_applied_before_deterministic_split(self):
        self.write()
        first = self.load(max_n_episodes=3)
        second = self.load(max_n_episodes=3)
        for lhs, rhs in zip(first, second):
            np.testing.assert_array_equal(lhs.episode_ids, rhs.episode_ids)
            np.testing.assert_array_equal(lhs.starts, rhs.starts)
            np.testing.assert_array_equal(lhs.traj_lengths, [5, 4, 6])
            self.assertEqual(len(lhs.actions), 15)
            self.assertEqual(len(lhs.states), 15)

    def test_two_episode_dataset_keeps_both_partitions_nonempty(self):
        self.write()
        train, validation = self.load(max_n_episodes=2, val_fraction=0.1)
        self.assertEqual(len(train.episode_ids), 1)
        self.assertEqual(len(validation.episode_ids), 1)

    def test_rejects_nonpositive_or_fractional_trajectory_lengths(self):
        for lengths in ([5, 0, 6, 7], [5, -1, 6, 8], [5.5, 3.5, 6, 3],
                        [5, float("nan"), 6, 3], [[5, 4, 6, 3]]):
            with self.subTest(lengths=lengths):
                self.write(traj_lengths=np.array(lengths))
                with self.assertRaisesRegex(ValueError, "length"):
                    self.load()

    def test_rejects_mismatched_frame_counts_and_coordinate_shapes(self):
        for changes in (dict(actions=self.actions[:-1]), dict(states=self.states[:-1]),
                        dict(actions=self.actions[:, :16]), dict(states=self.states[:, :16])):
            with self.subTest(changes=tuple(changes)):
                self.write(**changes)
                with self.assertRaises(ValueError):
                    self.load()

    def test_rejects_nonfinite_actions_or_states(self):
        for field in ("actions", "states"):
            invalid = getattr(self, field).copy()
            invalid[0, 0] = np.nan
            self.write(**{field: invalid})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "finite"):
                self.load()

    def test_rejects_invalid_window_and_partition_settings(self):
        self.write()
        for changes in (dict(horizon_steps=0), dict(horizon_steps=1.5),
                        dict(val_fraction=0), dict(val_fraction=1), dict(val_fraction=np.nan),
                        dict(max_n_episodes=0), dict(max_n_episodes=1),
                        dict(max_n_episodes=2.5), dict(horizon_steps=100)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.load(**changes)

    def test_array_constructor_rejects_partitions_without_full_windows(self):
        for episode_ids, horizon_steps in (([], 3), ([3], 4)):
            with self.subTest(episode_ids=episode_ids), self.assertRaisesRegex(ValueError, "no full-horizon"):
                ActionChunkDataset(self.actions, self.states, self.lengths, episode_ids, horizon_steps)


class IdentityChord(torch.nn.Module):
    hand_dim = 16

    def __init__(self):
        super().__init__()
        self.calls = []

    def check_inference(self):
        self.calls.append((self.training, torch.is_grad_enabled()))

    def get_relative_hand_targets(self, actions, state):
        self.check_inference()
        return actions[..., 6:] - state[:, None, 7:]

    def posterior(self, actions, state):
        self.check_inference()
        mean = actions[..., 6:8].mean(dim=1)
        return mean, torch.zeros_like(mean)

    def encode(self, actions, state):
        return actions

    def decode(self, encoded, state):
        decoded = encoded.clone()
        decoded[..., 6:] += 0.125
        return decoded

    def reconstruct(self, actions, state):
        mean, logvar = self.posterior(actions, state)
        return self.decode(self.encode(actions, state), state), mean, logvar

    def get_hand_targets(self, actions):
        return actions[..., 6:]

    def get_initial_hand_positions(self, state):
        return state[..., 7:]


class HandChordTrainingTests(unittest.TestCase):
    def dataset(self):
        actions = np.broadcast_to(np.arange(8, dtype=np.float32)[:, None], (8, 22)).copy()
        states = np.zeros((8, 23), dtype=np.float32)
        return ActionChunkDataset(actions, states, [4, 4], [0], 3)

    def test_statistics_use_only_selected_training_windows_in_eval_without_gradients(self):
        dataset = self.dataset()
        chord = IdentityChord()
        expected = np.concatenate([dataset.actions[start:start + 3, 6:].numpy()
                                   for start in dataset.starts])
        mean, std = fit_statistics(chord, dataset, 1, "cpu")
        np.testing.assert_allclose(mean.numpy(), expected.mean(axis=0))
        np.testing.assert_allclose(std.numpy(), expected.std(axis=0), rtol=1e-6)
        self.assertTrue(chord.calls)
        self.assertTrue(all(call == (False, False) for call in chord.calls))

    def test_latent_statistics_use_global_chunk_posterior_means(self):
        chord = IdentityChord()
        mean, std = fit_statistics(chord, self.dataset(), 1, "cpu", latent=True)
        np.testing.assert_allclose(mean.numpy(), [1.5, 1.5])
        np.testing.assert_allclose(std.numpy(), [0.5, 0.5])

    def test_evaluation_reports_only_physical_mae_without_changing_training_rng(self):
        dataset = self.dataset()
        chord = IdentityChord()
        rng_state = torch.get_rng_state().clone()
        metrics = evaluate(chord, dataset, 2, "cpu")
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_state))
        self.assertAlmostEqual(metrics["mae_deg"], np.rad2deg(0.125), places=4)
        self.assertEqual(set(metrics), {"mae_deg"})

    def test_kl_schedule_warms_up_and_reaches_configured_maximum(self):
        self.assertEqual(kl_weight(3999), 0)
        self.assertEqual(kl_weight(4000), 0)
        self.assertAlmostEqual(kl_weight(6000), 1.5e-4)
        self.assertEqual(kl_weight(8000), 3e-4)
        self.assertEqual(kl_weight(9000), 3e-4)
        self.assertEqual(kl_weight(1, maximum=0.2, zero_steps=1, ramp_steps=0), 0.2)


if __name__ == "__main__":
    unittest.main()
