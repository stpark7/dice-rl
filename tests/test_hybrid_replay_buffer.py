"""Regression tests for CPU replay storage and grouped training batches."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from util.hybrid_replay_buffer import HybridReplayBuffer


class HybridReplayBufferTests(unittest.TestCase):
    def make_buffer(self, **kwargs):
        options = dict(obs_dim=3, action_dim=2, noise_dim=2,
                       max_size=32, horizon_steps=2, device="cpu")
        options.update(kwargs)
        return HybridReplayBuffer(**options)

    def add_episode(self, buffer, length=8, offset=0, requires_grad=False,
                    complete=True):
        observations = [torch.tensor([offset + t, offset + t + .1,
                                      offset + t + .2],
                                     dtype=torch.float32,
                                     requires_grad=requires_grad)
                        for t in range(length + 1)]
        actions = [torch.tensor([t, t + .5], dtype=torch.float32,
                                requires_grad=requires_grad)
                   for t in range(length)]
        info = dict(initial_obs=observations[0], observations=observations[1:],
                    actions=actions, rewards=[.25] * length,
                    dones=[False] * (length - 1) + [complete],
                    include_initial=True)
        buffer.add(None, None, None, None, None,
                   torch.tensor([[complete]]), [info])
        return observations, actions

    def expert_dataset(self):
        return [SimpleNamespace(
            conditions={"state": np.full((1, 3), 100., dtype=np.float32),
                        "next_state": np.full((1, 3), 102., dtype=np.float32)},
            actions=np.full((2, 2), 50., dtype=np.float32),
            rewards=np.array([1.], dtype=np.float32),
            dones=np.array([True]),
            mc_return=np.array([2.], dtype=np.float32),
        )]

    def test_complete_episode_preserves_chunks_history_and_returns(self):
        buffer = self.make_buffer(cond_steps=2, log_q_overestimation=True,
                                  gamma=.5)
        observations, actions = self.add_episode(buffer, length=5)
        self.assertEqual(buffer.size, 3)
        self.assertEqual(buffer.num_episodes, 1)
        torch.testing.assert_close(buffer.chunk_states[0],
                                   torch.stack(observations[:2]))
        torch.testing.assert_close(buffer.chunk_next_states[0],
                                   torch.stack(observations[2:4]))
        torch.testing.assert_close(buffer.chunk_actions[0],
                                   torch.stack(actions[1:3]))
        torch.testing.assert_close(buffer.chunk_rewards[:3],
                                   torch.full((3, 1), .5))
        torch.testing.assert_close(buffer.chunk_dones[:3, 0],
                                   torch.tensor([False, False, True]))
        torch.testing.assert_close(buffer.chunk_mc_returns[0],
                                   torch.tensor([.75]))
        self.assertTrue(all(t.device.type == "cpu" for t in buffer.sample(4)))

    def test_cpu_ring_buffer_keeps_capacity_and_overwrites_old_chunks(self):
        buffer = self.make_buffer(max_size=3)
        self.add_episode(buffer, length=5)
        self.assertTrue(buffer.is_full())
        self.assertEqual(buffer.ptr, 1)
        self.assertEqual(sorted(buffer.chunk_states[:, 0, 0].tolist()),
                         [1., 2., 3.])
        self.add_episode(buffer, length=4, offset=10)
        self.assertEqual(buffer.size, 3)
        self.assertEqual(buffer.ptr, 1)
        self.assertEqual(sorted(buffer.chunk_states[:, 0, 0].tolist()),
                         [10., 11., 12.])

    def test_pending_and_completed_storage_detach_autograd(self):
        buffer = self.make_buffer()
        self.add_episode(buffer, length=3, requires_grad=True, complete=False)
        episode = buffer.ongoing_episodes[0]
        self.assertIsNotNone(episode)
        for tensor in episode["observations"] + episode["actions"]:
            self.assertEqual(tensor.device.type, "cpu")
            self.assertFalse(tensor.requires_grad)
        self.add_episode(buffer, requires_grad=True)
        for tensor in (buffer.chunk_states, buffer.chunk_next_states,
                       buffer.chunk_actions):
            self.assertFalse(tensor.requires_grad)
            self.assertIsNone(tensor.grad_fn)

    def test_n_step_mc_returns_and_dtypes_are_preserved(self):
        buffer = self.make_buffer(use_n_step=True, n_step=2,
                                  log_q_overestimation=True, gamma=.5)
        self.add_episode(buffer)
        indices = np.array([0, 3, 4])
        with patch("util.hybrid_replay_buffer.np.random.randint",
                   return_value=indices):
            batch = buffer.sample(3)
        state, noise, action, reward, next_state, done, mc, n_steps, source = batch
        torch.testing.assert_close(state[:, 0, 0], torch.tensor([0., 3., 4.]))
        torch.testing.assert_close(next_state[:, 0, 0], torch.tensor([4., 7., 8.]))
        torch.testing.assert_close(reward, torch.ones(3, 1))
        torch.testing.assert_close(done[:, 0], torch.tensor([False, False, True]))
        torch.testing.assert_close(mc[:, 0], torch.tensor([.9375, .8125, .75]))
        torch.testing.assert_close(n_steps, torch.full((3, 1), 2, dtype=torch.long))
        torch.testing.assert_close(source, torch.zeros(3, 1, dtype=torch.long))
        self.assertEqual(done.dtype, torch.bool)
        self.assertEqual(n_steps.dtype, torch.long)
        self.assertEqual(action.dtype, torch.float32)
        self.assertEqual(noise.shape, (3, 2, 2))

    def test_grouped_batches_keep_exact_expert_ratio_per_update(self):
        buffer = self.make_buffer(use_rlpd=True, expert_ratio=.3,
                                  expert_dataset=self.expert_dataset(),
                                  expert_n_step=3, log_q_overestimation=True)
        self.add_episode(buffer)
        batches = list(buffer.sample_batches(7, 4))
        self.assertEqual(len(batches), 4)
        for batch in batches:
            self.assertEqual(len(batch), 9)
            self.assertTrue(all(t.shape[0] == 7 for t in batch))
            state, _, _, reward, _, done, mc, steps, source = batch
            self.assertEqual(source.sum().item(), 2)
            mask = source[:, 0].bool()
            self.assertTrue(torch.all(state[mask] == 100.))
            self.assertTrue(torch.all(state[~mask] < 10.))
            self.assertTrue(torch.all(steps[mask] == 3))
            self.assertTrue(torch.all(steps[~mask] == 1))
            self.assertTrue(torch.all(reward[mask] == 1.))
            self.assertTrue(torch.all(mc[mask] == 2.))
            self.assertTrue(torch.all(done[mask]))

    def test_grouped_sampling_respects_ratio_override_and_zero_experts(self):
        buffer = self.make_buffer(use_rlpd=True, expert_ratio=.9,
                                  expert_dataset=self.expert_dataset())
        self.add_episode(buffer)
        for ratio, expected in ((0., 0), (.5, 3), (1., 7)):
            for batch in buffer.sample_batches(7, 3, expert_ratio=ratio):
                self.assertEqual(batch[-1].sum().item(), expected)
                self.assertEqual(batch[0].shape[0], 7)

    def test_grouped_online_batches_keep_shapes_and_independent_values(self):
        buffer = self.make_buffer(noise_action_dim=5)
        self.add_episode(buffer)
        np.random.seed(42)
        batches = list(buffer.sample_batches(4, 3))
        self.assertEqual(len(batches), 3)
        snapshots = [[tensor.clone() for tensor in batch] for batch in batches]
        for batch in batches:
            self.assertEqual(batch[0].shape, (4, 1, 3))
            self.assertEqual(batch[1].shape, (4, 2, 5))
            self.assertEqual(batch[2].shape, (4, 2, 2))
            self.assertTrue(all(t.device.type == "cpu" for t in batch))
        self.add_episode(buffer, length=8, offset=100)
        list(buffer.sample_batches(4, 3))
        for batch, snapshot in zip(batches, snapshots):
            for tensor, expected in zip(batch, snapshot):
                torch.testing.assert_close(tensor, expected)

    def test_empty_buffer_raises_and_clear_resets_episode_storage(self):
        buffer = self.make_buffer()
        with self.assertRaises(ValueError):
            buffer.sample(2)
        with self.assertRaises(ValueError):
            list(buffer.sample_batches(2, 3))
        self.add_episode(buffer)
        buffer.clear()
        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer.ongoing_episodes, [None])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cpu_storage_transfers_typed_batches_to_cuda_without_corruption(self):
        buffer = self.make_buffer(device="cuda:0", storage_device="cpu",
                                  pin_memory=True, log_q_overestimation=True)
        self.add_episode(buffer, requires_grad=True)
        self.assertEqual(buffer.chunk_states.device.type, "cpu")
        self.assertFalse(buffer.chunk_states.is_pinned())
        np.random.seed(12)
        actual = list(buffer.sample_batches(4, 3))
        snapshots = [[tensor.clone() for tensor in batch] for batch in actual]
        for batch in actual:
            self.assertTrue(all(t.device.type == "cuda" for t in batch))
            state, _, action, reward, next_state, done, mc, steps, source = batch
            torch.testing.assert_close(next_state[:, 0, 0], state[:, 0, 0] + 2)
            torch.testing.assert_close(action[:, 0, 0], state[:, 0, 0])
            self.assertEqual(done.dtype, torch.bool)
            self.assertEqual(steps.dtype, torch.long)
            self.assertEqual(source.dtype, torch.long)
            self.assertEqual(mc.dtype, torch.float32)
            weight = torch.ones(3, 2, device="cuda", requires_grad=True)
            ((state[:, 0] @ weight - action[:, 0]).square().mean()
             + reward.mean()).backward()
            self.assertIsNotNone(weight.grad)
        for _ in range(3):
            list(buffer.sample_batches(4, 3))
        torch.cuda.synchronize()
        for batch, snapshot in zip(actual, snapshots):
            for tensor, expected in zip(batch, snapshot):
                torch.testing.assert_close(tensor, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_mixed_expert_batches_match_cpu_n_step_and_mc_results(self):
        options = dict(use_rlpd=True, expert_ratio=.3,
                       expert_dataset=self.expert_dataset(), expert_n_step=3,
                       log_q_overestimation=True, use_n_step=True, n_step=2,
                       gamma=.5, noise_action_dim=5)
        cpu_buffer = self.make_buffer(**options)
        cuda_buffer = self.make_buffer(device="cuda:0", **options)
        for buffer in (cpu_buffer, cuda_buffer):
            self.add_episode(buffer)
        np.random.seed(37)
        expected = list(cpu_buffer.sample_batches(7, 3))
        np.random.seed(37)
        actual = list(cuda_buffer.sample_batches(7, 3))
        self.assertEqual(len(actual), 3)
        for cuda_batch, cpu_batch in zip(actual, expected):
            self.assertTrue(all(field.device == torch.device("cuda:0")
                                for field in cuda_batch))
            self.assertEqual(cuda_batch[1].shape, (7, 2, 5))
            self.assertTrue(torch.isfinite(cuda_batch[1]).all())
            self.assertEqual(cuda_batch[-1].sum().item(), 2)
            # Noise is generated on the sampling device; all stored fields and
            # computed returns must agree with the same CPU sampling indices.
            for index, (cuda_field, cpu_field) in enumerate(zip(cuda_batch, cpu_batch)):
                if index != 1:
                    torch.testing.assert_close(cuda_field.cpu(), cpu_field)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_transfer_without_pinned_memory_keeps_values_and_dtypes(self):
        buffer = self.make_buffer(device="cuda:0", pin_memory=False)
        self.add_episode(buffer)
        self.assertFalse(buffer.pin_memory)
        for batch in buffer.sample_batches(3, 2):
            state, _, action, reward, next_state, done, steps, source = batch
            self.assertTrue(all(t.device.type == "cuda" for t in batch))
            torch.testing.assert_close(next_state[:, 0, 0], state[:, 0, 0] + 2)
            torch.testing.assert_close(action[:, 0, 0], state[:, 0, 0])
            torch.testing.assert_close(reward, torch.full_like(reward, .5))
            self.assertEqual(done.dtype, torch.bool)
            self.assertEqual(steps.dtype, torch.long)
            self.assertEqual(source.dtype, torch.long)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_gpu_storage_override_keeps_sampling_and_indexed_batches_on_gpu(self):
        buffer = self.make_buffer(device="cuda:0", storage_device="cuda:0")
        self.add_episode(buffer, requires_grad=True)
        self.assertEqual(buffer.chunk_states.device, torch.device("cuda:0"))
        self.assertFalse(buffer.pin_memory)
        self.assertFalse(buffer.chunk_states.requires_grad)
        for batch in buffer.sample_batches(3, 2):
            self.assertTrue(all(t.device == torch.device("cuda:0") for t in batch))
            torch.testing.assert_close(batch[4][:, 0, 0], batch[0][:, 0, 0] + 2)
        states, actions = buffer.get_batch_from_indices(torch.tensor([0, 2]))
        torch.testing.assert_close(states[:, 0, 0],
                                   torch.tensor([0., 2.], device="cuda:0"))
        torch.testing.assert_close(actions[:, 0, 0], states[:, 0, 0])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_gpu_source_episode_tensors_are_detached_on_cpu(self):
        buffer = self.make_buffer(device="cuda:0")
        source = torch.arange(3., device="cuda:0", requires_grad=True)
        stored = buffer._to_tensor({"state": source * 2})["state"]
        self.assertEqual(stored.device.type, "cpu")
        self.assertFalse(stored.requires_grad)
        self.assertIsNone(stored.grad_fn)
        torch.testing.assert_close(stored, torch.tensor([0., 2., 4.]))


if __name__ == "__main__":
    unittest.main()
