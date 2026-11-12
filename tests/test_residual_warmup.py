"""Warmup regression tests without simulator or pretrained-policy assets."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from agent.finetune.train_distill_residual_flow_agent import TrainDistillResidualFlowAgent
from agent.finetune.train_distill_residual_flow_img_agent import TrainDistillResidualFlowImgAgent
from model.hand_chord.chunk_hand_chord import ChunkHandActionChord
from model.rl.hand_chord_residual_rl_img import HandChordResidualRLImgModel
from util.hybrid_replay_buffer import HybridReplayBuffer
from util.scheduler import CosineAnnealingWarmupRestarts


class RawPolicy(nn.Module):
    def forward_from_features(self, features, init_noise):
        return SimpleNamespace(trajectories=init_noise * .1)


class ConstantQ(nn.Module):
    def forward(self, state, noise, action):
        return torch.full((len(state), 1), 2.)


class ResidualWarmupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(42)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        normalizer = root / "normalization.npz"
        np.savez(normalizer, action_min=np.full(22, -1.),
                 action_max=np.full(22, 1.), obs_min=np.full(23, -2.),
                 obs_max=np.full(23, 2.))
        chord = ChunkHandActionChord(normalization_path=normalizer,
                                     horizon_steps=2, latent_dim=3,
                                     width=8, layers=1, heads=2)
        chord.save(root / "chord.pt")
        with patch.object(HandChordResidualRLImgModel, "_load_pretrained_policy",
                          return_value=RawPolicy()):
            self.model = HandChordResidualRLImgModel(
                obs_dim=27, action_dim=22, horizon_steps=2, cond_steps=1,
                pretrained_flow_policy_path="unused.pt", chord_path=str(root / "chord.pt"),
                actor_hidden_dims=[8], critic_hidden_dims=[8], device="cpu",
                use_n_step=True, num_multi_z_for_actor_loss=1,
            )

    def make_agent(self, use_rlpd=False, pretrain_steps=3, warmup=4, log_mc=False):
        agent = TrainDistillResidualFlowAgent.__new__(TrainDistillResidualFlowAgent)
        agent.cfg = OmegaConf.create({
            "use_rlpd": use_rlpd, "expert_ratio": .7,
            "train": {"buffer_warmup_transitions": warmup,
                      "critic_pretrain_steps": pretrain_steps,
                      "critic_pretrain_expert_ratio": .5,
                      "critic_pretrain_lr": .002, "critic_lr": .001,
                      "batch_size": 4},
            "replay_buffer": {"max_size": 32},
        })
        agent._configure_warmup(agent.cfg)
        agent.model = self.model
        agent.batch_size = 4
        agent.device = "cpu"
        agent.gamma = .99
        agent.tau = .1
        agent.max_grad_norm = 1.
        agent.gradient_steps = 2
        agent.critic_target_update_freq = 2
        agent.log_q_overestimation = log_mc
        agent.use_wandb = False
        agent.current_training_step = 7
        agent.use_adaptive_expert_ratio = True
        agent.adaptive_expert_ratio_start = .7
        agent.adaptive_expert_ratio_end = .1
        agent.adaptive_expert_ratio_steps = 100
        agent.actor_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=.001)
        agent.critic_optimizer = torch.optim.Adam(self.model.critic.parameters(), lr=.001)
        agent.actor_scheduler = CosineAnnealingWarmupRestarts(
            agent.actor_optimizer, first_cycle_steps=10, warmup_steps=1, min_lr=.0001)
        agent.critic_scheduler = CosineAnnealingWarmupRestarts(
            agent.critic_optimizer, first_cycle_steps=10, warmup_steps=1, min_lr=.0001)
        demo = SimpleNamespace(
            conditions={"state": torch.zeros(1, 27), "next_state": torch.ones(1, 27)},
            actions=torch.zeros(2, 22), rewards=torch.ones(1),
            dones=torch.ones(1), mc_return=torch.ones(1))
        agent.replay_buffer = HybridReplayBuffer(
            obs_dim=27, action_dim=22, noise_dim=44, max_size=32,
            horizon_steps=2, device="cpu", use_rlpd=use_rlpd,
            expert_dataset=[demo] if use_rlpd else None,
            log_q_overestimation=log_mc,
        )
        trajectory = dict(initial_obs=torch.zeros(27),
                          observations=[torch.ones(27) * t for t in range(1, 6)],
                          actions=[torch.zeros(22) for _ in range(5)],
                          rewards=[0., 0., 0., 0., 1.],
                          dones=[False] * 4 + [True], include_initial=True)
        agent.replay_buffer.add(None, None, None, None, None,
                                torch.tensor([[True]]), [trajectory])
        return agent

    def test_base_action_bypasses_actor_and_matches_codec_projection(self):
        state, noise = torch.zeros(2, 1, 27), torch.randn(2, 2, 22)
        expected = self.model.chord.decode(
            self.model.chord.encode(noise * .1, state[..., :23]), state[..., :23])
        with patch.object(self.model.actor, "forward", side_effect=AssertionError("actor called")):
            action = self.model.get_action(state, noise, use_residual=False)
        torch.testing.assert_close(action, expected)

    def test_critic_only_loss_skips_actor_and_preserves_n_step_target(self):
        self.model.target_critic = ConstantQ()
        for multi in (False, True):
            with self.subTest(multi_sample=multi):
                self.model.multi_sample_next_noise = multi
                with patch.object(self.model.actor, "forward", side_effect=AssertionError("actor called")):
                    target, _ = self.model._compute_target_q(
                        torch.zeros(2, 1, 27), torch.tensor([[1.], [3.]]),
                        torch.tensor([[False], [True]]), .5,
                        torch.tensor([[3], [1]]), use_residual=False)
                    losses = self.model.loss(
                        state=torch.zeros(2, 1, 27), noise=torch.zeros(2, 2, 22),
                        action=torch.zeros(2, 2, 22), next_state=torch.zeros(2, 1, 27),
                        reward=torch.ones(2, 1), done=torch.zeros(2, 1),
                        n_steps=torch.ones(2, 1), critic_only=True)
                torch.testing.assert_close(target, torch.tensor([[1.25], [3.]]))
                self.assertNotIn("actor_total", losses)
                losses["critic_loss"].backward()
                self.assertTrue(all(p.grad is None for p in self.model.actor.parameters()))

    def test_pretraining_updates_only_critic_and_restores_online_lr(self):
        agent = self.make_agent()
        before = {name: p.detach().clone() for name, p in self.model.named_parameters()}
        actor_schedule = agent.actor_scheduler.state_dict().copy()
        critic_schedule = agent.critic_scheduler.state_dict().copy()
        online_lr = agent.critic_optimizer.param_groups[0]["lr"]
        with patch.object(self.model.actor, "forward", side_effect=AssertionError("actor called")):
            agent._pretrain_critic()
        changed = {name for name, p in self.model.named_parameters()
                   if not torch.equal(before[name], p)}
        self.assertTrue(any(name.startswith("critic.") for name in changed))
        self.assertTrue(any(name.startswith("target_critic.") for name in changed))
        self.assertTrue(all(name.startswith(("critic.", "target_critic.")) for name in changed))
        self.assertEqual(agent.critic_pretrain_updates, 3)
        self.assertEqual(agent.current_training_step, 7)
        self.assertTrue(agent.warmup_complete)
        self.assertFalse(agent.actor_optimizer.state)
        self.assertEqual(agent.actor_scheduler.state_dict(), actor_schedule)
        self.assertEqual(agent.critic_scheduler.state_dict(), critic_schedule)
        self.assertEqual(agent.critic_optimizer.param_groups[0]["lr"], online_lr)

    def test_demo_toggle_controls_pretraining_batch_sources(self):
        for demos, log_mc in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(demos=demos, log_mc=log_mc):
                agent = self.make_agent(use_rlpd=demos, pretrain_steps=1, log_mc=log_mc)
                with patch.object(self.model, "loss", wraps=self.model.loss) as loss:
                    agent._pretrain_critic()
                sources = loss.call_args.kwargs["data_source"]
                self.assertEqual(sources.sum().item(), 2 if demos else 0)
                self.assertEqual(len(sources), 4)
                if not demos:
                    self.assertEqual(agent.get_current_expert_ratio(0), 0.)

    def test_implicit_warmup_does_not_update_weights_or_sample(self):
        agent = self.make_agent(pretrain_steps=0)
        with patch.object(agent.replay_buffer, "sample_batches", side_effect=AssertionError("sampled")):
            agent._pretrain_critic()
        self.assertTrue(agent.warmup_complete)
        self.assertEqual(agent.critic_pretrain_updates, 0)
        self.assertFalse(agent.critic_optimizer.state)

    def test_disabled_warmup_and_invalid_settings(self):
        agent = self.make_agent(pretrain_steps=0, warmup=0)
        self.assertTrue(agent.warmup_complete)
        for key, value in (("critic_pretrain_steps", -1),
                           ("critic_pretrain_steps", 1.5),
                           ("buffer_warmup_transitions", 33),
                           ("critic_pretrain_expert_ratio", 1.1),
                           ("critic_pretrain_lr", 0)):
            with self.subTest(key=key):
                cfg = OmegaConf.create(OmegaConf.to_container(agent.cfg))
                cfg.train[key] = value
                with self.assertRaises(ValueError):
                    agent._configure_warmup(cfg)

    def test_capacity_check_defers_to_buffer_when_max_size_is_omitted(self):
        agent = self.make_agent()
        cfg = OmegaConf.create(OmegaConf.to_container(agent.cfg))
        cfg.train.buffer_warmup_transitions = 200000
        del cfg.replay_buffer.max_size
        agent._configure_warmup(cfg)
        self.assertEqual(agent.buffer_warmup_transitions, 200000)

    def test_image_agent_without_demos_does_not_require_expert_config(self):
        cfg = OmegaConf.create({"obs_dim": 23, "base_policy_path": "/unused/checkpoint/state.pt",
                                "flow_steps": 10, "use_rlpd": False})
        pretrained = OmegaConf.create({"flow_steps": 10, "model": {"network": {
            "_target_": "VisionFlowMatchingMLP", "visual_feature_dim": 4}}})
        def initialize_parent(agent, cfg):
            agent.model = SimpleNamespace(pretrained_flow_policy=RawPolicy(), obs_dim=cfg.obs_dim)
        with patch("agent.finetune.train_distill_residual_flow_img_agent.OmegaConf.load",
                   return_value=pretrained), patch.object(
                       TrainDistillResidualFlowAgent, "__init__", initialize_parent):
            agent = TrainDistillResidualFlowImgAgent(cfg)
        self.assertEqual(agent.obs_dim, 27)

    def test_no_demo_mode_skips_expert_features_and_visualization_io(self):
        cfg = OmegaConf.create({
            "use_rlpd": False, "replay_buffer": {"max_size": 8,
                "expert_n_step": "${expert_dataset.n_step}"}, "train": {"gamma": .99}})
        agent = TrainDistillResidualFlowImgAgent.__new__(TrainDistillResidualFlowImgAgent)
        agent.cfg = cfg
        agent.obs_dim, agent.action_dim, agent.noise_dim = 27, 22, 44
        agent.n_envs, agent.cond_steps, agent.horizon_steps, agent.device = 1, 1, 2, "cpu"
        with patch.object(agent, "_preprocess_expert_dataset", side_effect=AssertionError("demo read")), \
                patch("numpy.load", side_effect=AssertionError("demo file read")):
            agent._setup_replay_buffer(cfg)
            self.assertEqual(agent.extract_expert_trajectories(), [])
        self.assertIsNone(agent.replay_buffer.expert_dataset)

    def test_pretraining_restores_lr_after_failure_without_marking_complete(self):
        agent = self.make_agent()
        online_lr = agent.critic_optimizer.param_groups[0]["lr"]
        with patch.object(self.model, "loss", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                agent._pretrain_critic()
        self.assertEqual(agent.critic_optimizer.param_groups[0]["lr"], online_lr)
        self.assertFalse(agent.warmup_complete)

    def test_training_loop_collects_before_learning_and_pretrains_once(self):
        class ToyVectorEnv:
            def __init__(self):
                self.calls = 0

            def step(self, actions):
                self.calls += 1
                first, done = self.calls % 2 == 1, self.calls % 2 == 0
                trajectory = dict(
                    initial_obs={"state": np.zeros((1, 27), dtype=np.float32)} if first else None,
                    observations=[{"state": np.zeros((1, 27), dtype=np.float32)}] * 2,
                    actions=list(actions[0]), rewards=[0., float(done)],
                    dones=[False, done], include_initial=first)
                return ({"state": np.zeros((1, 1, 27), dtype=np.float32)},
                        np.array([float(done)]), np.array([done]), np.array([False]),
                        [{"full_trajectory": trajectory}])

        cases = ((False, 0, 4, 6), (False, 3, 4, 6),
                 (True, 0, 4, 6), (True, 3, 4, 6),
                 (False, 0, 0, 6), (False, 3, 4, 3))
        for demos, updates, warmup, budget in cases:
            with self.subTest(demos=demos, updates=updates, warmup=warmup, budget=budget):
                agent = self.make_agent(use_rlpd=demos, pretrain_steps=updates, warmup=warmup)
                agent.replay_buffer.clear()
                agent.cfg.run_eval = False
                agent.venv = ToyVectorEnv()
                agent.n_envs, agent.horizon_steps, agent.act_steps, agent.action_dim = 1, 2, 2, 22
                agent.num_train_steps, agent.eval_freq, agent.save_freq = budget, 100, 100
                agent.log_freq, agent.update_freq = 1, 1
                agent.actor_update_freq = 2
                agent.log_q_overestimation = False
                agent.visualize_trajectories = False
                agent.online_explore_strategy = "standard"
                obs = {"state": np.zeros((1, 1, 27), dtype=np.float32)}
                with patch.object(agent, "reset_env_all", return_value=obs), \
                        patch.object(agent, "save_model"), \
                        patch.object(agent, "_pretrain_critic", wraps=agent._pretrain_critic) as pretrain, \
                        patch.object(agent, "get_action", wraps=agent.get_action) as action, \
                        patch.object(agent, "update_networks", wraps=agent.update_networks) as update:
                    agent.run()
                self.assertEqual(pretrain.call_count, int(warmup > 0 and budget >= 4))
                self.assertEqual(agent.critic_pretrain_updates, updates if budget >= 4 else 0)
                self.assertEqual(agent.simulator_steps, 2 * budget)
                self.assertEqual(agent.current_training_step, budget)
                collecting = min(4, budget) if warmup else 0
                self.assertEqual([call.kwargs.get("use_residual", True)
                                  for call in action.call_args_list],
                                 [False] * collecting + [True] * (budget - collecting))
                self.assertTrue(all(call.kwargs["training_step"] >= 3
                                    for call in update.call_args_list))
                self.assertEqual(update.call_count, 6 if budget >= 4 else 0)


if __name__ == "__main__":
    unittest.main()
