"""DAWN SAC regression tests using a small frozen native BC policy."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from model.rl.dawn import DAWNModel
from model.rl.resfit import ResFiTModel


class TinyVision(nn.Module):
    visual_feature_dim = 2
    num_img = 1

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 2)

    def extract_visual_features(self, cond):
        return self.projection(cond["rgb"][:, 0].mean(dim=(-1, -2)))


class TinyBC(nn.Module):
    action_dim = 2
    horizon_steps = 3
    cond_steps = 1

    def __init__(self):
        super().__init__()
        self.network = TinyVision()
        self.action = nn.Parameter(torch.tensor([.2, -.3]))

    def forward(self, cond, deterministic=True):
        return SimpleNamespace(trajectories=self.action.expand(len(cond["state"]), 3, 2))


class ConstantQs(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = values
        self.last_action = None

    def forward(self, features, action):
        self.last_action = action.detach().clone()
        return action.new_tensor(self.values)[:, None].expand(2, len(features))


class DAWNModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(5)
        self.model = DAWNModel(
            obs_dim=3, action_dim=2, horizon_steps=3, pretrained_policy=TinyBC(),
            actor_hidden_dim=8, critic_hidden_dim=8, critic_num_blocks=1,
            critic_expansion=2, sac_alpha=.2,
        )
        self.batch = dict(
            obs=torch.randn(2, 5), base_action=torch.zeros(2, 2),
            action=torch.randn(2, 2), reward=torch.tensor([[1.], [3.]]),
            discount=torch.tensor([[.5], [0.]]), next_obs=torch.randn(2, 5),
            next_base_action=torch.ones(2, 2),
        )

    def test_native_observations_and_bc_chunks_match_resfit(self):
        resfit = ResFiTModel(
            obs_dim=3, action_dim=2, horizon_steps=3,
            pretrained_policy=self.model.base_policy,
            actor_hidden_dims=[8], critic_hidden_dims=[8], critic_ensemble_size=2,
        )
        obs = dict(state=torch.randn(2, 1, 3), rgb=torch.ones(2, 1, 4, 4, 3) * 128)
        torch.testing.assert_close(self.model.encode(obs), resfit.encode(obs))
        torch.testing.assert_close(self.model.base_actions(obs), resfit.base_actions(obs))
        self.assertEqual(tuple(self.model.encode(obs).shape), (2, 5))
        self.assertFalse(self.model.encode(obs).requires_grad)
        self.assertTrue(all(not p.requires_grad for p in self.model.base_policy.parameters()))
        self.model.train()
        self.assertFalse(self.model.base_policy.training)
        self.assertFalse(self.model.critic_target.training)
        self.assertTrue(self.model.actor.training)
        self.assertFalse(hasattr(self.model, "actor_target"))

    def test_gaussian_log_probability_uses_raw_residual_without_scale_jacobian(self):
        for parameter in self.model.actor.parameters():
            nn.init.zeros_(parameter)
        features, base = self.batch["obs"], torch.ones(2, 2)
        torch.manual_seed(11)
        action, log_prob = self.model.sample_action(features, base)
        torch.manual_seed(11)
        distribution = torch.distributions.Normal(torch.zeros(2, 2), torch.full((2, 2), -9.).exp())
        latent = distribution.rsample()
        residual = latent.tanh()
        expected = (distribution.log_prob(latent) - torch.log(1 - residual.square() + 1e-6)).sum(-1, keepdim=True)
        torch.testing.assert_close(action, base + .1 * residual)
        torch.testing.assert_close(log_prob, expected)
        self.assertEqual(tuple(log_prob.shape), (2, 1))

    def test_eval_action_uses_mean_and_retains_unclipped_action_for_critic(self):
        for parameter in self.model.actor.parameters():
            nn.init.zeros_(parameter)
        nn.init.constant_(self.model.actor.fc_mean.bias, 2.)
        base = torch.ones(2, 2)
        expected = base + .1 * torch.full_like(base, 2.).tanh()
        actual = self.model.act(self.batch["obs"], base)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.all(actual > 1))
        with patch.object(self.model.actor, "get_action", wraps=self.model.actor.get_action) as sample:
            self.model.act(self.batch["obs"], base, sample=True)
        sample.assert_called_once()

    def test_soft_bellman_target_uses_minimum_q_entropy_and_terminal_discount(self):
        self.model.critics = ConstantQs([0., 0.])
        self.model.critic_target = ConstantQs([4., 6.])
        next_action = torch.full((2, 2), 1.5)
        with patch.object(self.model, "sample_action", return_value=(next_action, torch.full((2, 1), 2.))):
            loss = self.model.critic_loss(self.batch)
        target = torch.tensor([1. + .5 * (4. - .2 * 2.), 3.])
        torch.testing.assert_close(loss, 2 * target.square().mean())
        torch.testing.assert_close(self.model.critic_target.last_action, next_action)
        with patch.object(self.model, "sample_action", return_value=(next_action, torch.full((2, 1), 2.))):
            hard_loss = self.model.critic_loss(self.batch, include_entropy=False)
        hard_target = torch.tensor([1. + .5 * 4., 3.])
        torch.testing.assert_close(hard_loss, 2 * hard_target.square().mean())

    def test_actor_objective_uses_minimum_q_and_entropy(self):
        self.model.critics = ConstantQs([4., 6.])
        with patch.object(self.model, "sample_action", return_value=(torch.ones(2, 2), torch.full((2, 1), 2.))):
            loss = self.model.actor_loss(self.batch)
        torch.testing.assert_close(loss, torch.tensor(.2 * 2. - 4.))

    def test_actor_gradients_do_not_reach_critics_base_policy_or_temperature(self):
        loss = self.model.actor_loss(self.batch)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.actor.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.critics.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.model.critics.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.base_policy.parameters()))
        self.assertIsNone(self.model.log_alpha.grad)

    def test_critic_gradients_do_not_reach_actor_or_target(self):
        self.model.critic_loss(self.batch).backward()
        self.assertTrue(any(p.grad is not None for p in self.model.critics.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.actor.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.critic_target.parameters()))
        self.assertIsNone(self.model.log_alpha.grad)

    def test_temperature_loss_uses_detached_log_probability_and_configured_entropy(self):
        self.model.target_entropy = -3.
        log_prob = torch.full((2, 1), 2., requires_grad=True)
        with patch.object(self.model, "sample_action", return_value=(torch.ones(2, 2), log_prob)):
            loss = self.model.temperature_loss(self.batch)
        torch.testing.assert_close(loss, self.model.log_alpha)
        loss.backward()
        torch.testing.assert_close(self.model.log_alpha.grad, torch.ones_like(self.model.log_alpha))
        self.assertIsNone(log_prob.grad)
        self.assertTrue(all(p.grad is None for p in self.model.actor.parameters()))

    def test_target_update_only_changes_critics(self):
        actor_before = [p.clone() for p in self.model.actor.parameters()]
        for parameter in self.model.critics.parameters():
            nn.init.constant_(parameter, 2.)
        for parameter in self.model.critic_target.parameters():
            nn.init.constant_(parameter, 0.)
        self.model.update_targets(.25)
        for parameter in self.model.critic_target.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, .5))
        for actual, expected in zip(self.model.actor.parameters(), actor_before):
            torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            self.model.update_targets(1.1)

    def test_fixed_temperature_and_invalid_settings(self):
        model = DAWNModel(
            obs_dim=3, action_dim=2, horizon_steps=3, pretrained_policy=TinyBC(),
            actor_hidden_dim=8, critic_hidden_dim=8, autotune=False, sac_alpha=.3,
        )
        torch.testing.assert_close(model.alpha, torch.tensor(.3))
        self.assertFalse(model.log_alpha.requires_grad)
        self.assertEqual(model.target_entropy, -2.)
        for kwargs in (dict(residual_scale=-1.), dict(sac_alpha=0.),
                       dict(log_std_min=2., log_std_max=1.), dict(critic_num_blocks=-1)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DAWNModel(obs_dim=3, action_dim=2, horizon_steps=3,
                          pretrained_policy=TinyBC(), **kwargs)

    def test_resfit_checkpoint_keys_and_action_behavior_remain_compatible(self):
        model = ResFiTModel(obs_dim=3, action_dim=2, horizon_steps=3,
                            pretrained_policy=TinyBC(), actor_hidden_dims=[8],
                            critic_hidden_dims=[8], critic_ensemble_size=2)
        self.assertEqual({key.split('.')[0] for key in model.state_dict()},
                         {'base_policy', 'actor', 'critics', 'actor_target', 'critic_target'})
        base = torch.ones(2, 2) * .4
        torch.testing.assert_close(model.act(self.batch["obs"], base), base)
        model.train()
        self.assertFalse(model.base_policy.training)
        self.assertFalse(model.actor_target.training)
        self.assertFalse(model.critic_target.training)


if __name__ == "__main__":
    unittest.main()
