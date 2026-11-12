"""DAWN residual SAC over a frozen native BC policy.

Actor and normalized critic follow Guozheng Ma et al., "What Makes Value
Learning Efficient in Residual Reinforcement Learning?" (DAWN, ICML 2026),
specifically nets/dawn.py and online_dawn/pi_dec_diffusion_maniskill2_dawn.py
in the DAWN implementation. Native image features and per-control-step BC
queues adapt that state/chunk-based implementation to this repository.
"""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from model.rl.resfit import FrozenBCPolicyModel, frozen


class ResidualActor(nn.Module):
    """Gaussian raw residual in [-1, 1], before the residual scale is applied."""

    def __init__(self, input_dim, action_dim, hidden_dim=256,
                 log_std_min=-20., log_std_max=2.):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)
        for layer in (self.fc_mean, self.fc_logstd):
            nn.init.orthogonal_(layer.weight, .01)
            nn.init.zeros_(layer.bias)

    def forward(self, features):
        hidden = self.backbone(features)
        mean = self.fc_mean(hidden)
        log_std = self.fc_logstd(hidden).tanh()
        log_std = self.log_std_min + .5 * (self.log_std_max - self.log_std_min) * (log_std + 1.)
        return mean, log_std

    def get_eval_action(self, features):
        mean, _ = self(features)
        return mean.tanh()

    def get_action(self, features):
        mean, log_std = self(features)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        latent = distribution.rsample()
        residual = latent.tanh()
        log_prob = distribution.log_prob(latent) - torch.log(1. - residual.square() + 1e-6)
        return residual, log_prob.sum(-1, keepdim=True), mean.tanh()


class LayerNormResidualBlock(nn.Module):
    def __init__(self, hidden_dim=512, expansion=4):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * expansion)
        self.linear2 = nn.Linear(hidden_dim * expansion, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, features):
        hidden = self.linear2(F.relu(self.linear1(features)))
        return self.norm(features + hidden)


class LayerNormCritic(nn.Module):
    def __init__(self, n_obs, n_act, hidden_dim=512, num_blocks=2, expansion=4):
        super().__init__()
        self.embedder = nn.Sequential(nn.Linear(n_obs + n_act, hidden_dim), nn.LayerNorm(hidden_dim))
        self.blocks = nn.Sequential(*[
            LayerNormResidualBlock(hidden_dim, expansion) for _ in range(num_blocks)
        ])
        self.head_l1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.head_l2 = nn.Linear(hidden_dim, 1)

    def forward(self, features, actions):
        hidden = self.embedder(torch.cat((features, actions), dim=-1))
        return self.head_l2(self.head_l1(self.blocks(hidden)))


class TwinQ(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim, num_blocks, expansion):
        super().__init__()
        self.heads = nn.ModuleList([
            LayerNormCritic(feature_dim, action_dim, hidden_dim, num_blocks, expansion)
            for _ in range(2)
        ])

    def forward(self, features, actions):
        return torch.stack([head(features, actions).squeeze(-1) for head in self.heads])


class DAWNModel(FrozenBCPolicyModel):
    def __init__(
        self, obs_dim, action_dim, horizon_steps, cond_steps=1,
        pretrained_policy_path=None, pretrained_config_path=None,
        base_policy_weights="model", pretrained_policy=None, visual_feature_dim=None,
        residual_scale=.1, actor_hidden_dim=256, critic_hidden_dim=512,
        critic_num_blocks=2, critic_expansion=4, log_std_min=-20., log_std_max=2.,
        sac_alpha=.01, autotune=True, target_entropy=None, device="cpu",
    ):
        if not math.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError("residual_scale must be finite and nonnegative")
        if not math.isfinite(sac_alpha) or sac_alpha <= 0:
            raise ValueError("sac_alpha must be finite and positive")
        if not math.isfinite(log_std_min) or not math.isfinite(log_std_max) or log_std_min >= log_std_max:
            raise ValueError("log_std_min must be finite and smaller than log_std_max")
        for name, value, minimum in (("actor_hidden_dim", actor_hidden_dim, 1),
                                     ("critic_hidden_dim", critic_hidden_dim, 1),
                                     ("critic_num_blocks", critic_num_blocks, 0),
                                     ("critic_expansion", critic_expansion, 1)):
            if int(value) != value or value < minimum:
                raise ValueError("{} must be an integer >= {}".format(name, minimum))
        super().__init__(
            obs_dim, action_dim, horizon_steps, cond_steps,
            pretrained_policy_path, pretrained_config_path, base_policy_weights,
            pretrained_policy, visual_feature_dim, device,
        )
        self.residual_scale = float(residual_scale)
        self.autotune = bool(autotune)
        self.target_entropy = -float(action_dim) if target_entropy is None else float(target_entropy)
        if not math.isfinite(self.target_entropy):
            raise ValueError("target_entropy must be finite")
        self.actor = ResidualActor(
            self.feature_dim, self.action_dim, actor_hidden_dim, log_std_min, log_std_max,
        )
        self.critics = TwinQ(
            self.feature_dim, self.action_dim, critic_hidden_dim, critic_num_blocks, critic_expansion,
        )
        self.critic_target = copy.deepcopy(self.critics).requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(sac_alpha)), requires_grad=self.autotune)
        self.to(device)
        self.train()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self, mode=True):
        super().train(mode)
        self.critic_target.eval()
        return self

    def sample_action(self, features, base_action):
        residual, log_prob, _ = self.actor.get_action(features)
        # DAWN regularizes raw residual entropy; residual_scale adds no Jacobian.
        return base_action + self.residual_scale * residual, log_prob

    @torch.no_grad()
    def act(self, features, base_action, sample=False):
        """Return the raw sum; the trainer clips only actions sent to the env."""
        if sample:
            return self.sample_action(features, base_action)[0]
        return base_action + self.residual_scale * self.actor.get_eval_action(features)

    def critic_loss(self, batch, include_entropy=True):
        with torch.no_grad():
            next_action, log_prob = self.sample_action(batch["next_obs"], batch["next_base_action"])
            next_q = self.critic_target(batch["next_obs"], next_action).min(dim=0).values
            if include_entropy:
                next_q = next_q - self.alpha * log_prob.reshape(-1)
            target = batch["reward"].reshape(-1) + batch["discount"].reshape(-1) * next_q
        values = self.critics(batch["obs"].detach(), batch["action"].detach())
        return F.mse_loss(values[0], target) + F.mse_loss(values[1], target)

    def actor_loss(self, batch):
        features = batch["obs"].detach()
        action, log_prob = self.sample_action(features, batch["base_action"].detach())
        with frozen(self.critics):
            q = self.critics(features, action).min(dim=0).values
        return (self.alpha.detach() * log_prob.reshape(-1) - q).mean()

    def temperature_loss(self, batch):
        with torch.no_grad():
            _, log_prob = self.sample_action(batch["obs"], batch["base_action"])
        return (-self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

    @torch.no_grad()
    def update_targets(self, tau):
        if not 0. <= tau <= 1.:
            raise ValueError("tau must be in [0, 1]")
        for online, target in zip(self.critics.parameters(), self.critic_target.parameters()):
            target.lerp_(online, tau)
