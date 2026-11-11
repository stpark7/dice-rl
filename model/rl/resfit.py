"""Per-step off-policy residual finetuning of a frozen native BC policy.

This DexJoCo adaptation reuses the frozen BC visual encoder. ResFiT's residual
actor and LayerNorm Q ensemble operate on those features at every control step;
the caller owns the base policy's action-chunk queues and n-step replay.
"""

import copy
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch
from torch import nn
from torch.nn import functional as F


def load_base_policy(checkpoint_path, device, config_path=None, weights="model"):
    """Load a native BC checkpoint and its Hydra model definition."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if config_path is None:
        config_path = next(
            (parent / ".hydra" / "config.yaml"
             for parent in checkpoint_path.parents
             if (parent / ".hydra" / "config.yaml").is_file()),
            None,
        )
    if config_path is None:
        raise FileNotFoundError(
            "Could not find .hydra/config.yaml for {}; set pretrained_config_path"
            .format(checkpoint_path)
        )
    cfg = OmegaConf.load(config_path)
    cfg.device = str(device)
    if "device" in cfg.model:
        cfg.model.device = str(device)
    policy = hydra.utils.instantiate(cfg.model).to(device)
    # Native training checkpoints also contain NumPy RNG state.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if weights in checkpoint:
        state = checkpoint[weights]
    elif weights == "model" and all(torch.is_tensor(value) for value in checkpoint.values()):
        state = checkpoint
    else:
        raise KeyError("Checkpoint {} has no {!r} weights".format(checkpoint_path, weights))
    policy.load_state_dict(state, strict=True)
    return policy


def _mlp(input_dim, hidden_dims, output_dim):
    layers = []
    for width in hidden_dims:
        layers.extend((nn.Linear(input_dim, width), nn.LayerNorm(width), nn.ReLU()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class ResidualActor(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dims, residual_scale):
        super().__init__()
        self.network = _mlp(feature_dim + action_dim, hidden_dims, action_dim)
        self.residual_scale = residual_scale
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features, base_actions):
        return self.residual_scale * torch.tanh(
            self.network(torch.cat((features, base_actions), dim=-1))
        )


class QEnsemble(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dims, ensemble_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [_mlp(feature_dim + action_dim, hidden_dims, 1) for _ in range(ensemble_size)]
        )

    def forward(self, features, full_actions):
        inputs = torch.cat((features, full_actions), dim=-1)
        return torch.stack([head(inputs).squeeze(-1) for head in self.heads], dim=0)


class ResFiTModel(nn.Module):
    def __init__(
        self, obs_dim, action_dim, horizon_steps, cond_steps=1,
        pretrained_policy_path=None, pretrained_config_path=None,
        base_policy_weights="model", pretrained_policy=None, visual_feature_dim=None,
        actor_hidden_dims=(1024, 1024), critic_hidden_dims=(1024, 1024),
        critic_ensemble_size=10, target_subset_size=2, residual_scale=0.2,
        exploration_std=0.05, target_noise_clip=0.3, device="cpu",
    ):
        super().__init__()
        if not 1 <= target_subset_size <= critic_ensemble_size:
            raise ValueError("target_subset_size must be between 1 and critic_ensemble_size")
        if residual_scale < 0 or exploration_std < 0 or target_noise_clip < 0:
            raise ValueError("Residual scale and noise parameters must be nonnegative")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.horizon_steps = int(horizon_steps)
        self.cond_steps = int(cond_steps)
        self.target_subset_size = int(target_subset_size)
        self.residual_scale = float(residual_scale)
        self.exploration_std = float(exploration_std)
        self.target_noise_clip = float(target_noise_clip)
        if pretrained_policy is None:
            if pretrained_policy_path is None:
                raise ValueError("pretrained_policy_path or pretrained_policy is required")
            pretrained_policy = load_base_policy(
                pretrained_policy_path, device, pretrained_config_path, base_policy_weights
            )
        self.base_policy = pretrained_policy.to(device).requires_grad_(False).eval()
        for name, expected in (("action_dim", self.action_dim),
                               ("horizon_steps", self.horizon_steps),
                               ("cond_steps", self.cond_steps)):
            actual = getattr(self.base_policy, name, expected)
            if actual != expected:
                raise ValueError("BC {}={} differs from configured {}".format(name, actual, expected))
        network = getattr(self.base_policy, "network", self.base_policy)
        if visual_feature_dim is None:
            visual_feature_dim = getattr(network, "visual_feature_dim", 0)
        self.visual_feature_dim = int(visual_feature_dim)
        self.feature_dim = self.cond_steps * self.obs_dim + self.visual_feature_dim
        self.actor = ResidualActor(
            self.feature_dim, self.action_dim, actor_hidden_dims, self.residual_scale
        )
        self.critics = QEnsemble(
            self.feature_dim, self.action_dim, critic_hidden_dims, critic_ensemble_size
        )
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic_target = copy.deepcopy(self.critics).requires_grad_(False)
        self.to(device)
        self.train()

    @property
    def device(self):
        return next(self.actor.parameters()).device

    def train(self, mode=True):
        super().train(mode)
        self.base_policy.eval()
        self.actor_target.eval()
        self.critic_target.eval()
        return self

    def _condition(self, obs):
        state = torch.as_tensor(obs["state"], device=self.device, dtype=torch.float32)
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.ndim != 3 or tuple(state.shape[1:]) != (self.cond_steps, self.obs_dim):
            raise ValueError("Expected state shape (B, {}, {})".format(self.cond_steps, self.obs_dim))
        cond = {"state": state}
        if "rgb" in obs:
            # Keep native pixel scale: the pretrained encoder divides by 255.
            rgb = torch.as_tensor(obs["rgb"], device=self.device, dtype=torch.float32)
            if rgb.ndim == 4:
                rgb = rgb.unsqueeze(1)
            if rgb.ndim != 5:
                raise ValueError("Expected batched RGB with optional time dimension")
            network = getattr(self.base_policy, "network", self.base_policy)
            channels = 3 * getattr(network, "num_img", 1)
            if rgb.shape[2] != channels and rgb.shape[-1] == channels:
                rgb = rgb.permute(0, 1, 4, 2, 3)
            if rgb.shape[2] != channels or rgb.shape[0] != state.shape[0]:
                raise ValueError("RGB batch/channel dimensions do not match BC observations")
            cond["rgb"] = rgb.contiguous()
        return cond

    @torch.no_grad()
    def encode(self, obs):
        """Return flattened state history followed by frozen BC visual features."""
        cond = self._condition(obs)
        features = cond["state"].flatten(1)
        if self.visual_feature_dim:
            network = getattr(self.base_policy, "network", self.base_policy)
            visual = network.extract_visual_features(cond)
            if tuple(visual.shape) != (features.shape[0], self.visual_feature_dim):
                raise ValueError("BC visual features differ from visual_feature_dim")
            features = torch.cat((features, visual), dim=-1)
        return features.detach()

    @torch.no_grad()
    def base_actions(self, obs):
        """Predict a complete BC chunk; the trainer manages its per-step queue."""
        output = self.base_policy(cond=self._condition(obs), deterministic=True)
        actions = output.trajectories
        if tuple(actions.shape[1:]) != (self.horizon_steps, self.action_dim):
            raise ValueError("BC returned an incompatible action chunk")
        return actions.detach()

    @torch.no_grad()
    def act(self, features, base_action, std=0.0, warmup=False, warmup_noise=None):
        if warmup:
            scale = self.residual_scale if warmup_noise is None else float(warmup_noise)
            residual = torch.empty_like(base_action).uniform_(-scale, scale)
        else:
            residual = self.actor(features, base_action)
            if std:
                residual = residual + torch.randn_like(residual) * std
        return (base_action + residual).clamp(-1.0, 1.0)

    def critic_loss(self, batch):
        with torch.no_grad():
            residual = self.actor_target(batch["next_obs"], batch["next_base_action"])
            noise = (torch.randn_like(residual) * self.exploration_std).clamp(
                -self.target_noise_clip, self.target_noise_clip
            )
            next_action = (batch["next_base_action"] + residual + noise).clamp(-1., 1.)
            values = self.critic_target(batch["next_obs"], next_action)
            indices = torch.randperm(values.shape[0], device=values.device)[:self.target_subset_size]
            next_q = values[indices].min(dim=0).values
            target = batch["reward"].reshape(-1) + batch["discount"].reshape(-1) * next_q
        values = self.critics(batch["obs"].detach(), batch["action"].detach())
        return F.mse_loss(values, target.unsqueeze(0).expand_as(values))

    def actor_loss(self, batch):
        features = batch["obs"].detach()
        base = batch["base_action"].detach()
        action = (base + self.actor(features, base)).clamp(-1., 1.)
        # Preserve gradients through Q's action input without accumulating Q gradients.
        parameters = list(self.critics.parameters())
        flags = [parameter.requires_grad for parameter in parameters]
        try:
            for parameter in parameters:
                parameter.requires_grad_(False)
            return -self.critics(features, action).mean()
        finally:
            for parameter, flag in zip(parameters, flags):
                parameter.requires_grad_(flag)

    @torch.no_grad()
    def update_targets(self, tau, update_actor=True):
        if not 0. <= tau <= 1.:
            raise ValueError("tau must be in [0, 1]")
        pairs = [(self.critics, self.critic_target)]
        if update_actor:
            pairs.append((self.actor, self.actor_target))
        for online, target in pairs:
            for parameter, target_parameter in zip(online.parameters(), target.parameters()):
                target_parameter.lerp_(parameter, tau)
