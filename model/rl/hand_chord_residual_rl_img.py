"""
Distilled residual RL model for image-based online finetuning through a frozen hand chord.

This module extends DistillResidualRLModel so that the residual is added in the packed
coordinates of a pretrained hand chord instead of the raw action space.
The state already contains visual features merged by the agent, with proprioception first.

- HandChordResidualActor: residual actor that outputs packed [B, chord.policy_dim] coordinates
- HandChordResidualRLImgModel: main model; critics, replay, exploration noise, and
  executed actions keep the raw [B, H, action_dim] interface of the parent

"""

import torch
import torch.nn as nn
from typing import List
import logging

log = logging.getLogger(__name__)

from model.common.mlp import MLP
from model.hand_chord.codec import load_hand_chord
from model.rl.distill_residual_rl import DistillResidualRLModel


class HandChordResidualActor(nn.Module):
    """
    One-step residual actor in packed chord coordinates.

    Takes state s and a conditioning input (raw noise z or packed base action), outputs a packed residual.
    """

    def __init__(
        self,
        obs_dim: int,
        input_action_dim: int,
        chord_dim: int,
        cond_steps: int = 1,
        hidden_dims: List[int] = [256, 256, 256],
        activation_type: str = "Mish",
        use_layernorm: bool = False,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.input_action_dim = input_action_dim
        self.chord_dim = chord_dim
        self.cond_steps = cond_steps

        # Input: flattened state + flattened conditioning
        # state: (B, cond_steps, obs_dim) -> flattened to (B, cond_steps * obs_dim)
        # conditioning: (B, horizon_steps, action_dim) noise or (B, chord_dim) base action -> (B, input_action_dim)
        input_dim = cond_steps * obs_dim + input_action_dim

        mlp_dims = [input_dim] + hidden_dims + [chord_dim]

        self.mlp = MLP(
            mlp_dims,
            activation_type=activation_type,
            out_activation_type="Identity",
            use_layernorm=use_layernorm,
        )

    def forward(self, state: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: (B, cond_steps, obs_dim) - chunked state
            conditioning: (B, horizon_steps, action_dim) noise or (B, chord_dim) packed base action

        Returns:
            residual: (B, chord_dim) - packed residual
        """
        B = conditioning.shape[0]  # batch size
        # Concatenate flattened inputs
        x = torch.cat([state.reshape(B, -1), conditioning.reshape(B, -1)], dim=-1)  # (B, cond_steps * obs_dim + input_action_dim)
        return self.mlp(x)  # (B, chord_dim)


class HandChordResidualRLImgModel(DistillResidualRLModel):
    """
    An image-based distilled RL model with residual actions parameterized in hand chord coordinates.

    This model works with augmented state dimensions where visual features
    are merged after the low-dim state by the agent.

    The main differences from DistillResidualRLImgModel:
    1. get_action adds the residual to the chord-encoded pretrained action and decodes the sum
    2. the actor is a HandChordResidualActor that outputs [B, chord.policy_dim]
    3. the chord is loaded from chord_path and kept frozen in eval mode with the pretrained policy
    """

    def __init__(
        self,
        obs_dim: int,  # This should be augmented (original_obs_dim + visual_feature_dim)
        action_dim: int,
        pretrained_flow_policy_path: str,
        chord_path: str,
        # Actor settings for packed chord coordinates
        actor_hidden_dims: List[int] = [256, 256, 256],
        activation_type: str = "Mish",
        use_layernorm: bool = True,
        cond_steps: int = 1,
        horizon_steps: int = 4,
        condition_residual_on_base_action: bool = False,
        device: str = "cuda",
        # All other parameters with defaults to match parent
        **kwargs
    ):
        """
        Initialize image-based hand chord residual RL model.

        Args:
            obs_dim: Augmented observation dimension; the first chord.state_dim entries are proprioception
            action_dim: Raw action dimension, must match the chord
            pretrained_flow_policy_path: Path to pretrained flow policy
            chord_path: Path to a pretrained hand chord checkpoint (chord.pt)
            actor_hidden_dims, activation_type, use_layernorm: Same meaning as in parent class
            **kwargs: All other parameters passed to parent class
        """

        # Keep modules local until the parent initializes nn.Module.
        chord = load_hand_chord(chord_path, device=device)
        chord.requires_grad_(False)

        if condition_residual_on_base_action: # condition on the packed base action, which is (B, chord_dim)
            input_action_dim = chord.policy_dim #  chord_dim is read from the codec metadata
        else: # condition on noise, which is (B, horizon_steps, action_dim)
            input_action_dim = horizon_steps * action_dim

        actor = HandChordResidualActor(
            obs_dim=obs_dim,
            input_action_dim=input_action_dim,
            chord_dim=chord.policy_dim,
            cond_steps=cond_steps,
            hidden_dims=actor_hidden_dims,
            activation_type=activation_type,
            use_layernorm=use_layernorm,
        )

        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            pretrained_flow_policy_path=pretrained_flow_policy_path,
            actor_hidden_dims=actor_hidden_dims,
            activation_type=activation_type,
            use_layernorm=use_layernorm,
            cond_steps=cond_steps,
            horizon_steps=horizon_steps,
            condition_residual_on_base_action=condition_residual_on_base_action,
            device=device,
            actor=actor,
            **kwargs
        )
        self.chord = chord

        log.info(f"HandChordResidualRLImgModel initialized with augmented obs_dim={obs_dim}")
        log.info(f"  chord: {chord_path} (latent_dim={self.chord.latent_dim}, chord_dim={self.chord.policy_dim})")

    def train(self, mode: bool = True):
        """Train actor and critics while the frozen pretrained policy and chord stay in eval mode."""
        super().train(mode)
        self.pretrained_flow_policy.eval()
        self.chord.eval()
        return self

    def get_action(self, state: torch.Tensor, noise: torch.Tensor, return_pretrained_actions: bool = False):
        """
        Get action by adding the residual to the pretrained action in latent residual space.

        Action = D(E(π_pre(s,z)) + r_θ(s,z)) or D(E(π_pre(s,z)) + r_θ(s,E(a_base)))
        where E/D are chord encode/decode, both conditioned on the measured hand pose.

        The pretrained actions returned are D(E(π_pre(s,z))),
        the chord projection of the pretrained action,
        so a zero residual reproduces them and the BC loss and Q comparison in actor_loss stay like for like.

        Args:
            state: (B, cond_steps, augmented_obs_dim) - proprioception first, then visual features
            noise: (B, horizon_steps, action_dim) - noise for action generation
            return_pretrained_actions: if True, return tuple (total_actions, pretrained_actions)

        Returns:
            if return_pretrained_actions:
                tuple: (total_actions, pretrained_actions) both (B, horizon_steps, action_dim)
            else:
                action: (B, horizon_steps, action_dim) - total action (decoded pretrained + residual)
        """
        proprioception = state[..., :self.chord.state_dim]  # (B, cond_steps, state_dim)

        # Get pretrained action and its packed code (no gradient)
        with torch.no_grad():
            # The state already contains merged visual features from the agent
            output = self.pretrained_flow_policy.forward_from_features(
                features=state,
                init_noise=noise
            )
            pretrained_actions = output.trajectories  # (B, horizon_steps, action_dim)
            pretrained_packed = self.chord.encode(pretrained_actions, proprioception)  # (B, chord_dim)

        # Get packed residual from actor - condition on base action or noise
        if self.condition_residual_on_base_action:
            # Use packed base action as input: r_θ(s, E(a_base))
            residual_packed = self.actor(state, pretrained_packed)  # (B, chord_dim)
        else:
            # Use noise as input: r_θ(s, z)
            residual_packed = self.actor(state, noise)  # (B, chord_dim)

        # Total action = decode(pretrained + residual)
        # Chord weights are frozen, but gradients flow through its input back to the actor
        total_actions = self.chord.decode(pretrained_packed + residual_packed, proprioception)  # (B, horizon_steps, action_dim)

        if return_pretrained_actions:
            with torch.no_grad():
                projected_actions = self.chord.decode(pretrained_packed, proprioception)  # (B, horizon_steps, action_dim)
            return total_actions, projected_actions
        else:
            return total_actions
