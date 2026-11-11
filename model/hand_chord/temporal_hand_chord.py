"""Measured-q0-conditioned hand VAE with one K-dimensional code per frame."""

from pathlib import Path

import torch

from model.hand_chord.chunk_hand_chord import ChunkHandActionChord


class TemporalHandActionChord(ChunkHandActionChord):
    """Pack unchanged arm frames followed by frame-major temporal hand codes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_dim = self.horizon_steps * (self.arm_dim + self.latent_dim)

    def posterior(self, actions, initial_hand_state):
        """Return posterior mean and log-variance with shape [B, H, K]."""
        initial_hand_state = initial_hand_state[:, -1, :] if initial_hand_state.ndim == 3 else initial_hand_state
        relative = (
            self.get_relative_hand_targets(actions, initial_hand_state) - self.relative_mean
        ) / self.relative_std
        tokens = self._build_transformer_input(
            self.hand_embedding(relative), initial_hand_state, self.encoder_position
        )
        # Preserve each action token after attention to the measured-q0 token.
        frames = self.encoder(tokens)[:, 1:]
        return self.posterior_mean(frames), self.posterior_logvar(frames).clamp(-10, 10)

    def encode(self, actions, state):
        """Return [B, H*arm_dim+H*K] with both blocks in frame-major order."""
        mean, _ = self.posterior(actions, state)
        latent = (mean - self.latent_mean) / self.latent_std
        arm = actions[..., :self.arm_dim].flatten(1)
        return torch.cat([arm, latent.flatten(1)], dim=-1)

    def _decode_relative_hand_action(self, latent, state):
        """Decode each frame's latent without repeating a global chunk code."""
        if latent.ndim != 3 or latent.shape[1:] != (self.horizon_steps, self.latent_dim):
            raise ValueError(f"latent must have shape [B, {self.horizon_steps}, {self.latent_dim}]")
        tokens = self._build_transformer_input(
            self.latent_embedding(latent), state, self.decoder_position
        )
        return self.hand_output(self.decoder(tokens)[:, 1:])

    def decode(self, packed, state):
        """Restore [B, H, raw_action_dim], preserving the packed arm exactly."""
        if packed.ndim != 2 or packed.shape[-1] != self.policy_dim:
            raise ValueError(f"packed must have shape [B, {self.policy_dim}]")
        initial_hand_state = state[:, -1, :] if state.ndim == 3 else state
        if initial_hand_state.shape[0] != packed.shape[0]:
            raise ValueError("actions and state must have the same batch size")
        arm, latent = packed.split(
            [self.horizon_steps * self.arm_dim, self.horizon_steps * self.latent_dim], dim=-1
        )
        arm = arm.reshape(packed.shape[0], self.horizon_steps, self.arm_dim)
        latent = latent.reshape(packed.shape[0], self.horizon_steps, self.latent_dim)
        latent = latent * self.latent_std + self.latent_mean
        return torch.cat([arm, self._decode_hand(latent, initial_hand_state)], dim=-1)

    def metadata(self):
        """Describe the temporal checkpoint and its packed-action contract."""
        metadata = super().metadata()
        metadata.update({
            "format_version": 2,
            "latent_layout": "B,H,K",
            "latent_scalars_per_chunk": self.horizon_steps * self.latent_dim,
            "packed_layout": f"B,H*{self.arm_dim}+H*K",
            "packed_hand_order": "frame_major",
            "encoder_summary": "action_tokens_attending_to_measured_q0_and_all_hand_frames",
        })
        return metadata

    def save(self, path):
        """Save a self-contained temporal checkpoint and training statistics."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": 2, "config": self._configuration(),
            "state_dict": self.state_dict(), "metadata": self.metadata(),
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        """Load a temporal checkpoint without the original normalization NPZ."""
        from model.hand_chord.codec import load_hand_chord

        model = load_hand_chord(path, device=device)
        if not isinstance(model, cls):
            raise ValueError("unsupported temporal hand chord checkpoint")
        return model
