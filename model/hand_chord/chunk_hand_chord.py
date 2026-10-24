"""
Conditional VAE with one K-dimensional code for an entire hand chunk.

The packed interface is [B, H*arm_dim+K].
Leading columns contain the unchanged arm values in frame-major order;
the final K columns contain standardized global hand posterior means.
With the default DexJoCo layout and H=16, this means 96 arm values followed by K hand values.
Decoding restores the raw [B, H, raw_action_dim] action layout.
"""

from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ChunkHandActionChord(nn.Module):
    """Normalize, encode, and reconstruct a measured-q0-conditioned hand chunk."""

    normalization_epsilon = 1e-6

    def __init__(
        self,
        normalization_path=None,
        horizon_steps=16,
        latent_dim=8,
        width=128,
        layers=3,
        heads=4,
        dropout=0.0,
        joint_weights=None,
        reconstruction_weight=1.0,
        huber_delta=1.0,
        raw_action_dim=22,
        hand_dim=16,
        state_dim=23,
        hand_state_start=7,
    ):
        super().__init__()
        if normalization_path is None:
            raise ValueError("normalization_path is required")

        # Action and state dimensions
        self.raw_action_dim = int(raw_action_dim)
        self.hand_dim = int(hand_dim)
        self.state_dim = int(state_dim)
        self.hand_state_start = int(hand_state_start)
        self.arm_dim = self.raw_action_dim - self.hand_dim
        if (self.hand_dim <= 0 or self.arm_dim < 0 or self.hand_state_start < 0
                or self.hand_state_start + self.hand_dim > self.state_dim):
            raise ValueError("Hand action and state slices must fit the configured dimensions")
        self.hand_action_slice = slice(self.arm_dim, self.raw_action_dim)
        self.hand_state_slice = slice(self.hand_state_start, self.hand_state_start + self.hand_dim)

        # Model Hyperparameters
        self.horizon_steps = int(horizon_steps)
        self.latent_dim = int(latent_dim)

        # Loss Hyperparameters
        self.reconstruction_weight = float(reconstruction_weight)
        self.huber_delta = float(huber_delta)

        # Load normalization statistics from a DexJoCo NPZ file.
        with np.load(normalization_path, allow_pickle=False) as data:
            normalization = {
                name: data[name] for name in ("action_min", "action_max", "obs_min", "obs_max")
            }
            normalized = bool(data["normalized"].item()) if "normalized" in data else True

        # Register the normalization statistics as buffers.
        for name, values in normalization.items():
            self.register_buffer(name, torch.as_tensor(values, dtype=torch.float32))
        self.register_buffer("normalization_enabled", torch.tensor(normalized))
        self.register_buffer("relative_mean", torch.zeros(self.hand_dim))
        self.register_buffer("relative_std", torch.ones(self.hand_dim))
        self.register_buffer("latent_mean", torch.zeros(self.latent_dim))
        self.register_buffer("latent_std", torch.ones(self.latent_dim))

        # 기본 16관절 손은 약지에 0.5를 적용하고, 다른 차원은 동일 가중치를 쓴다.
        if joint_weights is None:
            joint_weights = torch.ones(self.hand_dim)
            if self.hand_dim == 16:
                joint_weights[8:12] = 0.5  # Right-hand order: index, middle, ring, thumb.
        joint_weights = torch.as_tensor(joint_weights, dtype=torch.float32)
        if joint_weights.shape != (self.hand_dim,):
            raise ValueError(f"joint_weights must have shape [{self.hand_dim}]")
        self.register_buffer("joint_weights", joint_weights.clone())

        self.width, self.layers, self.heads = int(width), int(layers), int(heads)
        self.dropout = float(dropout)
        self.policy_dim = self.horizon_steps * self.arm_dim + self.latent_dim

        # common
        self.q0_embedding = nn.Linear(self.hand_dim, self.width)

        # Encoder
        self.hand_embedding = nn.Linear(self.hand_dim, self.width)
        self.encoder_position = nn.Parameter(torch.zeros(1, self.horizon_steps + 1, self.width))
        self.encoder = self._build_transformer()
        self.posterior_mean = nn.Linear(self.width, self.latent_dim)
        self.posterior_logvar = nn.Linear(self.width, self.latent_dim)
        nn.init.normal_(self.encoder_position, std=0.02)

        # Decoder
        self.latent_embedding = nn.Linear(self.latent_dim, self.width)
        self.decoder_position = nn.Parameter(torch.zeros(1, self.horizon_steps + 1, self.width))
        self.decoder = self._build_transformer()
        self.hand_output = nn.Linear(self.width, self.hand_dim)
        nn.init.normal_(self.decoder_position, std=0.02)

    def _build_transformer(self):
        """Build a self-attention stack with independently initialized layers."""
        layer = nn.TransformerEncoderLayer(
            d_model=self.width, nhead=self.heads, dim_feedforward=2 * self.width,
            dropout=self.dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        transformer = nn.TransformerEncoder(
            layer, num_layers=self.layers, norm=nn.LayerNorm(self.width),
            enable_nested_tensor=False,
        )
        # 복제된 Transformer 층들이 같은 초기 가중치로 시작하지 않도록 재초기화한다.
        for parameter in transformer.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        return transformer

    def _build_transformer_input(self, embedded_frames, state, position):
        # 정규화된 측정 q0를 조건 토큰으로 앞에 붙이고 시간 위치를 더한다.
        q0_token = self.q0_embedding(state[:, self.hand_state_slice])[:, None]
        return torch.cat([q0_token, embedded_frames], dim=1) + position

    def _unnormalize(self, values, minimum, maximum):
        # 정규화된 입력만 라디안으로 복원하고 물리 좌표 입력은 그대로 둔다.
        physical = (values + 1) / 2 * (
            maximum - minimum + self.normalization_epsilon
        ) + minimum
        return torch.where(self.normalization_enabled, physical, values)

    def get_hand_targets(self, actions):
        """Extract absolute hand target angles in radians from an action chunk."""
        # action 열을 잘라내기 전에 고정된 청크 길이와 차원을 확인한다.
        if actions.ndim != 3 or actions.shape[1:] != (self.horizon_steps, self.raw_action_dim):
            raise ValueError(f"actions must have shape [B, {self.horizon_steps}, {self.raw_action_dim}]")
        hand = actions[..., self.hand_action_slice]
        return self._unnormalize(
            hand, self.action_min[self.hand_action_slice], self.action_max[self.hand_action_slice]
        )

    def get_initial_hand_positions(self, state):
        """Return measured chunk-start hand angles in radians from the latest state."""
        # 과거 입력까지 사용하면 state는 [B, T, state_dim] 형태의 3차원 텐서다.
        initial_hand_state = state[:, -1, :] if state.ndim == 3 else state
        hand = initial_hand_state[..., self.hand_state_slice]
        return self._unnormalize(
            hand, self.obs_min[self.hand_state_slice], self.obs_max[self.hand_state_slice]
        )

    def get_relative_hand_targets(self, actions, state):
        """Return hand target angles relative to the measured chunk-start pose, in radians."""
        # 명령과 관측의 서로 다른 정규화를 각각 되돌린 뒤 상대각을 구한다.
        targets = self.get_hand_targets(actions)
        initial_positions = self.get_initial_hand_positions(state)
        return targets - initial_positions[:, None]

    def normalize_hand(self, hand):
        """Convert physical hand commands back to the dataset's action coordinates."""
        normalized = 2 * (hand - self.action_min[self.hand_action_slice]) / (
            self.action_max[self.hand_action_slice] - self.action_min[self.hand_action_slice]
            + self.normalization_epsilon
        ) - 1
        return torch.where(self.normalization_enabled, normalized, hand)

    @torch.no_grad()
    def _set_statistics(self, mean_buffer, std_buffer, mean, std):
        mean = torch.as_tensor(mean, device=mean_buffer.device, dtype=mean_buffer.dtype)
        std = torch.as_tensor(std, device=std_buffer.device, dtype=std_buffer.dtype)
        # copy_의 브로드캐스팅으로 한 통계값이 모든 관절에 복사되는 것을 막는다.
        if mean.shape != mean_buffer.shape or std.shape != std_buffer.shape:
            raise ValueError(f"statistics must have shape {tuple(mean_buffer.shape)}")
        # 표준화 과정의 0 나눗셈과 NaN 전파를 막는다.
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("statistics must be finite with strictly positive standard deviations")
        mean_buffer.copy_(mean)
        std_buffer.copy_(std)

    def set_statistics(self, mean, std):
        """Set training-episode relative-hand statistics in radians."""
        self._set_statistics(self.relative_mean, self.relative_std, mean, std)

    def set_latent_statistics(self, mean, std):
        """Set posterior-mean statistics fitted over training latent values."""
        self._set_statistics(self.latent_mean, self.latent_std, mean, std)

    def posterior(self, actions, initial_hand_state):
        """
        Infer the chunk-level latent posterior from hand actions and the measured initial pose.
        args : actions: [B, H, raw_action_dim], initial_hand_state: [B, state_dim]
        Returns the posterior mean and log-variance, each with shape [B, K].
        """

        relative = (
            self.get_relative_hand_targets(actions, initial_hand_state) - self.relative_mean
        ) / self.relative_std  # [B, H, hand_dim]
        embedded_frames = self.hand_embedding(relative)  # [B, H, width]
        tokens = self._build_transformer_input(embedded_frames, initial_hand_state, self.encoder_position)  # [B, H+1, width]

        # The measured-q0 token attends to every hand frame and summarizes the chunk.
        chunk_embedding = self.encoder(tokens)[:, 0]  # [B, H+1, width] -> [B, width]
        mean = self.posterior_mean(chunk_embedding)  # [B, K]
        logvar = self.posterior_logvar(chunk_embedding).clamp(-10, 10)  # [B, K]
        return mean, logvar

    def encode(self, actions, state):
        """Pack frame-major arm values followed by K standardized global values."""
        # 과거 입력까지 사용하면 state는 [B, T, state_dim] 형태의 3차원 텐서다.
        initial_hand_state = state[:, -1, :] if state.ndim == 3 else state
        mean, _ = self.posterior(actions, initial_hand_state)
        latent = (mean - self.latent_mean) / self.latent_std
        arm = actions[..., :self.arm_dim].flatten(1)
        return torch.cat([arm, latent], dim=-1)

    def _decode_relative_hand_action(self, latent, state):
        """Reconstruct standardized relative hand actions from a global latent."""

        assert latent.ndim == 2
        assert latent.shape[-1] == self.latent_dim

        embedded_latent = self.latent_embedding(latent)  # [B, K] -> [B, width]
        frames = embedded_latent.unsqueeze(1)  # [B, 1, width]
        frames = frames.expand(-1, self.horizon_steps, -1)  # [B, H, width]
        tokens = self._build_transformer_input(frames, state, self.decoder_position)  # [B, H+1, width]

        decoded_tokens = self.decoder(tokens)  # [B, H+1, width]
        decoded_frames = decoded_tokens[:, 1:]  # [B, H, width]
        relative = self.hand_output(decoded_frames)  # [B, H, hand_dim]
        return relative

    def _decode_hand(self, latent, state):
        # 상대각의 표준화를 되돌리고 측정 q0를 더해 손 명령을 복원한다.
        relative = self._decode_relative_hand_action(latent, state) * self.relative_std + self.relative_mean
        return self.normalize_hand(relative + self.get_initial_hand_positions(state)[:, None])

    def decode(self, packed, state):
        """Decode [B, H*arm_dim+K] into raw actions, preserving all arm values."""
        if packed.ndim != 2 or packed.shape[-1] != self.policy_dim:
            raise ValueError(f"packed must have shape [B, {self.policy_dim}]")
        # 과거 입력까지 사용하면 state는 [B, T, state_dim] 형태의 3차원 텐서다.
        initial_hand_state = state[:, -1, :] if state.ndim == 3 else state
        if initial_hand_state.shape[0] != packed.shape[0]:
            raise ValueError("actions and state must have the same batch size")
        arm, latent = packed.split([self.horizon_steps * self.arm_dim, self.latent_dim], dim=-1)
        arm = arm.reshape(packed.shape[0], self.horizon_steps, self.arm_dim)
        latent = latent * self.latent_std + self.latent_mean
        return torch.cat([arm, self._decode_hand(latent, initial_hand_state)], dim=-1)

    def reconstruct(self, actions, state):
        """Return deterministic raw-action reconstructions and posterior moments."""
        # 과거 입력까지 사용하면 state는 [B, T, state_dim] 형태의 3차원 텐서다.
        initial_hand_state = state[:, -1, :] if state.ndim == 3 else state
        mean, logvar = self.posterior(actions, initial_hand_state)
        reconstructed = torch.cat([actions[..., :self.arm_dim], self._decode_hand(mean, initial_hand_state)], dim=-1)
        return reconstructed, mean, logvar

    def loss(self, actions, state, beta=1e-4):
        """
        Combine the chord's reconstruction objective with posterior KL.

        Reconstruction uses standardized relative hand coordinates.
        Evaluation uses posterior means; training samples the posterior.
        """

        initial_hand_state = state[:, -1, :] if state.ndim == 3 else state # 과거 입력까지 사용하면 state는 [B, T, state_dim] 형태의 3차원 텐서다.
        mean, logvar = self.posterior(actions, initial_hand_state)

        # 학습에서는 posterior를 샘플링하고 평가에서는 평균을 사용한다.
        latent = mean
        if self.training:
            latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        prediction = self._decode_relative_hand_action(latent, initial_hand_state)
        target = (
            self.get_relative_hand_targets(actions, initial_hand_state) - self.relative_mean
        ) / self.relative_std

        error = F.huber_loss(
            prediction, target, reduction="none", delta=self.huber_delta
        )
        # 관절별 가중치를 적용한 Huber 오차를 배치·시간·관절 전체에서 평균낸다.
        reconstruction = (error * self.joint_weights).mean()

        # posterior와 표준 정규분포 사이의 KL을 재구성 손실에 더한다.
        kl = -0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(dim=-1).mean()
        return {
            "loss": self.reconstruction_weight * reconstruction + beta * kl,
            "reconstruction": reconstruction,
            "kl": kl,
        }

    def _configuration(self):
        # 체크포인트에서 같은 모델을 재생성할 수 있도록 생성자 설정을 저장한다.
        return {
            "raw_action_dim": self.raw_action_dim, "hand_dim": self.hand_dim,
            "state_dim": self.state_dim, "hand_state_start": self.hand_state_start,
            "horizon_steps": self.horizon_steps, "latent_dim": self.latent_dim,
            "joint_weights": self.joint_weights.detach().cpu().tolist(),
            "reconstruction_weight": self.reconstruction_weight,
            "huber_delta": self.huber_delta,
            "width": self.width, "layers": self.layers,
            "heads": self.heads, "dropout": self.dropout,
        }

    def metadata(self):
        """Return hand-coordinate, model, and packed-action metadata."""
        return {
            "format_version": 1, "config": self._configuration(),
            "raw_action_dim": self.raw_action_dim,
            "hand_dim": self.hand_dim,
            "action_slice": [self.hand_action_slice.start, self.hand_action_slice.stop],
            "state_slice": [self.hand_state_slice.start, self.hand_state_slice.stop],
            "joint_names": [
                f"{finger}_{joint}"
                for finger in ("index", "middle", "ring", "thumb")
                for joint in range(4)
            ] if self.hand_dim == 16 else [f"joint_{joint}" for joint in range(self.hand_dim)],
            "physical_units": "radians", "relative_target": "q_command_minus_measured_q0",
            "normalization_enabled": bool(self.normalization_enabled.item()),
            "normalization_epsilon": self.normalization_epsilon,
            "policy_dim": self.policy_dim,
            "architecture": type(self).__name__,
            "latent_layout": "B,K",
            "q0_token": "stored_measured_hand_state",
            "latent_scalars_per_chunk": self.latent_dim,
            "packed_layout": f"B,H*{self.arm_dim}+K",
            "packed_arm_order": "frame_major",
            "packed_arm_slice": [0, self.horizon_steps * self.arm_dim],
            "packed_hand_slice": [self.horizon_steps * self.arm_dim, self.policy_dim],
            "conditioning": "measured_chunk_start_q0",
            "encoder_summary": "q0_token_attending_to_all_hand_frames",
        }

    def save(self, path):
        """Save weights, physical normalizers, and training statistics together."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 가중치와 정규화 통계를 함께 저장해 원본 NPZ 없이도 추론할 수 있게 한다.
        torch.save({
            "format_version": 1, "config": self._configuration(),
            "state_dict": self.state_dict(), "metadata": self.metadata(),
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        """Load a self-contained global chunk checkpoint for inference."""
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        metadata = checkpoint.get("metadata", {})
        if (
            checkpoint.get("format_version") != 1
            or metadata.get("latent_layout") != "B,K"
        ):
            raise ValueError("unsupported global chunk hand chord checkpoint")
        # Earlier chunk checkpoints recorded the unused frame-loss weights.
        config = checkpoint["config"].copy()
        config.pop("velocity_weight", None)
        config.pop("acceleration_weight", None)
        state_dict = checkpoint["state_dict"]
        # 체크포인트에 포함된 통계로 NPZ를 구성해 원본 파일 없이 복원한다.
        with BytesIO() as normalization:
            np.savez(
                normalization,
                **{name: state_dict[name].cpu().numpy()
                   for name in ("action_min", "action_max", "obs_min", "obs_max")},
                normalized=bool(state_dict["normalization_enabled"].item()),
            )
            normalization.seek(0)
            model = cls(normalization_path=normalization, **config).to(device)
        model.load_state_dict(state_dict)
        return model.eval()
