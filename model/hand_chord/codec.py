"""Load global and temporal hand codecs through their checkpoint metadata."""

from io import BytesIO

import numpy as np
import torch

from model.hand_chord.chunk_hand_chord import ChunkHandActionChord
from model.hand_chord.temporal_hand_chord import TemporalHandActionChord


def load_hand_chord(path, device="cpu"):
    """Restore a supported codec, including its embedded physical normalizers."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    layout = checkpoint.get("metadata", {}).get("latent_layout")
    codec_type = {
        (1, "B,K"): ChunkHandActionChord,
        (2, "B,H,K"): TemporalHandActionChord,
    }.get((checkpoint.get("format_version"), layout))
    if codec_type is None:
        raise ValueError("unsupported hand chord checkpoint version or latent layout")

    config = checkpoint["config"].copy()
    # Historical global codecs included these unused derivative-loss weights.
    config.pop("velocity_weight", None)
    config.pop("acceleration_weight", None)
    state_dict = checkpoint["state_dict"]
    with BytesIO() as normalization:
        np.savez(
            normalization,
            **{name: state_dict[name].cpu().numpy()
               for name in ("action_min", "action_max", "obs_min", "obs_max")},
            normalized=bool(state_dict["normalization_enabled"].item()),
        )
        normalization.seek(0)
        model = codec_type(normalization_path=normalization, **config).to(device)
    model.load_state_dict(state_dict)
    return model.eval()
