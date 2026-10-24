"""Statistics and held-out physical metrics shared by hand chord trainers."""

import math

import torch
from torch.utils.data import DataLoader


def kl_weight(step, maximum=3e-4, zero_steps=4000, ramp_steps=4000):
    if step < zero_steps:
        return 0.0
    if ramp_steps == 0:
        return maximum
    return maximum * min(1.0, (step - zero_steps) / ramp_steps)


@torch.no_grad()
def fit_statistics(chord, dataset, batch_size, device, latent=False):
    """Fit streaming moments over the supplied training windows in eval mode."""
    was_training = chord.training
    chord.eval()
    count, total, squares = 0, None, None
    try:
        for actions, state in DataLoader(dataset, batch_size=batch_size):
            actions, state = actions.to(device), state.to(device)
            values = chord.posterior(actions, state)[0] if latent else chord.get_relative_hand_targets(actions, state)
            values = values.detach().reshape(-1, values.shape[-1]).double().cpu()
            if total is None:
                total, squares = values.sum(0), values.square().sum(0)
            else:
                total += values.sum(0)
                squares += values.square().sum(0)
            count += len(values)
    finally:
        chord.train(was_training)
    if not count:
        raise ValueError("Cannot fit chord statistics on an empty dataset")
    mean = total / count
    std = (squares / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-4)
    return mean.float(), std.float()


@torch.no_grad()
def evaluate(chord, dataset, batch_size, device):
    """Report mean absolute hand reconstruction error in degrees."""
    chord.eval()
    absolute = torch.zeros((), dtype=torch.float64, device=device)
    count = 0
    # Iterator creation consumes a seed even without workers or shuffling.
    loader = DataLoader(dataset, batch_size=batch_size, generator=torch.Generator())
    for actions, state in loader:
        actions, state = actions.to(device), state.to(device)
        predicted, _, _ = chord.reconstruct(actions, state)
        reconstructed, target = chord.get_hand_targets(predicted), chord.get_hand_targets(actions)
        error = (reconstructed - target).double()
        absolute += error.abs().sum()
        count += error.numel()
    if not count:
        raise ValueError("Cannot evaluate a chord on an empty dataset")
    return {"mae_deg": float(absolute / count) * (180 / math.pi)}
