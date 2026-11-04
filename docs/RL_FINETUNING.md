# Replay buffer memory and transfers

Distill residual and hand-chord finetuning use `HybridReplayBuffer`. Replay
chunks, ongoing episodes, and preprocessed expert features are stored on CPU
by default. The model and sampled training batches stay on `cfg.device`.
Batch size, buffer capacity, n-step/MC returns, and the expert count in each
minibatch retain their configured values.

The pick_bucket configs expose:

```yaml
replay_buffer:
  storage_device: cpu
  pin_memory: true
```

Other configs use the same defaults. Add these keys with Hydra's `+` syntax
when overriding a config that does not declare them. Set `storage_device` to
the model's CUDA device to use GPU storage again, or set `pin_memory: false`
to disable pinned staging.

The training loop samples `train.gradient_steps` minibatches as one update
group. CPU fields are packed by dtype and transferred with non-blocking copies
from pinned staging. Each gradient step then consumes a slice on the GPU, so
there are no host-to-device copies between steps in that group. Random noise
is generated on the model device. Only sampled data uses pinned memory; the
full replay storage remains pageable. Temporary GPU and pinned memory scale
with batch size and gradient steps, rather than replay capacity.

Feature extraction still runs on the GPU. Completed feature batches are
transferred to the storage device before splitting into individual observations
or expert transitions, avoiding per-observation device copies.

These defaults apply when starting a new training process. Grouped sampling
preserves the sampling distribution and per-minibatch expert rounding, but
changes random-number draw order, so fixed seeds do not reproduce old runs
bit for bit. Actual training throughput depends on CPU, PCIe, and GPU load.

Run focused regression checks with:

```bash
python -m unittest discover -s tests -p test_hybrid_replay_buffer.py -v
```

CUDA tests exercise typed transfers and repeated backward passes when a GPU
is available; otherwise they are skipped.
