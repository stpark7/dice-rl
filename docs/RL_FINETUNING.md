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

# HandChord replay warmup and critic pretraining

The six DexJoCo `ft_hand_chord_residual_flow_unet_img` configs collect at least
1,000 valid online action chunks with the frozen, zero-residual policy before
learning. This policy is `chord.decode(chord.encode(BC action, state), state)`.
Collection bypasses the residual actor and Q-based candidate selection; the
BC policy still receives its usual random generation noise. No additional
exploration noise is added.

Two independent settings control the learning data and initialization:

| Setting | Behavior |
| --- | --- |
| `use_rlpd=true` | Mix expert demonstrations into critic pretraining and subsequent RL batches. |
| `use_rlpd=false` | Use rollout data throughout RL; skip demo loading, feature preprocessing, episode-count checks, and expert trajectory visualization. BC/chord checkpoints and normalization are still required. |
| `train.critic_pretrain_steps=0` | Implicit warmup: fill replay, then start joint actor/critic training. |
| `train.critic_pretrain_steps=N` | Explicit pretraining: after filling replay, perform exactly N critic-only gradient updates before joint training. |

Run the default demo-assisted implicit warmup on `hammer_nail` with:

```bash
python script/run.py \
  --config-name=ft_hand_chord_residual_flow_unet_img \
  --config-dir=cfg/dexjoco/finetune/hammer_nail/ \
  seed=42
```

Append these overrides to select the other comparisons:

```bash
# Demo-assisted explicit pretraining
use_rlpd=true train.critic_pretrain_steps=10000

# Base-rollout-only implicit warmup
use_rlpd=false train.critic_pretrain_steps=0

# Base-rollout-only explicit pretraining
use_rlpd=false train.critic_pretrain_steps=10000
```

`train.buffer_warmup_transitions` counts stored **online action chunks**, not
individual simulator steps, rollout calls, or expert rows. Replay stores chunks
only after complete episodes; collection therefore stops at an episode-driven
threshold and can exceed the requested count. At least `train.batch_size`
online chunks are required. Warmup collection uses the same rollout budget
as subsequent training (`train.num_train_steps`). If that budget ends before
the threshold is reached, the trainer warns that learning never started.

During explicit pretraining, `train.critic_pretrain_expert_ratio` defaults to
0.5 and is ignored when demos are disabled. Subsequent joint learning uses the
existing expert-ratio schedule. The critic bootstraps from the projected base
policy, retaining the online n-step discount and multi-noise target aggregation.
The residual actor, BC policy, and chord remain unchanged. Target critic updates
use `train.critic_target_update_freq`, counted in pretraining gradient updates.

`train.critic_pretrain_lr` defaults to `train.critic_lr`. Pretraining uses this
fixed LR without advancing either online LR scheduler; the critic's online LR
is restored afterward, while its learned optimizer moments are retained.
Pretraining updates do not advance the rollout counter, Q-filtering warmup,
or exploration warmup. Those existing schedules still count rollout calls,
including initial collection.

Logs distinguish `train/rollout_steps`, actual `train/simulator_steps`, replay
`buffer/size`, and `warmup/critic_updates`; explicit pretraining also logs its
final `warmup/critic_loss` and prints progress every 100 updates. Checkpoints
record warmup completion, pretraining update count, and simulator step count
as metadata. This does not add full replay/trajectory resume support.

For configs that omit the new settings, both collection warmup and explicit
pretraining default to zero. To reproduce that behavior in the HandChord
configs, set both `train.buffer_warmup_transitions=0` and
`train.critic_pretrain_steps=0`.

Run the warmup regression tests without simulator assets:

```bash
python -m unittest discover -s tests -p test_residual_warmup.py -v
```
