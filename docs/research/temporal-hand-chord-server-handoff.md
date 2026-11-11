# Temporal HandChord: server handoff

Branch: `temporal-hand-chord-pick-bucket-fold-glasses` in `stpark7/dice-rl`.
Local codec training is complete. Residual RL main training has **not** started.
The user requested server execution without lowering parameters for memory.
The final requested comparison reuses the existing Global K8 results and trains
only Temporal K9 on both tasks, with seeds 42, 142, 242 and 342. Do not launch
new Global K8 codec training, RL training, or evaluation. Follow
[the next-agent experiment prompt](hand-chord-four-seed-agent-prompt.md) for that
comparison with eight new Temporal RL runs. The two-task launcher below is the Temporal K9 seed-42 entry
point; it does not itself schedule the complete comparison matrix.

## Reference configuration

The source is the finished seed-42 W&B run
[`zbsqw25b`](https://wandb.ai/gist_ailab/dexjoco-pick_bucket_latent_residual_batch256/runs/zbsqw25b),
created 2026-11-05 03:59:09 UTC. All four runs in the supplied project use the
same training parameters apart from seed and output/run names.

Both new `ft_temporal_hand_chord_residual_flow_unet_img` configs preserve:

| Parameter | Value |
| --- | --- |
| Seed | 42 |
| Training environments / rollout calls | 10 / 10,000 |
| Batch size / gradient steps per rollout | 256 / 10 |
| Replay capacity / storage | 1,000,000 chunks / CPU, pinned batch staging |
| Actor and critic hidden layers | `[1024,1024,1024]` |
| Critic ensemble / actor noise candidates / next-noise candidates | 10 / 16 / 4 |
| Actor and critic learning rates / weight decay | 1e-4 / 1e-5 |
| Actor and target update intervals | 2 / 2 |
| Gamma / tau / n-step | 0.99 / 0.01 / 3 |
| BC weight / critic weight | 50 / 1 |
| Q-filter warmup / replay-flow warmup | 1,000 / 3,000 rollout calls |
| Adaptive expert fraction | 0.7 → 0.1 over 10,000 rollout calls |
| Evaluation and save frequency | Every 1,000 rollout calls |
| Evaluation | 50 rounds × 4 environments = 200 episodes |
| Flow steps / horizon / action execution steps | 10 / 16 / 16 |
| Noise / raw action / packed Temporal K9 actor dimensions | 352 / 22 / 240 |

Replay capacity 1,000,000 matches the reference; it was not lowered to fit local
memory. Task-specific BC checkpoint, dataset, normalizer, demonstration count and
episode length remain task-specific: PickBucket uses 98 demos and 700 steps;
FoldGlasses uses 100 demos and 900 steps. `obs_dim=23` in the config becomes 151
after the image trainer adds the frozen policy's 128 visual features, matching
the reference's runtime configuration. WandB logging defaults to disabled;
configs, checkpoints, logs and evaluation CSVs are saved locally.

## Checkpoints included in Git

The user explicitly requested committing these two small checkpoints, overriding
the repository's usual exclusion of checkpoints. Only the two exported codecs
and their metadata are included; datasets, base policies and run logs are excluded.
No Hugging Face transfer is required. The optional upload returned HTTP 401 and
did not create a commit there.

| Task | Codec path | Best step | Held-out hand MAE |
| --- | --- | ---: | ---: |
| PickBucket | `checkpoints/temporal-hand-chord/k9_seed42/pick_bucket/chord.pt` | 5,000 | 1.052837° |
| FoldGlasses | `checkpoints/temporal-hand-chord/k9_seed42/fold_glasses/chord.pt` | 19,500 | 0.919131° |

Each codec was trained for 20,000 optimizer steps, batch 128, seed 42, episode split
seed 0. Validation episodes are disjoint from training. Checkpoints are standalone:
weights, physical normalizers and training-fitted latent statistics are embedded.
`metadata.json`, `training_summary.json`, and `checksums.sha256` document provenance.
Fresh CUDA evaluation matches the recorded validation values. CPU/CUDA fused
Transformer kernels can differ slightly; this is numeric variation, not an export
weight mismatch. These MAEs do not measure simulator task success.

## Server setup and execution

Fetch the branch from the user's fork and activate the existing DICE-RL environment:

```bash
git fetch https://github.com/stpark7/dice-rl.git temporal-hand-chord-pick-bucket-fold-glasses
git switch -c temporal-hand-chord-pick-bucket-fold-glasses FETCH_HEAD
conda activate dice-rl

export DICE_RL_DATA_DIR=/absolute/server/data_dir
export DICE_RL_LOG_DIR=/absolute/server/log_dir
```

The following inputs must already exist on the server:

- `$DICE_RL_DATA_DIR/dexjoco/{pick_bucket,fold_glasses}-img/ph_pretrain/normalization.npz`
- `$DICE_RL_DATA_DIR/dexjoco/{pick_bucket,fold_glasses}-img/ph_pretrain/train.npz` for the additional four-seed codec runs.
- `$DICE_RL_DATA_DIR/dexjoco/{pick_bucket,fold_glasses}-img/ph_finetune/train.npz`
- `$DICE_RL_LOG_DIR/dexjoco-pretrain/pretrained_bc_policy_pick_bucket_img/checkpoint/state_2000.pt`
- `$DICE_RL_LOG_DIR/dexjoco-pretrain/pretrained_bc_policy_fold_glasses_img/checkpoint/state_1100.pt`
- The corresponding base-policy run's `.hydra/config.yaml` next to each `checkpoint/` directory.

Use the original dataset normalizers. Their SHA256 values are recorded in each
codec's metadata. The original base-policy configs are required to reconstruct
the BC image models; copying checkpoint weights alone is insufficient.

From the repository root, verify the included artifacts:

```bash
(cd checkpoints/temporal-hand-chord/k9_seed42 && sha256sum -c checksums.sha256)
```

Start the full reference configuration sequentially in a persistent session:

```bash
tmux new -s temporal-k9-rl
bash script/train_temporal_hand_chord_rl.sh \
  "$PWD/checkpoints/temporal-hand-chord/k9_seed42" \
  "$DICE_RL_LOG_DIR/temporal-hand-chord/rl_k9_seed42"
```

Detach with Ctrl-B, then D. The script starts PickBucket, then starts FoldGlasses
only after PickBucket exits successfully. It sets EGL rendering, keeps seed 42,
and overrides only codec/output paths and device. Select the physical GPU with
`CUDA_VISIBLE_DEVICES` before launch; `cuda:0` then addresses that visible device.
Use `DICE_RL_PYTHON=/absolute/path/to/python` if the activated interpreter is not
the intended environment. The output directory must be fresh; reruns are refused
to preserve existing evidence.

The initial raw BC and zero-residual codec projection evaluations each run 200
episodes before training. They use matching environment seeds and base-policy
noise. Later RL evaluations are recorded separately. Monitor:

```bash
cat "$DICE_RL_LOG_DIR/temporal-hand-chord/rl_k9_seed42/status"
tail -f "$DICE_RL_LOG_DIR/temporal-hand-chord/rl_k9_seed42/pick_bucket/train.log"
```

Each task directory contains `command.sh`, `status`, `train.log`, `.hydra/`,
`evaluation_results.csv`, and `checkpoint/model_step_<step>.pth` as it progresses.
The early part of `train.log` can be expert feature extraction or initial
evaluation, rather than optimizer updates. No training budget is reduced by the
launcher. Do not run both tasks concurrently on a memory-constrained machine.

## Validation and recovery

The implementation passed 48 tests with CUDA, including bundled-codec integrity
and default paths, real Hydra multiruns, decoder gradients, checkpoint
compatibility, reference configuration checks, matched baseline RNG, and launcher
failure/collision behavior. Both actual BC image models also passed compatibility
inference with the new codec. A local PickBucket simulator smoke completed four
rollout calls with a deliberately small smoke-only profile: actor optimizer
stepped four times and critic optimizer six times; all saved weights were finite.
This smoke profile is not used by the server launcher.

The original branch is `dexjoco-integration`, base commit `29d65d9`. Existing user
edits to ClickMouse/PinchTongs configs were preserved and excluded from these
commits. Stop any process using a checkout before switching its branch.
