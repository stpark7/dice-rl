# Temporal HandChord first training run

The implementation follows `temporal-hand-chord-implementation-plan.html`.
Branch: `temporal-hand-chord-pick-bucket-fold-glasses`, based on `29d65d9`.
The original `dexjoco-integration` branch remains available for recovery.
Existing uncommitted ClickMouse/PinchTongs config edits remain untouched.
An original diff and the previously untracked research documents are backed up
under `log_dir/temporal-hand-chord/2026-11-11/recovery/`.

## Implemented

- `codec=global_chunk` retains the existing whole-chunk VAE; `codec=temporal_chunk`
  selects one K-dimensional latent per action frame, conditioned on measured q0.
- Temporal K9 uses 144 hand latent values and 96 unchanged arm values, for a
  packed actor output of 240. Raw actions remain `[B,16,22]`; noise remains 352.
- A shared checkpoint loader restores global v1 and temporal v2 codecs, including
  embedded normalizers. The codec stays frozen during residual RL while decoder
  gradients reach the actor.
- Checkpoints, resolved configs, summaries and Hydra files share the actual run
  directory, including multiruns. Summaries include episode split IDs.
- KL sums over K and averages over batch/time for both variants. No velocity or
  acceleration loss is added in this first comparison.

## First experiment

| Setting | Both tasks |
| --- | --- |
| Tasks | `pick_bucket`, then `fold_glasses` |
| Codec / K / horizon | `temporal_chunk` / 9 / 16 |
| Model seed / episode split seed | 42 / 0 |
| Demonstrations | Up to 100: all 98 PickBucket and all 100 FoldGlasses episodes |
| Validation | 10% of whole episodes, disjoint from training |
| Optimizer | AdamW, learning rate 3e-4, weight decay 1e-4 |
| Budget / batch size | 20,000 steps / 128 |
| KL schedule | Zero for 4,000 steps, ramp for 4,000, maximum 3e-4 |
| Validation / checkpoint cadence | 500 / 1,000 steps |
| Runtime | Python 3.11.16, PyTorch 2.7.0+cu128, NumPy 1.26.4, Hydra 1.3.2 |
| Hardware | Shared RTX 5090, GPU 0; one codec job at a time |
| Tracking | Local logs, `wandb=null` |

The launcher explicitly overrides the task defaults so FoldGlasses uses the same
20,000-step budget as PickBucket. It opens only action/state arrays, without
loading dataset images or creating simulator environments.

Launch from the repository root using the installed environment:

```bash
DICE_RL_PYTHON=/home/lee/anaconda3/envs/dice-rl/bin/python \
  bash script/train_temporal_hand_chord.sh \
  "$PWD/log_dir/temporal-hand-chord/2026-11-11/k9_seed42"
```

Use a fresh output directory for each new run. The completed experiment ran
inside the persistent tmux session `temporal-chord-k9-20261111`. Per-task
directories contain `train.log`, `command.sh`, `status`, `history.jsonl`,
`config.yaml`, `.hydra/`, `best.pt`, `latest.pt`, `chord.pt`, and `summary.json`
as the run progresses. The final `chord.pt` exports the best validation model with
latent statistics fitted on training episodes.

```bash
cat log_dir/temporal-hand-chord/2026-11-11/k9_seed42/status
tail -n 10 log_dir/temporal-hand-chord/2026-11-11/k9_seed42/pick_bucket/train.log
tail -n 10 log_dir/temporal-hand-chord/2026-11-11/k9_seed42/fold_glasses/train.log
```

Both 20,000-step runs completed. Exported best checkpoints: PickBucket step 5,000,
held-out MAE 1.052837 degrees; FoldGlasses step 19,500, held-out MAE 0.919131 degrees.
The two self-contained codecs and metadata are committed under
`checkpoints/temporal-hand-chord/k9_seed42/` at the user's request.
See [server handoff](temporal-hand-chord-server-handoff.md) to start residual RL.

## Validation

`OMP_NUM_THREADS=1 /home/lee/anaconda3/envs/dice-rl/bin/python -m unittest discover -s tests -v`
passed all 48 tests with CUDA available, including bundled-codec integrity and default-path checks. This includes exact arm preservation,
temporal q0 conditioning, standalone checkpoint roundtrips, old global codecs,
KL normalization, actor gradients through a frozen codec, raw critic interfaces,
short trainer exports, a real two-codec Hydra multirun, reference RL settings,
matched-noise baseline evaluation, and collision-safe launchers. All three existing
PickBucket/FoldGlasses global codec checkpoints also loaded successfully.

Real-data smoke runs use 10 episodes and 50 GPU optimization steps for each task,
stored under `log_dir/temporal-hand-chord/2026-11-11/smoke/`. Their short budget
validates execution and export; it does not establish final policy quality.

## Follow-on comparisons

This run trained the codec. Residual RL will run on the server at the user's
request; local RL main training was not started. Keep the base policy, episode split,
seed, training budget and KL normalization fixed when comparing Global K8,
Temporal K8, Temporal K9, and the scalar-budget control Global K128.
Report raw-base, zero-residual codec projection and trained RL success separately.
The trainer now records `pretrained`, `codec_projection`, and `finetuned` rows in
`evaluation_results.csv`. The initial raw/projection pair shares environment seeds
and flow noise; the added projection does not advance the training RNG stream.
The standalone RL checkpoint evaluation script still reports only raw-base/RL.

To use a completed codec with the existing residual RL configuration, override
`model.chord_path` with that task's exported absolute `chord.pt` path. The actor
output size is read from the codec; retain the base checkpoint and raw dimensions.

## Recovery

The experiment code is committed only on the new branch. After stopping any
training that uses this checkout, `git switch dexjoco-integration` returns to the
original code. The original config edits remain uncommitted, and ignored training
outputs remain on disk. Do not use a destructive reset or clean to switch back.
`latest.pt` contains codec weights rather than optimizer/RNG state; it is not an
exact training-resume checkpoint.
