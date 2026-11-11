# Temporal HandChord K9, seed 42

Frozen hand codecs for PickBucket and FoldGlasses, trained for 20,000 steps each.
Per-frame latent layout B,H,K with H16/K9; packed actor dimension 240 (96 arm + 144 hand latent).
Checkpoints include weights, normalizers, and training-fitted latent statistics, and require no original normalization NPZ for codec inference.
Use the Temporal HandChord implementation on branch `temporal-hand-chord-pick-bucket-fold-glasses` in `stpark7/dice-rl`.

The exported weights come from the best held-out reconstruction step: PickBucket step 5000, MAE 1.0528372829 degrees; FoldGlasses step 19500, MAE 0.9191310827 degrees.
These are codec reconstruction metrics, not simulator task success rates. Both checkpoints were freshly verified on the original CUDA evaluation path. CPU/CUDA fused kernels can yield small numeric differences.

Training uses seed 42, episode split seed 0, batch 128, AdamW 3e-4, and KL maximum 3e-4 after 4000 zero-weight + 4000 ramp steps. No temporal derivative loss.
The next residual RL run uses seed 42 and the training parameters from W&B run `gist_ailab/dexjoco-pick_bucket_latent_residual_batch256/zbsqw25b`, with task-specific BC checkpoints and this temporal codec.

See `checksums.sha256` and per-task metadata for file integrity, model layout, normalization fingerprints, and exact episode split IDs.
