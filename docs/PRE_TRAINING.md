# DexJoCo BC Pretraining

Train the six single-arm tasks with the shared
[`TrainFlowMatchingImgAgent`](../agent/pretrain/train_flow_matching_img_agent.py)
and [`StitchedSequenceDataset`](../agent/dataset/sequence.py). The public dataset
already excludes the four rejected demonstrations, so no training-time filtering
is needed. Checkpoint results are pending (`TBD`).

## Run

Use the existing image config and override the task and experiment settings:

```bash
python script/run.py \
  --config-name=pre_flow_matching_unet_img \
  --config-dir=cfg/dexjoco/pretrain/pick_bucket/ \
  env_name=click_mouse horizon_steps=16 act_steps=16 \
  train.n_epochs=2000 train.batch_size=256 \
  train.lr_scheduler.first_cycle_steps=2000 \
  +train.val_freq=100 train.save_model_freq=100 \
  train_dataset.max_n_episodes=10000
```

Set `DICE_RL_DATA_DIR` and `DICE_RL_LOG_DIR` first. Replace `env_name` to select
another single-arm task. The command loads all episodes; change
`train_dataset.max_n_episodes` to `50` for the 50-demo condition. The loader uses the
normalization shipped with the dataset for both conditions; it does not refit
statistics when limiting the number of episodes.

## Hyperparameters

| Hyperparameter | Click Mouse | Fold Glasses | Hammer Nail | Pick Bucket | Pinch Tongs | Water Plant |
|---|---|---|---|---|---|---|
| Proprioceptive dim | 23 | 23 | 23 | 23 | 23 | 23 |
| Action dim | 22 | 22 | 22 | 22 | 22 | 22 |
| Cameras | front, wrist | front, wrist | front, wrist | front, wrist | front, wrist | front, wrist |
| Image resolution | 96 × 96 | 96 × 96 | 96 × 96 | 96 × 96 | 96 × 96 | 96 × 96 |
| Crop size | 84 × 84 | 84 × 84 | 84 × 84 | 84 × 84 | 84 × 84 | 84 × 84 |
| U-Net base channels | 128 | 128 | 128 | 128 | 128 | 128 |
| U-Net channels | [128, 256, 256] | [128, 256, 256] | [128, 256, 256] | [128, 256, 256] | [128, 256, 256] | [128, 256, 256] |
| Flow steps | 10 | 10 | 10 | 10 | 10 | 10 |
| Prediction / execution steps | 16 / 16 | 16 / 16 | 16 / 16 | 16 / 16 | 16 / 16 | 16 / 16 |
| Used demos (50 / all) | 50 / 99 | 50 / 100 | 50 / 99 | 50 / 98 | 50 / 100 | 50 / 100 |



Shared settings:

- **Model:** ResNet-18 encoders trained from scratch; one observation frame; Euler flow integration.
- **Training:** 2,000 epochs, batch size 256, seed 42. AdamW with LR `1e-4` and weight decay `1e-6`; cosine LR schedule with 100 warmup epochs and minimum LR `1e-5`.

## Results

During training, `train.val_freq` controls rollout frequency and each rollout
evaluates one episode per `env.n_envs`. The shared trainer saves periodic and
final checkpoints; it does not manage a separate `best.pt`.

Evaluate a saved checkpoint over 100 fixed-seed episodes (`10000`–`10099`) with:

```bash
python script/eval_pretrain_checkpoint.py \
  --ckpt_path /path/to/checkpoint/state_2000.pt \
  --num_eval_episodes 10 --eval_n_envs 10
```

Record the checkpoint and its measured success rate in the table below.

| Task | 50 demos: success (%) | All demos: success (%) | Checkpoint links (50 / all) |
|---|---|---|---|
| Click Mouse | TBD | TBD | TBD / TBD |
| Fold Glasses | TBD | TBD | TBD / TBD |
| Hammer Nail | TBD | TBD | TBD / TBD |
| Pick Bucket | — | TBD | — / TBD |
| Pinch Tongs | TBD | TBD | TBD / TBD |
| Water Plant | TBD | TBD | TBD / TBD |
