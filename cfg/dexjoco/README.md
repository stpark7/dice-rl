# DexJoCo image BC task configurations

The five task configs in `pretrain/<task>/pre_flow_matching_unet_img.yaml`
are standalone. Each file contains its task, environment, model, dataset, and
training settings; no shared base config is required.

| Task | State / action | Ordered cameras |
|---|---|---|
| hammer_nail | 23 / 22 | front, wrist |
| pick_bucket | 23 / 22 | front, wrist |
| bimanual_assembly | 46 / 44 | ego, wrist_left, wrist_right |
| bimanual_hanoi | 46 / 44 | ego, wrist_left, wrist_right |
| bimanual_microwave_cook | 46 / 44 | ego, wrist_left, wrist_right |

All configs use seed 42, 96x96 RGB, `use_6d_rot: false`, and `abs_action: false`.
Microwave Cook uses all 100 episodes; the other tasks use the first 50 episodes,
matching the Robomimic Tool Hang diffusion reference. None reserves a validation split.
The dataset stays on CPU; the existing trainer moves batches to `cuda:0`.
Expose one GPU per training process with `CUDA_VISIBLE_DEVICES`.
Camera order is shared by the dataset and environment. Model camera count,
RGB channels, state/action shapes, and conditioning width match each task.

Parameter order and shared training settings in all five configs follow
`cfg/robomimic/pretrain/tool_hang/pre_diffusion_mlp_img.yaml`: 8,000 epochs,
a 5,000-epoch LR cycle, dropout 0.1, EMA decay cap 0.995,
and checkpoint saving every 200 epochs. Microwave Cook now uses 5,000 training
epochs; other task configs retain 8,000. Batch size is 512 after a GPU memory
probe (the reference uses 128). With five bimanual assembly environments,
three optimizer updates at batch 512 peaked at 5.50 GiB allocated / 6.55 GiB
reserved by PyTorch on an RTX 3090; simulator/driver memory is additional.
They retain Flow Matching with 10 flow
steps, DexJoCo-specific dimensions/cameras/wrappers, 96x96 images with an 84x84
crop, the dataset-derived episode limit, and CPU dataset storage.

All use an 8-step action horizon/execution and one observation frame. These
are reference-based starting hyperparameters, not measured optimal DexJoCo
settings. Simulator domain randomization defaults to false; select the
evaluation distribution explicitly for an experiment.

## Episode limits and config inspection

Set each task's `env.max_episode_steps` to the maximum `traj_lengths` in its
BC dataset (`ph_pretrain/train.npz`), rounded up to a multiple of 100:
`((int(traj_lengths.max()) + 99) // 100) * 100`. For example, 1234 becomes
1300, while 1200 stays 1200. These are individual environment steps, not
action chunks. One override propagates to both environment wrappers. The
simulator may still truncate earlier according to its own internal limit.

The configured limits use `max_trajectory_length` from the pinned release
[`manifest.json`](https://huggingface.co/datasets/robopark/dexjoco-image-data/blob/68a239b1ddabe7228026f94ce57845d10f7d98f2/manifest.json).
The release audit computes this field directly from BC `train.npz`
`traj_lengths.max()`. The six task datasets have also been downloaded to
`data_dir/dexjoco/` at this revision with archive SHA-256 verification.
The installed BC arrays match the release episode counts, dimensions, and
trajectory maxima; first and last image frames are readable for every camera.
Local verification details are in `data_dir/dexjoco/download-verification.json`.

| Task | BC maximum steps | max_episode_steps |
|---|---:|---:|
| bimanual_assembly | 895 | 900 |
| bimanual_hanoi | 1429 | 1500 |
| bimanual_microwave_cook | 1028 | 1100 |
| hammer_nail | 522 | 600 |
| pick_bucket | 1055 | 1100 |

From the repository root, with the runtime dependencies installed:

```bash
export DICE_RL_DATA_DIR="$PWD/data_dir"
export DICE_RL_LOG_DIR="$PWD/log_dir"
task=pick_bucket
python script/run.py \
  --config-dir="cfg/dexjoco/pretrain/$task" \
  --config-name=pre_flow_matching_unet_img --cfg job --resolve
```

This inspects the config without launching training. Recompute the limit if
the BC dataset changes; override it with `env.max_episode_steps=<step-count>`.
`wandb=null` disables experiment tracking.

## Evaluation behavior

Microwave Cook uses `env.n_envs: 20`; the other four configs use `env.n_envs: 10`, with
`reset_within_step: true`, and the multi-step wrapper's default reward sum.
They have no separate `eval` block: the current image trainer runs one round
every 100 epochs using `model` weights, with no separate final evaluation.
Automatic reset retains the reference behavior; it does not implement fixed
episode accounting across rounds.

The previous unused `eval` protocol fields were removed rather than implying
that the trainer honors episode counts, final evaluation, seed overrides, or
weight selection. Supporting those options requires changes to the image
evaluator. `train.render.freq` matches the reference value of 2 but does not
control rollout frequency in the current image trainer. `n_envs` is per
process, so all five configs together would create 60 environments. After
stopping Hammer Nail and Assembly, the three active training jobs use 40 environments.
These config files do not change trainer code or establish policy success rates.
