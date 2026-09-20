# Dexjoco image BC: server setup notes

## Scope and completion target

The data pipeline supports six image BC tasks. Dataset download, conversion,
and loading are available; training configurations and offline trainer
integration still need to be completed before launching training.
No Dexjoco training checkpoints are included.

## Obtain the data

Repository: <https://huggingface.co/datasets/robopark/dexjoco-image-data> (private).
Published release commit: `68a239b1ddabe7228026f94ce57845d10f7d98f2`.
Authenticate with a read-capable HF token through the environment or standard
Hugging Face login. No credential is stored in this repository or the release.

Use `script/download_hf.sh dexjoco` from the repository root.

```bash
export DICE_RL_DATA_DIR="$PWD/data_dir"
bash script/download_hf.sh dexjoco
# Pin the verified release:
bash script/download_hf.sh dexjoco --revision 68a239b1ddabe7228026f94ce57845d10f7d98f2
# Optional single-task download:
bash script/download_hf.sh dexjoco --task pick_bucket
```

The downloader prints the resolved HF commit. Record it in each experiment and
use `--revision <commit>` for later reproduction. `--repo-id`, `--data-dir` and
`--cache-dir` are optional overrides. No `DICE_RL_LOG_DIR` is required for this
data-only command. The no-argument script retains its Robomimic behavior.

Archives and extracted data both occupy disk space. A failed installation leaves
no partially installed task directory; the HF cache can be reused on retry.
An existing dataset is never silently replaced. Download receipts allow an
unchanged completed installation to be reused, but do not detect subsequent
manual edits to individual chunks.

## Dataset contract

Each `${DICE_RL_DATA_DIR}/dexjoco/{task}-img/` contains:

- `episodes.json`: episode order, prefix cuts, rejections, camera names and dimensions.
- `images.zarr/{camera}`: all recorded views, RGB uint8 `(T, 3, 96, 96)`.
- `ph_pretrain/train.npz`: float32 normalized states/actions, trajectory lengths,
  camera names and relative image archive reference `../images.zarr`.
- `ph_pretrain/normalization.npz`: state/action min/max, normalized flag.
- `ph_finetune/`: identical shared data plus synthetic terminal rewards/dones.

Keep the sibling image archive with both NPZ directories. Recorded success
labels are not imported; the synthetic finetuning rewards do not establish
successful demonstrations. Normalization uses all retained episodes in a task;
it was not fitted on a separate validation split.

| Task | Proprioception | Actions | Available cameras | Initial training selection |
|---|---:|---:|---|---|
| hammer_nail | 23 | 22 | ego_left, ego_right, front, wrist | front, wrist |
| pick_bucket | 23 | 22 | ego_left, ego_right, front, wrist | front, wrist |
| fold_glasses | 23 | 22 | ego_left, ego_right, front, wrist | front, wrist |
| bimanual_assembly | 46 | 44 | ego, wrist_left, wrist_right | ego, wrist_left, wrist_right |
| bimanual_hanoi | 46 | 44 | ego, wrist_left, wrist_right | ego, wrist_left, wrist_right |
| bimanual_microwave_cook | 46 | 44 | ego, wrist_left, wrist_right | ego, wrist_left, wrist_right |

Single-arm states are `[tcp_xyz3, tcp_quat_wxyz4, hand16]`. Bimanual states are
`[right_tcp7, left_tcp7, right_hand16, left_hand16]`; they are not interleaved
23-dimensional arm blocks. Bimanual actions are `[right_action22, left_action22]`.
Each arm action is `[world_delta_xyz3, world_relative_rotvec3, absolute_hand16]`.
The rotation increment left-multiplies the measured orientation. Unnormalize
actions before interpreting them; these are not raw 23/46-dimensional absolute
quaternion targets. Set `use_6d_rot: false` and `abs_action: false` for the loader.

## Data-loader example

This is a dataset block, not a complete executable training configuration:

```yaml
train_dataset:
  _target_: agent.dataset.sequence.StitchedSequenceDataset
  dataset_path: ${oc.env:DICE_RL_DATA_DIR}/dexjoco/pick_bucket-img/ph_pretrain/train.npz
  use_img: true
  image_keys: [front, wrist]
  horizon_steps: 8
  cond_steps: 1
  img_cond_steps: 1
  max_n_episodes: 100
  device: cpu
  use_6d_rot: false
  abs_action: false
```

Set model `obs_dim`, `action_dim`, `cond_dim = obs_dim * cond_steps`, `num_img`,
image size and crop consistently. Suggested crops are 84×84 within the stored
96×96 images. Loader RGB samples are `(history, 3 * num_img, 96, 96)` and actions
are `(horizon, action_dim)`. The loader leaves RGB as uint8. The current vision
network scales RGB internally. The dataset loader must stay on CPU for worker
loading; the training loop transfers the batch to the model device.

## Required trainer integration before launching

1. Prepare six Dexjoco task configurations with the dimensions and camera
   selections above. No ready-to-run Dexjoco training configuration is included.
2. Add and test an offline-BC mode. `PreTrainAgent.__init__` currently always
   constructs and seeds a simulator. Merely increasing evaluation frequency
   does not remove this dependency.
3. `TrainFlowMatchingImgAgent.__init__` resets `val_freq` to 100. Disable rollout
   explicitly in offline mode and remove remaining environment accesses.
4. The current Dexjoco image wrapper handles single-arm execution. Bimanual
   dataset support does not imply bimanual simulator rollout support. Offline
   BC does not require that adapter; online evaluation/RL does.
5. Verify PyTorch/torchvision compatibility. The project pins torch 2.4.0 but
   does not list torchvision as a core dependency, although the image model
   imports it. Consult the release provenance for the actual conversion
   environment; do not blindly copy it as a server CUDA environment.
6. Each independent process should see one physical GPU through
   `CUDA_VISIBLE_DEVICES` and use logical `device=cuda:0`. First smoke-test a
   single-arm and a bimanual batch before launching all six.
7. Save early and verify model/EMA reloading. The current checkpoint stores
   model, EMA and epoch, not optimizer/scheduler state. Do not claim exact
   training resume without extending and testing checkpoint support.

Set `max_n_episodes: 100` to retain all 100 valid episodes per task.
Record seed (planned: 42), dataset revision, camera order, config, dependencies,
GPU and smoke-run results. A loss curve and checkpoint do not establish task
success; no policy success rate has been measured in this local preparation.
