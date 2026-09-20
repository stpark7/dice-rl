## Data processing scripts

### DexJoCo images

Convert recorded Zarr states/actions and every camera in each episode's
`videos/*.mp4` folder. Original recordings are left intact.

```console
python script/dataset/process_dexjoco_dataset.py --task pick_bucket --with_images
```

The default output is `data_dir/dexjoco/pick_bucket-img/`:

- `images.zarr/<camera>`: RGB uint8 arrays `(T, 3, 96, 96)`, one array per
  camera, compressed in 16-frame chunks. All cameras are saved, including
  `ego_left` and `ego_right`; names and counts are discovered, not hardcoded.
- `ph_pretrain/train.npz`: normalized `states`, `actions`, `traj_lengths`,
  `camera_names`, and the relative `image_store` reference `../images.zarr`.
- `ph_pretrain/normalization.npz`: state/action min/max statistics.
- `ph_finetune/`: the existing residual-RL output format, referencing the same
  image archive. Its existing terminal-success reward convention is unchanged;
  recorded success labels are not imported by this converter.
- `episodes.json`: source episode order, rejected episodes, prefix cuts,
  camera names, image size, and state/action dimensions.

MP4 frame `t` is assumed to match Zarr sample `t`. Conversion applies the same
prefix cut to all modalities and fails on differing frame counts or camera sets.
Images are never normalized or resized in the loader. To generate another
resolution, use `--image_size 128 --save_dir data_dir/dexjoco/pick_bucket-img-128`.
An existing image dataset is not overwritten; choose a new output directory.
Keep the NPZ directories and sibling `images.zarr` together when moving data.
Without `--with_images`, the previous state-only output remains available.

Choose cameras in the dataset configuration, in the order the model and online
environment should receive them:

```yaml
train_dataset:
  _target_: agent.dataset.sequence.StitchedSequenceDataset
  dataset_path: data_dir/dexjoco/pick_bucket-img/ph_pretrain/train.npz
  use_img: true
  image_keys: [front, wrist]  # Or [ego_left, ego_right, front, wrist]
  horizon_steps: 8
  cond_steps: 1
  img_cond_steps: 1
  device: cpu
```

The loader reads only requested observation frames and cameras; the full image
archive is not copied to GPU. Omitting `image_keys` selects all cameras in the
archive's recorded order. Legacy NPZ datasets with a single `images` array
continue to work, but do not support named camera selection. Match environment
`image_keys`, model `num_img`, input size (96), and crop size to the selection;
the snippet above is a dataset example, not a complete training configuration.

Bimanual conversion uses the same command with e.g. `--task bimanual_assembly`.
Its `ego`, `wrist_left`, and `wrist_right` videos are discovered automatically.
Image-policy states retain `[right_tcp7, left_tcp7, right_hand16, left_hand16]`
(46 dimensions). Raw recorded actions are `[right_target23, left_target23]`;
converted actions are `[right_delta22, left_delta22]` (44 dimensions). Single-arm
image policies use 23-dimensional proprioception and 22-dimensional actions.
Movement of either arm can end the initial held prefix. This supports dataset
conversion; the current single-arm online image wrapper is not a bimanual policy
execution adapter.

To generate data for training and finetuning, first download raw data from [this link](https://huggingface.co/datasets/wintermelontree/raw_robomimic_data/tree/main) or the official Robomimic repository. Then use the following commands to generate processed data for pretraining. 

Can (state-based)
```console
python script/dataset/process_robomimic_dataset.py --load_path path_to_raw_hdf5_data --save_dir ${DICE_RL_DATA_DIR}/robomimic/can-img/ph_pretrain --normalize
```

Can (image-based)
```console
python script/dataset/process_robomimic_dataset.py --load_path path_to_raw_hdf5_data --save_dir ${DICE_RL_DATA_DIR}/robomimic/can-img/ph_pretrain --normalize --cameras agentview robot0_eye_in_hand
```

Square (state-based)
```console
python script/dataset/process_robomimic_dataset.py --load_path path_to_raw_hdf5_data --save_dir ${DICE_RL_DATA_DIR}/robomimic/square/ph_pretrain --normalize
```

Square (image-based)
```console
python script/dataset/process_robomimic_dataset.py --load_path path_to_raw_hdf5_data --save_dir ${DICE_RL_DATA_DIR}/robomimic/square/ph_pretrain --normalize --cameras agentview robot0_eye_in_hand
```

Transport (state-based)
```console
python script/dataset/process_robomimic_dataset.py --load_path path_to_raw_hdf5_data --save_dir ${DICE_RL_DATA_DIR}/robomimic/transport-img/ph_pretrain --normalize
```     

Transport (image-based)
```console
python script/dataset/process_robomimic_dataset.py --load_path path_to_raw_hdf5_data --save_dir ${DICE_RL_DATA_DIR}/robomimic/transport-img/ph_pretrain --normalize --cameras robot0_eye_in_hand robot1_eye_in_hand shouldercamera0 shouldercamera1 
``` 

Tool Hang (state-based)
```console
python script/dataset/process_robomimic_dataset.py --load_path path_to_raw_hdf5_data --save_dir ${DICE_RL_DATA_DIR}/robomimic/tool_hang-img/ph_pretrain --normalize
```

Tool Hang (image-based)
```console
python script/dataset/process_robomimic_dataset.py --load_path path_to_raw_hdf5_data --save_dir ${DICE_RL_DATA_DIR}/robomimic/tool_hang-img/ph_pretrain --normalize --cameras sideview  robot0_eye_in_hand
```

By default, DICE-RL uses RLPD for finetuning. To generate data for finetuning, simply add `--truncate` to the command used for pretraining, which will truncate the trajectories to have exactly one success at the end. This is to ensure the value learning between offline data and online data is consistent.
