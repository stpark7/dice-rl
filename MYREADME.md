# 논문명
<학회명>

[[Paper](https://arxiv.org/abs/2603.10263)]&nbsp;&nbsp;[[Website](https://zhanyisun.github.io/dice.rl.2026/)]&nbsp;&nbsp;[[Datasets](https://huggingface.co/datasets/robopark/dexjoco-single-arm-data)]&nbsp;&nbsp;[[Checkpoints](https://huggingface.co/wintermelontree/robomimic-pretrain-checkpoints)]&nbsp;&nbsp;[[Real Robot Code](https://github.com/zhanyisun/DICE-RL-Robot)]

[Zhanyi Sun](https://zhanyisun.github.io/), [Shuran Song](https://shurans.github.io/)

<학교명>

<img src="media/teaser.png" alt="Teaser" width="100%">

> Figure 추가 후, 모델의 대표 figure 설명 적기

## Installation


1. Clone the repository

   ```bash
   git clone https://github.com/stpark7/dice-rl.git
   git clone https://github.com/brave-eai/dexjoco.git
   cd dice-rl
   ```

2. Install core dependencies with a conda environment.

   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential libegl1 libgl1 libglfw3 ffmpeg
   ```

3. Create a Python 3.11 environment. DexJoCo requires Python 3.10 or 3.11.

   ```bash
   conda create -n dice-rl python=3.11 -y
   conda activate dice-rl
   python -m pip install 'setuptools<81' wheel
   ```

4. Install DICE-RL, its image encoder dependency, and DexJoCo into that environment.

   ```bash
   python -m pip install -r installation/requirements-dexjoco.txt
   ```

   Run from the repository root with `dice-rl` activated. The requirements file installs DICE-RL and the sibling DexJoCo checkout in editable mode, with PyTorch 2.7.0 / torchvision 0.22.0 (CUDA 12.8), MuJoCo 3.4.0, and NumPy 1.26.4.

5. Set data/log paths from the repository root. Finetuning and HandChord configs use these variables, falling back to `data_dir/` and `log_dir/` when unset.

   ```bash
   export DICE_RL_DATA_DIR="$PWD/data_dir"
   export DICE_RL_LOG_DIR="$PWD/log_dir"
   export MUJOCO_GL=egl
   export PYOPENGL_PLATFORM=egl
   ```

   `source script/set_path.sh` is an interactive alternative that also appends exports to `~/.bashrc`. After moving the repository, update any old exported paths. Use `wandb=null` on the training command to disable WandB.

## DexJoCo Data Preprocessing

> [!TIP]
> **Using the Hugging Face datasets as provided? Skip this section** and continue to [Download datasets and checkpoints from Hugging Face](#download-datasets-and-checkpoints-from-hugging-face).
>
> Preprocessing is only needed when regenerating datasets from raw recordings, for example to **increase the image resolution**.

To process raw Zarr recordings and camera videos yourself, use `script/dataset/process_dexjoco_dataset.py`.

### Dataset Quality Notes

Raw DexJoCo demonstrations contain many idle frames, especially while waiting before a task begins. Training directly on these recordings can overrepresent waiting or holding behavior in the BC loss.

Four demonstrations were removed from the raw dataset due to poor trajectory quality: pick_bucket 39 (244967) and 12 (974934), click_mouse 42 (581035), and hammer_nail 2 (403013).

The processed 2026-10-21 release contains 596 demonstrations and 216,890 retained transitions: 98 pick_bucket, 99 click_mouse, 99 hammer_nail, and 100 for each remaining task. Normalization was recomputed for the three affected tasks using only the remaining demonstrations.

### Processing Raw Demonstrations

Preprocessing always saves every recorded camera together with states and actions.

Run from the repository root, replacing `--raw_path` with your task's raw recording directory:

```bash
python script/dataset/process_dexjoco_dataset.py \
  --task pick_bucket \
  --raw_path /path/to/raw/pick_bucket \
  --save_dir data_dir/dexjoco/pick_bucket-img
```

Outputs are saved under `ph_pretrain/` and `ph_finetune/`.

To train at another image resolution, regenerate the dataset from the raw videos with `--image_size`. For example, to save 128×128 images:

```bash
python script/dataset/process_dexjoco_dataset.py \
  --task pick_bucket \
  --raw_path /path/to/raw/pick_bucket \
  --save_dir data_dir/dexjoco/pick_bucket-img-128 \
  --image_size 128
```
❗ Update the image dimensions in your training config to match the dataset.


See [the preprocessing guide](script/dataset/README.md) for details.

## Download datasets and checkpoints from Hugging Face
Download datasets from [Hugging Face](https://huggingface.co/datasets/robopark/dexjoco-single-arm-data) with the following command. The DexJoCo downloader installs datasets under `$DICE_RL_DATA_DIR/dexjoco/`; it does **not** download BC or HandChord checkpoints. Prepare those separately using the paths in [Finetuning](#finetuning), or override the paths for your own pretrained models.

Run from the repository root:

```bash
# Download all six tasks
bash script/download_hf.sh

# Download selected tasks
bash script/download_hf.sh --task hammer_nail water_plant
```


## Evaluating finetuned RL checkpoints and pretrained BC checkpoints
To directly evaluate the finetuned RL checkpoints and pretrained BC checkpoints and to get success rates for both, use the following commands. Make sure to change the pretrained checkpoint path in the config file of the finetuning checkpoint. The
script/eval_rl_checkpoint.py script will automatically search for the corresponding pretrained BC checkpoint and evaluate it as well.

```console
python script/eval_rl_checkpoint.py \
  --ckpt_path path_to_finetuned_checkpoint \
  --num_eval_episodes 10 \
  --eval_n_envs 10
```
The output includes success rates for the finetuned RL policy and the original pretrained BC policy, plus their difference. `--num_eval_episodes` counts parallel evaluation **rounds**: the command above evaluates **100 episodes per policy** (`10 × 10`). This script uses `max_q_min` with 16 candidates for RL and skips exploration warmup. The BC comparison uses the original BC actions, without HandChord projection.

To evaluate a pretrained flow-matching BC checkpoint independently, use:

```console
python script/eval_pretrain_checkpoint.py \
  --ckpt_path path_to_pretrained_checkpoint \
  --num_eval_episodes 10 \
  --eval_n_envs 10
```

See [DexJoCo BC pretraining](docs/PRE_TRAINING.md) for hyperparameters and success rates.

## Pretraining
**Note**: You may skip pre-training if you would like to use the default checkpoint (available for download at Hugging Face) for finetuning.

### HandChord Pre-training:
```bash
python script/run.py --config-name=pre_chunk_hand_chord --config-dir=cfg/dexjoco/hand_chord/pretrain/{env_name}/
```

### BC Policy Pre-training
To pretrained image-based BC policies on the Dexjoco dataset, use the following command.
```bash
python script/run.py --config-name=pre_flow_matching_unet_img --config-dir=cfg/dexjoco/pretrain/{env_name}/
```

## Finetuning

### Raw-action residual RL

Finetune a pretrained BC policy with residuals in raw action space. Replace `{task}` with the target task.

```bash
python script/run.py \
  --config-dir=cfg/dexjoco/finetune/{task}/ \
  --config-name=ft_distill_residual_flow_unet_img
```

### HandChord residual RL

Finetune a pretrained BC policy with residuals in HandChord coordinates. Set `model.chord_path` in the task YAML before running.

```bash
python script/run.py \
  --config-dir=cfg/dexjoco/finetune/{task}/ \
  --config-name=ft_hand_chord_residual_flow_unet_img
```

# Code Acknowledgements(미)
Our code base is built on top of the following repositories. We thank the authors for open-sourcing their code.
- [DPPO](https://github.com/irom-princeton/dppo): our pretraining and finetuning stack is built on top of the DPPO codebase.
- [Robomimic](https://github.com/ARISE-Initiative/robomimic) and [Diffusion Policy](https://github.com/real-stanford/diffusion_policy): the encoder and policy architecture for image-based policies are adapted from the codebases of Robomimic and Diffusion Policy.
- [DICE-RL](https://github.com/real-stanford/dice-rl)


If you find this codebase useful, consider citing:

```bibtex
@article{sun2026prior,
  title={From Prior to Pro: Efficient Skill Mastery via Distribution Contractive RL Finetuning},
  author={Sun, Zhanyi and Song, Shuran},
  journal={arXiv preprint arXiv:2603.10263},
  year={2026}
}
```

# Contact
If you have any questions, please feel free to contact [Sangtae Park](stpark5013@gm.gist.ac.kr). If you leave an issue, please send me an accompanying email!
