# DAWN baseline on DexJoCo

DAWN (Data-Anchored Warmup and Normalization) is available through the same
Hydra launcher as ResFiT. The implementation follows the residual SAC recipe in
the local `../DAWN/online_dawn/pi_dec_diffusion_maniskill2_dawn.py`: initial
rollouts use only the frozen BC, and twin critics use Layer Normalization.
The original DAWN repository is a reference; it is not a runtime dependency.

Standalone `ft_dawn_unet_img.yaml` files are provided under
`cfg/dexjoco/finetune/<task>/` for `click_mouse`, `fold_glasses`, `hammer_nail`,
`pick_bucket`, `pinch_tongs`, and `water_plant`. Their environment settings,
camera order, BC checkpoint defaults, normalizer defaults, and evaluation
seeds match the corresponding ResFiT configurations.

## Train and evaluate

Run from this repository's root with the DICE-RL environment activated and
DexJoCo installed. Set `DICE_RL_LOG_DIR` and `DICE_RL_DATA_DIR` if the prepared
files are outside the default `log_dir` and `data_dir` directories.

```bash
python script/run.py \
  --config-dir=cfg/dexjoco/finetune/pick_bucket \
  --config-name=ft_dawn_unet_img \
  base_policy_path=/path/to/native_bc/checkpoint/state_2000.pt \
  normalization_path=/path/to/normalization.npz \
  wandb=null
```

The native BC checkpoint requires its training `.hydra/config.yaml`. The loader
searches the checkpoint's ancestor directories; use
`base_policy_config_path=/path/to/bc_config.yaml` if it was moved. Select
`base_policy_weights=ema` to use native EMA weights instead of the default
`model` weights. External Policy Decorator checkpoints from DAWN have a
different model format and cannot be substituted for native DICE-RL BC weights.

The configured `horizon_steps: 16` must match the BC's prediction horizon.
`bc_action_steps` controls how many predicted actions are consumed before
replanning and must be between 1 and that horizon. For an 8-step BC checkpoint,
add `horizon_steps=8 bc_action_steps=8`. Keep `act_steps=1`: the residual is
recomputed at every individual control step. Match the BC's state dimensions,
observation history, camera order, and normalization as well. Fold Glasses
retains ResFiT's `hf-2ea810265e23` checkpoint directory and its checkpoint-local
normalizer; override the paths if using another BC run.

Inspect a resolved configuration without loading weights or a simulator:

```bash
python script/run.py \
  --config-dir=cfg/dexjoco/finetune/pick_bucket \
  --config-name=ft_dawn_unet_img --cfg job --resolve wandb=null
```

Evaluate a trained checkpoint using its training configuration and BC assets:

```bash
python script/run.py \
  --config-dir=cfg/dexjoco/finetune/pick_bucket \
  --config-name=ft_dawn_unet_img \
  mode=eval checkpoint_path=/path/to/dawn/checkpoint/state_1600000.pt \
  base_policy_path=/path/to/native_bc/checkpoint/state_2000.pt \
  normalization_path=/path/to/normalization.npz \
  wandb=null
```

Repeat any training-time model, horizon, BC-weight, observation, or normalization
overrides during evaluation. Evaluation runs exactly `num_eval_episodes: 100`
episodes across `eval_n_envs: 5` workers, beginning at `eval_seed: 10000`.
With `eval_base_policy: true`, the frozen BC is evaluated with identical seeds
and its original chunk cadence. Set it to `false` to evaluate only DAWN.

Identical seeds reproduce each episode's initial scene, not its outcome.
DexJoCo camera frames are not bit-reproducible: two resets with one seed give
the same state but frames that differ in a few pixels by one intensity level,
and contact dynamics amplify the resulting action differences. BC and DAWN
results are therefore independent samples, not paired episodes. In a ResFiT
Pick Bucket run the zero-initialized residual, which executes exactly the BC's
actions, scored 0.68 against the BC's 0.80 on the same 100 seeds. Treat
differences of that size between single evaluations as noise.

Results are written to `eval.csv` and `eval_metrics.json` under the run log
directory, `dexjoco-dawn/<task>_dawn_unet_img/dawn-s<seed>_<timestamp>` in the
log root. That directory name is also the WandB run name. Each checkpoint
includes the frozen BC weights (290 MB per file for Fold Glasses), saved every
`train.save_freq: 10000` transitions. Training checkpoints contain the model,
configuration, optimizer state, and progress counters. Training with
`checkpoint_path` warm-starts weights with fresh replay and warmup; it is not
an exact replay-buffer resume.

## Adaptation and experiment settings

Upstream's ManiSkill diffusion experiment applies residual SAC to four-action
chunks. This DexJoCo adaptation uses the native image BC as a frozen chunk
predictor and applies a fresh residual to each queued control action. The BC
visual encoder stays frozen and in evaluation mode; replay stores its visual
features plus proprioception rather than RGB images. Each vector worker owns
its own BC action queue, which is cleared when that worker's episode ends.

The actor conditions on observation features and outputs a Gaussian residual.
Critics condition on observation features and the combined action
`base_action + residual_scale * residual`. Their losses use that raw combined
action; simulator execution clips it to `[-1, 1]`. Entropy and automatic
temperature tuning use the raw residual distribution, before residual scaling
and action clipping. With `target_entropy: null`, the target is minus the
per-step residual action dimension. `autotune: false` keeps `sac_alpha` fixed.

The first `train.buffer_warmup_steps: 20000` transitions execute only the BC.
Online replay then feeds SAC updates without expert demonstrations. The default
`train.critic_warmup_steps: 0` adds no separate critic-only phase. Critic targets
use the minimum of the two target critics and the SAC entropy term. Only true
terminations disable bootstrapping; timeouts bootstrap from their final
pre-reset observations.

Training lengths, warmup lengths, evaluation/save/log frequencies, and
`train.training_freq` count individual simulator transitions summed across
workers. `train.utd_ratio: 0.25` requests one gradient update per four collected
transitions after warmup. Updates are grouped into bursts controlled by
`train.training_freq: 64`; fractional update budgets carry forward. With
multiple workers, a rollout can cross a configured transition boundary by
up to one vector step. `policy_frequency` and `target_network_frequency` count
gradient updates. `train.reward_offset: -1.0` shifts rewards stored for learning
only; evaluation returns and success rates use the original environment rewards.
`train.max_grad_norm: 50` clips the actor and each critic separately, as
upstream does; the temperature is not clipped.

The default residual scale 0.1, SAC discount 0.97, target interpolation 0.01,
batch size 1024, update ratio 0.25, and learning rates 1e-4 are ManiSkill-based
starting settings, not tuned DexJoCo results. All six tasks use seed 42 and
1,600,000 training transitions. Run different seeds through Hydra overrides,
for example `seed=0`, and record each task's runtime dependencies and evaluation
metrics when comparing against ResFiT.

Configuration checks do not require a simulator:

```bash
python -m unittest discover -s tests -p 'test_dawn*.py' -v
```

## Validation

Validated on 2026-11-12 with Python 3.11.16, PyTorch 2.7.0+cu128,
Hydra 1.3.2, and the locally installed DexJoCo, with CUDA hidden
(`CUDA_VISIBLE_DEVICES=""`). The full unittest suite ran 48 tests: 43 passed
and five CUDA tests were skipped.

A native Fold Glasses image-BC checkpoint was also used for a CPU simulator
smoke run with seed 42:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python script/run.py \
  --config-dir=cfg/dexjoco/finetune/fold_glasses \
  --config-name=ft_dawn_unet_img \
  device=cpu wandb=null run_eval=false env.n_envs=1 \
  train.num_train_steps=4 train.buffer_warmup_steps=2 \
  train.training_freq=1 train.batch_size=2 train.utd_ratio=1 \
  train.eval_freq=0 train.save_freq=0 train.log_freq=1 \
  logdir=/tmp/dice-rl-dawn-smoke
```

This completed four simulator transitions, three SAC updates, and checkpoint
saving. Reloading that checkpoint with `mode=eval`, one evaluation worker,
one episode per policy, and `env.max_episode_steps=4` completed for both DAWN
and BC. Both returned 0 with success rate 0 over the four-step limit. This
checks execution and checkpoint compatibility; it does not measure learning
performance. On a machine without an EGL renderer, set both OpenGL variables
above to `osmesa`.
