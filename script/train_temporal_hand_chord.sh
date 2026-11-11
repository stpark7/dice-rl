#!/usr/bin/env bash
# Run the first Temporal K9 codec experiment on both DexJoCo tasks, sequentially.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${DICE_RL_PYTHON:-python}"
experiment_root="${1:-$repo_root/log_dir/temporal-hand-chord/$(date +%Y-%m-%d_%H-%M-%S)}"
mkdir -p "$experiment_root"
experiment_root="$(cd "$experiment_root" && pwd)"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export DICE_RL_DATA_DIR="${DICE_RL_DATA_DIR:-$repo_root/data_dir}"
export DICE_RL_LOG_DIR="${DICE_RL_LOG_DIR:-$repo_root/log_dir}"

printf '%s\n' "$$" > "$experiment_root/launcher.pid"
printf 'running %s\n' "$(date -Is)" > "$experiment_root/status"
trap 'result=$?; if (( result == 0 )); then state=completed; else state=failed; fi; printf "%s exit=%s %s\n" "$state" "$result" "$(date -Is)" > "$experiment_root/status"' EXIT

for task in pick_bucket fold_glasses; do
    output_dir="$experiment_root/$task"
    mkdir -p "$output_dir"
    command=(
        "$python_bin" script/run.py
        --config-name=pre_chunk_hand_chord
        "--config-dir=$repo_root/cfg/dexjoco/hand_chord/pretrain/$task"
        codec=temporal_chunk latent_dim=9 seed=42 device=cuda:0
        train_dataset.split_seed=0 train_dataset.max_n_episodes=100
        train.n_steps=20000 train.batch_size=128
        train.val_freq=500 train.save_model_freq=1000
        wandb=null "hydra.run.dir=$output_dir"
    )
    printf '%q ' "${command[@]}" > "$output_dir/command.sh"
    printf '\n' >> "$output_dir/command.sh"
    printf 'running %s\n' "$(date -Is)" > "$output_dir/status"
    printf 'Training %s; log: %s\n' "$task" "$output_dir/train.log"
    if "${command[@]}" > "$output_dir/train.log" 2>&1; then
        printf 'completed %s\n' "$(date -Is)" > "$output_dir/status"
    else
        result=$?
        printf 'failed exit=%s %s\n' "$result" "$(date -Is)" > "$output_dir/status"
        exit "$result"
    fi
done
