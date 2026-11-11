#!/usr/bin/env bash
# Continue Temporal K9 residual RL on a server, preserving the full reference configs.
set -euo pipefail

if (( $# != 2 )); then
    printf 'Usage: %s CODEC_ROOT RL_OUTPUT_ROOT\n' "$0" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${DICE_RL_PYTHON:-python}"
codec_root="$1"
experiment_root="$2"

# Validate the complete queue before creating or reserving any output artifacts.
for task in pick_bucket fold_glasses; do
    if [[ ! -f "$codec_root/$task/chord.pt" ]]; then
        printf 'Missing Temporal codec checkpoint: %s\n' "$codec_root/$task/chord.pt" >&2
        exit 1
    fi
done
codec_root="$(cd "$codec_root" && pwd)"
mkdir -p "$experiment_root"
experiment_root="$(cd "$experiment_root" && pwd)"
for artifact in status launcher.pid pick_bucket fold_glasses; do
    if [[ -e "$experiment_root/$artifact" || -L "$experiment_root/$artifact" ]]; then
        printf 'Refusing to reuse RL directory with existing artifact: %s\n' "$experiment_root/$artifact" >&2
        exit 1
    fi
done
# Permanent atomic ownership prevents concurrent launches and accidental reruns.
if ! mkdir "$experiment_root/.launcher-reservation" 2>/dev/null; then
    printf 'Refusing to reuse reserved RL directory: %s\n' "$experiment_root" >&2
    exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export DICE_RL_DATA_DIR="${DICE_RL_DATA_DIR:-$repo_root/data_dir}"
export DICE_RL_LOG_DIR="${DICE_RL_LOG_DIR:-$repo_root/log_dir}"
cd "$repo_root"

printf '%s\n' "$$" > "$experiment_root/launcher.pid"
printf 'running %s\n' "$(date -Is)" > "$experiment_root/status"
trap 'result=$?; if (( result == 0 )); then state=completed; else state=failed; fi; printf "%s exit=%s %s\n" "$state" "$result" "$(date -Is)" > "$experiment_root/status"' EXIT

for task in pick_bucket fold_glasses; do
    output_dir="$experiment_root/$task"
    mkdir -p "$output_dir"
    command=(
        "$python_bin" script/run.py
        --config-name=ft_temporal_hand_chord_residual_flow_unet_img
        "--config-dir=$repo_root/cfg/dexjoco/finetune/$task"
        "chord_run_dir=$codec_root" "logdir=$output_dir" "hydra.run.dir=$output_dir"
        device=cuda:0 seed=42
    )
    printf '%q ' "${command[@]}" > "$output_dir/command.sh"
    printf '\n' >> "$output_dir/command.sh"
    printf 'running %s\n' "$(date -Is)" > "$output_dir/status"
    printf 'Training residual RL for %s; log: %s\n' "$task" "$output_dir/train.log"
    if "${command[@]}" > "$output_dir/train.log" 2>&1; then
        printf 'completed %s\n' "$(date -Is)" > "$output_dir/status"
    else
        result=$?
        printf 'failed exit=%s %s\n' "$result" "$(date -Is)" > "$output_dir/status"
        exit "$result"
    fi
done
