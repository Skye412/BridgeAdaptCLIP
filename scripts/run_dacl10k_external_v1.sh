#!/usr/bin/env bash
set -euo pipefail

model="${1:-all}"
device="${DEVICE:-0}"
dataset_root="${DACL10K_ROOT:-/home/skye/data/Skye/databases/dacl10k-DatasetNinja}"
commit="$(git rev-parse --short HEAD)"
experiment_root="${EXPERIMENT_ROOT:-/home/skye/data/Skye/AdaptCLIP/results/dacl10k_external_eval_v1_${commit}}"
max_images_args=()
if [[ -n "${MAX_IMAGES:-}" ]]; then max_images_args=(--max_images "${MAX_IMAGES}"); fi

row0="/home/skye/data/Skye/AdaptCLIP/results/bridge2893_original_adaptclip_supervised_v004_82c2be3/checkpoints/epoch_14.pth"
v13_fine="/home/skye/data/Skye/AdaptCLIP/results/bridge2893_bridgeadaptclip_v13_signed_correction_9da68fa/checkpoints/epoch_3.pth"
v20_broad="/home/skye/data/Skye/AdaptCLIP/results/bridge2893_bridgeadaptclip_v20_decoupled_broad_fine_26f75de/checkpoints/epoch_15.pth"
v21_fine="/home/skye/data/Skye/AdaptCLIP/results/bridge2893_bridgeadaptclip_v21_fine_multilevel_4c44b8f/checkpoints/epoch_9.pth"
v21_broad="/home/skye/data/Skye/AdaptCLIP/results/bridge2893_bridgeadaptclip_v21_multilevel_broad_dad4806/checkpoints/epoch_11.pth"

run_model() {
  local name="$1"; shift
  mkdir -p "${experiment_root}/${name}"
  CUDA_VISIBLE_DEVICES="${device}" python evaluate_dacl10k_external.py \
    --model "${name}" --dataset_root "${dataset_root}" \
    --output_dir "${experiment_root}/${name}" --row0_checkpoint "${row0}" \
    --tile_size 1024 --stride 768 --tile_batch_size 1 \
    --histogram_bins 65536 --save_every 5 --seed 10 --amp --resume \
    "${max_images_args[@]}" "$@"
}

case "${model}" in
  row0) run_model row0 ;;
  v20) run_model v20 --fine_checkpoint "${v13_fine}" --fine_state_key bridgeadaptclip_v13 --broad_checkpoint "${v20_broad}" ;;
  v21) run_model v21 --fine_checkpoint "${v21_fine}" --fine_state_key bridgeadaptclip_v21_fine --broad_checkpoint "${v21_broad}" --broad_state_key bridgeadaptclip_v21 ;;
  all)
    run_model row0
    run_model v20 --fine_checkpoint "${v13_fine}" --fine_state_key bridgeadaptclip_v13 --broad_checkpoint "${v20_broad}"
    run_model v21 --fine_checkpoint "${v21_fine}" --fine_state_key bridgeadaptclip_v21_fine --broad_checkpoint "${v21_broad}" --broad_state_key bridgeadaptclip_v21
    ;;
  *) echo "Usage: $0 {row0|v20|v21|all}" >&2; exit 2 ;;
esac
