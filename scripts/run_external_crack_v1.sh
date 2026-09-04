#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-all}"
model="${2:-all}"
device="${DEVICE:-0}"
output_root="${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
database_root="${DATABASE_ROOT:-/home/skye/data/Skye/databases}"
max_images_args=()
if [[ -n "${MAX_IMAGES:-}" ]]; then max_images_args=(--max_images "${MAX_IMAGES}"); fi

row0="/home/skye/data/Skye/AdaptCLIP/results/bridge2893_original_adaptclip_supervised_v004_82c2be3/checkpoints/epoch_14.pth"
v13_fine="/home/skye/data/Skye/AdaptCLIP/results/bridge2893_bridgeadaptclip_v13_signed_correction_9da68fa/checkpoints/epoch_3.pth"
v20_broad="/home/skye/data/Skye/AdaptCLIP/results/bridge2893_bridgeadaptclip_v20_decoupled_broad_fine_26f75de/checkpoints/epoch_15.pth"
v21_fine="/home/skye/data/Skye/AdaptCLIP/results/bridge2893_bridgeadaptclip_v21_fine_multilevel_4c44b8f/checkpoints/epoch_9.pth"
v21_broad="/home/skye/data/Skye/AdaptCLIP/results/bridge2893_bridgeadaptclip_v21_multilevel_broad_dad4806/checkpoints/epoch_11.pth"

run_one() {
  local dataset_name="$1" model_name="$2"; shift 2
  local output_dir="${output_root}/${dataset_name}/${model_name}"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${device}" python evaluate_crack_external.py \
    --dataset_name "${dataset_name}" --dataset_root "${database_root}/${dataset_name}" \
    --model "${model_name}" --output_dir "${output_dir}" \
    --row0_checkpoint "${row0}" --tile_size 1024 --stride 768 \
    --histogram_bins 65536 --binary_threshold 0.5 --tolerance 3 \
    --min_component_pixels 10 --seed 10 --amp "${max_images_args[@]}" "$@"
}

run_models() {
  local dataset_name="$1"
  case "${model}" in
    row0) run_one "${dataset_name}" row0 ;;
    fine13) run_one "${dataset_name}" fine13 --fine_checkpoint "${v13_fine}" ;;
    v20) run_one "${dataset_name}" v20 --fine_checkpoint "${v13_fine}" --broad_checkpoint "${v20_broad}" ;;
    v21) run_one "${dataset_name}" v21 --fine_checkpoint "${v21_fine}" --broad_checkpoint "${v21_broad}" ;;
    all)
      run_one "${dataset_name}" row0
      run_one "${dataset_name}" fine13 --fine_checkpoint "${v13_fine}"
      run_one "${dataset_name}" v20 --fine_checkpoint "${v13_fine}" --broad_checkpoint "${v20_broad}"
      run_one "${dataset_name}" v21 --fine_checkpoint "${v21_fine}" --broad_checkpoint "${v21_broad}"
      ;;
    *) echo "Unknown model ${model}" >&2; exit 2 ;;
  esac
}

case "${dataset}" in
  CamCrack789|Crack500) run_models "${dataset}" ;;
  all) run_models CamCrack789; run_models Crack500 ;;
  *) echo "Usage: $0 {CamCrack789|Crack500|all} {row0|fine13|v20|v21|all}" >&2; exit 2 ;;
esac
