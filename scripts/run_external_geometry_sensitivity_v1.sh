#!/usr/bin/env bash
set -euo pipefail

scope="${1:-all}"
device="${DEVICE:-0}"
database_root="${DATABASE_ROOT:-/home/skye/data/Skye/databases}"
commit="$(git rev-parse --short HEAD)"
output_root="${OUTPUT_ROOT:-/home/skye/data/Skye/AdaptCLIP/results/external_geometry_sensitivity_v1_${commit}}"
max_images_args=()
if [[ -n "${MAX_IMAGES:-}" ]]; then max_images_args=(--max_images "${MAX_IMAGES}"); fi

row0="/home/skye/data/Skye/AdaptCLIP/results/bridge2893_original_adaptclip_supervised_v004_82c2be3/checkpoints/epoch_14.pth"
v13_fine="/home/skye/data/Skye/AdaptCLIP/results/bridge2893_bridgeadaptclip_v13_signed_correction_9da68fa/checkpoints/epoch_3.pth"
v20_broad="/home/skye/data/Skye/AdaptCLIP/results/bridge2893_bridgeadaptclip_v20_decoupled_broad_fine_26f75de/checkpoints/epoch_15.pth"
v21_fine="/home/skye/data/Skye/AdaptCLIP/results/bridge2893_bridgeadaptclip_v21_fine_multilevel_4c44b8f/checkpoints/epoch_9.pth"
v21_broad="/home/skye/data/Skye/AdaptCLIP/results/bridge2893_bridgeadaptclip_v21_multilevel_broad_dad4806/checkpoints/epoch_11.pth"

run_crack_model() {
  local dataset="$1" protocol="$2" model="$3"; shift 3
  local output_dir="${output_root}/${dataset}/${protocol}/${model}"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${device}" python evaluate_crack_geometry_sensitivity.py \
    --dataset_name "${dataset}" --dataset_root "${database_root}/${dataset}" \
    --geometry_protocol "${protocol}" --model "${model}" \
    --output_dir "${output_dir}" --row0_checkpoint "${row0}" \
    --tile_size 1024 --histogram_bins 65536 --binary_threshold 0.5 \
    --tolerance 3 --min_component_pixels 10 --seed 10 --amp \
    "${max_images_args[@]}" "$@"
}

run_crack() {
  local dataset protocol
  for dataset in CamCrack789 Crack500; do
    for protocol in \
      current_top_left_pad \
      symmetric_pad_native_scale \
      fit_long_side_1024; do
      run_crack_model "${dataset}" "${protocol}" row0
      run_crack_model "${dataset}" "${protocol}" fine13 \
        --fine_checkpoint "${v13_fine}"
      run_crack_model "${dataset}" "${protocol}" v20 \
        --fine_checkpoint "${v13_fine}" --broad_checkpoint "${v20_broad}"
      run_crack_model "${dataset}" "${protocol}" v21 \
        --fine_checkpoint "${v21_fine}" --broad_checkpoint "${v21_broad}"
    done
  done
}

run_valid_core_model() {
  local model="$1"; shift
  local output_dir="${output_root}/DACL10K/valid_core_128_halo/${model}"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${device}" python evaluate_dacl10k_valid_core.py \
    --model "${model}" --dataset_root "${database_root}/dacl10k-DatasetNinja" \
    --output_dir "${output_dir}" --row0_checkpoint "${row0}" \
    --tile_size 1024 --halo 128 --tile_batch_size 1 \
    --histogram_bins 65536 --save_every 5 --seed 10 --amp --resume \
    "${max_images_args[@]}" "$@"
}

run_dacl() {
  local diagnostic_dir="${output_root}/DACL10K/current_hann_geometry/row0"
  mkdir -p "${diagnostic_dir}"
  CUDA_VISIBLE_DEVICES="${device}" python diagnose_dacl10k_external.py \
    --model row0 --dataset_root "${database_root}/dacl10k-DatasetNinja" \
    --output_dir "${diagnostic_dir}" --row0_checkpoint "${row0}" \
    --tile_size 1024 --stride 768 --edge_width 128 --tile_batch_size 1 \
    --histogram_bins 65536 --seed 10 --amp "${max_images_args[@]}"
  run_valid_core_model row0
  run_valid_core_model v20 \
    --fine_checkpoint "${v13_fine}" --broad_checkpoint "${v20_broad}"
  run_valid_core_model v21 \
    --fine_checkpoint "${v21_fine}" --broad_checkpoint "${v21_broad}"
}

case "${scope}" in
  crack) run_crack ;;
  dacl) run_dacl ;;
  all) run_crack; run_dacl ;;
  *) echo "Usage: $0 {crack|dacl|all}" >&2; exit 2 ;;
esac
