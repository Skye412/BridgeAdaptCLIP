#!/usr/bin/env bash
set -euo pipefail

device="${DEVICE:-0}"
data_root="${BRIDGE2893_ROOT:-/home/skye/data/Skye/databases/Bridge2893_split_seed42}"
output_root="${OUTPUT_ROOT:?Set OUTPUT_ROOT for v2.1 level diagnostics}"
row0_checkpoint="${ROW0_CHECKPOINT:?Set ROW0_CHECKPOINT}"
fine_checkpoint="${FINE_CHECKPOINT:?Set FINE_CHECKPOINT to v2.1-Fine Epoch 9}"

run_variant() {
  local name="$1"; shift
  local output="${output_root}/${name}"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${device}" python test_bridgeadaptclip_v11.py \
    --model_name "BridgeAdaptCLIP-v2.1-Fine-${name}" \
    --checkpoint_state_key bridgeadaptclip_v21_fine \
    --test_data_path "${data_root}/val" \
    --checkpoint_path "${fine_checkpoint}" \
    --row0_checkpoint_path "${row0_checkpoint}" \
    --save_path "${output}" --features_list 6 12 18 24 \
    --model_input_size 518 --structural_input_size 1024 \
    --metric_resolution 1024 --reference_count 0 --batch_size 2 \
    --seed 10 --bridge_class_metrics --amp \
    --active_shallow_levels "$@"
}

run_variant all 6 12 18
run_variant without_6 12 18
run_variant without_12 6 18
run_variant without_18 6 12
run_variant only_6 6
run_variant only_12 12
run_variant only_18 18
