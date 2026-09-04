#!/usr/bin/env bash
set -euo pipefail

device="${DEVICE:-0}"
data_root="${BRIDGE2893_ROOT:-/home/skye/data/Skye/databases/Bridge2893_split_seed42}"
output_root="${OUTPUT_ROOT:?Set OUTPUT_ROOT for v2.1 diagnostics}"
row0_checkpoint="${ROW0_CHECKPOINT:?Set ROW0_CHECKPOINT}"
fine_checkpoint="${FINE_CHECKPOINT:?Set FINE_CHECKPOINT to v2.1-Fine Epoch 9}"

mkdir -p "${output_root}/fine_test"
CUDA_VISIBLE_DEVICES="${device}" python test_bridgeadaptclip_v11.py \
  --model_name BridgeAdaptCLIP-v2.1-Fine \
  --checkpoint_state_key bridgeadaptclip_v21_fine \
  --test_data_path "${data_root}/test" \
  --checkpoint_path "${fine_checkpoint}" \
  --row0_checkpoint_path "${row0_checkpoint}" \
  --save_path "${output_root}/fine_test" --features_list 6 12 18 24 \
  --model_input_size 518 --structural_input_size 1024 \
  --metric_resolution 1024 --reference_count 0 --batch_size 2 \
  --seed 10 --bridge_class_metrics --amp

OUTPUT_ROOT="${output_root}/level_ablation_val" \
ROW0_CHECKPOINT="${row0_checkpoint}" \
FINE_CHECKPOINT="${fine_checkpoint}" \
DEVICE="${device}" BRIDGE2893_ROOT="${data_root}" \
bash scripts/diagnose_bridgeadaptclip_v21_levels.sh

printf 'Status: complete\nFine checkpoint: %s\n' "${fine_checkpoint}" \
  > "${output_root}/STATUS.txt"
