#!/usr/bin/env bash
set -euo pipefail
device="${DEVICE:-0}"; data_root="${BRIDGE2893_ROOT:-/home/skye/data/Skye/databases/Bridge2893_split_seed42}"
experiment_dir="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR}"; checkpoint="${CHECKPOINT:?Set CHECKPOINT}"
row0_checkpoint="${ROW0_CHECKPOINT:?Set ROW0_CHECKPOINT}"; fine_checkpoint="${FINE_CHECKPOINT:?Set FINE_CHECKPOINT}"
mkdir -p "${experiment_dir}/evaluation"
CUDA_VISIBLE_DEVICES="${device}" python test_bridgeadaptclip_v20.py \
  --model_name BridgeAdaptCLIP-v2.1 --checkpoint_state_key bridgeadaptclip_v21 \
  --fine_checkpoint_state_key bridgeadaptclip_v21_fine \
  --test_data_path "${data_root}/test" --checkpoint_path "${checkpoint}" \
  --row0_checkpoint_path "${row0_checkpoint}" --fine_checkpoint_path "${fine_checkpoint}" \
  --save_path "${experiment_dir}/evaluation" --features_list 6 12 18 24 \
  --model_input_size 518 --structural_input_size 1024 --metric_resolution 1024 \
  --batch_size 2 --seed 10 --bridge_class_metrics --amp
