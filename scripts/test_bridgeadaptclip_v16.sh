#!/usr/bin/env bash
set -euo pipefail

device="${DEVICE:-0}"
data_root="${BRIDGE2893_ROOT:-/home/skye/data/Skye/databases/Bridge2893_split_seed42}"
experiment_dir="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to the v1.6 experiment}"
row0_checkpoint="${ROW0_CHECKPOINT:?Set ROW0_CHECKPOINT to Protocol-v2 Row0 Epoch 14}"
checkpoint="${CHECKPOINT:?Set CHECKPOINT to the validation-selected v1.6 checkpoint}"

mkdir -p "${experiment_dir}/evaluation"
CUDA_VISIBLE_DEVICES="${device}" python test_bridgeadaptclip_v11.py \
    --model_name BridgeAdaptCLIP-v1.6 \
    --checkpoint_state_key bridgeadaptclip_v16 \
    --test_data_path "${data_root}/test" \
    --checkpoint_path "${checkpoint}" \
    --row0_checkpoint_path "${row0_checkpoint}" \
    --save_path "${experiment_dir}/evaluation" \
    --model_input_size 518 \
    --structural_input_size 1024 \
    --metric_resolution 1024 \
    --reference_count 0 \
    --batch_size 2 \
    --seed 10 \
    --bridge_class_metrics \
    --amp
