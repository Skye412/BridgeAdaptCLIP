#!/usr/bin/env bash
set -euo pipefail

device="${DEVICE:-0}"
data_root="${BRIDGE2893_ROOT:-/home/skye/data/Skye/databases/Bridge2893_split_seed42}"
experiment_dir="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to the BridgeAdaptCLIP-v1 experiment}"
checkpoint="${CHECKPOINT:?Set CHECKPOINT to the validation-selected checkpoint}"

mkdir -p "${experiment_dir}/evaluation"
CUDA_VISIBLE_DEVICES="${device}" python test_bridgeadaptclip.py \
    --test_data_path "${data_root}/test" \
    --checkpoint_path "${checkpoint}" \
    --save_path "${experiment_dir}/evaluation" \
    --model_input_size 518 \
    --structural_input_size 1024 \
    --metric_resolution 1024 \
    --reference_count 0 \
    --batch_size 2 \
    --seed 10 \
    --bridge_class_metrics \
    --amp
