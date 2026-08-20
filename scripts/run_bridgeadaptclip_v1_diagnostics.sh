#!/usr/bin/env bash
set -euo pipefail

device="${DEVICE:-0}"
data_root="${BRIDGE2893_ROOT:-/home/skye/data/Skye/databases/Bridge2893_split_seed42}"
experiment_dir="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to an initialized diagnostic experiment}"
row0_checkpoint="${ROW0_CHECKPOINT:?Set ROW0_CHECKPOINT to Protocol-v2 Row 0 Epoch 14}"
full_checkpoint="${FULL_CHECKPOINT:?Set FULL_CHECKPOINT to Full-v1 Epoch 4}"

mkdir -p "${experiment_dir}/validation" "${experiment_dir}/evaluation"

CUDA_VISIBLE_DEVICES="${device}" python diagnose_bridgeadaptclip.py \
    --phase validation \
    --data_path "${data_root}/val" \
    --row0_checkpoint "${row0_checkpoint}" \
    --full_checkpoint "${full_checkpoint}" \
    --output_dir "${experiment_dir}/validation" \
    --expected_row0_p_ap 74.10541219833652 \
    --expected_full_p_ap 72.16794240003061 \
    --batch_size 2 \
    --amp

CUDA_VISIBLE_DEVICES="${device}" python diagnose_bridgeadaptclip.py \
    --phase test \
    --data_path "${data_root}/test" \
    --row0_checkpoint "${row0_checkpoint}" \
    --full_checkpoint "${full_checkpoint}" \
    --output_dir "${experiment_dir}/evaluation" \
    --fusion_selection "${experiment_dir}/validation/fusion_selection.json" \
    --expected_row0_p_ap 68.12294 \
    --expected_full_p_ap 63.72997170590384 \
    --batch_size 2 \
    --amp
