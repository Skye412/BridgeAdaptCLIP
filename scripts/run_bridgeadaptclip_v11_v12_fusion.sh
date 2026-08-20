#!/usr/bin/env bash
set -euo pipefail

device="${DEVICE:-0}"
data_root="${BRIDGE2893_ROOT:-/home/skye/data/Skye/databases/Bridge2893_split_seed42}"
experiment_dir="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to the fusion diagnostic directory}"
row0_checkpoint="${ROW0_CHECKPOINT:?Set ROW0_CHECKPOINT}"
v11_checkpoint="${V11_CHECKPOINT:?Set V11_CHECKPOINT}"
v12_checkpoint="${V12_CHECKPOINT:?Set V12_CHECKPOINT}"

mkdir -p "${experiment_dir}/validation" "${experiment_dir}/evaluation"

CUDA_VISIBLE_DEVICES="${device}" python diagnose_bridgeadaptclip_v11_v12_fusion.py \
    --phase validation \
    --data_path "${data_root}/val" \
    --row0_checkpoint "${row0_checkpoint}" \
    --v11_checkpoint "${v11_checkpoint}" \
    --v12_checkpoint "${v12_checkpoint}" \
    --output_dir "${experiment_dir}/validation" \
    --batch_size 2 \
    --amp

CUDA_VISIBLE_DEVICES="${device}" python diagnose_bridgeadaptclip_v11_v12_fusion.py \
    --phase test \
    --data_path "${data_root}/test" \
    --row0_checkpoint "${row0_checkpoint}" \
    --v11_checkpoint "${v11_checkpoint}" \
    --v12_checkpoint "${v12_checkpoint}" \
    --output_dir "${experiment_dir}/evaluation" \
    --fusion_selection "${experiment_dir}/validation/fusion_selection.json" \
    --batch_size 2 \
    --amp
