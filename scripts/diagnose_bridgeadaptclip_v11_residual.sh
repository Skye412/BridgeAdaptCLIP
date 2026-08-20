#!/usr/bin/env bash
set -euo pipefail

device="${DEVICE:-0}"
data_root="${BRIDGE2893_ROOT:-/home/skye/data/Skye/databases/Bridge2893_split_seed42}"
diagnostic_dir="${DIAGNOSTIC_DIR:?Set DIAGNOSTIC_DIR to an initialized diagnostic directory}"
checkpoint="${CHECKPOINT:?Set CHECKPOINT to v1.1 validation-selected Epoch 2}"
row0_checkpoint="${ROW0_CHECKPOINT:?Set ROW0_CHECKPOINT to Protocol-v2 Row0 Epoch 14}"

for split in val test; do
    mkdir -p "${diagnostic_dir}/${split}"
    CUDA_VISIBLE_DEVICES="${device}" python diagnose_bridgeadaptclip_v11_residual.py \
        --data_path "${data_root}/${split}" \
        --split_name "${split}" \
        --decision_split val \
        --checkpoint_path "${checkpoint}" \
        --row0_checkpoint_path "${row0_checkpoint}" \
        --save_path "${diagnostic_dir}/${split}" \
        --model_input_size 518 \
        --structural_input_size 1024 \
        --metric_resolution 1024 \
        --batch_size 2 \
        --samples_per_image 4096 \
        --sampling_seed 42 \
        --seed 10 \
        --amp
done
