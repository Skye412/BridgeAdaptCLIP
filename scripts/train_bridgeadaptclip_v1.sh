#!/usr/bin/env bash
set -euo pipefail

device="${DEVICE:-0}"
data_root="${BRIDGE2893_ROOT:-/home/skye/data/Skye/databases/Bridge2893_split_seed42}"
experiment_dir="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to an initialized results directory}"
epochs="${EPOCHS:-15}"
physical_batch="${PHYSICAL_BATCH_SIZE:-4}"
accumulation="${GRADIENT_ACCUMULATION_STEPS:-2}"

mkdir -p "${experiment_dir}/checkpoints"

CUDA_VISIBLE_DEVICES="${device}" python train_bridgeadaptclip.py \
    --train_data_path "${data_root}/train" \
    --save_path "${experiment_dir}/checkpoints" \
    --features_list 6 12 18 24 \
    --model_input_size 518 \
    --structural_input_size 1024 \
    --physical_batch_size "${physical_batch}" \
    --gradient_accumulation_steps "${accumulation}" \
    --effective_batch_size 8 \
    --epochs "${epochs}" \
    --adapter_learning_rate 0.001 \
    --new_module_learning_rate 0.001 \
    --seed 10 \
    --n_ctx 12 \
    --vl_reduction 4 \
    --fusion_channels 128 \
    --structural_channels 128 \
    --strip_kernel 5 \
    --focal_alpha 0.75 \
    --focal_gamma 2 \
    --amp
