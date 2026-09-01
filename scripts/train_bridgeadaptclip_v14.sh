#!/usr/bin/env bash
set -euo pipefail

device="${DEVICE:-0}"
data_root="${BRIDGE2893_ROOT:-/home/skye/data/Skye/databases/Bridge2893_split_seed42}"
experiment_dir="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to an initialized v1.4 directory}"
row0_checkpoint="${ROW0_CHECKPOINT:?Set ROW0_CHECKPOINT to Protocol-v2 Row0 Epoch 14}"
epochs="${EPOCHS:-15}"
physical_batch="${PHYSICAL_BATCH_SIZE:-4}"
accumulation="${GRADIENT_ACCUMULATION_STEPS:-2}"
max_train_steps="${MAX_TRAIN_STEPS:-0}"

mkdir -p "${experiment_dir}/checkpoints"
CUDA_VISIBLE_DEVICES="${device}" python train_bridgeadaptclip_v11.py \
    --model_name BridgeAdaptCLIP-v1.4 \
    --checkpoint_state_key bridgeadaptclip_v14 \
    --train_data_path "${data_root}/train" \
    --save_path "${experiment_dir}/checkpoints" \
    --row0_checkpoint_path "${row0_checkpoint}" \
    --features_list 6 12 18 24 \
    --model_input_size 518 \
    --structural_input_size 1024 \
    --physical_batch_size "${physical_batch}" \
    --gradient_accumulation_steps "${accumulation}" \
    --effective_batch_size 8 \
    --epochs "${epochs}" \
    --new_module_learning_rate 0.0003 \
    --residual_l1_weight 0 \
    --gate_loss_weight 0.1 \
    --preserve_loss_weight 0.01 \
    --signed_correction_loss_weight 0 \
    --margin_loss_weight 0.05 \
    --margin 1.0 \
    --seed 10 \
    --n_ctx 12 \
    --vl_reduction 4 \
    --fusion_channels 128 \
    --structural_channels 128 \
    --strip_kernel 5 \
    --focal_alpha 0.75 \
    --focal_gamma 2 \
    --sigma 4 \
    --max_train_steps "${max_train_steps}" \
    --amp
