#!/usr/bin/env bash
set -euo pipefail
device="${DEVICE:-0}"
data_root="${BRIDGE2893_ROOT:-/home/skye/data/Skye/databases/Bridge2893_split_seed42}"
experiment_dir="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to an initialized v2.0 directory}"
row0_checkpoint="${ROW0_CHECKPOINT:?Set ROW0_CHECKPOINT to Protocol-v2 Row0 Epoch 14}"
fine_checkpoint="${FINE_CHECKPOINT:?Set FINE_CHECKPOINT to validation-selected v1.3 Epoch 3}"
epochs="${EPOCHS:-15}"
mkdir -p "${experiment_dir}/checkpoints"
CUDA_VISIBLE_DEVICES="${device}" python train_bridgeadaptclip_v20.py \
  --train_data_path "${data_root}/train" --save_path "${experiment_dir}/checkpoints" \
  --row0_checkpoint_path "${row0_checkpoint}" --fine_checkpoint_path "${fine_checkpoint}" \
  --model_input_size 518 --structural_input_size 1024 --fusion_channels 128 \
  --structural_channels 128 --broad_channels 128 --strip_kernel 5 \
  --physical_batch_size "${PHYSICAL_BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-2}" \
  --effective_batch_size 8 --epochs "${epochs}" --learning_rate 0.0003 \
  --broad_gate_loss_weight 0.1 --positive_preserve_loss_weight 0.05 \
  --broad_ranking_loss_weight 0.01 --hard_positive_count 256 --hard_negative_count 256 \
  --seed 10 --sigma 4 --max_train_steps "${MAX_TRAIN_STEPS:-0}" --amp
