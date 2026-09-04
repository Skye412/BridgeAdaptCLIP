#!/usr/bin/env bash
set -euo pipefail

device="${DEVICE:-0}"
data_root="${BRIDGE2893_ROOT:-/home/skye/data/Skye/databases/Bridge2893_split_seed42}"
experiment_dir="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to an initialized results directory}"
epochs="${EPOCHS:-15}"

mkdir -p "${experiment_dir}/checkpoints"

CUDA_VISIBLE_DEVICES="${device}" python train.py \
    --dataset bridge2893 \
    --train_data_path "${data_root}/train" \
    --save_path "${experiment_dir}/checkpoints" \
    --features_list 6 12 18 24 \
    --image_size 518 \
    --batch_size 8 \
    --epoch "${epochs}" \
    --save_freq 1 \
    --print_freq 1 \
    --learning_rate 0.001 \
    --seed 10 \
    --k_shots 1 \
    --n_ctx 12 \
    --vl_reduction 4 \
    --pq_mid_dim 128 \
    --visual_learner \
    --textual_learner \
    --pq_learner \
    --pq_context
