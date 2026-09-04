#!/usr/bin/env bash
set -euo pipefail

device="${DEVICE:-0}"
data_root="${BRIDGE2893_ROOT:-/home/skye/data/Skye/databases/Bridge2893_split_seed42}"
experiment_dir="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to the v005 evaluation experiment directory}"
checkpoint="${CHECKPOINT:?Set CHECKPOINT to the validation-selected checkpoint}"

mkdir -p "${experiment_dir}/evaluation"

CUDA_VISIBLE_DEVICES="${device}" python test.py \
    --dataset bridge2893 \
    --test_data_path "${data_root}/test" \
    --checkpoint_path "${checkpoint}" \
    --save_path "${experiment_dir}/evaluation" \
    --seed 10 \
    --k_shots 0 \
    --features_list 6 12 18 24 \
    --model_input_size 518 \
    --metric_resolution 1024 \
    --batch_size 8 \
    --n_ctx 12 \
    --vl_reduction 4 \
    --pq_mid_dim 128 \
    --visual_learner \
    --textual_learner \
    --pq_learner \
    --pq_context \
    --eval_metrics I-AUROC I-AP I-F1max P-AUROC P-AP P-F1max \
    --cpu_eval \
    --bridge_class_metrics \
    --pixel_thresholds 2048
