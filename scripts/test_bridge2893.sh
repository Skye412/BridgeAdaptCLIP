#!/usr/bin/env bash
set -euo pipefail

device="${DEVICE:-0}"
data_root="${BRIDGE2893_ROOT:-/home/skye/data/Skye/databases/Bridge2893_split_seed42}"
checkpoint="${CHECKPOINT:-./adaptclip_checkpoints/12_4_128_train_on_mvtec_3adapters_batch8/epoch_15.pth}"
save_dir="${SAVE_DIR:-./results/bridge2893_official_mvtec_baseline}"
shots="${SHOTS:-0}"

for shot in ${shots}; do
    if [[ "${shot}" -eq 0 ]]; then
        seeds="10"
    else
        seeds="10 20 30"
    fi

    for seed in ${seeds}; do
        CUDA_VISIBLE_DEVICES="${device}" python test.py \
            --dataset bridge2893 \
            --test_data_path "${data_root}/test" \
            --seed "${seed}" \
            --k_shots "${shot}" \
            --checkpoint_path "${checkpoint}" \
            --save_path "${save_dir}" \
            --features_list 6 12 18 24 \
            --image_size 518 \
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
            --pixel_thresholds 2048 \
            --pro_thresholds 256
    done
done
