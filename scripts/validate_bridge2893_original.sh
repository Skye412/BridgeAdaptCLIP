#!/usr/bin/env bash
set -euo pipefail

device="${DEVICE:-0}"
data_root="${BRIDGE2893_ROOT:-/home/skye/data/Skye/databases/Bridge2893_split_seed42}"
experiment_dir="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to the training experiment directory}"
epochs="${EPOCHS:-15}"
seeds="${VAL_SEEDS:-10 20 30}"

mkdir -p "${experiment_dir}/validation"

for epoch in $(seq 1 "${epochs}"); do
    checkpoint="${experiment_dir}/checkpoints/epoch_${epoch}.pth"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "Missing checkpoint: ${checkpoint}" >&2
        exit 1
    fi
    epoch_dir="${experiment_dir}/validation/epoch_${epoch}"
    mkdir -p "${epoch_dir}"

    for seed in ${seeds}; do
        CUDA_VISIBLE_DEVICES="${device}" python test.py \
            --dataset bridge2893 \
            --test_data_path "${data_root}/val" \
            --checkpoint_path "${checkpoint}" \
            --save_path "${epoch_dir}" \
            --seed "${seed}" \
            --k_shots 1 \
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
            --pixel_thresholds 2048
    done
done

python tools/select_bridge_checkpoint.py \
    --validation_root "${experiment_dir}/validation" \
    --checkpoint_root "${experiment_dir}/checkpoints" \
    --output "${experiment_dir}/validation/selection.json"
