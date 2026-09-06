#!/usr/bin/env bash
set -euo pipefail
model="${MODEL:?Set MODEL}"
name="${EXPERIMENT_NAME:?Set EXPERIMENT_NAME}"
data_root="${BRIDGE2893_ROOT:-/home/skye/data/Skye/databases/Bridge2893_split_seed42}"
result_root="${RESULT_ROOT:-/home/skye/data/Skye/AdaptCLIP/results}"
experiment="${result_root}/${name}"
mkdir -p "${experiment}"/{checkpoints,evaluation,provenance}
CUDA_VISIBLE_DEVICES="${DEVICE:-0}" python train_supervised_baseline.py \
  --model "${model}" --train_data_path "${data_root}/train" \
  --val_data_path "${data_root}/val" --output_dir "${experiment}/checkpoints" \
  --epochs 45 --physical_batch_size "${PHYSICAL_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --effective_batch_size 8 --seed 10 --amp --resume
threshold=$(python -c "import torch; print(torch.load('${experiment}/checkpoints/best.pth',map_location='cpu')['validation_metrics']['P-F1max-threshold'])")
CUDA_VISIBLE_DEVICES="${DEVICE:-0}" python test_supervised_baseline.py \
  --test_data_path "${data_root}/test" --checkpoint "${experiment}/checkpoints/best.pth" \
  --output_dir "${experiment}/evaluation" --val_threshold "${threshold}" --amp

