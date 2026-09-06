#!/usr/bin/env bash
set -euo pipefail
data_root="${BRIDGE2893_ROOT:-/home/skye/data/Skye/databases/Bridge2893_split_seed42}"
result_root="${RESULT_ROOT:-/home/skye/data/Skye/AdaptCLIP/results}"
upstream="${ANOMALYCLIP_ROOT:?Set ANOMALYCLIP_ROOT}"
experiment="${result_root}/bridge2893_anomalyclip_bd_comparison"
mkdir -p "${experiment}"/{checkpoints,evaluation,provenance}
CUDA_VISIBLE_DEVICES="${DEVICE:-0}" python train_anomalyclip_bd.py \
  --upstream_root "${upstream}" --train_data_path "${data_root}/train" \
  --val_data_path "${data_root}/val" --output_dir "${experiment}/checkpoints" \
  --epochs 45 --batch_size 8 --seed 10 --resume
threshold=$(python -c "import torch; print(torch.load('${experiment}/checkpoints/best.pth',map_location='cpu')['validation_metrics']['P-F1max-threshold'])")
CUDA_VISIBLE_DEVICES="${DEVICE:-0}" python test_anomalyclip_bd.py \
  --upstream_root "${upstream}" --test_data_path "${data_root}/test" \
  --checkpoint "${experiment}/checkpoints/best.pth" \
  --output_dir "${experiment}/evaluation" --val_threshold "${threshold}"

