#!/usr/bin/env bash
set -euo pipefail
name="${ABLATION_NAME:?Set ABLATION_NAME}"
preserve="${POSITIVE_PRESERVE_WEIGHT:-0.05}"
ranking="${BROAD_RANKING_WEIGHT:-0.01}"
device="${DEVICE:-0}"
data_root="${BRIDGE2893_ROOT:-/home/skye/data/Skye/databases/Bridge2893_split_seed42}"
result_root="${RESULT_ROOT:-/home/skye/data/Skye/AdaptCLIP/results}"
row0="${ROW0_CHECKPOINT:-${result_root}/bridge2893_original_adaptclip_native1024_eval_v005_b0c6f4d/checkpoints/epoch_14.pth}"
fine="${FINE_CHECKPOINT:-${result_root}/bridge2893_bridgeadaptclip_v13_signed_correction_9da68fa/checkpoints/epoch_3.pth}"
experiment="${result_root}/${name}"
mkdir -p "${experiment}"/{checkpoints,validation,evaluation,provenance}
python - <<PY > "${experiment}/provenance/config.json"
import json
print(json.dumps({"paper_module":"BSC", "ablation":"${name}",
 "fine_checkpoint":"${fine}", "positive_preserve_weight":${preserve},
 "negative_only_ranking_weight":${ranking}, "seed":10, "epochs":15}, indent=2))
PY
CUDA_VISIBLE_DEVICES="${device}" python train_bridgeadaptclip_v20.py \
  --model_name "${name}" --train_data_path "${data_root}/train" \
  --save_path "${experiment}/checkpoints" --row0_checkpoint_path "${row0}" \
  --fine_checkpoint_path "${fine}" --model_input_size 518 \
  --structural_input_size 1024 --fusion_channels 128 --structural_channels 128 \
  --broad_channels 128 --strip_kernel 5 --physical_batch_size 4 \
  --gradient_accumulation_steps 2 --effective_batch_size 8 --epochs 15 \
  --learning_rate 0.0003 --broad_gate_loss_weight 0.1 \
  --positive_preserve_loss_weight "${preserve}" \
  --broad_ranking_loss_weight "${ranking}" --hard_positive_count 256 \
  --hard_negative_count 256 --seed 10 --sigma 4 --amp
for epoch in $(seq 1 15); do
  out="${experiment}/validation/epoch_${epoch}"; mkdir -p "${out}"
  CUDA_VISIBLE_DEVICES="${device}" python test_bridgeadaptclip_v20.py \
    --model_name "${name}" --test_data_path "${data_root}/val" \
    --checkpoint_path "${experiment}/checkpoints/epoch_${epoch}.pth" \
    --row0_checkpoint_path "${row0}" --fine_checkpoint_path "${fine}" \
    --save_path "${out}" --model_input_size 518 --structural_input_size 1024 \
    --metric_resolution 1024 --batch_size 2 --seed 10 --bridge_class_metrics --amp
done
python tools/select_bridge_checkpoint.py --validation_root "${experiment}/validation" \
  --checkpoint_root "${experiment}/checkpoints" --k_shots 0 --metric_resolution 1024 \
  --output "${experiment}/validation/selection.json"
best=$(python -c "import json; print(json.load(open('${experiment}/validation/selection.json'))['best_checkpoint'])")
CUDA_VISIBLE_DEVICES="${device}" python test_bridgeadaptclip_v20.py \
  --model_name "${name}" --test_data_path "${data_root}/test" \
  --checkpoint_path "${best}" --row0_checkpoint_path "${row0}" \
  --fine_checkpoint_path "${fine}" --save_path "${experiment}/evaluation" \
  --model_input_size 518 --structural_input_size 1024 --metric_resolution 1024 \
  --batch_size 2 --seed 10 --bridge_class_metrics --amp

