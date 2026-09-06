#!/usr/bin/env bash
set -euo pipefail

name="${ABLATION_NAME:?Set ABLATION_NAME}"
variant="${STRUCTURAL_VARIANT:-strip}"
gate_weight="${GATE_LOSS_WEIGHT:-0.1}"
preserve_weight="${PRESERVE_LOSS_WEIGHT:-0.01}"
device="${DEVICE:-0}"
data_root="${BRIDGE2893_ROOT:-/home/skye/data/Skye/databases/Bridge2893_split_seed42}"
result_root="${RESULT_ROOT:-/home/skye/data/Skye/AdaptCLIP/results}"
row0="${ROW0_CHECKPOINT:-${result_root}/bridge2893_original_adaptclip_native1024_eval_v005_b0c6f4d/checkpoints/epoch_14.pth}"
initial_state="${FINE_INITIAL_STATE:?Set FINE_INITIAL_STATE}"
experiment="${result_root}/${name}"
mkdir -p "${experiment}"/{checkpoints,validation,evaluation,provenance}

python - <<PY > "${experiment}/provenance/config.json"
import json
print(json.dumps({
 "paper_model":"DeRCLIP-F ablation", "ablation":"${name}",
 "structural_variant":"${variant}", "gate_loss_weight":${gate_weight},
 "preserve_loss_weight":${preserve_weight}, "signed_loss_weight":0.05,
 "seed":10, "epochs":15, "physical_batch":4, "accumulation":2,
 "effective_batch":8, "selection":"validation native-1024 overall P-AP"
}, indent=2))
PY

CUDA_VISIBLE_DEVICES="${device}" python train_bridgeadaptclip_v11.py \
  --model_name "${name}" --checkpoint_state_key bridgeadaptclip_v13 \
  --train_data_path "${data_root}/train" --save_path "${experiment}/checkpoints" \
  --row0_checkpoint_path "${row0}" --initial_state_path "${initial_state}" \
  --features_list 6 12 18 24 --model_input_size 518 \
  --structural_input_size 1024 --physical_batch_size 4 \
  --gradient_accumulation_steps 2 --effective_batch_size 8 --epochs 15 \
  --new_module_learning_rate 0.0003 --residual_l1_weight 0 \
  --gate_loss_weight "${gate_weight}" --preserve_loss_weight "${preserve_weight}" \
  --signed_correction_loss_weight 0.05 --seed 10 --n_ctx 12 \
  --vl_reduction 4 --fusion_channels 128 --structural_channels 128 \
  --strip_kernel 5 --structural_variant "${variant}" \
  --focal_alpha 0.75 --focal_gamma 2 --sigma 4 --amp

for epoch in $(seq 1 15); do
  out="${experiment}/validation/epoch_${epoch}"
  mkdir -p "${out}"
  CUDA_VISIBLE_DEVICES="${device}" python test_bridgeadaptclip_v11.py \
    --model_name "${name}" --checkpoint_state_key bridgeadaptclip_v13 \
    --test_data_path "${data_root}/val" \
    --checkpoint_path "${experiment}/checkpoints/epoch_${epoch}.pth" \
    --row0_checkpoint_path "${row0}" --save_path "${out}" \
    --model_input_size 518 --structural_input_size 1024 \
    --metric_resolution 1024 --structural_variant "${variant}" \
    --reference_count 0 --batch_size 2 --seed 10 --bridge_class_metrics --amp
done
python tools/select_bridge_checkpoint.py \
  --validation_root "${experiment}/validation" \
  --checkpoint_root "${experiment}/checkpoints" --k_shots 0 \
  --metric_resolution 1024 --output "${experiment}/validation/selection.json"
best=$(python -c "import json; print(json.load(open('${experiment}/validation/selection.json'))['best_checkpoint'])")
CUDA_VISIBLE_DEVICES="${device}" python test_bridgeadaptclip_v11.py \
  --model_name "${name}" --checkpoint_state_key bridgeadaptclip_v13 \
  --test_data_path "${data_root}/test" --checkpoint_path "${best}" \
  --row0_checkpoint_path "${row0}" --save_path "${experiment}/evaluation" \
  --model_input_size 518 --structural_input_size 1024 --metric_resolution 1024 \
  --structural_variant "${variant}" --reference_count 0 --batch_size 2 \
  --seed 10 --bridge_class_metrics --amp

