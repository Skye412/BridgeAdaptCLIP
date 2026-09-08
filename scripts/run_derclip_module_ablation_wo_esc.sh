#!/usr/bin/env bash
set -euo pipefail

revision="${EXPERIMENT_REVISION:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"
data_root="${BRIDGE2893_ROOT:-/home/skye/data/Skye/databases/Bridge2893_split_seed42}"
result_root="${RESULT_ROOT:-/home/skye/data/Skye/AdaptCLIP/results}"
row0="${ROW0_CHECKPOINT:-${result_root}/bridge2893_original_adaptclip_native1024_eval_v005_b0c6f4d/checkpoints/epoch_14.pth}"
name="bridge2893_derclip_module_M1_wo_esc_zero_bypass_${revision}"
experiment="${result_root}/${name}"
mkdir -p "${experiment}"/{checkpoints,validation,evaluation,provenance}

python - <<PY > "${experiment}/provenance/config.json"
import json
print(json.dumps({
  "paper_ablation": "DeRCLIP w/o ESC -- zero-feature bypass",
  "semantic_base": "formal Row 0 Epoch 14, frozen",
  "esc_instantiated": False,
  "esc_executed": False,
  "bsc_initialization": "from scratch",
  "bsc_inputs": "zero F_joint, Row-0 logits, Row-0 probability",
  "fusion": "Z_final = Z_0 + C_b",
  "optimizer": "Adam", "betas": [0.5, 0.999], "learning_rate": 3e-4,
  "physical_batch": 4, "gradient_accumulation": 2, "effective_batch": 8,
  "epochs": 15, "seed": 10,
  "loss_weights": {"focal": 1, "dice": 1, "broad_gate": 0.1,
                   "positive_preserve": 0.05, "negative_only_ranking": 0.01},
  "selection": "validation native-1024 overall P-AP"
}, indent=2))
PY

CUDA_VISIBLE_DEVICES="${DEVICE:-0}" python train_bridgeadaptclip_v20.py \
  --model_name DeRCLIP-w/o-ESC \
  --checkpoint_state_key bridgeadaptclip_v20_wo_esc \
  --fine_bypass row0_zero_feature \
  --train_data_path "${data_root}/train" --save_path "${experiment}/checkpoints" \
  --row0_checkpoint_path "${row0}" \
  --features_list 6 12 18 24 --model_input_size 518 \
  --structural_input_size 1024 --physical_batch_size 4 \
  --gradient_accumulation_steps 2 --effective_batch_size 8 --epochs 15 \
  --learning_rate 0.0003 --broad_gate_loss_weight 0.1 \
  --positive_preserve_loss_weight 0.05 --broad_ranking_loss_weight 0.01 \
  --seed 10 --n_ctx 12 --vl_reduction 4 --fusion_channels 128 \
  --structural_channels 128 --broad_channels 128 --strip_kernel 5 --amp

for epoch in $(seq 1 15); do
  output="${experiment}/validation/epoch_${epoch}"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${DEVICE:-0}" python test_bridgeadaptclip_v20.py \
    --model_name DeRCLIP-w/o-ESC \
    --checkpoint_state_key bridgeadaptclip_v20_wo_esc \
    --fine_bypass row0_zero_feature \
    --test_data_path "${data_root}/val" \
    --checkpoint_path "${experiment}/checkpoints/epoch_${epoch}.pth" \
    --row0_checkpoint_path "${row0}" --save_path "${output}" \
    --model_input_size 518 --structural_input_size 1024 --metric_resolution 1024 \
    --batch_size 2 --seed 10 --bridge_class_metrics --amp
done

python tools/select_bridge_checkpoint.py \
  --validation_root "${experiment}/validation" \
  --checkpoint_root "${experiment}/checkpoints" --k_shots 0 \
  --metric_resolution 1024 --output "${experiment}/validation/selection.json"
best=$(python -c "import json; print(json.load(open('${experiment}/validation/selection.json'))['best_checkpoint'])")
CUDA_VISIBLE_DEVICES="${DEVICE:-0}" python test_bridgeadaptclip_v20.py \
  --model_name DeRCLIP-w/o-ESC \
  --checkpoint_state_key bridgeadaptclip_v20_wo_esc \
  --fine_bypass row0_zero_feature \
  --test_data_path "${data_root}/test" --checkpoint_path "${best}" \
  --row0_checkpoint_path "${row0}" --save_path "${experiment}/evaluation" \
  --model_input_size 518 --structural_input_size 1024 --metric_resolution 1024 \
  --batch_size 2 --seed 10 --bridge_class_metrics --amp

