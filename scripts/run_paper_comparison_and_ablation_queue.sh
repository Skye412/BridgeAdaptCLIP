#!/usr/bin/env bash
set -euo pipefail

checkout="${CHECKOUT:-/home/skye/data/Skye/AdaptCLIP_paper_a55cfbf}"
result_root="${RESULT_ROOT:-/home/skye/data/Skye/AdaptCLIP/results}"
queue_root="${result_root}/paper_comparisons_ablations_a300db7"
mkdir -p "${queue_root}/status"
exec > >(tee -a "${queue_root}/queue.log") 2>&1

source /home/skye/miniconda3/etc/profile.d/conda.sh
cd "${checkout}"
export RESULT_ROOT="${result_root}"
export BRIDGE2893_ROOT=/home/skye/data/Skye/databases/Bridge2893_split_seed42
export DEVICE=0
export HF_ENDPOINT=https://hf-mirror.com

run_stage() {
  local name="$1"; shift
  if [[ -f "${queue_root}/status/${name}.done" ]]; then
    echo "SKIP completed ${name}"
    return
  fi
  date --iso-8601=seconds > "${queue_root}/status/${name}.started"
  echo "START ${name} $(date --iso-8601=seconds)"
  "$@"
  date --iso-8601=seconds > "${queue_root}/status/${name}.done"
  echo "DONE ${name} $(date --iso-8601=seconds)"
}

# Main-table comparisons use the isolated comparison environment.
conda activate bridgecomparisons
run_stage deeplabv3plus env \
  MODEL=deeplabv3plus_r50 \
  EXPERIMENT_NAME=bridge2893_deeplabv3plus_r50_comparison_a300db7 \
  PHYSICAL_BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=8 \
  bash scripts/run_supervised_comparison.sh
run_stage segformer_b1 env \
  MODEL=segformer_b1 \
  EXPERIMENT_NAME=bridge2893_segformer_b1_comparison_a300db7 \
  PHYSICAL_BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=8 \
  bash scripts/run_supervised_comparison.sh
run_stage anomalyclip_bd env \
  ANOMALYCLIP_ROOT=/home/skye/data/Skye/third_party/AnomalyCLIP \
  bash scripts/run_anomalyclip_bd.sh

# Controlled DeRCLIP ablations use the original frozen model environment.
conda activate adaptclip
initial_state="${queue_root}/fine_shared_seed10_initial_state.pth"
if [[ ! -f "${initial_state}" ]]; then
  python tools/make_fine_ablation_initial_state.py --output "${initial_state}"
fi
export FINE_INITIAL_STATE="${initial_state}"

run_stage A1_semantic_only env \
  ABLATION_NAME=bridge2893_derclip_f_A1_semantic_only_a300db7 \
  STRUCTURAL_VARIANT=semantic_only bash scripts/run_derclip_f_ablation.sh
run_stage A2_square_convolution env \
  ABLATION_NAME=bridge2893_derclip_f_A2_square_convolution_a300db7 \
  STRUCTURAL_VARIANT=square bash scripts/run_derclip_f_ablation.sh
run_stage A3_without_gate_supervision env \
  ABLATION_NAME=bridge2893_derclip_f_A3_no_gate_supervision_a300db7 \
  STRUCTURAL_VARIANT=strip GATE_LOSS_WEIGHT=0 \
  bash scripts/run_derclip_f_ablation.sh
run_stage A4_without_fine_preservation env \
  ABLATION_NAME=bridge2893_derclip_f_A4_no_preservation_a300db7 \
  STRUCTURAL_VARIANT=strip PRESERVE_LOSS_WEIGHT=0 \
  bash scripts/run_derclip_f_ablation.sh
run_stage B1_without_positive_preservation env \
  ABLATION_NAME=bridge2893_derclip_bsc_B1_no_positive_preservation_a300db7 \
  POSITIVE_PRESERVE_WEIGHT=0 BROAD_RANKING_WEIGHT=0.01 \
  bash scripts/run_derclip_bsc_ablation.sh
run_stage B2_without_negative_ranking env \
  ABLATION_NAME=bridge2893_derclip_bsc_B2_no_negative_ranking_a300db7 \
  POSITIVE_PRESERVE_WEIGHT=0.05 BROAD_RANKING_WEIGHT=0 \
  bash scripts/run_derclip_bsc_ablation.sh

date --iso-8601=seconds > "${queue_root}/COMPLETE"
echo "ALL COMPLETE $(date --iso-8601=seconds)"
