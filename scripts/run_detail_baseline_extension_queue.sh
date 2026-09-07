#!/usr/bin/env bash
set -euo pipefail

checkout="${CHECKOUT:-/home/skye/data/Skye/AdaptCLIP_paper_a55cfbf}"
result_root="${RESULT_ROOT:-/home/skye/data/Skye/AdaptCLIP/results}"
primary_root="${result_root}/paper_comparisons_ablations_a300db7"
revision="${EXPERIMENT_REVISION:-a5ef5e0}"
queue_root="${result_root}/paper_detail_baselines_${revision}"
mkdir -p "${queue_root}/status"
exec > >(tee -a "${queue_root}/queue.log") 2>&1

source /home/skye/miniconda3/etc/profile.d/conda.sh
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

echo "WAIT primary queue ${primary_root}/COMPLETE"
while [[ ! -f "${primary_root}/COMPLETE" ]]; do
  sleep 300
done
cd "${checkout}"
echo "PRIMARY COMPLETE; extension starts $(date --iso-8601=seconds)"

# C0: the complete DeRCLIP-F control from exactly the shared random initial state.
conda activate adaptclip
export FINE_INITIAL_STATE="${primary_root}/fine_shared_seed10_initial_state.pth"
run_stage C0_shared_initialization_full env \
  ABLATION_NAME="bridge2893_derclip_f_C0_shared_init_full_${revision}" \
  STRUCTURAL_VARIANT=strip GATE_LOSS_WEIGHT=0.1 PRESERVE_LOSS_WEIGHT=0.01 \
  bash scripts/run_derclip_f_ablation.sh

# Detail-preserving supervised baselines. Physical batch is recorded explicitly;
# accumulation changes gradients, not BatchNorm statistics.
conda activate bridgecomparisons
run_stage unetplusplus_r34 env \
  MODEL=unetplusplus_r34 \
  EXPERIMENT_NAME="bridge2893_unetplusplus_r34_comparison_${revision}" \
  PHYSICAL_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=4 \
  bash scripts/run_supervised_comparison.sh
run_stage hrnetv2_w18 env \
  MODEL=hrnetv2_w18 \
  EXPERIMENT_NAME="bridge2893_hrnetv2_w18_comparison_${revision}" \
  PHYSICAL_BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=4 \
  bash scripts/run_supervised_comparison.sh

date --iso-8601=seconds > "${queue_root}/COMPLETE"
echo "EXTENSION COMPLETE $(date --iso-8601=seconds)"
