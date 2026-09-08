#!/usr/bin/env bash
set -euo pipefail

checkout="${CHECKOUT:-/home/skye/data/Skye/AdaptCLIP_paper_a55cfbf}"
result_root="${RESULT_ROOT:-/home/skye/data/Skye/AdaptCLIP/results}"
revision="${EXPERIMENT_REVISION:?Set EXPERIMENT_REVISION}"
queue_root="${result_root}/paper_module_ablations_${revision}"
mkdir -p "${queue_root}/status"
exec > >(tee -a "${queue_root}/queue.log") 2>&1
source /home/skye/miniconda3/etc/profile.d/conda.sh
conda activate adaptclip
cd "${checkout}"
export RESULT_ROOT="${result_root}"
export BRIDGE2893_ROOT=/home/skye/data/Skye/databases/Bridge2893_split_seed42
export DEVICE=0

# M2 is the exact Fine Expert consumed by the formal v2.0 model. Register it;
# do not retrain it and do not substitute the shared-initialization C0 run.
python - <<PY > "${queue_root}/M2_wo_BSC_REUSED.json"
import json
print(json.dumps({
  "paper_ablation": "DeRCLIP w/o BSC",
  "action": "reuse; no retraining",
  "checkpoint": "${result_root}/bridge2893_bridgeadaptclip_v13_signed_correction_9da68fa/checkpoints/epoch_3.pth",
  "evaluation": "${result_root}/bridge2893_bridgeadaptclip_v13_signed_correction_9da68fa/evaluation/bridge2893_10seed_0shot_metrics.json",
  "expected_P_AP": 73.41,
  "expected_Crack_P_AP": 39.98,
  "expected_Macro_P_AP": 56.63,
  "explicitly_not_C0": True
}, indent=2))
PY
date --iso-8601=seconds > "${queue_root}/status/M2_wo_BSC_reused.done"

date --iso-8601=seconds > "${queue_root}/status/M1_wo_ESC.started"
EXPERIMENT_REVISION="${revision}" bash scripts/run_derclip_module_ablation_wo_esc.sh
date --iso-8601=seconds > "${queue_root}/status/M1_wo_ESC.done"
date --iso-8601=seconds > "${queue_root}/COMPLETE"
echo "MODULE ABLATIONS COMPLETE $(date --iso-8601=seconds)"
