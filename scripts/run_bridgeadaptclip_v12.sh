#!/usr/bin/env bash
set -euo pipefail

experiment_dir="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to the v1.2 experiment}"
bash scripts/train_bridgeadaptclip_v12.sh
bash scripts/validate_bridgeadaptclip_v12.sh

best_checkpoint="$(python3 -c 'import json, os; p=os.path.join(os.environ["EXPERIMENT_DIR"], "validation", "selection.json"); print(json.load(open(p))["best_checkpoint"])')"
export CHECKPOINT="${best_checkpoint}"
bash scripts/test_bridgeadaptclip_v12.sh

export DIAGNOSTIC_DIR="${experiment_dir}/diagnostics"
export CHECKPOINT_STATE_KEY=bridgeadaptclip_v12
bash scripts/diagnose_bridgeadaptclip_v11_residual.sh
