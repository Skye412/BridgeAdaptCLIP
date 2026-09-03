#!/usr/bin/env bash
set -euo pipefail
experiment_dir="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR}"
bash scripts/train_bridgeadaptclip_v21.sh
bash scripts/validate_bridgeadaptclip_v21.sh
export CHECKPOINT="$(python3 -c 'import json,os; print(json.load(open(os.path.join(os.environ["EXPERIMENT_DIR"],"validation","selection.json")))["best_checkpoint"])')"
bash scripts/test_bridgeadaptclip_v21.sh
printf '# BridgeAdaptCLIP-v2.1 result\n\nStatus: complete\nFine checkpoint: %s\nBroad checkpoint: %s\n' \
  "${FINE_CHECKPOINT}" "${CHECKPOINT}" > "${experiment_dir}/analysis/summary.md"
