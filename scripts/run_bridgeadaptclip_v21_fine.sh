#!/usr/bin/env bash
set -euo pipefail

experiment_dir="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR}"
bash scripts/train_bridgeadaptclip_v21_fine.sh
bash scripts/validate_bridgeadaptclip_v21_fine.sh
selected="$(python3 -c 'import json,os; print(json.load(open(os.path.join(os.environ["EXPERIMENT_DIR"],"validation","selection.json")))["best_checkpoint"])')"
printf '# BridgeAdaptCLIP-v2.1-Fine Phase 1\n\nStatus: validation complete; Test not run\nSelected checkpoint: %s\n' \
    "${selected}" > "${experiment_dir}/analysis/summary.md"
