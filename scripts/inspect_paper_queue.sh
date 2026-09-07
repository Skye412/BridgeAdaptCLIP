#!/usr/bin/env bash
set -u
queue=/home/skye/data/Skye/AdaptCLIP/results/paper_comparisons_ablations_a300db7
echo STATUS
find "${queue}/status" -maxdepth 1 -type f \
  -printf '%f %TY-%Tm-%TdT%TH:%TM:%TS\n' 2>/dev/null | sort
echo PROCESSES
pgrep -af 'run_paper_comparison|train_supervised_baseline|test_supervised_baseline|train_anomalyclip|train_bridgeadaptclip|test_bridgeadaptclip' || true
echo GPU
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo LOG
tail -c 10000 "${queue}/nohup.log" 2>/dev/null | tr '\r' '\n' | tail -45
echo CURVES
for file in /home/skye/data/Skye/AdaptCLIP/results/bridge2893_*comparison_a300db7/checkpoints/validation_curve.json; do
  [[ -f "${file}" ]] || continue
  python - "${file}" <<'PY'
import json
import os
import sys
path = sys.argv[1]
curve = json.load(open(path))
best = max(curve, key=lambda row: row['P-AP'])
print(
    os.path.basename(os.path.dirname(os.path.dirname(path))),
    'epochs', len(curve),
    'latest', curve[-1]['epoch'], round(curve[-1]['P-AP'], 4),
    'best', best['epoch'], round(best['P-AP'], 4),
)
PY
done
