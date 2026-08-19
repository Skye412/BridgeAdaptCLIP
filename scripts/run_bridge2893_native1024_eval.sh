#!/usr/bin/env bash
set -euo pipefail

experiment_dir="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to the v005 evaluation experiment directory}"
source_experiment_dir="${SOURCE_EXPERIMENT_DIR:?Set SOURCE_EXPERIMENT_DIR to the completed v004 experiment directory}"
checkpoint_dir="${source_experiment_dir}/checkpoints"

export CHECKPOINT_DIR="${checkpoint_dir}"
bash scripts/validate_bridge2893_native1024_zero_ref.sh

best_checkpoint="$(python3 -c 'import json, os; p=os.path.join(os.environ["EXPERIMENT_DIR"], "validation", "selection.json"); print(json.load(open(p))["best_checkpoint"])')"
best_epoch="$(python3 -c 'import json, os; p=os.path.join(os.environ["EXPERIMENT_DIR"], "validation", "selection.json"); print(json.load(open(p))["best_epoch"])')"

mkdir -p "${experiment_dir}/checkpoints"
ln "${best_checkpoint}" "${experiment_dir}/checkpoints/epoch_${best_epoch}.pth"
printf '%s\n' "${best_checkpoint}" > "${experiment_dir}/checkpoints/source_checkpoint.txt"

export CHECKPOINT="${best_checkpoint}"
bash scripts/test_bridge2893_native1024_zero_ref.sh
