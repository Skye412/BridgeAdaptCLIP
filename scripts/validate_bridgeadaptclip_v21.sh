#!/usr/bin/env bash
set -euo pipefail
device="${DEVICE:-0}"; data_root="${BRIDGE2893_ROOT:-/home/skye/data/Skye/databases/Bridge2893_split_seed42}"
experiment_dir="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to the v2.1 experiment}"
row0_checkpoint="${ROW0_CHECKPOINT:?Set ROW0_CHECKPOINT}"; fine_checkpoint="${FINE_CHECKPOINT:?Set FINE_CHECKPOINT}"
epochs="${EPOCHS:-15}"; mkdir -p "${experiment_dir}/validation"
for epoch in $(seq 1 "${epochs}"); do
  checkpoint="${experiment_dir}/checkpoints/epoch_${epoch}.pth"; out="${experiment_dir}/validation/epoch_${epoch}"
  [[ -f "${checkpoint}" ]] || { echo "Missing ${checkpoint}" >&2; exit 1; }; mkdir -p "${out}"
  CUDA_VISIBLE_DEVICES="${device}" python test_bridgeadaptclip_v20.py \
    --model_name BridgeAdaptCLIP-v2.1 --checkpoint_state_key bridgeadaptclip_v21 \
    --fine_checkpoint_state_key bridgeadaptclip_v21_fine \
    --test_data_path "${data_root}/val" --checkpoint_path "${checkpoint}" \
    --row0_checkpoint_path "${row0_checkpoint}" --fine_checkpoint_path "${fine_checkpoint}" \
    --save_path "${out}" --features_list 6 12 18 24 --model_input_size 518 \
    --structural_input_size 1024 --metric_resolution 1024 --batch_size 2 \
    --seed 10 --bridge_class_metrics --amp
done
python tools/select_bridge_checkpoint.py --validation_root "${experiment_dir}/validation" \
  --checkpoint_root "${experiment_dir}/checkpoints" --k_shots 0 --metric_resolution 1024 \
  --output "${experiment_dir}/validation/selection.json"
