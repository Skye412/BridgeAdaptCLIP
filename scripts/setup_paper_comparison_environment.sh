#!/usr/bin/env bash
set -euo pipefail

checkout=/home/skye/data/Skye/AdaptCLIP_paper_4cab9e4
if [[ ! -d "${checkout}/.git" ]]; then
  git clone --branch feature/high-res-branch --single-branch \
    https://github.com/Skye412/BridgeAdaptCLIP.git "${checkout}"
fi
git -C "${checkout}" fetch origin feature/high-res-branch
git -C "${checkout}" checkout 4cab9e4

source /home/skye/miniconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx bridgecomparisons; then
  conda create -y -n bridgecomparisons --clone adaptclip
fi
conda activate bridgecomparisons
python -m pip install -r "${checkout}/requirements-comparisons.txt"

mkdir -p /home/skye/data/Skye/third_party
upstream=/home/skye/data/Skye/third_party/AnomalyCLIP
if [[ ! -d "${upstream}/.git" ]]; then
  git clone https://github.com/zqhang/AnomalyCLIP.git "${upstream}"
fi
git -C "${upstream}" fetch origin
git -C "${upstream}" checkout 3911738c0867544f545a076ad78f3f11d9ecbfdf

cd "${checkout}"
python - <<'PY'
import torch
import transformers
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('cuda', torch.cuda.is_available())
PY
