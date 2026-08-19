#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <experiment_id>" >&2
    exit 2
fi

experiment_id="$1"
repo_root="$(git rev-parse --show-toplevel)"
experiment_dir="${repo_root}/results/${experiment_id}"

if [[ -e "${experiment_dir}" ]]; then
    echo "Refusing to overwrite existing experiment: ${experiment_dir}" >&2
    exit 1
fi

mkdir -p \
    "${experiment_dir}/analysis" \
    "${experiment_dir}/checkpoints" \
    "${experiment_dir}/evaluation" \
    "${experiment_dir}/provenance/dataset"

git -C "${repo_root}" rev-parse HEAD > "${experiment_dir}/provenance/git_commit.txt"
git -C "${repo_root}" status --short --branch > "${experiment_dir}/provenance/git_status.txt"
git -C "${repo_root}" diff --binary > "${experiment_dir}/provenance/worktree.patch"
git -C "${repo_root}" diff --binary main...HEAD > "${experiment_dir}/provenance/iteration_changes.patch"
git -C "${repo_root}" log --oneline main..HEAD > "${experiment_dir}/provenance/iteration_commits.txt"

if command -v conda >/dev/null 2>&1; then
    conda env export > "${experiment_dir}/provenance/conda_environment.yml"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -q > "${experiment_dir}/provenance/nvidia_smi.txt"
fi

printf '# %s\n\nStatus: initialized\n' "${experiment_id}" > "${experiment_dir}/analysis/summary.md"
printf 'experiment_id=%s\ngit_commit=%s\n' \
    "${experiment_id}" "$(git -C "${repo_root}" rev-parse HEAD)" \
    > "${experiment_dir}/provenance/config.env"

echo "Initialized ${experiment_dir}"
