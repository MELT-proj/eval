#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output="logs/%x.%j.out"
#SBATCH --job-name=melt-text-prior
#
# Runs `melteval text-prior` for one backbone against one frozen set, inside
# the venv named by VENV_PATH (set by the site file). Single GPU, single
# node -- no multi-node case to support, same as run_eval.sbatch.
#
# Not usually invoked directly: use infra/runners/submit_text_prior.sh, which
# sources a site file for VENV_PATH/SBATCH_ARGS and forwards everything below.
#
# Usage (host side, from the repo root):
#   VENV_PATH=/path/to/venv/bin/activate \
#     sbatch [slurm flags] infra/run_text_prior.sbatch <frozen_set_dir> <out.json> [text-prior args...]
#
# Example:
#   sbatch infra/run_text_prior.sbatch \
#     /mnt/scratch-artemis/giuseppe/melt-data/eval-sets/fleurs24-asr-dev \
#     /mnt/scratch-artemis/giuseppe/melt-data/text-prior/llama1b-ins.json \
#     --model meta-llama/Llama-3.2-1B-Instruct --chat-template-config llama3 --task asr

set -euo pipefail

echo "[run_text_prior] host-side setup"

if [[ -z "${1:-}" || -z "${2:-}" ]]; then
    echo "Usage: sbatch $0 <frozen_set_dir> <out.json> [text-prior args...]"
    exit 2
fi
FROZEN_SET="$1"
OUT_JSON="$2"
shift 2
EXTRA_ARGS=("$@")

VENV_PATH="${VENV_PATH:-}"
if [[ -z "${VENV_PATH}" ]]; then
    echo "ERROR: VENV_PATH is not set. This is normally set by a site file"
    echo "       (infra/sites/<site>.sh) via infra/runners/submit_text_prior.sh."
    exit 2
fi
if [[ ! -f "${VENV_PATH}" ]]; then
    echo "ERROR: VENV_PATH not found: ${VENV_PATH}"
    exit 2
fi
if [[ ! -d "${FROZEN_SET}" ]]; then
    echo "ERROR: frozen set directory not found: ${FROZEN_SET}"
    exit 2
fi

mkdir -p "$(dirname "${OUT_JSON}")" logs

echo "[run_text_prior] venv:       ${VENV_PATH}"
echo "[run_text_prior] frozen set: ${FROZEN_SET}"
echo "[run_text_prior] out:        ${OUT_JSON}"
echo "[run_text_prior] extra args: ${EXTRA_ARGS[*]:-(none)}"

# shellcheck disable=SC1090
source "${VENV_PATH}"

if [[ ! -f melteval/tasks.py ]]; then
    echo "ERROR: run this from the melt-eval repo root (melteval/tasks.py not found here)."
    exit 2
fi

exec melteval text-prior "${FROZEN_SET}" -o "${OUT_JSON}" "${EXTRA_ARGS[@]}"
