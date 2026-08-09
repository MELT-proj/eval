#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output="logs/%x.%j.out"
#SBATCH --job-name=melt-eval
#
# Runs `inspect eval` for one checkpoint against one frozen set, inside the
# venv named by VENV_PATH (set by the site file). Single GPU, single node --
# an eval run has no multi-node case to support.
#
# Not usually invoked directly: use infra/runners/submit_eval.sh, which
# sources a site file for VENV_PATH/OUTPUT_DIR/SBATCH_ARGS and forwards
# everything below.
#
# Usage (host side, from the repo root):
#   VENV_PATH=/path/to/venv/bin/activate OUTPUT_DIR=/path/to/eval-logs \
#     sbatch [slurm flags] infra/run_eval.sbatch <checkpoint_dir> <frozen_set_dir> [inspect eval args...]
#
# Example:
#   sbatch infra/run_eval.sbatch \
#     /mnt/scratch-artemis/giuseppe/melt-data/outputs/MA-v1.2.7 \
#     /mnt/scratch-artemis/giuseppe/melt-data/eval-sets/asr-test-v1 \
#     -T task_filter=asr -M batch_size=16

set -euo pipefail

echo "[run_eval] host-side setup for run_eval.sbatch"

if [[ -z "${1:-}" || -z "${2:-}" ]]; then
    echo "Usage: sbatch $0 <checkpoint_dir> <frozen_set_dir> [inspect eval args...]"
    echo "Example: sbatch $0 /path/to/outputs/MA-v1.2.7 /path/to/eval-sets/asr-test-v1 -T task_filter=asr"
    exit 2
fi
CHECKPOINT="$1"
FROZEN_SET="$2"
shift 2
EXTRA_ARGS=("$@")

VENV_PATH="${VENV_PATH:-}"
if [[ -z "${VENV_PATH}" ]]; then
    echo "ERROR: VENV_PATH is not set. This is normally set by a site file"
    echo "       (infra/sites/<site>.sh) via infra/runners/submit_eval.sh."
    exit 2
fi
if [[ ! -f "${VENV_PATH}" ]]; then
    echo "ERROR: VENV_PATH not found: ${VENV_PATH}"
    echo "       This site's venv does not exist yet -- see the comment at the"
    echo "       top of its infra/sites/<site>.sh for how to build one."
    exit 2
fi

if [[ ! -d "${CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint directory not found: ${CHECKPOINT}"
    exit 2
fi
if [[ ! -d "${FROZEN_SET}" ]]; then
    echo "ERROR: frozen set directory not found: ${FROZEN_SET}"
    echo "       Build one first with: melteval freeze <spec.yaml> -o <frozen_set_dir>"
    exit 2
fi

OUTPUT_DIR="${OUTPUT_DIR:-./eval-logs}"
mkdir -p "${OUTPUT_DIR}" logs

echo "[run_eval] venv:       ${VENV_PATH}"
echo "[run_eval] checkpoint: ${CHECKPOINT}"
echo "[run_eval] frozen set: ${FROZEN_SET}"
echo "[run_eval] output dir: ${OUTPUT_DIR}"
echo "[run_eval] extra args: ${EXTRA_ARGS[*]:-(none)}"

# shellcheck disable=SC1090
source "${VENV_PATH}"

# repo root check -- the task reference below is repo-relative.
if [[ ! -f melteval/tasks.py ]]; then
    echo "ERROR: run this from the melt-eval repo root (melteval/tasks.py not found here)."
    exit 2
fi

exec inspect eval melteval/tasks.py@speech \
    --model "melt/${CHECKPOINT}" \
    --log-dir "${OUTPUT_DIR}" \
    -T "frozen_set=${FROZEN_SET}" \
    "${EXTRA_ARGS[@]}"
