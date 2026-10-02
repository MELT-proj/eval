#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output="logs/%x.%j.out"
#SBATCH --job-name=melt-score
#
# Score finished eval logs with `inspect score`, one after another, inside a
# SLURM job. Post-hoc scoring loads a model of its own -- a judge for
# audio_chat, BERTScore's encoder for MCIF's official QA/SUM metric -- so it is
# GPU work like generation and goes through the scheduler, never a login shell.
#
#   VENV_PATH=<venv>/bin/activate sbatch <site SBATCH_ARGS> infra/run_score.sbatch \
#       <log>... -- <inspect score args...>
#
# Normally submitted by `scripts/evaluate.py score`, which picks the venv
# and arguments per kind of log. Pass `--model mockllm/model` when calling it by
# hand: `inspect score` otherwise rebuilds the log's own model, and a local
# provider loads its weights just to be constructed. Each log is rewritten in
# place (--overwrite), so the report picks the new scores up from the same file.
set -euo pipefail

LOGS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
    LOGS+=("$1")
    shift
done
[[ "${1:-}" == "--" ]] && shift
SCORE_ARGS=("$@")

if [[ ${#LOGS[@]} -eq 0 ]]; then
    echo "Usage: sbatch $0 <log>... -- <inspect score args...>"
    exit 2
fi
if [[ -z "${VENV_PATH:-}" || ! -f "${VENV_PATH}" ]]; then
    echo "ERROR: VENV_PATH not set or not found: '${VENV_PATH:-}'"
    exit 2
fi
if [[ ! -f melteval/tasks.py ]]; then
    echo "ERROR: run this from the melt-eval repo root (melteval/tasks.py not found here)."
    exit 2
fi

echo "[run_score] venv:  ${VENV_PATH}"
echo "[run_score] args:  ${SCORE_ARGS[*]:-(none)}"
source "${VENV_PATH}"

for log in "${LOGS[@]}"; do
    echo "[run_score] scoring ${log}"
    inspect score "${SCORE_ARGS[@]}" --overwrite "${log}"
done
