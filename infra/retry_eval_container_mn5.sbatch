#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --output="logs/%x.%j.out"
#SBATCH --job-name=melt-eval-retry-mn5
#
# Resume an eval that ran out of wall time, instead of paying for it twice.
#
# `inspect eval-retry` re-runs only the samples a log does not already have a
# successful score for, and takes the model, frozen set and every task/model
# argument from the log itself -- so this wrapper needs the log path and
# nothing else. A checkpoint whose ASR job was killed at 79% resumes at 79%.
#
# This exists because the qwen-2b campaign's ASR jobs turned out to need ~21 h
# against an 18 h (final) and 12 h (checkpoints) request: roughly 4x the
# llama-1b run they were sized from. Those wall times cannot be raised on a
# running job -- `scontrol update TimeLimit` is denied to the job's own user on
# MN5 -- so the work is resumed rather than restarted.
#
# Usage (from the melt-eval repo root on MN5):
#   MELT_SIF=/path/to/melt_eval_cuda126_v2.sif \
#     sbatch [slurm flags] infra/retry_eval_container_mn5.sbatch <log.eval> [...]
#
# The retry writes a *new* log next to the old one; the killed partial stays
# put, with its non-success status, and `projects/melt` skips it on that basis.
# Delete the partials once the retries have finished if the directory matters.

set -euo pipefail

echo "[retry_eval_container_mn5] host-side setup"

if [[ -z "${1:-}" ]]; then
    echo "Usage: sbatch $0 <log.eval> [more logs...]"
    exit 2
fi
LOGS=("$@")
for log in "${LOGS[@]}"; do
    [[ -f "${log}" ]] || { echo "ERROR: log file not found: ${log}"; exit 2; }
done

MELT_SIF="${MELT_SIF:-}"
[[ -n "${MELT_SIF}" ]] || { echo "ERROR: MELT_SIF is not set."; exit 2; }
[[ -f "${MELT_SIF}" ]] || { echo "ERROR: MELT_SIF not found: ${MELT_SIF}"; exit 2; }

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f "${SUBMIT_DIR}/melteval/tasks.py" ]] || {
    echo "ERROR: run this from the melt-eval repo root (melteval/tasks.py not found in ${SUBMIT_DIR})."
    exit 2
}
mkdir -p logs

if [[ -f /etc/profile.d/02-lmod.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/02-lmod.sh
fi
if command -v module >/dev/null 2>&1; then
    module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true
fi
SINGULARITY_BIN="${SINGULARITY_BIN:-}"
if [[ -z "${SINGULARITY_BIN}" ]]; then
    if command -v apptainer >/dev/null 2>&1; then
        SINGULARITY_BIN=apptainer
    elif command -v singularity >/dev/null 2>&1; then
        SINGULARITY_BIN=singularity
    else
        echo "ERROR: neither 'apptainer' nor 'singularity' found on this node."
        exit 2
    fi
fi

echo "[retry_eval_container_mn5] runtime: ${SINGULARITY_BIN}"
echo "[retry_eval_container_mn5] image:   ${MELT_SIF}"
echo "[retry_eval_container_mn5] repo:    ${SUBMIT_DIR}"
echo "[retry_eval_container_mn5] logs:    ${LOGS[*]}"

# Same binds as run_eval_container_mn5.sbatch, and for the same reasons: host
# /gpfs paths stay valid inside the container, and the checkout wins over the
# melteval baked into the image.
exec "${SINGULARITY_BIN}" exec --nv \
    --bind /gpfs \
    --bind "${SUBMIT_DIR}" \
    --pwd "${SUBMIT_DIR}" \
    "${MELT_SIF}" \
    bash -lc '
        set -euo pipefail
        source /workspace/venv/bin/activate
        exec inspect eval-retry "$@"
    ' _ "${LOGS[@]}"
