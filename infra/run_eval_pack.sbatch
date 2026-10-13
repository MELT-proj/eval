#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --output="logs/%x.%j.out"
#SBATCH --job-name=melt-eval-pack
#
# Runs a pack of eval slices of one model on one node: one worker per GPU
# (melteval/pack.py), each loading the model once and taking slices from the
# pack's manifest until none are left. Every slice writes its own inspect log
# into its own log_dir, exactly as infra/run_eval.sbatch would.
#
# Usage (host side, from the repo root):
#   VENV_PATH=/path/to/venv/bin/activate sbatch [slurm flags] infra/run_eval_pack.sbatch <manifest.json>

set -euo pipefail

MANIFEST="${1:?usage: sbatch $0 <manifest.json>}"
MANIFEST="$(realpath "${MANIFEST}")"

VENV_PATH="${VENV_PATH:-}"
if [[ -z "${VENV_PATH}" || ! -f "${VENV_PATH}" ]]; then
    echo "ERROR: VENV_PATH not set or not found: '${VENV_PATH}'"
    echo "       This is normally set by a site file (infra/sites/<site>.sh) via infra/runners/submit_pack.sh."
    exit 2
fi
if [[ ! -f melteval/tasks.py ]]; then
    echo "ERROR: run this from the melt-eval repo root (melteval/tasks.py not found here)."
    exit 2
fi

# The GPUs SLURM gave this job, one worker each. Outside SLURM (or with no
# CUDA_VISIBLE_DEVICES), MELT_GPUS workers on devices 0..N-1.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
else
    mapfile -t GPUS < <(seq 0 $(( ${MELT_GPUS:-1} - 1 )))
fi

if [[ -z "${NO_COLOR:-}" ]]; then
    export FORCE_COLOR="${FORCE_COLOR:-1}"
fi

echo "[run_eval_pack] venv:     ${VENV_PATH}"
echo "[run_eval_pack] manifest: ${MANIFEST}"
echo "[run_eval_pack] GPUs:     ${GPUS[*]} (one worker each)"

# shellcheck disable=SC1090
source "${VENV_PATH}"

# Split the node's CPUs between the workers, so four tokenizers/dataloaders do
# not each assume they have all of them.
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    export OMP_NUM_THREADS="$(( SLURM_CPUS_PER_TASK / ${#GPUS[@]} > 0 ? SLURM_CPUS_PER_TASK / ${#GPUS[@]} : 1 ))"
fi

PIDS=()
for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPUS[$i]}" python -m melteval.pack "${MANIFEST}" --worker-id "gpu${GPUS[$i]}" &
    PIDS+=($!)
done

STATUS=0
for pid in "${PIDS[@]}"; do
    wait "${pid}" || STATUS=1
done
echo "[run_eval_pack] all workers finished (exit ${STATUS}); run scripts/evaluate.py status for the per-slice view"
exit "${STATUS}"
