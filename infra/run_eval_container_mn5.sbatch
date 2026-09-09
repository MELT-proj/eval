#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --output="logs/%x.%j.out"
#SBATCH --job-name=melt-eval-container-mn5
#
# MN5 variant of run_eval_container.sbatch. The difference from the artemis
# version: MN5 has no outbound internet, so the code that runs has to be a
# bind-mounted checkout (pushed over SSH -- see infra/sync_repo.sh's sibling
# convention in the training repo) rather than whatever melteval was baked
# into the .sif at build time on a machine that *does* have internet. Same
# mental model as training/bash/run_train_singularity.sbatch: "the code that
# runs is the checkout on MN5, not the copy inside the image."
#
# Usage (host side, from the melt-eval repo root on MN5, after
# infra/sync_repo.sh mn5 --init has pushed a branch here):
#   MELT_SIF=/path/to/melt_eval_cuda126.sif OUTPUT_DIR=/path/to/eval-logs \
#     sbatch [slurm flags] infra/run_eval_container_mn5.sbatch \
#     <checkpoint_dir> <frozen_set_dir> [inspect eval args...]
#
# <checkpoint_dir> and <frozen_set_dir> are HOST paths (e.g. under /gpfs/...):
# the whole point of binding /gpfs wholesale (see BINDS below) rather than
# remapping to a fixed /workspace/... path is that a host path is valid
# unchanged inside the container, so callers don't need a second,
# container-relative form of every path they pass.

set -euo pipefail

echo "[run_eval_container_mn5] host-side setup"

if [[ -z "${1:-}" || -z "${2:-}" ]]; then
    echo "Usage: sbatch $0 <checkpoint_dir> <frozen_set_dir> [inspect eval args...]"
    exit 2
fi
CHECKPOINT="$1"
FROZEN_SET="$2"
shift 2
EXTRA_ARGS=("$@")

MELT_SIF="${MELT_SIF:-}"
[[ -n "${MELT_SIF}" ]] || { echo "ERROR: MELT_SIF is not set."; exit 2; }
[[ -f "${MELT_SIF}" ]] || { echo "ERROR: MELT_SIF not found: ${MELT_SIF}"; exit 2; }
[[ -d "${CHECKPOINT}" ]] || { echo "ERROR: checkpoint directory not found: ${CHECKPOINT}"; exit 2; }
[[ -d "${FROZEN_SET}" ]] || { echo "ERROR: frozen set directory not found: ${FROZEN_SET}"; exit 2; }

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
[[ -f "${SUBMIT_DIR}/melteval/tasks.py" ]] || {
    echo "ERROR: run this from the melt-eval repo root (melteval/tasks.py not found in ${SUBMIT_DIR})."
    exit 2
}

OUTPUT_DIR="${OUTPUT_DIR:-${SUBMIT_DIR}/eval-logs}"
mkdir -p "${OUTPUT_DIR}" logs

# Container runtime: MN5 exposes it as a module, same pattern as
# run_train_singularity.sbatch and run_eval_container.sbatch.
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

echo "[run_eval_container_mn5] runtime:    ${SINGULARITY_BIN}"
echo "[run_eval_container_mn5] image:      ${MELT_SIF}"
echo "[run_eval_container_mn5] checkpoint: ${CHECKPOINT}"
echo "[run_eval_container_mn5] frozen set: ${FROZEN_SET}"
echo "[run_eval_container_mn5] output dir: ${OUTPUT_DIR}"
echo "[run_eval_container_mn5] repo:       ${SUBMIT_DIR}"
echo "[run_eval_container_mn5] extra args: ${EXTRA_ARGS[*]:-(none)}"

# --nv: host GPU driver (the image's CUDA toolkit is only for JIT builds).
# --bind /gpfs: checkpoints, frozen sets and datasets all live under
#   /gpfs/{projects,scratch} on MN5; binding the real host path (not remapping
#   to a fixed /workspace/... path) means every path this job was called with
#   is valid unchanged inside the container. Same idea as artemis's
#   `--bind /mnt` in run_eval_container.sbatch.
# --bind "${SUBMIT_DIR}": the repo checkout, bind-mounted OVER whatever
#   melteval the image had baked in at build time -- see the file header.
exec "${SINGULARITY_BIN}" exec --nv \
    --bind /gpfs \
    --bind "${SUBMIT_DIR}" \
    --pwd "${SUBMIT_DIR}" \
    "${MELT_SIF}" \
    bash -lc '
        set -euo pipefail
        source /workspace/venv/bin/activate
        exec inspect eval melteval/tasks.py@speech \
            --model "melt/$1" \
            --log-dir "$2" \
            -T "frozen_set=$3" \
            "${@:4}"
    ' _ "${CHECKPOINT}" "${OUTPUT_DIR}" "${FROZEN_SET}" "${EXTRA_ARGS[@]}"
