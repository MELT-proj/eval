#!/bin/bash
#
# Submit a pack of eval slices (one model, many slices) as one SLURM job.
#
#   MELT_GPUS=4 infra/runners/submit_pack.sh <site> <manifest.json>
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

SITE="${1:?usage: $0 <site> <manifest.json>}"
MANIFEST="${2:?usage: $0 <site> <manifest.json>}"
[[ -f melteval/tasks.py ]] || die "run this from the melt-eval repo root (melteval/tasks.py not found here)"
[[ -f "${MANIFEST}" ]] || die "pack manifest not found: ${MANIFEST}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_FILE="${SCRIPT_DIR}/../sites/${SITE}.sh"
[[ -f "$SITE_FILE" ]] || die "unknown site '${SITE}' (expected ${SITE_FILE}); copy sites/example.sh to add one"
# shellcheck disable=SC1090
source "$SITE_FILE"

GPUS="${MELT_GPUS:-1}"
ARGS=()
for arg in "${SBATCH_ARGS[@]}"; do
    [[ "$arg" == --gpus-per-node=* || "$arg" == --gres=gpu* ]] || ARGS+=("$arg")
done
ARGS+=(--gpus-per-node="${GPUS}")

mkdir -p logs   # SLURM won't create the --output dir; a missing dir kills the job silently.
if [[ -n "${MELT_SLURM_OUT:-}" ]]; then
    mkdir -p "$(dirname "${MELT_SLURM_OUT}")"
    ARGS+=(--output="${MELT_SLURM_OUT}")
fi
[[ -n "${MELT_JOB_TAG:-}" ]] && ARGS+=(--comment="${MELT_JOB_TAG}")

echo "[submit_pack] site=${SITE} sbatch ${ARGS[*]} infra/run_eval_pack.sbatch ${MANIFEST}"
sbatch "${ARGS[@]}" infra/run_eval_pack.sbatch "${MANIFEST}"
