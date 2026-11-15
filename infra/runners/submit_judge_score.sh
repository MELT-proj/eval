#!/bin/bash
# Submit the persistent-vLLM judge scoring job.
#
#   VENV_PATH=<client>/bin/activate \
#   JUDGE_SERVER_VENV=<vllm-server>/bin/activate \
#   infra/runners/submit_judge_score.sh <site> <log>... -- <inspect score args...>
#
# The score job starts one vLLM server, scores every log against it, and stops it.
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

SITE="${1:?usage: $0 <site> <log>... -- <inspect score args...>}"; shift
[[ -f melteval/tasks.py ]] || die "run this from the melt-eval repo root (melteval/tasks.py not found here)"
[[ -n "${JUDGE_SERVER_VENV:-}" && -f "${JUDGE_SERVER_VENV}" ]] || die "JUDGE_SERVER_VENV is not set or not found"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_FILE="${SCRIPT_DIR}/../sites/${SITE}.sh"
[[ -f "$SITE_FILE" ]] || die "unknown site '${SITE}' (expected ${SITE_FILE})"
# shellcheck disable=SC1090
source "$SITE_FILE"

mkdir -p logs
if [[ -n "${MELT_SLURM_OUT:-}" ]]; then
    mkdir -p "$(dirname "${MELT_SLURM_OUT}")"
    SBATCH_ARGS+=(--output="${MELT_SLURM_OUT}")
fi
[[ -n "${MELT_JOB_TAG:-}" ]] && SBATCH_ARGS+=(--comment="${MELT_JOB_TAG}")

echo "[submit_judge_score] site=${SITE} server=${JUDGE_SERVER_VENV} client=${VENV_PATH}"
echo "[submit_judge_score] sbatch ${SBATCH_ARGS[*]} infra/run_judge_score.sbatch $*"
sbatch "${SBATCH_ARGS[@]}" infra/run_judge_score.sbatch "$@"
