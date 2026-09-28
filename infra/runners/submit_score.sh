#!/bin/bash
# Submit infra/run_score.sbatch with a site's scheduler arguments -- the
# scoring counterpart of submit_eval.sh (same site files, same rule: anything
# that loads a model goes through SLURM).
#
#   VENV_PATH=<venv>/bin/activate infra/runners/submit_score.sh <site> <log>... -- <inspect score args...>
#
# Normally called by `scripts/run_matrix.py score`.
set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }

SITE="${1:?usage: $0 <site> <log>... -- <inspect score args...>}"; shift
[[ -f melteval/tasks.py ]] || die "run this from the melt-eval repo root (melteval/tasks.py not found here)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_FILE="${SCRIPT_DIR}/../sites/${SITE}.sh"
[[ -f "$SITE_FILE" ]] || die "unknown site '${SITE}' (expected ${SITE_FILE})"
source "$SITE_FILE"

mkdir -p logs   # SLURM won't create the --output dir; a missing dir kills the job silently.
echo "[submit_score] site=${SITE} venv=${VENV_PATH} sbatch ${SBATCH_ARGS[*]} infra/run_score.sbatch $*"
# A caller can tag the job (scripts/run_matrix.py tags each one with its
# model and slice) so it can tell what is already queued or running.
[[ -n "${MELT_JOB_TAG:-}" ]] && SBATCH_ARGS+=(--comment="${MELT_JOB_TAG}")
sbatch "${SBATCH_ARGS[@]}" infra/run_score.sbatch "$@"
