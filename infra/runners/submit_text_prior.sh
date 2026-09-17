#!/bin/bash
#
# Submit a text-prior scoring job to SLURM.
#
#   infra/runners/submit_text_prior.sh <site> <frozen_set_dir> <out.json> [text-prior args...]
#
# Mirrors submit_eval.sh's shape: <site> selects infra/sites/<site>.sh for
# VENV_PATH/SBATCH_ARGS, everything after the two required paths goes to
# `melteval text-prior` (--model, --chat-template-config, --task, --lang, ...).
#
# Example:
#   infra/runners/submit_text_prior.sh artemis \
#     /mnt/scratch-artemis/giuseppe/melt-data/eval-sets/fleurs24-asr-dev \
#     /mnt/scratch-artemis/giuseppe/melt-data/text-prior/llama1b-ins-en.json \
#     --model meta-llama/Llama-3.2-1B-Instruct --chat-template-config llama3 \
#     --task asr --lang en
#
# Run from the repo root.
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

SITE="${1:?usage: $0 <site> <frozen_set_dir> <out.json> [text-prior args...]}"; shift
[[ -f melteval/tasks.py ]] || die "run this from the melt-eval repo root (melteval/tasks.py not found here)"
[[ $# -ge 2 ]] || die "usage: $0 <site> <frozen_set_dir> <out.json> [text-prior args...]"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_FILE="${SCRIPT_DIR}/../sites/${SITE}.sh"
[[ -f "$SITE_FILE" ]] || die "unknown site '${SITE}' (expected ${SITE_FILE})"
# shellcheck disable=SC1090
source "$SITE_FILE"

mkdir -p logs

echo "[submit_text_prior] site=${SITE} sbatch ${SBATCH_ARGS[*]} infra/run_text_prior.sbatch $*"
sbatch "${SBATCH_ARGS[@]}" infra/run_text_prior.sbatch "$@"
