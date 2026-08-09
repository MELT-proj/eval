#!/bin/bash
#
# Submit an evaluation job to SLURM.
#
#   infra/runners/submit_eval.sh <site> <checkpoint_dir> <frozen_set_dir> [inspect eval args...]
#
# This is the whole interface a collaborator needs: swap the checkpoint, swap
# the frozen set, run. Everything else (venv, output dir, GPU/QoS) comes from
# the site file.
#
# Examples:
#   # ASR only, batch size 16
#   infra/runners/submit_eval.sh artemis \
#     /mnt/scratch-artemis/giuseppe/melt-data/outputs/MA-v1.2.7 \
#     /mnt/scratch-artemis/giuseppe/melt-data/eval-sets/asr-test-v1 \
#     -T task_filter=asr -M batch_size=16
#
#   # A different checkpoint against the same set -- only one arg changes
#   infra/runners/submit_eval.sh artemis \
#     /mnt/scratch-artemis/giuseppe/melt-data/outputs/SFT-v1.3.0 \
#     /mnt/scratch-artemis/giuseppe/melt-data/eval-sets/asr-test-v1
#
#   # A short debug run: 5 samples, the debug QoS
#   MELT_QOS=gpu-debug infra/runners/submit_eval.sh artemis \
#     /path/to/checkpoint /path/to/frozen-set -T limit=5
#
# <site> selects infra/sites/<site>.sh, which exports VENV_PATH/OUTPUT_DIR/
# LOCAL_DATASETS_DIR and defines the SBATCH_ARGS array (partition/QoS/time).
# Run from the repo root.
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

SITE="${1:?usage: $0 <site> <checkpoint_dir> <frozen_set_dir> [inspect eval args...]}"; shift
[[ -f melteval/tasks.py ]] || die "run this from the melt-eval repo root (melteval/tasks.py not found here)"
[[ $# -ge 2 ]] || die "usage: $0 <site> <checkpoint_dir> <frozen_set_dir> [inspect eval args...]"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_FILE="${SCRIPT_DIR}/../sites/${SITE}.sh"
[[ -f "$SITE_FILE" ]] || die "unknown site '${SITE}' (expected ${SITE_FILE}); copy sites/example.sh to add one"
# shellcheck disable=SC1090
source "$SITE_FILE"

mkdir -p logs   # SLURM won't create the --output dir; a missing dir kills the job silently.

echo "[submit_eval] site=${SITE} sbatch ${SBATCH_ARGS[*]} infra/run_eval.sbatch $*"
sbatch "${SBATCH_ARGS[@]}" infra/run_eval.sbatch "$@"
