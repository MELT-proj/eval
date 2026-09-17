#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output="logs/%x.%j.out"
#SBATCH --job-name=melt-eval
#
# Runs `inspect eval` for one checkpoint against one evaluation set, inside
# the venv named by VENV_PATH (set by the site file). Single GPU, single node
# -- an eval run has no multi-node case to support.
#
# The evaluation set is either a frozen-set directory or a source spec YAML
# read live (the HuggingFace path, which needs no freeze pass -- see
# melteval/dataset.py). Which one is decided here from what the path *is*,
# rather than from a flag, so a caller swapping one for the other changes
# nothing else about the command.
#
# Not usually invoked directly: use infra/runners/submit_eval.sh, which
# sources a site file for VENV_PATH/OUTPUT_DIR/SBATCH_ARGS and forwards
# everything below.
#
# Usage (host side, from the repo root):
#   VENV_PATH=/path/to/venv/bin/activate OUTPUT_DIR=/path/to/eval-logs \
#     sbatch [slurm flags] infra/run_eval.sbatch <checkpoint_dir> <eval_set> [inspect eval args...]
#
# Examples:
#   sbatch infra/run_eval.sbatch \
#     /mnt/scratch-artemis/giuseppe/melt-data/outputs/MA-v1.2.7 \
#     /mnt/scratch-artemis/giuseppe/melt-data/eval-sets/asr-test-v1 \
#     -T task_filter=asr -M batch_size=16
#
#   sbatch infra/run_eval.sbatch \
#     /mnt/scratch-artemis/giuseppe/melt-data/outputs/MA-v1.2.7 \
#     configs/hf/air-bench.yaml \
#     -T task_filter=audio_mcq -T dataset_id=air-bench-foundation-speech

set -euo pipefail

echo "[run_eval] host-side setup for run_eval.sbatch"

if [[ -z "${1:-}" || -z "${2:-}" ]]; then
    echo "Usage: sbatch $0 <checkpoint_dir> <eval_set> [inspect eval args...]"
    echo "  <eval_set> is a frozen-set directory or a source spec YAML."
    echo "Example: sbatch $0 /path/to/outputs/MA-v1.2.7 /path/to/eval-sets/asr-test-v1 -T task_filter=asr"
    exit 2
fi
CHECKPOINT="$1"
# Resolved to an absolute path before it is handed to inspect. `inspect eval`
# chdir's into the task *file's* directory while it constructs the task (see
# chdir_python in inspect_ai/_eval/loader.py), so a relative path in a -T
# argument is resolved against melteval/, not against the directory the job
# was submitted from -- "No such file or directory: configs/hf/air-bench.yaml"
# from a job whose cwd plainly contains it.
EVAL_SET="$(realpath -m "$2")"
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
# A directory is a frozen set; a YAML file is a spec to read live.
if [[ -d "${EVAL_SET}" ]]; then
    EVAL_SET_ARG=(-T "frozen_set=${EVAL_SET}")
    EVAL_SET_KIND="frozen set"
elif [[ -f "${EVAL_SET}" && "${EVAL_SET}" == *.y*ml ]]; then
    EVAL_SET_ARG=(-T "spec=${EVAL_SET}")
    EVAL_SET_KIND="live spec"
else
    echo "ERROR: evaluation set not found (or not a recognised kind): ${EVAL_SET}"
    echo "       Expected a frozen-set directory, or a source spec .yaml/.yml file."
    echo "       Build a frozen set with: melteval freeze <spec.yaml> -o <frozen_set_dir>"
    exit 2
fi

OUTPUT_DIR="${OUTPUT_DIR:-./eval-logs}"
mkdir -p "${OUTPUT_DIR}" logs

echo "[run_eval] venv:       ${VENV_PATH}"
echo "[run_eval] checkpoint: ${CHECKPOINT}"
echo "[run_eval] eval set:   ${EVAL_SET} (${EVAL_SET_KIND})"
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
    "${EVAL_SET_ARG[@]}" \
    "${EXTRA_ARGS[@]}"
