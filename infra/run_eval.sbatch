#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output="logs/%x.%j.out"
#SBATCH --job-name=melt-eval
#
# Runs `inspect eval` for one model against one evaluation set, inside
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
#     sbatch [slurm flags] infra/run_eval.sbatch <model> <eval_set> [inspect eval args...]
#
# <model> is an inspect model name (see melteval/providers/__init__.py):
#   melt/<checkpoint>     a MELT checkpoint
#   melt/hf/<model>       an off-the-shelf baseline, e.g. melt/hf/qwen2_audio
#   smurf/<checkpoint>    a SMURF checkpoint
# A bare checkpoint path also works: it is prefixed with $MELTEVAL_PROVIDER
# (default `melt`). The matching -T prompt_style is added unless given.
#
# Examples:
#   sbatch infra/run_eval.sbatch \
#     /mnt/scratch-artemis/giuseppe/melt-data/outputs/MA-v1.2.7 \
#     /mnt/scratch-artemis/giuseppe/melt-data/eval-sets/asr-test-v1 \
#     -T task_filter=asr -M batch_size=16
#
#   sbatch infra/run_eval.sbatch melt/hf/qwen2_audio configs/hf/air-bench.yaml \
#     -T task_filter=audio_mcq -T dataset_id=air-bench-foundation-speech
#
# A SMURF checkpoint is a single Lightning .ckpt rather than a directory, and
# its stack (NeMo + fbk_speechllm) does not co-install with MELT's, so it needs
# its own venv too:
#
#   VENV_PATH=/path/to/venvs/smurf-eval/bin/activate \
#     infra/runners/submit_eval.sh artemis smurf/<path-to-.ckpt> /path/to/frozen-set \
#     -T instruction="Transcribe this English audio: "

set -euo pipefail

echo "[run_eval] host-side setup for run_eval.sbatch"

if [[ -z "${1:-}" || -z "${2:-}" ]]; then
    echo "Usage: sbatch $0 <model> <eval_set> [inspect eval args...]"
    echo "  <model> is melt/<checkpoint>, melt/hf/<model> or smurf/<checkpoint>."
    echo "  <eval_set> is a frozen-set directory or a source spec YAML."
    echo "Example: sbatch $0 melt/<path-to-checkpoint> /path/to/eval-sets/asr-test-v1 -T task_filter=asr"
    exit 2
fi
MODEL="$1"
# Resolved to an absolute path before it is handed to inspect. `inspect eval`
# chdir's into the task *file's* directory while it constructs the task (see
# chdir_python in inspect_ai/_eval/loader.py), so a relative path in a -T
# argument is resolved against melteval/, not against the directory the job
# was submitted from -- "No such file or directory: configs/hf/air-bench.yaml"
# from a job whose cwd plainly contains it.
EVAL_SET="$(realpath -m "$2")"
shift 2
EXTRA_ARGS=("$@")

# Split the model name into provider and the rest, and pick the prompt family
# (melteval.providers.model_family, in bash). A bare path is the old interface.
case "$MODEL" in
    melt/*|smurf/*) ;;
    *) MODEL="${MELTEVAL_PROVIDER:-melt}/${MODEL}" ;;
esac
PROVIDER="${MODEL%%/*}"
NAME="${MODEL#*/}"
CHECKPOINT=""
case "$PROVIDER/$NAME" in
    melt/hf/*)   FAMILY=hf ;;
    melt/vllm/*) FAMILY=vllm ;;
    melt/*)      FAMILY=melt;  CHECKPOINT="$NAME" ;;
    smurf/*)     FAMILY=smurf; CHECKPOINT="$NAME" ;;
    *) echo "ERROR: unknown provider in model ${MODEL} (expected melt/... or smurf/...)"; exit 2 ;;
esac

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

# A MELT checkpoint is a directory; a SMURF one is a .ckpt file (or a
# save_pretrained directory), so accept either and let the provider judge.
# Made absolute for the same chdir reason as the evaluation set above.
if [[ -n "${CHECKPOINT}" ]]; then
    if [[ ! -e "${CHECKPOINT}" ]]; then
        echo "ERROR: checkpoint not found: ${CHECKPOINT}"
        exit 2
    fi
    CHECKPOINT="$(realpath "${CHECKPOINT}")"
    MODEL="${PROVIDER}/${CHECKPOINT}"
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

# The prompt format has to match the model's family, so it follows the model
# unless the caller said otherwise (e.g. to run a deliberate mismatch).
if [[ ! " ${EXTRA_ARGS[*]:-} " == *"prompt_style="* ]]; then
    EXTRA_ARGS+=(-T "prompt_style=${FAMILY}")
fi

echo "[run_eval] venv:       ${VENV_PATH}"
echo "[run_eval] model:      ${MODEL} (prompt family ${FAMILY})"
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
    --model "${MODEL}" \
    --log-dir "${OUTPUT_DIR}" \
    "${EVAL_SET_ARG[@]}" \
    "${EXTRA_ARGS[@]}"
