#!/bin/bash
#SBATCH --job-name=smurf-crosscheck
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

##SBATCH --partition=<gpu-partition>          # a partition with an 80 GB GPU (h100/h200/a100-80)
##SBATCH --account=<account>
##SBATCH --qos=<qos>
##SBATCH --mem=64G

# Submit from the melt-eval repo root:
#
#   MODEL_CKPT=data/checkpoints/step=2784-last.ckpt \
#   VENV=$HOME/venvs/smurf-eval/bin/activate \
#   sbatch infra/crosscheck.sbatch
#
# (sbatch forwards the environment by default, so exported vars reach the job.)
#
# Required env:
#   MODEL_CKPT   SMURF .ckpt path, absolute or relative to the repo root
#   VENV         path to the SMURF venv's bin/activate (fbk_speechllm + NeMo +
#                melt-eval[hf,metrics], the paso-1 venv)
#
# Optional env (forwarded to scripts/run_crosscheck.sh):
#   CONFIG               freeze config      [configs/librispeech-hf-smoke.yaml]
#   SMURF_GEN_DIR        upstream gen-config dir  [../smurf/speechllm/working_config/generate]
#   SMURF_GEN_NAME       --config-name            [asr_inference]
#   INSTRUCTION          prompt text        ["Transcribe this English audio: "]
#   MAX_TOKENS           generation cap     [128]
#   MELTEVAL_DEVICE_MAP  "auto" to shard the melteval side; "" (default) = one GPU
#   WORK                 output root        [runs]
#   FORCE=1              redo finished steps
#   FROM_STEP / TO_STEP  run a sub-range (e.g. FROM_STEP=5 to only re-compare)
#   SKIP_UPSTREAM=1      skip fbk_speechllm.inference and the compare

set -euo pipefail

: "${MODEL_CKPT:?set MODEL_CKPT=/path/to/model.ckpt}"
: "${VENV:?set VENV=/path/to/smurf-eval/bin/activate}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f melteval/tasks.py ]] || { echo "ERROR: submit from the melt-eval repo root" >&2; exit 2; }
[[ -f "$VENV" ]]           || { echo "ERROR: VENV not found: $VENV" >&2; exit 2; }
[[ -e "$MODEL_CKPT" ]]     || { echo "ERROR: checkpoint not found: $MODEL_CKPT" >&2; exit 2; }

mkdir -p logs
export TMPDIR="${TMPDIR:-$HOME/.cache/triton-eval}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$TMPDIR}"
mkdir -p "$TMPDIR"

# Default to no sharding
export MELTEVAL_DEVICE_MAP="${MELTEVAL_DEVICE_MAP-}"

echo "[crosscheck] node=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[crosscheck] venv=${VENV}"
echo "[crosscheck] checkpoint=${MODEL_CKPT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# shellcheck disable=SC1090
source "$VENV"

exec scripts/run_crosscheck.sh "$MODEL_CKPT"
