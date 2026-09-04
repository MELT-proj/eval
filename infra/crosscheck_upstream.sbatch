#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output="logs/%x.%j.out"
#SBATCH --job-name=smurf-crosscheck-upstream
#

# runs the upstream inference step of the SMURF cross-check using fbk_speechllm.inference

#   MODEL_PATH=/abs/model.ckpt \
#   DATA_PATH=/abs/runs/crosscheck-smurf-cuts/cuts.jsonl.gz \
#   OUTPUT_PATH=/abs/runs/crosscheck-smurf/upstream.jsonl \
#   VENV_PATH=/path/to/venvs/smurf-eval/bin/activate \
#   sbatch --gpus-per-node=1 --partition=h100 infra/crosscheck_upstream.sbatch \
#     /abs/../speechllm/working_config/generate asr_inference \
#     generation.max_new_tokens=128
#
# Positional args: <config-path> <config-name> [extra hydra overrides...]
# `data.test_ds.batch_size=1` is always appended (the cross-check pins it).

set -euo pipefail

echo "[crosscheck_upstream] host-side setup"

CONFIG_PATH="${1:-}"
CONFIG_NAME="${2:-}"
if [[ -z "$CONFIG_PATH" || -z "$CONFIG_NAME" ]]; then
    echo "Usage: sbatch $0 <config-path> <config-name> [hydra overrides...]"
    exit 2
fi
shift 2
OVERRIDES=("$@")

for var in VENV_PATH MODEL_PATH DATA_PATH OUTPUT_PATH; do
    if [[ -z "${!var:-}" ]]; then
        echo "ERROR: $var is not set (normally exported by scripts/run_crosscheck.sh)."
        exit 2
    fi
done
[[ -f "$VENV_PATH" ]]  || { echo "ERROR: VENV_PATH not found: $VENV_PATH"; exit 2; }
[[ -e "$MODEL_PATH" ]] || { echo "ERROR: checkpoint not found: $MODEL_PATH"; exit 2; }
[[ -f "$DATA_PATH" ]]  || { echo "ERROR: cuts manifest not found: $DATA_PATH"; exit 2; }

mkdir -p "$(dirname "$OUTPUT_PATH")" logs

echo "[crosscheck_upstream] venv:        $VENV_PATH"
echo "[crosscheck_upstream] model:       $MODEL_PATH"
echo "[crosscheck_upstream] data:        $DATA_PATH"
echo "[crosscheck_upstream] output:      $OUTPUT_PATH"
echo "[crosscheck_upstream] config:      $CONFIG_PATH / $CONFIG_NAME"
echo "[crosscheck_upstream] overrides:   data.test_ds.batch_size=1 ${OVERRIDES[*]:-(none)}"

# shellcheck disable=SC1090
source "$VENV_PATH"

exec python -m fbk_speechllm.inference \
    --config-path "$CONFIG_PATH" \
    --config-name "$CONFIG_NAME" \
    data.test_ds.batch_size=1 \
    "${OVERRIDES[@]}"
