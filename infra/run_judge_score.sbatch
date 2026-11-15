#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output="logs/%x.%j.out"
#SBATCH --job-name=melt-judge
#
# Score logs with one persistent vLLM server for the whole job.
#
#   VENV_PATH=<client>/bin/activate \
#   JUDGE_SERVER_VENV=<vllm-server>/bin/activate \
#   sbatch infra/run_judge_score.sbatch <log>... -- <inspect score args...>
#
# The client venv must have inspect_ai + openai + melteval. The server venv
# must have vLLM. The server is started once on this job's GPU, waited on at
# /health, and killed on every exit path.
set -euo pipefail

LOGS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
    LOGS+=("$1")
    shift
done
[[ "${1:-}" == "--" ]] && shift
SCORE_ARGS=("$@")

[[ ${#LOGS[@]} -gt 0 ]] || { echo "Usage: sbatch $0 <log>... -- <inspect score args...>"; exit 2; }
[[ -n "${VENV_PATH:-}" && -f "${VENV_PATH}" ]] || { echo "ERROR: VENV_PATH not set or not found"; exit 2; }
[[ -n "${JUDGE_SERVER_VENV:-}" && -f "${JUDGE_SERVER_VENV}" ]] || { echo "ERROR: JUDGE_SERVER_VENV not set or not found"; exit 2; }
[[ -f melteval/tasks.py ]] || { echo "ERROR: run this from the melt-eval repo root"; exit 2; }

# Use this checkout ahead of any editable copy installed in the venv.
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-8B}"
JUDGE_PORT="${JUDGE_PORT:-8000}"
JUDGE_API_KEY="${JUDGE_API_KEY:-inspectai}"
JUDGE_SERVER_LOG="${JUDGE_SERVER_LOG:-judge-server-${SLURM_JOB_ID:-local}.log}"

SERVER_ARGS=(
    --max-model-len "${JUDGE_MAX_MODEL_LEN:-4096}"
    --gpu-memory-utilization "${JUDGE_GPU_MEMORY_UTILIZATION:-0.90}"
)
if [[ -n "${JUDGE_SERVER_ARGS:-}" ]]; then
    read -r -a EXTRA_SERVER_ARGS <<< "${JUDGE_SERVER_ARGS}"
    SERVER_ARGS+=("${EXTRA_SERVER_ARGS[@]}")
fi

echo "[run_judge_score] server venv: ${JUDGE_SERVER_VENV}"
echo "[run_judge_score] client venv: ${VENV_PATH}"
echo "[run_judge_score] model: ${JUDGE_MODEL} port: ${JUDGE_PORT}"
echo "[run_judge_score] logs: ${#LOGS[@]}"

# Start the server in its own venv. The background process keeps that venv's PATH.
# shellcheck disable=SC1090
source "${JUDGE_SERVER_VENV}"
vllm serve "${JUDGE_MODEL}" \
    --host 127.0.0.1 \
    --port "${JUDGE_PORT}" \
    --api-key "${JUDGE_API_KEY}" \
    "${SERVER_ARGS[@]}" > "${JUDGE_SERVER_LOG}" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

ready=false
for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${JUDGE_PORT}/health" >/dev/null 2>&1; then
        ready=true
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: vLLM server exited before becoming ready; see ${JUDGE_SERVER_LOG}" >&2
        tail -50 "${JUDGE_SERVER_LOG}" >&2 || true
        exit 1
    fi
    sleep 2
done
if [[ "${ready}" != true ]]; then
    echo "ERROR: vLLM server did not become ready; see ${JUDGE_SERVER_LOG}" >&2
    tail -50 "${JUDGE_SERVER_LOG}" >&2 || true
    exit 1
fi
echo "[run_judge_score] server ready"

export VLLM_BASE_URL="http://127.0.0.1:${JUDGE_PORT}/v1"
export VLLM_API_KEY="${JUDGE_API_KEY}"

# shellcheck disable=SC1090
source "${VENV_PATH}"
for log in "${LOGS[@]}"; do
    echo "[run_judge_score] scoring ${log}"
    inspect score "${SCORE_ARGS[@]}" --overwrite "${log}"
done
