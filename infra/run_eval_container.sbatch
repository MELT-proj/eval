#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output="logs/%x.%j.out"
#SBATCH --job-name=melt-eval-container
#
# Same job as run_eval.sbatch, but runs inside the infra/Singularity.def
# image (via apptainer) instead of a hand-built venv -- a machine only needs
# apptainer, not a melteval venv assembled with `uv sync`. See README.md
# "Install" and infra/setup/build_singularity.sh for how the image is built.
#
# Not usually invoked directly: infra/run_eval.sbatch's plain-venv path
# (infra/runners/submit_eval.sh) is the primary interface. This is for a
# machine that only has the image, not a hand-built venv.
#
# Usage (host side -- mkdir logs first, sbatch won't create it):
#   MELT_SIF=/path/to/melt_eval_cuda126.sif OUTPUT_DIR=/path/to/eval-logs \
#     sbatch [slurm flags] infra/run_eval_container.sbatch <checkpoint_dir> <frozen_set_dir> [inspect eval args...]
#
# Example (gpu-debug QoS, a6000 partition, a few samples):
#   MELT_SIF=/mnt/scratch-artemis/giuseppe/melt-data/melt_eval_cuda126.sif \
#   OUTPUT_DIR=/mnt/scratch-artemis/giuseppe/melt-data/eval-logs \
#     sbatch --partition=a6000 --qos=gpu-debug --gpus-per-node=1 --time=00:15:00 \
#     infra/run_eval_container.sbatch \
#     /mnt/scratch-artemis/giuseppe/melt-data/outputs/MA-v1.2.7 \
#     /mnt/scratch-artemis/giuseppe/melt-data/eval-sets/librispeech \
#     -T task_filter=asr -T limit=5

set -euo pipefail

echo "[run_eval_container] host-side setup"

if [[ -z "${1:-}" || -z "${2:-}" ]]; then
    echo "Usage: sbatch $0 <checkpoint_dir> <frozen_set_dir> [inspect eval args...]"
    echo "Example: sbatch $0 /path/to/outputs/MA-v1.2.7 /path/to/eval-sets/asr-test-v1 -T task_filter=asr"
    exit 2
fi
CHECKPOINT="$1"
FROZEN_SET="$2"
shift 2
EXTRA_ARGS=("$@")

MELT_SIF="${MELT_SIF:-}"
if [[ -z "${MELT_SIF}" ]]; then
    echo "ERROR: MELT_SIF is not set. Point it at the image built by"
    echo "       infra/setup/build_singularity.sh."
    exit 2
fi
if [[ ! -f "${MELT_SIF}" ]]; then
    echo "ERROR: MELT_SIF not found: ${MELT_SIF}"
    exit 2
fi

if [[ ! -d "${CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint directory not found: ${CHECKPOINT}"
    exit 2
fi
if [[ ! -d "${FROZEN_SET}" ]]; then
    echo "ERROR: frozen set directory not found: ${FROZEN_SET}"
    echo "       Build one first with: melteval freeze <spec.yaml> -o <frozen_set_dir>"
    exit 2
fi

OUTPUT_DIR="${OUTPUT_DIR:-./eval-logs}"
mkdir -p "${OUTPUT_DIR}" logs

echo "[run_eval_container] image:      ${MELT_SIF}"
echo "[run_eval_container] checkpoint: ${CHECKPOINT}"
echo "[run_eval_container] frozen set: ${FROZEN_SET}"
echo "[run_eval_container] output dir: ${OUTPUT_DIR}"
echo "[run_eval_container] extra args: ${EXTRA_ARGS[*]:-(none)}"

# --nv: map in the host GPU driver (the image's CUDA toolkit is only there
#   for JIT builds -- see infra/Singularity.def -- the driver itself still
#   has to come from the host).
# --bind /mnt: frozen sets, checkpoints and datasets live under
#   /mnt/scratch-* (see AGENTS.md -- nothing heavy goes under $HOME), each
#   its own NFS mount; apptainer's bind is recursive, so binding /mnt once
#   carries all of them in rather than needing one --bind per path.
# The venv at /workspace/venv and repo at /workspace/eval are both baked
# into the image at build time -- this does not read the host melt-eval
# checkout at all, so the job sees exactly what was in the image.
exec apptainer exec --nv --bind /mnt "${MELT_SIF}" \
    bash -lc '
        set -euo pipefail
        source /workspace/venv/bin/activate
        cd /workspace/eval
        exec inspect eval melteval/tasks.py@speech \
            --model "melt/$1" \
            --log-dir "$2" \
            -T "frozen_set=$3" \
            "${@:4}"
    ' _ "${CHECKPOINT}" "${OUTPUT_DIR}" "${FROZEN_SET}" "${EXTRA_ARGS[@]}"
