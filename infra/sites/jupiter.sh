# JSC JUPITER (booster partition: GH200, aarch64, 4 GPUs per node, whole nodes only).

# --- python environment ------------------------------------------------------
# A SMURF venv (NeMo + speechllm + melt-eval[metrics]).
# The wheels have to be aarch64 builds.
export VENV_PATH="${VENV_PATH:-${PROJECT:?run: jutil env activate -p <project>}/${USER}/venvs/smurf-eval/bin/activate}"
# Post-hoc metrics (MCIF's official scorers, COMET), in a venv of their own.
export METRICS_VENV_PATH="${METRICS_VENV_PATH:-${PROJECT}/${USER}/venvs/metrics-eval/bin/activate}"

# --- storage (host paths) ------------------------------------------------------
export HF_HOME="${HF_HOME:-${SCRATCH:?run: jutil env activate -p <project>}/${USER}/hf_cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/${USER}/eval-logs}"
# Where scripts/evaluate.py puts each campaign (configs/eval/*.yaml: output_dir).
export EVAL_ROOT="${EVAL_ROOT:-${SCRATCH}/${USER}/eval-runs}"
# The INDEXED copy of the Shar corpora (random access needs the .idx files).
export LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-${PROJECT}/${USER}/melt-data/shar-indexed}"
# The SMURF corpora (plain .jsonl.gz Shar: index a copy before freezing from it).
export AZEROS_DIR="${AZEROS_DIR:-/e/data1/smurf4eu-data/speech/azeros}"
# Its indexed copy (scripts/index_shar.py): manifests + .idx, audio symlinked back.
export AZEROS_INDEXED_DIR="${AZEROS_INDEXED_DIR:-${SCRATCH}/${USER}/azeros-indexed}"

# --- misc ------------------------------------------------------------------------
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# --- scheduler -------------------------------------------------------------------
# Nodes are allocated (and billed) whole, so scripts/evaluate.py run packs a
# model's slices into one job per node: one worker per GPU, the checkpoint
# loaded once per worker (infra/run_eval_pack.sbatch). A config's `pack:` block
# sets the pack's time and size; MELT_PACK_GPUS=0 goes back to one job per slice.
export MELT_PACK_GPUS="${MELT_PACK_GPUS:-4}"
# --gpus-per-node below is for single-slice jobs; infra/runners/submit_pack.sh
# replaces it with the pack's GPU count.
SBATCH_ARGS=(
    --account="${JUPITER_ACCOUNT:?export JUPITER_ACCOUNT=<your compute project>}"
    --partition="${MELT_PARTITION:-booster}"
    --time="${MELT_TIME:-02:00:00}"
    --nodes=1
    --gpus-per-node=1
    --cpus-per-task=72
)
