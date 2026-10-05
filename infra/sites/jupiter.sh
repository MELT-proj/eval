# JSC JUPITER (booster partition: GH200, aarch64, 4 GPUs per node, whole nodes only).

# --- python environment ------------------------------------------------------
# A SMURF venv (NeMo + speechllm + melt-eval[metrics]).
# The wheels have to be aarch64 builds.
export VENV_PATH="${VENV_PATH:-${PROJECT:?run: jutil env activate -p <project>}/${USER}/venvs/smurf-eval/bin/activate}"

# --- storage (host paths) ------------------------------------------------------
export HF_HOME="${HF_HOME:-${SCRATCH:?run: jutil env activate -p <project>}/${USER}/hf_cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH}/${USER}/eval-logs}"
# Where scripts/evaluate.py puts each campaign (configs/eval/*.yaml: output_dir).
export EVAL_ROOT="${EVAL_ROOT:-${SCRATCH}/${USER}/eval-runs}"
# The INDEXED copy of the Shar corpora (random access needs the .idx files).
export LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-${PROJECT}/${USER}/melt-data/shar-indexed}"

# --- misc ------------------------------------------------------------------------
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# --- scheduler -------------------------------------------------------------------
SBATCH_ARGS=(
    --account="${JUPITER_ACCOUNT:?export JUPITER_ACCOUNT=<your compute project>}"
    --partition="${MELT_PARTITION:-booster}"
    --time="${MELT_TIME:-02:00:00}"
    --nodes=1
    --gpus-per-node=1
    --cpus-per-task=72
)
