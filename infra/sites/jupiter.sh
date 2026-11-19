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

# --- PolySpeech (dev SQA/SLU); reconstruction and frozen set live on Jupiter ---
if [[ -d /e/data1/smurf4eu-data/speech/polyspeech-100 ]]; then
    export POLYSPEECH_ROOT="${POLYSPEECH_ROOT:-/e/data1/smurf4eu-data/speech/polyspeech-100}"
fi
_polyspeech_frozen="${POLYSPEECH_FROZEN_ROOT:-${PROJECT}/${USER}/frozen-dev/polyspeech-100-pa_qt-200}"
if [[ -d "$_polyspeech_frozen" ]]; then
    export POLYSPEECH_PA_QT_SET="${POLYSPEECH_PA_QT_SET:-$_polyspeech_frozen}"
fi

# --- speechval-dev frozen sets (dev partition of the baseline matrix) -----------
# The four Shar sets are in the shared data1 tree (frozen 2026-11-10); the sets
# frozen on Jupiter later -- LibriSpeech validation, Europarl -- are per-user.
# Locators are absolute paths, so a set is only usable on the machine it was
# frozen on: each export is guarded, and a site without them behaves as before
# (the suite's slice fails loudly at plan time instead of reading someone
# else's audio). Recipes: docs/speechval.md, "Freezing (CPU only)".
_dev_roots=(
    "${SPEECHVAL_DEV_FROZEN_ROOT:-}"
    /e/data1/smurf4eu-data/speech/frozen-dev
    "${PROJECT}/${USER}/frozen-dev"
)
_speechval_set() {
    local var=$1 name=$2 root
    for root in "${_dev_roots[@]}"; do
        if [[ -n "$root" && -d "$root/$name" ]]; then
            export "$var=${!var:-$root/$name}"
            return
        fi
    done
}
_speechval_set FLEURS24_ST_XEN_DEV_SET fleurs24-st-xen-dev-full
_speechval_set FLEURS24_ST_ENX_DEV_SET fleurs24-st-enx-dev
_speechval_set COVOST2_DEV_SET st-eval-campaign-v1-dev
_speechval_set VOXPOPULI_DEV_SET voxpopuli-dev
_speechval_set LIBRISPEECH_VALIDATION_SET librispeech-validation
_speechval_set EUROPARL_ST_DEV_SET europarl-st-dev

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
