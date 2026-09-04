# Bocconi HPC cluster (short_gpuh200 / short_gpunew / short_gpua100, etc).
# See _my_docs/crosscheck_bocconi.md for the sinfo/sacctmgr breakdown this is
# based on.
#
# Compute nodes have outbound internet (unlike mn5), so HF_HUB_OFFLINE is not
# forced -- override it yourself once a cache is warm and you want to pin to
# it.

# --- python environment -------------------------------------------------
# The smurf-eval venv built per docs/replication_notes.md (fbk_speechllm +
# NeMo pinned <3.0.0 + peft/fiddle + melt-eval[hf,metrics]). Lives inside the
# repo on this cluster, not under $HOME/venvs.
export VENV_PATH="${VENV_PATH:-$HOME/repos/eval/.venv/bin/activate}"

# --- storage (host paths) -------------------------------------------------
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-$HOME/scratch/eval-logs}"
export TMPDIR="${TMPDIR:-$HOME/scratch/triton-eval}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$TMPDIR}"

# --- scheduler -------------------------------------------------------------
# short_gpuh200: H200, 141 GB -- the checkpoint loads whole on one GPU, no
# device_map anywhere (the cleanest 1:1 with upstream). Account `calvo` has
# both `debug` and `normal` QOS, no explicit --account needed.
SBATCH_ARGS=(
    --time="${MELT_TIME:-00:45:00}"
    --nodes=1
    --gpus-per-node=1
    --partition="${MELT_PARTITION:-short_gpuh200}"
    --qos="${MELT_QOS:-normal}"
    --cpus-per-task="${MELT_CPUS:-8}"
    --mem="${MELT_MEM:-64G}"
)
