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

# MELT checkpoints need melt-proj (from a sibling ../training checkout), whose
# `shar` extra pins lhotse/torch differently from the NeMo stack above, so MELT
# gets its own venv here. configs/eval/*.yaml points MELT models at it; for a
# one-off run, override VENV_PATH with it.
#   uv venv --python 3.12 $HOME/repos/eval/.venv-melt
#   VIRTUAL_ENV=$HOME/repos/eval/.venv-melt uv pip install --prerelease=allow -e ".[shar,hf,metrics]"
export MELT_VENV_PATH="${MELT_VENV_PATH:-$HOME/repos/eval/.venv-melt/bin/activate}"

# --- storage (host paths) -------------------------------------------------
# The HF cache lives on BeeGFS (/scratch -> /mnt/beegfsnew/scratch), not under
# $HOME: /home has a 180 GB per-user quota, and one baseline's weights
# (Qwen3-Omni, ~70 GB) plus a benchmark's parquet fill most of it. Keep the
# login shell on the same cache (`export HF_HOME=...` in ~/.bashrc), or a
# download made there is not the one the job reads.
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-$HOME/scratch/eval-logs}"
# Where scripts/evaluate.py puts each campaign (configs/eval/*.yaml: output_dir).
export EVAL_ROOT="${EVAL_ROOT:-$HOME/scratch/eval-runs}"
export TMPDIR="${TMPDIR:-$HOME/scratch/triton-eval}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$TMPDIR}"

# --- scheduler -------------------------------------------------------------
# short_gpuh200: H200, 141 GB -- the checkpoint loads whole on one GPU, no
# device_map anywhere (the cleanest 1:1 with upstream). Account `calvo` has
# both `debug` and `normal` QOS, no explicit --account needed.
#
# The H200 partitions differ only in wall-clock ceiling: short_gpuh200 1h10,
# gpuh200 1 day, long_gpuh200 3 days (`scontrol show partition`). A job asking
# for more than its partition allows is rejected at submission, so unless
# MELT_PARTITION is given, pick the shortest partition that fits MELT_TIME
# (shorter partitions have more nodes and schedule sooner).
#
# QOS `normal` caps a user at 30 submitted jobs (pending + running) and 10
# running (`sacctmgr show qos`). scripts/evaluate.py reads MAX_QUEUED and
# waits for room before submitting past it, so a large matrix does not fail
# half-way at the 31st sbatch.
export MAX_QUEUED="${MAX_QUEUED:-28}"

# Default: the most short_gpuh200 allows. A time limit is only a ceiling -- a
# job that needs 6 minutes frees the GPU after 6 -- and asking for more would
# push every job onto the busier day-long partition. Slices that need longer
# say so with `time:` in the evaluation config.
_melt_time="${MELT_TIME:-01:10:00}"
_melt_seconds() {  # [D-]HH:MM:SS -> seconds
    local t="$1" d=0
    [[ "$t" == *-* ]] && { d="${t%%-*}"; t="${t#*-}"; }
    IFS=: read -r h m s <<<"$t"
    echo $(( d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#${s:-0} ))
}
if [[ -z "${MELT_PARTITION:-}" ]]; then
    if (( $(_melt_seconds "$_melt_time") <= 70 * 60 )); then
        MELT_PARTITION=short_gpuh200
    elif (( $(_melt_seconds "$_melt_time") <= 24 * 3600 + 10 * 60 )); then
        MELT_PARTITION=gpuh200
    else
        MELT_PARTITION=long_gpuh200
    fi
fi

SBATCH_ARGS=(
    --time="${_melt_time}"
    --nodes=1
    --gpus-per-node=1
    --partition="${MELT_PARTITION}"
    --qos="${MELT_QOS:-normal}"
    --cpus-per-task="${MELT_CPUS:-8}"
    --mem="${MELT_MEM:-64G}"
)
