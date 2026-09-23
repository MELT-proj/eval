# BSC MareNostrum5 (acc partition). Air-gapped compute nodes: run offline.
#
# Use CONTAINER MODE (infra/run_eval_container_mn5.sbatch), not this file's
# venv. VENV_PATH below still does not exist -- MN5 has no outbound internet,
# so a venv with inspect_ai + torch + melt-proj + lhotse has to be built once
# on a machine that does and copied over -- and nobody has done that; a job
# that tries to use it will fail fast in the sbatch script with a clear
# message rather than hang. Container mode is proven working (smoke-tested
# 2026-09-22/23, see the campaign board): build infra/Singularity.def
# elsewhere (it needs internet, e.g. nyx) with training checked out as a
# sibling, and copy the .sif here -- promote it under its own name, prove it
# with one run, then `ln -sfn` it to the MELT_SIF default below; never
# overwrite a .sif a queued job may be reading. Rebuild whenever the training
# side gains an axis that changes checkpoint shapes (e.g. stack_factor,
# PR #126) -- an old image will fail to load the new checkpoints' weights.
#
# See training's infra/sync_repo.sh for the push mechanism (git-over-SSH; MN5
# is a normal git remote, not a place code is copied to by hand). melt-eval
# has no sync_repo.sh of its own yet -- push directly:
#   git push mn5:eval HEAD:refs/heads/<branch>
# after enabling `git -C eval config receive.denyCurrentBranch updateInstead`
# on the remote once (sync_repo.sh --init does this for training; do the same
# by hand here until melt-eval gets its own copy of the script).

# --- python environment ------------------------------------------------------
export VENV_PATH="${VENV_PATH:-/gpfs/scratch/epor48/venvs/melteval/bin/activate}"

# --- container mode (infra/run_eval_container_mn5.sbatch) --------------------
# The venv above still doesn't exist (see the note at the top of this file);
# the container path works today -- build infra/Singularity.def elsewhere
# (it needs internet) and copy the .sif here. Overridable per-run:
#   MELT_SIF=/path/to/other.sif sbatch infra/run_eval_container_mn5.sbatch …
export MELT_SIF="${MELT_SIF:-/gpfs/scratch/epor48/melt_eval_cuda126.sif}"

# --- storage (host paths) -----------------------------------------------------
# The project is shared by several accounts; a directory here belongs to
# whoever created it. Set your own OUTPUT_DIR if you don't have write access to
# the default. LOCAL_DATASETS_DIR and HF_HOME are read-only in a run.
export HF_HOME="${HF_HOME:-/gpfs/scratch/epor48/hf_cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/scratch/epor48/eval-logs}"
# The INDEXED copy, not plain `shar`: generation does random-access reads
# (batched, out of shard order), and a plain Shar tree has no .idx sidecars,
# so the reader would rescan a shard per sample and raise rather than do that
# silently. `melteval freeze` itself reads sequentially and works against
# either tree -- this only bites at `inspect eval` time (smoke-tested
# 2026-09-22/23, see the campaign board).
export LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/gpfs/projects/epor48/melt-data/shar-indexed}"

# --- code sync (see training/infra/sync_repo.sh for the pattern) -------------
export REMOTE_SSH=mn5
export REMOTE_REPO=eval

# --- misc ----------------------------------------------------------------------
# No internet on compute nodes: everything must already be cached/staged.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- scheduler -------------------------------------------------------------
SBATCH_ARGS=(
    --time="${MELT_TIME:-01:00:00}"
    --nodes=1
    --gpus-per-node=1
    --account=epor48
    --qos="${MELT_QOS:-acc_debug}"
    --cpus-per-task=20
)
