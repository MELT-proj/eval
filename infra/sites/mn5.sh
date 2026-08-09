# BSC MareNostrum5 (acc partition). Air-gapped compute nodes: run offline.
#
# NOT YET USABLE as shipped: VENV_PATH below does not exist. MN5 has no
# outbound internet, so a venv with inspect_ai + torch + melt-proj + lhotse
# has to be built once on a machine that does and copied over (or built inside
# the existing training .sif and pip-installed from wheels staged in advance).
# Until that venv exists at the path below (or you override VENV_PATH), a job
# submitted with this site file will fail fast in the sbatch script with a
# clear message rather than hang.
#
# See training's infra/sync_repo.sh for the push mechanism (git-over-SSH; MN5
# is a normal git remote, not a place code is copied to by hand).

# --- python environment ------------------------------------------------------
export VENV_PATH="${VENV_PATH:-/gpfs/scratch/epor48/venvs/melteval/bin/activate}"

# --- storage (host paths) -----------------------------------------------------
# The project is shared by several accounts; a directory here belongs to
# whoever created it. Set your own OUTPUT_DIR if you don't have write access to
# the default. LOCAL_DATASETS_DIR and HF_HOME are read-only in a run.
export HF_HOME="${HF_HOME:-/gpfs/scratch/epor48/hf_cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/scratch/epor48/eval-logs}"
export LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/gpfs/projects/epor48/melt-data/shar}"

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
