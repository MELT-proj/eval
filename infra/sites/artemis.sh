# Artemis workstation (a6000 / h100 / h200 partitions).
#
# GPU work on artemis is shared and must go through SLURM -- never run
# `inspect eval` (or anything else that touches a GPU) directly on the login
# shell, even for a quick check. See ../../AGENTS.md.

# --- python environment ------------------------------------------------------
# A venv with melt-eval + inspect_ai + torch + melt-proj + lhotse, all in one
# environment (no site had all of these already, so this one exists purely for
# melt-eval; see README.md "Install" for how to rebuild it).
export VENV_PATH="${VENV_PATH:-/mnt/scratch-artemis/giuseppe/venvs/melteval/bin/activate}"

# --- storage (host paths) -----------------------------------------------------
# All under scratch, never $HOME: artemis $HOME is GlusterFS-synced and quota
# limited, and venvs/eval-logs/frozen-sets are exactly the kind of large,
# frequently-changing files that don't belong there.
export HF_HOME="${HF_HOME:-/mnt/scratch-artemis/giuseppe/melt-data/hf_cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/scratch-artemis/giuseppe/melt-data/eval-logs}"
export LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/mnt/scratch-nyx/giuseppe/melt/melt-data/shar}"

# --- misc ----------------------------------------------------------------------
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# --- scheduler -------------------------------------------------------------
# gpu-short (a6000, up to 4h) is the default -- long enough for a real frozen
# set, short enough to schedule quickly. Override per job:
#   MELT_QOS=gpu-debug MELT_TIME=00:20:00 infra/runners/submit_eval.sh artemis …
SBATCH_ARGS=(
    --time="${MELT_TIME:-04:00:00}"
    --nodes=1
    --gpus-per-node=1
    --partition="${MELT_PARTITION:-a6000}"
    --qos="${MELT_QOS:-gpu-short}"
)
