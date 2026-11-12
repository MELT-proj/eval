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

# The job must import the `melteval` package of the checkout this site file
# lives in, not whatever the venv's last editable install pointed at. SLURM
# jobs inherit PYTHONPATH from the submit shell.
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"

# Post-hoc metrics (MCIF's official scorers, COMET), in a venv of their own —
# infra/setup/metrics_venv.sh builds it. No external binary: mcif-bench 2.0
# resegments with mweralign in-process.
export METRICS_VENV_PATH="${METRICS_VENV_PATH:-/mnt/scratch-artemis/giuseppe/venvs/metrics-eval/bin/activate}"

# --- storage (host paths) -----------------------------------------------------
# All under scratch, never $HOME: artemis $HOME is GlusterFS-synced and quota
# limited, and venvs/eval-logs/frozen-sets are exactly the kind of large,
# frequently-changing files that don't belong there.
export HF_HOME="${HF_HOME:-/mnt/scratch-artemis/giuseppe/melt-data/hf_cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/scratch-artemis/giuseppe/melt-data/eval-logs}"
# Where scripts/evaluate.py puts each campaign (configs/eval/*.yaml: output_dir).
export EVAL_ROOT="${EVAL_ROOT:-/mnt/scratch-artemis/giuseppe/melt-data/eval-runs}"
# sacrebleu's cache. Holds models/flores200sacrebleuspm, the SentencePiece model
# behind spBLEU, staged once from a machine with internet (README.md, "spBLEU").
# Jobs never download it: the ST task checks for it before generating and stops.
# sacrebleu reads this once, when it is imported.
export SACREBLEU="${SACREBLEU:-/mnt/scratch-artemis/giuseppe/sacrebleu}"
export LOCAL_DATASETS_DIR="${LOCAL_DATASETS_DIR:-/mnt/scratch-nyx/giuseppe/melt/melt-data/shar}"

# --- speechval-dev frozen sets (docs/speechval.md) ----------------------------
# Guarded so a partial tree fails in `plan` with "environment variable is not
# set", not silently. Paths are per-machine; on artemis the full dev sets were
# frozen on 2026-11-12 (docs/speechval.md, "Canary baseline").
_speechval_dev="${SPEECHVAL_DEV_ROOT:-/mnt/scratch-artemis/giuseppe/frozen/speechval-dev}"
if [[ -d "$_speechval_dev/voxpopuli-dev" ]]; then
    export VOXPOPULI_DEV_SET="${VOXPOPULI_DEV_SET:-$_speechval_dev/voxpopuli-dev}"
fi
if [[ -d "$_speechval_dev/fleurs24-st-xen-dev-full" ]]; then
    export FLEURS24_ST_XEN_DEV_SET="${FLEURS24_ST_XEN_DEV_SET:-$_speechval_dev/fleurs24-st-xen-dev-full}"
fi
if [[ -d "$_speechval_dev/fleurs24-st-enx-dev" ]]; then
    export FLEURS24_ST_ENX_DEV_SET="${FLEURS24_ST_ENX_DEV_SET:-$_speechval_dev/fleurs24-st-enx-dev}"
fi
if [[ -d "$_speechval_dev/st-eval-campaign-v1-dev" ]]; then
    export COVOST2_DEV_SET="${COVOST2_DEV_SET:-$_speechval_dev/st-eval-campaign-v1-dev}"
fi
if [[ -d "$_speechval_dev/librispeech-validation" ]]; then
    export LIBRISPEECH_VALIDATION_SET="${LIBRISPEECH_VALIDATION_SET:-$_speechval_dev/librispeech-validation}"
fi
if [[ -d /mnt/scratch-artemis/giuseppe/frozen/speechval/europarl-st-dev ]]; then
    export EUROPARL_ST_DEV_SET="${EUROPARL_ST_DEV_SET:-/mnt/scratch-artemis/giuseppe/frozen/speechval/europarl-st-dev}"
fi
unset _speechval_dev

# Where inspect_ai keeps its own state (traces, view assets). It defaults to
# $HOME/.local/share, which is /mnt/home here -- churn on a GlusterFS-synced
# quota, and on some of this site's GPU nodes not even writable: a job there
# dies during logger setup with "PermissionError: [Errno 13] Permission
# denied: '/mnt/home'", before it reads a single sample. Pointing
# XDG_DATA_HOME at scratch fixes both, and makes every partition behave the
# same way.
export XDG_DATA_HOME="${XDG_DATA_HOME:-/mnt/scratch-artemis/giuseppe/.local/share}"

# --- misc ----------------------------------------------------------------------
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# --- scheduler -------------------------------------------------------------
# gpu-short (a6000, up to 4h) is the default -- long enough for a real frozen
# set, short enough to schedule quickly. Override per job:
#   MELT_QOS=gpu-debug MELT_TIME=00:20:00 infra/runners/submit_eval.sh artemis …
#
# The a6000 partition is the busiest of this site's partitions and is
# regularly fully allocated to multi-day jobs. Other partitions take their own
# QoS names, not the gpu-short/gpu-debug tiers, so a partition override needs
# a QoS override with it -- check `sinfo`/`sacctmgr` for this site's names:
#   MELT_PARTITION=<partition> MELT_QOS=<qos> infra/runners/submit_eval.sh artemis …
SBATCH_ARGS=(
    --time="${MELT_TIME:-04:00:00}"
    --nodes=1
    --gpus-per-node=1
    --partition="${MELT_PARTITION:-a6000}"
    --qos="${MELT_QOS:-gpu-short}"
)
