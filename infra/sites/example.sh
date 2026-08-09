# Site config template. Copy this to sites/<name>.sh and fill in the values.
#
# A site file is sourced by infra/runners/submit_eval.sh. It contains ONLY
# `export`s (inherited by the SLURM job) and one SBATCH_ARGS array (passed to
# sbatch). No logic, no eval-specific args — those are CLI arguments to
# submit_eval.sh / inspect eval itself.

# --- python environment -----------------------------------------------------
# A venv with melt-eval installed (`uv pip install -e ".[shar,metrics]"`)
# alongside inspect_ai, torch, melt-proj and lhotse -- everything `inspect eval`
# needs to run the `melt` provider. Point this at wherever that venv lives; it
# does not have to be inside the repo.
export VENV_PATH="${VENV_PATH:-/path/to/venvs/melteval/bin/activate}"

# --- storage (host paths) ---------------------------------------------------
# Write these as "${VAR:-default}" so a value exported by the caller wins:
#   OUTPUT_DIR=/my/eval-logs infra/runners/submit_eval.sh <site> …
export HF_HOME="${HF_HOME:-/path/to/hf_cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-/path/to/eval-logs}"    # where inspect writes .eval logs

# --- misc --------------------------------------------------------------------
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# --- scheduler ----------------------------------------------------------------
# Evals are single-GPU, single-node, and short relative to training -- there is
# no multi-node case to support.
SBATCH_ARGS=(--time=01:00:00 --nodes=1 --gpus-per-node=1)
