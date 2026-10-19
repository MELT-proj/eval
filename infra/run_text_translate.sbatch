#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output="logs/%x.%j.out"
#SBATCH --job-name=melt-text-translate
#
# Text-only translation baseline: a text LLM translates the reference transcript of a frozen ST set
# (scripts/text_translate.py). One GPU. The environment comes from a site file (VENV_PATH, HF_HOME,
# SACREBLEU), e.g. `source infra/sites/artemis.sh`; pass the partition and QOS on the sbatch line.
#
# Usage (host side, from the repo root, after sourcing the site file):
#   sbatch --partition=<p> --qos=<q> --gres=gpu:1 --time=04:00:00 infra/run_text_translate.sbatch \
#       <hf model id> <frozen set dir> <output dir> [--batch-size 64] [--limit N]
set -euo pipefail
source "${VENV_PATH:?source a site file first}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python scripts/text_translate.py --model "$1" --set "$2" --out "$3" "${@:4}"
