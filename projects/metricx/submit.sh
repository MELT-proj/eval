#!/bin/bash
# Submit MetricX-24 scoring of finished ST logs as one GPU job (the 26 GB model is loaded once per job).
#
#   projects/metricx/submit.sh OUT_ROOT NAME=LOG [NAME=LOG ...]      # any number of runs, any sets
#   QOS=gpu-h200 PARTITION=h200 projects/metricx/submit.sh OUT_ROOT NAME=LOG
#
# Runs whose summary.json already exists under OUT_ROOT are skipped, so rescoring one more backbone is the same
# command with just its two logs. Submit from a clone on /mnt/scratch-artemis: the h200 node cannot see /mnt/home.
set -euo pipefail
OUT_ROOT=$1; shift
REPO=$(cd "$(dirname "$0")/../.." && pwd)
VENVS=${VENVS:-/mnt/scratch-artemis/giuseppe/venvs}
PARTITION=${PARTITION:-h100}
QOS=${QOS:-gpu-h100}
TIME=${TIME:-04:00:00}
mkdir -p "$OUT_ROOT/slurm"
RUNS=(); for spec in "$@"; do RUNS+=(--run "$spec"); done
sbatch --partition="$PARTITION" --qos="$QOS" --gres=gpu:1 --cpus-per-task=8 --mem=96G --time="$TIME" \
  --job-name=metricx --output="$OUT_ROOT/slurm/%x.%j.out" --export=ALL,HF_HOME="${HF_HOME:-/mnt/scratch-artemis/giuseppe/melt-data/hf_cache}",HF_HUB_OFFLINE=1 \
  --wrap "$VENVS/metricx/bin/python $REPO/projects/metricx/score.py --out-root $OUT_ROOT ${RUNS[*]} ${SCORE_ARGS:-}"
