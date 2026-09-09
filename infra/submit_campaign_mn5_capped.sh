#!/bin/bash
# Capped variant of submit_campaign_mn5.sh: caps every (task, corpus, language)
# unit at CAP samples (default 2000, taken in manifest order -- see
# `melteval.dataset.frozen_dataset`'s `limit`) instead of running the corpus's
# full test split.
#
# Why a cap. Once the campaign was already split one job per unit
# (submit_campaign_mn5.sh), most of what a full run buys over a 2000-sample
# one is a tighter confidence interval on corpora that are already tens of
# thousands of samples -- cv22_sidon/en alone is 16.4k. 2000 is enough to see
# real signal per checkpoint without paying for the last few thousand samples
# of precision on every unit, every checkpoint.
#
# Why bfloat16 and batch_size=4 by default here: larger batches showed
# instability in later testing than the batch_size=16 the first two campaigns
# ran at. `dtype` is passed explicitly even though it already matches the
# provider's own default (melteval/providers/melt.py) so a future default
# change there does not silently change what this script runs. The uncapped
# sibling still defaults to batch_size=16 -- its wall times were sized from
# real batch_size=16 throughput, and there is not yet a real batch_size=4
# throughput number to size a full-scale (up to 16.4k samples/unit) run's
# wall time from. Revisit both scripts' defaults together once one exists.
#
# Writes to a LOG_ROOT under campaign-logs-capped2000/, not campaign-logs/ --
# a *different* directory tree from the full-scale campaign, deliberately.
# projects/melt's report reads every log under whatever root it is pointed at
# and has no notion of "these two logs for the same checkpoint used different
# sample counts, don't average them" -- so a capped and a full run must never
# share a log root, and a report run on one must not be pointed at the other.
#
# Run this ON MN5 from the eval checkout:
#   RUN_ROOT=/gpfs/scratch/epor48/outputs/<run> ./infra/submit_campaign_mn5_capped.sh
set -euo pipefail

RUN_ROOT="${RUN_ROOT:?usage: RUN_ROOT=/path/to/run ./infra/submit_campaign_mn5_capped.sh}"
RUN_NAME=$(basename "$RUN_ROOT")
ASR_SET="${ASR_SET:-/gpfs/scratch/epor48/itpt955676/eval-sets/asr-eval-campaign-v1}"
ST_SET="${ST_SET:-/gpfs/scratch/epor48/itpt955676/eval-sets/st-eval-campaign-v1}"
SIF="${MELT_SIF:-/gpfs/scratch/epor48/itpt955676/melt_eval_cuda126_v2.sif}"

LOG_ROOT="${LOG_ROOT:-/gpfs/scratch/epor48/itpt955676/campaign-logs-capped2000/${RUN_NAME}}"

CAP="${CAP:-2000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DTYPE="${DTYPE:-bfloat16}"

# A 2000-sample cap and batch_size=4 shrink every unit far below the 4 h this
# script's uncapped sibling budgets for its 16.4k-sample worst case -- expect
# well under an hour per unit even on the slower of the two models seen so
# far. Kept at 1.5 h anyway, not trimmed further: this is the first campaign
# run at batch_size=4, so there is no real throughput measurement for it yet
# to size a tighter number from. Revisit once one exists (see how ASR_TIME in
# submit_campaign_mn5.sh was derived from the first campaign's real numbers).
ASR_TIME="${ASR_TIME:-01:30:00}"
ST_TIME="${ST_TIME:-01:30:00}"

ACCOUNT="${ACCOUNT:-epor48}"
QOS="${QOS:-acc_ehpc}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f melteval/tasks.py ]] || { echo "ERROR: run this from the melt-eval repo root."; exit 2; }
[[ -d "$RUN_ROOT" ]] || { echo "ERROR: no such run root: $RUN_ROOT"; exit 2; }

# Unique (task, dataset_id, lang) triples actually present in a frozen set.
units_of() {
    python3 - "$1" <<'PY'
import json, sys
seen = []
with open(f"{sys.argv[1]}/manifest.jsonl", encoding="utf-8") as fh:
    for line in fh:
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r.get("task", ""), r.get("dataset_id", ""), r.get("lang", ""))
        if key not in seen:
            seen.append(key)
print("\n".join("\t".join(k) for k in seen))
PY
}

CHECKPOINTS=("$RUN_ROOT")
for d in "$RUN_ROOT"/checkpoint-*; do
    [ -d "$d" ] && CHECKPOINTS+=("$d")
done

mapfile -t ASR_UNITS < <(units_of "$ASR_SET")
mapfile -t ST_UNITS < <(units_of "$ST_SET")

echo "Run:        $RUN_NAME"
echo "Log root:   $LOG_ROOT"
echo "Cap:        $CAP samples/unit, batch_size=$BATCH_SIZE, dtype=$DTYPE"
echo "Checkpoints: ${#CHECKPOINTS[@]}"
echo "Units:      ${#ASR_UNITS[@]} ASR + ${#ST_UNITS[@]} ST = $(( (${#ASR_UNITS[@]} + ${#ST_UNITS[@]}) * ${#CHECKPOINTS[@]} )) jobs"

submit() {
    local ckpt="$1" name="$2" set_dir="$3" time="$4" task="$5" dataset_id="$6" lang="$7"
    local args=(-T "task_filter=${task}" -T "limit=${CAP}")
    [[ -n "$dataset_id" ]] && args+=(-T "dataset_id=${dataset_id}")
    [[ -n "$lang" ]] && args+=(-T "lang=${lang}")
    # Sub-checkpoints carry no tokenizer/processor/training config of their own
    # -- only the run root does -- so every job points those at the run root.
    # Identical, and harmless, for the run root itself.
    args+=(
        -M "processor=${RUN_ROOT}"
        -M "batch_size=${BATCH_SIZE}"
        -M "dtype=${DTYPE}"
        -T "tokenizer=${RUN_ROOT}"
        -T "format_config=${RUN_ROOT}/training_config.yaml"
        --tags "${name}"
    )
    if [[ "$DRY_RUN" != "0" ]]; then
        echo "DRY: ${name} ${task} ${dataset_id} ${lang}"
        return
    fi
    MELT_SIF="$SIF" OUTPUT_DIR="${LOG_ROOT}/${name}" \
        sbatch --account="$ACCOUNT" --qos="$QOS" --nodes=1 --gpus-per-node=1 \
        --cpus-per-task=20 --time="$time" \
        infra/run_eval_container_mn5.sbatch "$ckpt" "$set_dir" "${args[@]}"
}

for CKPT in "${CHECKPOINTS[@]}"; do
    NAME=$(basename "$CKPT")
    [[ "$CKPT" == "$RUN_ROOT" ]] && NAME="final"

    for unit in "${ASR_UNITS[@]}"; do
        IFS=$'\t' read -r task dataset_id lang <<<"$unit"
        echo "=== ${NAME}: ${task} ${dataset_id}/${lang} ==="
        submit "$CKPT" "$NAME" "$ASR_SET" "$ASR_TIME" "$task" "$dataset_id" "$lang"
    done

    for unit in "${ST_UNITS[@]}"; do
        IFS=$'\t' read -r task dataset_id lang <<<"$unit"
        echo "=== ${NAME}: ${task} ${dataset_id}/${lang} ==="
        submit "$CKPT" "$NAME" "$ST_SET" "$ST_TIME" "$task" "$dataset_id" "$lang"
    done
done
