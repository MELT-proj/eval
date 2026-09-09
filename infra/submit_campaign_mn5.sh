#!/bin/bash
# Submit the ASR+ST eval campaign for every checkpoint of one training run:
# the run root plus every checkpoint-N subfolder.
#
# **One job per (task, corpus, language).** Not one job per task. Those are the
# units the report in projects/melt is built from, so they have to be
# aggregated across logs either way, and splitting there buys two things:
#
#  * Wall time that fits. The single-job form asks for the whole frozen set at
#    once, and that estimate is model-dependent in a way an sbatch default
#    cannot track: llama-1b got through the 161k-sample ASR set in ~5 h, qwen-2b
#    needs ~21 h. The first campaign was sized from the former and the second
#    one's ASR jobs were all going to be killed at the wall. Per unit the
#    largest cell is cv22_sidon/en at 16.4k samples -- ~2 h even for the slow
#    model -- so one ceiling covers both.
#  * Backfill. MN5's scheduler slots a 4 h job into gaps an 18 h job cannot
#    fit, so shorter requests start sooner even at equal priority.
#
# The cost is one model load per unit (~2-5 min against hours of decode) and
# more jobs in the queue: 34 per checkpoint, 272 for an eight-checkpoint run.
# acc_ehpc allows 366 submitted per user, so submit one run at a time.
#
# The units come from the frozen set's own manifest, not from a list here: a
# corpus added to configs/*.yaml shows up in the campaign by rebuilding the
# frozen set, with nothing to keep in sync.
#
# Run this ON MN5 from the eval checkout:
#   RUN_ROOT=/gpfs/scratch/epor48/outputs/<run> ./infra/submit_campaign_mn5.sh
set -euo pipefail

RUN_ROOT="${RUN_ROOT:?usage: RUN_ROOT=/path/to/run ./infra/submit_campaign_mn5.sh}"
RUN_NAME=$(basename "$RUN_ROOT")
ASR_SET="${ASR_SET:-/gpfs/scratch/epor48/itpt955676/eval-sets/asr-eval-campaign-v1}"
ST_SET="${ST_SET:-/gpfs/scratch/epor48/itpt955676/eval-sets/st-eval-campaign-v1}"
SIF="${MELT_SIF:-/gpfs/scratch/epor48/itpt955676/melt_eval_cuda126_v2.sif}"

# One subfolder per checkpoint, so `inspect view --log-dir <folder>` scopes to
# one checkpoint and the report can still walk the whole tree at once.
LOG_ROOT="${LOG_ROOT:-/gpfs/scratch/epor48/itpt955676/campaign-logs/${RUN_NAME}}"

# Sized from the slowest model seen so far (qwen-2b, ~7.6k samples/hour) against
# the largest single unit (16.4k samples), with margin. Raise ASR_TIME rather
# than lower it if a bigger model lands: the failure mode is silent, a killed
# job leaves a partial log, and only `inspect eval-retry` gets that work back.
ASR_TIME="${ASR_TIME:-04:00:00}"
ST_TIME="${ST_TIME:-03:00:00}"
BATCH_SIZE="${BATCH_SIZE:-16}"

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
echo "Checkpoints: ${#CHECKPOINTS[@]}"
echo "Units:      ${#ASR_UNITS[@]} ASR + ${#ST_UNITS[@]} ST = $(( (${#ASR_UNITS[@]} + ${#ST_UNITS[@]}) * ${#CHECKPOINTS[@]} )) jobs"

submit() {
    local ckpt="$1" name="$2" set_dir="$3" time="$4" task="$5" dataset_id="$6" lang="$7"
    local args=(-T "task_filter=${task}")
    [[ -n "$dataset_id" ]] && args+=(-T "dataset_id=${dataset_id}")
    [[ -n "$lang" ]] && args+=(-T "lang=${lang}")
    # Sub-checkpoints carry no tokenizer/processor/training config of their own
    # -- only the run root does -- so every job points those at the run root.
    # Identical, and harmless, for the run root itself.
    args+=(
        -M "processor=${RUN_ROOT}"
        -M "batch_size=${BATCH_SIZE}"
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
