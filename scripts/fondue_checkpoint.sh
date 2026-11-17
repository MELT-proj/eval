#!/usr/bin/env bash
# Schedule the full speechval-dev eval (ASR + ST) for one Fondue checkpoint,
# then score its ST logs with MetricX.
#
#   scripts/fondue_checkpoint.sh 33000              # schedule, wait, score
#   scripts/fondue_checkpoint.sh 33000 --no-score   # eval only
#   scripts/fondue_checkpoint.sh 33000 --dry-run    # print the plan, change nothing
#   scripts/fondue_checkpoint.sh 33000 --config-only  # add the model entry, stop
#
# A `gpu-short` job is capped at 4 h while the whole battery needs about 5 h, so
# the eval goes out as three packs that each fit under the cap, one after the
# other, and any phase that ran out of time is resubmitted in another round
# (`run` only ever submits slices that have no successful log). Scoring runs
# once at the end: `evaluate.py score` submits one MetricX job per ST benchmark
# and skips the ASR logs, which declare no post scorer.
#
# Run this from the melt-eval checkout. When this host has no SLURM client the
# launcher is driven over ssh; when it has one (artemis itself), it runs in
# place.
#
# Environment overrides:
#   MELT_SITE              host to drive over ssh        (default: artemis)
#   MELT_MIRROR            melt-eval checkout there      (default: /mnt/scratch-artemis/giuseppe/melt-eval-speechval)
#   MELT_FONDUE_CONFIG     campaign config to edit       (default: $EVAL_ROOT/speechval-fondue/config.yaml)
#   MELT_FONDUE_CKPT_ROOT  directory holding checkpoint-<N>
#   MELT_TIME              per-pack wall time            (default: 03:00:00)
#   MELT_PY                interpreter with inspect_ai   (default: the melteval venv)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SITE="${MELT_SITE:-artemis}"
MIRROR="${MELT_MIRROR:-/mnt/scratch-artemis/giuseppe/melt-eval-speechval}"
CKPT_ROOT="${MELT_FONDUE_CKPT_ROOT:-/mnt/scratch-nyx/giuseppe/fondue-keep}"
CONFIG="${MELT_FONDUE_CONFIG:-/mnt/scratch-artemis/giuseppe/melt-data/eval-runs/speechval-fondue/config.yaml}"
PACK_TIME="${MELT_TIME:-03:00:00}"
PY="${MELT_PY:-/mnt/scratch-artemis/giuseppe/venvs/melteval/bin/python}"
ROUNDS="${MELT_ROUNDS:-3}"

# Each pack has to finish inside gpu-short's 4 h. Split by cost, not by name:
# ASR ~1 h, FLEURS ST ~2 h, the remaining ST ~2 h on a6000.
PHASES=(
  "librispeech-validation-clean librispeech-validation-other voxpopuli-dev"
  "fleurs24-st-xen-dev fleurs24-st-enx-dev"
  "covost2-dev covost2-enx-dev europarl-st-dev"
)

usage() {
  sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 2
}

STEP=""
DO_SCORE=1
DRY_RUN=0
CONFIG_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --no-score) DO_SCORE=0 ;;
    --config-only) CONFIG_ONLY=1 ;;
    --time) shift; PACK_TIME="${1:?--time needs a value}" ;;
    -h|--help) usage ;;
    -*) echo "unknown option: $1" >&2; usage ;;
    *) [[ -n "$STEP" ]] && { echo "one checkpoint per run, got '$STEP' and '$1'" >&2; exit 2; }
       STEP="$1" ;;
  esac
  shift
done

[[ -n "$STEP" ]] || { echo "usage: $(basename "$0") <checkpoint-step> [options]" >&2; exit 2; }
[[ "$STEP" =~ ^[0-9]+$ ]] || { echo "the checkpoint step must be a number, got '$STEP'" >&2; exit 2; }

CKPT="${CKPT_ROOT}/checkpoint-${STEP}"
MODEL="fondue-step-${STEP}"
[[ -d "$CKPT" ]] || { echo "no such checkpoint: $CKPT" >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "no such campaign config: $CONFIG" >&2; exit 1; }

# Add the model entry and set the per-pack wall time. The entry is cloned from
# an existing one, so the edit keeps whatever style the file has: the campaign
# snapshot written by yaml.safe_dump puts sequence items at column 0 with
# dumper-named anchors, while the hand-written configs in this repo indent them
# and share a hand-named one. Both must stay valid YAML.
edit_config() {
  "$(command -v python3)" - "$CONFIG" "$MODEL" "$CKPT" "$PACK_TIME" "$1" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
model, ckpt, pack_time, dry = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] == "1"
lines = path.read_text().splitlines()

try:
    models_at = lines.index("models:")
except ValueError:
    sys.exit(f"{path}: no top-level 'models:' key")
# The block ends at the next top-level key. A sequence item can sit at column 0
# too (yaml.safe_dump writes it there), so only a key -- not a dash -- ends it.
end = next(
    (i for i in range(models_at + 1, len(lines)) if re.match(r"^[^\s-]", lines[i])),
    len(lines),
)
block = lines[models_at + 1 : end]
starts = [i for i, line in enumerate(block) if re.match(r"^\s*- name:", line)]
if not starts:
    sys.exit(f"{path}: no model entries to clone")

bounds = starts + [len(block)]
entries = [block[starts[n] : bounds[n + 1]] for n in range(len(starts))]
while entries and entries[-1] and not entries[-1][-1].strip():
    entries[-1].pop()
had_blank = any(not line.strip() for line in block)
prefix = block[: starts[0]]

# Clone the last entry whose aliases are all defined in this file, so a broken
# entry (a dangling *alias) never becomes the template; the clone then never
# defines an anchor of its own.
defined = set(re.findall(r"&(\w+)", "\n".join(lines)))
usable = [
    entry
    for entry in entries
    if all(name in defined for name in re.findall(r"\*(\w+)", "\n".join(entry)))
]
if not usable:
    sys.exit(f"{path}: no model entry with resolvable aliases to clone")
template = usable[-1]
entry = list(template)
entry[0] = re.sub(r"name:\s*\S+", f"name: {model}", entry[0])
for i, line in enumerate(entry):
    if re.match(r"^\s*model:", line):
        entry[i] = f"{line[: len(line) - len(line.lstrip())]}model: melt/{ckpt}"
        break
entry = [re.sub(r"(args:\s*)&\S+", r"\1", line) for line in entry]

already = any(e[0].split("name:", 1)[1].strip() == model for e in entries)
kept = [e for e in entries if e[0].split("name:", 1)[1].strip() != model]
kept.append(entry)
body = []
for n, item in enumerate(kept):
    if n and had_blank:
        body.append("")
    body.extend(item)
new_block = prefix + body + ([""] if had_blank else [])
lines = lines[: models_at + 1] + new_block + lines[end:]

pack_at = next((i for i, line in enumerate(lines) if line.startswith("pack:")), None)
if pack_at is not None:
    for i in range(pack_at + 1, len(lines)):
        if not lines[i].strip():
            continue
        if not lines[i].startswith((" ", "\t")):
            break
        if lines[i].lstrip().startswith("time:"):
            lines[i] = f'  time: "{pack_time}"'
            break

if dry:
    print("model entry " + ("already present, would be rewritten:" if already else "would be added:"))
    print("\n".join(entry))
    print(f'pack time would be set to "{pack_time}"')
    sys.exit(0)

path.write_text("\n".join(lines) + "\n")
print(f"{'rewritten' if already else 'added'} {model} in {path}")
PY
}

edit_config "$DRY_RUN"

header() {
  echo "checkpoint: $CKPT"
  echo "config:     $CONFIG"
}

if [[ "$DRY_RUN" == 1 ]]; then
  header
  for i in "${!PHASES[@]}"; do
    echo "pack $((i + 1)) (<= ${PACK_TIME}): ${PHASES[$i]}"
  done
  if [[ "$DO_SCORE" == 1 ]]; then
    echo "then: $PY scripts/evaluate.py score $CONFIG --models $MODEL --site $SITE --wait"
    echo "      (MetricX on the ST logs; ASR declares no post scorer)"
  else
    echo "then: stop (scoring disabled)"
  fi
  echo "up to $ROUNDS round(s); each ends with scripts/evaluate.py status"
  exit 0
fi

if [[ "$CONFIG_ONLY" == 1 ]]; then
  header
  exit 0
fi

LOCAL=0
if command -v sbatch >/dev/null 2>&1; then
  LOCAL=1
fi

# Drive scripts/evaluate.py, here or on the host that has the scheduler.
launch() {
  if [[ "$LOCAL" == 1 ]]; then
    ( cd "$REPO" && MELT_TIME="$PACK_TIME" "$PY" scripts/evaluate.py "$@" )
  else
    local args
    printf -v args '%q ' "$@"
    ssh "$SITE" "cd $(printf '%q' "$MIRROR") && MELT_TIME=$(printf '%q' "$PACK_TIME") $(printf '%q' "$PY") scripts/evaluate.py $args"
  fi
}

# Slices still without a successful log, from the launcher's own status view.
needs_work() {
  local out
  out="$(launch status "$CONFIG" --models "$MODEL" 2>&1)" || return 1
  grep -qE '^(missing|failed)[[:space:]]' <<<"$out"
}

header
for round in $(seq 1 "$ROUNDS"); do
  echo
  echo "== round ${round}/${ROUNDS}"
  for phase in "${PHASES[@]}"; do
    echo "-- pack: ${phase}"
    # shellcheck disable=SC2086  # the phase is a list of benchmark names
    launch run "$CONFIG" --models "$MODEL" --benchmarks $phase --site "$SITE" --wait
  done
  if ! needs_work; then
    echo "every slice has a successful log"
    break
  fi
  echo "some slices are still missing; going round again"
done

if [[ "$DO_SCORE" == 1 ]]; then
  echo
  echo "== scoring ST with MetricX"
  launch score "$CONFIG" --models "$MODEL" --site "$SITE" --wait
fi

echo
echo "== final status"
launch status "$CONFIG" --models "$MODEL"
