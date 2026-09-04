#!/usr/bin/env bash
#
# Run the whole SMURF cross-check end to end (docs/replication_notes.md,
# section "Cross-check: melteval SMURF inference == fbk_speechllm.inference"):
#
#   2. melteval freeze  +  scripts/hf_to_lhotse.py   (CPU, always local)
#   3. python -m fbk_speechllm.inference             (GPU)
#   4. inspect eval  melteval/tasks.py@asr           (GPU)
#   5. scripts/compare_crosscheck.py                 (CPU, always local)
#        + scripts/score_asr.py: WER/CER of *each* side against the frozen
#          set's gold transcripts (informational; not part of the verdict).
#
# Two GPU modes:
#   local   (default)   -- run steps 3 and 4 here, in the current venv.
#   SLURM   (SITE set)   -- submit steps 3 and 4 with `sbatch --wait` on that
#                           site. Use this when the checkpoint does not fit on
#                           one local GPU: an 80 GB h100/h200 loads it whole,
#                           with no sharding on either side (the cleanest 1:1).
#
# Run from the repo root. Steps whose output already exists are skipped unless
# FORCE=1.
#
# Usage:
#   scripts/run_crosscheck.sh <checkpoint>
#   SITE=artemis MELT_PARTITION=h100 VENV_SMURF=/scratch/…/venvs/smurf-eval/bin/activate \
#     scripts/run_crosscheck.sh <checkpoint>
#
# Knobs (env):
#   CONFIG               melteval freeze config            [configs/librispeech-hf-smoke.yaml]
#   SMURF_GEN_DIR        dir with the upstream gen config  [../speechllm/working_config/generate]
#   SMURF_GEN_NAME       --config-name for that config     [asr_inference]
#   INSTRUCTION          prompt text (must match upstream) ["Transcribe this English audio: "]
#   MAX_TOKENS           generation cap, both sides        [128]
#   SCORE_NORMALIZER     normalizer for the step-5 accuracy table (lower |
#                        basic | english | none; basic/english need `melt`)  [lower]
#   MELTEVAL_DEVICE_MAP  "" (default) loads on one GPU; "auto" shards across all  [""]
#   WORK                output root                        [runs]
#   FORCE=1             redo steps whose output exists
#   FROM_STEP=N         start at step N (2..5), reuse earlier outputs
#   TO_STEP=N           stop after step N (e.g. TO_STEP=2 = CPU prep only)
#   SKIP_UPSTREAM=1     skip steps 3 and 5 (no fbk_speechllm available)
#
# SLURM mode only:
#   SITE               infra/sites/<SITE>.sh to submit against (enables SLURM mode)
#   VENV_SMURF         activate script of the SMURF venv on that site (required)
#   MELT_PARTITION     partition for both jobs                          [h100]
#   MELT_QOS / MELT_TIME  passed through to the site's SBATCH_ARGS

set -euo pipefail

# ---------------------------------------------------------------- colours ----
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
    B=$'\e[1m'; DIM=$'\e[2m'; R=$'\e[31m'; G=$'\e[32m'; Y=$'\e[33m'; C=$'\e[36m'; X=$'\e[0m'
else
    B=""; DIM=""; R=""; G=""; Y=""; C=""; X=""
fi
CC_COLOR=$([[ -n "$B" ]] && echo always || echo never)   # match the compare script to us
section() { printf '\n%s\n%s\n\n' "${B}${C}══ $* ══${X}" "${DIM}$(date '+%H:%M:%S')${X}"; }
info()    { printf '%s\n' "${C}·${X} $*"; }
warn()    { printf '%s\n' "${Y}!${X} $*"; }
ok()      { printf '%s\n' "${G}✓${X} $*"; }
die()     { printf '%s\n' "${R}✗ $*${X}" >&2; exit 1; }
run()     { printf '%s\n' "${DIM}\$ $*${X}"; "$@"; }

# ------------------------------------------------------------------ config ----
CONFIG="${CONFIG:-configs/librispeech-hf-smoke.yaml}"
MODEL_CKPT="${1:-${MODEL_CKPT:-}}"
SMURF_GEN_DIR="${SMURF_GEN_DIR:-$(pwd)/../speechllm/working_config/generate}"
SMURF_GEN_NAME="${SMURF_GEN_NAME:-asr_inference}"
INSTRUCTION="${INSTRUCTION:-Transcribe this English audio: }"
MAX_TOKENS="${MAX_TOKENS:-128}"
SCORE_NORMALIZER="${SCORE_NORMALIZER:-lower}"
WORK="${WORK:-runs}"
FROM_STEP="${FROM_STEP:-2}"
TO_STEP="${TO_STEP:-5}"
SITE="${SITE:-}"
# empty (default) = load the melteval checkpoint on one GPU; "auto" = shard it
# across every visible GPU (only if it does not fit on one).
MELTEVAL_DEVICE_MAP="${MELTEVAL_DEVICE_MAP-}"

FROZEN_DIR="${WORK}/librispeech-hf-smoke"
CUTS_DIR="${WORK}/crosscheck-smurf-cuts"
RUN_DIR="${WORK}/crosscheck-smurf"
LOG_DIR="${RUN_DIR}/eval-logs"
UPSTREAM_OUT="${RUN_DIR}/upstream.jsonl"

export TMPDIR="${TMPDIR:-$HOME/.cache/triton-eval}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.cache/triton-eval}"
mkdir -p "$TMPDIR" "$RUN_DIR" logs

# --------------------------------------------------------------- preflight ----
section "preflight"
[[ -f melteval/tasks.py ]] || die "run from the repo root (melteval/tasks.py not found)"
[[ -n "$MODEL_CKPT" ]]     || die "no checkpoint: pass it as arg 1 or set MODEL_CKPT="
[[ -e "$MODEL_CKPT" ]]     || die "checkpoint not found: $MODEL_CKPT"
[[ -f "$CONFIG" ]]         || die "config not found: $CONFIG"
MODEL_ABS="$(cd "$(dirname "$MODEL_CKPT")" && pwd)/$(basename "$MODEL_CKPT")"

WITH_UPSTREAM=1
[[ -n "${SKIP_UPSTREAM:-}" ]] && { WITH_UPSTREAM=0; warn "SKIP_UPSTREAM set -- steps 3 and 5 skipped"; }

if [[ -n "$SITE" ]]; then
    MODE="SLURM (${SITE})"
    SITE_FILE="infra/sites/${SITE}.sh"
    [[ -f "$SITE_FILE" ]] || die "unknown site '${SITE}' (expected ${SITE_FILE})"
    [[ -n "${VENV_SMURF:-}" ]] || die "SLURM mode needs VENV_SMURF=/path/to/smurf-eval/bin/activate"
    command -v sbatch >/dev/null || die "sbatch not on PATH but SITE is set"
    # The site file is sourced later (just before step 3), not here: it exports
    # HF_HUB_OFFLINE=1, which would break the local, network-touching step 2 on
    # a first run.
else
    MODE="local"
    if (( WITH_UPSTREAM )) && ! python -c "import fbk_speechllm" 2>/dev/null; then
        die "fbk_speechllm not importable here. Use SITE=<site> to submit via SLURM, or SKIP_UPSTREAM=1."
    fi
fi
if (( WITH_UPSTREAM )) && [[ ! -f "${SMURF_GEN_DIR}/${SMURF_GEN_NAME}.yaml" ]]; then
    die "upstream gen config not found: ${SMURF_GEN_DIR}/${SMURF_GEN_NAME}.yaml"
fi

info "mode:         ${B}${MODE}${X}"
info "checkpoint:   ${MODEL_ABS}"
info "config:       ${CONFIG}"
info "instruction:  '${INSTRUCTION}'"
info "max tokens:   ${MAX_TOKENS}"
info "device_map:   ${MELTEVAL_DEVICE_MAP:-<one GPU>}"
info "outputs:      ${RUN_DIR}/"
(( WITH_UPSTREAM )) && info "upstream cfg: ${SMURF_GEN_DIR}/${SMURF_GEN_NAME}.yaml"

skip_done() {  # $1 = path that marks the step done, $2 = human name
    [[ -z "${FORCE:-}" && -e "$1" ]] && { ok "$2 already present ($1) -- skipping (FORCE=1 to redo)"; return 0; }
    return 1
}

# ------------------------------------------------------ 2. build the sets ----
if (( FROM_STEP <= 2 && TO_STEP >= 2 )); then
    section "2 · build the frozen set and the matching lhotse cuts"
    if ! skip_done "${FROZEN_DIR}/manifest.jsonl" "frozen set"; then
        run melteval freeze "$CONFIG" -o "$FROZEN_DIR"
    fi
    run melteval show "$FROZEN_DIR"
    if ! skip_done "${CUTS_DIR}/cuts.jsonl.gz" "lhotse cuts"; then
        run python scripts/hf_to_lhotse.py --config "$CONFIG" -o "$CUTS_DIR"
    fi
    ok "step 2 done"
fi

# --- SLURM: bring in the site's SBATCH_ARGS and offline/cache exports, now
# that the network-touching step 2 is done ---------------------------------
if [[ -n "$SITE" ]] && (( TO_STEP >= 3 )); then
    export MELT_PARTITION="${MELT_PARTITION:-h100}"
    # shellcheck disable=SC1090
    source "$SITE_FILE"
    info "sbatch:       ${SBATCH_ARGS[*]}"
fi

# ------------------------------------------- 3. upstream: fbk_speechllm ------
if (( WITH_UPSTREAM )) && (( FROM_STEP <= 3 && TO_STEP >= 3 )); then
    section "3 · fbk_speechllm.inference  (GPU)"
    if ! skip_done "$UPSTREAM_OUT" "upstream hypotheses"; then
        mkdir -p "$(dirname "$UPSTREAM_OUT")"
        if [[ -n "$SITE" ]]; then
            run env \
                VENV_PATH="$VENV_SMURF" \
                MODEL_PATH="$MODEL_ABS" \
                DATA_PATH="$(pwd)/${CUTS_DIR}/cuts.jsonl.gz" \
                OUTPUT_PATH="$(pwd)/${UPSTREAM_OUT}" \
                sbatch --wait "${SBATCH_ARGS[@]}" infra/crosscheck_upstream.sbatch \
                    "$SMURF_GEN_DIR" "$SMURF_GEN_NAME" \
                    "generation.max_new_tokens=${MAX_TOKENS}"
        else
            run env \
                MODEL_PATH="$MODEL_ABS" \
                DATA_PATH="$(pwd)/${CUTS_DIR}/cuts.jsonl.gz" \
                OUTPUT_PATH="$(pwd)/${UPSTREAM_OUT}" \
                python -m fbk_speechllm.inference \
                    --config-path "$SMURF_GEN_DIR" \
                    --config-name "$SMURF_GEN_NAME" \
                    data.test_ds.batch_size=1 \
                    "generation.max_new_tokens=${MAX_TOKENS}"
        fi
    fi
    lines=$(wc -l < "$UPSTREAM_OUT" 2>/dev/null || echo 0)
    ok "step 3 done -- ${lines} hypotheses in ${UPSTREAM_OUT}"
fi

# --------------------------------------------- 4. melteval: inspect eval ----
if (( FROM_STEP <= 4 && TO_STEP >= 4 )); then
    section "4 · inspect eval  (GPU)"
    mkdir -p "$LOG_DIR"
    common_targs=(
        -T "frozen_set=$(pwd)/${FROZEN_DIR}"
        -T prompt_style=smurf
        # -T is parsed as YAML; INSTRUCTION contains ": " (colon-space), which
        # YAML would otherwise read as a mapping, not a string (see
        # docs/replication_notes.md) -- the embedded double quotes force it to
        # parse as a plain string.
        -T "instruction=\"${INSTRUCTION}\""
        -T normalizer=none
        --max-tokens "$MAX_TOKENS"
        -M batch_size=1
    )
    [[ -n "$MELTEVAL_DEVICE_MAP" ]] && common_targs+=(-M "device_map=${MELTEVAL_DEVICE_MAP}")

    if [[ -n "$SITE" ]]; then
        run env \
            MELTEVAL_PROVIDER=smurf \
            VENV_PATH="$VENV_SMURF" \
            OUTPUT_DIR="$(pwd)/${LOG_DIR}" \
            sbatch --wait "${SBATCH_ARGS[@]}" infra/run_eval.sbatch \
                "$MODEL_ABS" "$(pwd)/${FROZEN_DIR}" \
                -T task_filter=asr "${common_targs[@]}"
    else
        run inspect eval melteval/tasks.py@asr \
            --model "smurf/${MODEL_ABS}" \
            --log-dir "$LOG_DIR" \
            "${common_targs[@]}"
    fi
    ok "step 4 done"
fi

EVAL_LOG="$(ls -t "${LOG_DIR}"/*.eval 2>/dev/null | head -n1 || true)"

# ------------------------------------------------------- 5. compare ---------
if (( WITH_UPSTREAM )) && (( FROM_STEP <= 5 && TO_STEP >= 5 )); then
    section "5 · compare hypotheses"
    [[ -n "$EVAL_LOG" ]]      || die "no .eval log in ${LOG_DIR}"
    [[ -f "$UPSTREAM_OUT" ]]  || die "no upstream output at ${UPSTREAM_OUT}"
    info "eval log: ${EVAL_LOG}"
    info "upstream: ${UPSTREAM_OUT}"

    set +e
    python scripts/compare_crosscheck.py \
        --eval-log "$EVAL_LOG" \
        --upstream "$UPSTREAM_OUT" \
        --instruction "$INSTRUCTION" \
        --color "$CC_COLOR"
    rc=$?
    set -e

    # Accuracy of each side against the frozen set's gold transcripts. Purely
    # informational -- the cross-check verdict is compare_crosscheck.py's alone
    # (hyp-vs-hyp), so a failure here only drops the table, it does not flip rc.
    section "5b · accuracy vs. gold  (same normalizer both sides)"
    set +e
    python scripts/score_asr.py \
        --frozen-set "$FROZEN_DIR" \
        --normalizer "$SCORE_NORMALIZER" \
        "melteval=${EVAL_LOG}" \
        "upstream=${UPSTREAM_OUT}"
    score_rc=$?
    set -e
    (( score_rc == 0 )) || warn "score_asr.py exited ${score_rc}; accuracy table skipped (verdict unaffected)"

    echo
    if (( rc == 0 )); then
        printf '%s\n' "${B}${G}╔══════════════════════════════════════════════╗${X}"
        printf '%s\n' "${B}${G}║  CROSS-CHECK PASSED                           ║${X}"
        printf '%s\n' "${B}${G}╚══════════════════════════════════════════════╝${X}"
    else
        printf '%s\n' "${B}${R}╔══════════════════════════════════════════════╗${X}"
        printf '%s\n' "${B}${R}║  CROSS-CHECK FAILED -- see the diffs above    ║${X}"
        printf '%s\n' "${B}${R}╚══════════════════════════════════════════════╝${X}"
    fi
    exit $rc
fi

section "done"
ok "frozen set:  ${FROZEN_DIR}"
ok "lhotse cuts: ${CUTS_DIR}/cuts.jsonl.gz"
(( WITH_UPSTREAM )) && [[ -f "$UPSTREAM_OUT" ]] && ok "upstream:     ${UPSTREAM_OUT}"
[[ -n "$EVAL_LOG" ]] && ok "eval log:     ${EVAL_LOG}"
if (( ! WITH_UPSTREAM )); then
    warn "upstream skipped: run compare later with scripts/compare_crosscheck.py"
    [[ -n "$EVAL_LOG" ]] && info "melteval WER/CER vs gold:  python scripts/score_asr.py --frozen-set ${FROZEN_DIR} melteval=${EVAL_LOG}"
fi
