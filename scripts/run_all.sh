#!/bin/bash
# Run the whole evaluation matrix end to end, unattended:
#
#   prefetch -> generate -> retry what failed -> score free-text slices -> report
#
# Each GPU step is submitted to SLURM by scripts/run_matrix.py; this script only
# waits for the jobs it submitted and moves on, so it runs on the login node for
# hours. Run it inside tmux:
#
#   tmux new -s eval
#   MODELS="qwen2-audio-7b-instruct qwen3-omni-30b-a3b-instruct smurf" scripts/run_all.sh bocconi
#   (Ctrl-b d to detach; tmux attach -t eval to come back)
#
# Arguments / environment:
#   $1              site (infra/sites/<site>.sh; its defaults -- HF_HOME and the
#                   venv paths -- are loaded, so nothing needs exporting first)
#   $2              matrix config (default configs/matrix/baselines-v1.yaml)
#   MODELS          space-separated model names to run (default: every model in the matrix)
#   BENCHMARKS      space-separated benchmark names (default: all)
#   REPORT_OUT      where results.xlsx goes (default ~/scratch/report-baselines)
#   SKIP_PREFETCH=1 skip the download step (already done)
#   SKIP_SCORE=1    skip grading audio_chat (e.g. no .venv-mcif yet); those cells stay "pending"
#   POLL_SECONDS    how often to check the queue (default 120)
#
# Safe to re-run at any point: every step skips what is already done.
set -euo pipefail

SITE="${1:?usage: [MODELS=...] $0 <site> [matrix.yaml]}"
MATRIX="${2:-configs/matrix/baselines-v1.yaml}"
REPORT_OUT="${REPORT_OUT:-$HOME/scratch/report-baselines}"
POLL_SECONDS="${POLL_SECONDS:-120}"

[[ -f melteval/tasks.py ]] || { echo "ERROR: run this from the melt-eval repo root." >&2; exit 2; }
SITE_FILE="infra/sites/${SITE}.sh"
[[ -f "$SITE_FILE" ]] || { echo "ERROR: unknown site '${SITE}' (no ${SITE_FILE})." >&2; exit 2; }
# The site file's defaults (HF_HOME, MELT_VENV_PATH, MCIF_VENV_PATH, ...) are
# what the matrix's ${VAR} paths resolve against, so load them here too: the
# submit scripts source it for each job, but run_matrix.py itself does not.
# Anything already exported in this shell still wins (every default is ${VAR:-...}).
# shellcheck source=/dev/null
source "$SITE_FILE"
[[ -n "${HF_HOME:-}" ]] || { echo "ERROR: HF_HOME is not set (downloads would fill \$HOME)." >&2; exit 2; }
echo "site ${SITE}: HF_HOME=${HF_HOME}"
echo "  venvs: default=${VENV_PATH:-?} melt=${MELT_VENV_PATH:-unset} mcif=${MCIF_VENV_PATH:-unset}"
python -c "import melteval" 2>/dev/null || { echo "ERROR: activate the venv first (source .venv/bin/activate)." >&2; exit 2; }

FILTER=()
[[ -n "${MODELS:-}" ]] && FILTER+=(--models ${MODELS})
[[ -n "${BENCHMARKS:-}" ]] && FILTER+=(--benchmarks ${BENCHMARKS})

step() { printf '\n==== %s  [%s]\n' "$*" "$(date '+%F %T')"; }

run_matrix() { python scripts/run_matrix.py "$MATRIX" --site "$SITE" "$@"; }

# Run a submitting command, echo its output, and collect the SLURM job ids it printed.
SUBMITTED=()
submit() {
    # Shown live (the runner may wait for room in the queue for hours) and kept
    # to pick the job ids out of.
    local out log
    log="$(mktemp)"
    if ! "$@" 2>&1 | tee "$log"; then
        echo "ERROR: submission failed" >&2
        rm -f "$log"
        exit 1
    fi
    out="$(cat "$log")"; rm -f "$log"
    mapfile -t SUBMITTED < <(grep -oP 'Submitted batch job \K[0-9]+' <<<"$out" || true)
}

# Block until every job in SUBMITTED has left the queue.
wait_for_jobs() {
    [[ ${#SUBMITTED[@]} -eq 0 ]] && { echo "(nothing submitted, nothing to wait for)"; return; }
    local ids; ids="$(IFS=,; echo "${SUBMITTED[*]}")"
    echo "waiting for ${#SUBMITTED[@]} job(s): ${ids}"
    while true; do
        local left; left="$(squeue -h -j "$ids" 2>/dev/null | wc -l)"
        [[ "$left" -eq 0 ]] && break
        echo "  $(date '+%T')  ${left} job(s) still queued/running"
        sleep "$POLL_SECONDS"
    done
    local failed=()
    for id in "${SUBMITTED[@]}"; do
        for f in logs/*."${id}".out; do
            [[ -f "$f" ]] && grep -q Traceback "$f" && failed+=("$f")
        done
    done
    if [[ ${#failed[@]} -gt 0 ]]; then
        echo "  ${#failed[@]} job(s) ended with a traceback:"
        printf '    %s\n' "${failed[@]}"
    fi
}

if [[ "${SKIP_PREFETCH:-0}" != 1 ]]; then
    step "1/5 prefetch datasets into ${HF_HOME}"
    run_matrix prefetch "${FILTER[@]}"
fi

step "2/5 generate"
submit run_matrix "${FILTER[@]}"
wait_for_jobs

# Anything killed at the wall clock or failed transiently has no successful log,
# so a second pass resubmits exactly that. A deterministic failure fails again
# and is reported by the report's warnings; it is not retried a third time.
step "3/5 retry what did not finish"
submit run_matrix "${FILTER[@]}"
wait_for_jobs

if [[ "${SKIP_SCORE:-0}" != 1 ]]; then
    step "4/5 score free-text slices (audio_chat)"
    submit run_matrix score "${FILTER[@]}"   # subcommand first: --models would swallow it
    wait_for_jobs
fi

step "5/5 report"
LOG_ROOT="$(run_matrix log-root)"
python projects/baselines/report.py --log-root "$LOG_ROOT" --out "$REPORT_OUT" \
    || echo "WARNING: the report could not be built (see above)."

step "done"
echo "Excel: ${REPORT_OUT}/results.xlsx"
echo "Still missing (resubmit with the same command once fixed):"
run_matrix --dry-run "${FILTER[@]}" | grep -E "^plan|^skip" || echo "  nothing"
