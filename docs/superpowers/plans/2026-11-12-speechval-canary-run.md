# Canary-1B-v2 on speechval-dev — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score NVIDIA Canary-1B-v2 on every direction of `speechval-dev` it supports, on artemis, with a campaign folder the worklog can read.

**Architecture:** Refreeze the suite's dev sets on artemis (the existing CoVoST2 set is the old 5-direction one), add a campaign config with explicit skips for languages Canary does not cover, and run two waves through `scripts/evaluate.py` with a single-GPU pack so one SLURM job reuses one model load. WER/BLEU/chrF are inline; MetricX is run post-hoc by `score`.

**Tech Stack:** `scripts/evaluate.py`, `melteval`, SLURM (`a6000` / `gpu-short`), NeMo via `melt/hf/canary`, MetricX-24 through `$METRICS_VENV_PATH`.

**Spec:** `docs/superpowers/specs/2026-11-12-speechval-canary-and-worklog-design.md` (Part A)

## Global Constraints

- Never run anything GPU-bound outside SLURM on artemis; freezing, prefetch, `plan` and `status` are CPU-only and may run directly.
- Use the nyx Shar tree: `/mnt/scratch-nyx/giuseppe/melt/melt-data/shar` (`LOCAL_DATASETS_DIR` and `AZEROS_INDEXED_DIR`).
- Frozen sets live under `/mnt/scratch-artemis/giuseppe/frozen/`; never write to `$HOME`.
- Canary covers `bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk` and translates only with English on one side.
- Expected frozen counts: VoxPopuli 13 034, FLEURS X→en 8 261, FLEURS en→X 8 819, CoVoST2 61 452, EuroparlST 36 000.
- Pack wall clock 4 h (`gpu-short`); a slice without a successful log is retried by the next `run`.

## Review Focus

- A slice whose language Canary does not cover reaches generation → the provider raises mid-run and the slice errors. `plan` must report it skipped instead, for every one of the 76 excluded slices.
- A missing frozen set or unset env var → `plan` says "environment variable is not set" rather than submitting jobs against nothing.
- Compute nodes are offline → LibriSpeech (`spec:`-read) must be prefetched, or two benchmarks and the smoke run fail in a way that looks like a provider bug.
- A pack that runs out of its 4 h → slices are left missing; `status` must show them and the next `run` must pick up only those.
- Two generations of a frozen set → old campaign logs must not silently start resolving to a different set.

---

### Task 1: Freeze the four dev sets and verify counts

**Files:**
- Modify: artemis filesystem only
- Verify: `frozen_set.json` under each new directory

**Interfaces:**
- Produces: `.../frozen/speechval-dev/{voxpopuli-dev,fleurs24-st-xen-dev-full,fleurs24-st-enx-dev,st-eval-campaign-v1-dev}`, used by the site file in Task 3 and the config in Task 2.

- [ ] **Step 1: Preserve the old CoVoST2 set**

```bash
F=/mnt/scratch-artemis/giuseppe/frozen/speechval-dev
[[ -d $F/st-eval-campaign-v1-dev && ! -d $F/st-eval-campaign-v1-dev.five-dir ]] && \
  mv $F/st-eval-campaign-v1-dev $F/st-eval-campaign-v1-dev.five-dir
```

Expected: the old 10 000-sample set is at `st-eval-campaign-v1-dev.five-dir`;
the canonical name is free. (No artemis campaign referenced it — checked across
`eval-runs/*/logs` — so no log stops resolving.)

- [ ] **Step 2: Freeze all four sets**

```bash
source /mnt/scratch-artemis/giuseppe/venvs/melteval/bin/activate
export LOCAL_DATASETS_DIR=/mnt/scratch-nyx/giuseppe/melt/melt-data/shar
export AZEROS_INDEXED_DIR=$LOCAL_DATASETS_DIR
F=/mnt/scratch-artemis/giuseppe/frozen/speechval-dev
for s in fleurs24-st-xen-dev-full fleurs24-st-enx-dev voxpopuli-dev st-eval-campaign-v1-dev; do
  melteval freeze configs/shar/$s.yaml -o $F/$s
done
```

Expected: four directories, each with `frozen_set.json` and `manifest.jsonl`.

- [ ] **Step 3: Verify the counts**

```bash
/usr/bin/python3 - <<'PY'
import json
F = "/mnt/scratch-artemis/giuseppe/frozen/speechval-dev"
want = {"voxpopuli-dev": 13034, "fleurs24-st-xen-dev-full": 8261,
        "fleurs24-st-enx-dev": 8819, "st-eval-campaign-v1-dev": 61452}
for name, n in want.items():
    j = json.load(open(f"{F}/{name}/frozen_set.json"))
    print(name, j["total_samples"], f'{j["total_hours"]:.1f} h',
          "OK" if j["total_samples"] == n else f"EXPECTED {n}")
PY
```

Expected: four lines ending in `OK`. Record the hours; they inform the wave split.

---

### Task 2: Write the Canary campaign config

**Files:**
- Create: `configs/eval/speechval-canary.yaml`

**Interfaces:**
- Consumes: the suite `configs/eval/suites/speechval-dev.yaml`
- Produces: a config `plan` can enumerate; `output_dir` `${EVAL_ROOT}/speechval-canary`

- [ ] **Step 1: Generate the Europarl skip lines**

```bash
/usr/bin/python3 - <<'PY'
langs = "fr de es it pt pl ro nl".split()
print("\n".join(f"  - europarl-st-{a}_{b}-dev" for a in langs for b in langs if a != b))
PY
```

Expected: 56 lines. Paste them under the Europarl comment in the next step.

- [ ] **Step 2: Write the config**

```yaml
# Canary-1B-v2 on speechval-dev (dev). Canary is NeMo, not a chat model: its
# "instruction" is <task>|<source>|<target>, turned into transcribe() arguments.
# It covers 25 European languages and translates only with English on one side,
# so every unsupported direction is skipped explicitly (see docs/speechval.md).
#
#   python scripts/evaluate.py plan configs/eval/speechval-canary.yaml --site artemis
#   python scripts/evaluate.py run  configs/eval/speechval-canary.yaml --site artemis --wait

name: speechval-canary
output_dir: ${EVAL_ROOT}/speechval-canary

models:
  - name: canary-1b-v2
    model: melt/hf/canary
    venv: /mnt/scratch-artemis/giuseppe/melt-data/pyextras/canary-shim/activate
    args: [-M, batch_size=16]
    instruction:
      asr: "asr|{lang_code}|{lang_code}"
      st: "st|{src_lang_code}|{tgt_lang_code}"

benchmarks: configs/eval/suites/speechval-dev.yaml

skip:
  # FLEURS ga: Irish is not one of the 25 languages canary covers.
  - fleurs-ga_en-dev
  - fleurs-en_ga-dev
  # CoVoST2 directions with a language canary does not cover (ar ca cy fa id ja
  # mn ta tr zh), both ways.
  - covost2-ar_en-dev
  - covost2-ca_en-dev
  - covost2-cy_en-dev
  - covost2-fa_en-dev
  - covost2-id_en-dev
  - covost2-ja_en-dev
  - covost2-mn_en-dev
  - covost2-ta_en-dev
  - covost2-tr_en-dev
  - covost2-zh_en-dev
  - covost2-en_ar-dev
  - covost2-en_ca-dev
  - covost2-en_cy-dev
  - covost2-en_fa-dev
  - covost2-en_id-dev
  - covost2-en_ja-dev
  - covost2-en_mn-dev
  - covost2-en_ta-dev
  - covost2-en_tr-dev
  - covost2-en_zh-dev
  # EuroparlST pairs with no English side (56 of 72). All eight languages are
  # covered by canary, but it translates only en<->X.
  # <generated block from Step 1>

# One SLURM job per wave, one model load for all its slices (artemis has no
# MELT_PACK_GPUS by default; this is the only pack setting needed).
pack:
  time: "04:00:00"
  max_slices: 200
  gpus: 1
```

- [ ] **Step 3: Commit**

```bash
git add configs/eval/speechval-canary.yaml
git commit -m "speechval: canary-1b-v2 campaign config"
```

---

### Task 3: Point the artemis site file at the speechval sets and verify `plan`

**Files:**
- Modify: `infra/sites/artemis.sh` (after the `LOCAL_DATASETS_DIR` block)
- Test: the `plan` output

**Interfaces:**
- Produces: `VOXPOPULI_DEV_SET`, `FLEURS24_ST_XEN_DEV_SET`,
  `FLEURS24_ST_ENX_DEV_SET`, `COVOST2_DEV_SET`, `EUROPARL_ST_DEV_SET` exported
  when their directories exist.

- [ ] **Step 1: Add the guarded exports**

```bash
# --- speechval-dev frozen sets (docs/speechval.md); guarded so a partial tree
# fails in `plan` with "environment variable is not set", not silently.
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
if [[ -d /mnt/scratch-artemis/giuseppe/frozen/speechval/europarl-st-dev ]]; then
    export EUROPARL_ST_DEV_SET="${EUROPARL_ST_DEV_SET:-/mnt/scratch-artemis/giuseppe/frozen/speechval/europarl-st-dev}"
fi
unset _speechval_dev
```

- [ ] **Step 2: Verify the file still sources cleanly under `set -e`**

```bash
bash -c 'set -euo pipefail; source infra/sites/artemis.sh; echo "ok=$COVOST2_DEV_SET"'
```

Expected: `ok=/mnt/scratch-artemis/giuseppe/frozen/speechval-dev/st-eval-campaign-v1-dev`.

- [ ] **Step 3: Verify `plan` sees exactly the supported work**

```bash
python scripts/evaluate.py plan configs/eval/speechval-canary.yaml --site artemis
```

Expected:
- **92** jobs: 2 LibriSpeech + 14 VoxPopuli + 22 FLEURS X→en + 22 FLEURS en→X +
  11 CoVoST X→en + 5 CoVoST en→X + 8 Europarl X→en + 8 Europarl EN→X;
- the rest skipped with `excluded by the config's skip list`;
- no `environment variable is not set`, no `venv ... not found`;
- the last line names the packs a run would submit.

- [ ] **Step 4: Commit**

```bash
git add infra/sites/artemis.sh
git commit -m "artemis: export speechval-dev frozen sets"
```

---

### Task 4: LibriSpeech validation (superseded: frozen, see ledger)

> Superseded during execution: LibriSpeech is now a frozen set
> (`configs/shar/librispeech-validation.yaml`) because `datasets` prepares every
> split of the Hub repo even for `validation` (~60 GB). The steps below were not
> run; the freeze and the suite change are in the ledger.

**Files:**
- Modify: the artemis HF cache

**Interfaces:**
- Consumes: Task 2's config, Task 3's site exports
- Produces: `openslr/librispeech_asr` validation in the cache the compute nodes read

- [ ] **Step 1: Check that it is absent**

```bash
ls /mnt/scratch-artemis/giuseppe/melt-data/hf_cache/datasets | grep -i openslr || echo ABSENT
```

Expected: `ABSENT`.

- [ ] **Step 2: Prefetch from the login shell (egress required)**

```bash
cd /home/giuseppe/melt-proj/melt-eval
python scripts/evaluate.py prefetch configs/eval/speechval-canary.yaml --site artemis
```

Expected: downloads the two `validation` splits (and the Canary weights, already
cached); ends without error. Idempotent.

- [ ] **Step 3: Verify the cache**

```bash
ls /mnt/scratch-artemis/giuseppe/melt-data/hf_cache/datasets | grep -i openslr
```

Expected: an `openslr___librispeech_asr` entry (or
`models--openslr--librispeech_asr` in the hub cache — record the actual path in
Task 9's note).

---

### Task 5: Smoke run

**Files:**
- Create: `${EVAL_ROOT}/speechval-canary/smoke-limit20/` (generated)

**Interfaces:**
- Consumes: Task 2's config
- Produces: evidence that provider loading, the control-string instruction and
  the frozen sets work before spending a full wave

- [ ] **Step 1: Run 20 samples of every wave-1 benchmark**

```bash
python scripts/evaluate.py run configs/eval/speechval-canary.yaml --site artemis \
  --benchmarks librispeech-validation-clean librispeech-validation-other \
               voxpopuli-dev fleurs24-st-xen-dev fleurs24-st-enx-dev europarl-st-dev \
  --limit 20 --wait
```

Expected: exits 0; logs under `smoke-limit20/logs/`.

- [ ] **Step 2: Inspect one ASR and one ST log**

```bash
/usr/bin/python3 - <<'PY'
import glob, json
base = "/mnt/scratch-artemis/giuseppe/melt-data/eval-runs/speechval-canary/smoke-limit20/logs"
for pat in ("**/asr-*/*.json", "**/st-fleurs-de_en*/*.json"):
    p = sorted(glob.glob(f"{base}/{pat}", recursive=True))[0]
    j = json.load(open(p))
    print(p.split("/")[-2], j["status"], j["eval"]["metadata"]["melteval_slice"],
          j["eval"]["task_args"]["instruction"], len(j.get("samples", [])))
    print(" first output:", (j["samples"][0]["output"]["completion"] or "")[:80])
PY
```

Expected: `success` for both; the instruction field is the spec template; 20
samples; a fluent transcript/translation. A non-empty transcript in the wrong
language means the control spec was not applied — stop and check `instruction`
before the full run.

- [ ] **Step 3: Fix or proceed.** If a benchmark failed, read its `slurm-*.out`
  in the smoke log folder; the next `run` retries only the missing slices.

---

### Task 6: Wave 1 (everything except CoVoST2)

**Files:**
- Create: `${EVAL_ROOT}/speechval-canary/{logs,submissions.jsonl}`

**Interfaces:**
- Consumes: Task 5's smoke result
- Produces: successful logs for 6 of the 8 benchmark families

- [ ] **Step 1: Submit and wait**

```bash
python scripts/evaluate.py run configs/eval/speechval-canary.yaml --site artemis \
  --benchmarks librispeech-validation-clean librispeech-validation-other \
               voxpopuli-dev fleurs24-st-xen-dev fleurs24-st-enx-dev europarl-st-dev \
  --wait
```

Expected: one or more packs submitted; the command returns when the queue is
empty. `submissions.jsonl` records them.

- [ ] **Step 2: Check status**

```bash
python scripts/evaluate.py status configs/eval/speechval-canary.yaml --site artemis \
  --benchmarks librispeech-validation-clean librispeech-validation-other \
               voxpopuli-dev fleurs24-st-xen-dev fleurs24-st-enx-dev europarl-st-dev
```

Expected: every wave-1 slice `done`; anything `missing`/`failed` is picked up by
re-running Step 1 (a pack cut off by its 4 h wall clock is the normal cause).
Repeat until the only non-done entries are the documented skips.

- [ ] **Step 3: Count errors and note durations**

```bash
grep -c '"status": "error"' ${EVAL_ROOT}/speechval-canary/logs/*/*/*/*.json || true
```

Expected: 0, or investigate the named files. Note pack durations from
`submissions.jsonl` for Task 9.

---

### Task 7: Wave 2 (CoVoST2)

**Files:**
- Create: logs for `covost2-dev`, `covost2-enx-dev`

**Interfaces:**
- Consumes: the refrozen 36-direction set from Task 1
- Produces: 16 successful CoVoST2 slices

- [ ] **Step 1: Submit and wait**

```bash
python scripts/evaluate.py run configs/eval/speechval-canary.yaml --site artemis \
  --benchmarks covost2-dev covost2-enx-dev --wait
python scripts/evaluate.py status configs/eval/speechval-canary.yaml --site artemis \
  --benchmarks covost2-dev covost2-enx-dev
```

Expected: 16 slices `done` (11 X→en, 5 en→X); the other 20 directions reported
as skipped by the skip list.

- [ ] **Step 2: Sanity-check the direction coverage**

```bash
/usr/bin/python3 - <<'PY'
import glob, json
paths = glob.glob("/mnt/scratch-artemis/giuseppe/melt-data/eval-runs/speechval-canary/logs/canary-1b-v2/covost2-*/*/*.json")
print(len(paths), sorted({json.load(open(p))["eval"]["metadata"]["melteval_slice"] for p in paths})[:20])
PY
```

Expected: `16 [...]` and the expected slice names.

---

### Task 8: MetricX post-hoc scoring

**Files:**
- Modify: the ST logs in place (`score` writes into them)

**Interfaces:**
- Consumes: ST slices declaring `post_scorers`, `$METRICS_VENV_PATH`
- Produces: `metricx` in each ST log's metrics; `pending` disappears for them

- [ ] **Step 1: Run the scorers**

```bash
python scripts/evaluate.py score configs/eval/speechval-canary.yaml --site artemis --wait
```

Expected: SLURM job(s) loading MetricX-24 hybrid XXL in
`/mnt/scratch-artemis/giuseppe/venvs/metrics-eval`; ends 0.

- [ ] **Step 2: Verify one log**

```bash
/usr/bin/python3 - <<'PY'
import glob, json
p = sorted(glob.glob("/mnt/scratch-artemis/giuseppe/melt-data/eval-runs/speechval-canary/logs/canary-1b-v2/fleurs24-st-xen-dev/*/*.json"))[0]
j = json.load(open(p))
for s in j["results"]["scores"]:
    print(s["name"], {k: round(v, 2) for k, v in s["metrics"].items()})
PY
```

Expected: `metricx_scorer` with a `metricx` value in roughly 1–25 (lower is
better), alongside the inline `st_scorer` metrics.

- [ ] **Step 3: Check nothing is left unscored**

`score` is idempotent (it skips logs that already carry their scorers). Run this
check and re-run `score` if it prints anything:

```bash
/usr/bin/python3 - <<'PY'
import glob, json
for p in glob.glob("/mnt/scratch-artemis/giuseppe/melt-data/eval-runs/speechval-canary/logs/canary-1b-v2/*/*/*.json"):
    j = json.load(open(p))
    declared = {s.split("@")[-1] for s in j["eval"]["metadata"].get("melteval_post_scorers", "").split(",") if s}
    got = {s["name"] for s in j.get("results", {}).get("scores", [])}
    if j.get("status") == "success" and declared and not got & declared:
        print("UNSCORED", p)
PY
```

Expected: no output.

---

### Task 9: Report and record the baseline

**Files:**
- Create: `${EVAL_ROOT}/speechval-canary/results/{summary.csv,results_long.csv,results.xlsx}`
- Modify: `docs/speechval.md` (short "Canary baseline (artemis)" section)
- Modify: this plan's "Recorded results" section

**Interfaces:**
- Produces: the first row of the worklog's data

- [ ] **Step 1: Build the report**

```bash
python scripts/evaluate.py report configs/eval/speechval-canary.yaml --site artemis
```

Expected: `results.xlsx` and both CSVs; one headline number per slice.

- [ ] **Step 2: Read the headline numbers**

```bash
/usr/bin/python3 - <<'PY'
import csv
p = "/mnt/scratch-artemis/giuseppe/melt-data/eval-runs/speechval-canary/results/summary.csv"
for r in csv.DictReader(open(p)):
    if r["benchmark"].startswith(("librispeech", "voxpopuli")):
        print(r["benchmark"], r["slice"], r.get("canary-1b-v2"))
PY
```

Expected: plausible WER (LibriSpeech validation clean roughly 2–6 %, VoxPopuli
roughly 5–15 %). A number near 100 % or 0 % is a prompt/reference bug — stop
and check the log before publishing it to the worklog.

- [ ] **Step 3: Record the results**

Append a table under "Recorded results" with: benchmark family, macro WER/BLEU,
mean MetricX, number of slices, total samples, pack wall clock, and the
`spec_hash` from each `frozen_set.json`. Add the same short table to
`docs/speechval.md` with the frozen paths, counts and date.

- [ ] **Step 4: Commit**

```bash
git add configs/eval/speechval-canary.yaml infra/sites/artemis.sh docs/speechval.md \
        docs/superpowers/plans/2026-11-12-speechval-canary-run.md
git commit -m "speechval: canary-1b-v2 baseline on artemis"
```

(`results/` stays on scratch; it is the worklog's input.)

---

## Recorded results

Run of 2026-11-12 on `a6000` (poseidon) plus `h200` (hades) for the smoke and
MetricX. Campaign `${EVAL_ROOT}/speechval-canary`; 92 slices, 76 wave-1 + 16
CoVoST; 76/76 and 16/16 successful; 60 ST logs scored with MetricX (FLEURS and
Europarl; CoVoST declares no post-scorer).

| family | slices | metric | value |
|---|---|---|---|
| LibriSpeech validation-clean | 1 | WER | 2.3 % |
| LibriSpeech validation-other | 1 | WER | 4.3 % |
| VoxPopuli dev | 14 | WER (macro) | 9.5 % |
| FLEURS X->en | 22 | BLEU / MetricX | 28.0 / 5.14 |
| FLEURS en->X | 22 | BLEU / MetricX | 28.9 / 4.58 |
| CoVoST2 X->en | 11 | BLEU | 38.0 |
| CoVoST2 en->X | 5 | BLEU | 30.1 |
| EuroparlST en<->X | 16 | BLEU / MetricX | 32.2 / 4.12 |

Frozen set hashes: librispeech-validation `6cfe26369984` (5567), voxpopuli-dev
`6c8ac5cc07c7` (13034), fleurs24-st-xen-dev-full `968f8151b34e` (8261),
fleurs24-st-enx-dev `e77d0be19482` (8819), st-eval-campaign-v1-dev
`cb0993207c8e` (61452). Full table in `docs/speechval.md`.
