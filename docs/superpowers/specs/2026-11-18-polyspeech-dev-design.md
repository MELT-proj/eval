# PolySpeech in the dev set: definition, frozen subset, and runs

**Status:** design, awaiting approval
**Date:** 2026-11-18
**Author:** Giuseppe (with the agent)

## Intent

The dev/test partition assigns `SQA/SLU` one dev benchmark: PolySpeech-100.
It is the only dev row with no definition (a `TODO` in `speechval-dev.yaml`) and
no results anywhere. This design adds the definition, freezes a 200-per-language
subset, and schedules the evaluation on Jupiter, where the audio and the
checkpoints already are.

The `pa_qt` variant is chosen (passage audio, question and options as text),
because it isolates listening comprehension from speech-in-prompt handling.

## Why 200 per language

The reconstruction has **24 languages × 801 samples = 19,224 samples, 567 h of
audio**, not the small split the name suggests. At ~80 s of passage audio per
sample that is 15–20 GPU-h per checkpoint, ~1,400 GPU-h for the 84-checkpoint
battery. Upstream has only one split, so there is no smaller official set to
use; a seeded subset is the only way to make the dev row affordable.

`PolySpeechReader.freeze` already takes `max_samples` and `seed` per source and
is deterministic (`test_max_samples_is_deterministic_and_keeps_order`):

- **200/language → 4,800 records, ~107 h of passage audio.**
- ~3.7–5 GPU-h per checkpoint, **310–420 GPU-h for 84**; ~1–1.5 h wall on a
  4-GPU Jupiter node.
- Noise at 25% chance: about ±3.1% per language, ±0.6% on the 24-language macro.

The subset is selected **after the pa_qt validity filter**, so it is
variant-specific; with seed 0 the same 200 are chosen on every re-freeze.
Results are labelled `polyspeech-100-pa_qt-200` and must not be compared with
full-set PolySpeech numbers.

## Goals

- A `speechval-polyspeech` suite with 24 slices, runnable and frozen on Jupiter.
- A frozen set of exactly 200 samples per language, with stable sample keys.
- A Jupiter campaign over the battery checkpoints, measured first on two
  checkpoints, then scaled.
- Logs ingested by the worklog on artemis after an rsync (run list/inspector);
  a matrix family for `audio_mcq` is a follow-up in the worklog repo.
- Documentation that keeps the dev partition honest about the subset.

## Non-goals

- `pa_qa`, and the full 801/language set.
- Folding PolySpeech into `speechval-dev.yaml` (that would make every artemis
  dev-suite run fail on an unset `${POLYSPEECH_SET}`; see "Approach").
- Copying the reconstruction or the checkpoints to nyx/artemis.
- MCIF, AIR-Bench, and the rest of the test partition.
- A worklog dashboard family for `audio_mcq` (separate repo, follow-up).

## Constraints found during recon

- The reconstruction is **Jupiter-only** (`/e/data1/smurf4eu-data/speech/polyspeech-100`,
  192 GB of wav, 103 GB passages), pinned to revision `5c92353`, CC-BY-NC-4.0.
  Frozen records embed the absolute root, so the freeze and the runs belong on
  Jupiter.
- Battery checkpoints are on Jupiter: **102k–297k under `/p/scratch/...`**
  (no `checkpoints/` component), **300k–600k under
  `/e/project1/.../checkpoints/`**. The nyx staged copies were deleted by the
  sweep, so `/p/scratch` is currently the only copy of the low steps.
- Jupiter is full: 3,818 of 5,884 booster nodes allocated; `sbatch --test-only`
  for a 4-GPU/6 h job estimates a start ~17 h out. QOS `normal` allows 512
  running and 8,192 submitted jobs, so the batch can be queued at once.
- The prior attempt failed on the shared venv: its editable `fbk_speechllm`
  points at a checkout without `prompts/`. A `PYTHONPATH` override cannot win
  against an editable finder. The venv is group-owned and **not writable**, so
  we need a copy we own.

## Approaches considered

**A. Jupiter-native (chosen).** Freeze and run where the audio and checkpoints
are. No 192 GB data copy and no 1.8 TB checkpoint re-staging. Cost: a Jupiter
venv we own, a queue wait, and an rsync of logs to artemis.

**B. Artemis-native.** Copy the reconstruction (192 GB, or 103 GB of passage
audio) to nyx and re-stage every checkpoint (21.8 GB × 84 = 1.8 TB). Uniform
with the rest of the campaign and no new venv, but days of transfer for no
evaluation benefit. Rejected.

**C. Separate suite vs. main-suite integration.** Adding the benchmark to
`speechval-dev.yaml` makes every config that uses the suite resolve
`${POLYSPEECH_SET}`; on artemis that variable has no value and no frozen set,
so existing dev-suite runs would fail at `plan`, including the ST fill. Chosen:
**a separate `speechval-polyspeech.yaml` suite** (the dev partition is the union
of the two), leaving artemis untouched.

## Design

### 1. Spec

`configs/hf/polyspeech-100-pa_qt-200.yaml`: the existing `pa_qt` spec with
`max_samples: 200` and `seed: 0` added to every one of the 24 sources. A test
asserts it differs from `polyspeech-100-pa_qt.yaml` only by those two keys and
that both have the same 24 `dataset_id`s.

### 2. Suite

`configs/eval/suites/speechval-polyspeech.yaml`:

```yaml
benchmarks:
  - name: polyspeech-100-pa_qt-200
    frozen_set: ${POLYSPEECH_PA_QT_SET}
    slices:
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-bg}
      ... 24 languages ...
```

No judge and no post-scorer: `audio_mcq` uses `mcq_scorer` and reports
`accuracy_<dataset_id>` plus `unresolved_rate`.

### 3. Site and frozen set

- `infra/sites/jupiter.sh`: guarded exports of `POLYSPEECH_ROOT` (the
  reconstruction) and `POLYSPEECH_PA_QT_SET` (the frozen set). Guarded, so a
  missing tree fails loudly in `plan`.
- Frozen set at `$PROJECT/$USER/frozen-dev/polyspeech-100-pa_qt-200`, built with:

  ```bash
  export POLYSPEECH_ROOT=/e/data1/smurf4eu-data/speech/polyspeech-100
  melteval freeze configs/hf/polyspeech-100-pa_qt-200.yaml -o $FROZEN
  ```

  Acceptance: `frozen_set.json` reports 4,800 samples, 24 dataset_ids, exactly
  200 per language, and every record's `cut_id` is present in the manifest.

- **Length distribution of the selected subset**, recorded next to the frozen
  set: pooled and per language, n, mean, median, p90, p95, min, max, and the
  fraction above 40 s (the training window). A small
  `scripts/frozen_length_stats.py <frozen-dir>` reads `manifest.jsonl`, prints
  the table and writes `length_stats.json`; `docs/speechval.md` quotes the
  pooled line. This is the evidence behind the extrapolation caveat.

### 4. Jupiter environment

The shared venv is readable but not writable, and its editable installs point at
another user's checkouts (`fbk_speechllm` at `calvo-bartolome1/speechllm`, which
has no `prompts/`; `melteval` at `calvo-bartolome1/eval`). **`PYTHONPATH` wins
against those editable finders** — verified on Jupiter:

```
PYTHONPATH=$REPO:/e/.../speech_stream/speechllm/src \
  .../venvs/smurf-eval/bin/python -c "import fbk_speechllm.prompts, melteval"
fbk:      .../speech_stream/speechllm/src/fbk_speechllm/prompts/__init__.py
melteval: .../attanasio1/eval-polyspeech/melteval/__init__.py
```

So **no venv copy**. The campaign runs from `eval-polyspeech`, synced to the
repo commit, with that `PYTHONPATH`; `sbatch` inherits it. The 6.8 GB copy is a
fallback only, and if used must repoint *both* editable entries, not just
`fbk_speechllm`.

**Code of record:** `$PROJECT/$USER/eval-polyspeech` (the prior attempt's copy,
with `submit_polyspeech.sh`), overwritten from the workstation repo with rsync
and run from there. `jup_env.sh` ends with `cd $PROJECT/$USER/eval`; the submit
path must `cd` into `eval-polyspeech` afterwards, or the launcher runs the wrong
melteval.

### 5. Validity gate: the 40 s training window

The run's `exp_config.yaml` sets `max_duration: 40.0` on every training source.
The sampled passages are much longer: median 78.6 s, p95 145.6 s, max 264.6 s,
and **89% exceed 40 s**. The SMURF provider has no truncation or over-limit
guard (unlike `phi4.py`/`gemma4.py`/`desta25.py`), so the model will be asked to
process 2–6× its training window rather than silently cropping. That is
long-form extrapolation, not the benchmark the model was trained for.

The pilot is the validity gate, not just a throughput measurement: if accuracy
sits at the 25% chance level with a high `unresolved_rate` across languages, the
passages are beyond the model and this dev row is not measurable as defined.

The row is therefore **SMURF-only for now**: the 30–40 s baselines (Qwen2-Audio,
Gemma 4, Phi-4, DeSTA2.5) cannot run it, and `docs/speechval.md` will say so.

### 6. Campaign

Jupiter scratch, `$USER/eval-polyspeech/configs/eval/speechval-polyspeech.yaml`:

- **Checkpoint list, by rule not by prose:** glob `step=*.ckpt` under
  `/p/scratch/e-ext-2025e01-100/speech_stream/eurollm_speech_16_nodes_run_1_asr_long`
  (102k–297k) and under the same run's `/e/project1/.../checkpoints` (300k–600k),
  keep `N % 6000 == 0` with `102000 <= N <= 594000`, then add `600000`. Drop the
  `-v1` and `-last` variants. That yields the 83 sweep checkpoints plus 600k.
- 84 model entries, `smurf//<path>` with one anchored stanza shared by all:
  the eurollm prompt override, `batch_size=8`, `map_location=cpu`, and the
  instructions/system prompts copied **verbatim** from the campaign config the
  sweep used, `/mnt/scratch-artemis/giuseppe/smurf-sweep/speechval-smurf-sweep.yaml`
  (not in the repo). Prompt parity is a known bug class; nothing here is invented.
- `benchmarks: <repo>/configs/eval/suites/speechval-polyspeech.yaml`.
- `pack: {time: "03:00:00", max_slices: 24, gpus: 4}` — one 4-GPU job per
  checkpoint; time tightened after the pilot.
- `output_dir: $SCRATCH/$USER/eval-runs/speechval-polyspeech`.

A small generator (run once, outside the repo) applies the rule above and writes
the model list with anchors; the rendered config is kept next to the campaign.

### 7. Execution order

1. Repo work: spec, suite, site export, docs, tests (`pytest tests -q`, ruff).
2. Sync the repo to `eval-polyspeech`, export `PYTHONPATH`, and verify
   `import fbk_speechllm.prompts, melteval` resolves to the repo copy (login
   node, no GPU).
3. Freeze the 200 subset, run `scripts/frozen_length_stats.py` on it, and
   validate the counts.
4. Smoke: 1 GPU, `-T dataset_id=polyspeech-pa_qt-en -T limit=20` on 102k.
5. Pilot: 102k and 594k × 24 slices, one 4-GPU pack each; record wall time,
   submit→start wait, and accuracy against the 25% chance level.
6. Scale: remaining 82 (or a smaller subset if the pilot's cost per checkpoint
   is worse than estimated) submitted in one batch.
7. `rsync` the campaign folder to
   `/mnt/scratch-artemis/giuseppe/melt-data/eval-runs/speechval-polyspeech`
   so the worklog ingests it; verify the 24 per-language accuracies and the
   `unresolved_rate`. Adding a matrix family/headline for `audio_mcq` is a
   follow-up in the worklog repo, not this plan.

### 8. Tests

- Spec parity: the 200 spec is the full pa_qt spec plus `max_samples`/`seed`.
- Suite completeness: 24 slices, one per source `dataset_id`; no duplicate ids.
- Freeze behaviour: a small reconstruction fixture with `max_samples` (already
  covered) stays green; no test reads the Jupiter data.
- Docs: the dev partition table names the subset and the separate suite.

## Risks

- **`/p/scratch` retention.** 102k–297k exist only there now. If scratch purges,
  those models are gone from Jupiter (and were deleted from nyx). Worth copying
  them under `/e` before the campaign, independent of PolySpeech.
- **Provider/API drift.** The `prompts` fix may expose a second incompatibility
  in the Jupiter speechllm checkout; the smoke run is the gate.
- **Throughput estimate.** 3.7–5 GPU-h/checkpoint is extrapolated from ASR
  packs; the pilot replaces it with a measurement.
- **Passage length vs training window.** 89% of sampled passages exceed the
  40 s `max_duration`; the pilot is the validity gate (see §5).
- **Subset validity.** 200/language is a screening set, and seven languages use
  a different question set; compare on the overlap, as `docs/polyspeech.md`
  already says.
- **License.** CC-BY-NC-4.0; internal evaluation only.

## Open decision left to the owner

After the pilot: all 84 checkpoints, or only the 20 that already have ST? The
20 are derived from the worklog cache (numeric ST across the dev ST families):
102k, 108k, 114k, 120k, 126k, 132k, 138k, 144k, 150k, 156k, 168k, 180k, 192k,
204k, 216k, 228k, 240k, 264k, 300k, 600k. Either set is a `--models` filter; the
pilot's measured cost decides.

The other decision the owner must confirm: whether long passages (89% above the
40 s training window) are acceptable for this dev row, with the pilot's accuracy
as the gate. If not, the dev SQA row needs a different benchmark.
