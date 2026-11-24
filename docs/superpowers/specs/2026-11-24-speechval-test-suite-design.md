# speechval-test suite — design

Date: 2026-11-24
Status: decisions confirmed in conversation; builder check pending

## Intent

The speechval campaigns so far evaluate the **validation/dev** splits
(`configs/eval/suites/speechval-dev.yaml`, `speechval-polyspeech.yaml`). The
final numbers, though, are the **test** ones, and no suite defines them in the
speechval notation: `suites/v1.yaml` is the baselines/whisper-flavoured test
suite (different benchmark names, no MCIF short/SFP selection policy, no MLS).
This design adds `configs/eval/suites/speechval-test.yaml` — the test-split
mirror of `speechval-dev.yaml` — so an upcoming `speechval-test*` campaign can
point at it and every number lands in the worklog under a `speechval-*`
campaign folder.

Written **before** any campaign runs it, on purpose: the worklog's 🚀 Test
page (speecheval-worklog) keys its split classification off these benchmark
ids, so the ids must exist and be stable first.

## Goals

- One suite file, `configs/eval/suites/speechval-test.yaml`, in the exact
  notation of `speechval-dev.yaml` (benchmark `name:`, `spec:`/`frozen_set:`,
  `needs_instruction`, `post_venv` where a post-scorer needs it, one slice per
  language/direction **where the spec's own `dataset_id` tags allow it** —
  `fleurs24-asr-test` tags all languages with one flat id, so it stays one
  pooled slice exactly as v1 runs it).
- **Entire test splits, no subsampling anywhere** — no `limit`, no caps. A
  debug run uses `-T limit=N` against the same frozen set instead of a
  shrunken source (the convention `fleurs24-asr-test.yaml`'s header already
  states).
- Nothing runs, nothing is frozen, no driver config (`configs/eval/*.yaml`
  with models) is written — checkpoint choices come later.

## Non-goals

- New dataset sources for VoiceBench, AdvBench and RedVox. They get
  clearly-marked commented stub entries in the suite; building their sources
  is a separate follow-up project.
- The `long` MCIF track and MCIF SUM — not on the benchmark list this suite is
  written from.
- Any change to `speechval-dev.yaml`, `v1.yaml` or the eval configs.

## The suite's benchmarks

ASR:

| suite benchmark | source | slices |
|---|---|---|
| `librispeech-test-clean` | `spec: configs/hf/librispeech-hf-test-clean.yaml` (v1's own choice for the test split; HF cache must be pulled from a login shell first — README "Evaluating a HuggingFace benchmark") | `asr` × `librispeech-test-clean`, `time: "02:00:00"` |
| `librispeech-test-other` | `spec: configs/hf/librispeech-hf-test-other.yaml` | `asr` × `librispeech-test-other`, `time: "02:00:00"` |
| `voxpopuli-test` | `frozen_set: ${VOXPOPULI_SET}` from `configs/shar/voxpopuli-test.yaml` | 16 languages, `asr` × `voxpopuli-<lang>` — the test spec's own tags carry **no** `-test` suffix (unlike the dev spec's `voxpopuli-<lang>-dev`), and a slice id must match the frozen set's `dataset_id`; `en` gets `args: [-T, normalizer=english]` |
| `mls-test` | `frozen_set: ${MLS_TEST_SET}` from a **new** `configs/shar/mls-test.yaml` (below) | 8 languages (nl en fr de it pl pt es), `asr` × `mls-<lang>`, ids unsuffixed like VoxPopuli's |
| `fleurs24-asr-test` | `frozen_set: ${FLEURS24_ASR_SET}` from `configs/shar/fleurs24-asr-test.yaml` | **one pooled slice, `- {task: asr}`**, exactly as v1's `fleurs24-asr`: the spec tags every language `dataset_id: fleurs`, so per-language slices cannot exist; per-language numbers come from the scorer's grouped metrics |
| `mcif-chunked-asr` | `spec: configs/hf/mcif.yaml` | `chunked_asr` × `mcif-short-fixed-en`, `mcif-short-mixed-en` — the corpus has ASR for target-language `en` only (mcif.yaml header), so these are all the slices that exist |

ST:

| suite benchmark | source | slices |
|---|---|---|
| `fleurs24-st-xen-test` | `frozen_set: ${FLEURS24_ST_XEN_SET}` from `configs/shar/fleurs24-st-xen-test.yaml` (re-frozen from the 25-source spec, see below) | X→en, **25 slices**: the 23 EU locales plus the ru/uk probes — this work un-comments the spec's own ru/uk blocks (`fleurs-ru_en`, `fleurs-uk_en`; the spec header already documents them as intended probes, they were only left commented pending this decision) — each slice with `post_scorers: [melteval/rescore.py@metricx_scorer]`, `post_venv: ${METRICS_VENV_PATH}` |
| `fleurs24-st-enx-test` | `frozen_set: ${FLEURS24_ST_ENX_SET}` from `configs/shar/fleurs24-st-enx-test.yaml` | en→X, one slice per target, MetricX post-scorer, same as dev's enx pattern |
| `covost2-test` | `frozen_set: ${COVOST2_SET}` (the same frozen set v1 uses — it already exists) | 21 X→en slices, MetricX post-scorer |
| `covost2-enx-test` | `frozen_set: ${COVOST2_SET}` | 15 en→X slices, MetricX post-scorer |
| `europarl-st-test` | `frozen_set: ${EUROPARL_ST_TEST_SET}` from `configs/shar/europarl-st-test.yaml` | all 72 directions among the 9 languages, MetricX post-scorer from `custom.sentence` as in dev |
| `mcif-chunked-st` | `spec: configs/hf/mcif.yaml` | `chunked_st` × {`mcif-short-fixed-de`, `mcif-short-mixed-de`} and the same for `it` — TRANS targets are de/it/zh and **zh is skipped per user decision, MCIF needs no Chinese** — each with `post_scorers: [melteval/mcif_scoring.py@mcif_official_trans_scorer]` |

Frozen-set env names match the repo's existing exports for these artifacts,
so a single export serves every suite that reads them: `VOXPOPULI_SET`,
`FLEURS24_ASR_SET`, `FLEURS24_ST_ENX_SET` and `COVOST2_SET` as
`suites/v1.yaml` uses them, `EUROPARL_ST_TEST_SET` as
`docs/jupiter-continuous-eval.md` documents it, and `MLS_TEST_SET` is new.
The one exception is `FLEURS24_ST_XEN_SET`:
this suite uses the X→en set with all 25 test ids (23 EU + ru/uk), while v1's
set was frozen from the 23-source version of the same spec — it must be
(re)frozen from the current 25-source `configs/shar/fleurs24-st-xen-test.yaml`
before this suite runs, or the two probe slices match no sample and their
jobs die at dataset build.

SQA/SLU:

| suite benchmark | source | slices |
|---|---|---|
| `mcif-sfp` | `spec: configs/hf/mcif.yaml` | `audio_chat` × `mcif-short-fixed-{en,de,it}`, each with `post_scorers: [melteval/mcif_scoring.py@mcif_official_qa_scorer]` — SFP read as the short-fixed QA track; zh is skipped here too, per the same user decision |
| `air-bench-chat-speech` | `spec: configs/hf/air-bench.yaml`, **`judge: melt/hf/qwen3_omni`** — audio_chat is a judge task; without a benchmark-level judge the slice generates but never scores (v1's air-bench sets exactly this judge) | `audio_chat` × `air-bench-chat-speech` only — the Chat speech subset, no foundation/sound/music, no mixed |

Safety (stubs — commented entries, not loadable ones):

```yaml
  # - name: voicebench        # test-only, partial coverage; source TBD
  # - name: advbench          # audio jailbreak set; source TBD
  # - name: redvox            # audio safety corpus; source TBD
```

## The new MLS spec

`configs/shar/mls-test.yaml`, extracted from the `mls_sidon` block of
`configs/shar/asr-eval-campaign-v1.yaml` (the only local MLS source — all 8
languages' `test` trees): eight `lhotse_shar` sources under
`${LOCAL_DATASETS_DIR}/mls_sidon/<language>/test`, `text_field:
custom.pnc_text`, and — mirroring `voxpopuli-test.yaml` — **one `dataset_id`
per language** (`mls-<lang>`) so the suite can spread one language per slice
and the worklog can group rows by language. `mls_sidon`'s cut ids repeat
across languages, so nothing may ever be keyed by corpus id (same rule the
repo already applies to VoxPopuli and FLEURS).

## Decisions made in conversation (not builder's calls)

- **FLEURS X→en includes the ru/uk probes** (user decision): the suite's
  25 slices include them, which means un-commenting the spec's own
  `fleurs-ru_en` / `fleurs-uk_en` blocks in
  `configs/shar/fleurs24-st-xen-test.yaml` as part of this work — the spec
  header already documents them as intended probes; they were waiting on
  this decision. The frozen set, when built later, then carries 25 ids.
- **MCIF skips zh everywhere** (user decision): chunked ASR stays `en`
  only, chunked ST runs de/it (not zh), SFP runs en/de/it. The long track
  and SUM remain excluded.

## Calls made here (correct them if wrong)

- **AIR-Bench judge**: `judge: melt/hf/qwen3_omni` at the benchmark level,
  v1's own choice — without it an `audio_chat` slice generates but never
  scores. Only the Chat speech subset is listed, per the benchmark list this
  suite is written from.
- **VoxPopuli/MLS slice ids are unsuffixed** (`voxpopuli-<lang>`,
  `mls-<lang>`): slice ids must match the frozen set's `dataset_id` tags,
  and the test spec's tags carry no `-test`.
- **Wall-clock `time:` entries**: only the ones dev/v1 already set
  (LibriSpeech `02:00:00`). VoxPopuli's spec warns one language needs >6 h
  on a single GPU, so the suite header says plainly: **calibrate `time:`
  per slice before the campaign is scheduled** — the driver-config author
  sets them against the actual models; wrong guesses here mean mass timeouts
  or over-booked queues, and nothing has run yet to calibrate against.
- **PolySpeech** is absent: it is not on the test list; it stays
  validation-only.
- **LibriSpeech via HF spec, not a frozen set**: v1 already made that call
  for the test split; the dev suite freezes only because the HF script
  prepares all splits at once. If offline nodes struggle, freezing is the
  fallback, not a suite change.

## Constraints

- `speechval-dev.yaml` notation exactly: env-var frozen-set refs, quoted
  `time:` strings, one slice per line, comments that say where a number's
  reference text comes from.
- The suite must load through `evaluate.load_benchmarks` as-is (stub comments
  included) — a suite entry pointing at a nonexistent spec fails at load, so
  nothing references a file that does not exist yet.
- ruff line length 119, no `typing` imports, `uv` — repo conventions
  (AGENTS.md) apply to the new spec file too.

## Testing

- Extend the suite-loading test in `tests/test_evaluate.py` (the one that
  loads `suites/v1.yaml` today) to also load `suites/speechval-test.yaml`
  and assert: the **14** live benchmark names, slice counts (16 VoxPopuli,
  8 MLS, 25 FLEURS X→en, 72 Europarl, 21+15 CoVoST, 2+4 MCIF), the
  MetricX/MCIF post-scorers and the AIR-Bench judge on the right entries,
  and that no slice carries a `limit`.
- The MLS spec needs no audio to be *parsed*: a unit test asserts its eight
  sources, per-language `dataset_id`s and `custom.pnc_text` text field.
- Corpus integration stays behind `MELTEVAL_SHAR_ROOT` as everywhere else.
