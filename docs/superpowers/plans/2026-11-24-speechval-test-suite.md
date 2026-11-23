# speechval-test Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write `configs/eval/suites/speechval-test.yaml` (plus one new Shar spec and one spec edit) defining the full test-split benchmark suite, without freezing or running anything.

**Architecture:** A suite file in the exact notation of `speechval-dev.yaml`, referencing existing source configs (HF specs read live, Shar specs via frozen-set env vars) and one new extracted Shar spec for MLS. Nothing executes: the deliverable is configuration plus the tests that pin it.

**Tech Stack:** YAML config files consumed by `scripts/evaluate.py` (`load_config` / `load_benchmarks`); pytest with the melteval venv.

**Spec:** `docs/superpowers/specs/2026-11-24-speechval-test-suite-design.md` (this plan argues from it; read both)

## Global Constraints

- Work in a worktree/branch off `origin/main`. Never touch the `speechval-canary` branch or its uncommitted `configs/eval/speechval-fondue.yaml` modification.
- Nothing is frozen, nothing is run, no GPU/SLURM anything. Verification is pytest + CPU-only reads.
- **Entire test splits, no subsampling:** no `limit` key on any slice, no caps in any source.
- Every file the suite references must already exist; a suite entry pointing at a nonexistent spec fails at load.
- Repo conventions (AGENTS.md): ruff line length 119, no `typing` imports, add tests to existing test files.
- Test runner: `/mnt/scratch-artemis/giuseppe/venvs/melteval/bin/python -m pytest`.
- Slice ids must equal the referenced spec's own `dataset_id` tags, which are **unsuffixed** for the Shar test specs (`voxpopuli-de`, not `voxpopuli-de-test`), except LibriSpeech (`librispeech-test-clean/-other`) and MCIF (`mcif-short-fixed-en`…), whose sources define their own ids.

## Review Focus

Inputs and failure modes the spec implies but no run has exercised yet — each pinned by a test in the task that owns it:

1. A slice `dataset_id` that matches no `dataset_id` tag in the referenced spec → a job that dies at dataset build (`_raise_no_matches`). Pinned by the cross-reference assertions in Task 3 (suite ids ⊆ spec ids) and Task 2's test (FLEURS ids).
2. A judge task (`audio_chat`) with no judge → generates but never scores, silently. Pinned by Task 3's assertion that `air-bench-chat-speech` carries `judge: melt/hf/qwen3_omni`.
3. Subsampling sneaking in as a `limit` on a slice. Pinned by Task 3's no-`limit` assertion over every slice.
4. A benchmark whose frozen-set env var name drifts from the one the driver will export (`${VOXPOPULI_TEST_SET}` typo'd as `${VOXPOPULI_SET}`, say) → dead campaign. Pinned by Task 3's assertion of the exact env var strings.
5. `needs_instruction` missing from a speech benchmark → prompts silently wrong. Pinned by Task 3's assertion on the ten speech benchmarks.
6. (Check, not a test) FLEURS ru/uk cuts whose sentence ids fall outside `configs/refs/fleurs-en-reference-test.json` → empty references, garbage scores. Task 2 verifies coverage where the corpus tree is reachable and documents the assumption otherwise.

---

### Task 1: MLS source spec

**Files:**
- Create: `configs/shar/mls-test.yaml`
- Test: `tests/test_freeze.py` (existing file — source-spec tests live there)

**Interfaces:**
- Produces: a source spec named `mls-test` whose eight sources carry `dataset_id: mls-<lang>` for `{nl en fr de it pl pt es}`; the suite (Task 3) will reference its frozen set as `${MLS_TEST_SET}` and its slices by these ids.

- [ ] **Step 1: Commit the design doc alone on the worktree branch**

```bash
git add docs/superpowers/specs/2026-11-24-speechval-test-suite-design.md
git commit -m "docs: specify the speechval-test suite"
```

- [ ] **Step 2: Write the failing test in `tests/test_freeze.py`**

Add a test class `TestSourceSpecs` with `test_mls_test_spec()`:
`yaml.safe_load` of `configs/shar/mls-test.yaml` asserts — `name == "mls-test"`, exactly 8 sources, all `type == "lhotse_shar"`, the `dataset_id` set is exactly `{"mls-" + code}` for the eight codes above, every source's `text_field == "custom.pnc_text"`, every `shar_path` ends with `/test`, and the `lang` tags match each source's id code.

- [ ] **Step 3: Run it to verify it fails**

Run: `/mnt/scratch-artemis/giuseppe/venvs/melteval/bin/python -m pytest tests/test_freeze.py::TestSourceSpecs -v`
Expected: FAIL — file not found.

- [ ] **Step 4: Write `configs/shar/mls-test.yaml`**

Mirror `configs/shar/voxpopuli-test.yaml`'s structure and comment style: `name: mls-test`, `seed: 0`, eight `lhotse_shar` sources under `${LOCAL_DATASETS_DIR}/mls_sidon/<language>/test` for dutch/english/french/german/italian/polish/portuguese/spanish (tags: `task: asr`, `lang: <code>`, `dataset_id: mls-<code>`), `text_field: custom.pnc_text`. Header comment states: sources extracted from the `mls_sidon` block of `configs/shar/asr-eval-campaign-v1.yaml`; one `dataset_id` per language so runs split per language; MLS cut ids repeat across languages, so nothing may ever be keyed by corpus id (the repo's standing rule for VoxPopuli/FLEURS).

- [ ] **Step 5: Run the test to verify it passes, then run the freeze tests**

Run: `/mnt/scratch-artemis/giuseppe/venvs/melteval/bin/python -m pytest tests/test_freeze.py -q`
Expected: PASS, no regressions.

- [ ] **Step 6: Commit**

```bash
git add configs/shar/mls-test.yaml tests/test_freeze.py
git commit -m "feat: add the mls test source spec"
```

### Task 2: FLEURS X→en ru/uk probes on

**Files:**
- Modify: `configs/shar/fleurs24-st-xen-test.yaml` (un-comment the two ru/uk blocks near lines 198–210; reword the line ~54 comment that says they stay commented)
- Test: `tests/test_freeze.py` (`TestSourceSpecs`)

**Interfaces:**
- Produces: a spec with 25 active sources, `dataset_id`s ending in `fleurs-ru_en` and `fleurs-uk_en`; the suite (Task 3) lists 25 slices for this benchmark.

- [ ] **Step 1: Write the failing test**

`TestSourceSpecs.test_fleurs_xen_test_spec()`: `yaml.safe_load` of `configs/shar/fleurs24-st-xen-test.yaml` asserts — exactly 25 sources, the `dataset_id` set contains `fleurs-ru_en` and `fleurs-uk_en`, all ids unique, every source `task == "st"` and `tgt_lang == "en"`.

- [ ] **Step 2: Run it to verify it fails**

Run: `/mnt/scratch-artemis/giuseppe/venvs/melteval/bin/python -m pytest tests/test_freeze.py::TestSourceSpecs -v`
Expected: FAIL — 23 sources, no ru/uk.

- [ ] **Step 3: Verify ru/uk reference coverage, then un-comment**

First check the assumption the scoring rests on: cuts in `${LOCAL_DATASETS_DIR}/fleurs/ru_ru/test` and `uk_ua/test` must have sentence ids covered by `configs/refs/fleurs-en-reference-test.json` (the shared English reference map; try the corpus trees the repo already documents — the nyx/scratch Shar roots in AGENTS.md and README). Read manifests only, CPU-only. If the trees are unreachable or coverage is partial, STOP and report back with what you found instead of editing; otherwise un-comment the two blocks exactly as written (keeping their tags and the provenance comments), and reword the "ru/uk stay commented" note to record the 2026-11-24 decision and, if you checked it, that reference coverage was verified.

- [ ] **Step 4: Run the test to verify it passes**

Run: `/mnt/scratch-artemis/giuseppe/venvs/melteval/bin/python -m pytest tests/test_freeze.py::TestSourceSpecs -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/shar/fleurs24-st-xen-test.yaml tests/test_freeze.py
git commit -m "feat: enable the fleurs x-to-en ru/uk test probes"
```

### Task 3: The suite file

**Files:**
- Create: `configs/eval/suites/speechval-test.yaml`
- Test: `tests/test_evaluate.py` (`class TestSuite`, which already loads `suites/v1.yaml`)

**Interfaces:**
- Consumes: the Task 1 spec (`mls-<lang>` ids) and the Task 2 spec (25 FLEURS X→en ids).
- Produces: the suite the future `speechval-test*` campaign points at; the worklog's split classification (separate project) keys off these benchmark `name:`s verbatim.

- [ ] **Step 1: Write the failing tests in `class TestSuite`**

Two tests, both loading via `evaluate.load_config(self._write(tmp_path, "configs/eval/suites/speechval-test.yaml"), None)`:

`test_speechval_test_suite_structure` asserts:
- The 14 benchmark names, in order: `librispeech-test-clean`, `librispeech-test-other`, `voxpopuli-test`, `mls-test`, `fleurs24-asr-test`, `mcif-chunked-asr`, `fleurs24-st-xen-test`, `fleurs24-st-enx-test`, `covost2-test`, `covost2-enx-test`, `europarl-st-test`, `mcif-chunked-st`, `mcif-sfp`, `air-bench-chat-speech`.
- Slice counts, same order: 1, 1, 16, 8, 1, 2, 25, 23, 21, 15, 72, 4, 3, 1.
- No slice contains a `limit` key.
- `needs_instruction` is true on the ten speech benchmarks and absent on the three MCIF ones and `air-bench-chat-speech`.
- `air-bench-chat-speech` has `judge == "melt/hf/qwen3_omni"`.
- The eight frozen-set benchmarks reference exactly `${VOXPOPULI_TEST_SET}`, `${MLS_TEST_SET}`, `${FLEURS24_ASR_TEST_SET}`, `${FLEURS24_ST_XEN_TEST_SET}`, `${FLEURS24_ST_ENX_TEST_SET}`, `${COVOST2_SET}` (×2, both CoVoST2 entries) and `${EUROPARL_ST_TEST_SET}`.
- Every st slice of the FLEURS/CoVoST2/Europarl benchmarks carries `post_scorers: [melteval/rescore.py@metricx_scorer]` and its benchmark has `post_venv: ${METRICS_VENV_PATH}`.

`test_speechval_test_suite_slice_ids_match_their_specs` cross-references every slice id against the source it claims (the dead-job guard):
- For each Shar spec (`configs/shar/{voxpopuli-test,mls-test,fleurs24-st-xen-test,fleurs24-st-enx-test,europarl-st-test}.yaml`): the suite slice ids for that benchmark equal the set of `dataset_id` tags in the spec.
- The two CoVoST2 benchmarks' ids ⊆ the `dataset_id` tags in `configs/shar/st-eval-campaign-v1.yaml` (the spec behind `${COVOST2_SET}`), split X→en / en→X by direction.
- `librispeech-test-clean`/`-other` match the `dataset_id` in `configs/hf/librispeech-hf-test-{clean,other}.yaml`.
- MCIF ids ⊆ the dataset ids `configs/hf/mcif.yaml` defines; `air-bench-chat-speech` matches the id in `configs/hf/air-bench.yaml`.

- [ ] **Step 2: Run them to verify they fail**

Run: `/mnt/scratch-artemis/giuseppe/venvs/melteval/bin/python -m pytest tests/test_evaluate.py::TestSuite -v`
Expected: FAIL — suite file not found.

- [ ] **Step 3: Write `configs/eval/suites/speechval-test.yaml`**

Follow `speechval-dev.yaml` notation exactly. Benchmark by benchmark:

- `librispeech-test-clean` / `librispeech-test-other`: `spec: configs/hf/librispeech-hf-test-{clean,other}.yaml`, `needs_instruction: true`, slice `{task: asr, dataset_id: <benchmark name>, time: "02:00:00"}`.
- `voxpopuli-test`: `frozen_set: ${VOXPOPULI_TEST_SET}`, `needs_instruction: true`; 16 slices `voxpopuli-<lang>` (the spec's tags, no `-test` suffix); the `en` slice gets `args: [-T, normalizer=english]`.
- `mls-test`: `frozen_set: ${MLS_TEST_SET}`, `needs_instruction: true`; 8 slices `mls-<lang>`.
- `fleurs24-asr-test`: `frozen_set: ${FLEURS24_ASR_TEST_SET}`, `needs_instruction: true`; **one pooled slice `- {task: asr}`** with no `dataset_id`, exactly as v1's `fleurs24-asr` runs it (the spec tags every language with one flat id).
- `mcif-chunked-asr`: `spec: configs/hf/mcif.yaml`; slices `{task: chunked_asr, dataset_id: mcif-short-fixed-en}` and `{task: chunked_asr, dataset_id: mcif-short-mixed-en}` — the corpus defines ASR for target `en` only.
- `fleurs24-st-xen-test`: `frozen_set: ${FLEURS24_ST_XEN_TEST_SET}`, `needs_instruction: true`, `post_venv: ${METRICS_VENV_PATH}`; 25 slices `fleurs-<lang>_en` (23 EU locales + `fleurs-ru_en`, `fleurs-uk_en`), each with `post_scorers: [melteval/rescore.py@metricx_scorer]`.
- `fleurs24-st-enx-test`: same shape, `${FLEURS24_ST_ENX_TEST_SET}`, 23 slices `fleurs-en_<lang>`.
- `covost2-test` / `covost2-enx-test`: both `frozen_set: ${COVOST2_SET}` (the set v1 already uses), `needs_instruction: true`, `post_venv: ${METRICS_VENV_PATH}`; 21 slices `covost2-<src>_en` and 15 slices `covost2-en_<tgt>`, MetricX post-scorer on each.
- `europarl-st-test`: `frozen_set: ${EUROPARL_ST_TEST_SET}`, `needs_instruction: true`, `post_venv: ${METRICS_VENV_PATH}`; 72 slices `europarl-st-<src>_<tgt>` (all ordered directions among en fr de es it pt pl ro nl), MetricX post-scorer on each.
- `mcif-chunked-st`: `spec: configs/hf/mcif.yaml`; 4 slices `{task: chunked_st, dataset_id: mcif-short-{fixed,mixed}-{de,it}, post_scorers: [melteval/mcif_scoring.py@mcif_official_trans_scorer]}` — zh skipped by decision.
- `mcif-sfp`: `spec: configs/hf/mcif.yaml`; 3 slices `{task: audio_chat, dataset_id: mcif-short-fixed-{en,de,it}, post_scorers: [melteval/mcif_scoring.py@mcif_official_qa_scorer]}`.
- `air-bench-chat-speech`: `spec: configs/hf/air-bench.yaml`, `judge: melt/hf/qwen3_omni` (comment: v1's own judge choice — audio_chat never scores without it); one slice `{task: audio_chat, dataset_id: air-bench-chat-speech}`.

Suite header comment: purpose (the test half of the validation/test split, `speechval-dev.yaml`'s mirror), the no-subsampling rule (`-T limit=N` for debug runs, never a shrunken source), the stub note below, and the wall-clock warning — only LibriSpeech carries `time:`; VoxPopuli's spec warns of >6 h per language on one GPU, so **calibrate `time:` per slice before the campaign is scheduled**.

At the end, three commented stubs (not loadable entries), each naming the intended dataset:

```yaml
  # - name: voicebench       # test-only, partial coverage; source TBD
  # - name: advbench         # audio jailbreak set; source TBD
  # - name: redvox           # audio safety corpus; source TBD
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/mnt/scratch-artemis/giuseppe/venvs/melteval/bin/python -m pytest tests/test_evaluate.py::TestSuite -v`
Expected: PASS, and the existing v1 tests in the class still pass.

- [ ] **Step 5: Full check and commit**

Run: `/mnt/scratch-artemis/giuseppe/venvs/melteval/bin/python -m pytest tests/test_evaluate.py tests/test_freeze.py -q` and `ruff check .`
Expected: PASS / clean.

```bash
git add configs/eval/suites/speechval-test.yaml tests/test_evaluate.py
git commit -m "feat: add the speechval-test suite"
```
