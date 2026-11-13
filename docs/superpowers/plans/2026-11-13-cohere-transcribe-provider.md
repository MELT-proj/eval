# Cohere Transcribe Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `CohereLabs/cohere-transcribe-03-2026` as the `melt/hf/cohere_transcribe` baseline, open a PR for it, and evaluate it on `speechval-dev`'s ASR slices.

**Architecture:** A thin `HFSpeechChatAPI` subclass (the Whisper pattern) that overrides only the batch step: group the batch by the ISO language code the solver rendered as the instruction, run the processor once per group (it takes one language per call), and decode with the processor's `audio_chunk_index` so long clips come back as one string.

**Tech Stack:** Python 3.12, `transformers>=5.4` (`CohereAsrForConditionalGeneration`, `AutoProcessor`), inspect_ai, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-13-cohere-transcribe-provider-design.md`

## Global Constraints

- Pinned revision `d552dadc2922ba6f6c1f994f8db904d52dcaf942` (the `main` commit staged in the artemis HF cache).
- 14 languages only: `ar de el en es fr it ja ko nl pl pt vi zh`.
- ASR only. No `st`, no judge (`text_only` refused), no entry in any ST benchmark.
- ruff line length 119; no `from typing import ...` (builtin generics only).
- Tests go in the existing `tests/test_provider.py`.
- The PR is branched from `origin/main` and contains provider + router + tests + docs only.
- No GPU work outside SLURM; the smoke and the run go through `scripts/evaluate.py run ... --site artemis`.

## Review Focus

- A batch whose samples carry different languages: one language for the whole batch would transcribe some samples in the wrong language with no error. Pinned by `test_a_batch_is_split_by_language`.
- Audio longer than ~35 s: passing it through without `audio_chunk_index` to `decode` repeats or loses chunks. Pinned by `test_chunk_indices_reach_decode`.
- A missing or unsupported language: defaulting to English silently departs from what was asked. Pinned by `test_an_unsupported_language_is_refused` and by the base class's empty-prompt error.
- Audio not at 16 kHz or missing entirely: the feature extractor would accept and mis-transcribe it. Pinned by `test_audio_at_the_wrong_rate_is_refused`.
- Result order after grouping: a completion landing on another sample's audio is invisible in the score. Pinned by `test_completions_come_back_in_order_and_stripped`.

---

### Task 1: Isolated branch and the design doc

**Files:**
- Create (worktree): `docs/superpowers/specs/2026-11-13-cohere-transcribe-provider-design.md`

**Interfaces:**
- Produces: a git worktree on branch `cohere-transcribe-provider`, based on `origin/main`, where every later task commits.

- [ ] **Step 1: Create the worktree**

Use the `using-git-worktrees` skill. The branch must be based on `origin/main` (not on the local `speechval-canary`), e.g.:

```bash
git -C /home/giuseppe/melt-proj/melt-eval worktree add \
  .claude/worktrees/cohere-transcribe-provider -b cohere-transcribe-provider origin/main
```

- [ ] **Step 2: Move the spec into the worktree and commit it**

```bash
cp /home/giuseppe/melt-proj/melt-eval/docs/superpowers/specs/2026-11-13-cohere-transcribe-provider-design.md \
   <worktree>/docs/superpowers/specs/
cd <worktree>
git add docs/superpowers/specs/2026-11-13-cohere-transcribe-provider-design.md
git commit -m "docs: design for the Cohere Transcribe provider"
```

- [ ] **Step 3: Confirm the base**

Run: `git log --oneline -1 && git merge-base --is-ancestor origin/main HEAD && echo based-on-main`
Expected: the design commit, then `based-on-main`.

---

### Task 2: The provider module

**Files:**
- Create: `melteval/providers/cohere_transcribe.py`
- Test: `tests/test_provider.py` (new class `TestCohereTranscribe`, after `TestWhisper`)

**Interfaces:**
- Consumes: `melteval.providers.base.BatchedSpeechAPI`, `_generate_kwargs`, `_Request`; `melteval.providers.hf.HFSpeechChatAPI`, `check_audio`.
- Produces:
  - `FAMILY = "Cohere Transcribe"`, `LANGUAGES: frozenset[str]`
  - `class CohereTranscribeAPI(HFSpeechChatAPI)` with `family`, `repo`, `revision`, `_load_classes() -> tuple[type, type]`, `_generate_batch(batch: list[_Request]) -> list[str]`
  - `parse_language(text: str) -> str`
  - `_transcribe(model, processor, device: str, batch: list[_Request], language: str, config: GenerateConfig) -> list[str]`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_provider.py`, modelled on `TestWhisper`'s fakes:

```python
class TestCohereTranscribe:
    """Cohere Transcribe's batch step, with a fake model and processor."""

    class _Processor:
        def __init__(self):
            self.feature_extractor = types.SimpleNamespace(sampling_rate=16000)
            self.calls: list[dict] = []
            self.decode_kwargs: dict = {}

        def __call__(self, audios, **kwargs):
            import torch

            self.calls.append({"n": len(audios), **kwargs})
            n = len(audios)
            return {
                "input_features": torch.zeros(n, 2, dtype=torch.float32),
                "attention_mask": torch.ones(n, 2, dtype=torch.long),
                "decoder_input_ids": torch.zeros(n, 3, dtype=torch.long),
                "audio_chunk_index": [(i, None) for i in range(n)],
            }

        def decode(self, ids, **kwargs):
            self.decode_kwargs = kwargs
            language = kwargs.get("language", "?")
            return [f" {language}{i} " for i in range(ids.shape[0])]

    class _Model:
        def __init__(self):
            import torch

            self.dtype = torch.bfloat16
            self.seen: dict = {}

        def generate(self, **kwargs):
            import torch

            self.seen = kwargs
            return torch.arange(kwargs["input_features"].shape[0]).unsqueeze(1)

    def _run(self, languages, config=None):
        pytest.importorskip("torch")
        from melteval.providers.cohere_transcribe import _transcribe

        processor, model = self._Processor(), self._Model()
        batch = [_QwenRequest(text=code) for code in languages]
        config = config or GenerateConfig()
        return _transcribe(model, processor, "cpu", batch, languages[0], config), processor, model
```

`_QwenRequest` already declares `text` and `sample_rate` as dataclass fields, so
`_QwenRequest(text="en")` and `_QwenRequest(sample_rate=8000)` work as written.

Six behaviours to pin (each its own test, exact assertions):

```python
def test_completions_come_back_in_order_and_stripped(self):
    completions, _, _ = self._run(["en", "en"])
    assert completions == ["en0", "en1"]

def test_a_batch_is_split_by_language(self):
    """One language per processor call, results back in request order."""
    pytest.importorskip("torch")
    from melteval.providers.cohere_transcribe import CohereTranscribeAPI

    api = object.__new__(CohereTranscribeAPI)
    api.model, api.processor, api.device = self._Model(), self._Processor(), "cpu"
    api.text_only = False
    batch = [_QwenRequest(text="en"), _QwenRequest(text="de"), _QwenRequest(text="en")]
    assert api._generate_batch(batch) == ["en0", "de0", "en1"]
    assert [call["language"] for call in api.processor.calls] == ["en", "de"]
    assert [call["n"] for call in api.processor.calls] == [2, 1]

def test_chunk_indices_and_language_reach_decode(self):
    _, processor, _ = self._run(["de"])
    assert processor.decode_kwargs["language"] == "de"
    assert processor.decode_kwargs["audio_chunk_index"] == [(0, None)]
    assert processor.decode_kwargs["skip_special_tokens"] is True

def test_generation_is_greedy_and_capped(self):
    _, _, model = self._run(["en"])
    assert model.seen["do_sample"] is False
    assert model.seen["max_new_tokens"] == 256

def test_input_features_take_the_model_dtype(self):
    _, _, model = self._run(["en"])
    assert model.seen["input_features"].dtype == model.dtype

def test_audio_at_the_wrong_rate_is_refused(self):
    with pytest.raises(ValueError, match="16000 Hz"):
        pytest.importorskip("torch")
        from melteval.providers.cohere_transcribe import _transcribe
        processor, model = self._Processor(), self._Model()
        _transcribe(
            model, processor, "cpu",
            [_QwenRequest(text="en", sample_rate=8000)],
            "en", GenerateConfig(),
        )
```

Plus a `TestCohereTranscribeLanguage` class:

```python
def test_a_code_is_lowercased_and_stripped(self):
    from melteval.providers.cohere_transcribe import parse_language
    assert parse_language(" EN\n") == "en"

def test_an_unsupported_code_is_refused_with_the_supported_list(self):
    from melteval.providers.cohere_transcribe import parse_language
    with pytest.raises(ValueError, match="Supported: ar, de, el, en, es, fr, it, ja, ko, nl, pl, pt, vi, zh"):
        parse_language("xx")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_provider.py -q -k CohereTranscribe`
Expected: collection error / `ModuleNotFoundError: melteval.providers.cohere_transcribe`.

- [ ] **Step 3: Implement `melteval/providers/cohere_transcribe.py`**

Module docstring: the model is a dedicated 2B ASR model, the processor takes audio + one language, it chunks long audio itself and reassembles in `decode`, and the language is the instruction (not a chat prompt).

```python
FAMILY = "Cohere Transcribe"
LANGUAGES = frozenset("ar de el en es fr it ja ko nl pl pt vi zh".split())

class CohereTranscribeAPI(HFSpeechChatAPI):
    family = FAMILY
    repo = "CohereLabs/cohere-transcribe-03-2026"
    revision = "d552dadc2922ba6f6c1f994f8db904d52dcaf942"

    def _load_classes(self):
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration
        return CohereAsrForConditionalGeneration, AutoProcessor

    def _generate_batch(self, batch):
        """... groups by language, results back by index; refuse text_only ..."""
```

- `_generate_batch` raises `ValueError` when `self.text_only` ("has no text-only mode: it cannot act as a judge", Whisper's wording).
- `_generate_batch` groups `{parse_language(request.text): [indices]}`, calls `_transcribe(...)` per group with that group's requests and `batch[0].config`, and rebuilds the list in request order.
- `parse_language`: `code = text.strip().lower()`; if not in `LANGUAGES`, raise listing them; return the code.
- `_transcribe`: `check_audio(batch, processor.feature_extractor.sampling_rate, FAMILY)`; `inputs = processor([r.audio for r in batch], sampling_rate=16000, return_tensors="pt", language=language)`; pop `audio_chunk_index`; move tensors to device; cast `input_features` to `model.dtype`; `with torch.no_grad(): generated = model.generate(**inputs, **_generate_kwargs(config))`; `decoded = processor.decode(generated, skip_special_tokens=True, audio_chunk_index=chunk_index, language=language)`; normalise a lone string to a one-item list; return stripped strings.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_provider.py -q -k CohereTranscribe`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add melteval/providers/cohere_transcribe.py tests/test_provider.py
git commit -m "Add a provider for Cohere Transcribe (cohere-transcribe-03-2026)"
```

---

### Task 3: Register the model and document it

**Files:**
- Modify: `melteval/providers/router.py` (import + `HF_MODELS`)
- Modify: `docs/baseline-providers.md` (table row + section)
- Test: `tests/test_provider.py` (registration assertion)

**Interfaces:**
- Consumes: `CohereTranscribeAPI` from Task 2.
- Produces: `melt/hf/cohere_transcribe` resolving through `melteval.providers.router.HF_MODELS`.

- [ ] **Step 1: Write the failing registration test**

```python
def test_cohere_transcribe_is_registered_as_an_hf_model():
    from melteval.providers import family_of
    from melteval.providers.cohere_transcribe import CohereTranscribeAPI
    from melteval.providers.router import HF_MODELS
    assert HF_MODELS["cohere_transcribe"] is CohereTranscribeAPI
    assert family_of("melt/hf/cohere_transcribe") == "hf"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_provider.py -q -k registered_as_an_hf_model`
Expected: FAIL with `KeyError: 'cohere_transcribe'`.

- [ ] **Step 3: Register it**

In `melteval/providers/router.py`, import `CohereTranscribeAPI` and add `"cohere_transcribe": CohereTranscribeAPI,` to `HF_MODELS`.

- [ ] **Step 4: Document it in `docs/baseline-providers.md`**

Table row after Whisper:

```markdown
| Cohere Transcribe (2B)        | `melt/hf/cohere_transcribe` | `d552dadc2922ba6f6c1f994f8db904d52dcaf942` | no limit (chunked) |
```

Section with: the model is ASR-only in 14 languages; the language is the instruction (`-T instruction='"{lang_code}"'`, validated, one language per processor call so batches are grouped); long-form audio is chunked and reassembled, nothing is truncated; `transformers>=5.4`; the Hub repo is gated, so `prefetch` needs an accepted account and `HF_TOKEN` from a login node. Include a `submit_eval.sh` example on a VoxPopuli frozen set.

- [ ] **Step 5: Run the test and lint**

Run: `pytest tests/test_provider.py -q && ruff check melteval tests`
Expected: pass, no lint errors.

- [ ] **Step 6: Commit**

```bash
git add melteval/providers/router.py docs/baseline-providers.md tests/test_provider.py
git commit -m "Register melt/hf/cohere_transcribe and document it"
```

---

### Task 4: Whole-branch check and the PR

**Files:** none (verification and publication).

- [ ] **Step 1: Run the full unit suite and lint**

```bash
pytest tests -q && ruff check .
```

Expected: everything that ran passes (corpus integration tests skip without `MELTEVAL_SHAR_ROOT`).

- [ ] **Step 2: Review the diff against main**

Run: `git diff --stat origin/main...HEAD`
Expected: the spec, the provider, the router line, the docs, the tests. Nothing from `speechval-canary` (no campaign configs, no worklog).

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin cohere-transcribe-provider
gh pr create --base main --head cohere-transcribe-provider \
  --title "Add a provider for Cohere Transcribe (cohere-transcribe-03-2026)" \
  --body "Adds melt/hf/cohere_transcribe: a dedicated 2B ASR model in 14 languages.
The language is the instruction ({lang_code}); the provider groups each batch by
language because the processor takes one per call, and decodes with the
processor's chunk index so long audio is not truncated. ASR only: no ST, no judge.
Pinned to revision d552dad.... Tests use a fake model/processor; docs/baseline-providers.md
covers invocation and the gated-repo staging note."
```

- [ ] **Step 4: Report the PR URL**

Run: `gh pr view --json url -q .url`
Expected: a `https://github.com/MELT-proj/eval/pull/N` URL.

---

### Task 5: Campaign config, mirror sync, and a smoke run

**Files:**
- Create: `configs/eval/speechval-cohere.yaml` (on the `speechval-canary` branch, NOT in the PR)
- Modify: `docs/speechval.md` (staging section, after the run)

**Interfaces:**
- Consumes: `melt/hf/cohere_transcribe` merged into the `speechval-canary` working tree.
- Produces: `${EVAL_ROOT}/speechval-cohere` logs that the worklog ingests.

- [ ] **Step 1: Merge the provider branch into `speechval-canary`**

```bash
git -C /home/giuseppe/melt-proj/melt-eval checkout speechval-canary
git merge --no-ff cohere-transcribe-provider
```

Expected: clean merge; `melteval/providers/cohere_transcribe.py` present on the branch.

- [ ] **Step 2: Write `configs/eval/speechval-cohere.yaml`**

```yaml
# Cohere Transcribe (ASR-only, 14 languages) on speechval-dev's ASR slices.
#
#   python scripts/evaluate.py plan configs/eval/speechval-cohere.yaml --site artemis
#   python scripts/evaluate.py run  configs/eval/speechval-cohere.yaml --site artemis --wait

name: speechval-cohere
output_dir: ${EVAL_ROOT}/speechval-cohere

models:
  - name: cohere-transcribe-03-2026
    model: melt/hf/cohere_transcribe
    args: [-M, batch_size=8]
    # The language code is the instruction: the processor takes one language per
    # call and the model has no language detection. The provider validates it.
    instruction:
      asr: "{lang_code}"

benchmarks: configs/eval/suites/speechval-dev.yaml

skip:
  # ASR only: no speech translation.
  - fleurs24-st-xen-dev
  - fleurs24-st-enx-dev
  - covost2-dev
  - covost2-enx-dev
  - europarl-st-dev
  # VoxPopuli languages outside the model's 14.
  - voxpopuli-ro-dev
  - voxpopuli-hu-dev
  - voxpopuli-cs-dev
  - voxpopuli-fi-dev
  - voxpopuli-hr-dev
  - voxpopuli-sk-dev
  - voxpopuli-sl-dev
```

- [ ] **Step 3: Stage the checkpoint into the site's HF cache**

On the artemis login node (the compute nodes run `HF_HUB_OFFLINE=1`):

```bash
ssh artemis 'cd /mnt/scratch-artemis/giuseppe/melt-eval-speechval && \
  source /mnt/scratch-artemis/giuseppe/venvs/melteval/bin/activate && \
  export HF_TOKEN=$(cat ~/.cache/huggingface/token) && \
  python scripts/evaluate.py prefetch configs/eval/speechval-cohere.yaml --site artemis'
```

Expected: `fetch melt/hf/cohere_transcribe: CohereLabs/cohere-transcribe-03-2026@d552dad...` and no error (already cached as of 2026-11-13).

- [ ] **Step 4: Sync code and config to the artemis mirror**

The mirror is an rsync copy, not a checkout:

```bash
rsync -a configs/eval/speechval-cohere.yaml artemis:/mnt/scratch-artemis/giuseppe/melt-eval-speechval/configs/eval/
rsync -a melteval/providers/cohere_transcribe.py melteval/providers/router.py artemis:/mnt/scratch-artemis/giuseppe/melt-eval-speechval/melteval/providers/
```

- [ ] **Step 5: Plan and smoke five samples**

```bash
ssh artemis 'cd /mnt/scratch-artemis/giuseppe/melt-eval-speechval && \
  source /mnt/scratch-artemis/giuseppe/venvs/melteval/bin/activate && \
  python scripts/evaluate.py plan configs/eval/speechval-cohere.yaml --site artemis | tail -3 && \
  MELT_PARTITION=h200 MELT_QOS=gpu-h200 python scripts/evaluate.py run configs/eval/speechval-cohere.yaml \
    --site artemis --benchmarks librispeech-validation-clean voxpopuli-dev --limit 5 --wait'
```

Expected: `would submit 9 slice(s) as 1 pack(s)`; the smoke finishes with every log `success`.

- [ ] **Step 6: Inspect the smoke transcripts**

Read one LibriSpeech and one non-English VoxPopuli log: the completion must be a plausible transcript in the sample's language, not English-by-default and not the reference echoed. This is the check that the language grouping reached the processor (`melteval:format_spec` shows the rendered instruction).

---

### Task 6: Full run, scores, and the record

**Files:**
- Modify: `docs/speechval.md` (Cohere row + staging notes)

- [ ] **Step 1: Run all nine slices**

```bash
ssh artemis 'cd /mnt/scratch-artemis/giuseppe/melt-eval-speechval && \
  source /mnt/scratch-artemis/giuseppe/venvs/melteval/bin/activate && \
  MELT_PARTITION=h200 MELT_QOS=gpu-h200 python scripts/evaluate.py run configs/eval/speechval-cohere.yaml --site artemis --wait'
```

Expected: 9/9 logs `success` (WER and the normalizers are inline; there are no post-scorers for ASR).

- [ ] **Step 2: Verify the log count and WER values**

```bash
ssh artemis 'python3 - <<PY
import glob, json, collections
paths = glob.glob("/mnt/scratch-artemis/giuseppe/melt-data/eval-runs/speechval-cohere/logs/*/*/*/*.json")
print(len(paths), collections.Counter(json.load(open(p)).get("status") for p in paths))
for p in sorted(paths):
    j = json.load(open(p))
    score = j["results"]["scores"][0]["metrics"].get("corpus_wer_raw", {}).get("value")
    print(j["eval"]["metadata"]["melteval_slice"], round(score, 4))
PY'
```

Expected: `9 Counter({'success': 9})` and Latin-script VoxPopuli WERs in a plausible range (0.05–0.20), not ~1.0 (wrong language) or ~0.0.

- [ ] **Step 3: Add the row to `docs/speechval.md`**

Extend the model table with the eight columns used by the other rows: LibriSpeech and VoxPopuli WER, and `–` in every ST column; add a short staging paragraph (pinned revision, gated repo, language-as-instruction, long-form chunking, 9 slices and why).

- [ ] **Step 4: Check the worklog sees the campaign**

```bash
curl -s localhost:8765/api/snapshot | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['kpis']); print(d['filters']['models'])"
```

Expected: one more model and nine more slices, `pending` back to 0. If the model shows up without a family, note the family mapping in the report rather than editing the app in this task.

- [ ] **Step 5: Commit the campaign and the docs**

```bash
git add configs/eval/speechval-cohere.yaml docs/speechval.md
git commit -m "speechval: cohere transcribe baseline"
```
