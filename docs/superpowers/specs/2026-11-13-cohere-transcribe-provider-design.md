# Cohere Transcribe provider — design

## Intent

Add `CohereLabs/cohere-transcribe-03-2026` to the harness as an off-the-shelf
ASR baseline, `melt/hf/cohere_transcribe`, so it can be run on `speechval-dev`
like the other baselines, and open that as a PR against `main`.

Why: it is a dedicated, 2B, conformer encoder–decoder ASR model reported at
5.42 mean WER on the Open ASR Leaderboard, in 14 languages. The suite's only
ASR rows today are LibriSpeech, VoxPopuli and a pair of general speech LLMs;
this is the first model whose whole purpose is transcription.

## What the model gives us

From the model card and the installed `transformers` (5.16.1, which has
`CohereAsrForConditionalGeneration`; the card requires >= 5.4):

- **ASR only.** No translation, no chat, no timestamps, no diarization.
- **14 languages:** `ar de el en es fr it ja ko nl pl pt vi zh`. Card is
  explicit that it has no language detection and should be told the language.
- **The processor is the whole prompt.** `processor(audio, sampling_rate=16000,
  language="en")` builds a fixed decoder prompt from the language token and
  returns `input_features`, `attention_mask`, `decoder_input_ids` and
  `audio_chunk_index`. There is no text prompt and no chat template.
- **One language per call.** `language` is a single string for the whole call,
  so a mixed-language batch cannot be sent as one.
- **Long-form is internal.** The feature extractor splits audio longer than
  ~35 s at quiet boundaries and returns one row per chunk plus
  `audio_chunk_index=[(sample, chunk|None), ...]`; `processor.decode(...,
  audio_chunk_index=..., language=...)` reassembles per-sample text. Nothing
  is truncated and no 30 s reject is needed.
- **Punctuation on by default** (`punctuation=True`), which is the card's
  recommended usage.
- Hub repo is gated but the account has accepted it; pinned revision is the
  current `main` commit, `d552dadc2922ba6f6c1f994f8db904d52dcaf942`.

## Goals

- A provider registered as `melt/hf/cohere_transcribe`, family `hf`, that
  produces one completion per sample, in request order.
- The language comes from the sample, not from a launch argument.
- Long-form audio is transcribed whole, not cut at a window.
- Unit tests with a fake model/processor (no checkpoint), docs, a PR.
- A campaign config and a run on `speechval-dev`'s ASR slices.

## Non-goals

- Speech translation (the model cannot).
- Adding FLEURS ASR to `speechval-dev` so the non-EU languages are measured;
  that changes the suite for every model, so it is a separate decision.
- vLLM serving, timestamps, diarization, language auto-detection.

## Approaches

**(A) The language code is the instruction, and the provider groups a batch by
language.** The campaign writes `asr: "{lang_code}"`; the solver renders the
sample's ISO code (`en`, `de`, …); the provider validates it against the 14
languages, groups the batch's indices by code, and calls the processor once per
group with that language. Chosen: it keeps one config for all slices, uses the
launcher's existing per-sample language metadata, and matches how Canary already
receives its language pair.

**(B) `-M language=` fixed per run.** One run per language (nine runs), no
per-slice language, and a wrong language silently mis-transcribes. Rejected.

**(C) Leave the language out and let the model decide.** The card says it has no
language detection; omitting it is a guaranteed silent quality loss. Rejected.

## Design

**`melteval/providers/cohere_transcribe.py`**

- `CohereTranscribeAPI(HFSpeechChatAPI)` — same loading path as Whisper
  (`repo`, `revision`, `_load_classes`, dtype/device/revision handling,
  HF-cache-offline support). It overrides only the batch step; the chat-template
  machinery in `hf.py` is never used.
- `LANGUAGES` is a frozenset of the 14 codes; `parse_language(text)` strips and
  lowercases the rendered prompt, requires exactly one supported code, and
  raises listing what is supported. An empty prompt already fails in
  `BatchedSpeechAPI.generate()`, because `ignores_instruction` stays `False`
  (the "instruction" is the language, and a sample without one cannot be run
  correctly).
- `_generate_batch`: group indices by language, call a module-level
  `_transcribe(model, processor, device, batch, language)` per group (module
  function so tests can pass fakes), place results back by index. `text_only`
  is refused, as for Whisper: this model cannot judge.
- `_transcribe`: processor call with `sampling_rate=16000`, `return_tensors=pt`,
  `language=<code>`; move tensors to device; cast `input_features` to the model
  dtype; greedy `generate(max_new_tokens=config.max_tokens or 256)`; decode with
  `skip_special_tokens=True, audio_chunk_index=..., language=<code>`; strip.
- No `_reject` override: the processor's chunking is the long-form story.

**`melteval/providers/router.py`** — `HF_MODELS["cohere_transcribe"]`.

**`configs/eval/speechval-cohere.yaml`** (kept on the `speechval-canary`
branch with the other campaigns, not in the PR): one model, no special venv,
`args: [-M, batch_size=8]`, `instruction: {asr: "{lang_code}"}`, and a `skip:`
list of the five ST benchmarks plus the seven VoxPopuli languages the model does
not cover (`ro hu cs fi hr sk sl`). Nine slices: LibriSpeech clean/other,
VoxPopuli en/de/fr/es/it/nl/pl.

**Docs** — a row in `docs/baseline-providers.md` and a short section: model
string, pinned revision, language spec, long-form behaviour, ASR-only.

**PR** — branch `cohere-transcribe-provider` from `origin/main`, containing
provider + registration + tests + docs. The campaign config and campaign docs
stay with the speechval work.

## Verification

- `pytest tests/test_provider.py -q` (fake model/processor: order, grouping by
  language, chunk args reach decode, unsupported/missing language errors,
  `text_only` refused).
- `ruff check`.
- A `--limit 5` smoke on LibriSpeech + VoxPopuli through SLURM, checked for
  sane transcripts and WER, then the full nine slices.
- The worklog picks the campaign up from the eval root automatically.

## Open items settled by the build

- Whether the model accepts `attn_implementation="sdpa"` (the shared loader's
  default) or needs `eager`; decided by loading it once, not by guessing.
- Exact batch size (8 unless the smoke says otherwise).
- Whether `processor.decode` returns a list or a string when a batch has no
  chunking; handled defensively and covered by a test.
