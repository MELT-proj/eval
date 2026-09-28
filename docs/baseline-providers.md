# Baseline providers (qwen only so far)

Baslines are used as reference points next to MELT and SMURF.

So far this only includes speech LLMs loaded through `transformers.`

Each one is a thin subclass of`melteval/providers/hf.py`, which owns what they have in common. A subclass says only which classes to load, whether its processor truncates long clips, and how its `generate()` takes its arguments and returns its output.

| Provider                    | `--model`                | `-T prompt_style` | Longest clip          |
| --------------------------- | -------------------------- | ------------------- | --------------------- |
| Qwen2-Audio-7B-Instruct     | `qwen2_audio/<snapshot>` | `qwen2_audio`     | 30 s (refused beyond) |
| Qwen3-Omni-30B-A3B-Instruct | `qwen3_omni/<snapshot>`  | `qwen3_omni`      | no fixed limit        |

**Who builds what.** The solver (`instruction_prompt`) hands over the bare
instruction; the provider adds the chat template and the audio token.

## Qwen2-Audio-7B-Instruct

```bash
inspect eval melteval/tasks.py@asr \
  --model qwen2_audio/$HF_HOME/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/<sha> \
  -T frozen_set=/path/to/frozen-set \
  -T prompt_style=qwen2_audio \
  -T instruction='"Transcribe this audio."' \
  -M batch_size=8 --log-format json
```

On a cluster, through SLURM:

```bash
MELTEVAL_PROVIDER=qwen2_audio infra/runners/submit_eval.sh <site> \
  $HF_HOME/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/<sha> \
  /path/to/frozen-set-or-spec.yaml \
  -T task_filter=asr -T instruction='"Transcribe this audio."' -M batch_size=8 --log-format json
```

## What the model does for itself

**The chat template is the model's.** `instruction_prompt` renders the bare
instruction. The provider wraps it in the model card's conversation shape,
`[{"type": "audio"}, {"type": "text", "text": instruction}]`, and renders it
with `processor.apply_chat_template(..., add_generation_prompt=True)`. The
template adds its default system turn ("You are a helpful assistant.") and
`Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>` before the instruction. The
fully rendered prompt of the first sample is logged at INFO. The instruction
itself is in each sample's store (`melteval:prompt`, `melteval:format_spec`).

**The audio is expanded by the processor.** It replaces `<|AUDIO|>` with one
token per encoder frame.

## Prompt variants

The instruction takes the placeholders of `render_smurf_prompt`: `{lang}`
(the language name, from the training package's table), `{lang_code}` (the
ISO code) and `{task}`. The choice changes the result, so it is always
explicit:

- with language: `-T instruction='"Transcribe this audio in {lang}."'`
- without language: `-T instruction='"Transcribe this audio."'`

A frozen set whose samples carry their own `instruction` uses that one
instead. The outer double quotes matter: inspect parses `-T` values as YAML
(see docs/replication_notes.md).

## Things to know

- **30 s limit.** The processor pads every clip to 30 s and silently cuts off
  anything longer. By default the provider refuses such a batch rather than
  score a truncated transcription against the full reference.
  `-M truncate_long_audio=true` opts into the cut explicitly instead (what the
  model does out of the box): each cut sample gets
  `audio_truncated_from_seconds` in its output metadata, the run logs a warning
  with the count, and `projects/baselines/report.py` counts them per run. The
  baseline matrix uses it for Qwen2-Audio (LibriSpeech test-clean has 9 of 2620
  clips over 30 s) and skips MCIF `long`, whose clips are whole talks.
- **Left padding.** The provider sets `processor.tokenizer.padding_side = "left"` explicitly. `generate()` runs from `input_ids` here, so the output
  starts with the prompt, and left padding lets one slice strip it from every
  row. Compare runs only at equal `batch_size`.
- **Output is not post-edited.** Special tokens are dropped and whitespace
  stripped, nothing else. If the model wraps its answer in a preamble ("The
  original content of this audio is: '…'"), that is scored as is: that is
  `corpus_wer`, the headline. The ASR scorer also reports
  `corpus_wer_extracted` (the quoted text only, see
  `melteval.scorers.strip_transcript_preamble`) and `preamble_rate`, for every
  model alike. On LibriSpeech test-clean, Qwen2-Audio wraps 34.5 % of its
  answers even when told to output only the transcription: 12.1 % WER as it
  stands, 2.9 % extracted.
- **Decoding** is greedy by default, via the same `_generate_kwargs` as the MELT
  provider. `-M dtype=` (default `bfloat16`) and `-M attn_implementation=`
  (default `sdpa`) are configurable.

## Environment

Needs `transformers` with the `qwen2_audio` model type (≥ 4.45). The processor's
audio keyword is `audio=`, checked against 4.57.6; older releases called it
`audios=`. MELT's training package is not needed to *generate*. Without it:

- `{lang}` fails with a clear error: use `{lang_code}` or a literal
  instruction.
- The ASR scorer's normalizers are melteval's own copy of the training
  package's (`melteval/normalizers`, checked identical by
  `tests/test_normalizers.py`), so the default WER here is computed exactly as
  for a MELT run. Do not pass `-T normalizer=none` for a number meant to be
  compared with MELT's.

**Status:** the solver, prompt style, provider guard and batch logic are
unit-tested with a fake processor and model (`tests/test_provider.py`,
`tests/test_prompt_smurf.py`), and the solver has been run through a real
`inspect_ai.eval()` with `mockllm`. No run with the real weights yet.

## Qwen3-Omni-30B-A3B-Instruct

```bash
MELTEVAL_PROVIDER=qwen3_omni infra/runners/submit_eval.sh <site> \
  $HF_HOME/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/<sha> \
  configs/hf/air-bench.yaml \
  -T task_filter=audio_mcq -T dataset_id=air-bench-foundation-speech -M batch_size=4
```

Stage it like Qwen2-Audio (`huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct`), but check `df -h "$HF_HOME"` first: the
weights are **~70 GB**. At bf16 they fit one H200; on 80 GB cards pass
`-M device_map=auto` to shard across two GPUs.

What differs from Qwen2-Audio:

- **Text only.** The model is a thinker (the LLM) plus a talker that speaks the
  answer. The talker is dropped after loading and `return_audio=False` is
  passed; `generate()` then returns `(token_ids, None)`.
- **Decoding arguments carry a `thinker_` prefix.** Its `generate()` does not
  reject a plain `max_new_tokens` — it drops it in favour of its own
  `thinker_max_new_tokens=1024`, so `--max-tokens 256` would silently decode
  up to 1024. The provider prefixes every argument (tested in
  `tests/test_provider.py::TestQwen3Omni`).
- **No 30 s window.** The processor pads audio without truncating it, so long
  clips (MCIF `long`) go in whole. Size `batch_size` for the longest audio.
- Needs `transformers` with the `qwen3_omni_moe` model type (checked against
  4.57.6).
