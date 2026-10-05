# Baseline providers

Baselines are reference points next to MELT and SMURF: off-the-shelf models loaded through `transformers` (Qwen2-Audio, Qwen3-Omni, Whisper) and DeSTA2.5-Audio, which comes from its own package.

Each is a thin subclass of `melteval/providers/hf.py`, which owns what they share. A subclass says which classes to load, how long a clip the model takes in one piece, and how its `generate()` takes arguments and returns output. Whisper and DeSTA2.5 override the batch step.

| Model                         | `--model`             | Hub revision (pinned)                        | Longest clip         |
| ----------------------------- | ----------------------- | -------------------------------------------- | -------------------- |
| Qwen2-Audio-7B-Instruct       | `melt/hf/qwen2_audio` | `0a095220c30b7b31434169c3086508ef3ea5bf0a` | 30 s                 |
| Qwen3-Omni-30B-A3B-Instruct   | `melt/hf/qwen3_omni`  | `26291f793822fb6be9555850f06dfe95f2d7e695` | no fixed limit       |
| Whisper large-v3              | `melt/hf/whisper`     | `06f233fe06e710322aca913c1bc4249a0d71fce1` | no limit (long-form) |
| DeSTA2.5-Audio (Llama-3.1-8B) | `melt/hf/desta25`     | `1427f8725d35a0c9da0c7c54b1ffcc157a5948af` | 30 s                 |

All run with `-T prompt_style=hf` (the launcher and `run_eval.sbatch` set it). The revision is set on the class (`repo`, `revision` in `melteval/providers/qwen*.py`) and loaded from the HF cache, offline too once downloaded (`scripts/evaluate.py prefetch`, or `huggingface-cli download <repo> --revision <sha>`). `-M revision=<sha>` loads another commit; `-M path=<dir>` loads a local checkpoint of the same architecture.

**Who builds what.** The solver (`instruction_prompt`) hands over the bare instruction; the provider adds the chat template and the audio token.

**Never write the audio token in an instruction.** The chat template adds its own (`<|AUDIO|>` for Qwen2-Audio, `<|audio_pad|>` for Qwen3-Omni). An instruction is plain text, e.g. `"Transcribe this audio."`.

Some benchmarks (AIR-Bench, MCIF) ship a prompt with an `{audio_token}` marker, e.g. `"{audio_token}\nWhat is the emotion?"`. For these models the marker is removed (`melteval.prompt.strip_audio_token`), since the template already adds the audio.

## Qwen2-Audio-7B-Instruct

```bash
inspect eval melteval/tasks.py@asr \
  --model melt/hf/qwen2_audio \
  -T frozen_set=/path/to/frozen-set \
  -T prompt_style=hf \
  -T instruction='"Transcribe this audio."' \
  -M batch_size=8 --log-format json
```

On a cluster, one job through SLURM (many at once: [running-evaluations.md](running-evaluations.md)):

```bash
infra/runners/submit_eval.sh <site> melt/hf/qwen2_audio /path/to/frozen-set-or-spec.yaml \
  -T task_filter=asr -T instruction='"Transcribe this audio."' -M batch_size=8 --log-format json
```

### What the model does for itself

**The chat template is the model's.** The provider wraps the bare instruction in the model card's conversation shape, `[{"type": "audio"}, {"type": "text", "text": instruction}]`, and renders it with `processor.apply_chat_template(..., add_generation_prompt=True)`. Qwen2-Audio's template adds its default system turn ("You are a helpful assistant.") and `Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>` before the instruction. The first sample's rendered prompt is logged at INFO; the instruction is in each sample's store (`melteval:prompt`, `melteval:format_spec`).

**The audio is expanded by the processor**, which replaces `<|AUDIO|>` with one token per encoder frame.

### Prompt variants

The instruction takes the placeholders of `render_smurf_prompt`: `{lang}` (language name, from the training package's table), `{lang_code}` (ISO code) and `{task}`. The choice changes the result, so it is always explicit:

- with language: `-T instruction='"Transcribe this audio in {lang}."'`
- without language: `-T instruction='"Transcribe this audio."'`

A frozen set whose samples carry their own `instruction` uses that one instead. The outer double quotes matter: inspect parses `-T` values as YAML (see docs/replication_notes.md).

## Qwen3-Omni-30B-A3B-Instruct

```bash
infra/runners/submit_eval.sh <site> melt/hf/qwen3_omni configs/hf/air-bench.yaml \
  -T task_filter=audio_mcq -T dataset_id=air-bench-foundation-speech -M batch_size=4
```

### What differs from Qwen2-Audio:

- **A different rendered prompt.** No default system turn and no `Audio 1:` prefix; the audio goes in as `<|audio_start|><|audio_pad|><|audio_end|>` right before the instruction, and the processor expands `<|audio_pad|>`. For "Transcribe this audio." the template renders:
  `<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>Transcribe this audio.<|im_end|>\n<|im_start|>assistant\n`
- **Text only.** The model is a thinker (the LLM) plus a talker that speaks the answer. The talker is dropped after loading and `return_audio=False` is passed; `generate()` then returns `(token_ids, None)`.
- **Decoding arguments carry a `thinker_` prefix.** Its `generate()` does not reject a plain `max_new_tokens`; it drops it for its own `thinker_max_new_tokens=1024`, so `--max-tokens 256` would silently decode up to 1024. The provider prefixes every argument (tested in `tests/test_provider.py::TestQwen3Omni`).
- **No 30 s window, but only because the provider asks.** The encoder takes any length, but the feature extractor still truncates to 30 s by default (`truncation=True`), without a warning. The provider passes `truncation=False`, so long clips (MCIF `long`) go in whole. Size `batch_size` for the longest audio.
- Needs `transformers` with the `qwen3_omni_moe` model type (checked against 4.57.6).

## Whisper large-v3

```bash
inspect eval melteval/tasks.py@asr --model melt/hf/whisper -T frozen_set=... \
  -T prompt_style=hf -T instruction='"Transcribe this audio."' -M batch_size=8
```

Not a chat model: no template, and the **instruction is ignored** (it is passed only because the launcher requires one per task for benchmarks with none, `needs_instruction`). Model arguments set what it does: `-M task=transcribe` (default) or `-M task=translate` (into English). The baseline matrix has one entry per direction.

- **Language is auto-detected.** A provider never sees a sample's metadata, so a short clip can get the wrong language.
- **Long audio** is decoded long-form (sequential 30 s windows, `return_timestamps=True`), not refused.
- Only ASR and X->en ST apply; the matrix skips AIR-Bench and MCIF for it.

## DeSTA2.5-Audio

```bash
inspect eval melteval/tasks.py@asr --model melt/hf/desta25 -T frozen_set=... \
  -T prompt_style=hf -T instruction='"Transcribe this audio."' -M batch_size=8
```

The Hub repo holds only the connector weights; the model class is `DeSTA25AudioModel` in the `desta` package ([github.com/kehanlu/DeSTA2.5-Audio](https://github.com/kehanlu/DeSTA2.5-Audio)), which also loads a gated Llama-3.1-8B-Instruct backbone and a Whisper encoder. It pins its own `transformers`/`torch`, so it needs **its own venv**; the matrix entry (`venv: ${DESTA_VENV_PATH}`) is skipped when that venv does not exist.

**Install.** Accept the [Llama-3.1 licence](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), then:

```bash
uv venv --python 3.12 ~/repos/eval/.venv-desta
source ~/repos/eval/.venv-desta/bin/activate
VIRTUAL_ENV=$PWD/.venv-desta uv pip install -e ".[hf,metrics]"
git clone https://github.com/kehanlu/DeSTA2.5-Audio.git ~/repos/DeSTA2.5-Audio
VIRTUAL_ENV=$PWD/.venv-desta uv pip install -e ~/repos/DeSTA2.5-Audio
huggingface-cli login   # token goes to $HF_HOME/token, which jobs read
# Silero-VAD: torch.hub asks to trust the repo and a job cannot answer, so cache it once
python -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)"
```

Leave NumPy at 2.x; `Desta25API` restores the `np.sctypes` that `desta` needs. Silero-VAD is cached in `~/.cache/torch/hub`, which `scripts/evaluate.py prefetch` does not cover. `DESTA_VENV_PATH` defaults to `~/repos/eval/.venv-desta/bin/activate` in `infra/sites/bocconi.sh`.

### Things to now

- **System prompt.** Results are wrong without one: the model is an instruction-tuned LLM and otherwise answers like a chat assistant. The launcher passes the model entry's `system:` map, one prompt per task and `default` for the rest, as `-M system_prompt=...`; outside the launcher `Desta25API` sends the authors' default. Prompts follow the [README](https://github.com/kehanlu/DeSTA2.5-Audio) and `docs/evaluation_tips.md`; only the ASR and ST user instructions are ours, and so is applying the MCQ system prompt to AIR-Bench. ST and `audio_chat` use the default.
- **30 s window.** As for Qwen2-Audio, a clip over 30 s fails that sample alone (`AudioTooLongError`, `--fail-on-error` applies) and MCIF `long` is skipped.
