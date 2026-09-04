# Evaluating SMURF checkpoints

SMURF is the parallel research line trained with
[NVIDIA NeMo Speech](https://github.com/NVIDIA-NeMo/Speech) rather than with
MELT. Its model is `fbk_speechllm.models.speech_llm.SpeechLLM`, a thin subclass
of NeMo's `SALM`, and its checkpoints are Lightning `.ckpt` files.

They are evaluated through the same frozen sets, tasks and scorers as MELT
checkpoints. Only two things differ, and both are contained: the provider
(`--model smurf/<ckpt>`) and the prompt solver (`-T prompt_style=smurf`).

```bash
inspect eval melteval/tasks.py@asr \
  --model smurf/path/to/epoch=0-step=3600.ckpt \
  -T frozen_set=runs/asr-test-v1 \
  -T prompt_style=smurf \
  -T instruction="Transcribe this English audio: " \
  -M batch_size=4
```

On a cluster, the runner takes it from an environment variable, because the
checkpoint and the venv both change with the family:

```bash
MELTEVAL_PROVIDER=smurf VENV_PATH=/path/to/venvs/smurf-eval/bin/activate \
  infra/runners/submit_eval.sh artemis \
  /path/to/epoch=0-step=3600.ckpt /path/to/frozen-set \
  -T instruction="Transcribe this English audio: "
```

`-T prompt_style=smurf` is added automatically there, since the provider and
the prompt format have to agree.

## Why not `fbk_speechllm.inference`

Upstream's documented inference path builds a Lhotse `DataModule` from a NeMo
data config and iterates its test dataloader. Reusing it would hand the sample
list back to the model's data config — the one thing a frozen set exists to
take away. Which samples a number is computed over has to be decided once, at
freeze time, and be identical across checkpoints and across families.

So the provider uses SALM's own lower-level generation API instead, which
upstream documents alongside the dataloader one:

```python
model.generate(
    prompts=[[{"role": "user", "content": f"Transcribe this: {model.audio_locator_tag}"}]],
    audios=audios,        # float32 (B, T), zero-padded
    audio_lens=audio_lens, # int64 (B,)
    generation_config=GenerationConfig(...),
)
```

Everything the dataloader path does that matters for generation happens on this
path too, inside `generate()`: the prompt formatter, the placeholder expansion,
the left padding.

## What the model does for itself

This is the part worth reading before changing anything.

**The chat template is the model's.** Given turns rather than token ids, SALM
formats them with the `PromptFormatter` named by its own `cfg.prompt_format`
(`qwen` for the current runs). So `smurf_prompt` renders the *bare
instruction* and stops. Running a SMURF checkpoint under `speech_prompt`
instead would apply MELT's chat template first and NeMo's on top of it —
`melteval.solver._require_provider` raises rather than let that happen.

**The audio placeholder is the model's.** SALM splices encoder frames into the
positions of `cfg.audio_locator_tag` (`<|audioplaceholder|>`), so the provider
reads the tag off the loaded checkpoint and appends it to the prompt. It cannot
be configured here, and it cannot drift from the weights it belongs to. A
prompt that already contains the tag is left untouched, which is how a
benchmark shipping its own prompt keeps control of where the audio goes.

**The instruction is not the model's.** In SMURF training it is the
conversation's `context` tag, baked into the Lhotse manifests at data-prep
time (`fbk_speechllm.data_preprocessing.prepare_data_for_training`, drawing
from the pools in `constants/asr_prompts.py`), and at inference time it comes
from the data config:

```yaml
input_cfg:
  - type: lhotse_as_conversation
    audio_locator_tag: "<|audioplaceholder|>"
    tags:
      context: "Transcribe this English audio: "
```

`-T prompt_config=<that yaml>` reads it from there; `-T instruction=...` states
it directly. With neither, and no per-sample `instruction` in the frozen set,
the run fails. That is deliberate: an empty user turn is a valid input to a
speech LLM, and it produces fluent text and a plausible score.

## Environment

The two stacks do not co-install — SMURF pins `nemo_toolkit>=2.7`,
`megatron-core==0.17.0` and Python 3.12 against its own torch build, MELT pins
a different one — so a SMURF eval needs its own venv:

```bash
uv venv --python 3.12 /path/to/venvs/smurf-eval
VIRTUAL_ENV=/path/to/venvs/smurf-eval uv pip install -e /path/to/speechllm
VIRTUAL_ENV=/path/to/venvs/smurf-eval uv pip install -e "/path/to/melt-eval[metrics]"
```

Note `melt-eval` without the `shar` extra: that extra pulls in `melt-proj`,
which is MELT's training package and has no business in this environment.
Frozen sets that reference Shar sources still need `lhotse` and `soundfile`,
which the SMURF stack already installs.

Nothing in `melteval` imports torch, NeMo or transformers at module scope, so
this half-populated environment imports the package (and registers both
providers) without the MELT side present.

The `{lang}` placeholder in an instruction is the one thing that *does* reach
for MELT's package — it resolves ISO codes to the same language names MELT
prompts use. In a SMURF-only environment use `{lang_code}`, or write the
language into the instruction as SMURF's own configs do.

## Cross-checking against upstream

Same idea as `configs/crosscheck.yaml` does for MELT (see the commit message on
that file): run the same checkpoint over the same samples through both tools
and compare, so the harness's own generation path is not quietly wrong.

1. Build a frozen set from the same cuts the SMURF inference config points at.
2. Run `python -m fbk_speechllm.inference --config-path … --config-name …`
   with `batch_size: 1`, `max_new_tokens` matching, `do_sample: False`.
3. Run `inspect eval` as above with `-M batch_size=1` and the same
   `instruction`, and `--max-tokens` set to the same value.
4. Compare the hypotheses, not just the WER: `melteval rescore` is not needed,
   the eval log holds the completions.

Expect small differences from batching and dtype, the same two sources the MELT
cross-check documented. Systematic differences — a truncated hypothesis, a
prompt echoed back, an off-by-one language — mean the prompt is not what the
model was trained on, and the instruction is the first thing to check.

**Status:** not yet run. The provider's message handling, audio collation,
config translation and prompt rendering are unit-tested (`tests/test_provider.py`,
`tests/test_prompt_smurf.py`), and the solver is covered through a real
`inspect_ai.eval()`, but no SMURF checkpoint has been generated from through
this path yet.
