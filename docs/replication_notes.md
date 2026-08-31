 

# SMURF in melt-eval integration

# 1. Does the new SMURF eval provider work?

## 1.1 Create the venv (SMURF stack + melt-eval)

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ../smurf/speechllm --override smurf_overrides.txt
uv pip install -e ".[hf,metrics]"
```

## 1.2 Unit tests

```bash
pytest tests -q
```

## 1.3 Build a frozen set

### Config template (`hf_dataset`)

```yaml
name: <frozen_set_name>
seed: 0

input_cfg:
  - type: hf_dataset
    repo: <user/dataset>          # HF repo
    revision: <commit-sha-or-tag> # pinning
    split: test                   # optional, default "test"
    # name: <dataset-config>      # optional: HF's "config" argument (some datasets have several)
    audio_column: audio           # optional, default "audio"
    text_column: text             # optional, default "text"
    # id_column: id               # optional: column to use as the sample ID
    # source_text_column: ...     # optional: ST only, the source text (for COMET later)
    # min_duration: 1.0           # optional: duration filter, seconds
    # max_duration: 30.0
    # max_samples: 50             # optional: cap the sample count (random subsample with `seed`)
    tags:
      task: asr                   # asr | st | ...
      lang: en                    # expected output language
      # src_lang: en              # source/target-language tasks only (e.g. st)
      # tgt_lang: de
      dataset_id: <label>         # for per-corpus breakdowns in the summary
```

### LibriSpeech

Config for testing in [configs/librispeech-hf-smoke.yaml](../configs/librispeech-hf-smoke.yaml) (`hf-internal-testing/librispeech_asr_dummy`, `clean/validation`, ~73 samples).

```bash
melteval freeze configs/librispeech-hf-smoke.yaml -o runs/librispeech-hf-smoke
melteval show runs/librispeech-hf-smoke
```

## 1.4 Evaluate

```bash
TMPDIR=~/.cache/triton-eval TRITON_CACHE_DIR=~/.cache/triton-eval \
inspect eval melteval/tasks.py@asr \
  --model smurf/data/checkpoints/step=2784-last.ckpt \
  -T frozen_set="$(pwd)/runs/librispeech-hf-smoke" \
  -T prompt_style=smurf \
  -T instruction='"Transcribe this English audio: "' \
  -T normalizer=none \
  -M batch_size=4 -M device_map=auto
```

`-M device_map=auto` shards the checkpoint across every visible GPU; this is needed if it does not fit on one.

---

# 2. Cross-check evaluation

Here we run the same checkpoint over the same samples, in the same order, with the same decoding settings, through both tools, and compare the transcriptions. Both use the same model (`SpeechLLM`), tokenizer and `qwen` prompt formatter, but differ in how the prompt reaches it:

* **melteval** hands `generate()` one user turn (`instruction` + `<|audioplaceholder|>`) and lets the formatter run inside.
* **upstream** runs the formatter up front and hands `generate()` the token ids.

Breakdown of differences:

| Setting                                                                       | upstream (`asr_inference.yaml`)                   | melteval side                                                 | agrees by default?                                                     |
| ----------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------- |
| instruction (`context`)                                                     | `tags.context: "Transcribe this English audio: "` | `-T instruction=...` (same string, trailing space included) | **no:** the wrapper set it                                      |
| `max_new_tokens`                                                            | `128`                                             | `--max-tokens 128`                                          | **no:** the provider default is 256                             |
| `batch_size`                                                                | `1` (`data.test_ds.batch_size=1`)               | `-M batch_size=1`                                           | **no:** with batching, sequences are left-padded to the longest |
| in the batch, and padding can nudge the result. With`batch_size=1` there is |                                                     |                                                               |                                                                        |
| no padding, so any remaining difference is signal, not noise.                 |                                                     |                                                               |                                                                        |
| `num_beams`                                                                 | `1` (greedy)                                      | default`1`                                                  | yes                                                                    |
| `do_sample`                                                                 | `False` (deterministic)                           | no`temperature` ⇒ `False`                                | yes                                                                    |
| `use_model_defaults`  (tells HF `generate()` *not* to pull in          |                                                     |                                                               |                                                                        |
| whatever decoding params the base LLM ships with, and use exactly what the    |                                                     |                                                               |                                                                        |
| config says)                                                                  | `False`                                           | default`False`                                              | yes                                                                    |
| `dtype`                                                                     | `bfloat16`                                        | default`bfloat16`                                           | yes                                                                    |
| bos/eos/pad token ids                                                         | `model.text_{bos,eos,pad}_id`                     | identical (`_generation_config`)                            | yes                                                                    |

If both routes tokenize to the same thing, the transcriptions come out the same. So the transcriptions are the check:

* **The two transcript sets come out almost the same.** A word off here and thereis fine, as long as the differences have no pattern. That small amount of noise is expected: the two runs group the audio into batches differently and round numbers a bit differently, which can flip the odd word. **It is not caused by the prompt.**
* **They differ, and the differences repeat.** For example: every long transcript is cut off at the same point, the instruction text shows up inside the model's answer, or the output is in the wrong language. A repeating pattern is not noise; this means the model got a different prompt than upstream gives it.

## 2.2  Run it

```bash
# SLURM: everything in one ~80 GB GPU job.
MODEL_CKPT=data/checkpoints/step=2784-last.ckpt \
VENV=$HOME/venvs/smurf-eval/bin/activate \
  sbatch infra/crosscheck.sbatch

# local: steps 2.4 and 2.5 on the current GPU, if the checkpoint fits
scripts/run_crosscheck.sh data/checkpoints/step=2784-last.ckpt
```

`infra/crosscheck.sbatch` activates the venv and runs `scripts/run_crosscheck.sh` inside the allocation.

Knobs (environment variables, forwarded into the job):

| Var                         | Default                                        | Purpose                                                 |
| --------------------------- | ---------------------------------------------- | ------------------------------------------------------- |
| `MODEL_CKPT`              | — (required)                                  | path to the`.ckpt`                                    |
| `VENV`                    | — (required)                                  | the SMURF venv's`bin/activate`                        |
| `CONFIG`                  | `configs/librispeech-hf-smoke.yaml`          | frozen-set config                                       |
| `INSTRUCTION`             | `"Transcribe this English audio: "`          | prompt (must match upstream)                            |
| `MAX_TOKENS`              | `128`                                        | generation cap on both sides                            |
| `MELTEVAL_DEVICE_MAP`     | (one GPU)                                      | `auto` to shard the melteval side                     |
| `SMURF_GEN_DIR`           | `../smurf/speechllm/working_config/generate` | dir of the upstream config                              |
| `WORK`                    | `runs`                                       | output root                                             |
| `FORCE=1`                 | —                                             | redo steps already finished                             |
| `FROM_STEP` / `TO_STEP` | `2` / `5`                                  | run a sub-range (e.g.`FROM_STEP=5` = re-compare only) |
| `SKIP_UPSTREAM=1`         | —                                             | skip 2.4 and 2.6 (no`fbk_speechllm`)                  |

## 2.3.1 Prepare the data (CPU)

```bash
# melteval side: the frozen set (reuse the one from 1.3)
melteval freeze configs/librispeech-hf-smoke.yaml -o runs/librispeech-hf-smoke
melteval show runs/librispeech-hf-smoke                 # confirm ~73 samples

# upstream side: the SAME HF split -> lhotse CutSet, same order and same ids
python scripts/hf_to_lhotse.py \
  --config configs/librispeech-hf-smoke.yaml \
  -o runs/crosscheck-smurf-cuts
# -> runs/crosscheck-smurf-cuts/cuts.jsonl.gz  (+ audio/*.wav at 16 kHz)
```

We use an HF dataset because the SMURF venv has no `melt-proj`, which melteval's shar reader needs. `scripts/hf_to_lhotse.py` converts it into the lhotse file `fbk_speechllm.inference` needs, keeping the samples in the same order and with the same ids melteval uses, so the two result files line up sample by sample.

## 2.3.2 `fbk_speechllm.inference` (GPU)

`asr_inference.yaml` already carries `context="Transcribe this English audio: "`, `max_new_tokens=128`, `num_beams=1`, `do_sample=False`,`use_model_defaults=False`.

**SLURM (h100/h200, 80 GB):**

```bash
MELT_PARTITION=h100 \
VENV_PATH=/path/venvs/smurf-eval/bin/activate \
MODEL_PATH=$(pwd)/data/checkpoints/step=2784-last.ckpt \
DATA_PATH=$(pwd)/runs/crosscheck-smurf-cuts/cuts.jsonl.gz \
OUTPUT_PATH=$(pwd)/runs/crosscheck-smurf/upstream.jsonl \
sbatch --wait --nodes=1 --gpus-per-node=1 --partition=h100 --qos=<qos> \
  infra/crosscheck_upstream.sbatch \
  $(pwd)/../smurf/speechllm/working_config/generate asr_inference \
  generation.max_new_tokens=128
```

**Local (only if the checkpoint fits on the GPU):**

```bash
MODEL_PATH=$(pwd)/data/checkpoints/step=2784-last.ckpt \
DATA_PATH=$(pwd)/runs/crosscheck-smurf-cuts/cuts.jsonl.gz \
OUTPUT_PATH=$(pwd)/runs/crosscheck-smurf/upstream.jsonl \
python -m fbk_speechllm.inference \
  --config-path $(pwd)/../smurf/speechllm/working_config/generate \
  --config-name asr_inference \
  data.test_ds.batch_size=1 generation.max_new_tokens=128
```

Output: `runs/crosscheck-smurf/upstream.jsonl`, one `{id, hypothesis, num_tokens}` line per sample.

## 2.3.3 `inspect eval` via melteval (GPU)

**SLURM:** reuses `infra/run_eval.sbatch` (it appends `-T prompt_style=smurf` itself).

```bash
MELT_PARTITION=h100 MELTEVAL_PROVIDER=smurf \
VENV_PATH=/path/venvs/smurf-eval/bin/activate \
OUTPUT_DIR=$(pwd)/runs/crosscheck-smurf/eval-logs \
sbatch --wait --nodes=1 --gpus-per-node=1 --partition=h100 --qos=<qos> \
  infra/run_eval.sbatch \
  $(pwd)/data/checkpoints/step=2784-last.ckpt $(pwd)/runs/librispeech-hf-smoke \
  -T task_filter=asr -T instruction="Transcribe this English audio: " \
  -T normalizer=none --max-tokens 128 -M batch_size=1
```

**Local:**

```bash
TMPDIR=~/.cache/triton-eval TRITON_CACHE_DIR=~/.cache/triton-eval \
inspect eval melteval/tasks.py@asr \
  --model smurf/$(pwd)/data/checkpoints/step=2784-last.ckpt \
  -T frozen_set="$(pwd)/runs/librispeech-hf-smoke" \
  -T prompt_style=smurf \
  -T instruction='"Transcribe this English audio: "' \
  -T normalizer=none \
  --max-tokens 128 \
  -M batch_size=1
# add -M device_map=auto ONLY if the checkpoint does not fit on one GPU
```

Output: a `.eval` in `runs/crosscheck-smurf/eval-logs/` (or `logs/` locally).

## 2.4 Compare (CPU)

```bash
python scripts/compare_crosscheck.py \
  --eval-log runs/crosscheck-smurf/eval-logs/<the .eval from step C> \
  --upstream runs/crosscheck-smurf/upstream.jsonl
```

Joins the two outputs (by `cut_id`/`id`, with a positional fallback) and reports:

- exact matches;
- a WER/CER between the two transcription sets, computed as if upstream's were the ground truth, i.e., the number is how far melteval's text has drifted from upstream's;
- and systematic-pattern flags (one hypothesis a prefix of the other; the instruction present in the answer; length off by >50 %). Shows the most divergent pairs with a colour word-level diff.

Then it reports

* **PASS** (exit 0): no systematic flags and hyp-vs-hyp WER ≤ 2 %.
* **FAIL** (exit 1): a flag, or a higher WER. Inspect the pairs and fix the prompt assembly in `melteval/providers/smurf.py` (e.g. a separator between instruction and placeholder, or emitting two turns).
