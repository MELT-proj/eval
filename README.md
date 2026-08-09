# melt-eval

Downstream evaluation for MELT speech models, built as an extension to
[`inspect_ai`](https://github.com/UKGovernmentBEIS/inspect_ai) — not a fork.
Providers, datasets, solvers and scorers register through inspect's own
extension points, so the framework can be upgraded without a rebase.

It exists because everything the training repo measures is a loss or a
teacher-forced proxy. This measures what the model actually generates.

## The shape of a run

```
melteval freeze  ──►  frozen set  ──►  inspect eval  ──►  melteval rescore
  (corpora)          (manifest)        (generation)        (neural metrics)
```

1. **Freeze** resolves a source spec into a fixed list of samples: which audio,
   which reference, which task and language. Runs once; two runs of the same
   spec evaluate exactly the same samples.
2. **Evaluate** with `inspect eval`, which generates and scores.
3. **Rescore** adds COMET/MetricX from a separate environment, so a GPU metric
   stack never has to install next to the model.

Frozen sets **reference** audio rather than copying it — see
[docs/frozen-sets.md](docs/frozen-sets.md).

## Install

melt-eval needs both `inspect_ai` and the training package (`melt-proj`) —
`lhotse`, `torch`, `transformers` — in one environment, which no site's
existing venv or container image happens to provide together. Build a
dedicated venv **outside your home directory** (see "A note on artemis" below)
with the training repo checked out as a sibling:

```bash
# expects ../training to exist (sibling checkout); see [tool.uv.sources] in
# pyproject.toml if you keep it somewhere else
uv venv --python 3.12 /path/to/venvs/melteval
VIRTUAL_ENV=/path/to/venvs/melteval uv pip install --prerelease=allow -e ".[shar,metrics,dev]"
```

`--prerelease=allow` is needed because `melt-proj` pins a pre-release lhotse
(`2.0.0a3`). Extras are split by what you actually need: `shar` (local lhotse
corpora, pulls in the training package), `hf` (remote HuggingFace test
splits), `metrics` (jiwer, sacrebleu).

## Usage

```bash
# Build a frozen set (CPU-only -- safe to run directly, no SLURM needed)
LOCAL_DATASETS_DIR=/path/to/shar melteval freeze configs/smoke.yaml -o runs/smoke

# Look at what it contains
melteval show runs/smoke

# Try it against a mock model, no GPU or checkpoint required
inspect eval melteval/tasks.py@speech --model mockllm/model \
  -T frozen_set=runs/smoke -T format_config=/path/to/a/training_config.yaml
```

## Running an evaluation on a cluster

`infra/runners/submit_eval.sh <site> <checkpoint_dir> <frozen_set_dir> [inspect eval args...]`
is the whole interface: swap the checkpoint, swap the frozen set, run. The
site (`infra/sites/<site>.sh`) supplies the venv, output directory and SLURM
resources.

```bash
infra/runners/submit_eval.sh artemis \
  /path/to/outputs/MA-v1.2.7 \
  /path/to/eval-sets/asr-test-v1 \
  -T task_filter=asr -M batch_size=16
```

Common overrides, all `-T` (task parameter) or `-M` (model parameter) flags
appended after the two required paths:

| Want to... | Add |
|---|---|
| Restrict to one task | `-T task_filter=asr` (or `st`) |
| Restrict to one language | `-T lang=de` |
| Restrict to one corpus | `-T dataset_id=fleurs` |
| Cap the sample count | `-T limit=50` |
| Bigger/smaller batches | `-M batch_size=32` |
| Format from a different config than the checkpoint | `-T format_config=/path/to/training_config.yaml` |

**ST specifically:** a frozen set covering more than one target language needs
one run per language (`-T lang=de`, then a separate run with `-T lang=ar`,
...). BLEU's tokenizer is chosen per target language, so a corpus mixing
languages has no single correct tokenizer for the mix — `corpus_bleu` raises
rather than silently pick one.

A debug run before a full one — few samples, short QoS:

```bash
MELT_QOS=gpu-debug MELT_TIME=00:15:00 infra/runners/submit_eval.sh artemis \
  /path/to/checkpoint /path/to/frozen-set -T limit=5
```

`melteval freeze` itself is CPU-only and safe to run directly (no `sbatch`
needed) — only the generation step touches a GPU.

**Available sites:** `artemis` (a6000/h100/h200, working). `mn5` is scaffolded
but its venv has not been built yet — MN5 has no outbound internet, so that
venv has to be assembled elsewhere and copied over first; see the comment at
the top of `infra/sites/mn5.sh`. Copy `infra/sites/example.sh` to add a new
site.

## Rescoring ST output with COMET/MetricX

`inspect eval` reports BLEU/chrF; a neural MT metric is a separate step, run
from the `comet`/`metricx` venv rather than this one, so a GPU metric stack
never has to install next to the model under evaluation:

```bash
# in the melteval venv: extract (src, mt, ref) triples from a finished ST run
melteval rescore path/to/log.eval -o triples.jsonl
```

```python
# then, from the comet venv:
import json
from comet import download_model, load_from_checkpoint

samples = [json.loads(line) for line in open("triples.jsonl")]
model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
print(model.predict(samples, batch_size=8, gpus=1).system_score)
```

Only samples whose frozen-set source set `source_text_field` at freeze time
produce a triple — that field is the `src` a reference-based MT metric needs,
and it doesn't exist unless a source config asked for it (see
[docs/frozen-sets.md](docs/frozen-sets.md)).

## Development

```bash
pytest tests -q                                     # unit tests only
MELTEVAL_SHAR_ROOT=/path/to/shar pytest tests -q    # adds corpus integration tests
ruff check melteval tests
```

### A note on artemis

- **GPU work always goes through SLURM.** Never run `inspect eval`, or
  anything else that loads a model, directly on the workstation — even for a
  quick check. Use `infra/runners/submit_eval.sh`. Freezing, unit tests and
  linting are CPU-only and fine to run directly.
- **Nothing heavy goes under `/mnt/home/giuseppe`.** It's GlusterFS-synced and
  quota-limited. Venvs, eval outputs and frozen sets belong under
  `/mnt/scratch-artemis/giuseppe` or `/mnt/data-artemis/giuseppe`.

See `AGENTS.md` for the full detail behind both rules.
