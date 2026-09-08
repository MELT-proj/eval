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
   spec evaluate exactly the same samples. Skippable for a published
   HuggingFace benchmark, which is already immutable — see "Evaluating a
   HuggingFace benchmark" below.
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
LOCAL_DATASETS_DIR=/path/to/shar melteval freeze configs/shar/smoke.yaml -o runs/smoke

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
| Name the judge for a graded task | `-T grader_model=openai/gpt-4o` |

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

## Evaluating a HuggingFace benchmark

A published benchmark pinned to a revision is already immutable and already
has a canonical sample set, so there is nothing for a freeze pass to decide.
Pass the **spec** where a frozen set would go and the readers run at eval
time instead:

```bash
infra/runners/submit_eval.sh artemis \
  /path/to/checkpoint configs/hf/air-bench.yaml \
  -T task_filter=audio_mcq -T dataset_id=air-bench-foundation-speech
```

Same argument slot, same everything else: a directory is read as a frozen
set, a `.yaml` file as a spec. Sample keys come out identical either way, so
the same spec can still be frozen later without invalidating a run made this
way. **Local Shar mixtures should still be frozen** — there, "which samples?"
is a real decision and nothing else records it.

Always pass `-T dataset_id=` for a spec covering several splits. It selects
records, and it also stops the other splits being *resolved at all*, which
for AIR-Bench is the difference between materialising one split and all seven.

`configs/hf/air-bench.yaml` covers
[AIR-Bench](https://huggingface.co/datasets/pbcong/AIR-bench), both
configurations, all seven splits. It brings in two task types:

| Task | Splits | Scored by |
|---|---|---|
| `audio_mcq` | Foundation speech/sound/music | accuracy over the options the sample carries, plus the share of completions no option could be read out of |
| `audio_chat` | Chat speech/sound/music/mixed | a judge model — `-T grader_model=<provider/model>` |

Each sample brings its own question, so nothing is drawn from the training
prompt pool; the source's `instruction_template` decides the layout.

There is deliberately no lexical fallback for `audio_chat`: its references are
free text that a correct answer need not share any words with, so BLEU or an
exact match would rank a fluent wrong answer above a terse right one. On a
cluster with no route to a judge, generate now and grade later:

```bash
# on the cluster
infra/runners/submit_eval.sh artemis /path/to/checkpoint configs/hf/air-bench.yaml \
  -T task_filter=audio_chat -T dataset_id=air-bench-chat-speech --no-score
# later, from somewhere that can reach a judge
inspect score path/to/log.eval --scorer melteval/scorers.py@chat_scorer \
  -S grader_model=openai/gpt-4o
```

**Cache the splits first.** Compute nodes run with `HF_HUB_OFFLINE=1` (see
`infra/sites/artemis.sh`), so anything a run touches has to be in `HF_HOME`
before the job starts. Do it from the login shell, which is CPU-only work:

```bash
HF_HOME=/mnt/scratch-artemis/giuseppe/melt-data/hf_cache HF_HUB_OFFLINE=0 \
  python -c "from datasets import load_dataset; load_dataset('pbcong/AIR-bench', name='Foundation', split='speech', revision='ae924f11e8658e263fe8521c55664970552e15ca')"
```

AIR-Bench in full is about 12 GiB of parquet, plus as much again once
`datasets` has built its arrow cache. Check `df` first.

`configs/hf/mcif.yaml` covers
[MCIF](https://huggingface.co/datasets/FBK-MT/MCIF) (paper:
[arXiv:2507.19634](https://arxiv.org/abs/2507.19634)), both tracks, both
prompt styles, all four target languages. Its reference is not a column on
the sample's own row -- it lives in a separate file, keyed by an id that can
group several samples into one reference -- so this one brings in a dedicated
reader (`melteval/readers/mcif.py`) and two grouped scorers instead of reusing
`asr`/`st` directly:

| Task | Where | Scored by |
|---|---|---|
| `chunked_asr` | ASR, target language `en` | WER/CER, computed once per group after joining that group's completions in order -- see `melteval.scorers.chunked_asr_scorer` |
| `chunked_st` | translation, the other three target languages | BLEU/chrF, same grouping. The paper's own metric is COMET after a resegmentation step this harness does not perform |
| `audio_chat` | QA (every language) and summarisation (`long` track only) | a judge model, same as AIR-Bench Chat -- `-T grader_model=<provider/model>` |

In the `short` track a `chunked_asr`/`chunked_st` reference is a whole talk's
transcript and spans dozens of samples, each a short segment the model
transcribes on its own. In the `long` track every group has exactly one
member -- each sample already *is* a whole talk (several minutes), so that
track is a genuine single-shot long-form test rather than a chunked one; size
batches and wall time accordingly (see the comment in `configs/hf/mcif.yaml`).

**Caching MCIF is a different shape than AIR-Bench's.** Its audio is a plain
file path in the repo, not embedded in the parquet, so `load_dataset` alone
does not prime it -- read the samples you intend to run and resolve their
audio explicitly:

```bash
HF_HOME=/mnt/scratch-artemis/giuseppe/melt-data/hf_cache HF_HUB_OFFLINE=0 python -c "
from melteval.dataset import spec_dataset
from melteval.readers.base import reader_for_locator
from melteval.manifest import AudioLocator

ds = spec_dataset('configs/hf/mcif.yaml', dataset_id='mcif-short-fixed-en', task='chunked_asr')
for s in ds.samples:
    loc = AudioLocator.from_dict(s.metadata['audio'])
    reader_for_locator(loc).load_audio(loc)
"
```

**Available sites:** `artemis` (a6000/h100/h200, working). `mn5` is scaffolded
but its venv has not been built yet — MN5 has no outbound internet, so that
venv has to be assembled elsewhere and copied over first; see the comment at
the top of `infra/sites/mn5.sh`. Copy `infra/sites/example.sh` to add a new
site.

Recap of recent jobs on a site — QoS, state, resources used:

```bash
infra/job_recap.sh artemis 20   # or: infra/job_recap.sh mn5 20
```

## Rescoring ST output with COMET/MetricX

`inspect eval` reports BLEU/chrF; a neural MT metric is a separate step, run
from the `comet`/`metricx` venv rather than this one, so a GPU metric stack
never has to install next to the model under evaluation. `infra/Singularity.def`
bakes this in as a second venv (`unbabel-comet` pins `transformers<5`, which
cannot share an environment with melt-proj's `transformers>=5.16`) — activate
it with `source $COMET_VENV_PATH` inside the container.

```bash
# in the melteval venv: extract (src, mt, ref) triples from a finished ST run
melteval rescore path/to/log.eval -o triples.jsonl
```

```python
# then, from the comet venv ($COMET_VENV_PATH in the container image):
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

## Reproducing MCIF's official metrics (WER/COMET/BERTScore)

`chunked_asr`/`chunked_st`/`audio_chat` are fast, dependency-light
approximations that run inline during generation. The MCIF paper's own
numbers — WER against its exact normalizer, COMET, BERTScore — are a
**separate, post-hoc** step via `inspect score`, calling straight into the
official [`mcif`](https://github.com/hlt-mt/mcif) package
(`pip install mcif-bench`) rather than reimplementing any of it:

```bash
# generate first (as above); --no-score if this environment cannot also
# carry mcif-bench's dependencies
infra/runners/submit_eval.sh artemis /path/to/checkpoint configs/hf/mcif.yaml \
  -T task_filter=chunked_asr -T dataset_id=mcif-short-fixed-en --no-score

# then, from an environment with `pip install mcif-bench` and this package
# (no extras of its own needed -- just inspect_ai and huggingface_hub, both
# base dependencies):
inspect score path/to/log.eval \
  --scorer melteval/mcif_scoring.py@mcif_official_asr_scorer
```

| Scorer | Macro-task | Needs |
|---|---|---|
| `mcif_official_asr_scorer` | ASR | `jiwer` + `whisper-normalizer` (both pulled in by `mcif-bench`) |
| `mcif_official_trans_scorer` | TRANS | COMET (downloads a checkpoint on first use) **and** the external `mwerSegmenter` tool the official pipeline itself depends on — see `mcif.evaluation.MwerSegmenter`'s docstring for where to get it and `MWERSEGMENTER_ROOT` |
| `mcif_official_qa_scorer` / `mcif_official_sum_scorer` | QA / summarisation | `bert-score` (downloads a baseline-rescaling file per language on first use) |

No `-S` arguments: every sample already carries which `(repo, revision,
track, lang)` produced it, so the scorer re-downloads the exact reference file
generation used straight from the log. **The log must cover every member of
every reference group** — the official functions reassemble a group
themselves and raise if one is missing, so a partial run (`-T limit=`, a
debug slice) can't be scored this way; generate the whole `dataset_id` first.

`mcif-bench` pulls in COMET/`bert-score`/`torch` unconditionally (its
`evaluation` module imports all three at module load, even to compute WER
alone) — keep it in its own venv, the same reasoning as the COMET/MetricX venv
above. Its own pin of `torchmetrics` also expects `pkg_resources`, which a
fresh venv's `setuptools` may not provide anymore; `pip install "setuptools<81"`
if `import mcif.evaluation` fails with `ModuleNotFoundError: pkg_resources`.

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
