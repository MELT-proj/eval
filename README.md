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

```bash
uv venv && uv pip install -e ".[shar,metrics,dev]"
```

Extras are split by what you actually need: `shar` (local lhotse corpora, pulls
in the training package), `hf` (remote HuggingFace test splits), `metrics`
(jiwer, sacrebleu).

## Usage

```bash
# Build a frozen set
LOCAL_DATASETS_DIR=/path/to/shar melteval freeze configs/smoke.yaml -o runs/smoke

# Look at what it contains
melteval show runs/smoke
```

## Development

```bash
pytest tests -q                                     # unit tests only
MELTEVAL_SHAR_ROOT=/path/to/shar pytest tests -q    # adds corpus integration tests
ruff check melteval tests
```

On artemis the tests run inside the container, which is the only environment
with the full stack:

```bash
singularity exec --userns \
    -B /mnt/scratch-artemis:/mnt/scratch-artemis \
    -B /mnt/scratch-nyx:/mnt/scratch-nyx \
    -B /mnt/home/giuseppe:/mnt/home/giuseppe \
    /mnt/scratch-artemis/giuseppe/melt-data/melt_cuda126_lhotse2_td.sif \
    bash -c 'source /workspace/venv/bin/activate
             export PYTHONPATH=/mnt/scratch-artemis/giuseppe/pytest-shim:/path/to/training:/path/to/melt-eval
             cd /path/to/melt-eval && python -m pytest tests -q'
```

The `--userns` flag is required on artemis, and the container ships neither
pytest nor pip — hence the shim on `PYTHONPATH`.
