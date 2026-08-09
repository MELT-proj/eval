# AGENTS.md — melt-eval

Guidance for code agents working in this repository. It follows the conventions
of the sibling `MELT-proj/training` repo; where they differ, this file wins.

## What this is

A downstream-evaluation harness for MELT speech models, built as an **extension**
to `inspect_ai` (pinned), not a fork. Providers, datasets, solvers and scorers
register through inspect's decorators and the `inspect_ai` entry-point group.

Before proposing a change to vendored or forked framework code: don't. If
something genuinely cannot be done from outside, name the specific hook that
forces it.

## Layout

- `melteval/manifest.py` — the frozen-set record schema. The boundary between
  reading corpora and running evals.
- `melteval/freeze.py` — builds frozen sets from a source spec.
- `melteval/readers/` — one module per source type. The only code that knows
  about corpus formats.
- `melteval/providers/` — one module per model family. The seam that lets a
  NeMo/Smurf model be evaluated without touching anything else.
- `configs/` — frozen-set specs.
- `docs/` — design notes worth reading before changing behaviour.

## Coding conventions

- `ruff` for style; line length 119.
- Type hints on function signatures; docstrings on public functions and classes.
- **Do not import from `typing`** — Python ≥3.10 builtins (`X | None`,
  `list[str]`, `dict[str, int]`).
- `uv`, not `pip`.
- Add tests to existing test files where one fits.
- Keep PRs focused and minimal.

## Things that are easy to get wrong here

- **Reference resolution.** The same corpus keeps its reference text in
  different places (`supervisions[0].text`, `custom.pnc_text`,
  `custom.metadata.sentence`), and per-cut `tags.text_field` overrides the
  source-level setting. Getting this wrong does not crash; it scores every
  sample against the wrong string and reports a plausible number.
- **Prompt/format parity.** A checkpoint must be evaluated in the format it was
  trained in. Read it from the run's `training_config.yaml`; never default
  silently. This has already been shipped as a bug once
  ([training#58](https://github.com/MELT-proj/training/issues/58)).
- **Corpus vs per-sample metrics.** Corpus WER is total errors over total
  reference words, not the mean of per-sample rates. Same for BLEU.
- **Sample identity.** FLEURS cut IDs are not unique. Never key an eval log by
  a corpus's own ID.
- **Disk.** Frozen sets reference audio, they do not copy it. Check `df` before
  adding anything that writes audio, caches datasets, or builds an image.
- **Shar layouts.** Both plain (`cuts.*.jsonl.gz`) and indexed
  (`cuts.*.jsonl` + `.idx`) exist. Globbing only the gzipped form reports a
  fully populated source as empty — use `shar_manifest_files`.

## Testing

```bash
pytest tests -q                                     # unit only
MELTEVAL_SHAR_ROOT=/path/to/shar pytest tests -q    # + corpus integration
```

Integration tests skip themselves without `MELTEVAL_SHAR_ROOT`. Both need an
environment with `inspect_ai`, `melt-proj` and `lhotse` together — see
README.md "Install" for the venv (no existing site venv or container image had
all three, which is why a dedicated one exists at
`/mnt/scratch-artemis/giuseppe/venvs/melteval` on artemis). Provider tests that
load an actual checkpoint go through `infra/runners/submit_eval.sh` instead
(GPU work — see below), not through `pytest`.

## Environment notes (artemis)

- Dev/test venv: `/mnt/scratch-artemis/giuseppe/venvs/melteval` — the full
  stack (`inspect_ai`, `melt-proj`, `lhotse`, `torch`) in one place.
- Container: `/mnt/scratch-artemis/giuseppe/melt-data/melt_cuda126_lhotse2_td.sif`
  has `melt`/`lhotse`/`torch` (the `_td` suffix matters: the other image has no
  torchdata, which the training package imports) but no `inspect_ai` — useful
  for exercising the shar reader against real corpora without the venv above,
  not for a full eval run. `singularity` needs `--userns`, and the repo must be
  bind-mounted and `cd`-ed into, or Python imports the container's baked-in
  copy of `melt`.
- Corpora: `/mnt/scratch-nyx/giuseppe/melt/melt-data/shar` (indexed tree).
- **Never run anything on an artemis GPU directly — always go through a SLURM
  job (`sbatch`), never a bare `CUDA_VISIBLE_DEVICES=...` invocation on the
  workstation, even for a quick smoke test.** artemis is shared, and GPUs sit
  outside SLURM's accounting when used this way, invisible to everyone else
  scheduling against them. CPU-only work (freezing, unit tests, linting) is
  fine to run directly; anything that loads a model or touches a GPU is not.
  Use `infra/runners/submit_eval.sh <site> <checkpoint> <frozen_set> [...]`
  (see README.md "Running an evaluation on a cluster"), which mirrors the
  `sbatch` + site-file pattern `training`'s `infra/runners/submit-container.sh`
  uses.
- **Never write anything sizeable under `/mnt/home/giuseppe`.** It is synced
  via GlusterFS, so large or frequently-changing files there both eat a home
  quota and generate sync churn. Venvs, eval outputs, frozen sets, logs and
  caches all belong under `/mnt/scratch-artemis/giuseppe` or
  `/mnt/data-artemis/giuseppe` — the repo checkout itself (source only) is the
  one thing that belongs in `$HOME`.
