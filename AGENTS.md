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
- `melteval/freeze.py` — builds frozen sets from a source spec. `read_sources`
  is the half that a live (unfrozen) run shares, so `spec_dataset` and a
  frozen set produce identical records and identical sample keys.
- `melteval/dataset.py` — the two ways into an eval: a frozen manifest, or a
  spec read at eval time (HuggingFace benchmarks, which are already immutable
  and have nothing left for a freeze pass to pin down).
- `melteval/readers/` — one module per source type. The only code that knows
  about corpus formats.
- `melteval/providers/` — one module per model family. The seam that lets a
  NeMo/Smurf model be evaluated without touching anything else.
- `melteval/rescore.py`, `melteval/mcif_scoring.py` — post-hoc scorers for
  metrics that need a neural model (COMET/MetricX, or MCIF's own
  WER/COMET/BERTScore via the official `mcif` package) run from a separate
  venv via `inspect score`, never inline during generation. Not reachable
  through `melteval.registry` on purpose.
- `configs/` — source specs. `configs/shar/` for local Shar mixtures (frozen before
  use), `configs/hf/` for HuggingFace benchmarks (read live, or frozen -- see
  "Evaluating a HuggingFace benchmark" in README.md).
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
- **Relative paths in `-T` arguments do not mean what they look like.**
  `inspect eval` chdir's into the task file's directory (`melteval/`) while it
  builds the task, so `-T spec=configs/hf/air-bench.yaml` is looked for under
  `melteval/configs/`, and fails from a job whose working directory obviously
  contains it. `infra/run_eval.sbatch` resolves the evaluation set with
  `realpath` before passing it on; anything else path-shaped needs the same
  treatment.
- **A per-sample `instruction` is a template, not a string.** It goes through
  `str.format` so a benchmark can place `{audio_token}` itself. Corpus text
  spliced into one must go through `prompt.escape_literal` first: braces do
  occur in real questions, and an unescaped one is either a `KeyError`
  mid-generation or, worse, a silently rewritten question.
- **Matching an answer to a multiple-choice option needs word boundaries.**
  "Male" is a substring of "female", so a plain `in` test matches both options
  of a gender question, resolves to neither, and books the sample as an
  unreadable completion rather than a correct one. `mcq_scorer` reports
  `unresolved_rate` alongside accuracy precisely so this class of thing is
  visible instead of just depressing the score.
- **A reference can span more than one sample.** MCIF's `short` track cuts a
  whole talk's transcript across dozens of short segments, each transcribed
  on its own, and scores the whole group at once against one reference —
  `EvalRecord.extra` (`group_id`/`group_order`/`group_size`) carries that
  through to the scorer, which reassembles a group's completions in order
  before comparing to it (`chunked_asr_scorer`/`chunked_st_scorer`). Scoring
  each chunk against the whole-group reference independently produces a
  number, just not a meaningful one. `extra` itself is the general escape
  hatch: a reader whose corpus needs metadata no other reader does should add
  to it rather than growing `EvalRecord`'s named fields.
- **`Score.value` is not a free-form payload slot.** It goes through inspect's
  epoch-reduction machinery even for a single-epoch task, and a dict there is
  treated as named *numeric* sub-scores — `value_to_float()` runs on every
  entry. Text payloads (anything a corpus-level metric needs to recompute
  over, like `st_scorer`'s hypothesis/reference pairs) belong in
  `Score.metadata` instead, which is never touched by that reduction. This
  bit `st_scorer` for real (see the git history on `scorers.py`): every
  reference/hypothesis silently became `0.0` before `corpus_bleu` ever ran.
  A test that hand-builds `SampleScore` objects and calls a metric function
  directly bypasses this reduction entirely and cannot catch a regression
  here — only a test that runs a real `inspect_ai.eval()` can (see
  `TestStScorerThroughRealInspect` in `test_scorers.py`).

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
- HuggingFace cache: `/mnt/scratch-artemis/giuseppe/melt-data/hf_cache`.
  Compute nodes run with `HF_HUB_OFFLINE=1`, so a split has to be pulled from
  the login shell before any job that reads it — see README.md, "Evaluating a
  HuggingFace benchmark".
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
