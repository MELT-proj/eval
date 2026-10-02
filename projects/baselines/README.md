# Baseline report

Every model (baselines, SMURF, MELT) on every benchmark slice (AIR-Bench, MCIF,
ASR/ST sets, etc.), as one table.

```
configs/eval/baselines-v1.yaml     # which models × which benchmark slices
scripts/evaluate.py                # runs them, one job per (model, slice)
projects/baselines/report.py       # logs → summary.csv, results_long.csv, results.xlsx
```

## Run it

How the launcher works: [docs/running-evaluations.md](../../docs/running-evaluations.md).

```bash
python scripts/evaluate.py plan  configs/eval/baselines-v1.yaml --site bocconi
python scripts/evaluate.py run   configs/eval/baselines-v1.yaml --site bocconi --wait
python scripts/evaluate.py score configs/eval/baselines-v1.yaml --site bocconi --wait
```

The table lands in `<output_dir>/results/`. `report.py` also runs on its own,
e.g. to merge logs from several machines after an rsync:

```bash
python projects/baselines/report.py --log-root <dir> --log-root <dir> --out report-out
```

Models: the Qwen baselines are `melt/hf/qwen2_audio` and `melt/hf/qwen3_omni`,
at the revisions pinned in [docs/baseline-providers.md](../../docs/baseline-providers.md).
SMURF comes from the one `.ckpt` in `checkpoints/smurf/`, MELT from the one run
folder in `checkpoints/melt/` (both gitignored, paths relative to the repo
root). The MELT venv and the Shar frozen sets come from the environment
(`MELT_VENV_PATH`, `FLEURS24_ASR_SET`, `FLEURS24_ST_XEN_SET`). Anything unset
or missing on the current machine is skipped, with the reason.

## How the free-text slices are graded

Per benchmark, with `judge:` in the config, so every model under test is graded
the same way:

- **AIR-Bench Chat** — `judge: melt/hf/qwen3_omni`, loaded text-only
  (`model_args: {text_only: true}`, see `melteval/providers/hf.py`). An API
  model (`openai/gpt-4o`, its key in the environment) can be named instead.
  Caveat: Qwen3-Omni is also a model under test, so it grades its own answers.

## What comes out

| File                           | Contents                                                          |
| ------------------------------ | ----------------------------------------------------------------- |
| `summary.csv`                  | Headline metric per (benchmark, slice) × model                    |
| `results_long.csv`             | One row per `(model, benchmark, slice, metric)`                   |
| `results.xlsx` › `summary`     | Models × headline metric per slice                                |
| `results.xlsx` › `<benchmark>` | Models × every metric the scorer logged                           |
| `results.xlsx` › `runs`        | The log each row came from, its model, date and sample count      |

Headline metrics: `corpus_wer_raw` (ASR), `corpus_bleu` (ST), `wer_<id>`/`bleu_<id>`
(MCIF chunked), `choice_accuracy` (AIR-Bench Foundation), `accuracy` (judge).
