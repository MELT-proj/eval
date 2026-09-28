# Baseline matrix report

Every model (baselines, SMURF, MELT) on every benchmark slice (AIR-Bench, MCIF, ASR/ST sets, etc.), as one table.

```
configs/matrix/baselines-v1.yaml   # which models × which benchmark slices
scripts/run_matrix.py              # submits one SLURM job per (model, slice)
projects/baselines/report.py       # logs → results_long.csv + results.xlsx
```

## Run it

```bash
# 1. See what would be submitted on this site, and what is skipped and why
python scripts/run_matrix.py configs/matrix/baselines-v1.yaml --site bocconi --dry-run

# 2. Submit (re-running later submits only slices without a successful log)
python scripts/run_matrix.py configs/matrix/baselines-v1.yaml --site bocconi

# 3. Grade the free-text slices (audio_chat)
python scripts/run_matrix.py configs/matrix/baselines-v1.yaml --site bocconi --dry-run score
python scripts/run_matrix.py configs/matrix/baselines-v1.yaml --site bocconi score

# 4. Build the table (several --log-root to merge sites after an rsync)
python projects/baselines/report.py --log-root ~/scratch/eval-logs/matrix-baselines-v1 --out report-out
```

Checkpoints: the Qwen baselines from `$HF_HOME`, SMURF from the one `.ckpt`
in `checkpoints/smurf/`, MELT from the one run folder in `checkpoints/melt/`
(both gitignored, paths relative to the repo root). The MELT venv and the Shar
frozen sets come from the environment (`MELT_VENV_PATH`, `FLEURS24_ASR_SET`,
`FLEURS24_ST_XEN_SET`). Anything unset or missing on the current site is
skipped with a reaso.

## How the free-text slices are graded

Set per benchmark in the matrix's `score` block, so every model under test is
graded the same way and the choice is recorded next to the models:

- **AIR-Bench Chat** — `judge: qwen3-omni-30b-a3b-instruct`: a model from the
  matrix, loaded text-only (`model_args: {text_only: true}`, see
  `melteval/providers/hf.py`) from its own checkpoint. An API model can be
  named as `provider/model` instead. It may be better to use GPT here.
- **MCIF QA/SUM** — the paper's own BERTScore (`melteval/mcif_scoring.py`),
  run from a venv with `mcif-bench` (`MCIF_VENV_PATH`; see
  `infra/sites/bocconi.sh` for how to build it). SUM exists in the long track
  only, so those slices add its scorer.

Each pass is one SLURM job per (model under test, benchmark, scorer) via
`infra/runners/submit_score.sh`, rewriting the logs in place. A log that
already carries a scorer's results is skipped (`--rescore` to redo it).

## What comes out

| File                                | Contents                                                          |
| ----------------------------------- | ----------------------------------------------------------------- |
| `results_long.csv`                | One row per`(model, benchmark, slice, metric)`                  |
| `results.xlsx` › `summary`     | Models × headline metric per slice                               |
| `results.xlsx` › `<benchmark>` | Models × every metric the scorer logged                          |
| `results.xlsx` › `runs`        | The log each row came from, its checkpoint, date and sample count |

Headline metrics: `corpus_wer` (ASR), `corpus_bleu` (ST), `wer_<id>`/`bleu_<id>`
(MCIF chunked), `choice_accuracy` (AIR-Bench Foundation), `accuracy` (judge).
