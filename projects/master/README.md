# Master results table

Every finished campaign, every model, in one workbook. Built from the
`results/results_long.csv` each campaign already writes — it does not read
logs or rescore anything.

```bash
python projects/master/build.py \
    --source mcif207k=.../eval-runs/mcif-207k/results/results_long.csv \
    --source jupiter=.../eval-runs/eurollm-16n-run1/results/results_long.csv \
    --source mcifcomet=.../eval-runs/mcif-comet/results/results_long.csv \
    --out .../eval-runs/master
```

`--source NAME=PATH` is repeatable, and **order is precedence**: the first
campaign that has a `(model, benchmark, slice, metric)` cell wins it. A numeric
value always beats a later campaign's `pending`, so a campaign that scored a
metric is never shadowed by one that merely declared it. That is how the master
keeps the ArtemIS fresh run of step-0207000 for MCIF while taking everything
else from JUPITER.

## What comes out

| File / sheet | Contents |
| --- | --- |
| `master_at_a_glance.csv` | One row per model, one column per headline metric |
| `master_long.csv` | The merged long table, with the winning `source` per row |
| `master.xlsx` › `at_a_glance` | The same table as the CSV, formatted |
| `master.xlsx` › one per benchmark | Models × every slice/metric the scorer logged |
| `master.xlsx` › `coverage` | Slices / values / pending per model and family |
| `master.xlsx` › `sources` | Path and mtime of every input |

Headline columns: FLEURS-24 ST BLEU/COMET in both directions, FLEURS ASR WER,
VoxPopuli WER, LibriSpeech clean/other WER, MCIF ASR short/long WER, MCIF ST
BLEU/COMET for de and it, MCIF QA short en/de/it and long QA/SUM BERTScore,
MCIF judge score, AIR-Bench accuracy, CoVoST2 BLEU.

## Caveats to read the table with

- **Pending is not missing.** A slice a campaign generated but did not score
  shows `pending` in `master_long.csv` and is summed in the `coverage` sheet.
  Blank in `at_a_glance` means nobody declared that metric for that model.
- **WER is the raw corpus figure.** `MCIF ASR short/long WER` is
  `wer_<dataset_id>` (equivalently `corpus_wer_raw`), the same number the
  campaign table reports. A model that wraps its transcript (Qwen2-Audio does
  on 88% of MCIF segments) also has a preamble-stripped figure, kept in the
  `MCIF` sheet as `wer_extracted_<dataset_id>`.
- **Two generations of the same checkpoint.** The 0207000 checkpoint was run
  on ArtemIS with the rebuilt SMURF venv and earlier on JUPITER; their MCIF
  numbers differ (BLEU it 42.37 vs 41.66). The ArtemIS run wins in the master
  because it is first in `--source`.
- **`-test` benchmark ids are merged into their family.** `fleurs24-st-enx` and
  `fleurs24-st-enx-test` land in the same column; no model has run both, so
  there is nothing to cross-check yet.
- **DeSTA2.5 is absent.** Its 56 JUPITER logs are all `status=error`
  (`ModuleNotFoundError: lhotse` in its venv); the report skips them.
- **Apertus ST is not a usable number.** Its BLEU is 1.4–2.1, which is the
  provider under development (PR #34), not the model.
- **Earlier campaigns.** `jup-smurf` and `jup-smurf-run` are separate,
  superseded runs. They are not merged by default; add them as extra
  `--source` entries at the end of the list if a number only exists there.
