# SMURF4EU results

The cross-campaign results table and the charts built from it.

```
master_long.csv  ──►  aggregate.py  ──►  aggregated.xlsx  ──►  plotting/plot_eval.py
```

`master_long.csv` is written by `projects/master/build.py` (one row per
`model x benchmark x slice x metric`). `aggregate.py` turns it into the
workbook the paper/communication tables read; `plotting/` is the port of the
`eval-plots` branch of the sibling `speechllm` repository, rewired to read the
workbook instead of the raw long table.

## Build the aggregated table

```bash
python projects/smurf4eu/aggregate.py \
    --long /path/to/master_long.csv \
    --out /path/to/aggregated
```

Writes `aggregated.xlsx` and `aggregated_long.csv` into `--out`.

`aggregated.xlsx`:

| Sheet | Contents |
| --- | --- |
| `aggregate` | One row per model, one column per headline metric: ASR WER/CER (LibriSpeech clean/other, VoxPopuli, MCIF), ST COMET-22 (FLEURS both directions, MCIF EN→DE/IT), MCIF SQA BERTScore. |
| `models` | Kind (checkpoint/baseline), run, step and source campaign for every model. |
| `coverage` | For every model x headline cell: `ok`, `pending` or `missing`. |
| `fleurs-st-enx`, `fleurs-st-xen`, `fleurs-asr`, `voxpopuli`, `librispeech`, `mcif` | Disaggregated by language (or LibriSpeech condition / MCIF task) with every metric the scorer logged, so a different aggregate (geometric mean, language subset, pooled WER) can be built without re-reading logs. |

Conventions: WER/CER are fractions on normalized text; the MCIF preamble-stripped
WER is kept next to the raw one. The `aggregate` tab uses a macro mean over
languages and records `*_n`. AIR-Bench and CoVoST2 are not aggregated.

## Refresh the pipeline

```bash
# 1. one results_long.csv per campaign, from its logs
python projects/baselines/report.py --log-root <campaign>/logs --out <campaign>/results

# 2. merge the campaigns (order is precedence; a number beats pending)
python projects/master/build.py \
    --source mcif207k=<...>/mcif-207k/results/results_long.csv \
    --source jupiter=<...>/master/sources/jup-eurollm/results_long.csv \
    ... \
    --out <master>

# 3. aggregate and plot
python projects/smurf4eu/aggregate.py --long <master>/master_long.csv --out <master>
python projects/smurf4eu/plotting/plot_eval.py --aggregate <master>/aggregated.xlsx \
    --out-dir <master>/plots
```

The fill campaigns that populated the empty baseline cells
(`configs/eval/baselines-fill-*.yaml` and `configs/eval/baselines-comet-backfill.yaml`)
are ordinary `evaluate.py run` / `score` / `report` campaigns; add their
`results_long.csv` to the `--source` list once they finish.

## Plot

```bash
python projects/smurf4eu/plotting/plot_eval.py \
    --aggregate /path/to/aggregated/aggregated.xlsx \
    --out-dir /path/to/figures
```

See [plotting/README.md](plotting/README.md).
