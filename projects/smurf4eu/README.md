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
| `aggregate` | One row per model, one column per headline metric: ASR WER/CER (LibriSpeech clean/other, VoxPopuli, MCIF), ST COMET-22 (FLEURS both directions, MCIF), MCIF SQA BERTScore. MCIF ST and MCIF SQA are one column each, macro-averaged over their languages (`*_n` records how many). |
| `models` | Kind (checkpoint/baseline), run, step, source campaign and `aliases` for every model. |
| `coverage` | For every model x headline cell: `ok`, `pending` or `missing`. |
| `fleurs-st-enx`, `fleurs-st-xen`, `fleurs-asr`, `voxpopuli`, `librispeech`, `mcif` | Disaggregated by language (or MCIF task) with every metric the scorer logged, so a different aggregate (geometric mean, language subset, pooled WER) can be built without re-reading logs. LibriSpeech is the exception: its two conditions are columns (`wer_clean` / `wer_other`, `cer_clean` / `cer_other`), not rows. |

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

## Baseline coverage

The eight baselines were filled with the campaigns in `configs/eval/`
(`baselines-comet-backfill`, `baselines-fill-mcif-sqa`,
`baselines-fill-mcif-st`, `baselines-fill-asr`, plus
`baselines-fill-asr-qwen3` for the languages Qwen3's one-log run left
unfinished). A cell is empty for one of two reasons, never because a run was
forgotten:

**Not applicable to the model**

| Model | Empty cells | Why |
| --- | --- | --- |
| `whisper-large-v3` | FLEURS En→X, MCIF ST, MCIF SQA | Whisper translates only into English and has no chat/QA path. Its X→En numbers live under `whisper-large-v3-translate`, the same checkpoint in translate mode; the plots alias the two. |
| `whisper-large-v3-translate` | everything except FLEURS X→En | Translate-only mode. |
| `qwen2-audio-7b-instruct` | MCIF SQA (short and long) | Its 30 s audio window rejects MCIF's longer clips (a 34.1 s clip aborts the task). |
| `canary-1b-v2` | MCIF SQA | Not a chat model. Its MCIF ASR/ST are filled by `baselines-fill-mcif-canary.yaml`, which overrides MCIF's free-text prompt with the `<task>\|<src>\|<tgt>` spec (`instruction_override`). |
| `gemma-4-e4b-it`, `phi-4-multimodal-instruct` | MCIF SQA long track | Skipped by design: the short track is filled; their audio context is not documented for multi-minute talks. |

**Excluded on purpose**

- DeSTA2.5-Audio: deferred.
- AIR-Bench and CoVoST2: not part of this table.

Note that `whisper-large-v3` and `whisper-large-v3-translate` are one checkpoint
in two modes; the workbook merges them into one row (the original name is kept
in `models.aliases`), as do the plots.

## Known timeouts and how they were finished

Two 24 h-queue slices hit their wall clock while the table was being filled:

- **Qwen3 FLEURS ASR** completed 15,316/19,463 samples in 4 h. A timed-out log
  is not a successful log, so none of it is reusable; the whole benchmark is
  re-run one language per slice by `baselines-fill-asr-qwen3.yaml`, four at a
  time on the H200 partition. Re-running the config skips languages that
  already have a successful log.
- **Apertus MCIF long-track SQA** reached 290/320 in 3 h 30. Retried with the
  slice's limit raised from 3 h 30 to 6 h.

Both write into the same `fleurs-asr` / `mcif` tabs as the rest of the
campaign; nothing needs a separate merge step.

