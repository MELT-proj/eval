# MELT campaign report

Turns a tree of `inspect` eval logs — one `infra/submit_campaign_mn5.sh` sweep,
every checkpoint of a training run against the ASR and ST frozen sets — into two
CSVs sized for a spreadsheet and a set of charts.

A campaign submits one job per `(task, corpus, language)`, so a checkpoint has
many logs rather than one, and how they are split does not matter here: every
number is aggregated from the sample records up, and the pooled `OVERALL` WER is
summed from the raw counts across whatever logs contributed. A campaign run as a
single ASR job over the whole frozen set reports identically.

```
projects/melt/
├── campaign.py   # logs → one long-format table (no plotting, no marimo)
├── charts.py     # that table → matplotlib figures
└── report.py     # marimo notebook wiring the two together and writing the files
```

## Install

The dependencies live in the repo's `analysis` extra:

```bash
uv pip install -e ".[analysis]"
```

Or, without `uv`:

```bash
python -m venv .venv && .venv/bin/pip install -e ".[analysis]"
```

Reading logs needs nothing from the model side — no torch, no `melt-proj`, no
GPU. It is fine on a login node or a laptop.

## Run it

Interactively:

```bash
marimo edit projects/melt/report.py
```

Headless — same cells, same outputs, no browser:

```bash
MELT_LOG_ROOT=/mnt/scratch-nyx/giuseppe/melt/eval-logs/mn5-campaign-v1 \
MELT_REPORT_OUT=report-out \
python projects/melt/report.py
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `MELT_LOG_ROOT` | Directory searched recursively for `*.eval` | the nyx campaign root above |
| `MELT_REPORT_OUT` | Where the CSVs, charts and cache are written | `report-out` |
| `MELT_REPORT_REFRESH` | Set to `1` to ignore the cache and re-read every log | unset |

In the notebook the same three values are editable in the form at the top, so
the environment only matters for the headless path.

Point `MELT_LOG_ROOT` at a parent directory to pull several runs into one
report — the run name comes from each log's own checkpoint path, not from where
the file sits, so nesting is free-form.

### Where the logs are

The campaign runs on MN5 and writes to
`/gpfs/scratch/epor48/itpt955676/campaign-logs/<run>/`. They are mirrored to nyx
(reachable as `/mnt/scratch-nyx/...` from artemis, which is also where
`inspect view` runs):

```bash
rsync -a --info=progress2 \
  mn5transfer:/gpfs/scratch/epor48/itpt955676/campaign-logs/<run> \
  /mnt/scratch-nyx/giuseppe/melt/eval-logs/mn5-campaign-v1/
```

A log whose job was killed at the wall clock keeps a non-`success` status and is
skipped, with a warning naming it — so a directory holding both a killed partial
and the `inspect eval-retry` that finished it reports the finished one. See
`infra/retry_eval_container_mn5.sbatch`.

## What comes out

Written to `$MELT_REPORT_OUT`:

| File | Contents |
| --- | --- |
| `asr_results.csv` | One row per model, one column per `wer/<corpus>/<lang>` and `cer/<corpus>/<lang>` |
| `st_results.csv` | One row per model, one column per `bleu/<corpus>/<direction>` and `chrf/<corpus>/<direction>` |
| `metrics_long.csv` | The same numbers in long form — one row per `(model, task, corpus, language, metric)`. What the charts read |
| `frame.parquet` (+ `.json`) | Extraction cache. Reused until a log's path, size or mtime changes |
| `headline_by_checkpoint.{png,pdf}` | Pooled ASR WER and mean ST BLEU per checkpoint, one colour per run |
| `asr_wer_by_checkpoint.{png,pdf}`, `asr_cer_…` | Per-corpus panels, one line per language |
| `st_bleu_by_checkpoint.{png,pdf}`, `st_chrf_…` | One line per translation direction |
| `asr_final_wer_heatmap.{png,pdf}` | Corpus × language grid at the end of training |

Both CSVs carry the same identifying columns up front:

* `model` — the run name, or `<run>-ckptNNNN` for an intermediate checkpoint.
* `run_name`, `seed`, `checkpoint`, `step`, `is_final`.

`seed` is parsed from the `-s<n>-` segment of the run name and kept as its own
column specifically so repeats of one recipe under different initialisations can
be averaged later (`groupby` everything except `seed`) without re-deriving it.

Two summary columns are not per-cell measurements and are named in caps to say
so: `wer/OVERALL/all` is the corpus rate over every ASR cell pooled (summed from
the counts, so it equals the log's own top-line number), while
`bleu/MACRO-AVG` is the unweighted mean over ST directions — a summary of the
six corpus scores next to it, not a corpus score itself, because BLEU's brevity
penalty is only defined over one corpus.

## How the numbers are obtained

Metrics are **re-aggregated from each sample's raw counts**, not read off the log
header. The header's ASR breakdown groups on language only
(`melteval/scorers.py`), while one ASR log covers four corpora at once — so its
`wer_it` pools voxpopuli, mls_sidon and cv22_sidon, and the per-corpus figure the
CSV needs does not exist there. Summing each cell's error and reference counts
and dividing once is the same arithmetic the corpus metric does, and it is
checked: `report.py` asserts the pooled cell reproduces the header's
`corpus_wer`/`corpus_cer` exactly, and that every sample lands in exactly one
cell.

**ST is the exception, and deliberately so.** Its numbers come from the header.
`inspect` truncates long strings when it writes the log's `summaries.json`, and a
checkpoint that has not learned to stop emits exactly those — degenerate
repetition loops, clipped mid-sentence at ~1000 characters. Recomputing BLEU over
the truncated text moves the score (on the llama-1b final checkpoint's pt→en log,
5 of 4023 hypotheses were clipped and BLEU came out 1.763 instead of 1.753) and it
moves *more* the worse the checkpoint is. The header value was computed by the
real scorer over the untruncated output, and a `melteval` ST run is always a
single direction — the scorer refuses mixed target languages rather than pick one
BLEU tokenizer for two — so the header number *is* the cell. `campaign.py` raises
if it ever meets an ST log that breaks that assumption. ASR is unaffected: its
scorer stores integer edit counts, not text.

## Reading the charts

* **Colour** is the breakdown — language for ASR, direction for ST — on a warm
  plum → brick → amber → gold ramp.
* **Marker shape** is the training run, and colour is never the only cue. Runs
  are what the campaign accumulates over time (a second model, a second seed),
  and a chart of one run reads the same way as a chart of four.
* **The enlarged marker** is the end-of-training checkpoint.

One caveat on the x-axis: an eval log records which checkpoint it loaded but not
that run's `max_steps`, so the `final` point is placed one median checkpoint gap
past the last numbered one. Read it by its label, not by its step. When a run's
last numbered checkpoint *is* the end of training, `final` simply repeats it and
the curve ends flat — that is the data, not an artefact.
