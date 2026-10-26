# MetricX-24 for ST logs

Neural MT quality of finished speech-translation logs, offline. `melteval st-metrics` covers the lexical
metrics (BLEU, spBLEU, chrF++); this adds MetricX-24 from the same log, the way the README's COMET/MetricX
section prescribes: from its own venv (`metricx`: torch + transformers 4.40, no `inspect_ai`), never inside
`melteval`.

```
projects/metricx/score.py     log(s) -> per-sample scores + per-language / macro summary   (metricx venv, 1 GPU)
projects/metricx/table.py     summaries (+ chrF++ from st-metrics) -> markdown + long CSV    (melteval venv, CPU)
projects/metricx/submit.sh    one sbatch job that scores any list of logs
tests/test_metricx.py         unit tests (fake scorer, mock model) + one real-model `hub` smoke test
```

## What it computes

Checkpoint `google/metricx-24-hybrid-xxl-v2p6-bfloat16` at revision `3d9c6f7`, tokenizer `google/mt5-xl` at
`63fc645`, inference code ported from `google-research/metricx` at `fc4978e` (`metricx24/predict.py`,
`metricx24/models.py`). A regression model: it returns an **error score in [0, 25], lower is better**. One
hybrid checkpoint, two modes:

| column | mode | input string |
|---|---|---|
| `metricx_ref` | reference-based (primary) | `source: {src} candidate: {hyp} reference: {ref}` |
| `metricx_qe` | reference-free (secondary) | `source: {src} candidate: {hyp}` |

Inputs are tokenized with `google/mt5-xl`, truncated to **1536 tokens** (the length it was trained with), the
final EOS token is dropped, and the score is the bf16 logit of token 250089 (`<extra_id_10>`) after one decoder
step from token 0, clamped to [0, 25]. The checkpoint's `MT5ForRegression` is an `MT5ForConditionalGeneration`
read at one position, so the stock class loads it and the regression class is not vendored.

Where the three texts come from, per sample of a log: source = `metadata.source_text` (the transcript in the
source language, not the audio; English for en->X), hypothesis = `output.completion`, reference = `target`.
Languages are `src_lang` for X->en and `tgt_lang` for en->X, grouped and macro-averaged like the existing ST
tables, with their high/mid/low tiers.

Two things in the logs shape the numbers:

* **No source for Irish.** FLEURS Irish has no source transcript, in the logs or the manifest. Those samples are
  scored reference-based with an empty source (the upstream README allows it) and get no QE score.
  `macro_mean_with_source` leaves Irish out so reference-based and QE macros cover the same 22 languages.
* **Casing.** The backbones write almost no capitals; the references are cased and MetricX is case-sensitive.
  `--casing both` (default) scores every sample as written (`metricx_ref`, `metricx_qe`) and with source,
  hypothesis and reference lowercased (`..._lc`). The `_lc` columns are comparable to the lowercased chrF++ of
  the existing tables.

Samples are scored independently of their batch: sorted by length for speed, padded with an attention mask,
scattered back to input order. bf16 padding noise remains (the `hub` test bounds it).

## Run

```bash
# one job, any number of logs, model loaded once; skips runs that already have a summary.json
projects/metricx/submit.sh OUT_ROOT  llama1bBaseT=path/to/xen.json  llama1bBaseT=path/to/enx.json  ...
# tables, from the melteval venv
python projects/metricx/table.py --root OUT_ROOT --out OUT_ROOT/tables
```

Outputs per run: `OUT_ROOT/<name>/<dataset>/{scores.jsonl,summary.json,summary.csv}`. `scores.jsonl` has one row
per sample (`id`, `src_lang`, `tgt_lang`, `source_missing`, `metricx_ref`, `metricx_qe`, `metricx_ref_lc`,
`metricx_qe_lc`); `summary.json` starts with a `provenance` block (log, pinned revisions, templates, casing).

Tests: `pytest tests/test_metricx.py -m "not hub"` anywhere; the `hub` test needs the checkpoint in `$HF_HOME`
and a GPU, run from the `metricx` venv.
