# SpeechEval plots

Bar plots (PDF + PNG) and CSV tables for the SpeechEval results of one SMURF
checkpoint (reported as `EuroSpeech-9B-26-v0.1`, i.e. `ckpt-207k`) against the
baselines.

Ported from the `eval-plots` branch of the sibling `speechllm` repository and
rewired to read `aggregated.xlsx` (`projects/smurf4eu/aggregate.py`) instead of
`results_long.csv`.

```bash
python projects/smurf4eu/plotting/plot_eval.py \
    --aggregate /path/to/aggregated/aggregated.xlsx \
    --out-dir eval_plots/output
```

Outputs:

* `summary.{pdf,png}`: the headline chart, always on the languages supported by
  every model (no model is averaged over a different set of languages):
  VoxPopuli ASR (English; multilingual mean over the common languages), FLEURS
  ST X→EN and EN→X, and MCIF chunked ST EN→DE / EN→IT in COMET-22. The common
  languages are listed under each panel.
* In three variants (`all/`, `full_support/`, `common/`):
  `heatmap_<task>.{pdf,png}`, model x language heatmaps with the scores in the
  cells, and `tables/per_language_<task>_<variant>.{csv,md}`.
* `english_benchmarks.{pdf,png}` (LibriSpeech clean/other WER) and
  `tables/summary.csv`.

Conventions: Chinese and other non-EU languages are dropped; Qwen is restricted
to its speech-side languages (en, de, ru, it, fr, es, pt, nl) for ASR and X→EN;
EN→X is reported in full. Whisper large-v3 is one model: ASR comes from its
transcribe mode and X→EN from its translate mode (no EN→X). FLEURS and MCIF ST
are COMET-22 DA (x100). A model without a COMET value in the workbook has no
FLEURS ST bars.

The model list, display names, colours and supported languages live in
`MODELS` at the top of `plot_eval.py`. A model absent from `MODELS` is simply
left out of the figures; add an entry to include a new baseline.
