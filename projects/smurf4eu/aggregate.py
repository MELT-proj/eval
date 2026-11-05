"""Turn the master long table into ``aggregated.xlsx``: one tab of headline scores,
one tab per dataset disaggregated by language.

The input is the long table written by ``projects/master/build.py`` (or any
concatenation of campaign ``results_long.csv`` files with the same columns).
Nothing here is specific to a model list: models, languages and slices are
discovered from the table, so a new campaign is picked up by re-running it.

Output (``--out``):

* ``aggregated.xlsx``
  - ``aggregate``: one row per model, one column per headline metric, with the
    number of languages each mean covers (``*_n``).
  - ``models``: kind/run/step/campaign for every model.
  - ``coverage``: for each model x dataset x headline metric, whether a numeric
    value exists, is still ``pending`` or is absent entirely.
  - one tab per dataset (``fleurs-st-enx``, ``fleurs-st-xen``, ``fleurs-asr``,
    ``voxpopuli``, ``librispeech``, ``mcif``), disaggregated by language (or by
    LibriSpeech condition / MCIF task) with every metric the scorer logged.
* ``aggregated_long.csv``: the same rows in tidy form, for plotting or for a
  different aggregation.

Conventions:

* ASR ``wer``/``cer`` are the scorer's normalized corpus metrics, as fractions.
  MCIF's preamble-stripped variant is kept next to the raw one (``wer_extracted``).
* FLEURS ST ``comet`` and MCIF ``comet`` are COMET-22 DA (0-1); the workbook does
  not multiply by 100 (the plots do).
* The ``aggregate`` tab collapses per-language values with a macro mean and
  records ``n``; it never pools error counts, which the long table does not carry.

AIR-Bench and CoVoST2 are intentionally not aggregated.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)

#: Excel forbids these in sheet names and caps them at 31 characters.
_SHEET_BAD = set("[]:*?/\\")

#: Values that mean "the slice ran but the post-hoc scorer has not produced a number".
_PENDING = "pending"


@dataclass(frozen=True)
class Dataset:
    """A benchmark family and how to read its slice ids."""

    key: str
    title: str
    benchmarks: tuple[str, ...]
    #: Regex whose first group is the language (or LibriSpeech condition).
    slice_pattern: str
    #: Which group of *slice_pattern* carries the language/condition.
    group: int = 1

    def sheet(self) -> str:
        """Excel-safe sheet name; the plotting code reads these keys."""
        return "".join(c for c in self.key if c not in _SHEET_BAD)[:31]


DATASETS = {
    "fleurs-st-enx": Dataset(
        "fleurs-st-enx",
        "FLEURS ST en->x",
        ("fleurs24-st-enx", "fleurs24-st-enx-test"),
        r"^st-fleurs-en_([a-z]{2})$",
    ),
    "fleurs-st-xen": Dataset(
        "fleurs-st-xen",
        "FLEURS ST x->en",
        ("fleurs24-st-xen", "fleurs24-st-xen-test"),
        r"^st-fleurs-([a-z]{2})_en$",
    ),
    "fleurs-asr": Dataset("fleurs-asr", "FLEURS ASR", ("fleurs24-asr-test",), r"^asr-([a-z]{2})$"),
    "voxpopuli": Dataset(
        "voxpopuli",
        "VoxPopuli ASR",
        ("voxpopuli", "voxpopuli-test"),
        r"^asr-(?:voxpopuli-)?([a-z]{2})$",
    ),
    "librispeech": Dataset(
        "librispeech",
        "LibriSpeech ASR",
        ("librispeech", "librispeech-other"),
        r"^asr-librispeech-(?:test-)?(clean|other)$",
    ),
    "mcif": Dataset("mcif", "MCIF", ("mcif",), r"^.+$"),
}

#: MCIF slice -> (task, language, variant). The metric names also carry the
#: target language for ST; keeping it here makes the mapping explicit.
_MCIF_SLICES = {
    "chunked_asr-mcif-short-fixed-en": ("asr", "en", "short"),
    "chunked_asr-mcif-long-fixed-en": ("asr", "en", "long"),
    "chunked_st-mcif-short-fixed-de": ("st", "de", "short"),
    "chunked_st-mcif-short-fixed-it": ("st", "it", "short"),
    "audio_chat-mcif-short-fixed-en": ("qa", "en", "short"),
    "audio_chat-mcif-short-fixed-de": ("qa", "de", "short"),
    "audio_chat-mcif-short-fixed-it": ("qa", "it", "short"),
    "audio_chat-mcif-long-fixed-en": ("qa_long", "en", "long"),
}

#: Canonical metric per long-table metric name. A callable is used where the
#: language has to be read from the name (``wer_de``, ``bleu_de``, ...).
_METRIC_ALIASES = {
    "corpus_bleu": "bleu",
    "corpus_chrf": "chrf",
    "corpus_spbleu": "spbleu",
    "comet": "comet",
    "comet_coverage": "comet_coverage",
    "official_trans_comet": "comet",
    "mcif_official_trans_scorer": "comet",
    "corpus_wer_raw": "wer",
    "corpus_wer_extracted": "wer_extracted",
    "corpus_cer": "cer",
    "corpus_cer_extracted": "cer_extracted",
    "preamble_rate": "preamble_rate",
    "official_bertscore": "bertscore",
    "mcif_official_qa_scorer/official_bertscore": "bertscore",
    "mcif_official_sum_scorer/official_bertscore": "bertscore_sum",
    "wer_mcif-short-fixed-en": "wer",
    "wer_mcif-long-fixed-en": "wer",
    "wer_extracted_mcif-short-fixed-en": "wer_extracted",
    "wer_extracted_mcif-long-fixed-en": "wer_extracted",
    "cer_mcif-short-fixed-en": "cer",
    "cer_mcif-long-fixed-en": "cer",
    "bleu_mcif-short-fixed-de": "bleu",
    "bleu_mcif-short-fixed-it": "bleu",
    "chrf_mcif-short-fixed-de": "chrf",
    "chrf_mcif-short-fixed-it": "chrf",
}

#: ``bleu_<lang>`` / ``chrf_<lang>`` / ``spbleu_<lang>`` / ``wer_<lang>`` / ``cer_<lang>``.
_LANG_METRIC = re.compile(r"^(bleu|chrf|spbleu|wer|cer)_([a-z]{2})$")


def _canonical_metric(metric: str) -> str | None:
    """Map a long-table metric name to the canonical name, or ``None`` to ignore it."""
    if metric in _METRIC_ALIASES:
        return _METRIC_ALIASES[metric]
    match = _LANG_METRIC.match(metric)
    return match.group(1) if match else None


def _classify(benchmark: str, slice_id: str) -> tuple[str, str | None, str | None, str | None] | None:
    """Return ``(dataset, task, language/condition, variant)`` for one row, or ``None``.

    Unknown benchmarks are reported and skipped rather than guessed at.
    """
    if benchmark == "mcif":
        try:
            task, lang, variant = _MCIF_SLICES[slice_id]
        except KeyError:
            return None
        return "mcif", task, lang, variant
    for dataset in DATASETS.values():
        if dataset.key == "mcif":
            continue
        if benchmark not in dataset.benchmarks:
            continue
        match = re.match(dataset.slice_pattern, slice_id)
        if match:
            return dataset.key, None, match.group(dataset.group), None
        # A one-log run (the suite's FLEURS ASR) has slice "asr"; the language
        # is then in the metric name (``wer_de``), filled in by normalize().
        if dataset.key in ("fleurs-asr", "voxpopuli"):
            return dataset.key, None, None, None
        return None
    return None


def normalize(long: pd.DataFrame) -> pd.DataFrame:
    """Classify every row into ``(dataset, task, lang, metric)`` and drop unmapped ones.

    Args:
        long: A long results table with at least ``model``, ``benchmark``,
            ``slice``, ``metric`` and ``value``.

    Returns:
        A frame with ``dataset``, ``task``, ``lang``, ``metric``, ``value_num``
        and ``state`` (``ok`` / ``pending``).
    """
    needed = {"model", "benchmark", "slice", "metric", "value"}
    missing = needed - set(long.columns)
    if missing:
        raise ValueError(f"not a results_long table: missing columns {sorted(missing)}")

    frame = long.copy()
    frame["value_num"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["state"] = frame["value"].map(lambda value: "pending" if str(value) == _PENDING else "ok")
    frame["dataset"] = None
    frame["task"] = None
    frame["lang"] = None
    frame["variant"] = None
    frame["metric"] = frame["metric"].map(lambda metric: _canonical_metric(metric) or metric)
    frame["raw_metric"] = long["metric"]
    # Prefer the corpus metric (``corpus_bleu``) to a per-language duplicate
    # (``bleu_de``), which carry the same number; then a number to ``pending``.
    frame["_corpus"] = long["metric"].map(lambda m: 0 if str(m).startswith("corpus_") else 1)

    keep = []
    unmapped = set()
    for index, row in frame.iterrows():
        classified = _classify(str(row["benchmark"]), str(row["slice"]))
        if classified is None:
            unmapped.add(f"{row['benchmark']}/{row['slice']}")
            keep.append(False)
            continue
        dataset, task, lang, variant = classified
        if lang is None:
            # One-log FLEURS ASR / VoxPopuli runs: the language is the metric suffix.
            match = _LANG_METRIC.match(str(row["raw_metric"]))
            lang = match.group(2) if match else None
        if lang is None and dataset in ("fleurs-asr", "voxpopuli"):
            # A corpus-level duplicate of the per-language rows in a one-log run.
            keep.append(False)
            continue
        frame.at[index, "dataset"] = dataset
        frame.at[index, "task"] = task
        frame.at[index, "lang"] = lang
        frame.at[index, "variant"] = variant
        keep.append(True)
    if unmapped:
        logger.warning("skipped %d slice(s) outside the aggregated datasets: %s", len(unmapped), sorted(unmapped)[:8])

    out = frame[keep].copy()
    out = out.sort_values(["_corpus", "value_num"], ascending=[True, False]).drop_duplicates(
        ["model", "dataset", "task", "lang", "variant", "metric"], keep="first"
    )
    return out[["model", "dataset", "task", "lang", "variant", "metric", "value_num", "state", "source"]].reset_index(
        drop=True
    )


@dataclass(frozen=True)
class Headline:
    """One column of the aggregate tab."""

    label: str
    dataset: str
    metric: str
    task: str | None = None
    lang: str | None = None
    #: ``mean`` over every selected row (with ``n``), or ``first`` for one value.
    agg: str = "mean"


#: The aggregate tab, in display order.
HEADLINES = [
    Headline("LibriSpeech clean WER", "librispeech", "wer", lang="clean", agg="first"),
    Headline("LibriSpeech clean CER", "librispeech", "cer", lang="clean", agg="first"),
    Headline("LibriSpeech other WER", "librispeech", "wer", lang="other", agg="first"),
    Headline("LibriSpeech other CER", "librispeech", "cer", lang="other", agg="first"),
    Headline("VoxPopuli WER", "voxpopuli", "wer"),
    Headline("VoxPopuli CER", "voxpopuli", "cer"),
    Headline("MCIF ASR WER", "mcif", "wer", task="asr", lang="en", agg="first"),
    Headline("MCIF ASR CER", "mcif", "cer", task="asr", lang="en", agg="first"),
    Headline("FLEURS En->X COMET", "fleurs-st-enx", "comet"),
    Headline("FLEURS X->En COMET", "fleurs-st-xen", "comet"),
    Headline("MCIF ST EN->DE COMET", "mcif", "comet", task="st", lang="de", agg="first"),
    Headline("MCIF ST EN->IT COMET", "mcif", "comet", task="st", lang="it", agg="first"),
    Headline("MCIF SQA en BERTScore", "mcif", "bertscore", task="qa", lang="en", agg="first"),
    Headline("MCIF SQA de BERTScore", "mcif", "bertscore", task="qa", lang="de", agg="first"),
    Headline("MCIF SQA it BERTScore", "mcif", "bertscore", task="qa", lang="it", agg="first"),
    Headline("MCIF SQA long QA BERTScore", "mcif", "bertscore", task="qa_long", lang="en", agg="first"),
    Headline("MCIF SQA long SUM BERTScore", "mcif", "bertscore_sum", task="qa_long", lang="en", agg="first"),
]


def _step(model: str) -> int | None:
    match = re.search(r"(?:^|-)step-(\d+)$", model)
    return int(match.group(1)) if match else None


def models_table(long: pd.DataFrame) -> pd.DataFrame:
    """One row per model with kind, run, step and the campaign it came from."""
    first = long.drop_duplicates("model").set_index("model")
    rows = []
    for model, row in first.iterrows():
        step = _step(model)
        run = re.sub(r"-?step-\d+$", "", model) or model
        rows.append(
            {
                "model": model,
                "kind": "checkpoint" if step is not None else "baseline",
                "run": run if run != model else str(row.get("source", model)),
                "step": step,
                "campaign": row.get("source", ""),
            }
        )
    table = pd.DataFrame(rows)
    table["_order"] = table["kind"].map({"baseline": 0, "checkpoint": 1})
    table = table.sort_values(["_order", "step", "model"], na_position="first").drop(columns="_order")
    return table.reset_index(drop=True)


def aggregate_table(normalized: pd.DataFrame) -> pd.DataFrame:
    """One row per model, one column per headline metric, plus ``*_n`` language counts."""
    rows: dict[str, dict[str, object]] = {}
    for model, part in normalized.groupby("model", sort=False):
        row: dict[str, object] = {"model": model}
        for head in HEADLINES:
            cell = part[part["dataset"] == head.dataset]
            if head.task is not None:
                cell = cell[cell["task"] == head.task]
            if head.lang is not None:
                cell = cell[cell["lang"] == head.lang]
            cell = cell[cell["metric"] == head.metric]
            values = cell["value_num"].dropna()
            if values.empty:
                value = float("nan")
            elif head.agg == "first":
                value = float(values.iloc[0])
            else:
                value = float(values.mean())
            row[head.label] = value
            if head.agg == "mean":
                row[f"{head.label} n"] = int(values.shape[0])
        rows[model] = row
    table = pd.DataFrame(list(rows.values()))
    return table


def dataset_tab(normalized: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Model x language (or condition/task) for one dataset, every canonical metric."""
    part = normalized[normalized["dataset"] == dataset].copy()
    if part.empty:
        return pd.DataFrame()
    index = ["model", "task", "lang", "variant"] if dataset == "mcif" else ["model", "lang"]
    table = part.pivot_table(index=index, columns="metric", values="value_num", aggfunc="first")
    return table.reset_index().sort_values(index)


def coverage_table(normalized: pd.DataFrame) -> pd.DataFrame:
    """Filled / pending / missing state of every headline cell per model."""
    rows = []
    for model, part in normalized.groupby("model", sort=False):
        for head in HEADLINES:
            cell = part[(part["dataset"] == head.dataset) & (part["metric"] == head.metric)]
            if head.task is not None:
                cell = cell[cell["task"] == head.task]
            if head.lang is not None:
                cell = cell[cell["lang"] == head.lang]
            numeric = int(cell["value_num"].notna().sum())
            pending = int((cell["state"] == _PENDING).sum())
            state = "ok" if numeric else ("pending" if pending else "missing")
            rows.append({"model": model, "cell": head.label, "state": state, "values": numeric, "pending": pending})
    return pd.DataFrame(rows)


def write(long: pd.DataFrame, out: Path) -> dict[str, Path]:
    """Normalize *long* and write ``aggregated.xlsx`` plus ``aggregated_long.csv``."""
    normalized = normalize(long)
    out.mkdir(parents=True, exist_ok=True)
    xlsx = out / "aggregated.xlsx"
    tidy = out / "aggregated_long.csv"
    normalized.rename(columns={"value_num": "value"}).to_csv(tidy, index=False)

    models = models_table(normalized)
    aggregate = aggregate_table(normalized)
    aggregate = models.merge(aggregate, on="model", how="left")
    coverage = coverage_table(normalized)

    with pd.ExcelWriter(xlsx, engine="openpyxl") as book:
        aggregate.to_excel(book, sheet_name="aggregate", index=False)
        models.to_excel(book, sheet_name="models", index=False)
        coverage.to_excel(book, sheet_name="coverage", index=False)
        for key, dataset in DATASETS.items():
            tab = dataset_tab(normalized, key)
            if not tab.empty:
                tab.to_excel(book, sheet_name=dataset.sheet(), index=False)
        for sheet in book.sheets.values():
            sheet.freeze_panes = "B2"
    return {"aggregated.xlsx": xlsx, "aggregated_long.csv": tidy}


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, aggregate, and write the workbook."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--long", type=Path, required=True, help="Long results table (master_long.csv).")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    long = pd.read_csv(args.long)
    written = write(long, args.out)
    print(f"{len(long)} rows -> {len(normalize(long))} aggregated rows")
    for name, path in written.items():
        print(f"wrote {name}: {path}")


if __name__ == "__main__":
    main()
