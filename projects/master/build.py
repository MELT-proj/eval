"""One master workbook of every finished evaluation campaign.

Each campaign already writes ``results/results_long.csv`` (one row per
``model x benchmark x slice x metric``) through ``projects/baselines/report.py``.
This script merges those files into a single workbook meant to be read at a
glance:

* ``at_a_glance`` -- one row per model (all checkpoints of the training run,
  plus every baseline) and one column per headline metric.
* one sheet per benchmark family, with every metric the scorer logged.
* ``coverage`` -- how many numeric / pending values each model has per family.
* ``sources`` -- the file every number came from.

Usage::

    python projects/master/build.py \
        --source mcif207k=.../mcif-207k/results/results_long.csv \
        --source jupiter=.../eurollm-16n-run1/results/results_long.csv \
        --out /path/to/master

**Source order is precedence.** A campaign listed first wins a
``(model, benchmark, slice, metric)`` cell over a later one. A numeric value
always wins over ``pending`` regardless of order: a campaign that actually
scored a metric must not be shadowed by one that merely declared it.

The metrics are read off each campaign's long table, so nothing here rescoring
or re-reads logs. To refresh a campaign, rebuild its table first with
``projects/baselines/report.py --log-root <campaign>/logs``.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)

#: Long-table columns every campaign writes; ``checkpoint*`` is optional.
_COLUMNS = ["model", "benchmark", "slice", "task", "dataset_id", "scorer", "metric", "value"]

#: Scaffolder metric names differ between the batch and the post-hoc scorer.
_METRIC_ALIASES = {
    # ``st_scorer`` writes ``comet``; the post-hoc path writes ``comet_scorer``.
    "comet_scorer": "comet",
    # Older MCIF ST logs name the module rather than the metric.
    "mcif_official_trans_scorer": "official_trans_comet",
}

#: Excel forbids these in sheet names and caps them at 31 characters.
_SHEET_BAD = set("[]:*?/\\")


@dataclass(frozen=True)
class Glance:
    """One column of the at-a-glance sheet."""

    label: str
    family: str
    metric: str
    #: Restrict to one slice id (``None`` pools every slice of the family).
    slice_id: str | None = None
    #: ``mean`` for per-language slices, ``first`` for one-slice metrics.
    agg: str = "mean"


#: The at-a-glance columns, in display order.
GLANCE = [
    Glance("FLEURS ST en->x BLEU", "FLEURS-24 ST en->x", "corpus_bleu"),
    Glance("FLEURS ST en->x COMET", "FLEURS-24 ST en->x", "comet"),
    Glance("FLEURS ST x->en BLEU", "FLEURS-24 ST x->en", "corpus_bleu"),
    Glance("FLEURS ST x->en COMET", "FLEURS-24 ST x->en", "comet"),
    Glance("FLEURS ASR WER", "FLEURS-24 ASR", "corpus_wer_raw"),
    Glance("VoxPopuli WER", "VoxPopuli ASR", "corpus_wer_raw"),
    Glance("LibriSpeech clean WER", "LibriSpeech ASR", "corpus_wer_raw", "asr-librispeech-test-clean", "first"),
    Glance("LibriSpeech other WER", "LibriSpeech ASR", "corpus_wer_raw", "asr-librispeech-test-other", "first"),
    Glance("MCIF ASR short WER", "MCIF", "wer_mcif-short-fixed-en", "chunked_asr-mcif-short-fixed-en", "first"),
    Glance("MCIF ASR long WER", "MCIF", "wer_mcif-long-fixed-en", "chunked_asr-mcif-long-fixed-en", "first"),
    Glance("MCIF ST de BLEU", "MCIF", "bleu_mcif-short-fixed-de", "chunked_st-mcif-short-fixed-de", "first"),
    Glance("MCIF ST de COMET", "MCIF", "official_trans_comet", "chunked_st-mcif-short-fixed-de", "first"),
    Glance("MCIF ST it BLEU", "MCIF", "bleu_mcif-short-fixed-it", "chunked_st-mcif-short-fixed-it", "first"),
    Glance("MCIF ST it COMET", "MCIF", "official_trans_comet", "chunked_st-mcif-short-fixed-it", "first"),
    Glance("MCIF QA short en", "MCIF", "official_bertscore", "audio_chat-mcif-short-fixed-en", "first"),
    Glance("MCIF QA short de", "MCIF", "official_bertscore", "audio_chat-mcif-short-fixed-de", "first"),
    Glance("MCIF QA short it", "MCIF", "official_bertscore", "audio_chat-mcif-short-fixed-it", "first"),
    Glance(
        "MCIF QA long",
        "MCIF",
        "mcif_official_qa_scorer/official_bertscore",
        "audio_chat-mcif-long-fixed-en",
        "first",
    ),
    Glance(
        "MCIF SUM long",
        "MCIF",
        "mcif_official_sum_scorer/official_bertscore",
        "audio_chat-mcif-long-fixed-en",
        "first",
    ),
    Glance("MCIF judge score", "MCIF", "score"),
    Glance("AIR-Bench acc", "AIR-Bench", "choice_accuracy"),
    Glance("CoVoST2 BLEU", "CoVoST2", "corpus_bleu"),
]


def benchmark_family(benchmark: str) -> str:
    """Group a campaign's benchmark id into the family shown on the master.

    The same benchmark is named with and without a ``-test`` suffix depending
    on which cluster produced it (``fleurs24-st-enx`` vs
    ``fleurs24-st-enx-test``); both map to one family so a model that ran on
    either cluster lands in the same column.

    Args:
        benchmark: The ``benchmark`` field of a long-table row.

    Returns:
        The family name.
    """
    for prefix, family in (
        ("fleurs24-st-enx", "FLEURS-24 ST en->x"),
        ("fleurs24-st-xen", "FLEURS-24 ST x->en"),
        ("fleurs24-asr", "FLEURS-24 ASR"),
        ("voxpopuli", "VoxPopuli ASR"),
        ("librispeech", "LibriSpeech ASR"),
        ("covost2", "CoVoST2"),
        ("air-bench", "AIR-Bench"),
    ):
        if benchmark.startswith(prefix):
            return family
    if benchmark == "mcif":
        return "MCIF"
    return benchmark


def read_source(name: str, path: Path) -> pd.DataFrame:
    """Read one campaign's long table and tag it with its source name.

    Args:
        name: Short campaign name, used as the provenance label.
        path: The ``results_long.csv`` to read.

    Returns:
        A frame with the common columns, a ``source`` column and a numeric
        ``num`` column (``NaN`` for ``pending``).
    """
    frame = pd.read_csv(path)
    missing = [c for c in ("model", "benchmark", "slice", "metric", "value") if c not in frame]
    if missing:
        raise ValueError(f"{path} is not a results_long.csv: missing {missing}")
    frame = frame[[c for c in _COLUMNS if c in frame]].copy()
    frame["source"] = name
    frame["metric"] = frame["metric"].fillna("score").replace(_METRIC_ALIASES)
    frame["value"] = frame["value"].astype(str)
    frame["num"] = pd.to_numeric(frame["value"], errors="coerce")
    frame["family"] = frame["benchmark"].map(benchmark_family)
    frame["_rank"] = 0
    return frame


def merge(sources: list[tuple[str, Path]]) -> pd.DataFrame:
    """Merge campaigns, keeping one row per (model, benchmark, slice, metric).

    A numeric value beats ``pending`` regardless of source order; among
    numeric candidates, and among pending candidates, the earlier source wins.

    Args:
        sources: ``(name, path)`` pairs, highest precedence first.

    Returns:
        The merged long frame, without the internal rank column.
    """
    frames = []
    for rank, (name, path) in enumerate(sources):
        frame = read_source(name, path)
        frame["_rank"] = rank
        frames.append(frame)
        logger.info("%-24s %6d rows  %2d models  %s", name, len(frame), frame["model"].nunique(), path)
    merged = pd.concat(frames, ignore_index=True)
    merged["_has"] = merged["num"].notna()
    merged = (
        merged.sort_values(["_has", "_rank"], ascending=[False, True])
        .drop_duplicates(["model", "benchmark", "slice", "metric"], keep="first")
        .drop(columns=["_has", "_rank"])
        .reset_index(drop=True)
    )
    return merged


def _step(model: str) -> int | None:
    """Training step encoded in a checkpoint's model name, or ``None``.

    Matches both ``eurollm-16n-run1-step-0207000`` and the bare
    ``step-0001000`` naming an earlier campaign used.
    """
    match = re.search(r"(?:^|-)step-(\d+)$", model)
    return int(match.group(1)) if match else None


def at_a_glance(long: pd.DataFrame) -> pd.DataFrame:
    """One row per model and one column per headline metric.

    Args:
        long: The merged long frame.

    Returns:
        A frame sorted baselines first, then checkpoints by training step.
    """
    rows: dict[str, dict[str, object]] = {}
    for model, part in long.groupby("model", sort=False):
        rows[model] = {"model": model}
        for glance in GLANCE:
            cell = part[(part["family"] == glance.family) & (part["metric"] == glance.metric)]
            if glance.slice_id is not None:
                cell = cell[cell["slice"] == glance.slice_id]
            values = cell["num"].dropna()
            if values.empty:
                rows[model][glance.label] = float("nan")
            elif glance.agg == "first":
                rows[model][glance.label] = float(values.iloc[0])
            else:
                rows[model][glance.label] = float(values.mean())
        pending = part.loc[part["value"] == "pending", ["family", "slice", "metric"]]
        rows[model]["pending"] = "; ".join(sorted({f"{r.family}:{r.metric}" for r in pending.itertuples()}))

    table = pd.DataFrame(list(rows.values()))
    table["step"] = pd.array([_step(m) for m in table["model"]], dtype="Int64")
    table["kind"] = table["step"].map(lambda s: "baseline" if pd.isna(s) else "checkpoint")
    table["run"] = table["model"].map(lambda m: re.sub(r"-?step-\d+$", "", m) or m)
    campaign_of = long.drop_duplicates("model").set_index("model")["source"]
    table["campaign"] = table["model"].map(campaign_of)
    # A bare ``step-0001000`` carries no run prefix; the campaign names the run.
    bare = table["run"] == table["model"]
    table.loc[bare, "run"] = table.loc[bare, "campaign"]
    front = ["model", "kind", "run", "step", "campaign"]
    table = table[front + [c.label for c in GLANCE] + ["pending"]]
    table["_order"] = table["kind"].map({"baseline": 0, "checkpoint": 1})
    table = table.sort_values(["_order", "step", "model"], na_position="first").drop(columns="_order")
    return table.reset_index(drop=True)


def _sheet_name(family: str) -> str:
    """Excel-safe sheet name for a benchmark family."""
    return "".join(c for c in family if c not in _SHEET_BAD)[:31]


def write(long: pd.DataFrame, glance: pd.DataFrame, sources: list[tuple[str, Path]], out: Path) -> dict[str, Path]:
    """Write the workbook and its two CSV companions into *out*."""
    out.mkdir(parents=True, exist_ok=True)
    written = {
        "master.xlsx": out / "master.xlsx",
        "master_at_a_glance.csv": out / "master_at_a_glance.csv",
        "master_long.csv": out / "master_long.csv",
    }
    glance.to_csv(written["master_at_a_glance.csv"], index=False)
    long.drop(columns=["num"]).to_csv(written["master_long.csv"], index=False)

    coverage = (
        long.groupby(["model", "family"])
        .agg(
            slices_with_value=("slice", "nunique"),
            values=("num", "count"),
            pending=("value", lambda s: int((s == "pending").sum())),
        )
        .reset_index()
        .sort_values(["model", "family"])
    )
    provenance = pd.DataFrame(
        [
            {
                "source": name,
                "path": str(path),
                "modified": pd.Timestamp(path.stat().st_mtime, unit="s").isoformat(timespec="seconds"),
                "rows": len(pd.read_csv(path, usecols=["model"])),
            }
            for name, path in sources
        ]
    )

    with pd.ExcelWriter(written["master.xlsx"], engine="openpyxl") as xlsx:
        glance.to_excel(xlsx, sheet_name="at_a_glance", index=False)
        for family in sorted(long["family"].unique()):
            part = long[long["family"] == family]
            table = part.pivot_table(
                index="model", columns=["slice", "metric"], values="num", aggfunc="first"
            ).sort_index()
            # A flat "slice / metric" header is what a reader can sort and filter;
            # a two-row MultiIndex header is not.
            table.columns = [f"{slice_id} / {metric}" for slice_id, metric in table.columns]
            table.to_excel(xlsx, sheet_name=_sheet_name(family))
        coverage.to_excel(xlsx, sheet_name="coverage", index=False)
        provenance.to_excel(xlsx, sheet_name="sources", index=False)
        for sheet in xlsx.sheets.values():
            sheet.freeze_panes = "B2"
    return written


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, merge the campaigns, and write the master tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Campaign results_long.csv; repeatable, highest precedence first.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sources = []
    for item in args.source:
        if "=" not in item:
            raise SystemExit(f"--source expects NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        sources.append((name, Path(path)))

    long = merge(sources)
    glance = at_a_glance(long)
    written = write(long, glance, sources, args.out)
    print(glance.to_string(index=False, float_format="{:.4f}".format))
    print()
    for name, path in written.items():
        print(f"wrote {name}: {path}")


if __name__ == "__main__":
    main()
