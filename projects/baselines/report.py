"""Baseline matrix report: one table of every model on every benchmark slice, to CSV and Excel.

    python projects/baselines/report.py --log-root <dir> [--log-root <dir> ...] --out report-out

Reads the logs ``scripts/run_matrix.py`` produced and writes:

* ``results_long.csv``: one row per (model, benchmark, slice, metric).
* ``results.xlsx``: a ``summary`` sheet (models x headline metric per slice),
  one sheet per benchmark with every metric, and a ``runs`` sheet listing
  which log each number came from.  It includes, for a model run with
  ``truncate_long_audio``, how many of its samples were cut to its window.

Every number is read off the log header. Only successful logs count; when a
slice has more than one, the newest wins and the others are reported.

A judge-graded slice (``audio_chat``) generated with ``--no-score`` and not
graded yet appears as ``pending`` rather than being left out, so a missing
grade is visible instead of looking like a missing run.
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log, read_eval_log_samples


logger = logging.getLogger(__name__)

#: Metric(s) shown on the summary sheet for each task, as fnmatch patterns
#: (the chunked scorers name theirs after the dataset_id). Every metric the log
#: carries still goes to the per-benchmark sheets and the long CSV.
HEADLINE = {
    # As it stands, then after stripping a "The transcription is: '...'" wrapper,
    # and how often that applied (see melteval.scorers.strip_transcript_preamble).
    "asr": ["corpus_wer", "corpus_wer_extracted", "preamble_rate"],
    "st": ["corpus_bleu"],
    # Same three for MCIF, per dataset_id: wer_<id>, wer_extracted_<id>.
    "chunked_asr": ["wer_*", "preamble_rate"],
    "chunked_st": ["bleu_*"],
    "audio_mcq": ["choice_accuracy"],
    # A judge's grade (AIR-Bench Chat) or MCIF's official BERTScore (QA/SUM).
    "audio_chat": ["accuracy", "*bertscore*"],
}

#: Value written for a slice that ran but has no scores yet.
PENDING = "pending"


def discover(roots: list[str | Path]) -> list[Path]:
    """Every inspect log under *roots*, recursively."""
    paths: list[Path] = []
    for root in roots:
        root = Path(root)
        paths += sorted([*root.rglob("*.eval"), *root.rglob("*.json")])
    return [p for p in paths if p.name != "logs.json"]


def read_runs(roots: list[str | Path]) -> pd.DataFrame:
    """One row per successful log: who, what, and where from."""
    rows = []
    for path in discover(roots):
        try:
            log = read_eval_log(str(path), header_only=True)
        except Exception as exc:  # noqa: BLE001 - not every .json is a log
            logger.warning("skipping unreadable %s: %s", path, exc)
            continue
        if log.status != "success":
            logger.warning("skipping %s: status %s", path, log.status)
            continue
        meta = log.eval.metadata or {}
        task_args = log.eval.task_args or {}
        task = meta.get("melteval_task") or task_args.get("task_filter") or ""
        dataset_id = task_args.get("dataset_id") or ""
        rows.append(
            {
                "model": meta.get("melteval_model") or log.eval.model,
                "benchmark": meta.get("melteval_benchmark") or _benchmark_from_args(task_args),
                "slice": meta.get("melteval_slice") or (f"{task}-{dataset_id}" if dataset_id else task),
                "task": task,
                "dataset_id": dataset_id,
                "provider_model": log.eval.model,
                "created": log.eval.created,
                # `created` has one-second resolution; the file's mtime breaks ties.
                "_mtime": path.stat().st_mtime_ns,
                "samples": (log.results.completed_samples if log.results else None),
                "scored": bool(log.results and log.results.scores),
                "truncated_samples": _count_truncated(path, log),
                "log_path": str(path),
                "_log": log,
            }
        )
    runs = pd.DataFrame(rows)
    if runs.empty:
        return runs

    runs = runs.sort_values(["created", "_mtime"])
    for run in runs[runs["truncated_samples"].fillna(0) > 0].to_dict("records"):
        logger.warning(
            "%s × %s/%s: %d sample(s) had their audio truncated to the model's window (see the runs sheet)",
            run["model"], run["benchmark"], run["slice"], run["truncated_samples"],
        )
    duplicated = runs.duplicated(["model", "benchmark", "slice"], keep="last")
    for path in runs.loc[duplicated, "log_path"]:
        logger.warning("superseded by a newer log for the same slice: %s", path)
    return runs[~duplicated].reset_index(drop=True)


def _count_truncated(path: Path, log) -> int | None:
    """How many samples had their audio cut to the model's window, or None if the run could not cut any.

    Only a run started with ``truncate_long_audio`` can have cut anything (see
    ``melteval/providers/hf.py``), so only those logs are read sample by sample;
    every other log stays a header-only read.
    """
    if str((log.eval.model_args or {}).get("truncate_long_audio", "")).lower() not in ("true", "1", "yes"):
        return None
    return sum(
        1
        for sample in read_eval_log_samples(str(path), all_samples_required=False)
        if sample.output and (sample.output.metadata or {}).get("audio_truncated_from_seconds")
    )


def _benchmark_from_args(task_args: dict) -> str:
    """Name a log's benchmark from its spec or frozen set when metadata lacks it."""
    source = task_args.get("spec") or task_args.get("frozen_set") or ""
    return Path(str(source)).stem or "unknown"


def metrics_long(runs: pd.DataFrame) -> pd.DataFrame:
    """Every metric of every run, one row each; unscored runs get one ``pending`` row."""
    rows = []
    for run in runs.to_dict("records"):
        base = {k: run[k] for k in ("model", "benchmark", "slice", "task", "dataset_id")}
        log = run["_log"]
        if not run["scored"]:
            rows.append({**base, "scorer": "", "metric": "", "value": PENDING, "headline": True})
            continue
        patterns = HEADLINE.get(run["task"], [])
        # MCIF's long-track logs carry the same metric (BERTScore) from two
        # scorers, QA and SUM; name those by scorer so neither hides the other.
        seen: dict[str, int] = {}
        for score in log.results.scores:
            for name in score.metrics:
                seen[name] = seen.get(name, 0) + 1
        for score in log.results.scores:
            for name, metric in score.metrics.items():
                label = f"{score.name.split('/')[-1]}/{name}" if seen[name] > 1 else name
                rows.append(
                    {
                        **base,
                        "scorer": score.name,
                        "metric": label,
                        "value": metric.value,
                        "headline": any(fnmatch.fnmatch(name, p) for p in patterns),
                    }
                )
    return pd.DataFrame(rows)


def summary_table(long: pd.DataFrame) -> pd.DataFrame:
    """Models × (benchmark, slice, metric), headline metrics only."""
    head = long[long["headline"]].copy()
    head["metric"] = head["metric"].replace("", "score")
    return head.pivot_table(
        index="model", columns=["benchmark", "slice", "metric"], values="value", aggfunc="first"
    ).sort_index(axis=1)


def benchmark_table(long: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """Models × (slice, metric) for one benchmark, every metric."""
    part = long[long["benchmark"] == benchmark].copy()
    part["metric"] = part["metric"].replace("", "score")
    return part.pivot_table(index="model", columns=["slice", "metric"], values="value", aggfunc="first").sort_index(
        axis=1
    )


def write(runs: pd.DataFrame, long: pd.DataFrame, out: Path) -> dict[str, Path]:
    """Write the CSV and the workbook into *out*."""
    out.mkdir(parents=True, exist_ok=True)
    written = {"results_long.csv": out / "results_long.csv", "results.xlsx": out / "results.xlsx"}
    long.to_csv(written["results_long.csv"], index=False)

    with pd.ExcelWriter(written["results.xlsx"], engine="openpyxl") as xlsx:
        summary_table(long).to_excel(xlsx, sheet_name="summary")
        for benchmark in sorted(long["benchmark"].unique()):
            # Excel caps sheet names at 31 characters and forbids a few symbols.
            name = "".join(c for c in benchmark if c not in "[]:*?/\\")[:31]
            benchmark_table(long, benchmark).to_excel(xlsx, sheet_name=name)
        runs.drop(columns=["_log", "_mtime"]).to_excel(xlsx, sheet_name="runs", index=False)
    return written


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, build the tables, write them, and print the summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-root", action="append", required=True, help="Directory of logs (repeatable).")
    parser.add_argument("--out", type=Path, default=Path("report-out"), help="Output directory.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    runs = read_runs(args.log_root)
    if runs.empty:
        raise SystemExit(f"No successful logs under {args.log_root}.")
    long = metrics_long(runs)
    written = write(runs, long, args.out)

    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.4f}".format):
        print(summary_table(long).T.to_string())
    print()
    for path in written.values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
