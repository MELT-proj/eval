#!/usr/bin/env python3
"""Duration distribution of a frozen set, pooled and per dataset_id.

Usage:
    python scripts/frozen_length_stats.py <frozen-dir> [--json OUT]

Reads ``<frozen-dir>/manifest.jsonl``, skips records without a duration, and
reports n, mean, median, p90, p95, min, max and the fraction above ``--over``
(default 40 s, the SMURF training window). Percentiles are nearest-rank.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


#: Column names and widths of the printed table.
COLUMNS = (("dataset", 32), ("n", 5), ("mean", 7), ("median", 7), ("p90", 7), ("p95", 7), ("min", 6), ("max", 7),
           ("over40", 6))  # fmt: skip


def load_durations(manifest: Path) -> list[tuple[str, float]]:
    """Return ``(dataset_id, duration_s)`` for every record that has a duration."""
    out: list[tuple[str, float]] = []
    with manifest.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            duration = record.get("duration")
            if duration is None:
                continue
            out.append((str(record.get("dataset_id") or ""), float(duration)))
    return out


def nearest_rank(values: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already sorted list."""
    rank = max(1, math.ceil(pct / 100.0 * len(values)))
    return values[rank - 1]


def summarize(values: list[float], over_s: float) -> dict[str, float]:
    """The summary row for one group of durations."""
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "over40": 0.0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 2),
        "median": round(statistics.median(ordered), 2),
        "p90": round(nearest_rank(ordered, 90), 2),
        "p95": round(nearest_rank(ordered, 95), 2),
        "min": round(ordered[0], 2),
        "max": round(ordered[-1], 2),
        "over40": round(sum(v > over_s for v in ordered) / len(ordered), 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path, help="frozen-set directory containing manifest.jsonl")
    parser.add_argument("--json", type=Path, help="write the table as JSON here")
    parser.add_argument("--over", type=float, default=40.0, help="over-limit threshold in seconds")
    args = parser.parse_args(argv)

    rows = load_durations(args.frozen / "manifest.jsonl")
    if not rows:
        sys.exit(f"{args.frozen}: no durations in manifest.jsonl")

    by_dataset: dict[str, list[float]] = {}
    for dataset_id, duration in rows:
        by_dataset.setdefault(dataset_id, []).append(duration)
    table = {
        "over_s": args.over,
        "pooled": summarize([d for _, d in rows], args.over),
        "by_dataset": {k: summarize(v, args.over) for k, v in sorted(by_dataset.items())},
    }

    print(" ".join(f"{name:<{width}}" for name, width in COLUMNS))
    for name, row in [("pooled", table["pooled"]), *table["by_dataset"].items()]:
        cells = (
            f"{name:<32}",
            f"{row['n']:>5d}",
            f"{row['mean']:>7.2f}",
            f"{row['median']:>7.2f}",
            f"{row['p90']:>7.2f}",
            f"{row['p95']:>7.2f}",
            f"{row['min']:>6.2f}",
            f"{row['max']:>7.2f}",
            f"{row['over40']:>6.2%}",
        )
        print(" ".join(cells))
    if args.json:
        args.json.write_text(json.dumps(table, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
