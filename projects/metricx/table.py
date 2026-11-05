"""Backbone x set tables of MetricX-24 next to chrF++, from ``score.py`` outputs.

Runs in the ``melteval`` venv (it calls :mod:`melteval.st_metrics` for chrF++, which needs ``inspect_ai`` and
the staged SentencePiece model, ``$SACREBLEU``), not in the ``metricx`` venv. chrF++ is computed from the same
log a summary was scored from, lowercased and grouped like the existing ST tables (X->en per source language,
en->X per target language, plain mean over languages): ``melteval st-metrics --lowercase``.

It writes ``table.md`` (macro-means and tiers per backbone, per-language tables, and whether the backbone ranking
agrees between metrics) and ``table_long.csv`` (one row per model, set, group and metric) into ``--out``. It does
not touch ``results_master.csv``; the long CSV is meant to be handed to whoever owns that file.

Usage::

    python projects/metricx/table.py --root OUT --out OUT/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))  # projects/ is not a package: import the sibling module

from score import TIERS  # noqa: E402


#: (key in summary.json, header, lower is better). chrF++ is higher-better and comes from st-metrics.
METRICS = (
    ("metricx_ref_lc", "MetricX ref (lc)", True),
    ("metricx_ref", "MetricX ref (cased)", True),
    ("metricx_qe_lc", "MetricX QE (lc)", True),
    ("metricx_qe", "MetricX QE (cased)", True),
    ("chrfpp_lc", "chrF++ (lc)", False),
)
TIER_ORDER = ("high", "mid", "low")


def kendall_tau(a: list[float], b: list[float]) -> float:
    """Kendall's tau-a between two equally long score lists; ``1`` means the same ranking."""
    pairs = list(combinations(range(len(a)), 2))
    concordant = sum((a[i] - a[j]) * (b[i] - b[j]) > 0 for i, j in pairs)
    discordant = sum((a[i] - a[j]) * (b[i] - b[j]) < 0 for i, j in pairs)
    return (concordant - discordant) / len(pairs)


def ranking(values: dict[str, float], lower_is_better: bool) -> list[str]:
    """Model names from best to worst."""
    return sorted(values, key=lambda m: values[m], reverse=not lower_is_better)


def load_runs(root: Path) -> dict[str, dict[str, dict]]:
    """``{set: {model: summary}}`` from ``<root>/<model>/<set>/summary.json``."""
    runs: dict[str, dict[str, dict]] = {}
    for path in sorted(root.glob("*/*/summary.json")):
        runs.setdefault(path.parent.name, {})[path.parent.parent.name] = json.loads(path.read_text())
    return runs


def add_chrfpp(summary: dict) -> dict:
    """Attach lowercased chrF++ per language to *summary*, scored from the log it came from."""
    from melteval.st_metrics import score_log

    axis = summary["axis"]
    result = score_log(summary["provenance"]["log"], group_by=axis, lowercase=True)
    by_lang = {g["group"]: g["chrfpp"] for g in result["groups"]}
    for entry in summary["languages"]:
        entry["chrfpp_lc"] = by_lang[entry["lang"]]
    for scope in (summary["macro_mean"], summary["macro_mean_with_source"], *summary["tiers"].values()):
        scope["chrfpp_lc"] = None
    langs = {e["lang"]: e for e in summary["languages"]}
    summary["macro_mean"]["chrfpp_lc"] = sum(by_lang.values()) / len(by_lang)
    complete = [e["chrfpp_lc"] for e in summary["languages"] if not e["n_source_missing"]]
    summary["macro_mean_with_source"]["chrfpp_lc"] = sum(complete) / len(complete)
    for name, members in TIERS.items():
        values = [langs[lang]["chrfpp_lc"] for lang in members if lang in langs]
        summary["tiers"][name]["chrfpp_lc"] = sum(values) / len(values) if values else None
    return summary


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render(runs: dict[str, dict[str, dict]]) -> tuple[str, list[dict]]:
    """Markdown for every set, and the long-format rows behind it."""
    lines: list[str] = []
    long_rows: list[dict] = []
    for dataset, models in sorted(runs.items()):
        names = sorted(models)
        lines += [f"## {dataset}", ""]
        provenance = next(iter(models.values()))["provenance"]
        lines += [
            f"MetricX-24 `{provenance['model']}` @ `{provenance['model_revision'][:7]}`; error score 0-25, "
            "lower is better. chrF++ is higher-better. `lc` = source, hypothesis and reference lowercased. "
            f"{provenance['n_samples']} samples per model.",
            "",
        ]

        for scope_key, scope_name in (("macro_mean_with_source", "macro, languages with a source"),
                                      ("macro_mean", "macro, all languages")):  # fmt: skip
            header = "| backbone | " + " | ".join(h for _, h, _ in METRICS) + " | " + " | ".join(TIER_ORDER) + " |"
            lines += [
                f"**{scope_name}** (tiers: MetricX ref lc)",
                "",
                header,
                "|---|" + "---:|" * (len(METRICS) + len(TIER_ORDER)),
            ]
            for name in names:
                s = models[name]
                cells = [_fmt(s[scope_key].get(key)) for key, _, _ in METRICS]
                tiers = [_fmt(s["tiers"][t].get("metricx_ref_lc")) for t in TIER_ORDER]
                lines.append(f"| {name} | " + " | ".join(cells + tiers) + " |")
            lines.append("")

        for key, header, lower in METRICS:
            macro = {n: models[n]["macro_mean_with_source"].get(key) for n in names}
            if any(v is None for v in macro.values()):
                continue
            order = ranking(macro, lower)
            lines.append(f"- ranking by {header}: " + " > ".join(order))
        reference = {n: models[n]["macro_mean_with_source"]["chrfpp_lc"] for n in names}
        for key, header, lower in METRICS[:-1]:
            macro = {n: models[n]["macro_mean_with_source"].get(key) for n in names}
            if any(v is None for v in macro.values()):
                continue
            # chrF++ is higher-better, MetricX lower-better: negate MetricX so that "agree" means tau > 0.
            tau = kendall_tau([-macro[n] for n in names], [reference[n] for n in names])
            lines.append(f"- Kendall tau, {header} vs chrF++ (backbone order): {tau:+.2f}")
        lines.append("")

        languages = sorted(
            {e["lang"] for m in models.values() for e in m["languages"]}, key=lambda lang: (_tier_index(lang), lang)
        )
        for key, header, _ in METRICS:
            lines += [f"**{dataset}: {header} per language**", "", "| lang | " + " | ".join(names) + " |",
                      "|---|" + "---:|" * len(names)]  # fmt: skip
            for lang in languages:
                cells = []
                for name in names:
                    entry = next((e for e in models[name]["languages"] if e["lang"] == lang), None)
                    cells.append(_fmt(entry.get(key) if entry else None))
                lines.append(f"| {lang} | " + " | ".join(cells) + " |")
            lines.append("")

        for name in names:
            s = models[name]
            for key, _, _ in METRICS:
                for entry in s["languages"]:
                    long_rows.append(_long(name, dataset, "language", entry["lang"], key, entry.get(key), entry["n"]))
                for tier in TIER_ORDER:
                    long_rows.append(_long(name, dataset, "tier", tier, key, s["tiers"][tier].get(key), None))
                for scope in ("macro_mean", "macro_mean_with_source"):
                    long_rows.append(_long(name, dataset, "macro", scope, key, s[scope].get(key), None))
    return "\n".join(lines) + "\n", long_rows


def _tier_index(lang: str) -> int:
    return next((i for i, members in enumerate(TIERS.values()) if lang in members), len(TIERS))


def _long(model: str, dataset: str, kind: str, group: str, metric: str, value: float | None, n: int | None) -> dict:
    return {"model": model, "set": dataset, "kind": kind, "group": group, "metric": metric, "value": value, "n": n}


def main(argv: list[str] | None = None) -> int:
    """Command line: load every summary under ``--root``, add chrF++, write the tables to ``--out``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path, help="score.py output root")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    runs = load_runs(args.root)
    if not runs:
        parser.error(f"no <model>/<set>/summary.json under {args.root}")
    for models in runs.values():
        for summary in models.values():
            add_chrfpp(summary)
    markdown, long_rows = render(runs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "table.md").write_text(markdown, encoding="utf-8")
    with open(args.out / "table_long.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "set", "kind", "group", "metric", "value", "n"])
        writer.writeheader()
        writer.writerows(long_rows)
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
