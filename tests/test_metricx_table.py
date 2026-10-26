"""Tests for the MetricX / chrF++ table builder (``projects/metricx/table.py``): ranking and rendering only.

chrF++ itself comes from ``melteval.st_metrics`` and is covered by its own tests; here it is injected.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "projects" / "metricx" / "table.py"
    spec = importlib.util.spec_from_file_location("metricx_table", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


table = _load()
score = sys.modules["score"]  # table.py imports its sibling module under this name


def _summary(ref_lc: dict[str, float], chrfpp: dict[str, float]) -> dict:
    """A summary over languages ``ref_lc`` (MetricX ref lc per language), with chrF++ injected per language."""
    rows = []
    for lang, value in ref_lc.items():
        rows.append({"src_lang": lang, "source_missing": False, "metricx_ref_lc": value, "metricx_qe_lc": value + 1})
    summary = score.summarize(rows, "src_lang")
    summary["provenance"] = {"model": "m", "model_revision": "0123456789", "n_samples": len(rows)}
    for entry in summary["languages"]:
        entry["chrfpp_lc"] = chrfpp[entry["lang"]]
    for scope in (summary["macro_mean"], summary["macro_mean_with_source"], *summary["tiers"].values()):
        scope["chrfpp_lc"] = None
    summary["macro_mean"]["chrfpp_lc"] = sum(chrfpp.values()) / len(chrfpp)
    summary["macro_mean_with_source"]["chrfpp_lc"] = sum(chrfpp.values()) / len(chrfpp)
    return summary


def test_kendall_tau_is_one_for_the_same_order_and_minus_one_for_the_reverse():
    assert table.kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert table.kendall_tau([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0
    assert table.kendall_tau([1, 2, 3], [1, 3, 2]) == pytest.approx(1 / 3)


def test_ranking_respects_direction():
    values = {"a": 3.0, "b": 1.0, "c": 2.0}
    assert table.ranking(values, lower_is_better=True) == ["b", "c", "a"]
    assert table.ranking(values, lower_is_better=False) == ["a", "c", "b"]


def test_render_reports_agreement_between_metricx_and_chrfpp():
    runs = {
        "fleurs24-st-xen-dev": {
            "good": _summary({"es": 2.0, "de": 3.0}, {"es": 60.0, "de": 50.0}),
            "bad": _summary({"es": 9.0, "de": 11.0}, {"es": 20.0, "de": 10.0}),
        }
    }
    markdown, long_rows = table.render(runs)
    assert "ranking by MetricX ref (lc): good > bad" in markdown
    assert "ranking by chrF++ (lc): good > bad" in markdown
    assert "Kendall tau, MetricX ref (lc) vs chrF++ (backbone order): +1.00" in markdown
    assert "| es | 9.00 | 2.00 |" in markdown  # per-language rows, models in sorted order: bad, good
    macro = next(
        r
        for r in long_rows
        if (r["model"], r["kind"], r["group"], r["metric"]) == ("bad", "macro", "macro_mean", "metricx_ref_lc")
    )
    assert macro["value"] == pytest.approx(10.0)
    assert {r["metric"] for r in long_rows} >= {"metricx_ref_lc", "chrfpp_lc"}
