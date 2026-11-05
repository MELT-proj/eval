"""Tests for the cross-campaign master table (``projects/master/build.py``)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

# The master table lives behind the `analysis` extra; skip rather than fail
# for a checkout installed with `[dev]` alone.
pd = pytest.importorskip("pandas")


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build = _load("master_build", "projects/master/build.py")

COLUMNS = ["model", "benchmark", "slice", "task", "dataset_id", "scorer", "metric", "value", "headline"]


def _write(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


def _row(model="m", benchmark="fleurs24-st-enx", slice_="st-fleurs-en_de", metric="corpus_bleu", value="20.0"):
    return {
        "model": model,
        "benchmark": benchmark,
        "slice": slice_,
        "task": "st",
        "dataset_id": "fleurs-en_de",
        "scorer": "st_scorer",
        "metric": metric,
        "value": value,
        "headline": "True",
    }


class TestFamily:
    def test_both_dataset_naming_conventions_share_one_family(self):
        assert build.benchmark_family("fleurs24-st-enx") == build.benchmark_family("fleurs24-st-enx-test")
        assert build.benchmark_family("voxpopuli") == build.benchmark_family("voxpopuli-test")

    def test_unknown_benchmark_is_kept_verbatim(self):
        assert build.benchmark_family("some-new-set") == "some-new-set"


class TestMerge:
    def test_earlier_source_wins_between_numbers(self, tmp_path):
        first = _write(tmp_path / "a.csv", [_row(value="20.0")])
        second = _write(tmp_path / "b.csv", [_row(value="30.0")])
        merged = build.merge([("first", first), ("second", second)])
        assert merged.loc[0, "num"] == 20.0
        assert merged.loc[0, "source"] == "first"

    def test_a_number_beats_pending_from_a_higher_priority_source(self, tmp_path):
        """A campaign that scored a metric must not be shadowed by one that declared it only."""
        pending = _write(tmp_path / "p.csv", [_row(value="pending")])
        scored = _write(tmp_path / "s.csv", [_row(value="31.5")])
        merged = build.merge([("pending-first", pending), ("scored", scored)])
        assert len(merged) == 1
        assert merged.loc[0, "num"] == 31.5
        assert merged.loc[0, "source"] == "scored"

    def test_metric_aliases_are_applied(self, tmp_path):
        path = _write(
            tmp_path / "a.csv",
            [
                _row(metric="comet_scorer", value="0.5"),
                _row(metric="mcif_official_trans_scorer", value="0.7"),
                _row(metric="", value="pending"),
            ],
        )
        merged = build.merge([("a", path)])
        assert set(merged["metric"]) == {"comet", "official_trans_comet", "score"}


class TestAtAGlance:
    def _long(self, tmp_path):
        rows = [
            # Two languages of one direction -> macro mean.
            _row(slice_="st-fleurs-en_de", value="10.0"),
            _row(slice_="st-fleurs-en_it", value="30.0"),
            # A checkpoint and a baseline.
            _row(
                model="run-step-0003000",
                benchmark="mcif",
                slice_="chunked_st-mcif-short-fixed-de",
                metric="bleu_mcif-short-fixed-de",
                value="25.0",
            ),
            _row(
                model="qwen",
                benchmark="mcif",
                slice_="chunked_st-mcif-short-fixed-de",
                metric="bleu_mcif-short-fixed-de",
                value="20.0",
            ),
        ]
        path = _write(tmp_path / "a.csv", rows)
        return build.at_a_glance(build.merge([("a", path)]))

    def test_macro_mean_over_language_slices(self, tmp_path):
        table = self._long(tmp_path)
        assert table.loc[table.model == "m", "FLEURS ST en->x BLEU"].iloc[0] == pytest.approx(20.0)

    def test_step_and_kind_come_from_the_name(self, tmp_path):
        table = self._long(tmp_path).set_index("model")
        assert table.loc["run-step-0003000", "step"] == 3000
        assert table.loc["run-step-0003000", "kind"] == "checkpoint"
        assert table.loc["run-step-0003000", "run"] == "run"
        assert table.loc["qwen", "kind"] == "baseline"

    def test_bare_step_names_stay_checkpoints(self):
        assert build._step("step-0001000") == 1000


class TestWrite:
    def test_workbook_has_the_glance_sheet_and_one_per_family(self, tmp_path):
        pytest.importorskip("openpyxl")

        path = _write(tmp_path / "a.csv", [_row(), _row(benchmark="mcif", slice_="x", metric="score", value="1")])
        long = build.merge([("a", path)])
        written = build.write(long, build.at_a_glance(long), [("a", path)], tmp_path / "out")
        sheets = pd.ExcelFile(written["master.xlsx"]).sheet_names
        assert "at_a_glance" in sheets
        assert "FLEURS-24 ST en->x" in sheets
        assert "coverage" in sheets and "sources" in sheets
