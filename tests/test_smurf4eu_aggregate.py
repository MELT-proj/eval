"""Tests for the SMURF aggregation of the long results table (``projects/smurf4eu/aggregate.py``)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

# The aggregation lives behind the `analysis` extra; skip rather than fail
# for a checkout installed with `[dev]` alone.
pd = pytest.importorskip("pandas")


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


aggregate = _load("smurf4eu_aggregate", "projects/smurf4eu/aggregate.py")

COLUMNS = ["model", "benchmark", "slice", "task", "dataset_id", "scorer", "metric", "value", "source"]


def _row(benchmark, slice_, metric, value, model="m"):
    return {
        "model": model,
        "benchmark": benchmark,
        "slice": slice_,
        "task": "x",
        "dataset_id": "",
        "scorer": "x",
        "metric": metric,
        "value": value,
        "source": "test",
    }


class TestClassify:
    def test_fleurs_direction_and_language(self):
        assert aggregate._classify("fleurs24-st-enx", "st-fleurs-en_de") == ("fleurs-st-enx", None, "de", None)
        assert aggregate._classify("fleurs24-st-xen-test", "st-fleurs-de_en") == ("fleurs-st-xen", None, "de", None)

    def test_librispeech_condition_and_test_variant(self):
        assert aggregate._classify("librispeech", "asr-librispeech-clean") == ("librispeech", None, "clean", None)
        assert aggregate._classify("librispeech", "asr-librispeech-test-other") == (
            "librispeech",
            None,
            "other",
            None,
        )

    def test_voxpopuli_and_fleurs_asr(self):
        assert aggregate._classify("voxpopuli-test", "asr-cs") == ("voxpopuli", None, "cs", None)
        assert aggregate._classify("fleurs24-asr-test", "asr-en") == ("fleurs-asr", None, "en", None)

    def test_one_log_asr_slice_has_no_language_yet(self):
        assert aggregate._classify("fleurs24-asr-test", "asr") == ("fleurs-asr", None, None, None)

    def test_mcif_tasks_carry_language_and_variant(self):
        assert aggregate._classify("mcif", "chunked_st-mcif-short-fixed-de") == ("mcif", "st", "de", "short")
        assert aggregate._classify("mcif", "audio_chat-mcif-short-fixed-it") == ("mcif", "qa", "it", "short")

    def test_out_of_scope_benchmark_is_not_guessed(self):
        assert aggregate._classify("air-bench", "audio_mcq-air-bench-foundation-speech") is None


class TestNormalize:
    def test_benchmark_suffix_and_corpus_metric_are_collapsed(self, tmp_path):
        long = pd.DataFrame(
            [
                _row("fleurs24-st-enx-test", "st-fleurs-en_de", "corpus_bleu", 30.0),
                _row("fleurs24-st-enx-test", "st-fleurs-en_de", "bleu_de", 30.0),
                _row("fleurs24-st-enx-test", "st-fleurs-en_de", "comet", 0.9),
            ],
            columns=COLUMNS,
        )
        normalized = aggregate.normalize(long)
        assert set(normalized["metric"]) == {"bleu", "comet"}
        assert (normalized["dataset"] == "fleurs-st-enx").all()

    def test_a_number_beats_pending(self, tmp_path):
        long = pd.DataFrame(
            [
                _row("fleurs24-st-enx", "st-fleurs-en_de", "comet", "pending"),
                _row("fleurs24-st-xen", "st-fleurs-de_en", "comet", 0.8),
            ],
            columns=COLUMNS,
        )
        normalized = aggregate.normalize(long)
        assert "pending" in set(normalized["state"])

    def test_one_log_asr_gets_its_language_from_the_metric(self):
        long = pd.DataFrame(
            [
                _row("fleurs24-asr-test", "asr", "wer_de", 0.20),
                _row("fleurs24-asr-test", "asr", "cer_de", 0.10),
                _row("fleurs24-asr-test", "asr", "corpus_wer_raw", 0.20),
            ],
            columns=COLUMNS,
        )
        normalized = aggregate.normalize(long)
        assert set(normalized["metric"]) == {"wer", "cer"}
        assert set(normalized["lang"]) == {"de"}

    def test_unknown_slices_are_dropped(self):
        long = pd.DataFrame(
            [
                _row("air-bench", "audio_mcq-x", "accuracy", 0.5),
                _row("fleurs24-st-enx", "st-fleurs-en_de", "comet", 0.9),
            ],
            columns=COLUMNS,
        )
        assert len(aggregate.normalize(long)) == 1

    def test_missing_columns_are_an_error(self):
        with pytest.raises(ValueError, match="missing columns"):
            aggregate.normalize(pd.DataFrame({"model": ["m"]}))


class TestTables:
    def _normalized(self):
        long = pd.DataFrame(
            [
                _row("voxpopuli", "asr-voxpopuli-en", "wer_en", 0.10),
                _row("voxpopuli", "asr-voxpopuli-de", "wer_de", 0.20),
                _row("voxpopuli", "asr-voxpopuli-en", "cer_en", 0.05),
                _row("voxpopuli", "asr-voxpopuli-de", "cer_de", 0.10),
                _row("mcif", "audio_chat-mcif-short-fixed-en", "official_bertscore", 0.25),
            ],
            columns=COLUMNS,
        )
        return aggregate.normalize(long)

    def test_aggregate_is_a_macro_mean_with_a_language_count(self):
        table = aggregate.aggregate_table(self._normalized())
        assert table.loc[0, "VoxPopuli WER"] == pytest.approx(0.15)
        assert table.loc[0, "VoxPopuli WER n"] == 2
        assert table.loc[0, "MCIF SQA en BERTScore"] == pytest.approx(0.25)

    def test_missing_headline_is_nan_not_zero(self):
        table = aggregate.aggregate_table(self._normalized())
        assert pd.isna(table.loc[0, "FLEURS En->X COMET"])

    def test_coverage_distinguishes_filled_pending_missing(self):
        long = pd.DataFrame(
            [
                _row("fleurs24-st-enx", "st-fleurs-en_de", "comet", "pending"),
                _row("voxpopuli", "asr-voxpopuli-en", "wer_en", 0.1, model="other"),
            ],
            columns=COLUMNS,
        )
        coverage = aggregate.coverage_table(aggregate.normalize(long)).set_index(["model", "cell"])
        assert coverage.loc["m", "FLEURS En->X COMET"].state == "pending"
        assert coverage.loc["m", "VoxPopuli WER"].state == "missing"
        assert coverage.loc["other", "VoxPopuli WER"].state == "ok"

    def test_models_table_marks_checkpoints(self):
        long = pd.DataFrame(
            [_row("voxpopuli", "asr-voxpopuli-en", "wer_en", 0.1, model="run-step-0003000")],
            columns=COLUMNS,
        )
        models = aggregate.models_table(aggregate.normalize(long)).set_index("model")
        assert models.loc["run-step-0003000", "kind"] == "checkpoint"
        assert models.loc["run-step-0003000", "step"] == 3000
        assert models.loc["run-step-0003000", "run"] == "run"


class TestWrite:
    def test_workbook_has_the_expected_tabs(self, tmp_path):
        pytest.importorskip("openpyxl")
        long = pd.DataFrame(
            [
                _row("fleurs24-st-enx", "st-fleurs-en_de", "comet", 0.9),
                _row("voxpopuli", "asr-voxpopuli-en", "wer_en", 0.1),
                _row("mcif", "chunked_st-mcif-short-fixed-de", "official_trans_comet", 0.8),
            ],
            columns=COLUMNS,
        )
        written = aggregate.write(long, tmp_path / "out")
        sheets = pd.ExcelFile(written["aggregated.xlsx"]).sheet_names
        assert {"aggregate", "models", "coverage", "fleurs-st-enx", "voxpopuli", "mcif"} <= set(sheets)
        assert written["aggregated_long.csv"].exists()
