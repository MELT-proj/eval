"""Smoke tests for the ported SpeechEval plots (``projects/smurf4eu/plotting/plot_eval.py``)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plots = _load("smurf4eu_plot_eval", "projects/smurf4eu/plotting/plot_eval.py")


def _workbook(path: Path) -> Path:
    pytest.importorskip("openpyxl")
    ours = plots.CHECKPOINT_MODEL
    tabs = {
        "voxpopuli": pd.DataFrame(
            [
                {"model": ours, "lang": "en", "wer": 0.09},
                {"model": ours, "lang": "de", "wer": 0.15},
                {"model": "qwen2-audio-7b-instruct", "lang": "en", "wer": 0.13},
            ]
        ),
        "fleurs-st-xen": pd.DataFrame(
            [
                {"model": ours, "lang": "de", "comet": 0.83},
                {"model": "qwen2-audio-7b-instruct", "lang": "de", "comet": 0.81},
            ]
        ),
        "fleurs-st-enx": pd.DataFrame([{"model": ours, "lang": "de", "comet": 0.87}]),
        "librispeech": pd.DataFrame([{"model": ours, "wer_clean": 0.03, "wer_other": 0.05}]),
        "mcif": pd.DataFrame(
            [
                {"model": ours, "task": "st", "lang": "de", "variant": "short", "comet": 0.78},
                {"model": ours, "task": "st", "lang": "it", "variant": "short", "comet": 0.81},
            ]
        ),
    }
    with pd.ExcelWriter(path, engine="openpyxl") as book:
        for name, frame in tabs.items():
            frame.to_excel(book, sheet_name=name, index=False)
    return path


def test_load_results_scales_and_maps_the_checkpoint(tmp_path):
    tidy = plots.load_results(_workbook(tmp_path / "aggregated.xlsx"))
    asr = tidy[(tidy["task"] == plots.ASR) & (tidy["lang"] == "en") & (tidy["model"] == plots.OURS)]
    assert asr["model"].tolist() == [plots.OURS]
    assert asr["value"].iloc[0] == pytest.approx(9.0)
    assert set(tidy["task"]) == {plots.ASR, plots.ST_XEN, plots.ST_ENX}


def test_qwen_is_restricted_to_its_speech_languages(tmp_path):
    tidy = plots.load_results(_workbook(tmp_path / "aggregated.xlsx"))
    st_xen = tidy[(tidy["task"] == plots.ST_XEN) & (tidy["model"] == "qwen2-audio-7b-instruct")]
    assert st_xen["lang"].tolist() == ["de"]
    assert st_xen["value"].iloc[0] == pytest.approx(81.0)


def test_english_benchmarks_and_mcif_come_from_their_tabs(tmp_path):
    path = _workbook(tmp_path / "aggregated.xlsx")
    english = plots.load_english_benchmarks(path)
    assert set(english["benchmark"]) == {"LibriSpeech clean", "LibriSpeech other"}
    assert english.loc[english["benchmark"] == "LibriSpeech clean", "value"].iloc[0] == pytest.approx(3.0)
    mcif = plots.load_english_benchmarks(path, plots.MCIF_BENCHMARKS)
    assert set(mcif["benchmark"]) == {"EN→DE", "EN→IT"}
    assert mcif["value"].max() == pytest.approx(81.0)


def test_a_workbook_without_a_dataset_is_an_error(tmp_path):
    empty = tmp_path / "empty.xlsx"
    pd.DataFrame({"model": [], "lang": [], "wer": []}).to_excel(empty, sheet_name="voxpopuli", index=False)
    with pytest.raises(ValueError, match="No results found"):
        plots.load_english_benchmarks(empty)
