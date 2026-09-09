"""Tests for the campaign report's aggregation (``projects/melt/campaign.py``).

Only the pure table logic is covered here: log reading needs real `.eval` files,
and `report.py` already asserts against the log headers every time it runs,
which is the stronger check for that half.

``projects/`` is not a package (it is analysis code that reads eval logs, not
part of `melteval`), so the module is loaded by path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


# The report tooling lives behind the `analysis` extra; skip rather than fail
# for a checkout installed with `[dev]` alone.
pd = pytest.importorskip("pandas")


def _load_campaign():
    path = Path(__file__).resolve().parents[1] / "projects" / "melt" / "campaign.py"
    spec = importlib.util.spec_from_file_location("melt_campaign", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


campaign = _load_campaign()


def _asr_row(model: str, step: int, corpus: str, lang: str, metric: str, errors: float, total: float) -> dict:
    return {
        "run_name": "run-s1337-8g",
        "model": model,
        "seed": 1337,
        "checkpoint": "final" if step == 0 else f"checkpoint-{step}",
        "step": step,
        "is_final": step == 0,
        "task": "asr",
        "dataset_id": corpus,
        "corpus": corpus,
        "lang": lang,
        "src_lang": "",
        "tgt_lang": "",
        "direction": "",
        "n_samples": 10,
        "hours": 1.0,
        "log_path": f"/logs/{model}-{corpus}-{lang}.eval",
        "metric": metric,
        "value": errors / total,
        "numerator": errors,
        "denominator": total,
    }


class TestModelPath:
    def test_checkpoint_path_splits_into_run_and_step(self):
        assert campaign._parse_model_path("melt//gpfs/out/IFT-700-s1337-8g/checkpoint-902") == (
            "IFT-700-s1337-8g",
            "checkpoint-902",
            902,
            False,
        )

    def test_run_root_is_the_final_checkpoint(self):
        run, checkpoint, _, is_final = campaign._parse_model_path("melt//gpfs/out/IFT-700-s1337-8g")
        assert (run, checkpoint, is_final) == ("IFT-700-s1337-8g", campaign.FINAL, True)

    def test_provider_prefix_is_optional(self):
        assert campaign._parse_model_path("/gpfs/out/run/checkpoint-5")[2] == 5


class TestCorpusOf:
    def test_direction_suffix_is_stripped_from_an_st_corpus(self):
        assert campaign._corpus_of("covost2-pt_en", "pt_en") == "covost2"

    def test_asr_corpus_is_left_alone(self):
        assert campaign._corpus_of("librispeech-clean", "") == "librispeech-clean"

    def test_a_corpus_that_merely_ends_in_the_direction_text_is_not_truncated(self):
        # No `-` boundary, so nothing to strip: `_corpus_of` matches the
        # separator, not just the tail.
        assert campaign._corpus_of("covost2pt_en", "pt_en") == "covost2pt_en"


class TestPoolAsr:
    def test_pooled_wer_sums_counts_rather_than_averaging_rates(self):
        # A long cell at 10% and a short one at 50%: the corpus rate is 15%,
        # while the mean of the rates would be 30%. Getting this wrong is the
        # length bias corpus WER exists to avoid, so it is asserted on numbers
        # far enough apart that rounding cannot hide it.
        frame = pd.DataFrame(
            [
                _asr_row("m", 100, "big", "en", "wer", errors=90, total=900),
                _asr_row("m", 100, "small", "en", "wer", errors=50, total=100),
            ]
        )
        pooled = campaign.pool_asr(frame)
        overall = pooled[pooled["corpus"] == campaign.OVERALL]
        assert len(overall) == 1
        assert overall["value"].iloc[0] == pytest.approx(140 / 1000)

    def test_pooling_is_independent_of_how_the_cells_were_split_across_logs(self):
        # The campaign submits one job per (task, corpus, language), so a
        # checkpoint's cells arrive from several logs. The pooled number must
        # not depend on that: it is the same corpus rate either way.
        cells = [
            _asr_row("m", 100, "a", "en", "wer", errors=10, total=100),
            _asr_row("m", 100, "b", "de", "wer", errors=30, total=300),
            _asr_row("m", 100, "c", "fr", "wer", errors=5, total=50),
        ]
        split = pd.DataFrame(cells)
        merged = pd.DataFrame([{**c, "log_path": "/logs/one-big-job.eval"} for c in cells])
        assert campaign.pool_asr(split)[lambda d: d["corpus"] == campaign.OVERALL]["value"].iloc[0] == (
            campaign.pool_asr(merged)[lambda d: d["corpus"] == campaign.OVERALL]["value"].iloc[0]
        )

    def test_st_rows_are_left_out_of_the_pool(self):
        # BLEU has no denominator to pool over, which is why the ST summary is
        # a macro average; a pooled BLEU row here would look like a corpus
        # score and would not be one.
        st = {**_asr_row("m", 100, "covost2", "en", "bleu", 1, 1), "task": "st"}
        st["numerator"] = st["denominator"] = float("nan")
        frame = pd.DataFrame([_asr_row("m", 100, "a", "en", "wer", 10, 100), st])
        pooled = campaign.pool_asr(frame)
        assert set(pooled.loc[pooled["corpus"] == campaign.OVERALL, "metric"]) == {"wer"}


class TestPlaceFinal:
    def test_final_lands_one_median_gap_past_the_last_checkpoint(self):
        frame = pd.DataFrame(
            [
                _asr_row("m-ckpt0100", 100, "a", "en", "wer", 1, 10),
                _asr_row("m-ckpt0200", 200, "a", "en", "wer", 1, 10),
                _asr_row("m", 0, "a", "en", "wer", 1, 10),
            ]
        )
        placed = campaign._place_final(frame)
        assert placed.loc[placed["is_final"], "step"].iloc[0] == 300

    def test_a_run_with_only_a_final_checkpoint_has_no_scale_to_place_it_on(self):
        frame = pd.DataFrame([_asr_row("m", 0, "a", "en", "wer", 1, 10)])
        assert campaign._place_final(frame)["step"].iloc[0] == 0


class TestTables:
    def test_asr_table_is_one_row_per_model_with_a_column_per_cell(self):
        frame = campaign.pool_asr(
            pd.DataFrame(
                [
                    _asr_row("m", 100, "voxpopuli", "it", "wer", 30, 100),
                    _asr_row("m", 100, "voxpopuli", "it", "cer", 10, 100),
                    _asr_row("m", 100, "mls_sidon", "it", "wer", 20, 100),
                    _asr_row("m", 100, "mls_sidon", "it", "cer", 5, 100),
                ]
            )
        )
        table = campaign.asr_table(frame)
        assert len(table) == 1
        assert table["wer/voxpopuli/it"].iloc[0] == pytest.approx(0.30)
        assert table["wer/OVERALL/all"].iloc[0] == pytest.approx(0.25)
        # The seed has to survive into the spreadsheet: averaging repeats of one
        # recipe over initialisations is what it is carried for.
        assert table["seed"].iloc[0] == 1337

    def test_st_table_macro_averages_the_directions(self):
        rows = []
        for direction, bleu in (("de_en", 10.0), ("fr_en", 20.0)):
            rows.append(
                {
                    **_asr_row("m", 100, "covost2", "en", "bleu", 1, 1),
                    "task": "st",
                    "direction": direction,
                    "value": bleu,
                    "numerator": float("nan"),
                    "denominator": float("nan"),
                }
            )
        table = campaign.st_table(pd.DataFrame(rows))
        assert table["bleu/covost2/de_en"].iloc[0] == 10.0
        assert table["bleu/MACRO-AVG"].iloc[0] == pytest.approx(15.0)
