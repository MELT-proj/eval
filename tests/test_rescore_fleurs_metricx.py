"""Tests for the FLEURS MetricX source backfill driver (``scripts/rescore_fleurs_metricx.py``).

The driver is loaded the way ``tests/test_evaluate.py`` loads the launcher. Its
fixture is a minimal campaign: one suite, one benchmark, one slice log and the
frozen set that log's samples should get their source from.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


driver = _load("rescore_fleurs_metricx", "scripts/rescore_fleurs_metricx.py")


def _log(samples, metrics, scorer_names=("st_scorer", "metricx_scorer"), rescored=False):
    metricx_metrics = {
        "metricx": {"name": "metricx", "value": metrics["metricx"]},
        "metricx_coverage": {"name": "metricx_coverage", "value": 1.0},
    }
    if rescored:
        metricx_metrics["metricx_source_coverage"] = {"name": "metricx_source_coverage", "value": 1.0}
    return {
        "eval": {"scorers": [{"name": n, "options": {}} for n in scorer_names]},
        "results": {
            "scores": [
                {"name": "st_scorer", "metrics": {"corpus_bleu": {"name": "corpus_bleu", "value": metrics["bleu"]}}},
                {"name": "metricx_scorer", "metrics": metricx_metrics},
            ]
        },
        "samples": samples,
    }


def _manifest_record(key, dataset_id, source_text, cut_id="1"):
    return {
        "sample_key": key,
        "task": "st",
        "target": "ref",
        "audio": {"kind": "shar", "params": {}},
        "lang": "en",
        "src_lang": "fr",
        "tgt_lang": "en",
        "dataset_id": dataset_id,
        "source": "fleurs/fr_fr/validation",
        "source_text": source_text,
        "cut_id": cut_id,
    }


@pytest.fixture
def campaign(tmp_path):
    """A campaign config + suite + one slice log + the frozen set it refers to."""
    set_dir = tmp_path / "set"
    set_dir.mkdir()
    records = [
        _manifest_record("00-000000", "fleurs-fr_en-dev", "Bonjour.", "1"),
        _manifest_record("00-000001", "fleurs-fr_en-dev", "Salut.", "2"),
    ]
    (set_dir / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")

    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "benchmarks:\n"
        "  - name: fleurs24-st-xen-dev\n"
        f"    frozen_set: {set_dir}\n"
        "    post_venv: /venvs/metrics/activate\n"
        "    slices:\n"
        "      - {task: st, dataset_id: fleurs-fr_en-dev}\n"
    )

    out = tmp_path / "out"
    log_dir = out / "logs" / "model-a" / "fleurs24-st-xen-dev" / "st-fleurs-fr_en-dev"
    log_dir.mkdir(parents=True)
    log = _log(
        samples=[
            {
                "id": "00-000000",
                "metadata": {"dataset_id": "fleurs-fr_en-dev"},
                "output": {"completion": "a"},
                "scores": {
                    "metricx_scorer": {
                        "value": 2.0,
                        "metadata": {"metricx_triple": {"src": "", "mt": "a", "ref": "a"}},
                    }
                },
            },
            {
                "id": "00-000001",
                "metadata": {"dataset_id": "fleurs-fr_en-dev"},
                "output": {"completion": "b"},
                "scores": {
                    "metricx_scorer": {
                        "value": 4.0,
                        "metadata": {"metricx_triple": {"src": "", "mt": "b", "ref": "b"}},
                    }
                },
            },
        ],
        metrics={"bleu": 10.0, "metricx": 3.0},
    )
    log_path = log_dir / "2026-01-01T00-00-00_task.json"
    log_path.write_text(json.dumps(log))
    (log_dir / "2026-01-01T00-00-00_task.json.superseded").write_text("{}")

    config = tmp_path / "config.yaml"
    config.write_text(
        "name: speechval-test\n"
        f"output_dir: {out}\n"
        f"benchmarks: {suite}\n"
    )
    return {"config": config, "out": out, "log": log_path, "set": set_dir, "log_dir": log_dir}


def _ref(campaign, model="model-a", benchmark="fleurs24-st-xen-dev", slice="st-fleurs-fr_en-dev"):
    return driver.LogRef("speechval-test", model, benchmark, slice, campaign["log"])


class TestDiscover:
    def test_finds_only_current_logs_of_the_requested_benchmarks(self, campaign):
        other = campaign["out"] / "logs" / "model-a" / "voxpopuli-dev" / "st-voxpopuli-en-dev"
        other.mkdir(parents=True)
        (other / "x.json").write_text("{}")

        refs = driver.discover(campaign["out"], "speechval-test", set(driver.FLEURS_BENCHMARKS))

        assert [(r.campaign, r.model, r.benchmark, r.slice) for r in refs] == [
            ("speechval-test", "model-a", "fleurs24-st-xen-dev", "st-fleurs-fr_en-dev")
        ]
        assert refs[0].path == campaign["log"]


class TestCoverage:
    def test_is_one_when_every_sample_resolves(self, campaign):
        refs = [_ref(campaign)]
        rows = driver.coverage_rows(refs, {"fleurs24-st-xen-dev": campaign["set"]})
        assert rows[0]["samples"] == 2
        assert rows[0]["with_source"] == 2
        assert rows[0]["coverage"] == 1.0

    def test_uses_dataset_id_as_well_as_sample_id(self, campaign):
        """The same sample key under another dataset must not count as a source."""
        campaign["set"].joinpath("manifest.jsonl").write_text(
            json.dumps(_manifest_record("00-000000", "fleurs-es_en-dev", "Hola.")) + "\n"
        )
        with pytest.raises(driver.CoverageError) as caught:
            driver.coverage_rows([_ref(campaign)], {"fleurs24-st-xen-dev": campaign["set"]})
        assert caught.value.rows[0]["coverage"] == 0.0
        assert "st-fleurs-fr_en-dev" in str(caught.value)

    def test_raises_below_one(self, campaign):
        campaign["set"].joinpath("manifest.jsonl").write_text(
            json.dumps(_manifest_record("00-000000", "fleurs-fr_en-dev", "Bonjour.")) + "\n"
        )
        with pytest.raises(driver.CoverageError) as caught:
            driver.coverage_rows([_ref(campaign)], {"fleurs24-st-xen-dev": campaign["set"]})
        assert caught.value.rows[0]["with_source"] == 1
        assert caught.value.rows[0]["samples"] == 2


class TestSnapshot:
    def test_records_old_metricx_bleu_and_triples(self, campaign):
        rows, triples = driver.snapshot([_ref(campaign)])
        assert rows == [
            {
                "campaign": "speechval-test",
                "model": "model-a",
                "benchmark": "fleurs24-st-xen-dev",
                "slice": "st-fleurs-fr_en-dev",
                "metricx": 3.0,
                "metricx_coverage": 1.0,
                "corpus_bleu": 10.0,
                "corpus_spbleu": None,
                "corpus_chrf": None,
            }
        ]
        assert [(t["sample_id"], t["src"], t["mt"]) for t in triples] == [("00-000000", "", "a"), ("00-000001", "", "b")]


class TestAlreadyRescored:
    def test_detects_the_new_metric(self, campaign):
        assert driver.already_rescored(campaign["log"]) is False
        campaign["log"].write_text(json.dumps(_log([], {"bleu": 1.0, "metricx": 1.0}, rescored=True)))
        assert driver.already_rescored(campaign["log"]) is True


class TestDryRun:
    def test_prints_one_command_per_group_with_the_manifest(self, campaign, monkeypatch, capsys):
        monkeypatch.setattr(driver.evaluate, "load_site_env", lambda site: None)

        driver.main(["--campaign", str(campaign["config"]), "--dry-run"])

        out = capsys.readouterr().out
        assert out.count("submit_score.sh") == 1
        assert str(campaign["set"]) in out
        assert "--action overwrite" in out
        assert "--scorer" not in out
        results = campaign["out"] / "results"
        assert (results / "fleurs-source-coverage.csv").exists()
        assert (results / "metricx-before.csv").exists()
        assert (results / "metricx-before.jsonl").exists()
