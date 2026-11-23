"""Tests for extracting COMET/MetricX rescoring triples from an eval log.

`read_eval_log` is monkeypatched rather than run for real: producing a real
`.eval` log needs a live model, and what's under test here is triple
extraction, not `inspect eval` itself.
"""

import json

import pytest
from inspect_ai.log import EvalError
from inspect_ai.model import ModelOutput

from melteval.rescore import extract_triples, write_triples


class _Sample:
    """Minimal stand-in for `EvalSample` -- only the fields extract_triples reads."""

    def __init__(self, id, target, completion, source_text=None, error=None):
        self.id = id
        self.target = target
        self.output = ModelOutput.from_content(model="melt/x", content=completion) if completion else None
        self.metadata = {"source_text": source_text} if source_text is not None else {}
        self.error = error


def _log(samples):
    class _Log:
        pass

    log = _Log()
    log.samples = samples
    return log


@pytest.fixture
def patched_log(monkeypatch):
    """Route `extract_triples`'s `read_eval_log` call to a fake log."""
    import inspect_ai.log as log_module

    holder = {}

    def fake_read_eval_log(path, resolve_attachments=False):
        holder["path"] = path
        return holder["log"]

    monkeypatch.setattr(log_module, "read_eval_log", fake_read_eval_log)

    def _set(samples):
        holder["log"] = _log(samples)

    return _set


class TestExtractTriples:
    def test_usable_sample_produces_a_triple(self, patched_log):
        patched_log([_Sample("00-0", target="Ich gehe.", completion="I go.", source_text="Ich gehe.")])
        triples, stats = extract_triples("fake.eval")
        assert triples == [{"sample_id": "00-0", "src": "Ich gehe.", "mt": "I go.", "ref": "Ich gehe."}]
        assert stats["usable"] == 1

    def test_missing_source_text_is_skipped_and_counted(self, patched_log):
        patched_log([_Sample("00-0", target="a", completion="b", source_text=None)])
        triples, stats = extract_triples("fake.eval")
        assert triples == []
        assert stats["skipped_no_source_text"] == 1

    def test_errored_sample_is_skipped_and_counted(self, patched_log):
        patched_log(
            [_Sample("00-0", target="a", completion="b", source_text="c", error=EvalError(message="boom", traceback="", traceback_ansi=""))]
        )
        triples, stats = extract_triples("fake.eval")
        assert triples == []
        assert stats["skipped_error"] == 1

    def test_empty_completion_is_skipped_and_counted(self, patched_log):
        patched_log([_Sample("00-0", target="a", completion="   ", source_text="c")])
        triples, stats = extract_triples("fake.eval")
        assert triples == []
        assert stats["skipped_no_completion"] == 1

    def test_list_target_uses_the_first_reference(self, patched_log):
        patched_log([_Sample("00-0", target=["first", "second"], completion="b", source_text="c")])
        triples, _ = extract_triples("fake.eval")
        assert triples[0]["ref"] == "first"

    def test_mixed_batch_counts_each_outcome_independently(self, patched_log):
        patched_log(
            [
                _Sample("00-0", target="a", completion="b", source_text="c"),  # usable
                _Sample("00-1", target="a", completion="b", source_text=None),  # no source_text
                _Sample("00-2", target="a", completion="", source_text="c"),  # no completion
            ]
        )
        triples, stats = extract_triples("fake.eval")
        assert len(triples) == 1
        assert stats == {
            "total_samples": 3,
            "usable": 1,
            "skipped_error": 0,
            "skipped_no_source_text": 1,
            "skipped_no_completion": 1,
        }

    def test_empty_log_produces_no_triples_not_an_error(self, patched_log):
        patched_log([])
        triples, stats = extract_triples("fake.eval")
        assert triples == []
        assert stats["total_samples"] == 0


class TestWriteTriples:
    def test_round_trips_through_jsonl(self, tmp_path):
        triples = [
            {"sample_id": "00-0", "src": "a", "mt": "b", "ref": "c"},
            {"sample_id": "00-1", "src": "d", "mt": "e", "ref": "f"},
        ]
        out = tmp_path / "triples.jsonl"
        write_triples(triples, out)
        lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        assert lines == triples

    def test_creates_parent_directories(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "triples.jsonl"
        write_triples([{"sample_id": "k", "src": "a", "mt": "b", "ref": "c"}], out)
        assert out.exists()

    def test_non_ascii_is_not_escaped(self, tmp_path):
        out = tmp_path / "triples.jsonl"
        write_triples([{"sample_id": "k", "src": "أهلاً", "mt": "hello", "ref": "hi"}], out)
        assert "أهلاً" in out.read_text(encoding="utf-8")


class TestCometScorerThroughRealInspect:
    """``comet_scorer`` under a real ``inspect_ai.eval()``: the triples must survive inspect's score
    reduction (they live in ``Score.metadata``) and reach COMET once, as one corpus."""

    def test_comet_scores_the_corpus_and_reports_what_it_could_not_score(self, tmp_path, monkeypatch):
        import sys
        import types

        from inspect_ai import Task
        from inspect_ai import eval as inspect_eval
        from inspect_ai.dataset import MemoryDataset, Sample

        from melteval.rescore import comet_scorer

        seen = []
        fake = types.ModuleType("comet")
        fake.download_model = lambda name: f"/models/{name}"

        class _Model:
            def predict(self, samples, batch_size, gpus):
                seen.append(samples)
                return types.SimpleNamespace(system_score=0.75)

        fake.load_from_checkpoint = lambda path: _Model()
        monkeypatch.setitem(sys.modules, "comet", fake)

        dataset = MemoryDataset(
            [
                Sample(input="x", target="I go.", id="a", metadata={"source_text": "Ich gehe."}),
                Sample(input="x", target="Good day.", id="b", metadata={"source_text": "Guten Tag."}),
                # ga in FLEURS: no source transcript, so COMET cannot score it.
                Sample(input="x", target="Hello.", id="c", metadata={}),
            ]
        )
        outputs = [ModelOutput.from_content(model="mockllm/model", content=c) for c in ("I go.", "Good day.", "Hi.")]
        [log] = inspect_eval(
            Task(dataset=dataset, scorer=comet_scorer()),
            model="mockllm/model",
            model_args={"custom_outputs": outputs},
            display="none",
            log_dir=str(tmp_path),
        )

        assert log.status == "success"
        metrics = log.results.scores[0].metrics
        assert metrics["comet"].value == pytest.approx(0.75)
        assert metrics["comet_coverage"].value == pytest.approx(2 / 3)
        [samples] = seen
        assert sorted(samples, key=lambda t: t["src"]) == [
            {"src": "Guten Tag.", "mt": "Good day.", "ref": "Good day."},
            {"src": "Ich gehe.", "mt": "I go.", "ref": "I go."},
        ]

    def test_non_english_source_and_target_pass_through_unchanged(self, tmp_path, monkeypatch):
        """EuroparlST fr->pl: neither side is English; the triple is (source transcript, hypothesis, reference)."""
        import sys
        import types

        from inspect_ai import Task
        from inspect_ai import eval as inspect_eval
        from inspect_ai.dataset import MemoryDataset, Sample

        from melteval.rescore import comet_scorer

        seen = []
        fake = types.ModuleType("comet")
        fake.download_model = lambda name: name

        class _Model:
            def predict(self, samples, batch_size, gpus):
                seen.append(samples)
                return types.SimpleNamespace(system_score=0.8)

        fake.load_from_checkpoint = lambda path: _Model()
        monkeypatch.setitem(sys.modules, "comet", fake)

        src = "Madame la Présidente, je voudrais qu ' elle réponde."
        ref = "Pani Przewodnicząca, chciałbym, aby odpowiedziała. Zażółć gęślą jaźń."
        dataset = MemoryDataset([Sample(input="x", target=ref, id="a", metadata={"source_text": src})])
        outputs = [ModelOutput.from_content(model="mockllm/model", content="Pani Przewodnicząca, proszę.")]
        [log] = inspect_eval(
            Task(dataset=dataset, scorer=comet_scorer()),
            model="mockllm/model",
            model_args={"custom_outputs": outputs},
            display="none",
            log_dir=str(tmp_path),
        )
        assert log.status == "success"
        assert seen == [[{"src": src, "mt": "Pani Przewodnicząca, proszę.", "ref": ref}]]


class TestMetricxScorerThroughRealInspect:
    """``metricx_scorer`` under a real ``inspect_ai.eval()``, with the model replaced by a recorder."""

    def test_per_sample_values_mean_and_source_coverage(self, tmp_path, monkeypatch):
        from inspect_ai import Task
        from inspect_ai import eval as inspect_eval
        from inspect_ai.dataset import MemoryDataset, Sample

        import melteval.rescore as rescore

        seen = []
        monkeypatch.setattr(rescore, "metricx_predict", lambda inputs: seen.extend(inputs) or [2.0])
        dataset = MemoryDataset(
            [
                Sample(input="x", target="Pani Przewodnicząca.", id="a", metadata={"source_text": "Madame la Présidente."}),
                # no source transcript: still scored, with an empty source
                Sample(input="x", target="Guten Tag.", id="b", metadata={}),
                Sample(input="x", target="Tschüss.", id="c", metadata={"source_text": "Au revoir."}),
                # empty output: not scorable
                Sample(input="x", target="Danke.", id="d", metadata={"source_text": "Merci."}),
            ]
        )
        contents = ("Pani Przewodnicząca.", "Guten Tag.", "Tschüss.", "")
        outputs = [ModelOutput.from_content(model="mockllm/model", content=c) for c in contents]
        [log] = inspect_eval(
            Task(dataset=dataset, scorer=rescore.metricx_scorer()),
            model="mockllm/model",
            model_args={"custom_outputs": outputs},
            display="none",
            log_dir=str(tmp_path),
        )
        assert log.status == "success"
        metrics = log.results.scores[0].metrics
        assert metrics["metricx"].value == pytest.approx(2.0)
        assert metrics["metricx_coverage"].value == pytest.approx(3 / 4)
        assert metrics["metricx_source_coverage"].value == pytest.approx(3 / 4)
        assert len(seen) == 3
        assert sorted(seen) == sorted(
            [
                "source: Madame la Présidente. candidate: Pani Przewodnicząca. reference: Pani Przewodnicząca.",
                "source:  candidate: Guten Tag. reference: Guten Tag.",
                "source: Au revoir. candidate: Tschüss. reference: Tschüss.",
            ]
        )
        values = {s.id: s.scores["metricx_scorer"].value for s in log.samples}
        assert values["a"] == pytest.approx(2.0)
        assert values["d"] == -1.0  # no completion: the out-of-range UNSCORED sentinel


class TestMetricxSourceManifest:
    """A frozen-set manifest fills the source where the sample metadata has none."""

    def _manifest(self, tmp_path, records):
        from melteval.manifest import MANIFEST_NAME

        path = tmp_path / "set"
        path.mkdir()
        with open(path / MANIFEST_NAME, "w") as fh:
            fh.writelines(r.to_json() + "\n" for r in records)
        return str(path)

    def _record(self, key, dataset_id, source_text, src_lang="fr", tgt_lang="en"):
        from melteval.manifest import AudioLocator, EvalRecord

        return EvalRecord(
            sample_key=key,
            task="st",
            target="ref",
            audio=AudioLocator(kind="shar", params={}),
            lang=tgt_lang,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            dataset_id=dataset_id,
            source_text=source_text,
        )

    def _run(self, tmp_path, monkeypatch, scorer, samples):
        from inspect_ai import Task
        from inspect_ai import eval as inspect_eval
        from inspect_ai.dataset import MemoryDataset

        import melteval.rescore as rescore

        monkeypatch.setattr(rescore, "metricx_predict", lambda inputs: [1.0])
        outputs = [ModelOutput.from_content(model="mockllm/model", content="out") for _ in samples]
        [log] = inspect_eval(
            Task(dataset=MemoryDataset(samples), scorer=scorer),
            model="mockllm/model",
            model_args={"custom_outputs": outputs},
            display="none",
            log_dir=str(tmp_path),
        )
        return log

    def test_manifest_fills_a_missing_source(self, tmp_path, monkeypatch):
        from inspect_ai.dataset import Sample

        import melteval.rescore as rescore

        manifest = self._manifest(tmp_path, [self._record("00-000000", "fleurs-fr_en-dev", "Bonjour.")])
        scorer = rescore.metricx_scorer(source_manifest=manifest)
        log = self._run(
            tmp_path,
            monkeypatch,
            scorer,
            [
                Sample(
                    input="x",
                    target="ref",
                    id="00-000000",
                    metadata={"dataset_id": "fleurs-fr_en-dev", "src_lang": "fr", "tgt_lang": "en"},
                )
            ],
        )
        assert log.results.scores[0].metrics["metricx_source_coverage"].value == 1.0
        meta = log.samples[0].scores["metricx_scorer"].metadata
        assert meta["metricx_triple"]["src"] == "Bonjour."
        assert meta["metricx_source"] == "manifest"

    def test_same_key_under_another_dataset_is_not_used(self, tmp_path, monkeypatch):
        from inspect_ai.dataset import Sample

        import melteval.rescore as rescore

        manifest = self._manifest(tmp_path, [self._record("00-000000", "fleurs-es_en-dev", "Hola.")])
        scorer = rescore.metricx_scorer(source_manifest=manifest)
        log = self._run(
            tmp_path,
            monkeypatch,
            scorer,
            [
                Sample(
                    input="x",
                    target="ref",
                    id="00-000000",
                    metadata={"dataset_id": "fleurs-fr_en-dev", "src_lang": "fr", "tgt_lang": "en"},
                )
            ],
        )
        assert log.results.scores[0].metrics["metricx_source_coverage"].value == 0.0

    def test_manifest_language_mismatch_fails_the_eval(self, tmp_path, monkeypatch):
        from inspect_ai.dataset import Sample

        import melteval.rescore as rescore

        manifest = self._manifest(
            tmp_path, [self._record("00-000000", "fleurs-fr_en-dev", "Bonjour.", src_lang="de")]
        )
        scorer = rescore.metricx_scorer(source_manifest=manifest)
        log = self._run(
            tmp_path,
            monkeypatch,
            scorer,
            [
                Sample(
                    input="x",
                    target="ref",
                    id="00-000000",
                    metadata={"dataset_id": "fleurs-fr_en-dev", "src_lang": "fr", "tgt_lang": "en"},
                )
            ],
        )
        assert log.status == "error"
        assert "src_lang" in str(log.error.message)

    def test_env_var_is_the_fallback(self, tmp_path, monkeypatch):
        from inspect_ai.dataset import Sample

        import melteval.rescore as rescore

        manifest = self._manifest(tmp_path, [self._record("00-000000", "fleurs-fr_en-dev", "Bonjour.")])
        monkeypatch.setenv(rescore.SOURCE_MANIFEST_ENV, manifest)
        scorer = rescore.metricx_scorer()
        log = self._run(
            tmp_path,
            monkeypatch,
            scorer,
            [
                Sample(
                    input="x",
                    target="ref",
                    id="00-000000",
                    metadata={"dataset_id": "fleurs-fr_en-dev", "src_lang": "fr", "tgt_lang": "en"},
                )
            ],
        )
        assert log.results.scores[0].metrics["metricx_source_coverage"].value == 1.0


def test_metricx_scorer_is_registered_by_the_package_import():
    """The ``inspect_ai`` entry point imports ``melteval``; that alone must register every post-hoc scorer.

    ``inspect score`` with no ``--scorer`` rebuilds a log's scorers by bare name, so a scorer whose
    module only gets imported by that lookup cannot be found. A fresh interpreter, not this test
    process (which imports ``melteval.rescore`` at the top), is the only honest check.
    """
    import subprocess
    import sys

    code = (
        "from inspect_ai._util.registry import registry_lookup\n"
        "import melteval\n"
        "assert registry_lookup('scorer', 'st_scorer') is not None\n"
        "assert registry_lookup('scorer', 'metricx_scorer') is not None, 'metricx_scorer not registered'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
