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
