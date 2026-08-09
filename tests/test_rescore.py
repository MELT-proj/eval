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
