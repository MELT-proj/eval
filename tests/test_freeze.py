"""Tests for freeze orchestration and the shar reader's resolution rules.

The shar reader's *reference resolution* is the part worth guarding hardest: a
mistake there does not crash, it scores every sample of a corpus against the
wrong string and reports a plausible-looking number.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from melteval.freeze import freeze, load_spec, spec_hash
from melteval.manifest import MANIFEST_NAME, SUMMARY_NAME, AudioLocator, EvalRecord, read_manifest
from melteval.readers import base as reader_base
from melteval.readers.base import SourceResult


class FakeReader:
    """Stands in for a real corpus so freeze orchestration can be tested alone."""

    source_type = "fake"
    locator_kind = "fake"

    def __init__(self, per_source: dict[int, list[EvalRecord]] | None = None):
        self.per_source = per_source or {}

    def freeze(self, source_cfg, source_index):
        records = self.per_source.get(source_index, [])
        for ordinal, record in enumerate(records):
            record.sample_key = f"{source_index:02d}-{ordinal:06d}"
        return SourceResult(
            records=records,
            stats={"read": len(records), "kept": len(records), "hours": 0.0},
        )

    def load_audio(self, locator):  # pragma: no cover - not exercised here
        raise NotImplementedError


def _record(target="hi", **overrides) -> EvalRecord:
    defaults = {
        "sample_key": "",
        "task": "asr",
        "target": target,
        "audio": AudioLocator(kind="fake", params={}),
        "lang": "en",
        "duration": 1.0,
    }
    defaults.update(overrides)
    return EvalRecord(**defaults)


@pytest.fixture
def fake_reader(monkeypatch):
    """Install a fake reader and return a setter for its records."""
    reader = FakeReader()
    monkeypatch.setattr(reader_base, "get_reader", lambda _t: reader)
    monkeypatch.setattr("melteval.freeze.get_reader", lambda _t: reader)
    return reader


class TestFreeze:
    def test_writes_manifest_and_summary(self, tmp_path: Path, fake_reader):
        fake_reader.per_source = {0: [_record("a"), _record("b")]}
        spec = {"name": "t", "input_cfg": [{"type": "fake"}]}

        summary = freeze(spec, tmp_path)

        assert summary["total_samples"] == 2
        assert (tmp_path / MANIFEST_NAME).exists()
        assert json.loads((tmp_path / SUMMARY_NAME).read_text())["name"] == "t"

    def test_sample_keys_are_unique_across_sources(self, tmp_path: Path, fake_reader):
        """Two sources both starting at ordinal 0 must not collide.

        FLEURS cut IDs repeat within and across languages, so keys are assigned
        by the freezer rather than taken from the corpus.
        """
        fake_reader.per_source = {
            0: [_record("a", cut_id="2003"), _record("b", cut_id="2004")],
            1: [_record("c", cut_id="2003"), _record("d", cut_id="2004")],
        }
        spec = {"input_cfg": [{"type": "fake"}, {"type": "fake"}]}

        freeze(spec, tmp_path)

        records = read_manifest(tmp_path / MANIFEST_NAME)
        assert len(records) == 4
        assert len({r.sample_key for r in records}) == 4
        assert len({r.cut_id for r in records}) == 2  # the collision is real

    def test_duplicate_keys_are_rejected(self, tmp_path: Path, monkeypatch):
        """A broken key scheme must fail loudly, not merge samples silently."""

        class CollidingReader(FakeReader):
            def freeze(self, source_cfg, source_index):
                records = [_record("a"), _record("b")]
                for record in records:
                    record.sample_key = "same"
                return SourceResult(records=records, stats={})

        monkeypatch.setattr("melteval.freeze.get_reader", lambda _t: CollidingReader())
        with pytest.raises(ValueError, match="Duplicate sample_key"):
            freeze({"input_cfg": [{"type": "fake"}]}, tmp_path)

    def test_empty_result_raises(self, tmp_path: Path, fake_reader):
        """An eval set of zero samples is a bug report, not a valid artefact."""
        fake_reader.per_source = {}
        with pytest.raises(ValueError, match="No source yielded"):
            freeze({"input_cfg": [{"type": "fake"}]}, tmp_path)

    def test_spec_without_sources_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="no `input_cfg`"):
            freeze({"name": "x"}, tmp_path)

    def test_summary_counts_tasks_and_languages(self, tmp_path: Path, fake_reader):
        fake_reader.per_source = {
            0: [_record("a", task="asr", lang="en"), _record("b", task="st", lang="de")]
        }
        summary = freeze({"input_cfg": [{"type": "fake"}]}, tmp_path)
        assert summary["tasks"] == {"asr": 1, "st": 1}
        assert summary["languages"] == {"de": 1, "en": 1}

    def test_hours_not_tokens(self, tmp_path: Path, fake_reader):
        """Budgeting is in audio hours; num_tokens is absent from most sources."""
        fake_reader.per_source = {0: [_record("a", duration=1800.0), _record("b", duration=1800.0)]}
        assert freeze({"input_cfg": [{"type": "fake"}]}, tmp_path)["total_hours"] == 1.0


class TestSpecLoading:
    def test_expands_plain_and_omegaconf_env_syntax(self, tmp_path: Path, monkeypatch):
        """Specs get copied out of training configs, which use ${oc.env:VAR}."""
        monkeypatch.setenv("DATA_ROOT", "/corpora")
        path = tmp_path / "spec.yaml"
        path.write_text(
            "input_cfg:\n"
            "  - shar_path: ${DATA_ROOT}/a/test\n"
            "  - shar_path: ${oc.env:DATA_ROOT}/b/test\n"
        )
        spec = load_spec(path)
        assert [s["shar_path"] for s in spec["input_cfg"]] == ["/corpora/a/test", "/corpora/b/test"]

    def test_hash_is_stable_and_sensitive(self):
        a = {"input_cfg": [{"shar_path": "/x", "max_samples": 10}]}
        b = {"input_cfg": [{"max_samples": 10, "shar_path": "/x"}]}
        c = {"input_cfg": [{"shar_path": "/x", "max_samples": 11}]}
        assert spec_hash(a) == spec_hash(b)
        assert spec_hash(a) != spec_hash(c)


class TestTextFieldResolution:
    """The precedence chain that decides what a sample is scored against."""

    @staticmethod
    def _cut(tags=None):
        return SimpleNamespace(tags=tags)

    def test_per_cut_tag_overrides_source_default(self):
        from melteval.readers.shar import effective_text_field

        cut = self._cut(tags={"text_field": "custom.metadata.sentence"})
        assert effective_text_field(cut, "text") == "custom.metadata.sentence"

    def test_source_default_applies_without_an_override(self):
        from melteval.readers.shar import effective_text_field

        assert effective_text_field(self._cut(tags={"lang": "de"}), "custom.pnc_text") == "custom.pnc_text"

    def test_untagged_cut_falls_back_to_source_default(self):
        from melteval.readers.shar import effective_text_field

        assert effective_text_field(self._cut(), "text") == "text"

    def test_empty_override_does_not_win(self):
        from melteval.readers.shar import effective_text_field

        assert effective_text_field(self._cut(tags={"text_field": ""}), "custom.pnc_text") == "custom.pnc_text"


class TestGroupFormRejection:
    def test_nested_group_form_is_rejected(self):
        """Group tags replace cut tags wholesale, clobbering src_lang/tgt_lang."""
        from melteval.readers.shar import _reject_group_form

        with pytest.raises(ValueError, match="nested group form"):
            _reject_group_form({"tags": {"task": "st"}, "input_cfg": [{"shar_path": "/x"}]}, 0)

    def test_flat_form_is_accepted(self):
        from melteval.readers.shar import _reject_group_form

        _reject_group_form({"shar_path": "/x", "tags": {"task": "asr"}}, 0)
