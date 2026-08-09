"""Tests for the frozen-set manifest schema."""

from pathlib import Path

import pytest

from melteval.manifest import (
    MANIFEST_NAME,
    AudioLocator,
    EvalRecord,
    read_manifest,
    resolve_frozen_set,
    write_manifest,
)


def _record(**overrides) -> EvalRecord:
    """Build a record with sensible defaults for the fields under test."""
    defaults = {
        "sample_key": "00-000000",
        "task": "asr",
        "target": "hello world",
        "audio": AudioLocator(kind="shar", params={"dir": "/data/x", "index": 3}),
        "lang": "en",
        "src_lang": "en",
        "duration": 1.5,
    }
    defaults.update(overrides)
    return EvalRecord(**defaults)


class TestRoundTrip:
    def test_record_survives_serialisation(self):
        record = _record(source_text="ciao", choices=["a", "b"], cut_id="abc")
        restored = EvalRecord.from_dict(__import__("json").loads(record.to_json()))
        assert restored == record

    def test_locator_params_survive(self):
        locator = AudioLocator(kind="shar", params={"dir": "/d", "index": 7, "indexes_root": "/i"})
        assert AudioLocator.from_dict(locator.to_dict()) == locator

    def test_list_targets_are_preserved(self):
        """Multiple acceptable answers must not be flattened into a string."""
        record = _record(task="qa", target=["Paris", "paris, france"])
        restored = EvalRecord.from_dict(__import__("json").loads(record.to_json()))
        assert restored.target == ["Paris", "paris, france"]

    def test_write_then_read(self, tmp_path: Path):
        records = [_record(sample_key=f"00-{i:06d}") for i in range(3)]
        write_manifest(tmp_path / MANIFEST_NAME, records)
        assert read_manifest(tmp_path / MANIFEST_NAME) == records

    def test_non_ascii_is_not_escaped(self, tmp_path: Path):
        """Manifests are read by humans debugging references; keep them legible."""
        write_manifest(tmp_path / MANIFEST_NAME, [_record(target="أهلاً")])
        assert "أهلاً" in (tmp_path / MANIFEST_NAME).read_text(encoding="utf-8")


class TestResolveFrozenSet:
    def test_accepts_a_directory(self, tmp_path: Path):
        write_manifest(tmp_path / MANIFEST_NAME, [_record()])
        assert resolve_frozen_set(tmp_path) == tmp_path / MANIFEST_NAME

    def test_accepts_the_manifest_itself(self, tmp_path: Path):
        path = tmp_path / MANIFEST_NAME
        write_manifest(path, [_record()])
        assert resolve_frozen_set(path) == path

    def test_missing_manifest_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            resolve_frozen_set(tmp_path)
