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


class TestEuroparlStSpecs:
    """The EuroparlST dev/test specs: 56 non-English directions, one dataset_id each."""

    LANGS = {"fr", "de", "es", "it", "pt", "pl", "ro", "nl"}

    @staticmethod
    def _load(name: str, monkeypatch) -> list[dict]:
        monkeypatch.setenv("LOCAL_DATASETS_DIR", "/shar")
        return load_spec(Path(__file__).parent.parent / "configs" / "shar" / name)["input_cfg"]

    def _check(self, sources: list[dict], split: str, suffix: str, cap: int | None) -> None:
        pairs = {(s["tags"]["src_lang"], s["tags"]["tgt_lang"]) for s in sources}
        assert len(sources) == len(pairs) == 56
        assert pairs == {(a, b) for a in self.LANGS for b in self.LANGS if a != b}
        for s in sources:
            src, tgt = s["tags"]["src_lang"], s["tags"]["tgt_lang"]
            assert s["shar_path"] == f"/shar/europarl_st/{src}_{tgt}/{split}"
            assert s["tags"]["dataset_id"] == f"europarl-st-{src}_{tgt}{suffix}"
            assert s["tags"]["task"] == "st"
            assert s["source_text_field"] == "custom.sentence"
            assert s.get("max_samples") == cap
            assert "text_field" not in s  # reference stays supervisions[0].text (the translation)

    def test_dev_is_validation_capped_at_100(self, monkeypatch):
        self._check(self._load("europarl-st-dev.yaml", monkeypatch), "validation", "-dev", 100)

    def test_test_is_the_full_test_split(self, monkeypatch):
        self._check(self._load("europarl-st-test.yaml", monkeypatch), "test", "", None)

    def test_dev_dataset_ids_do_not_collide_with_test(self, monkeypatch):
        dev = {s["tags"]["dataset_id"] for s in self._load("europarl-st-dev.yaml", monkeypatch)}
        test = {s["tags"]["dataset_id"] for s in self._load("europarl-st-test.yaml", monkeypatch)}
        assert not dev & test


class TestSpeechvalDevSpecs:
    """The speechval-dev specs: full validation/dev splits, no cap, and every suite slice has a source."""

    ROOT = Path(__file__).parent.parent
    SHAR = {
        "voxpopuli-dev.yaml": "validation",
        "fleurs24-st-xen-dev-full.yaml": "validation",
        "fleurs24-st-enx-dev.yaml": "validation",
        "st-eval-campaign-v1-dev.yaml": "dev",
    }

    def _load(self, name: str, monkeypatch) -> list[dict]:
        monkeypatch.setenv("LOCAL_DATASETS_DIR", "/shar")
        monkeypatch.setenv("AZEROS_INDEXED_DIR", "/azeros")
        return load_spec(self.ROOT / "configs" / "shar" / name)["input_cfg"]

    @pytest.mark.parametrize("name,split", SHAR.items())
    def test_full_split_and_dev_ids(self, name, split, monkeypatch):
        sources = self._load(name, monkeypatch)
        ids = [s["tags"]["dataset_id"] for s in sources]
        assert len(ids) == len(set(ids))
        for s in sources:
            assert "max_samples" not in s
            assert s["shar_path"].endswith(f"/{split}")
            assert s["tags"]["dataset_id"].endswith("-dev")
            if "reference_map" in s:
                assert (self.ROOT / s["reference_map"]).is_file() and "reference-dev" in s["reference_map"]

    def test_enx_pairs_english_validation_audio_with_target_references(self, monkeypatch):
        sources = self._load("fleurs24-st-enx-dev.yaml", monkeypatch)
        assert len(sources) == 23
        for s in sources:
            tgt = s["tags"]["tgt_lang"]
            assert s["shar_path"] == "/shar/fleurs/en_us/validation"
            assert s["reference_map"] == f"configs/refs/fleurs-{tgt}-reference-dev.json"
            assert s["tags"]["dataset_id"] == f"fleurs-en_{tgt}-dev"

    def test_xen_uses_the_dev_english_reference(self, monkeypatch):
        sources = self._load("fleurs24-st-xen-dev-full.yaml", monkeypatch)
        assert len(sources) == 17  # jupiter has no FLEURS for el/pl/ro/sk/sl/sv
        assert {s["reference_map"] for s in sources} == {"configs/refs/fleurs-en-reference-dev.json"}

    def test_covost_dev_covers_the_suite_directions(self, monkeypatch):
        ids = {s["tags"]["dataset_id"] for s in self._load("st-eval-campaign-v1-dev.yaml", monkeypatch)}
        assert ids == {f"covost2-{s}_en-dev" for s in ("de", "es", "fr", "it", "pt")}

    @pytest.mark.parametrize("c", ["clean", "other"])
    def test_librispeech_validation_mirrors_test(self, c):
        val = load_spec(self.ROOT / "configs" / "hf" / f"librispeech-hf-validation-{c}.yaml")["input_cfg"][0]
        test = load_spec(self.ROOT / "configs" / "hf" / f"librispeech-hf-test-{c}.yaml")["input_cfg"][0]
        assert val["split"] == "validation" and "max_samples" not in val
        assert {k: v for k, v in val.items() if k not in ("split", "tags")} == {
            k: v for k, v in test.items() if k not in ("split", "tags")
        }
        assert val["tags"]["dataset_id"] == f"librispeech-validation-{c}"

    def test_every_suite_slice_has_a_source(self, monkeypatch):
        import yaml

        suite = yaml.safe_load((self.ROOT / "configs/eval/suites/speechval-dev.yaml").read_text())["benchmarks"]
        known = {f"librispeech-validation-{c}" for c in ("clean", "other")}
        for name in self.SHAR:
            known |= {s["tags"]["dataset_id"] for s in self._load(name, monkeypatch)}
        for b in suite:
            if b["name"].startswith("europarl"):
                continue
            for sl in b["slices"]:
                assert sl["dataset_id"] in known, sl
