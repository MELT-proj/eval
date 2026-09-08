"""Tests for turning a frozen set -- or a spec read live -- into a dataset."""

import pytest
import yaml

from melteval.dataset import AUDIO_DATA_KEY, frozen_dataset, record_to_sample, spec_dataset
from melteval.manifest import MANIFEST_NAME, AudioLocator, EvalRecord, write_manifest


def _record(key, **overrides) -> EvalRecord:
    defaults = {
        "sample_key": key,
        "task": "asr",
        "target": "hello",
        "audio": AudioLocator(kind="shar", params={"dir": "/d", "index": 1}),
        "lang": "en",
        "dataset_id": "librispeech",
        "duration": 2.0,
    }
    defaults.update(overrides)
    return EvalRecord(**defaults)


@pytest.fixture
def frozen(tmp_path):
    """A small mixed frozen set on disk."""
    write_manifest(
        tmp_path / MANIFEST_NAME,
        [
            _record("00-000000", lang="en"),
            _record("00-000001", lang="en"),
            _record("01-000000", task="st", lang="de", dataset_id="covost2"),
        ],
    )
    return tmp_path


class TestFiltering:
    def test_loads_everything_by_default(self, frozen):
        assert len(frozen_dataset(frozen)) == 3

    def test_filters_by_task(self, frozen):
        assert len(frozen_dataset(frozen, task="st")) == 1

    def test_filters_by_language(self, frozen):
        assert len(frozen_dataset(frozen, lang="en")) == 2

    def test_filters_by_corpus(self, frozen):
        assert len(frozen_dataset(frozen, dataset_id="covost2")) == 1

    def test_limit_takes_manifest_order(self, frozen):
        """Deterministic, so a limited run is still comparable to itself."""
        assert [s.id for s in frozen_dataset(frozen, limit=2)] == ["00-000000", "00-000001"]

    def test_empty_selection_raises(self, frozen):
        """A filter that matches nothing would otherwise report a score of 0."""
        with pytest.raises(ValueError, match="No samples"):
            frozen_dataset(frozen, task="speechqe")


class TestSampleConversion:
    def test_identity_and_target_are_carried(self):
        sample = record_to_sample(_record("00-000042", target="the reference"))
        assert sample.id == "00-000042"
        assert sample.target == "the reference"

    def test_audio_locator_reaches_the_metadata(self):
        """The provider is handed messages only, so the solver moves this into
        the message content; it starts life here."""
        sample = record_to_sample(_record("k"))
        assert sample.metadata["audio"]["kind"] == "shar"
        assert sample.metadata["audio"]["index"] == 1

    def test_grouping_keys_are_present(self):
        """grouped() reads these straight off sample metadata."""
        sample = record_to_sample(_record("k", task="st", lang="de", dataset_id="covost2"))
        for key in ("task", "lang", "dataset_id", "src_lang", "tgt_lang", "source"):
            assert key in sample.metadata

    def test_instruction_becomes_the_visible_input(self):
        sample = record_to_sample(_record("k", instruction="Summarise the talk."))
        assert sample.input == "Summarise the talk."

    def test_placeholder_input_describes_the_sample(self):
        """Without an instruction the prompt is not known until the solver runs,
        so the input column carries a description rather than a lie."""
        sample = record_to_sample(_record("k", task="asr", lang="en", duration=3.25))
        assert sample.input == "[asr en 3.2s]"

    def test_choices_are_carried_for_multiple_choice_tasks(self):
        sample = record_to_sample(_record("k", task="qa", choices=["a", "b", "c"]))
        assert sample.choices == ["a", "b", "c"]

    def test_extra_reaches_the_metadata(self):
        """The open-ended extension point readers like MCIF's use for
        metadata no other reader needs -- chunk grouping, in that case."""
        sample = record_to_sample(_record("k", extra={"group_id": "ASR_9", "group_order": 3}))
        assert sample.metadata["group_id"] == "ASR_9"
        assert sample.metadata["group_order"] == 3


class TestAudioKey:
    def test_key_is_stable(self):
        """The provider looks this key up by name; renaming it breaks generation."""
        assert AUDIO_DATA_KEY == "melteval_audio"


class TestSpecDataset:
    """Reading a spec at eval time, with no frozen set in between."""

    @pytest.fixture
    def spec_file(self, tmp_path):
        spec = {
            "name": "two-corpora",
            "seed": 0,
            "input_cfg": [
                {"type": "fake", "tags": {"task": "asr", "lang": "en", "dataset_id": "alpha"}},
                {"type": "fake", "tags": {"task": "st", "tgt_lang": "de", "dataset_id": "beta"}},
            ],
        }
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.safe_dump(spec), encoding="utf-8")
        return path

    @pytest.fixture
    def fake_reader(self, monkeypatch):
        """A reader that records which sources it was asked to read."""
        from melteval.manifest import AudioLocator, EvalRecord
        from melteval.readers.base import SourceResult

        read_sources: list[str] = []

        class FakeReader:
            source_type = "fake"
            locator_kind = "fake"

            def freeze(self, source_cfg, source_index):
                tags = dict(source_cfg.get("tags") or {})
                read_sources.append(str(tags.get("dataset_id")))
                records = [
                    EvalRecord(
                        sample_key=f"{source_index:02d}-{ordinal:06d}",
                        task=str(tags.get("task", "")),
                        target=f"target {ordinal}",
                        audio=AudioLocator(kind="fake", params={"index": ordinal}),
                        lang=str(tags.get("lang") or tags.get("tgt_lang") or ""),
                        tgt_lang=str(tags.get("tgt_lang", "")),
                        dataset_id=str(tags.get("dataset_id", "")),
                    )
                    for ordinal in range(3)
                ]
                return SourceResult(records=records, stats={"read": 3, "kept": 3})

            def load_audio(self, locator):
                raise NotImplementedError

        import melteval.readers.base as base

        monkeypatch.setitem(base._READERS, "fake", FakeReader())
        monkeypatch.setattr(base, "_LOADED", True)
        return read_sources

    def test_reads_a_spec_without_a_manifest(self, spec_file, fake_reader):
        dataset = spec_dataset(spec_file)
        assert len(dataset) == 6

    def test_sample_keys_match_what_freezing_the_same_spec_would_assign(
        self, spec_file, fake_reader
    ):
        """A run made live must stay comparable to one made from a frozen set."""
        dataset = spec_dataset(spec_file)
        assert [s.id for s in dataset][:4] == [
            "00-000000", "00-000001", "00-000002", "01-000000",
        ]

    def test_filters_select_records(self, spec_file, fake_reader):
        dataset = spec_dataset(spec_file, task="asr")
        assert len(dataset) == 3
        assert all(s.metadata["dataset_id"] == "alpha" for s in dataset)

    def test_an_excluded_source_is_never_read(self, spec_file, fake_reader):
        """Not an optimisation: reading a source means materialising a split."""
        spec_dataset(spec_file, dataset_id="beta")
        assert fake_reader == ["beta"]

    def test_lang_matches_a_source_tagged_only_by_target_language(
        self, spec_file, fake_reader
    ):
        """Readers collapse an ST record's `lang` to its target language, so a
        source tagged `tgt_lang: de` must not be skipped by `lang="de"`."""
        dataset = spec_dataset(spec_file, lang="de")
        assert len(dataset) == 3
        assert fake_reader == ["beta"]

    def test_a_source_that_declares_nothing_is_always_read(self, tmp_path, fake_reader):
        """An unlabelled source means "unknown", not "excluded"."""
        path = tmp_path / "untagged.yaml"
        path.write_text(
            yaml.safe_dump({"name": "x", "input_cfg": [{"type": "fake"}]}), encoding="utf-8"
        )
        assert len(spec_dataset(path)) == 3

    def test_limit_truncates_in_spec_order(self, spec_file, fake_reader):
        assert len(spec_dataset(spec_file, limit=4)) == 4

    def test_filters_selecting_nothing_is_an_error(self, spec_file, fake_reader):
        with pytest.raises(ValueError, match="An eval over zero samples"):
            spec_dataset(spec_file, dataset_id="nonexistent")
