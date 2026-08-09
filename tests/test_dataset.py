"""Tests for turning a frozen set into an inspect dataset."""

import pytest

from melteval.dataset import AUDIO_DATA_KEY, frozen_dataset, record_to_sample
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


class TestAudioKey:
    def test_key_is_stable(self):
        """The provider looks this key up by name; renaming it breaks generation."""
        assert AUDIO_DATA_KEY == "melteval_audio"
