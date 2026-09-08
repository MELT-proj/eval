"""Tests for the HuggingFace ``datasets`` reader.

Entirely offline: a tiny in-memory dataset is built with real WAV bytes and
`datasets.load_dataset` is monkeypatched to return it, so these need neither
network access nor the `torchcodec` dependency the installed `datasets`
release would otherwise require to *construct* audio data in memory. (Freeze
itself never needs it either way -- see the reader's module docstring.)
"""

import io

import numpy as np
import pyarrow as pa
import pytest
import soundfile as sf
from datasets import Audio, Dataset, DatasetInfo, Features, Value

from melteval.readers.base import get_reader, reader_for_locator
from melteval.readers.hf import TARGET_SAMPLE_RATE, HFReader


def _wav_bytes(num_samples: int, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.linspace(-0.5, 0.5, num_samples, dtype=np.float32), sample_rate, format="WAV")
    return buf.getvalue()


def _build_dataset(rows: list[dict], sample_rate: int = 16000) -> Dataset:
    """Build an in-memory HF dataset with real (decodable) WAV audio.

    Constructed as an Arrow table directly rather than via ``Dataset.from_dict``,
    which would route raw arrays through ``Audio.encode_example`` -- and the
    installed `datasets` release hard-requires `torchcodec` for that step, even
    though decoding an already-encoded audio column does not need it.
    """
    columns = {key: [row[key] for row in rows] for key in rows[0]}
    audio_type = pa.struct({"bytes": pa.binary(), "path": pa.string()})
    table = pa.table(
        {
            **{k: pa.array(v) for k, v in columns.items() if k != "audio"},
            "audio": pa.array(columns["audio"], type=audio_type),
        }
    )
    features = Features(
        {
            **{k: Value("string") if k != "duration" else Value("float32") for k in columns if k != "audio"},
            "audio": Audio(decode=False),
        }
    )
    return Dataset(table, info=DatasetInfo(features=features))


@pytest.fixture
def dataset():
    return _build_dataset(
        [
            {"id": "s0", "text": "hello world", "source_text": "salut le monde",
             "audio": {"bytes": _wav_bytes(1600), "path": None}},
            {"id": "s1", "text": "  ", "source_text": "",  # blank reference -> dropped
             "audio": {"bytes": _wav_bytes(1600), "path": None}},
            {"id": "s2", "text": "goodbye", "source_text": "au revoir",
             "audio": {"bytes": _wav_bytes(3200), "path": None}},
        ]
    )


@pytest.fixture
def patched_load(monkeypatch, dataset):
    """Route the reader's `load_dataset` calls to the fixture dataset."""
    import melteval.readers.hf as hf_module

    calls: list[dict] = []

    def fake_load_dataset(repo, name=None, split=None, revision=None):
        calls.append({"repo": repo, "name": name, "split": split, "revision": revision})
        return dataset

    monkeypatch.setattr(hf_module, "load_dataset", fake_load_dataset, raising=False)
    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    hf_module._DATASET_CACHE.clear()
    return calls


def _source_cfg(**overrides):
    cfg = {"repo": "org/speech-test", "revision": "abc123", "text_column": "text"}
    cfg.update(overrides)
    return cfg


class TestFreeze:
    def test_requires_repo(self):
        reader = HFReader()
        with pytest.raises(ValueError, match="requires `repo`"):
            reader.freeze({"revision": "abc"}, 0)

    def test_requires_revision(self):
        reader = HFReader()
        with pytest.raises(ValueError, match="revision.*required"):
            reader.freeze({"repo": "org/speech-test"}, 0)

    def test_drops_blank_references_and_counts_them(self, patched_load):
        result = HFReader().freeze(_source_cfg(), 0)
        assert result.stats["read"] == 3
        assert result.stats["kept"] == 2
        assert result.stats["dropped_no_reference"] == 1

    def test_sample_keys_follow_the_source_index_scheme(self, patched_load):
        result = HFReader().freeze(_source_cfg(), 3)
        assert [r.sample_key for r in result.records] == ["03-000000", "03-000001"]

    def test_targets_are_read_correctly(self, patched_load):
        records = HFReader().freeze(_source_cfg(), 0).records
        assert [r.target for r in records] == ["hello world", "goodbye"]

    def test_ids_are_read_via_the_id_column(self, patched_load):
        records = HFReader().freeze(_source_cfg(id_column="id"), 0).records
        assert [r.cut_id for r in records] == ["s0", "s2"]

    def test_id_column_overrides_the_row_index_fallback(self, patched_load):
        records = HFReader().freeze(_source_cfg(id_column="id"), 0).records
        assert records[0].cut_id == "s0"

    def test_falls_back_to_row_index_without_an_id_column(self, patched_load):
        records = HFReader().freeze(_source_cfg(), 0).records
        # global index 0 and 2 in the underlying dataset (row 1 was dropped)
        assert [r.cut_id for r in records] == ["0", "2"]

    def test_source_text_is_captured_for_st(self, patched_load):
        records = HFReader().freeze(_source_cfg(source_text_column="source_text"), 0).records
        assert records[0].source_text == "salut le monde"
        assert records[1].source_text == "au revoir"

    def test_no_source_text_column_means_none(self, patched_load):
        records = HFReader().freeze(_source_cfg(), 0).records
        assert records[0].source_text is None

    def test_tags_populate_task_and_languages(self, patched_load):
        records = HFReader().freeze(
            _source_cfg(tags={"task": "asr", "lang": "en", "dataset_id": "speech-test"}), 0
        ).records
        assert records[0].task == "asr"
        assert records[0].lang == "en"
        assert records[0].dataset_id == "speech-test"

    def test_dataset_id_defaults_to_the_repo(self, patched_load):
        records = HFReader().freeze(_source_cfg(), 0).records
        assert records[0].dataset_id == "org/speech-test"

    def test_locator_carries_enough_to_resolve_audio_later(self, patched_load):
        record = HFReader().freeze(_source_cfg(), 0).records[0]
        assert record.audio.kind == "hf"
        assert record.audio.params["repo"] == "org/speech-test"
        assert record.audio.params["revision"] == "abc123"
        assert record.audio.params["index"] == 0

    def test_max_samples_is_a_deterministic_seeded_subset(self, patched_load):
        first = HFReader().freeze(_source_cfg(max_samples=1, seed=7), 0).records
        second = HFReader().freeze(_source_cfg(max_samples=1, seed=7), 0).records
        assert [r.cut_id for r in first] == [r.cut_id for r in second]

    def test_registered_under_hf_dataset(self):
        assert get_reader("hf_dataset") is not None


class TestLoadAudio:
    def test_resolves_the_right_row_by_index(self, patched_load):
        records = HFReader().freeze(_source_cfg(), 0).records
        reader = reader_for_locator(records[1].audio)  # the "goodbye" row, index 2
        audio, sample_rate = reader.load_audio(records[1].audio)
        assert sample_rate == TARGET_SAMPLE_RATE
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        # 3200 samples at the source rate (16 kHz here) -> unchanged length
        assert audio.shape[0] == 3200

    def test_resamples_to_16khz(self, monkeypatch):
        """A non-16kHz dataset must not silently reach the model at its
        native rate -- the provider batches on one assumed sample rate."""
        import melteval.readers.hf as hf_module

        low_rate_dataset = _build_dataset(
            [{"id": "s0", "text": "hi", "source_text": "", "audio": {"bytes": _wav_bytes(800, sample_rate=8000), "path": None}}]
        )
        monkeypatch.setattr(hf_module, "load_dataset", lambda *a, **k: low_rate_dataset, raising=False)
        monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: low_rate_dataset)
        hf_module._DATASET_CACHE.clear()

        record = HFReader().freeze(_source_cfg(), 0).records[0]
        audio, sample_rate = HFReader().load_audio(record.audio)
        assert sample_rate == TARGET_SAMPLE_RATE
        # 800 samples @ 8kHz (0.1s) resampled to 16kHz -> ~1600 samples
        assert 1500 <= audio.shape[0] <= 1700

    def test_dataset_is_loaded_once_per_source(self, patched_load):
        records = HFReader().freeze(_source_cfg(), 0).records
        reader = reader_for_locator(records[0].audio)
        reader.load_audio(records[0].audio)
        reader.load_audio(records[1].audio)
        load_calls = [c for c in patched_load if c["repo"] == "org/speech-test"]
        # one call from freeze(), and load_audio must not repeat it per sample
        assert len(load_calls) <= 2


class TestInstructionsAndChoices:
    """Benchmarks that ship their own prompt per sample (AIR-Bench and kin)."""

    @pytest.fixture
    def qa_dataset(self):
        return _build_dataset(
            [
                {
                    "id": "q0", "text": "Female", "source_text": "",
                    "question": "What is the speaker's gender?",
                    "choice_a": "Male", "choice_b": "Female", "choice_c": "", "choice_d": "",
                    "audio": {"bytes": _wav_bytes(1600), "path": None},
                },
                {
                    "id": "q1", "text": "play_music", "source_text": "",
                    # braces in a real question: this must not reach str.format
                    "question": "What does {intent} evaluate to here?",
                    "choice_a": "play_music", "choice_b": "post",
                    "choice_c": "createoradd", "choice_d": "audio_volume_other",
                    "audio": {"bytes": _wav_bytes(1600), "path": None},
                },
                {
                    "id": "q2", "text": "an answer", "source_text": "",
                    "question": "   ",  # no prompt of its own -> dropped
                    "choice_a": "x", "choice_b": "y", "choice_c": "", "choice_d": "",
                    "audio": {"bytes": _wav_bytes(1600), "path": None},
                },
            ]
        )

    @pytest.fixture
    def patched_qa_load(self, monkeypatch, qa_dataset):
        import melteval.readers.hf as hf_module

        monkeypatch.setattr(hf_module, "load_dataset", lambda *a, **k: qa_dataset, raising=False)
        monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: qa_dataset)
        hf_module._DATASET_CACHE.clear()

    def _cfg(self, **overrides):
        return _source_cfg(
            text_column="text",
            instruction_column="question",
            choices_columns=["choice_a", "choice_b", "choice_c", "choice_d"],
            **overrides,
        )

    def test_question_becomes_the_instruction(self, patched_qa_load):
        record = HFReader().freeze(self._cfg(), 0).records[0]
        assert "What is the speaker's gender?" in record.instruction

    def test_audio_token_stays_a_placeholder(self, patched_qa_load):
        """The solver expands it later; a reader that expanded it here would
        hand the processor a prompt with no audio slot in it."""
        record = HFReader().freeze(self._cfg(), 0).records[0]
        assert "{audio_token}" in record.instruction

    def test_blank_choice_columns_are_dropped(self, patched_qa_load):
        record = HFReader().freeze(self._cfg(), 0).records[0]
        assert record.choices == ["Male", "Female"]
        assert "C." not in record.instruction

    def test_choices_are_labelled_in_order(self, patched_qa_load):
        record = HFReader().freeze(self._cfg(), 0).records[1]
        assert "A. play_music" in record.instruction
        assert "D. audio_volume_other" in record.instruction

    def test_braces_in_a_question_survive_prompt_rendering(self, patched_qa_load):
        """A stray brace used to be a KeyError at generation time, not here."""
        from melteval.prompt import FormatSpec, render_user_prompt

        record = HFReader().freeze(self._cfg(), 0).records[1]
        spec = FormatSpec(
            apply_chat_template=False, prompt_template=None,
            prompt_template_selection="random", chat_template_config="chatml",
            audio_token="<|audio|>", source="test",
        )
        rendered = render_user_prompt(
            task="audio_mcq", sample_key=record.sample_key, spec=spec,
            lang="en", instruction=record.instruction,
        )
        assert "What does {intent} evaluate to here?" in rendered
        assert "<|audio|>" in rendered

    def test_rows_without_a_question_are_dropped_and_counted(self, patched_qa_load):
        result = HFReader().freeze(self._cfg(), 0)
        assert result.stats["kept"] == 2
        assert result.stats["dropped_no_instruction"] == 1

    def test_no_instruction_column_leaves_the_record_alone(self, patched_qa_load):
        record = HFReader().freeze(_source_cfg(text_column="text"), 0).records[0]
        assert record.instruction is None
        assert record.choices is None

    def test_a_custom_template_controls_the_layout(self, patched_qa_load):
        record = HFReader().freeze(
            self._cfg(instruction_template="Listen: {audio_token}\nQ: {question}\n{choices}"), 0
        ).records[0]
        assert record.instruction.startswith("Listen: {audio_token}")
        assert "Q: What is the speaker's gender?" in record.instruction

    def test_free_form_sources_get_no_choices_block(self, patched_qa_load):
        record = HFReader().freeze(
            _source_cfg(text_column="text", instruction_column="question"), 0
        ).records[0]
        assert record.choices is None
        assert "A." not in record.instruction

    def test_a_template_that_drops_the_question_is_refused(self, patched_qa_load):
        with pytest.raises(ValueError, match="no .question. placeholder"):
            HFReader().freeze(self._cfg(instruction_template="{audio_token}\n{choices}"), 0)

    def test_a_template_that_drops_existing_options_is_refused(self, patched_qa_load):
        """Scoring against options the model was never shown measures nothing."""
        with pytest.raises(ValueError, match="no .choices. placeholder"):
            HFReader().freeze(self._cfg(instruction_template="{audio_token}\n{question}"), 0)
