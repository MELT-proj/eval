"""Tests for the MCIF reader.

Entirely offline: `datasets.load_dataset` and `huggingface_hub.hf_hub_download`
are both monkeypatched, the first to a tiny in-memory dataset (MCIF's `audio`/
`video` columns are plain repo-relative paths, not a `datasets.Audio` feature,
so no Arrow/torchcodec workaround is needed here the way `test_hf_reader.py`
needs one), the second to a small hand-built reference XML this module's tests
construct directly, matching the shape confirmed against the real
`FBK-MT/MCIF` reference files (see the reader's module docstring).
"""

from __future__ import annotations

import gzip
import io

import numpy as np
import pytest
import soundfile as sf
from datasets import Dataset

from melteval.readers.mcif import TARGET_SAMPLE_RATE, MCIFReader


def _wav_bytes(num_samples: int, sample_rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.linspace(-0.5, 0.5, num_samples, dtype=np.float32), sample_rate, format="WAV")
    return buf.getvalue()


#: The parquet split. Row `3` deliberately has no reference-bearing sample
#: (dropped from the ref XML directly); row `6`'s blank prompt exercises the
#: no-instruction drop; row `5` has no `audio` at all (its ref sample is
#: video-only).
ROWS = [
    {"id": "0", "prompt_en": "Transcribe part 1.", "prompt_de": "Transkribiere Teil 1.", "audio": "a.wav"},
    {"id": "1", "prompt_en": "Transcribe part 2.", "prompt_de": "Transkribiere Teil 2.", "audio": "b.wav"},
    {"id": "2", "prompt_en": "What is this?", "prompt_de": "Was ist das?", "audio": "c.wav"},
    {"id": "3", "prompt_en": "Unused.", "prompt_de": "Unbenutzt.", "audio": "d.wav"},
    {"id": "4", "prompt_en": "Future task.", "prompt_de": "Zukunftsaufgabe.", "audio": "e.wav"},
    {"id": "5", "prompt_en": "Video only question.", "prompt_de": "Nur Video.", "audio": ""},
    {"id": "6", "prompt_en": "", "prompt_de": "", "audio": "g.wav"},
]

REF_XML = b"""<?xml version='1.0' encoding='utf-8'?>
<testset name="MCIF" type="output">
  <task track="short" text_lang="en">
    <sample id="0,1" iid="ASR_1" task="ASR">
      <audio_path>a.wav,b.wav</audio_path>
      <reference>hello world</reference>
    </sample>
    <sample id="2" iid="QA_1" task="QA" qa_type="A" qa_origin="Transcript">
      <audio_path>c.wav</audio_path>
      <reference>the answer</reference>
    </sample>
    <sample id="3" iid="QA_2" task="QA">
      <audio_path>d.wav</audio_path>
      <reference></reference>
    </sample>
    <sample id="99" iid="ASR_2" task="ASR">
      <audio_path>missing.wav</audio_path>
      <reference>unreachable</reference>
    </sample>
    <sample id="4" iid="FUTURE_1" task="FUTURE_TASK">
      <audio_path>e.wav</audio_path>
      <reference>ignored</reference>
    </sample>
    <sample id="5" iid="QA_3" task="QA">
      <video_path>f.mp4</video_path>
      <reference>video only, no audio</reference>
    </sample>
    <sample id="6" iid="QA_4" task="QA">
      <audio_path>g.wav</audio_path>
      <reference>some answer</reference>
    </sample>
  </task>
</testset>
"""


@pytest.fixture
def dataset() -> Dataset:
    return Dataset.from_dict({key: [row[key] for row in ROWS] for key in ROWS[0]})


@pytest.fixture
def patched(monkeypatch, dataset, tmp_path):
    """Route `load_dataset`/`hf_hub_download` to the fixtures above."""

    def fake_load_dataset(repo, name=None, split=None, revision=None):
        return dataset

    ref_path = tmp_path / "ref.xml.gz"
    ref_path.write_bytes(gzip.compress(REF_XML))

    calls: list[str] = []

    def fake_hf_hub_download(repo_id, repo_type, revision, filename):
        calls.append(filename)
        if filename.endswith(".xml.gz"):
            return str(ref_path)
        audio_path = tmp_path / filename
        if not audio_path.exists():
            audio_path.write_bytes(_wav_bytes(800))
        return str(audio_path)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    return calls


def _source_cfg(**overrides) -> dict:
    defaults = {
        "repo": "FBK-MT/MCIF",
        "revision": "deadbeef",
        "track": "short",
        "prompt_type": "fixed",
        "lang": "en",
    }
    defaults.update(overrides)
    return defaults


class TestFreeze:
    def test_keeps_asr_group_and_one_qa_sample(self, patched):
        result = MCIFReader().freeze(_source_cfg(), source_index=0)
        assert result.stats["read"] == 7
        assert result.stats["kept"] == 3
        assert result.stats["task_asr"] == 2
        assert result.stats["task_qa"] == 1

    def test_drops_are_attributed_correctly(self, patched):
        result = MCIFReader().freeze(_source_cfg(), source_index=0)
        assert result.stats["dropped_no_reference"] == 1  # id 3, blank <reference>
        assert result.stats["dropped_missing_row"] == 1  # id 99, not in the parquet
        assert result.stats["dropped_unknown_task"] == 1  # id 4, task="FUTURE_TASK"
        assert result.stats["dropped_no_modality"] == 1  # id 5, video-only
        assert result.stats["dropped_no_instruction"] == 1  # id 6, blank prompt_en

    def test_asr_group_maps_to_chunked_asr_with_grouping_metadata(self, patched):
        records = MCIFReader().freeze(_source_cfg(), source_index=0).records
        asr = sorted((r for r in records if r.task == "chunked_asr"), key=lambda r: r.cut_id)
        assert [r.cut_id for r in asr] == ["0", "1"]
        assert [r.extra["group_order"] for r in asr] == [0, 1]
        assert all(r.extra["group_id"] == "ASR_1" for r in asr)
        assert all(r.extra["group_size"] == 2 for r in asr)
        assert all(r.target == "hello world" for r in asr)

    def test_qa_maps_to_audio_chat_with_qa_metadata(self, patched):
        records = MCIFReader().freeze(_source_cfg(), source_index=0).records
        qa = next(r for r in records if r.task == "audio_chat")
        assert qa.target == "the answer"
        assert qa.extra["qa_type"] == "A"
        assert qa.extra["qa_origin"] == "Transcript"
        assert qa.extra["group_size"] == 1

    def test_instruction_carries_the_audio_token_placeholder_and_the_question(self, patched):
        records = MCIFReader().freeze(_source_cfg(), source_index=0).records
        qa = next(r for r in records if r.task == "audio_chat")
        assert "{audio_token}" in qa.instruction
        assert "What is this?" in qa.instruction

    def test_lang_selects_the_matching_prompt_column(self, monkeypatch, dataset, tmp_path):
        """The German reference file has its own task/sample set in real MCIF;
        here it is the same XML (English-labelled) reused only to check that
        `prompt_de` -- not `prompt_en` -- is what ends up in the instruction."""

        def fake_load_dataset(repo, name=None, split=None, revision=None):
            return dataset

        ref_path = tmp_path / "ref.xml.gz"
        # Reuse the English-track XML, relabelled `text_lang="de"`, since only
        # the join logic (not real German content) is under test here.
        monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
        ref_path.write_bytes(gzip.compress(REF_XML.replace(b'text_lang="en"', b'text_lang="de"')))

        def fake_hf_hub_download(repo_id, repo_type, revision, filename):
            if filename.endswith(".xml.gz"):
                return str(ref_path)
            audio_path = tmp_path / filename
            if not audio_path.exists():
                audio_path.write_bytes(_wav_bytes(800))
            return str(audio_path)

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)

        records = MCIFReader().freeze(_source_cfg(lang="de"), source_index=0).records
        qa = next(r for r in records if r.task == "audio_chat")
        assert "Was ist das?" in qa.instruction
        assert all(r.lang == "de" for r in records)

    def test_dataset_id_defaults_from_track_prompt_type_and_lang(self, patched):
        records = MCIFReader().freeze(_source_cfg(), source_index=0).records
        assert all(r.dataset_id == "mcif-short-fixed-en" for r in records)

    def test_tags_dataset_id_overrides_the_default(self, patched):
        records = MCIFReader().freeze(
            _source_cfg(tags={"dataset_id": "custom"}), source_index=0
        ).records
        assert all(r.dataset_id == "custom" for r in records)

    def test_sample_keys_are_source_index_prefixed(self, patched):
        records = MCIFReader().freeze(_source_cfg(), source_index=7).records
        assert all(r.sample_key.startswith("07-") for r in records)

    def test_missing_revision_is_rejected(self, patched):
        cfg = _source_cfg()
        del cfg["revision"]
        with pytest.raises(ValueError, match="revision"):
            MCIFReader().freeze(cfg, source_index=0)

    def test_missing_lang_is_rejected(self, patched):
        cfg = _source_cfg()
        del cfg["lang"]
        with pytest.raises(ValueError, match="lang"):
            MCIFReader().freeze(cfg, source_index=0)

    def test_bad_track_is_rejected(self, patched):
        with pytest.raises(ValueError, match="track"):
            MCIFReader().freeze(_source_cfg(track="medium"), source_index=0)

    def test_no_matching_task_block_lists_whats_available(self, patched):
        with pytest.raises(ValueError, match="Available"):
            MCIFReader().freeze(_source_cfg(lang="fr"), source_index=0)


class TestLoadAudio:
    def test_downloads_decodes_and_resamples(self, patched):
        records = MCIFReader().freeze(_source_cfg(), source_index=0).records
        record = next(r for r in records if r.task == "audio_chat")
        data, sample_rate = MCIFReader().load_audio(record.audio)
        assert sample_rate == TARGET_SAMPLE_RATE
        assert data.ndim == 1
        assert data.dtype == np.float32
