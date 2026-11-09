"""Tests for the PolySpeech-100 reader.

Entirely offline: a miniature reconstruction (README, ``manifest.jsonl`` and real
wav files at mixed sample rates) is written to a temp directory in the layout the
reader documents. The assertions that matter are the ones a plausible-looking
mistake would not trip: that ``pa_qa`` never shows the question text, that the
reference is the option's text, that corpus braces survive the instruction
template, and that the joined waveform has the length the gaps imply.
"""

import json

import numpy as np
import pytest
import soundfile as sf
import yaml

from melteval.dataset import spec_dataset
from melteval.freeze import load_spec, read_sources
from melteval.readers.base import get_reader, reader_for_locator
from melteval.readers.polyspeech import (
    DEFAULT_QUESTION_GAP_S,
    DEFAULT_SEGMENT_GAP_S,
    PA_QA_QUESTION,
    TARGET_SAMPLE_RATE,
    PolySpeechReader,
)


REVISION = "5c9235323df63e175819d5bee46b82efc6965bbd"

QUESTION = "Wer ist gemeint?"
OPTIONS = ["der Koch", "die Lehrerin", "der Bauer", "die Ärztin"]


def _write_wav(root, rel: str, seconds: float, sample_rate: int) -> dict:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.linspace(-0.5, 0.5, int(seconds * sample_rate), dtype=np.float32)
    sf.write(path, data, sample_rate, format="WAV")
    return {"path": rel, "duration_s": seconds, "sample_rate": sample_rate, "channels": 1, "subtype": "FLOAT"}


def _row(root, lang: str, index: int, qid: int, *, answer: int = 2, question: str = QUESTION,
         options: list[str] | None = None) -> dict:
    sample = f"idx_{index:05d}_qid_{qid}"
    base = f"audio/lang={lang}/{sample}"
    return {
        "id": f"{lang}/{sample}",
        "lang": lang,
        "index": index,
        "qid": qid,
        "correct_answer": answer,
        "passage": "Ein Text.",
        "question": question,
        "options": list(options) if options is not None else list(OPTIONS),
        # Two segments at different rates: the reader has to resample each on its own.
        "passage_audio": [
            _write_wav(root, f"{base}/01_passage_s01_seg00.wav", 1.0, 8000),
            _write_wav(root, f"{base}/01_passage_s02_seg00.wav", 0.5, 24000),
        ],
        "question_audio": [_write_wav(root, f"{base}/02_question_00.wav", 0.5, 16000)],
        "option_audio": [],
    }


@pytest.fixture
def reconstruction(tmp_path):
    """A miniature reconstruction: four German samples (two unusable) and one English."""
    root = tmp_path / "polyspeech-100"
    rows = [
        _row(root, "deu_Latn", 5, 1, answer=2),
        _row(root, "deu_Latn", 6, 2, answer=4, question="Was ist {x}?",
             options=["{y} eins", "zwei", "drei", "vier {"]),
        _row(root, "deu_Latn", 7, 1, options=["gleich", "Gleich", "drei", "vier"]),  # duplicate options
        _row(root, "deu_Latn", 8, 2, answer=7),  # answer out of range
        _row(root, "eng_Latn", 5, 1, answer=1, question="Who is meant?",
             options=["the cook", "the teacher", "the farmer", "the doctor"]),
    ]
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (root / "README.md").write_text(f"Revision (commit): `{REVISION}`\n", encoding="utf-8")
    return root


def _cfg(root, variant: str = "pa_qa", language: str = "deu_Latn", **extra) -> dict:
    return {
        "type": "polyspeech",
        "root": str(root),
        "revision": REVISION,
        "language": language,
        "variant": variant,
        "tags": {"task": "audio_mcq", "lang": "de", "dataset_id": f"polyspeech-{variant}-de"},
        **extra,
    }


class TestFreeze:
    def test_registered_under_its_source_type(self):
        assert isinstance(get_reader("polyspeech"), PolySpeechReader)

    def test_reference_is_the_correct_options_text(self, reconstruction):
        records = PolySpeechReader().freeze(_cfg(reconstruction, "pa_qt"), 0).records
        assert [r.target for r in records] == ["die Lehrerin", "vier {"]
        assert all(r.target in r.choices and len(r.choices) == 4 for r in records)
        assert {r.task for r in records} == {"audio_mcq"}

    def test_unusable_samples_are_dropped_and_counted(self, reconstruction):
        result = PolySpeechReader().freeze(_cfg(reconstruction), 0)
        assert result.stats["read"] == 4
        assert result.stats["kept"] == 2
        assert result.stats["dropped_duplicate_options"] == 1
        assert result.stats["dropped_bad_answer"] == 1

    def test_pa_qt_shows_the_question_and_carries_only_passage_audio(self, reconstruction):
        [first, _] = PolySpeechReader().freeze(_cfg(reconstruction, "pa_qt"), 0).records
        assert QUESTION in first.instruction
        assert "{audio_token}" in first.instruction
        assert "A. der Koch" in first.instruction and "D. die Ärztin" in first.instruction
        assert "question" not in first.audio.params
        assert first.extra["question"] == QUESTION

    def test_pa_qa_never_shows_the_question_text(self, reconstruction):
        """The whole point of the variant: the question reaches the model as audio only."""
        for record in PolySpeechReader().freeze(_cfg(reconstruction, "pa_qa"), 0).records:
            assert record.extra["question_text"] not in record.instruction
            assert PA_QA_QUESTION in record.instruction
            assert record.audio.params["question"]
            assert "A. " in record.instruction  # options are still text
        # ...but the real question is kept for analysis
        assert PolySpeechReader().freeze(_cfg(reconstruction, "pa_qa"), 0).records[0].extra["question_text"] == QUESTION

    def test_corpus_braces_survive_the_instruction_template(self, reconstruction):
        """A per-sample instruction is run through ``str.format`` downstream."""
        second = PolySpeechReader().freeze(_cfg(reconstruction, "pa_qt"), 0).records[1]
        rendered = second.instruction.format(audio_token="<AUDIO>")
        assert "Was ist {x}?" in rendered
        assert "A. {y} eins" in rendered and "D. vier {" in rendered
        assert rendered.startswith("<AUDIO>")

    def test_identity_and_cross_language_join_keys(self, reconstruction):
        records = PolySpeechReader().freeze(_cfg(reconstruction, "pa_qt"), 3).records
        assert [r.sample_key for r in records] == ["03-000000", "03-000001"]
        assert [r.cut_id for r in records] == ["deu_Latn/idx_00005_qid_1", "deu_Latn/idx_00006_qid_2"]
        assert [(r.extra["index"], r.extra["qid"]) for r in records] == [(5, 1), (6, 2)]
        assert {r.lang for r in records} == {"de"} and {r.dataset_id for r in records} == {"polyspeech-pa_qt-de"}

    def test_language_and_dataset_id_defaults(self, reconstruction):
        cfg = _cfg(reconstruction, "pa_qt", language="eng_Latn")
        del cfg["tags"]
        [record] = PolySpeechReader().freeze(cfg, 0).records
        assert (record.lang, record.dataset_id) == ("en", "polyspeech-pa_qt-en")

    def test_duration_covers_the_audio_the_model_hears(self, reconstruction):
        qt = PolySpeechReader().freeze(_cfg(reconstruction, "pa_qt"), 0).records[0]
        qa = PolySpeechReader().freeze(_cfg(reconstruction, "pa_qa"), 0).records[0]
        assert qt.duration == pytest.approx(1.0 + DEFAULT_SEGMENT_GAP_S + 0.5)
        assert qa.duration == pytest.approx(qt.duration + DEFAULT_QUESTION_GAP_S + 0.5)

    def test_max_samples_is_deterministic_and_keeps_order(self, reconstruction):
        cfg = _cfg(reconstruction, "pa_qt", max_samples=1, seed=3)
        a = PolySpeechReader().freeze(cfg, 0).records
        b = PolySpeechReader().freeze(cfg, 0).records
        assert len(a) == 1 and [r.cut_id for r in a] == [r.cut_id for r in b]
        assert a[0].sample_key == "00-000000"

    @pytest.mark.parametrize(
        ("change", "match"),
        [
            ({"variant": "qa_qt"}, "variant"),
            ({"language": "hrv_Latn"}, "not in"),
            ({"revision": "deadbeef"}, "does not record upstream revision"),
            ({"revision": None}, "revision"),
            ({"root": "${POLYSPEECH_ROOT}"}, "unexpanded"),
            ({"root": None}, "requires `root`"),
        ],
    )
    def test_bad_source_configs_are_refused(self, reconstruction, change, match):
        cfg = {**_cfg(reconstruction), **change}
        with pytest.raises(ValueError, match=match):
            PolySpeechReader().freeze(cfg, 0)

    def test_a_directory_that_is_not_a_reconstruction_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="manifest.jsonl"):
            PolySpeechReader().freeze(_cfg(tmp_path), 0)


class TestLoadAudio:
    def test_pa_qt_joins_the_passage_segments_at_16k(self, reconstruction):
        [record, _] = PolySpeechReader().freeze(_cfg(reconstruction, "pa_qt"), 0).records
        audio, rate = reader_for_locator(record.audio).load_audio(record.audio)
        assert rate == TARGET_SAMPLE_RATE and audio.dtype == np.float32 and audio.ndim == 1
        # 1.0 s @ 8k and 0.5 s @ 24k, each resampled alone to 16k, 0.25 s apart.
        assert len(audio) == 16000 + 4000 + 8000

    def test_pa_qa_appends_the_spoken_question_after_a_pause(self, reconstruction):
        [record, _] = PolySpeechReader().freeze(_cfg(reconstruction, "pa_qa"), 0).records
        audio, _ = reader_for_locator(record.audio).load_audio(record.audio)
        assert len(audio) == 16000 + 4000 + 8000 + 16000 + 8000
        pause = audio[16000 + 4000 + 8000 : 16000 + 4000 + 8000 + 16000]
        assert not pause.any()  # the question gap is silence

    def test_gaps_are_configurable(self, reconstruction):
        cfg = _cfg(reconstruction, "pa_qa", segment_gap_s=0.0, question_gap_s=0.5)
        [record, _] = PolySpeechReader().freeze(cfg, 0).records
        audio, _ = reader_for_locator(record.audio).load_audio(record.audio)
        assert len(audio) == 16000 + 8000 + 8000 + 8000

    def test_an_undecodable_file_names_itself(self, reconstruction):
        [record, _] = PolySpeechReader().freeze(_cfg(reconstruction, "pa_qt"), 0).records
        (reconstruction / record.audio.params["passage"][0]).write_bytes(b"not audio")
        with pytest.raises(RuntimeError, match="01_passage_s01_seg00"):
            reader_for_locator(record.audio).load_audio(record.audio)


def _write_spec(path, root, variant: str, *languages: tuple[str, str]) -> None:
    sources = [
        {
            "type": "polyspeech", "root": str(root), "revision": REVISION, "language": language,
            "variant": variant,
            "tags": {"task": "audio_mcq", "lang": lang, "dataset_id": f"polyspeech-{variant}-{lang}"},
        }
        for language, lang in languages
    ]
    path.write_text(yaml.safe_dump({"name": "ps", "seed": 0, "input_cfg": sources}), encoding="utf-8")


class TestThroughSpecAndInspect:
    def test_spec_dataset_filters_by_dataset_id_without_reading_other_sources(self, reconstruction, tmp_path):
        spec = tmp_path / "spec.yaml"
        _write_spec(spec, reconstruction, "pa_qt", ("deu_Latn", "de"), ("eng_Latn", "en"))
        dataset = spec_dataset(spec, dataset_id="polyspeech-pa_qt-en")
        assert len(dataset) == 1
        sample = dataset[0]
        # source #1 keeps its number although source #0 was skipped
        assert sample.id == "01-000000" and sample.target == "the cook"
        assert sample.metadata["task"] == "audio_mcq" and sample.metadata["index"] == 5
        assert sample.metadata["audio"]["kind"] == "polyspeech"

    def test_spec_and_freeze_agree_on_keys(self, reconstruction, tmp_path):
        spec = tmp_path / "spec.yaml"
        _write_spec(spec, reconstruction, "pa_qa", ("deu_Latn", "de"), ("eng_Latn", "en"))
        records, per_source = read_sources(load_spec(spec))
        assert [r.sample_key for r in records] == ["00-000000", "00-000001", "01-000000"]
        assert [s["kept"] for s in per_source] == [2, 1]

    @pytest.mark.parametrize("variant", ["pa_qa", "pa_qt"])
    def test_real_inspect_run_scores_accuracy_per_language(self, reconstruction, tmp_path, variant):
        """Spec -> solver -> mock model -> ``mcq_scorer``, through a real ``inspect_ai.eval()``."""
        from inspect_ai import eval as inspect_eval
        from inspect_ai.model import ModelOutput

        from melteval.tasks import audio_mcq

        spec = tmp_path / "spec.yaml"
        _write_spec(spec, reconstruction, variant, ("deu_Latn", "de"), ("eng_Latn", "en"))
        # German: B is right, then a wrong letter (the answer is D); English: A is right.
        replies = ["B", "A", "A"]
        outputs = [ModelOutput.from_content(model="mockllm/model", content=r) for r in replies]
        [log] = inspect_eval(
            audio_mcq(spec=str(spec), prompt_style="hf", instruction="unused"),
            model="mockllm/model",
            model_args={"custom_outputs": outputs},
            max_connections=1,
            display="none",
            log_dir=str(tmp_path / "logs"),
        )
        metrics = {n: m.value for s in log.results.scores for n, m in s.metrics.items()}
        assert metrics[f"accuracy_polyspeech-{variant}-de"] == pytest.approx(0.5)
        assert metrics[f"accuracy_polyspeech-{variant}-en"] == pytest.approx(1.0)

        prompts = [m.text for s in log.samples for m in s.messages if m.role == "user"]
        assert all("A. " in p and "D. " in p for p in prompts)
        if variant == "pa_qa":
            assert not any(QUESTION in p or "Was ist" in p for p in prompts)
        else:
            assert QUESTION in prompts[0]
