"""Tests for locating the spBLEU SentencePiece model, and for failing before generation without it."""

from pathlib import Path

import pytest

from melteval.manifest import MANIFEST_NAME, AudioLocator, EvalRecord, write_manifest
from melteval.spm import SPMModelError, require_spm_model, spm_model_path
from tests.conftest import SPM_FILE_NAMES


class TestModelPath:
    def test_is_where_sacrebleu_looks(self, spm_dir):
        assert spm_model_path("flores200") == spm_dir / "models" / SPM_FILE_NAMES["flores200"]
        assert spm_model_path("flores101") == spm_dir / "models" / SPM_FILE_NAMES["flores101"]

    def test_the_model_sacrebleu_loads_is_the_one_checked(self, spm_dir):
        """Not just a plausible path: sacrebleu's own tokenizer must load from it."""
        import sacrebleu

        path = require_spm_model("flores200")
        assert path.is_file()
        sacrebleu.BLEU(tokenize="flores200")  # would try to download if the path were wrong

    def test_unknown_tokenizer_raises(self):
        with pytest.raises(ValueError, match="flores200"):
            spm_model_path("13a")


class TestRequireSpmModel:
    def test_present_model_passes(self, spm_dir):
        assert require_spm_model() == spm_dir / "models" / SPM_FILE_NAMES["flores200"]

    def test_missing_model_names_the_file_the_variable_and_the_fix(self, no_spm_dir):
        with pytest.raises(SPMModelError) as err:
            require_spm_model()
        message = str(err.value)
        assert str(no_spm_dir / "models" / "flores200sacrebleuspm") in message
        assert "SACREBLEU" in message
        assert "curl" in message and "spbleu=false" in message

    def test_a_file_that_is_not_a_model_is_reported_not_loaded_silently(self, no_spm_dir):
        models = no_spm_dir / "models"
        models.mkdir()
        (models / "flores200sacrebleuspm").write_bytes(b"not a sentencepiece model")
        with pytest.raises(SPMModelError, match="does not load"):
            require_spm_model()

    def test_never_downloads(self, no_spm_dir):
        """`no_spm_dir` makes sacrebleu's download_file raise AssertionError if reached."""
        with pytest.raises(SPMModelError):
            require_spm_model()


def _frozen_st_set(tmp_path: Path) -> Path:
    write_manifest(
        tmp_path / MANIFEST_NAME,
        [
            EvalRecord(
                sample_key="00-000000",
                task="st",
                target="Ich gehe.",
                audio=AudioLocator(kind="shar", params={"dir": "/d", "index": 1}),
                lang="de",
                dataset_id="fleurs-en_de",
                duration=2.0,
            )
        ],
    )
    return tmp_path


class TestFailBeforeGeneration:
    """Building the task is the last point before the model is loaded and samples are generated."""

    def test_st_task_construction_stops_when_the_model_is_missing(self, no_spm_dir, tmp_path):
        from melteval.tasks import st

        with pytest.raises(SPMModelError, match="flores200sacrebleuspm"):
            st(str(_frozen_st_set(tmp_path)))

    def test_the_check_comes_before_the_dataset_is_read(self, no_spm_dir):
        """A bad frozen-set path would raise something else if the dataset were built first."""
        from melteval.tasks import st

        with pytest.raises(SPMModelError):
            st("/nonexistent/frozen-set")

    def test_st_task_builds_when_the_model_is_staged(self, spm_dir, tmp_path):
        from melteval.tasks import st

        assert st(str(_frozen_st_set(tmp_path))) is not None

    def test_the_explicit_opt_out_builds_without_the_model(self, no_spm_dir, tmp_path):
        from melteval.tasks import st

        assert st(str(_frozen_st_set(tmp_path)), spbleu=False) is not None

    def test_asr_never_needs_the_model(self, no_spm_dir, tmp_path):
        from melteval.tasks import asr

        write_manifest(
            tmp_path / MANIFEST_NAME,
            [
                EvalRecord(
                    sample_key="00-000000",
                    task="asr",
                    target="hello",
                    audio=AudioLocator(kind="shar", params={"dir": "/d", "index": 1}),
                    lang="en",
                    dataset_id="librispeech",
                    duration=2.0,
                )
            ],
        )
        assert asr(str(tmp_path)) is not None

    def test_opt_out_is_refused_for_other_tasks(self, no_spm_dir, tmp_path):
        from melteval.tasks import asr

        with pytest.raises(ValueError, match="spbleu=False"):
            asr(str(tmp_path), spbleu=False)


class TestRealModel:
    """Against the production flores200 model; skipped where it is not staged (it is never committed)."""

    SHA256 = "14bb8dfb35c0ffdea7bc01e56cea38b9e3d5efcdcb9c251d6b40538e1aab555a"

    @pytest.fixture
    def real_model(self):
        path = spm_model_path("flores200")
        if not path.is_file():
            pytest.skip(f"flores200 SentencePiece model not staged at {path}")
        return path

    def test_is_the_documented_file(self, real_model):
        import hashlib

        assert hashlib.sha256(real_model.read_bytes()).hexdigest() == self.SHA256

    def test_loads_and_tokenizes(self, real_model):
        import sacrebleu

        assert require_spm_model("flores200") == real_model
        bleu = sacrebleu.BLEU(tokenize="flores200")
        assert bleu.corpus_score(["Ich gehe nach Hause."], [["Ich gehe nach Hause."]]).score > 99.0
