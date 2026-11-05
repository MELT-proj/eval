"""Shared fixtures.

``spm_dir`` stages a toy SentencePiece model where sacrebleu's flores tokenizers
look for the real one, so spBLEU runs through the real code path -- sacrebleu's
own ``TokenizerSPM`` loading a file named as it expects -- with no network and
without the 4.8 MB production model. ``SACREBLEU_DIR`` is patched on
``sacrebleu.tokenizers.tokenizer_spm``, which binds it by value when sacrebleu is
imported; setting ``$SACREBLEU`` afterwards would do nothing.
"""

import io
from pathlib import Path

import pytest


#: The file names sacrebleu's tokenizers load from ``<SACREBLEU_DIR>/models/``.
#: Spelled out on purpose: a test that derived them from sacrebleu would pass
#: whatever sacrebleu decided to expect.
SPM_FILE_NAMES = {"flores200": "flores200sacrebleuspm", "flores101": "sacrebleu_tokenizer_spm.model"}

_TOY_CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Ich gehe nach Hause und lese ein gutes Buch.",
    "La ONU tiene la intención de completar un fondo de asistencia.",
    "Guten Tag, wie geht es Ihnen heute?",
    "Good day to you, and a good day to us all.",
    "The U.N. also hopes to finalize a fund to help countries.",
    "Das Wetter ist heute schön und die Sonne scheint.",
    "El gato come pescado en la cocina todos los días.",
] * 8


@pytest.fixture(scope="session")
def toy_spm_bytes() -> bytes:
    """A tiny SentencePiece model, trained once per session on a handful of sentences."""
    import sentencepiece as spm

    model = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(_TOY_CORPUS),
        model_writer=model,
        vocab_size=80,
        model_type="unigram",
        hard_vocab_limit=False,
        character_coverage=1.0,
    )
    return model.getvalue()


@pytest.fixture
def spm_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, toy_spm_bytes: bytes) -> Path:
    """Point sacrebleu at a temp directory holding the toy model under both file names."""
    root = tmp_path / "sacrebleu"
    (root / "models").mkdir(parents=True)
    for name in SPM_FILE_NAMES.values():
        (root / "models" / name).write_bytes(toy_spm_bytes)
    monkeypatch.setattr("sacrebleu.tokenizers.tokenizer_spm.SACREBLEU_DIR", str(root))
    return root


@pytest.fixture
def no_spm_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point sacrebleu at an empty directory, and fail loudly if anything tries to download."""
    root = tmp_path / "empty-sacrebleu"
    root.mkdir()
    monkeypatch.setattr("sacrebleu.tokenizers.tokenizer_spm.SACREBLEU_DIR", str(root))

    def no_download(*args, **kwargs):
        raise AssertionError(f"attempted a download: {args}")

    monkeypatch.setattr("sacrebleu.tokenizers.tokenizer_spm.download_file", no_download)
    return root
