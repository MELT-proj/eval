"""Locate and verify the SentencePiece model behind spBLEU.

sacrebleu's ``flores200``/``flores101`` tokenizers load a SentencePiece model
from ``$SACREBLEU/models/`` (default ``~/.sacrebleu``) and, if the file is not
there, **download it** -- inside ``BLEU.__init__``, i.e. at the end of a run, in
the scoring step, on a node that may have no internet at all. That is the worst
moment to learn the file is missing: the generation is already spent.

So this module does two things sacrebleu does not. It answers "is the model
where sacrebleu will look, and does it load?" up front (:func:`require_spm_model`,
called when the ST task is built, and again before any metric touches
sacrebleu), and it never lets a download happen: a missing file is an error that
says how to stage it, not a fetch. The path is read from sacrebleu's own
constants rather than re-derived, so it cannot drift from where the tokenizer
really looks.
"""

from __future__ import annotations

import os
from pathlib import Path


#: sacrebleu tokenizer names that need the SentencePiece model, default first.
SPM_TOKENIZERS = ("flores200", "flores101")


class SPMModelError(RuntimeError):
    """The SentencePiece model spBLEU needs is missing, unreadable or unloadable."""


def spm_model_path(tokenizer: str = "flores200") -> Path:
    """Return the path sacrebleu's *tokenizer* loads its SentencePiece model from.

    Read at call time from ``sacrebleu.tokenizers.tokenizer_spm`` -- the module
    that binds ``SACREBLEU_DIR`` by value when sacrebleu is first imported -- so
    this is the directory the tokenizer will actually use, including when a
    test patches that attribute. Setting ``$SACREBLEU`` after sacrebleu has been
    imported changes nothing, for sacrebleu and for this function alike.

    Raises:
        ValueError: If *tokenizer* is not one of :data:`SPM_TOKENIZERS`.
    """
    if tokenizer not in SPM_TOKENIZERS:
        raise ValueError(f"spBLEU tokenizer must be one of {SPM_TOKENIZERS}, got {tokenizer!r}.")
    from sacrebleu.tokenizers import tokenizer_spm

    url = tokenizer_spm.SPM_MODELS[tokenizer]["url"]
    return Path(tokenizer_spm.SACREBLEU_DIR) / "models" / os.path.basename(url)


def require_spm_model(tokenizer: str = "flores200") -> Path:
    """Check that the SentencePiece model for *tokenizer* is present and loads.

    Args:
        tokenizer: ``flores200`` (default) or ``flores101``.

    Returns:
        The path of the model file.

    Raises:
        SPMModelError: If ``sentencepiece`` is not installed, the file does not
            exist, or it exists but ``sentencepiece`` cannot load it. The
            message names the file, ``SACREBLEU`` and how to stage it.
        ValueError: If *tokenizer* is unknown.
    """
    from sacrebleu.tokenizers import tokenizer_spm

    path = spm_model_path(tokenizer)
    url = tokenizer_spm.SPM_MODELS[tokenizer]["url"]
    staging = (
        f"spBLEU ({tokenizer}) needs the SentencePiece model {path}. sacrebleu looks for it under "
        f"$SACREBLEU (here {tokenizer_spm.SACREBLEU_DIR}; unset means ~/.sacrebleu), read once when "
        "sacrebleu is imported. Stage it once from a machine with internet, never from a job: "
        f"mkdir -p {path.parent} && curl -L -o {path} {url} "
        "(see README.md, \"spBLEU\"). To run without spBLEU instead, pass -T spbleu=false."
    )

    try:
        import sentencepiece
    except ImportError as exc:
        raise SPMModelError(f"sentencepiece is not installed (`uv sync --extra metrics`). {staging}") from exc

    if not path.is_file():
        raise SPMModelError(f"SentencePiece model not found: {path}. {staging}")
    try:
        sentencepiece.SentencePieceProcessor().Load(str(path))
    except (OSError, RuntimeError) as exc:
        raise SPMModelError(f"SentencePiece model {path} exists but does not load ({exc}). {staging}") from exc
    return path
