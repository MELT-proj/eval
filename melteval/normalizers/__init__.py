"""Whisper's text normalizers, as MELT's training package ships them.

``basic.py``, ``english.py`` and ``english.json`` are byte-for-byte copies of
``melt/evaluation/normalizers/`` in MELT-proj/training (itself taken verbatim
from https://github.com/openai/whisper/tree/main/whisper/normalizers, MIT).
They live here too so that every environment -- MELT's, SMURF's, a baseline's
-- scores WER with the *same* code without needing ``melt-proj`` installed:
the SMURF/NeMo stack cannot co-install it, and a model evaluated there with a
different (or no) normalizer would report a WER that is not comparable.

Do not edit these files. ``tests/test_normalizers.py`` checks that they are
still identical to the training package's copy whenever that is importable,
and that both produce the same output on a fixed set of strings.
"""

from .basic import BasicTextNormalizer as BasicTextNormalizer
from .english import EnglishTextNormalizer as EnglishTextNormalizer
