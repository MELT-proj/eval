"""melteval's copy of the WER normalizers must stay identical to the training package's.

WER is only comparable across models if every run normalizes with the same
code. melteval carries its own copy (``melteval/normalizers``) so SMURF and
baseline environments, which cannot install ``melt-proj``, still use it. These
tests pin that the copy has not drifted -- byte for byte, and on output --
whenever the training package is importable to compare against.
"""

import hashlib
from pathlib import Path

import pytest

from melteval.scorers import get_normalizer


FILES = ("basic.py", "english.py", "english.json")

#: Strings chosen to exercise each normalizer's rules: numbers and currency,
#: British spellings, contractions, fillers, bracketed noise, diacritics,
#: non-Latin scripts and stray punctuation.
SAMPLES = [
    "Hello, World!",
    "I've got twenty-one $3.50 coffees, it's 10:30 a.m.",
    "The colour of the centre was grey (laughs) [noise]",
    "Umm, uh, hmm... I mean, like, you know?",
    "Mr. Smith paid £1,000,000 on 1st January 2020; that's 50%.",
    "Ça, c'est très élégant — Æsop's œuvre, Straße, łódź",
    "Das ist 1 Test mit Umlauten: äöü ÄÖÜ ß",
    "这是一个测试。 Это тест. これはテストです",
    "  multiple   spaces\tand\nnewlines  ",
    "one hundred and five point two, 1/2, 3rd, 21st, 1990s",
]


def _melt_normalizers():
    return pytest.importorskip("melt.evaluation.normalizers")


def test_files_are_byte_identical_to_the_training_package():
    melt_normalizers = _melt_normalizers()
    ours = Path(__file__).resolve().parents[1] / "melteval" / "normalizers"
    theirs = Path(melt_normalizers.__file__).parent
    for name in FILES:
        assert hashlib.sha256((ours / name).read_bytes()).hexdigest() == hashlib.sha256(
            (theirs / name).read_bytes()
        ).hexdigest(), f"melteval/normalizers/{name} differs from {theirs / name}"


@pytest.mark.parametrize("name, cls", [("basic", "BasicTextNormalizer"), ("english", "EnglishTextNormalizer")])
def test_output_matches_the_training_package(name, cls):
    melt_normalizers = _melt_normalizers()
    ours, theirs = get_normalizer(name), getattr(melt_normalizers, cls)()
    for text in SAMPLES:
        assert ours(text) == theirs(text), text


@pytest.mark.parametrize("name", ["basic", "english"])
def test_available_without_the_training_package(name):
    """The point of the copy: this path imports nothing from melt."""
    import sys

    already_loaded = {m for m in sys.modules if m == "melt" or m.startswith("melt.")}
    normalize = get_normalizer(name)
    assert normalize("Hello, World!").strip() == "hello world"
    assert {m for m in sys.modules if m == "melt" or m.startswith("melt.")} == already_loaded
