"""Tests for ASR/ST scorers and their corpus-level metrics.

The central claim under test: corpus WER/BLEU is not the mean of per-sample
rates. Everything else here supports getting that one number right.
"""

import asyncio

import pytest
from inspect_ai.model import ModelName, ModelOutput
from inspect_ai.scorer import SampleScore, Score, Target
from inspect_ai.solver import TaskState

from melteval.registry import TASK_SCORERS, default_scorer
from melteval.scorers import (
    corpus_bleu,
    corpus_cer,
    corpus_chrf,
    corpus_wer,
    get_normalizer,
)


pytest.importorskip("melt.evaluation")


def _state(completion: str, sample_id: str = "k") -> TaskState:
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id=sample_id,
        epoch=1,
        input=completion,
        messages=[],
        output=ModelOutput.from_content(model="mockllm/model", content=completion),
    )


def _run(coro):
    return asyncio.run(coro)


def _sample_score(value: dict, metadata: dict | None = None) -> SampleScore:
    return SampleScore(score=Score(value=value), sample_metadata=metadata or {})


class TestNormalizers:
    def test_basic_lowercases_and_strips_punctuation(self):
        normalize = get_normalizer("basic")
        assert normalize("Hello, World!").strip() == normalize("hello world").strip()

    def test_none_is_identity(self):
        assert get_normalizer("none")("Hello, World!") == "Hello, World!"

    def test_unknown_normalizer_raises(self):
        with pytest.raises(ValueError, match="Unknown normalizer"):
            get_normalizer("whisper-large")


class TestAsrScorer:
    """`asr_scorer()` is the factory; `score` is the returned async scorer."""

    @pytest.fixture
    def score_fn(self):
        from melteval.scorers import asr_scorer

        return asr_scorer(normalizer="basic")

    def test_perfect_match_has_no_errors(self, score_fn):
        state = _state("the cat sat on the mat")
        result = _run(score_fn(state, Target("the cat sat on the mat")))
        assert result.value["wer_errors"] == 0
        assert result.value["ref_words"] == 6

    def test_counts_are_raw_not_a_rate(self, score_fn):
        """A one-word substitution out of six must show up as 1/6, not 0.1667
        pre-divided -- corpus_wer does the division, once, over the corpus.
        "rug" -> "mat" is 3 character substitutions, not 1 -- CER operates on
        the whole string, not word-aligned."""
        state = _state("the cat sat on the rug")
        result = _run(score_fn(state, Target("the cat sat on the mat")))
        assert result.value == {"wer_errors": 1, "ref_words": 6, "cer_errors": 3, "ref_chars": 22}

    def test_both_sides_are_normalized_symmetrically(self, score_fn):
        """run_inference.py's bug: it lowercased only the reference. A
        hypothesis differing only in case must score as a perfect match."""
        state = _state("THE CAT SAT")
        result = _run(score_fn(state, Target("the cat sat")))
        assert result.value["wer_errors"] == 0

    def test_empty_reference_is_excluded_not_crashed(self, score_fn):
        state = _state("some hypothesis")
        result = _run(score_fn(state, Target("   ")))
        assert result.value == {"wer_errors": 0, "ref_words": 0, "cer_errors": 0, "ref_chars": 0}

    def test_none_normalizer_is_case_sensitive(self):
        from melteval.scorers import asr_scorer

        score_fn = asr_scorer(normalizer="none")
        state = _state("THE CAT SAT")
        result = _run(score_fn(state, Target("the cat sat")))
        assert result.value["wer_errors"] == 3


class TestCorpusWerCer:
    def test_sums_errors_over_reference_units(self):
        scores = [
            _sample_score({"wer_errors": 1, "ref_words": 10, "cer_errors": 0, "ref_chars": 0}),
            _sample_score({"wer_errors": 1, "ref_words": 2, "cer_errors": 0, "ref_chars": 0}),
        ]
        assert corpus_wer()(scores) == pytest.approx(2 / 12)

    def test_differs_from_the_mean_of_per_sample_rates(self):
        """The architectural point of this module: a short bad sample must not
        dominate a corpus dominated by long good ones."""
        scores = [
            _sample_score({"wer_errors": 5, "ref_words": 5, "cer_errors": 0, "ref_chars": 0}),
            _sample_score({"wer_errors": 0, "ref_words": 995, "cer_errors": 0, "ref_chars": 0}),
        ]
        mean_of_rates = (1.0 + 0.0) / 2
        corpus_rate = corpus_wer()(scores)
        assert corpus_rate == pytest.approx(5 / 1000)
        assert corpus_rate != pytest.approx(mean_of_rates)

    def test_empty_corpus_is_zero_not_a_division_error(self):
        assert corpus_wer()([]) == 0.0

    def test_cer_uses_its_own_counters(self):
        scores = [_sample_score({"wer_errors": 9, "ref_words": 9, "cer_errors": 1, "ref_chars": 20})]
        assert corpus_cer()(scores) == pytest.approx(1 / 20)


class TestStScorer:
    @pytest.fixture
    def score_fn(self):
        from melteval.scorers import st_scorer

        return st_scorer()

    def test_records_the_pair_verbatim(self, score_fn):
        state = _state("Hello there")
        result = _run(score_fn(state, Target("Hi there")))
        assert result.value == {"reference": "Hi there", "hypothesis": "Hello there"}


class TestCorpusBleuChrf:
    def test_bleu_of_identical_text_is_near_100(self):
        scores = [
            _sample_score(
                {"reference": "the cat sat on the mat", "hypothesis": "the cat sat on the mat"},
                {"tgt_lang": "en"},
            )
        ]
        assert corpus_bleu()(scores) > 99.0

    def test_chrf_does_not_need_a_language(self):
        scores = [_sample_score({"reference": "hello", "hypothesis": "hello"})]
        assert corpus_chrf()(scores) > 99.0

    def test_zh_routes_to_the_zh_tokenizer(self):
        """Would raise from sacrebleu if word-splitting were applied to
        unsegmented Chinese instead of the zh tokenizer."""
        scores = [_sample_score({"reference": "你好世界", "hypothesis": "你好世界"}, {"tgt_lang": "zh"})]
        assert corpus_bleu()(scores) > 99.0

    def test_mixed_target_languages_raise(self):
        scores = [
            _sample_score({"reference": "a", "hypothesis": "a"}, {"tgt_lang": "en"}),
            _sample_score({"reference": "b", "hypothesis": "b"}, {"tgt_lang": "de"}),
        ]
        with pytest.raises(ValueError, match="Mixed target languages"):
            corpus_bleu()(scores)

    def test_falls_back_to_lang_when_tgt_lang_absent(self):
        """speechqe-style metadata may only carry `lang`, not `tgt_lang`."""
        scores = [_sample_score({"reference": "a", "hypothesis": "a"}, {"lang": "en"})]
        assert corpus_bleu()(scores) is not None


class TestRegistry:
    def test_asr_and_st_have_defaults(self):
        assert set(TASK_SCORERS) == {"asr", "st"}

    def test_no_task_filter_falls_back_to_a_plumbing_check(self):
        """Not a real metric -- just proves the default_scorer() doesn't need
        a task filter to return *something* runnable."""
        scorer = default_scorer(None)
        assert scorer is not None

    def test_known_task_filter_resolves(self):
        assert default_scorer("asr") is not None
        assert default_scorer("st") is not None

    def test_unknown_task_filter_raises_rather_than_guessing(self):
        with pytest.raises(ValueError, match="No default scorer"):
            default_scorer("ars")  # typo of "asr"
