"""Tests for ASR/ST scorers and their corpus-level metrics.

The central claim under test: corpus WER/BLEU is not the mean of per-sample
rates. Everything else here supports getting that one number right.

`TestStScorerThroughRealInspect` exists because of a real failure this design
already had once: hand-building `SampleScore` objects and calling a metric
function directly, as most tests below do, skips inspect's own scoring
pipeline entirely -- including the epoch-reduction step that goes through
every `Score.value` dict and applies `value_to_float()` to each entry. That
step is exactly what silently turned `st_scorer`'s reference/hypothesis text
into `0.0` when it lived in `Score.value` instead of `Score.metadata`, and no
amount of directly-constructed `SampleScore` tests could have caught it --
they all bypass the step that broke. Only a test that runs a real
`inspect_ai.eval()` exercises it.
"""

import asyncio

import pytest
from inspect_ai.model import ModelName, ModelOutput
from inspect_ai.scorer import SampleScore, Score, Target
from inspect_ai.solver import TaskState

from melteval.registry import GRADED_TASKS, TASK_SCORERS, default_scorer
from melteval.scorers import (
    chat_scorer,
    choice_accuracy,
    chunked_asr_scorer,
    chunked_bleu,
    chunked_cer,
    chunked_chrf,
    chunked_st_scorer,
    chunked_wer,
    corpus_bleu,
    corpus_cer,
    corpus_chrf,
    corpus_wer,
    get_normalizer,
    resolve_choice,
    unresolved_rate,
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
    """Build a `SampleScore` with a numeric `Score.value`, as `corpus_wer`/
    `corpus_cer` (and any real scorer) expect -- *not* a vehicle for text; see
    `_st_sample_score` for that."""
    return SampleScore(score=Score(value=value), sample_metadata=metadata or {})


def _st_sample_score(reference: str, hypothesis: str, metadata: dict | None = None) -> SampleScore:
    """Build a `SampleScore` shaped like `st_scorer`'s real output: text in
    `Score.metadata`, not `Score.value` -- see this module's docstring."""
    return SampleScore(
        score=Score(
            value={"hyp_tokens": len(hypothesis.split()), "ref_tokens": len(reference.split())},
            metadata={"reference": reference, "hypothesis": hypothesis},
        ),
        sample_metadata=metadata or {},
    )


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

    def test_records_the_pair_in_metadata_not_value(self, score_fn):
        """Not `Score.value`: see this module's docstring for why that
        specific placement is load-bearing, not a style choice."""
        state = _state("Hello there")
        result = _run(score_fn(state, Target("Hi there")))
        assert result.metadata == {"reference": "Hi there", "hypothesis": "Hello there"}

    def test_value_carries_token_lengths_not_text(self, score_fn):
        state = _state("Hello there")
        result = _run(score_fn(state, Target("Hi there you")))
        assert result.value == {"hyp_tokens": 2, "ref_tokens": 3}


class TestCorpusBleuChrf:
    def test_bleu_of_identical_text_is_near_100(self):
        scores = [
            _st_sample_score(
                "the cat sat on the mat", "the cat sat on the mat", {"tgt_lang": "en"}
            )
        ]
        assert corpus_bleu()(scores) > 99.0

    def test_chrf_does_not_need_a_language(self):
        scores = [_st_sample_score("hello", "hello")]
        assert corpus_chrf()(scores) > 99.0

    def test_zh_routes_to_the_zh_tokenizer(self):
        """Would raise from sacrebleu if word-splitting were applied to
        unsegmented Chinese instead of the zh tokenizer."""
        scores = [_st_sample_score("你好世界", "你好世界", {"tgt_lang": "zh"})]
        assert corpus_bleu()(scores) > 99.0

    def test_mixed_target_languages_raise(self):
        scores = [
            _st_sample_score("a", "a", {"tgt_lang": "en"}),
            _st_sample_score("b", "b", {"tgt_lang": "de"}),
        ]
        with pytest.raises(ValueError, match="Mixed target languages"):
            corpus_bleu()(scores)

    def test_falls_back_to_lang_when_tgt_lang_absent(self):
        """speechqe-style metadata may only carry `lang`, not `tgt_lang`."""
        scores = [_st_sample_score("a", "a", {"lang": "en"})]
        assert corpus_bleu()(scores) is not None

    def test_a_single_sample_does_not_crash(self):
        """The regression case: even n=1 hit the old bug, since the very
        first live metric update after one sample already sees whatever
        Score.value was reduced to."""
        scores = [_st_sample_score("Ich gehe.", "I go.", {"tgt_lang": "de"})]
        assert isinstance(corpus_bleu()(scores), float)


class TestStScorerThroughRealInspect:
    """Runs a real `inspect_ai.eval()` -- the only way to exercise inspect's
    own score-reduction step, which is what actually broke last time."""

    def test_corpus_bleu_survives_a_real_eval_run(self, tmp_path):
        from inspect_ai import Task
        from inspect_ai import eval as inspect_eval
        from inspect_ai.dataset import MemoryDataset, Sample
        from inspect_ai.model import ModelOutput as _ModelOutput

        from melteval.scorers import st_scorer

        dataset = MemoryDataset(
            samples=[
                Sample(
                    input="translate",
                    target="Ich gehe.",
                    id="00-0",
                    metadata={"tgt_lang": "de", "lang": "de"},
                ),
                Sample(
                    input="translate",
                    target="Guten Tag.",
                    id="00-1",
                    metadata={"tgt_lang": "de", "lang": "de"},
                ),
            ]
        )
        task = Task(dataset=dataset, scorer=st_scorer())

        [log] = inspect_eval(
            task,
            model="mockllm/model",
            model_args={
                "custom_outputs": [
                    _ModelOutput.from_content(model="mockllm/model", content="I go."),
                    _ModelOutput.from_content(model="mockllm/model", content="Good day."),
                ]
            },
            display="none",
            log_dir=str(tmp_path),
        )

        assert log.status == "success"
        metrics = log.results.scores[0].metrics
        assert "corpus_bleu" in metrics
        assert isinstance(metrics["corpus_bleu"].value, float)


def _grouped_score(
    reference: str, hypothesis: str, group_id: str, group_order: int, **metadata
) -> SampleScore:
    """A `SampleScore` shaped like `chunked_asr_scorer`/`chunked_st_scorer`'s
    real output: the (reference, hypothesis) pair in `Score.metadata`, and the
    grouping key in `sample_metadata`, exactly where `_grouped_pairs` reads it
    from."""
    metadata.setdefault("dataset_id", "d")
    return SampleScore(
        score=Score(
            value={"hyp_tokens": len(hypothesis.split()), "ref_tokens": len(reference.split())},
            metadata={"reference": reference, "hypothesis": hypothesis},
        ),
        sample_metadata={"group_id": group_id, "group_order": group_order, **metadata},
    )


class TestChunkedScorers:
    """MCIF's `short` track cuts one long reference across many samples; these
    metrics have to reassemble them before scoring, not score each on its own
    against the whole reference."""

    def test_wer_is_computed_over_the_joined_group_not_per_chunk(self):
        scores = [
            _grouped_score("hello world", "hello", "g1", 0),
            _grouped_score("hello world", "world", "g1", 1),
        ]
        assert chunked_wer()(scores) == pytest.approx(0.0)

    def test_group_order_does_not_depend_on_input_order(self):
        """Chunks can arrive from inspect in any order; the group's own
        `group_order` -- not list position -- decides the join order."""
        in_order = [
            _grouped_score("hello world", "hello", "g1", 0),
            _grouped_score("hello world", "world", "g1", 1),
        ]
        reversed_input = list(reversed(in_order))
        assert chunked_wer()(reversed_input) == chunked_wer()(in_order)

    def test_wrong_join_order_would_have_scored_worse(self):
        """Sanity check that the test above is not vacuous: joining out of
        order genuinely changes the WER, so getting the order right matters."""
        scores = [
            _grouped_score("the cat sat", "sat", "g1", 0),
            _grouped_score("the cat sat", "the cat", "g1", 1),
        ]
        assert chunked_wer()(scores) > 0.0

    def test_a_sample_without_grouping_metadata_is_its_own_group(self):
        """No `group_id` -- e.g. a corpus that never needed grouping -- must
        still score, one sample at a time, exactly like `corpus_wer` would."""
        scores = [
            SampleScore(
                score=Score(value={}, metadata={"reference": "hi", "hypothesis": "hi"}),
                sample_metadata={"dataset_id": "d"},
                sample_id="s0",
            )
        ]
        assert chunked_wer()(scores) == pytest.approx(0.0)

    def test_distinct_references_in_one_group_raise(self):
        """Two chunks placed in the same group must agree on the reference --
        disagreement means a grouping bug, not a scoring question."""
        scores = [
            _grouped_score("ref one", "a", "g1", 0),
            _grouped_score("ref two", "b", "g1", 1),
        ]
        with pytest.raises(ValueError, match="distinct references"):
            chunked_wer()(scores)

    def test_same_group_id_in_different_corpora_does_not_collide(self):
        """`dataset_id` is part of the grouping key, since two unrelated
        corpora scored in one run could otherwise share a group label. If
        these two were merged into one group of two members, they would
        disagree on the reference and `chunked_wer` would raise; scored as
        the two separate one-member groups they actually are, "alpha" is
        perfect (0 errors / 2 words) and "beta" is entirely wrong (2 errors /
        2 words), for a corpus WER of 2/4."""
        scores = [
            _grouped_score("hello world", "hello world", "g1", 0, dataset_id="alpha"),
            _grouped_score("goodnight moon", "wrong", "g1", 0, dataset_id="beta"),
        ]
        assert chunked_wer()(scores) == pytest.approx(0.5)

    def test_empty_corpus_is_zero_not_a_division_error(self):
        assert chunked_wer()([]) == 0.0
        assert chunked_bleu()([]) == 0.0

    def test_cer_uses_the_joined_group_too(self):
        """Chunks are joined with a space (like the official MCIF evaluation
        script joins per-segment hypotheses), so the reference here has one
        too -- a reference with no word boundary at the join point is a
        different, unrelated question from what this test is checking."""
        scores = [
            _grouped_score("a b", "a", "g1", 0),
            _grouped_score("a b", "b", "g1", 1),
        ]
        assert chunked_cer()(scores) == pytest.approx(0.0)

    def test_bleu_and_chrf_score_the_joined_hypothesis(self):
        scores = [
            _grouped_score("the cat sat on the mat", "the cat sat", "g1", 0, tgt_lang="en"),
            _grouped_score("the cat sat on the mat", "on the mat", "g1", 1, tgt_lang="en"),
        ]
        assert chunked_bleu()(scores) > 99.0
        assert chunked_chrf()(scores) > 99.0

    def test_asr_scorer_records_the_pair_like_st_scorer_does(self):
        score_fn = chunked_asr_scorer()
        result = _run(score_fn(_state("hello"), Target("hello world")))
        assert result.metadata == {"reference": "hello world", "hypothesis": "hello"}

    def test_st_scorer_records_the_pair_like_st_scorer_does(self):
        score_fn = chunked_st_scorer()
        result = _run(score_fn(_state("I go"), Target("Ich gehe")))
        assert result.metadata == {"reference": "Ich gehe", "hypothesis": "I go"}


class TestChunkedScorersThroughRealInspect:
    """As with `TestStScorerThroughRealInspect`: only a real `inspect_ai.eval()`
    run exercises epoch-reduction, which is what a text payload in
    `Score.value` would silently break."""

    def test_two_chunks_reassemble_into_one_correct_group(self, tmp_path):
        from inspect_ai import Task
        from inspect_ai import eval as inspect_eval
        from inspect_ai.dataset import MemoryDataset, Sample
        from inspect_ai.model import ModelOutput as _ModelOutput

        dataset = MemoryDataset(
            samples=[
                Sample(
                    input="transcribe",
                    target="hello world",
                    id="00-0",
                    metadata={"dataset_id": "mcif-short-fixed-en", "group_id": "ASR_1", "group_order": 0},
                ),
                Sample(
                    input="transcribe",
                    target="hello world",
                    id="00-1",
                    metadata={"dataset_id": "mcif-short-fixed-en", "group_id": "ASR_1", "group_order": 1},
                ),
            ]
        )
        task = Task(dataset=dataset, scorer=chunked_asr_scorer())

        [log] = inspect_eval(
            task,
            model="mockllm/model",
            model_args={
                "custom_outputs": [
                    _ModelOutput.from_content(model="mockllm/model", content="hello"),
                    _ModelOutput.from_content(model="mockllm/model", content="world"),
                ]
            },
            display="none",
            log_dir=str(tmp_path),
        )

        assert log.status == "success"
        metrics = log.results.scores[0].metrics
        assert metrics["chunked_wer"].value == pytest.approx(0.0)


class TestRegistry:
    def test_speech_and_mcq_tasks_have_argument_free_defaults(self):
        assert set(TASK_SCORERS) == {"asr", "st", "chunked_asr", "chunked_st", "audio_mcq"}

    def test_a_graded_task_is_not_in_the_argument_free_table(self):
        """audio_chat cannot be built without being told which model judges,
        so it is constructed in default_scorer() rather than looked up here."""
        assert "audio_chat" not in TASK_SCORERS
        assert "audio_chat" in GRADED_TASKS

    def test_a_graded_task_resolves_once_a_judge_is_named(self):
        assert default_scorer("audio_chat", "mockllm/model") is not None

    def test_a_graded_task_without_a_judge_still_builds(self):
        """It refuses when it scores, not when it is built -- see
        TestChatScorer."""
        assert default_scorer("audio_chat") is not None

    def test_mcq_resolves_to_the_choice_scorer(self):
        assert default_scorer("audio_mcq") is not None

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


# =============================================================================
# Multiple choice
# =============================================================================


class TestResolveChoice:
    """What counts as "the model picked option X"."""

    CHOICES = ["Male", "Female", "Nonbinary", "Unclear"]

    def test_a_bare_letter(self):
        assert resolve_choice("B", self.CHOICES) == "Female"

    def test_a_letter_with_punctuation(self):
        assert resolve_choice("(b) Female", self.CHOICES) == "Female"
        assert resolve_choice("B. Female", self.CHOICES) == "Female"
        assert resolve_choice("b: the speaker is female", self.CHOICES) == "Female"

    def test_lowercase_letters_count(self):
        assert resolve_choice("d", self.CHOICES) == "Unclear"

    def test_a_letter_past_the_end_is_not_a_choice(self):
        """Only four options were shown; "E" names none of them."""
        assert resolve_choice("E", self.CHOICES) is None

    def test_a_capitalised_label_the_answer_ends_on(self):
        """The most natural way an instruction-following model answers."""
        assert resolve_choice("The answer is B.", self.CHOICES) == "Female"
        assert resolve_choice("I would say (C)", self.CHOICES) == "Nonbinary"

    def test_a_trailing_lowercase_letter_is_not_a_label(self):
        """Otherwise a sentence ending in the article "a" scores as option A."""
        assert resolve_choice("it sounds like a", self.CHOICES) is None

    def test_the_option_text_alone(self):
        assert resolve_choice("Female", self.CHOICES) == "Female"

    def test_option_text_inside_a_sentence(self):
        assert resolve_choice("I think the speaker is female.", self.CHOICES) == "Female"

    def test_matching_ignores_case_and_punctuation(self):
        assert resolve_choice("  female!  ", self.CHOICES) == "Female"

    def test_echoing_every_option_is_not_an_answer(self):
        """Crediting the first option mentioned would turn a non-answer into a
        coin flip weighted by option order."""
        assert resolve_choice("Male, Female, Nonbinary or Unclear?", self.CHOICES) is None

    def test_an_unrelated_completion_resolves_to_nothing(self):
        assert resolve_choice("the weather is nice today", self.CHOICES) is None

    def test_an_empty_completion_resolves_to_nothing(self):
        assert resolve_choice("", self.CHOICES) is None

    def test_no_choices_at_all_resolves_to_nothing(self):
        assert resolve_choice("B", []) is None

    def test_a_leading_word_is_not_read_as_a_letter(self):
        """"a bird singing" starts with "a" but is not answering "A"."""
        assert resolve_choice("a bird singing", ["dog", "bird", "cat"]) == "bird"


class TestMcqMetrics:
    def _sample_score(self, correct: int, resolved: int) -> SampleScore:
        return SampleScore(score=Score(value={"correct": correct, "resolved": resolved}))

    def test_accuracy_is_the_share_correct(self):
        scores = [self._sample_score(1, 1), self._sample_score(0, 1), self._sample_score(1, 1)]
        assert choice_accuracy()(scores) == pytest.approx(2 / 3)

    def test_unresolved_rate_counts_unreadable_completions(self):
        scores = [self._sample_score(1, 1), self._sample_score(0, 0), self._sample_score(0, 0)]
        assert unresolved_rate()(scores) == pytest.approx(2 / 3)

    def test_a_wrong_answer_is_not_an_unresolved_one(self):
        """Both score 0 on accuracy; only one says the prompt was misunderstood."""
        scores = [self._sample_score(0, 1)]
        assert choice_accuracy()(scores) == 0.0
        assert unresolved_rate()(scores) == 0.0

    def test_empty_corpus_does_not_divide_by_zero(self):
        assert choice_accuracy()([]) == 0.0
        assert unresolved_rate()([]) == 0.0


class TestChatScorer:
    def test_constructing_without_a_judge_is_allowed(self):
        """So that `inspect eval --no-score` can run on a cluster that cannot
        reach a judge -- generate now, score the log later."""
        assert chat_scorer(None) is not None

    def test_scoring_without_a_judge_is_refused(self):
        """No lexical fallback on purpose: it would rank fluent wrong answers
        above terse right ones."""
        with pytest.raises(ValueError, match="grader_model"):
            _run(chat_scorer(None)(_state("an answer"), Target("the reference")))

    def test_builds_a_graded_scorer_when_given_one(self):
        assert chat_scorer("mockllm/model") is not None
