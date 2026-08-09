"""ASR and ST scorers with corpus-level metrics.

**Corpus metrics are not the mean of per-sample metrics**, and the difference
is not a rounding detail. WER is total edit operations over total reference
words across the whole corpus; the mean of per-sample WERs weights every
sample equally regardless of length, so it is systematically distorted by
short samples. Same story for BLEU, which is not meaningfully defined
per-sentence at all (its brevity penalty depends on the corpus's total
length).

So every scorer here stores raw counts in ``Score.value`` — error counts and
reference-unit counts for ASR — and the corpus-level number is a
:class:`Metric` that sums over the whole ``list[SampleScore]``. Reading
``score.value["wer_errors"]`` off one sample is not "that sample's error
rate" in any meaningful sense; the raw count exists so the corpus metric has
something to reduce over.

ST's hypothesis/reference pairs live in ``Score.metadata`` instead, **not**
``Score.value``, and that split is load-bearing rather than a style choice:
``Score.value`` goes through inspect's epoch-reduction machinery even for a
single-epoch run, and a dict there is treated as named *numeric* sub-scores —
``value_to_float()`` is applied to every entry. For ASR's integer counts that
happens to be harmless (an int survives being read as a float). For ST's
*text* it silently replaces every reference and hypothesis with ``0.0``
(logging one "Unable to convert value to float" warning per string), and
``corpus_bleu``/``corpus_chrf`` then hand sacrebleu a batch of zeros, which
sacrebleu correctly refuses: ``TypeError: BLEU: refs should be a sequence of
sequence of strings``, and it happens from the very first sample scored, not
just at scale. Found by actually running a batch through it (both on real
hardware and reproduced locally with `mockllm`, since the failure is in
metric computation, not generation) — a scorer unit test that hand-builds
`SampleScore` objects and calls the metric directly, as this module's tests
originally did, bypasses inspect's reduction step entirely and cannot see
this. ``Score.metadata`` carries arbitrary ``dict[str, Any]`` and is never
touched by that reduction, which is why it is the right place for text.

Metrics are attached to the scorers themselves (rather than left for a task to
wire up) so ``inspect eval melteval/tasks.py@speech`` reports corpus WER/CER or
BLEU/chrF, with a per-language breakdown, with no extra flags.
"""

from __future__ import annotations

from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    Value,
    grouped,
    metric,
    scorer,
)
from inspect_ai.solver import TaskState


# =============================================================================
# Text normalization
# =============================================================================

_NORMALIZER_NAMES = ("basic", "english", "none")


def get_normalizer(name: str):
    """Return a ``str -> str`` normalizer by name.

    Args:
        name: ``"basic"`` (Whisper-style, language-agnostic),
            ``"english"`` (adds English number/spelling normalization), or
            ``"none"`` (score raw strings).

    Raises:
        ValueError: If *name* is not one of the above.
    """
    if name == "none":
        return lambda text: text
    if name == "basic":
        from melt.evaluation import BasicTextNormalizer

        return BasicTextNormalizer()
    if name == "english":
        from melt.evaluation import EnglishTextNormalizer

        return EnglishTextNormalizer()
    raise ValueError(f"Unknown normalizer {name!r}. Expected one of: {_NORMALIZER_NAMES}")


# =============================================================================
# ASR: WER / CER
# =============================================================================


def _sum_ratio(scores: list[SampleScore], error_key: str, total_key: str) -> Value:
    """Sum two counters across samples and divide, guarding an empty corpus."""
    errors = sum(s.score.value[error_key] for s in scores)
    total = sum(s.score.value[total_key] for s in scores)
    return errors / total if total > 0 else 0.0


@metric
def corpus_wer() -> Metric:
    """Corpus WER: total word errors over total reference words."""

    def calculate(scores: list[SampleScore]) -> Value:
        return _sum_ratio(scores, "wer_errors", "ref_words")

    return calculate


@metric
def corpus_cer() -> Metric:
    """Corpus CER: total character errors over total reference characters."""

    def calculate(scores: list[SampleScore]) -> Value:
        return _sum_ratio(scores, "cer_errors", "ref_chars")

    return calculate


@scorer(
    metrics=[
        corpus_wer(),
        corpus_cer(),
        grouped(corpus_wer(), "lang", all=False, name_template="wer_{group_name}"),
        grouped(corpus_cer(), "lang", all=False, name_template="cer_{group_name}"),
    ]
)
def asr_scorer(normalizer: str = "basic") -> Scorer:
    """Score a transcription by word/character edit counts.

    Both sides are normalized identically before alignment — the prior art in
    ``run_inference.py`` lowercased only the reference, an asymmetry that is a
    bug rather than a policy, since it can only ever make the hypothesis look
    worse than it is.

    Args:
        normalizer: Which text normalizer to apply to both reference and
            hypothesis before computing edit distance.

    Returns:
        A scorer whose ``Score.value`` carries
        ``{wer_errors, ref_words, cer_errors, ref_chars}`` — counts to be
        summed by :func:`corpus_wer` / :func:`corpus_cer`, not rates.
    """
    normalize = get_normalizer(normalizer)

    async def score(state: TaskState, target: Target) -> Score:
        import jiwer

        reference = normalize(target.text)
        hypothesis = normalize(state.output.completion)

        # An empty reference makes jiwer's alignment degenerate (division by
        # zero reference length); skip rather than let one bad sample corrupt
        # a corpus WER that legitimately depends on every reference having a
        # positive number of words.
        if not reference.strip():
            return Score(
                value={"wer_errors": 0, "ref_words": 0, "cer_errors": 0, "ref_chars": 0},
                answer=hypothesis,
                explanation="Empty reference after normalization; excluded from corpus WER/CER.",
            )

        words = jiwer.process_words([reference], [hypothesis])
        chars = jiwer.process_characters([reference], [hypothesis])

        return Score(
            value={
                "wer_errors": words.substitutions + words.deletions + words.insertions,
                "ref_words": words.substitutions + words.deletions + words.hits,
                "cer_errors": chars.substitutions + chars.deletions + chars.insertions,
                "ref_chars": chars.substitutions + chars.deletions + chars.hits,
            },
            answer=hypothesis,
        )

    return score


# =============================================================================
# ST: BLEU / chrF
# =============================================================================

#: sacrebleu tokenizer per target language. Anything not listed uses "13a"
#: (sacrebleu's own default, tuned for space-delimited scripts). "ja-mecab"
#: needs the `sacrebleu[ja]` extra; corpus_bleu surfaces that requirement via
#: sacrebleu's own error rather than hiding it behind a fallback tokenizer.
BLEU_TOKENIZER_BY_LANG: dict[str, str] = {
    "zh": "zh",
    "zh-cn": "zh",
    "zh-hk": "zh",
    "zh-tw": "zh",
    "ja": "ja-mecab",
}


def _target_lang(scores: list[SampleScore]) -> str:
    """Read the (single) target language off the samples' metadata.

    Raises:
        ValueError: If the samples disagree on target language, since one
            sacrebleu tokenizer cannot be correct for both and silently
            picking one would produce a number that looks precise but is not
            comparable to a same-language run.
    """
    langs = {
        (s.sample_metadata or {}).get("tgt_lang") or (s.sample_metadata or {}).get("lang", "")
        for s in scores
    }
    langs.discard("")
    if len(langs) > 1:
        raise ValueError(
            f"Mixed target languages in one corpus_bleu/corpus_chrf call: {sorted(langs)}. "
            "Score languages separately (e.g. via the `lang` filter on frozen_dataset) so one "
            "tokenizer signature applies to the whole number."
        )
    return next(iter(langs), "")


@metric
def corpus_bleu() -> Metric:
    """Corpus BLEU via sacrebleu, tokenizer routed by target language.

    The full sacrebleu signature (tokenizer, smoothing, case, version) is
    logged at INFO — scores computed under different sacrebleu configurations
    are not comparable, so the signature is exactly the information a reader
    needs before trusting a comparison between two runs.
    """

    def calculate(scores: list[SampleScore]) -> Value:
        import logging

        import sacrebleu

        logger = logging.getLogger(__name__)
        hypotheses = [s.score.metadata["hypothesis"] for s in scores]
        references = [s.score.metadata["reference"] for s in scores]

        tokenizer = BLEU_TOKENIZER_BY_LANG.get(_target_lang(scores).lower(), "13a")
        bleu = sacrebleu.BLEU(tokenize=tokenizer)
        result = bleu.corpus_score(hypotheses, [references])
        logger.info("corpus_bleu signature: %s", bleu.get_signature())
        return result.score

    return calculate


@metric
def corpus_chrf() -> Metric:
    """Corpus chrF++ via sacrebleu. Language-agnostic, unlike BLEU."""

    def calculate(scores: list[SampleScore]) -> Value:
        import logging

        import sacrebleu

        logger = logging.getLogger(__name__)
        hypotheses = [s.score.metadata["hypothesis"] for s in scores]
        references = [s.score.metadata["reference"] for s in scores]

        chrf = sacrebleu.CHRF(word_order=2)  # chrF++
        result = chrf.corpus_score(hypotheses, [references])
        logger.info("corpus_chrf signature: %s", chrf.get_signature())
        return result.score

    return calculate


@scorer(
    metrics=[
        corpus_bleu(),
        corpus_chrf(),
        grouped(corpus_bleu(), "lang", all=False, name_template="bleu_{group_name}"),
        grouped(corpus_chrf(), "lang", all=False, name_template="chrf_{group_name}"),
    ]
)
def st_scorer() -> Scorer:
    """Score a translation by recording the (reference, hypothesis) pair.

    Corpus BLEU/chrF are not the mean of per-sample scores (see the module
    docstring), so nothing meaningful is computed per sample here. The pair is
    carried in ``Score.metadata`` — deliberately not ``Score.value``, which
    goes through inspect's epoch-reduction/``value_to_float`` machinery even
    for a single epoch and would silently replace this pair's text with
    ``0.0`` — purely so :func:`corpus_bleu` / :func:`corpus_chrf` have
    something to recompute over.

    ``Score.value`` itself carries token lengths: not a metric, just enough to
    make an empty hypothesis or a wildly short/long one visible in the log's
    per-sample view without decoding the metadata by hand.

    Returns:
        A scorer whose ``Score.metadata`` carries ``{reference, hypothesis}``.
    """

    async def score(state: TaskState, target: Target) -> Score:
        hypothesis = state.output.completion
        reference = target.text
        return Score(
            value={"hyp_tokens": len(hypothesis.split()), "ref_tokens": len(reference.split())},
            answer=hypothesis,
            metadata={"reference": reference, "hypothesis": hypothesis},
        )

    return score
