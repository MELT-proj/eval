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

import re

from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    Value,
    accuracy,
    grouped,
    metric,
    scorer,
    stderr,
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


# =============================================================================
# Chunked ASR / ST: many short generations scored against one long reference
# =============================================================================


def _grouped_pairs(scores: list[SampleScore]) -> list[tuple[str, str]]:
    """Reassemble chunk-level completions into one (hypothesis, reference) pair per group.

    Some corpora already give one sample one reference. Others -- MCIF's
    ``short`` track, for one -- cut a long recording into several samples that
    the model transcribes independently, and define the reference over the
    whole thing; scoring it means joining those completions back together, in
    order, before comparing to that one reference. A sample without grouping
    metadata is its own group of one, so this degrades to plain per-sample
    pairing when nothing upstream needs grouping at all.

    The group key is ``(dataset_id, group_id)``, not ``group_id`` alone,
    because two corpora scored in the same run could otherwise collide on the
    same group label by coincidence.

    Raises:
        ValueError: If a group's members disagree on the reference text --
            that would mean two chunks were placed in the same group by
            mistake, since every member of a real group is scored against the
            same whole-reference string.
    """
    groups: dict[tuple[str, str], list[tuple[float, str, str]]] = {}
    for s in scores:
        meta = s.sample_metadata or {}
        group_id = meta.get("group_id", s.sample_id)
        key = (str(meta.get("dataset_id", "")), str(group_id))
        order = meta.get("group_order", 0)
        groups.setdefault(key, []).append(
            (order, s.score.metadata["hypothesis"], s.score.metadata["reference"])
        )

    pairs = []
    for key, members in groups.items():
        members.sort(key=lambda m: m[0])
        references = {ref for _, _, ref in members}
        if len(references) > 1:
            raise ValueError(
                f"Group {key!r} has {len(references)} distinct references: "
                f"{sorted(references)}. Every chunk in a group must be scored against the "
                "same reference."
            )
        hypothesis = " ".join(hyp for _, hyp, _ in members)
        pairs.append((hypothesis, next(iter(references))))
    return pairs


@metric
def chunked_wer() -> Metric:
    """Corpus WER over grouped (not raw) hypothesis/reference pairs."""

    def calculate(scores: list[SampleScore]) -> Value:
        import jiwer

        normalize = get_normalizer("english")
        total_errors = total_words = 0
        for hypothesis, reference in _grouped_pairs(scores):
            ref_n, hyp_n = normalize(reference), normalize(hypothesis)
            if not ref_n.strip():
                continue
            words = jiwer.process_words([ref_n], [hyp_n])
            total_errors += words.substitutions + words.deletions + words.insertions
            total_words += words.substitutions + words.deletions + words.hits
        return total_errors / total_words if total_words > 0 else 0.0

    return calculate


@metric
def chunked_cer() -> Metric:
    """Corpus CER over grouped (not raw) hypothesis/reference pairs."""

    def calculate(scores: list[SampleScore]) -> Value:
        import jiwer

        normalize = get_normalizer("english")
        total_errors = total_chars = 0
        for hypothesis, reference in _grouped_pairs(scores):
            ref_n, hyp_n = normalize(reference), normalize(hypothesis)
            if not ref_n.strip():
                continue
            chars = jiwer.process_characters([ref_n], [hyp_n])
            total_errors += chars.substitutions + chars.deletions + chars.insertions
            total_chars += chars.substitutions + chars.deletions + chars.hits
        return total_errors / total_chars if total_chars > 0 else 0.0

    return calculate


@scorer(
    metrics=[
        chunked_wer(),
        chunked_cer(),
        grouped(chunked_wer(), "dataset_id", all=False, name_template="wer_{group_name}"),
        grouped(chunked_cer(), "dataset_id", all=False, name_template="cer_{group_name}"),
    ]
)
def chunked_asr_scorer() -> Scorer:
    """ASR scorer for corpora where a reference can span several samples.

    Each sample is scored for its own record only in the sense of carrying its
    completion forward -- the actual WER/CER is computed once per group by
    :func:`chunked_wer` / :func:`chunked_cer`, which reassemble each group's
    completions in order before comparing to its reference. Normalization is
    fixed to ``"english"``: this scorer exists for MCIF, whose grouped ASR
    reference is only ever produced for an English target.

    Returns:
        A scorer whose ``Score.metadata`` carries ``{reference, hypothesis}``,
        matching :func:`st_scorer`'s shape for the same reason: corpus WER is
        not the mean of per-sample rates, so nothing meaningful is computed
        per sample here.
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


@metric
def chunked_bleu() -> Metric:
    """Corpus BLEU over grouped (not raw) hypothesis/reference pairs."""

    def calculate(scores: list[SampleScore]) -> Value:
        import sacrebleu

        pairs = _grouped_pairs(scores)
        if not pairs:
            return 0.0
        tokenizer = BLEU_TOKENIZER_BY_LANG.get(_target_lang(scores).lower(), "13a")
        bleu = sacrebleu.BLEU(tokenize=tokenizer)
        result = bleu.corpus_score([h for h, _ in pairs], [[r for _, r in pairs]])
        return result.score

    return calculate


@metric
def chunked_chrf() -> Metric:
    """Corpus chrF++ over grouped (not raw) hypothesis/reference pairs."""

    def calculate(scores: list[SampleScore]) -> Value:
        import sacrebleu

        pairs = _grouped_pairs(scores)
        if not pairs:
            return 0.0
        chrf = sacrebleu.CHRF(word_order=2)
        result = chrf.corpus_score([h for h, _ in pairs], [[r for _, r in pairs]])
        return result.score

    return calculate


@scorer(
    metrics=[
        chunked_bleu(),
        chunked_chrf(),
        grouped(chunked_bleu(), "dataset_id", all=False, name_template="bleu_{group_name}"),
        grouped(chunked_chrf(), "dataset_id", all=False, name_template="chrf_{group_name}"),
    ]
)
def chunked_st_scorer() -> Scorer:
    """ST scorer for corpora where a reference can span several samples.

    See :func:`chunked_asr_scorer` -- same shape, BLEU/chrF instead of WER/CER.
    This reports the same corpus BLEU/chrF :func:`st_scorer` does, over
    reassembled groups rather than raw samples; it is not the paper metric for
    a benchmark like MCIF, which scores translation with COMET after a
    sentence-resegmentation step this harness does not perform (see
    ``melteval/rescore.py`` -- neural MT metrics live in their own venv, and
    the same applies here).

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


# =============================================================================
# Multiple choice: accuracy, and how often an answer could be read at all
# =============================================================================

#: Leading "B", "B.", "(b)", "b:" and friends. Anchored, because a letter
#: found anywhere in a sentence is usually a word ("a bird"), not an answer.
_LEADING_LABEL = re.compile(r"^\W*([a-z])\s*(?:[.):\-]|$)", re.IGNORECASE)

#: A label the completion *ends* on -- "The answer is B." Case-sensitive on
#: purpose, and that is the whole guard: an isolated lowercase letter at the
#: end of a sentence is usually the article "a", while a model announcing its
#: choice writes it capitalised. Without this, a model that answers in the
#: most natural way there is scores zero and looks like it cannot follow the
#: prompt.
_TRAILING_LABEL = re.compile(r"(?:^|\W)\(?([A-Z])\)?\.?\s*$")

_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def _normalize_choice_text(text: str) -> str:
    """Casefold, drop punctuation and collapse whitespace, for comparison only."""
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", text.casefold())).strip()


def _mentions(haystack: str, needle: str) -> bool:
    """Whether *needle* appears in *haystack* as a whole word.

    A plain substring test is wrong here in a way that is easy to miss and
    hard to see in a score: "Male" is a substring of "female", so an answer of
    "female" matched both options of a gender question, resolved to neither,
    and was counted as an unreadable completion.
    """
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


def resolve_choice(completion: str, choices: list[str]) -> str | None:
    """Read which option *completion* picked, or ``None`` if it picked none.

    Tried in order, most explicit first:

    1. a leading option label (``"B"``, ``"B."``, ``"(b)"``);
    2. a capitalised label the completion ends on ("The answer is B.");
    3. the whole completion equal to one option's text;
    4. exactly one option's text appearing somewhere in the completion.

    Step 4 requires the match to be *unique*: a model that echoes the full
    option list has not answered, and crediting it with the first option it
    happens to mention would turn a non-answer into a coin flip weighted by
    option order.

    Args:
        completion: What the model generated.
        choices: The sample's options, in the order they were presented.

    Returns:
        The chosen option's text, or ``None`` if no single option was picked.
    """
    if not choices:
        return None

    stripped = completion.strip()

    for pattern, search in ((_LEADING_LABEL, False), (_TRAILING_LABEL, True)):
        label = pattern.search(stripped) if search else pattern.match(stripped)
        if label:
            position = ord(label.group(1).upper()) - ord("A")
            if 0 <= position < len(choices):
                return choices[position]

    normalized = _normalize_choice_text(stripped)
    if not normalized:
        return None

    normalized_choices = [_normalize_choice_text(choice) for choice in choices]

    for choice, normalized_choice in zip(choices, normalized_choices):
        if normalized_choice and normalized == normalized_choice:
            return choice

    contained = [
        choice
        for choice, normalized_choice in zip(choices, normalized_choices)
        if normalized_choice and _mentions(normalized, normalized_choice)
    ]
    if len(contained) == 1:
        return contained[0]

    return None


@metric
def choice_accuracy() -> Metric:
    """Share of samples whose chosen option matches the reference."""

    def calculate(scores: list[SampleScore]) -> Value:
        if not scores:
            return 0.0
        return sum(s.score.value["correct"] for s in scores) / len(scores)

    return calculate


@metric
def unresolved_rate() -> Metric:
    """Share of completions no single option could be read out of.

    Reported alongside accuracy rather than folded into it, because the two
    failures are not the same thing and the fix is not the same either. A model
    that answers "D" and is wrong scores 0 here and so does one that recites a
    paragraph, but only the second is telling you the prompt never asked for a
    letter in a form it understood.
    """

    def calculate(scores: list[SampleScore]) -> Value:
        if not scores:
            return 0.0
        return sum(1 - s.score.value["resolved"] for s in scores) / len(scores)

    return calculate


@scorer(
    metrics=[
        choice_accuracy(),
        unresolved_rate(),
        grouped(choice_accuracy(), "dataset_id", all=False, name_template="accuracy_{group_name}"),
    ]
)
def mcq_scorer() -> Scorer:
    """Score a multiple-choice answer by which option the model picked.

    The reference is the option's **text**, not its label: benchmarks that ship
    their options as columns (AIR-Bench's ``choice_a``…``choice_d``) give the
    right answer the same way, and the letter a given option gets is an
    artefact of the order this harness rendered them in.

    Returns:
        A scorer whose ``Score.value`` carries ``{correct, resolved}`` as 0/1
        counters — the two things :func:`choice_accuracy` and
        :func:`unresolved_rate` reduce over. The completion and the option it
        resolved to go in ``Score.metadata``, which is not put through
        inspect's numeric epoch reduction (see the module docstring).
    """

    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        choices = [choice.value for choice in state.choices] if state.choices else []
        chosen = resolve_choice(completion, choices)

        correct = chosen is not None and _normalize_choice_text(chosen) == _normalize_choice_text(
            target.text
        )
        return Score(
            value={"correct": int(correct), "resolved": int(chosen is not None)},
            answer=completion,
            explanation=(
                None
                if chosen is not None
                else "No option could be read out of the completion; counted as incorrect."
            ),
            metadata={"chosen": chosen, "reference": target.text, "completion": completion},
        )

    return score


# =============================================================================
# Open-ended audio chat: graded by a judge model
# =============================================================================

#: Sent to the judge. Deliberately hands over the reference answer: the judge
#: cannot hear the audio, so without it there is nothing to grade against and
#: the "score" would be a fluency rating.
CHAT_GRADER_TEMPLATE = """You are grading a model's answer to a question about an audio clip.
You cannot hear the audio. Grade only against the reference answer, which was
written by someone who could.

Question put to the model:
{question}

Reference answer:
{criterion}

The model's answer:
{answer}

{instructions}"""

CHAT_GRADER_INSTRUCTIONS = """Decide how well the model's answer agrees with the reference.
Differences in wording, length or style do not matter; differences in what is
claimed about the audio do. An answer that is merely fluent, or that answers a
different question, is incorrect.

First explain your reasoning in one or two sentences. Then, on the last line,
write exactly one of:

GRADE: C   (agrees with the reference)
GRADE: P   (partially agrees -- right about some of it, wrong or silent about the rest)
GRADE: I   (disagrees with the reference, or does not answer)"""


NO_JUDGE_MESSAGE = (
    "Task `audio_chat` needs a judge model: pass -T grader_model=<provider/model> "
    "(e.g. -T grader_model=openai/gpt-4o). Open-ended answers about audio have no "
    "lexical metric that measures the task, and grading them with the model under test "
    "would be marking its own homework. To generate now and grade later, run "
    "`inspect eval --no-score` and score the log afterwards with `inspect score`."
)


@scorer(
    metrics=[
        accuracy(),
        stderr(),
        grouped(accuracy(), "dataset_id", all=False, name_template="accuracy_{group_name}"),
    ]
)
def chat_scorer(grader_model: str | None = None) -> Scorer:
    """Grade a free-form answer about audio against the reference, via a judge.

    There is no lexical metric worth reporting here. The references are one or
    two sentences of free text and a correct answer routinely shares almost no
    words with them, so BLEU or an exact match would rank a fluent wrong answer
    above a terse right one — a number that looks like a score and ranks models
    backwards. So a missing judge is an error rather than a fallback.

    The judge is resolved on the first sample scored, not here. Raising at
    construction time would also fire under ``inspect eval --no-score``, which
    is the one way to run this task on a cluster that cannot reach a judge:
    generate now, ``inspect score`` the log later from somewhere that can.

    Args:
        grader_model: The judge, as an inspect model string (for example
            ``openai/gpt-4o``).

    Returns:
        A scorer delegating to inspect's
        :func:`~inspect_ai.scorer.model_graded_qa` with the rubric above and
        partial credit.
    """
    delegate: list[Scorer] = []

    async def score(state: TaskState, target: Target) -> Score:
        if not delegate:
            from inspect_ai.scorer import model_graded_qa

            if not grader_model:
                raise ValueError(NO_JUDGE_MESSAGE)
            delegate.append(
                model_graded_qa(
                    template=CHAT_GRADER_TEMPLATE,
                    instructions=CHAT_GRADER_INSTRUCTIONS,
                    partial_credit=True,
                    model=grader_model,
                )
            )
        return await delegate[0](state, target)

    return score
