"""Evaluation tasks.

Tasks are thin: a frozen set, the speech solver, and a scorer. The scorer is a
parameter rather than a hard-coded choice, because the same generation path
serves ASR, translation and — with inspect's own
``exact``/``f1``/``choice``/``model_graded_qa`` scorers — spoken question
answering, summarisation and instruction following. :func:`asr` and :func:`st`
are convenience wrappers that also pick the matching speech metric by default;
see :mod:`melteval.registry`.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver

from melteval.dataset import frozen_dataset, spec_dataset
from melteval.registry import default_scorer
from melteval.solver import speech_prompt


@task
def speech(
    frozen_set: str | None = None,
    *,
    spec: str | None = None,
    task_filter: str | None = None,
    lang: str | None = None,
    dataset_id: str | None = None,
    limit: int | None = None,
    format_config: str | None = None,
    tokenizer: str | None = None,
    grader_model: str | None = None,
    scorer: Scorer | None = None,
    solver: Solver | None = None,
) -> Task:
    """Generate from audio and score the result.

    Samples come either from a frozen set or, with *spec*, from a source spec
    read at eval time. The second is for corpora that are already immutable —
    a HuggingFace benchmark pinned to a revision — where a freeze pass would
    only restate what the revision hash already guarantees. Local Shar
    mixtures should still be frozen: nothing else records which samples a
    number was computed over.

    Args:
        frozen_set: Path to a frozen-set directory.
        spec: Path to a source spec YAML, read live instead of frozen.
            Mutually exclusive with *frozen_set*.
        task_filter: Restrict to one task (``asr``, ``st``, ``chunked_asr``,
            ``chunked_st``, ``audio_mcq``, ``audio_chat``, …). Also selects
            the default scorer — see
            :func:`melteval.registry.default_scorer`.
        lang: Restrict to one output language.
        dataset_id: Restrict to one corpus.
        limit: Take at most this many samples, in manifest order.
        format_config: Training config to take the prompt format from. Defaults
            to the checkpoint being evaluated.
        tokenizer: Where to load the chat template from. Defaults to the
            checkpoint being evaluated.
        grader_model: Judge model for tasks scored by one (``audio_chat``).
        scorer: Scorer to apply. Defaults to the task's registered scorer, or
            ``exact()`` (a plumbing check, not a real metric) when
            *task_filter* is unset.
        solver: Override the solver. Defaults to :func:`speech_prompt`.

    Returns:
        The configured task.

    Raises:
        ValueError: If neither or both of *frozen_set* and *spec* are given.
            There is no useful default: guessing which one was meant would
            evaluate a different sample set than the caller asked for.
    """
    if bool(frozen_set) == bool(spec):
        raise ValueError(
            "Pass exactly one of `frozen_set` (a directory melteval freeze wrote) or "
            "`spec` (a source spec YAML, read live). "
            f"Got frozen_set={frozen_set!r}, spec={spec!r}."
        )

    filters = {"task": task_filter, "lang": lang, "dataset_id": dataset_id, "limit": limit}
    dataset = (
        spec_dataset(spec, **filters) if spec else frozen_dataset(frozen_set, **filters)
    )

    return Task(
        dataset=dataset,
        solver=solver or speech_prompt(format_config=format_config, tokenizer=tokenizer),
        scorer=scorer or default_scorer(task_filter, grader_model),
        name=f"speech-{task_filter or 'all'}",
    )


@task
def asr(frozen_set: str | None = None, **kwargs) -> Task:
    """:func:`speech` restricted to ``task_filter="asr"`` with WER/CER scoring."""
    kwargs.setdefault("task_filter", "asr")
    return speech(frozen_set, **kwargs)


@task
def st(frozen_set: str | None = None, **kwargs) -> Task:
    """:func:`speech` restricted to ``task_filter="st"`` with BLEU/chrF scoring.

    A frozen set covering more than one target language must also be given
    ``lang=`` (one run per target language). BLEU's tokenizer is chosen per
    target language, and mixing languages in one ``corpus_bleu`` call has no
    single correct tokenizer for the mix — the scorer raises rather than
    silently pick one and produce a number that looks precise but is not.
    """
    kwargs.setdefault("task_filter", "st")
    return speech(frozen_set, **kwargs)


@task
def chunked_asr(frozen_set: str | None = None, **kwargs) -> Task:
    """:func:`speech` restricted to ``task_filter="chunked_asr"``.

    For corpora (MCIF's ``short`` track) where several samples are chunks of
    one long recording and the reference is defined over the whole thing —
    each chunk is still its own generation, but WER/CER is computed once per
    group of chunks, not per sample; see
    :func:`melteval.scorers.chunked_asr_scorer`.
    """
    kwargs.setdefault("task_filter", "chunked_asr")
    return speech(frozen_set, **kwargs)


@task
def chunked_st(frozen_set: str | None = None, **kwargs) -> Task:
    """:func:`speech` restricted to ``task_filter="chunked_st"``.

    Same grouping as :func:`chunked_asr`, BLEU/chrF instead of WER/CER; see
    :func:`melteval.scorers.chunked_st_scorer`.
    """
    kwargs.setdefault("task_filter", "chunked_st")
    return speech(frozen_set, **kwargs)


@task
def audio_mcq(frozen_set: str | None = None, **kwargs) -> Task:
    """:func:`speech` restricted to multiple-choice samples, scored by accuracy.

    For benchmarks whose samples carry their own question and options —
    AIR-Bench Foundation, for one. The options travel on the sample, so the
    reference is the option's text and the letter it was shown under is
    incidental; see :func:`melteval.scorers.mcq_scorer`.
    """
    kwargs.setdefault("task_filter", "audio_mcq")
    return speech(frozen_set, **kwargs)


@task
def audio_chat(frozen_set: str | None = None, **kwargs) -> Task:
    """:func:`speech` restricted to open-ended audio QA, graded by a judge model.

    Needs ``grader_model=``. Free-text answers about audio have no lexical
    metric that measures the task, so there is nothing to fall back on — run
    ``inspect eval --no-score`` to generate now and grade the log later if no
    judge is reachable from the cluster.
    """
    kwargs.setdefault("task_filter", "audio_chat")
    return speech(frozen_set, **kwargs)
