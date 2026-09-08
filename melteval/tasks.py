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

from melteval.dataset import frozen_dataset
from melteval.registry import default_scorer
from melteval.solver import speech_prompt


@task
def speech(
    frozen_set: str,
    *,
    task_filter: str | None = None,
    lang: str | None = None,
    dataset_id: str | None = None,
    limit: int | None = None,
    format_config: str | None = None,
    tokenizer: str | None = None,
    scorer: Scorer | None = None,
    solver: Solver | None = None,
) -> Task:
    """Generate from audio and score the result.

    Args:
        frozen_set: Path to a frozen-set directory.
        task_filter: Restrict to one task (``asr``, ``st``, …). Also selects
            the default scorer — see :func:`melteval.registry.default_scorer`.
        lang: Restrict to one output language.
        dataset_id: Restrict to one corpus.
        limit: Take at most this many samples, in manifest order.
        format_config: Training config to take the prompt format from. Defaults
            to the checkpoint being evaluated.
        tokenizer: Where to load the chat template from. Defaults to the
            checkpoint being evaluated.
        scorer: Scorer to apply. Defaults to the task's registered scorer, or
            ``exact()`` (a plumbing check, not a real metric) when
            *task_filter* is unset.
        solver: Override the solver. Defaults to :func:`speech_prompt`.

    Returns:
        The configured task.
    """
    name = f"speech-{task_filter or 'all'}"
    # dataset_id and lang fold into the task name (not just left in the task
    # args) because together they're what tells two runs of the same
    # task_filter apart in inspect view's log list -- e.g. six covost2
    # directions all filtered to task_filter=st would otherwise all show up
    # as "speech-st", and voxpopuli/it vs voxpopuli/fr would both show up as
    # "speech-asr-voxpopuli" with nothing but the random log ID to tell them
    # apart.
    if dataset_id:
        name = f"{name}-{dataset_id}"
    if lang:
        name = f"{name}-{lang}"
    return Task(
        dataset=frozen_dataset(
            frozen_set, task=task_filter, lang=lang, dataset_id=dataset_id, limit=limit
        ),
        solver=solver or speech_prompt(format_config=format_config, tokenizer=tokenizer),
        scorer=scorer or default_scorer(task_filter),
        name=name,
    )


@task
def asr(frozen_set: str, **kwargs) -> Task:
    """:func:`speech` restricted to ``task_filter="asr"`` with WER/CER scoring."""
    kwargs.setdefault("task_filter", "asr")
    return speech(frozen_set, **kwargs)


@task
def st(frozen_set: str, **kwargs) -> Task:
    """:func:`speech` restricted to ``task_filter="st"`` with BLEU/chrF scoring.

    A frozen set covering more than one target language must also be given
    ``lang=`` (one run per target language). BLEU's tokenizer is chosen per
    target language, and mixing languages in one ``corpus_bleu`` call has no
    single correct tokenizer for the mix — the scorer raises rather than
    silently pick one and produce a number that looks precise but is not.
    """
    kwargs.setdefault("task_filter", "st")
    return speech(frozen_set, **kwargs)
