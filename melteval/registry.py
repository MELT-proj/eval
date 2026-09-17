"""Task → default scorer mapping.

Speech-specific metrics (WER/CER, BLEU/chrF) exist for ``asr`` and ``st``, and
for ``chunked_asr``/``chunked_st`` -- the same metrics, but computed over
several samples' completions reassembled into one hypothesis first, for
corpora (MCIF's ``short`` track) whose reference spans more than one sample.
Audio benchmarks that ship their own prompt per sample add two more shapes:
``audio_mcq``, where the sample carries its options and accuracy is well
defined, and ``audio_chat``, where the answer is free text and only a judge
model can grade it. Every other task — summarisation, instruction following —
scores with whatever ``inspect_ai.scorer`` ships (``exact``, ``f1``,
``choice``, ``model_graded_qa``, …), chosen explicitly per benchmark. This
module is not meant to grow a case for each of those; it holds the defaults
that have a genuinely single right answer.
"""

from __future__ import annotations

from collections.abc import Callable

from inspect_ai.scorer import Scorer

from melteval.scorers import (
    asr_scorer,
    chat_scorer,
    chunked_asr_scorer,
    chunked_st_scorer,
    mcq_scorer,
    st_scorer,
)


#: Factories taking no required argument. ``audio_chat`` is deliberately not
#: here: its scorer cannot be built without being told which model judges, so
#: it is constructed in :func:`default_scorer` where that argument is in scope.
TASK_SCORERS: dict[str, Callable[[], Scorer]] = {
    "asr": asr_scorer,
    "st": st_scorer,
    "chunked_asr": chunked_asr_scorer,
    "chunked_st": chunked_st_scorer,
    "audio_mcq": mcq_scorer,
}

#: Tasks whose default scorer needs a judge model.
GRADED_TASKS = ("audio_chat",)


def default_scorer(task_filter: str | None, grader_model: str | None = None) -> Scorer:
    """Return the default scorer for *task_filter*.

    With no task filter (a mixed or unfiltered frozen set) this falls back to
    ``exact()`` — a plumbing check that the pipeline runs end to end, not a
    real metric, since no single scorer is right for a set that mixes tasks.

    Args:
        task_filter: The task a frozen set was filtered to, or ``None``.
        grader_model: Judge model, for the tasks in :data:`GRADED_TASKS`.

    Raises:
        ValueError: If *task_filter* is set but has no registered default.
            Silently falling back to ``exact()`` here would hide a typo (e.g.
            ``task_filter="ars"``) behind a number that looks like a real
            score but is not. Also if a graded task is asked for without a
            judge — see :func:`melteval.scorers.chat_scorer`.
    """
    if task_filter is None:
        from inspect_ai.scorer import exact

        return exact()

    if task_filter in GRADED_TASKS:
        return chat_scorer(grader_model)

    factory = TASK_SCORERS.get(task_filter)
    if factory is None:
        raise ValueError(
            f"No default scorer for task {task_filter!r} (have defaults for: "
            f"{sorted([*TASK_SCORERS, *GRADED_TASKS])}). Pass scorer=... explicitly -- e.g. one "
            "of inspect_ai.scorer's exact()/f1()/choice()/model_graded_qa()."
        )
    return factory()
