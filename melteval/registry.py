"""Task → default scorer mapping.

Speech-specific metrics (WER/CER, BLEU/chrF) exist for ``asr`` and ``st``.
Every other task — spoken QA, summarisation, instruction following — scores
with whatever ``inspect_ai.scorer`` ships (``exact``, ``f1``, ``choice``,
``model_graded_qa``, …), chosen explicitly per benchmark. This module is not
meant to grow a case for each of those; it only holds the two defaults that
have a genuinely single right answer.
"""

from __future__ import annotations

from collections.abc import Callable

from inspect_ai.scorer import Scorer

from melteval.scorers import asr_scorer, st_scorer


TASK_SCORERS: dict[str, Callable[[], Scorer]] = {
    "asr": asr_scorer,
    "st": st_scorer,
}


def default_scorer(task_filter: str | None) -> Scorer:
    """Return the default scorer for *task_filter*.

    With no task filter (a mixed or unfiltered frozen set) this falls back to
    ``exact()`` — a plumbing check that the pipeline runs end to end, not a
    real metric, since no single scorer is right for a set that mixes tasks.

    Args:
        task_filter: The task a frozen set was filtered to, or ``None``.

    Raises:
        ValueError: If *task_filter* is set but has no registered default.
            Silently falling back to ``exact()`` here would hide a typo (e.g.
            ``task_filter="ars"``) behind a number that looks like a real
            score but is not.
    """
    if task_filter is None:
        from inspect_ai.scorer import exact

        return exact()

    factory = TASK_SCORERS.get(task_filter)
    if factory is None:
        raise ValueError(
            f"No default scorer for task {task_filter!r} (have defaults for: "
            f"{sorted(TASK_SCORERS)}). Pass scorer=... explicitly -- e.g. one of "
            "inspect_ai.scorer's exact()/f1()/choice()/model_graded_qa()."
        )
    return factory()
