"""Evaluation tasks.

Tasks are thin: a frozen set, the speech solver, and a scorer. The scorer is a
parameter rather than a hard-coded choice per task, because the same generation
path serves ASR, translation and — with inspect's own ``exact``/``f1``/``choice``
scorers — spoken question answering, summarisation and instruction following.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.scorer import Scorer, exact
from inspect_ai.solver import Solver

from melteval.dataset import frozen_dataset
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
        task_filter: Restrict to one task (``asr``, ``st``, …).
        lang: Restrict to one output language.
        dataset_id: Restrict to one corpus.
        limit: Take at most this many samples, in manifest order.
        format_config: Training config to take the prompt format from. Defaults
            to the checkpoint being evaluated.
        tokenizer: Where to load the chat template from. Defaults to the
            checkpoint being evaluated.
        scorer: Scorer to apply. Defaults to ``exact``, which is only sensible
            as a smoke test — pass an ASR or ST scorer for real numbers.
        solver: Override the solver. Defaults to :func:`speech_prompt`.

    Returns:
        The configured task.
    """
    return Task(
        dataset=frozen_dataset(
            frozen_set, task=task_filter, lang=lang, dataset_id=dataset_id, limit=limit
        ),
        solver=solver or speech_prompt(format_config=format_config, tokenizer=tokenizer),
        scorer=scorer or exact(),
        name=f"speech-{task_filter or 'all'}",
    )
