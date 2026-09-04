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
from melteval.scorers import asr_scorer
from melteval.solver import smurf_prompt, speech_prompt


#: Model family whose prompt format a run uses, and the solver that builds it.
#: Explicit rather than inferred from ``--model``: a task is constructed before
#: the model is loaded, and this repo's rule is that the format is chosen in a
#: place somebody can read, not derived. Mismatches are caught at generation
#: time by :func:`melteval.solver._require_provider`.
PROMPT_STYLES = ("melt", "smurf")


@task
def speech(
    frozen_set: str,
    *,
    task_filter: str | None = None,
    lang: str | None = None,
    dataset_id: str | None = None,
    limit: int | None = None,
    prompt_style: str = "melt",
    format_config: str | None = None,
    tokenizer: str | None = None,
    instruction: str | None = None,
    prompt_config: str | None = None,
    normalizer: str | None = None,
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
        prompt_style: Which model family's prompt format to build — ``melt``
            (default) or ``smurf``. Must match the ``--model`` provider.
        format_config: ``melt`` only. Training config to take the prompt format
            from. Defaults to the checkpoint being evaluated.
        tokenizer: ``melt`` only. Where to load the chat template from.
            Defaults to the checkpoint being evaluated.
        instruction: ``smurf`` only. Fixed instruction text for samples that do
            not carry their own.
        prompt_config: ``smurf`` only. SMURF data/inference config to read the
            instruction (``tags.context``) from.
        normalizer: ``task_filter="asr"`` only. Text normalizer for the WER/CER
            scorer -- ``"basic"`` (default) and ``"english"`` both import
            ``melt.evaluation``, which is not installed in a SMURF-only
            environment (see "Environment" in docs/smurf-provider.md); pass
            ``"none"`` there to score raw strings instead. See
            :func:`melteval.scorers.get_normalizer`.
        scorer: Scorer to apply. Defaults to the task's registered scorer, or
            ``exact()`` (a plumbing check, not a real metric) when
            *task_filter* is unset. Mutually exclusive with *normalizer*.
        solver: Override the solver. Defaults to the one *prompt_style* selects.

    Returns:
        The configured task.

    Raises:
        ValueError: If *normalizer* is given together with an explicit
            *scorer*, or with a *task_filter* other than ``"asr"``.
    """
    if normalizer is not None:
        if scorer is not None:
            raise ValueError(
                "normalizer is ignored when scorer is given explicitly; pass it into your own "
                "asr_scorer(normalizer=...) instead."
            )
        if task_filter != "asr":
            raise ValueError(
                f"normalizer configures the WER/CER scorer and only applies to task_filter='asr', "
                f"got task_filter={task_filter!r}."
            )
    if scorer is not None:
        resolved_scorer = scorer
    elif normalizer is not None:
        resolved_scorer = asr_scorer(normalizer=normalizer)
    else:
        resolved_scorer = default_scorer(task_filter)

    return Task(
        dataset=frozen_dataset(
            frozen_set, task=task_filter, lang=lang, dataset_id=dataset_id, limit=limit
        ),
        solver=solver
        or _prompt_solver(
            prompt_style,
            format_config=format_config,
            tokenizer=tokenizer,
            instruction=instruction,
            prompt_config=prompt_config,
        ),
        scorer=resolved_scorer,
        name=f"speech-{task_filter or 'all'}",
    )


def _prompt_solver(
    prompt_style: str,
    format_config: str | None,
    tokenizer: str | None,
    instruction: str | None,
    prompt_config: str | None,
) -> Solver:
    """Build the prompt solver for *prompt_style*.

    Arguments belonging to the other style are rejected rather than ignored: a
    ``-T format_config=...`` silently dropped from a SMURF run looks exactly
    like one that was honoured, and the whole point of naming the config is to
    know which one shaped the result.

    Raises:
        ValueError: If *prompt_style* is unknown, or an argument belongs to a
            different style.
    """
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"Unknown prompt_style {prompt_style!r}; expected one of {PROMPT_STYLES}.")

    given = {
        "format_config": format_config,
        "tokenizer": tokenizer,
        "instruction": instruction,
        "prompt_config": prompt_config,
    }
    belongs_to = {
        "format_config": "melt",
        "tokenizer": "melt",
        "instruction": "smurf",
        "prompt_config": "smurf",
    }
    misplaced = [
        name for name, value in given.items() if value is not None and belongs_to[name] != prompt_style
    ]
    if misplaced:
        raise ValueError(
            f"{', '.join(sorted(misplaced))} {'belongs' if len(misplaced) == 1 else 'belong'} to "
            f"prompt_style={belongs_to[misplaced[0]]!r}, but this task is running "
            f"prompt_style={prompt_style!r}."
        )

    if prompt_style == "smurf":
        return smurf_prompt(instruction=instruction, prompt_config=prompt_config)
    return speech_prompt(format_config=format_config, tokenizer=tokenizer)


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
