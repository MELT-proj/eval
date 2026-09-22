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
from melteval.scorers import asr_scorer
from melteval.solver import smurf_prompt, speech_prompt


#: Model family whose prompt format a run uses, and the solver that builds it.
#: Explicit rather than inferred from ``--model``: a task is constructed before
#: the model is loaded, and this repo's rule is that the format is chosen in a
#: place somebody can read, not derived. Mismatches are caught at generation
#: time by :func:`melteval.solver._require_provider`.
SOLVERS = {
    "melt": (speech_prompt, {"format_config", "tokenizer"}),
    "smurf": (smurf_prompt, {"instruction", "prompt_config"}),
}


@task
def speech(
    frozen_set: str | None = None,
    *,
    spec: str | None = None,
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
        ValueError: If neither or both of *frozen_set* and *spec* are given;
            if *normalizer* is given together with an explicit *scorer*, or
            with a *task_filter* other than ``"asr"``.
    """
    if bool(frozen_set) == bool(spec):
        raise ValueError(
            "Pass exactly one of `frozen_set` (a directory melteval freeze wrote) or "
            "`spec` (a source spec YAML, read live). "
            f"Got frozen_set={frozen_set!r}, spec={spec!r}."
        )
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

    filters = {"task": task_filter, "lang": lang, "dataset_id": dataset_id, "limit": limit}
    dataset = (
        spec_dataset(spec, **filters) if spec else frozen_dataset(frozen_set, **filters)
    )

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
        dataset=dataset,
        solver=solver
        or _prompt_solver(
            prompt_style,
            format_config=format_config,
            tokenizer=tokenizer,
            instruction=instruction,
            prompt_config=prompt_config,
        ),
        scorer=resolved_scorer,
        name=name,
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
    if prompt_style not in SOLVERS:
        raise ValueError(f"Unknown prompt_style {prompt_style!r}; expected one of {tuple(SOLVERS)}.")

    solver_fn, valid_kwargs = SOLVERS[prompt_style]

    given = {
        "format_config": format_config,
        "tokenizer": tokenizer,
        "instruction": instruction,
        "prompt_config": prompt_config,
    }

    misplaced = [
        name for name, value in given.items() if value is not None and name not in valid_kwargs
    ]

    if misplaced:
        owner = {name: style for style, (_, kwargs) in SOLVERS.items() for name in kwargs}
        raise ValueError(
            f"{', '.join(sorted(misplaced))} {'belongs' if len(misplaced) == 1 else 'belong'} to "
            f"prompt_style={owner[misplaced[0]]!r}, but this task is running "
            f"prompt_style={prompt_style!r}."
        )

    return solver_fn(**{name: given[name] for name in valid_kwargs if given[name] is not None})


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

    Needs a model bound to the ``grader`` role (``--model-role
    grader=<provider/model>``, e.g. ``--model-role grader=openai/gpt-4o``):
    the bound model judges each free-text answer against the reference and
    grades it as agreeing, partially agreeing or disagreeing (partial
    credit), via :func:`melteval.scorers.chat_scorer`. Free-text answers
    about audio have no lexical metric that measures the task, so there is
    nothing to fall back on — run ``inspect eval --no-score`` to generate now
    and grade the log later with ``inspect score`` if no judge is reachable
    from the cluster.
    """
    kwargs.setdefault("task_filter", "audio_chat")
    return speech(frozen_set, **kwargs)
