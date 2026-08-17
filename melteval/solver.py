"""The solvers: render the prompt, attach the audio, generate.

This is where format parity is applied and — just as importantly — made
visible. The rendered prompt goes into the message the model receives, so the
eval log shows exactly what was sent, and the format spec it came from is
recorded alongside it. A parity bug should be readable in the log rather than
inferred from a disappointing score.

There is one solver per model family, because "the format the model was trained
in" is answered by a different file for each: :func:`speech_prompt` reads a
MELT ``training_config.yaml`` and renders the whole sequence including the chat
template; :func:`smurf_prompt` renders only the instruction, because a SMURF
checkpoint applies its own chat template inside ``generate()``.
"""

from __future__ import annotations

from inspect_ai.model import ChatMessageUser, ContentData, ContentText
from inspect_ai.solver import Generate, Solver, TaskState, solver

from melteval.dataset import AUDIO_DATA_KEY
from melteval.prompt import (
    FormatSpec,
    SmurfPromptSpec,
    apply_generation_format,
    load_format_spec,
    load_smurf_prompt_spec,
    render_smurf_prompt,
    render_user_prompt,
)


@solver
def speech_prompt(
    format_config: str | None = None,
    tokenizer: str | None = None,
    apply_chat_template: bool | None = None,
    prompt_template_selection: str | None = None,
) -> Solver:
    """Build the model input for a speech sample.

    The format is read from the checkpoint's ``training_config.yaml`` — located
    from the model name when *format_config* is not given — so evaluation
    matches training by construction rather than by a flag somebody remembered
    to set.

    Args:
        format_config: Path to a training config, or the run/checkpoint
            directory holding one. Defaults to the model's own path.
        tokenizer: Where to load the chat template from. Defaults to the
            checkpoint being evaluated, which is where it belongs: the template
            ships with the model, while the format keys live in the run config,
            and the two are not always the same directory.
        apply_chat_template: Override the trained setting. Use only to measure
            the effect of the mismatch deliberately.
        prompt_template_selection: Override the trained selection strategy.

    Returns:
        A solver that replaces the message list and generates.
    """
    cache: dict[str, FormatSpec] = {}

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        _require_provider(state, "melt")
        source = format_config or state.model.name
        spec = cache.get(source)
        if spec is None:
            spec = load_format_spec(source)
            if apply_chat_template is not None:
                spec = _replace(spec, apply_chat_template=apply_chat_template)
            if prompt_template_selection is not None:
                spec = _replace(spec, prompt_template_selection=prompt_template_selection)
            cache[source] = spec

        metadata = state.metadata or {}
        prompt = render_user_prompt(
            task=str(metadata.get("task", "")),
            sample_key=str(state.sample_id),
            spec=spec,
            lang=str(metadata.get("lang") or ""),
            src_lang=str(metadata.get("src_lang") or ""),
            tgt_lang=str(metadata.get("tgt_lang") or ""),
            instruction=metadata.get("instruction"),
        )
        chat_tokenizer = (
            _tokenizer_for(tokenizer or state.model.name) if spec.apply_chat_template else None
        )
        formatted = apply_generation_format(prompt, spec, chat_tokenizer)

        content: list = [ContentText(text=formatted)]
        audio = metadata.get("audio")
        if audio is not None:
            content.append(ContentData(data={AUDIO_DATA_KEY: audio}))

        state.messages = [ChatMessageUser(content=content)]

        # Recorded so a result can be traced to the config that shaped it, and
        # so two runs can be compared on format before they are compared on score.
        state.store.set("melteval:format_spec", spec.to_dict())
        state.store.set("melteval:prompt", prompt)

        return await generate(state)

    return solve


@solver
def smurf_prompt(
    instruction: str | None = None,
    prompt_config: str | None = None,
) -> Solver:
    """Build the model input for a SMURF (NeMo/SALM) checkpoint.

    Deliberately shorter than :func:`speech_prompt`, and the difference is the
    point. A SMURF checkpoint carries its own ``PromptFormatter`` and its own
    audio placeholder, and applies both inside ``generate()``. Everything this
    solver would add on top — a chat template, an audio token — would be
    applied twice. So it renders the instruction and stops.

    The instruction is resolved per sample, in this order:

    1. the frozen set's own ``instruction``, when the sample carries one, so a
       benchmark that ships its prompts is evaluated with them;
    2. *instruction*, when given;
    3. the ``context`` tag in *prompt_config* — a SMURF data or inference YAML,
       which is where the training conversations got theirs.

    With none of the three, it raises. An empty user turn is a valid input to a
    speech LLM and produces plausible text, so guessing here would be scored
    rather than noticed.

    Args:
        instruction: A fixed instruction, with the placeholders
            :func:`melteval.prompt.render_smurf_prompt` documents.
        prompt_config: Path to a SMURF data/inference config to read the
            instruction from.

    Returns:
        A solver that replaces the message list and generates.

    Raises:
        ValueError: If both *instruction* and *prompt_config* are given. One of
            them would win, and the log would name the other.
    """
    if instruction is not None and prompt_config is not None:
        raise ValueError(
            "Pass either instruction= or prompt_config=, not both: the prompt a run used has to "
            "be traceable to one place."
        )

    spec: SmurfPromptSpec | None = None
    if prompt_config is not None:
        spec = load_smurf_prompt_spec(prompt_config)

    template = instruction if instruction is not None else (spec.instruction if spec else None)
    source = "argument" if instruction is not None else ("config" if spec else None)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        _require_provider(state, "smurf")
        metadata = state.metadata or {}

        sample_instruction = metadata.get("instruction")
        sample_template = sample_instruction or template
        if not sample_template:
            raise ValueError(
                f"No instruction for sample {state.sample_id}: it carries none of its own, and "
                "neither instruction= nor prompt_config= was given. A SMURF checkpoint is prompted "
                "by the conversation's context tag, so evaluating without one would measure a "
                "prompt no run was trained on."
            )

        prompt = render_smurf_prompt(
            sample_template,
            task=str(metadata.get("task") or ""),
            lang=str(metadata.get("lang") or ""),
            src_lang=str(metadata.get("src_lang") or ""),
            tgt_lang=str(metadata.get("tgt_lang") or ""),
        )

        content: list = [ContentText(text=prompt)]
        audio = metadata.get("audio")
        if audio is not None:
            content.append(ContentData(data={AUDIO_DATA_KEY: audio}))

        state.messages = [ChatMessageUser(content=content)]

        recorded = spec.to_dict() if spec else {"provider": "smurf", "source": "argument"}
        recorded["instruction"] = sample_template
        recorded["instruction_source"] = "sample" if sample_instruction else (source or "sample")
        # The chat template is the model's, applied in the provider. Recorded so
        # a log makes clear that its absence here is a decision, not an omission.
        recorded["apply_chat_template"] = False
        state.store.set("melteval:format_spec", recorded)
        state.store.set("melteval:prompt", prompt)

        return await generate(state)

    return solve


#: Which solver each speech provider must be run with. Only these two are
#: checked: anything else (``mockllm``, an API model used to sanity-check the
#: plumbing) has no format of its own to be wrong about.
_SOLVER_FOR_PROVIDER = {"melt": "speech_prompt", "smurf": "smurf_prompt"}


def _require_provider(state: TaskState, expected: str) -> None:
    """Fail fast when a checkpoint is paired with the other family's solver.

    The two prompt formats are not merely different, they are each other's
    double-application: MELT's solver emits a chat-templated string containing
    ``<|audio|>``, which a SMURF checkpoint would wrap in a *second* chat
    template and never expand, while SMURF's bare instruction reaches a MELT
    processor with no audio token to put the encoder frames in. Both produce
    fluent, wrong text and a plausible score — the failure mode this repo keeps
    tripping over (MELT-proj/training#58).

    Args:
        state: The sample's task state, whose ``model.api`` names the provider.
        expected: The provider this solver belongs to.

    Raises:
        ValueError: If the model comes from the other known speech provider.
    """
    api = getattr(state.model, "api", "")
    if api in _SOLVER_FOR_PROVIDER and api != expected:
        raise ValueError(
            f"Model {state.model} is a {api!r} checkpoint but the task is running "
            f"{_SOLVER_FOR_PROVIDER[expected]}(). Its prompt format is not this one's: run it "
            f"with -T prompt_style={api} (or pass solver={_SOLVER_FOR_PROVIDER[api]}(...))."
        )


def _replace(spec: FormatSpec, **changes) -> FormatSpec:
    """Return *spec* with *changes* applied."""
    from dataclasses import replace

    return replace(spec, **changes)


_TOKENIZERS: dict[str, object] = {}


def _tokenizer_for(source: str) -> object:
    """Load (once) the tokenizer whose chat template training used.

    Loaded from the checkpoint rather than reconstructed, because the chat
    template is part of the checkpoint and a reimplementation of it would be
    one more thing that can drift out of parity.
    """
    if source in _TOKENIZERS:
        return _TOKENIZERS[source]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    _TOKENIZERS[source] = tokenizer
    return tokenizer
