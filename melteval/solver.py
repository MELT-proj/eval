"""The solver: render the prompt, attach the audio, generate.

This is where format parity is applied and — just as importantly — made
visible. The rendered prompt goes into the message the model receives, so the
eval log shows exactly what was sent, and the format spec it came from is
recorded alongside it. A parity bug should be readable in the log rather than
inferred from a disappointing score.
"""

from __future__ import annotations

from inspect_ai.model import ChatMessageUser, ContentData, ContentText
from inspect_ai.solver import Generate, Solver, TaskState, solver

from melteval.dataset import AUDIO_DATA_KEY
from melteval.prompt import FormatSpec, apply_generation_format, load_format_spec, render_user_prompt


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
