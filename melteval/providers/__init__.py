"""Model providers, and the names that address them.

inspect names a model ``<provider>/<model>``: the provider is who runs it, the
rest is which model (``openai/gpt-4o``, ``vllm/openai-community/gpt2``). Here:

    melt/<checkpoint>     a MELT checkpoint, through MELT's own generation path
    melt/hf/<model>       an off-the-shelf speech LLM, through transformers'
                          ``generate()`` (the models are listed in
                          ``melteval.providers.router.HF_MODELS``)
    melt/vllm/<model>     reserved for a speech LLM served by vLLM; not implemented
    smurf/<checkpoint>    a SMURF (NeMo/SALM) checkpoint, through SALM's generation path

Each of these is a *family*: it decides which prompt path the task must run
(``-T prompt_style=<family>``, see ``melteval.tasks.SOLVERS``). The families
are not interchangeable.

This module imports nothing, so a shell script or a launcher can use
:func:`model_family` without loading a provider.
"""

#: Families that have a prompt path in this harness.
FAMILIES = ("melt", "hf", "smurf")


def model_family(api: str, name: str) -> str | None:
    """Return the family of the model ``<api>/<name>``.

    Args:
        api: The provider, e.g. ``melt``.
        name: Everything after it, e.g. ``hf/qwen2_audio`` or a checkpoint path.

    Returns:
        ``"melt"``, ``"hf"``, ``"vllm"`` or ``"smurf"``, or ``None`` for a model
        from any other provider (``mockllm``, an API judge), which has no
        prompt format of its own to get wrong.
    """
    if api == "smurf":
        return "smurf"
    if api != "melt":
        return None
    for family in ("hf", "vllm"):
        if name.startswith(f"{family}/"):
            return family
    return "melt"


def family_of(model: str) -> str | None:
    """:func:`model_family` for a whole model string, ``melt/hf/qwen2_audio``."""
    api, _, name = model.partition("/")
    return model_family(api, name)


def ignores_instruction(model: str) -> bool:
    """Whether *model* consumes only audio and never reads the prompt text.

    Whisper is the one case so far: it transcribes (or, with ``task=translate``,
    translates into English) whatever audio it is given, so an instruction would
    be a string the model cannot use. The launcher uses this to avoid demanding
    one for benchmarks whose samples carry none, and the solver to allow a
    sample to go in without one.

    Args:
        model: A whole model string, e.g. ``melt/hf/whisper``.

    Returns:
        ``True`` for a provider whose class sets ``ignores_instruction``.
    """
    if family_of(model) != "hf":
        return False
    # Imported lazily: this module deliberately loads no provider (or torch)
    # just to answer a naming question.
    from melteval.providers.router import HF_MODELS

    key = model.partition("/")[2].removeprefix("hf/")
    return bool(getattr(HF_MODELS.get(key), "ignores_instruction", False))
