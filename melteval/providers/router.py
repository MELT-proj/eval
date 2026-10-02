"""The ``melt`` provider: turns a model name into the class that runs it.

inspect takes everything before the first ``/`` of ``--model`` as the
provider and hands this module the rest. Only one provider can be registered
per name, so ``melt/<checkpoint>`` and ``melt/hf/<model>`` both arrive here, and
this module picks the class (see :mod:`melteval.providers` for the scheme):

    melt/hf/<key>       HF_MODELS[<key>], loaded from its pinned revision
    melt/vllm/<model>   reserved: an error
    melt/<checkpoint>   MELTAPI

The generation code of each model lives in its own module; nothing here does
anything but choose.

inspect's own ``hf`` provider cannot run the baselines: it refuses audio
content (``message_content_to_string`` in ``inspect_ai/model/_providers/hf.py``).
"""

from __future__ import annotations

from typing import Any

from inspect_ai.model import GenerateConfig, ModelAPI, modelapi

from melteval.providers.hf import HFSpeechChatAPI
from melteval.providers.melt import MELTAPI
from melteval.providers.qwen2_audio import Qwen2AudioAPI
from melteval.providers.qwen3_omni import Qwen3OmniAPI


#: The off-the-shelf models ``melt/hf/<key>`` can name. Each class carries its
#: Hub ``repo`` and pinned ``revision``.
HF_MODELS: dict[str, type[HFSpeechChatAPI]] = {
    "qwen2_audio": Qwen2AudioAPI,
    "qwen3_omni": Qwen3OmniAPI,
}


@modelapi(name="melt")
def melt():
    """Register the ``melt`` provider; inspect calls the returned factory with the model name."""
    return create_model


def create_model(
    model_name: str,
    base_url: str | None = None,
    api_key: str | None = None,
    config: GenerateConfig = GenerateConfig(),
    **model_args: Any,
) -> ModelAPI:
    """Build the model ``melt/<model_name>``.

    Args:
        model_name: What follows ``melt/``: ``hf/<key>``, ``vllm/<model>`` or a
            MELT checkpoint path.
        base_url: Unused; a local model has no endpoint.
        api_key: Unused; a local model needs no credential.
        config: Default generation config.
        **model_args: ``-M`` arguments, forwarded to the class. For
            ``hf/<key>``, ``revision=<sha>`` loads another Hub commit and
            ``path=<dir>`` a local checkpoint of the same architecture.

    Raises:
        ValueError: For ``hf/`` with a key not in :data:`HF_MODELS`, or for
            ``vllm/``, which is not implemented.
    """
    kind, _, rest = model_name.partition("/")
    if kind == "hf":
        if rest not in HF_MODELS:
            known = ", ".join(f"melt/hf/{key}" for key in HF_MODELS)
            raise ValueError(f"Unknown model melt/hf/{rest}; known: {known}.")
        cls = HF_MODELS[rest]
        path = model_args.pop("path", None)
        revision = model_args.pop("revision", None if path else cls.revision)
        return cls(model_name, base_url, api_key, config, path=path or cls.repo, revision=revision, **model_args)
    if kind == "vllm":
        raise ValueError(f"melt/{model_name}: the vllm path is reserved and not implemented.")
    return MELTAPI(model_name, base_url, api_key, config, **model_args)
