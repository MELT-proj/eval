"""Phi-4-multimodal-instruct, addressed as ``--model melt/hf/phi4_multimodal``.

The Hub repository ships the model's own code (``Phi4MMForCausalLM``, loaded
with ``trust_remote_code``), not the transformers-native port, and its prompt
is not a chat template: ``<|user|><|audio_1|>{instruction}<|end|><|assistant|>``,
which this provider writes itself. The audio goes to the processor as
``(samples, sample_rate)``.

The card suggests clips of at most 40 s but the model does not raise on longer
ones. The audio is passed as it is; clips over the limit are recorded in the
sample's metadata (``audio_over_model_limit_seconds``) and counted in a warning.
"""

from __future__ import annotations

import logging
from typing import Any

from melteval.providers.base import _generate_kwargs, _Request
from melteval.providers.gemma4 import mark_over_limit
from melteval.providers.hf import HFSpeechChatAPI


logger = logging.getLogger(__name__)

FAMILY = "Phi-4-multimodal"
#: The model card's suggested audio limit in seconds.
MAX_AUDIO_SECONDS = 40.0


class _RemoteCode:
    """Wraps a transformers Auto class so ``from_pretrained`` always trusts the repo's code."""

    def __init__(self, auto_class: Any) -> None:
        self._auto_class = auto_class

    def from_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        """Load with ``trust_remote_code=True``.

        The repository's ``config.json`` fixes ``_attn_implementation`` to
        ``flash_attention_2`` and its code reads that field, so an
        ``attn_implementation`` argument alone does not change what is used: the
        config is loaded first and gets the value.
        """
        if "dtype" in kwargs and self._auto_class.__name__ == "AutoModelForCausalLM":
            import transformers

            if tuple(int(x) for x in transformers.__version__.split(".")[:2]) < (4, 56):
                kwargs["torch_dtype"] = kwargs.pop("dtype")  # called ``dtype`` only from 4.56 on
        attn = kwargs.get("attn_implementation")
        if attn is not None and "config" not in kwargs and self._auto_class.__name__ == "AutoModelForCausalLM":
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(*args, trust_remote_code=True, revision=kwargs.get("revision"))
            config._attn_implementation = attn
            kwargs["config"] = config
        return self._auto_class.from_pretrained(*args, trust_remote_code=True, **kwargs)


class Phi4MultimodalAPI(HFSpeechChatAPI):
    """Generate from Phi-4-multimodal-instruct.

    ``attn_implementation`` defaults to ``sdpa``; the model card uses ``flash_attention_2``, which
    the remote code refuses under transformers 5. Pass ``-M attn_implementation=eager`` if sdpa fails.
    """

    family = FAMILY
    repo = "microsoft/Phi-4-multimodal-instruct"
    revision = "93f923e1a7727d1c4f446756212d9d3e8fcc5d81"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Take the arguments of :class:`HFSpeechChatAPI`."""
        kwargs.setdefault("attn_implementation", "sdpa")
        super().__init__(*args, **kwargs)

    def _load_classes(self) -> tuple[Any, Any]:
        """The Auto classes, loading the repository's own code."""
        from transformers import AutoModelForCausalLM, AutoProcessor

        return _RemoteCode(AutoModelForCausalLM), _RemoteCode(AutoProcessor)

    def _generate_batch(self, batch: list[_Request]) -> list[str]:
        """Run one padded batch through the model."""
        import torch

        mark_over_limit(batch, MAX_AUDIO_SECONDS, FAMILY)
        missing = [r for r in batch if r.audio is None]
        if missing:
            raise ValueError(f"{len(missing)} sample(s) in the batch carry no audio; {FAMILY} needs one per prompt.")
        prompts = [f"<|user|><|audio_1|>{r.text}<|end|><|assistant|>" for r in batch]
        if not self._logged_prompt:
            logger.info("%s prompt as sent (first sample): %r", self.family, prompts[0])
            self._logged_prompt = True
        inputs = self.processor(
            text=prompts,
            audios=[(r.audio, r.sample_rate) for r in batch],
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        for key, value in inputs.items():
            if hasattr(value, "is_floating_point") and value.is_floating_point():
                inputs[key] = value.to(self.model.dtype)
        with torch.no_grad():
            generated = self.model.generate(**inputs, **_generate_kwargs(batch[0].config))
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        return [
            text.strip()
            for text in self.processor.batch_decode(
                new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        ]
