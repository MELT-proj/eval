"""In-process model provider for Qwen3-Omni (``Qwen/Qwen3-Omni-30B-A3B-Instruct``).

Registered as ``qwen3_omni``, so the model is addressed as
``--model qwen3_omni/<path-or-hub-id>``; pass the snapshot directory in the
HuggingFace cache, as for Qwen2-Audio (see docs/baseline-providers.md).

Everything but three model-specific facts comes from
:mod:`melteval.providers.hf`:

* **Text only.** Qwen3-Omni is a thinker (the LLM) plus a talker that speaks
  the answer. The talker is dropped after loading and ``return_audio=False``
  is passed, so ``generate()`` returns ``(token_ids, None)`` and only the
  first element is used.
* **Its ``generate()`` renames the decoding arguments.** A plain
  ``max_new_tokens`` is *not* an error there: it lands in a shared bucket that
  loses to the method's own ``thinker_max_new_tokens=1024``, so a run asking
  for 256 tokens silently decodes up to 1024. Every decoding argument goes in
  with the ``thinker_`` prefix instead.
* **No 30 s window -- but only with ``truncation=False``.** The encoder takes
  long input in chunks, so long clips (MCIF ``long``) can go in whole. Its
  processor, however, extracts features with a ``WhisperFeatureExtractor``,
  whose ``truncation=True`` default cuts every clip to Whisper's 30 s; the
  processor overrides ``padding`` "to avoid default truncation" and leaves
  ``truncation`` alone (transformers 4.57.6). Nothing fails when it does: a
  333 s MCIF talk became the same 390 audio tokens as its first 30 s, and the
  model transcribed those 30 s and stopped. So the call passes
  ``truncation=False`` explicitly.

Nothing heavy is imported at module scope: registering the provider must not
cost a torch import.
"""

from __future__ import annotations

from typing import Any

from inspect_ai.model import GenerateConfig, modelapi

from melteval.providers.base import _generate_kwargs, _Request
from melteval.providers.hf import HFSpeechChatAPI, generate, render


FAMILY = "Qwen3-Omni"
PROCESSOR_KWARGS = {"truncation": False}


@modelapi(name="qwen3_omni")
class Qwen3OmniAPI(HFSpeechChatAPI):
    """Generate text from Qwen3-Omni (or a checkpoint of the same architecture).

    Takes the constructor arguments of :class:`HFSpeechChatAPI`. At bf16 the
    30B-A3B weights are ~70 GB: one H200 holds them, an 80 GB card needs
    ``-M device_map=auto`` across two.
    """

    family = FAMILY

    def _load_classes(self) -> tuple[Any, Any]:
        """Qwen3-Omni's model and processor classes."""
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        return Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    def _prepare_model(self) -> None:
        """Drop the speech-output half; an eval only reads text."""
        self.model.disable_talker()

    def _processor_kwargs(self) -> dict[str, Any]:
        """Keep the feature extractor from cutting audio to 30 s (see module docstring)."""
        return dict(PROCESSOR_KWARGS)

    def _gen_kwargs(self, config: GenerateConfig) -> dict[str, Any]:
        """Route the decoding arguments to the thinker (see module docstring)."""
        return _thinker_kwargs(config)

    def _sequences(self, output: Any) -> Any:
        """``generate()`` returns ``(token_ids, audio)``; keep the ids."""
        return output[0]


def _thinker_kwargs(config: GenerateConfig) -> dict[str, Any]:
    """The shared greedy-by-default kwargs, each prefixed for Qwen3-Omni's thinker."""
    kwargs = {f"thinker_{key}": value for key, value in _generate_kwargs(config).items()}
    kwargs["return_audio"] = False
    return kwargs


# Kept as module functions (see melteval.providers.hf) so a test can drive the
# batch logic around a fake processor and model.
_render = render


def _generate(model, processor, device: str, batch: list[_Request], prompts: list[str]) -> list[str]:
    """Generate one completion per request, as :meth:`Qwen3OmniAPI._generate_batch` does."""
    return generate(
        model,
        processor,
        device,
        batch,
        prompts,
        gen_kwargs=_thinker_kwargs(batch[0].config),
        family=FAMILY,
        sequences=lambda output: output[0],
        processor_kwargs=PROCESSOR_KWARGS,
    )
