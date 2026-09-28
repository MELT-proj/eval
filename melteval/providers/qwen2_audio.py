"""In-process model provider for Qwen2-Audio-7B-Instruct.

Registered as ``qwen2_audio``, so the model is addressed as
``--model qwen2_audio/<path-or-hub-id>``. A baseline has no training run of
ours behind it, so "the checkpoint" is simply what ``from_pretrained`` loads:
pass the snapshot directory in the HuggingFace cache
(``$HF_HOME/hub/models--Qwen--Qwen2-Audio-7B-Instruct/snapshots/<sha>``) so
the revision is pinned in the log and nothing reaches for the Hub.

Same division of labour as SMURF (see docs/baseline-providers.md); what is
not specific to this model lives in :mod:`melteval.providers.hf`:

* **The chat template is the model's.** The solver
  (:func:`melteval.solver.instruction_prompt`) hands over the bare
  instruction; this provider wraps it, with the audio, in the conversation
  shape the model card uses and renders it with the processor's own
  ``apply_chat_template``.
* **The audio token is the model's.** The template emits
  ``<|audio_bos|><|AUDIO|><|audio_eos|>`` and the processor expands it to as
  many positions as the encoder produces frames.

Nothing heavy is imported at module scope: registering the provider must not
cost a torch import.
"""

from __future__ import annotations

from typing import Any

from inspect_ai.model import modelapi

from melteval.providers.base import _generate_kwargs, _Request
from melteval.providers.hf import HFSpeechChatAPI, generate, render


FAMILY = "Qwen2-Audio"


@modelapi(name="qwen2_audio")
class Qwen2AudioAPI(HFSpeechChatAPI):
    """Generate from Qwen2-Audio-7B-Instruct (or a checkpoint of the same architecture).

    Takes the constructor arguments of :class:`HFSpeechChatAPI`.
    """

    family = FAMILY

    def _load_classes(self) -> tuple[Any, Any]:
        """Qwen2-Audio's model and processor classes."""
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        return Qwen2AudioForConditionalGeneration, AutoProcessor

    def _max_audio_seconds(self) -> float | None:
        """The processor pads every clip to its window and cuts off the rest."""
        return self.processor.feature_extractor.chunk_length


# Kept as module functions (see melteval.providers.hf) so a test can drive the
# batch logic around a fake processor and model.
_render = render


def _generate(model, processor, device: str, batch: list[_Request], prompts: list[str]) -> list[str]:
    """Generate one completion per request, as :meth:`Qwen2AudioAPI._generate_batch` does."""
    return generate(
        model,
        processor,
        device,
        batch,
        prompts,
        gen_kwargs=_generate_kwargs(batch[0].config),
        family=FAMILY,
        max_seconds=processor.feature_extractor.chunk_length,
    )
