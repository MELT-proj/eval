"""Apertus-v1.5-8B (swiss-ai), addressed as ``--model melt/hf/apertus``.

Apertus 1.5 is an omni model: text, images and audio in, text out. It is loaded
through ``transformers`` like the Qwen baselines (:mod:`melteval.providers.hf`),
but from Swiss AI's fork of ``transformers``, pinned on the model card, because the
integration is not in a released version yet. The fork pins its own stack, so it needs
**its own venv** (docs/baseline-providers.md), as DeSTA2.5 does.

What differs from the Qwen models:

* **24 kHz audio.** The model tokenizes audio with a WavTokenizer fed 24 kHz mono
  waveforms (40 tokens per second), and its processor rejects another declared
  rate. The readers resample to 16 kHz on load, so :meth:`ApertusAPI._generate_batch`
  upsamples each clip to 24 kHz first. The band above 8 kHz is gone already, which
  speech recognition does not need, but the model sees slightly different audio from
  one fed the original recording.
* **The audio tower stays in float32.** The model card says not to recast the model:
  the vision and audio parts remain float32 inside a half-precision load. The model is
  loaded with ``dtype=auto`` and the processor's features are not cast to the model dtype.
* **The chat template is the model's.** It renders ``{"type": "audio"}`` as ``<|audio|>``,
  which the processor expands to the audio's tokens, and adds the model's own system
  turn. The solver hands over the bare instruction, as for the Qwen models.

The checkpoint is gated: accept the license on the Hub and make ``HF_TOKEN`` (or a
logged-in ``huggingface-cli``) available when downloading.

Nothing heavy is imported at module scope: registering the provider must not cost a
torch import.
"""

from __future__ import annotations

import dataclasses
import logging

from inspect_ai.model import GenerateConfig

from melteval.providers.base import DEFAULT_BATCH_SIZE, DEFAULT_BATCH_WINDOW, _Request
from melteval.providers.hf import HFSpeechChatAPI


logger = logging.getLogger(__name__)

FAMILY = "Apertus-1.5"

#: Sampling rate of the audio the model's tokenizer takes.
SAMPLE_RATE = 24000


class ApertusAPI(HFSpeechChatAPI):
    """Generate from Apertus-v1.5-8B (or a checkpoint of the same architecture).

    Takes the constructor arguments of :class:`HFSpeechChatAPI`, with ``dtype`` defaulting
    to ``auto`` (see the module docstring).
    """

    family = FAMILY
    repo = "swiss-ai/Apertus-v1.5-8B"
    revision = "1519de0ae7f3d114879291810a58259966f64c99"
    cast_audio_features = False

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        dtype: str = "auto",
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_window: int = DEFAULT_BATCH_WINDOW,
        **model_args: object,
    ) -> None:
        """Load the model and its processor; arguments as in :class:`HFSpeechChatAPI`.

        ``attn_implementation`` defaults to ``None`` (transformers' own choice) instead of
        the ``sdpa`` the base class forces: the vision tokenizer submodel has no SDPA
        support, so forcing it raises at load time.
        """
        model_args.setdefault("attn_implementation", None)
        super().__init__(
            model_name,
            base_url,
            api_key,
            config,
            dtype=dtype,
            batch_size=batch_size,
            batch_window=batch_window,
            **model_args,
        )

    def _load_classes(self) -> tuple[type, type]:
        """Apertus's multimodal model and processor classes (from the swiss-ai fork)."""
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        return AutoModelForMultimodalLM, AutoProcessor

    def _generate_batch(self, batch: list[_Request]) -> list[str]:
        """Resample the batch's audio to 24 kHz, then generate as every HF chat model does."""
        return super()._generate_batch([to_sample_rate(r, SAMPLE_RATE) for r in batch])


def to_sample_rate(request: _Request, sample_rate: int) -> _Request:
    """Return *request* with its audio resampled to *sample_rate*.

    Args:
        request: A request whose audio, if any, is a mono float array at ``request.sample_rate``.
        sample_rate: The rate the model takes.

    Returns:
        *request* itself if it carries no audio or is already at *sample_rate*; otherwise a
        copy with the resampled audio and its duration (unchanged, up to rounding).
    """
    if request.audio is None or request.sample_rate == sample_rate:
        return request
    import librosa

    audio = librosa.resample(request.audio, orig_sr=request.sample_rate, target_sr=sample_rate)
    return dataclasses.replace(request, audio=audio, sample_rate=sample_rate, duration=len(audio) / sample_rate)
