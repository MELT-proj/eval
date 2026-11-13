"""Cohere Transcribe (cohere-transcribe-03-2026) as an ASR baseline.

A 2B conformer encoder-decoder trained from scratch for transcription only: no
translation, no chat, no timestamps. The processor is the whole prompt -- there
is no chat template and no free-form instruction. It takes the audio and one
language, builds the decoder prompt from that language token, and returns the
audio features, an attention mask, the decoder prompt and ``audio_chunk_index``.

Two consequences shape the batch step:

* **One language per processor call.** ``language`` is a single string for the
  whole call, so a batch that mixes languages is split into one call per
  language, and the results go back in request order.
* **Long audio is chunked by the processor.** Audio past ~35 s is split at quiet
  boundaries into several feature rows, listed in ``audio_chunk_index``;
  ``processor.decode(..., audio_chunk_index=..., language=...)`` reassembles them
  into one string per sample. Nothing is truncated, so there is no 30 s reject
  as for Qwen2-Audio.

The language arrives as the sample's instruction (``asr: "{lang_code}"`` in the
campaign config): the solver renders it per sample, the model has no language
detection of its own, and :func:`parse_language` validates it.
"""

from __future__ import annotations

from inspect_ai.model import GenerateConfig

from melteval.providers.base import _generate_kwargs, _Request
from melteval.providers.hf import HFSpeechChatAPI, check_audio


FAMILY = "Cohere Transcribe"
#: The languages the model covers (``CohereAsrProcessor.LANGUAGES``), ISO 639-1.
LANGUAGES = frozenset(["ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh"])


class CohereTranscribeAPI(HFSpeechChatAPI):
    """``CohereLabs/cohere-transcribe-03-2026`` through ``transformers``."""

    family = FAMILY
    repo = "CohereLabs/cohere-transcribe-03-2026"
    revision = "d552dadc2922ba6f6c1f994f8db904d52dcaf942"

    def _load_classes(self) -> tuple[type, type]:
        """Cohere Transcribe's model and processor classes."""
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        return CohereAsrForConditionalGeneration, AutoProcessor

    def _generate_batch(self, batch: list[_Request]) -> list[str]:
        """Transcribe one batch, one processor call per language present in it."""
        if self.text_only:
            raise ValueError(f"{FAMILY} has no text-only mode: it cannot act as a judge.")
        results: dict[int, str] = {}
        groups: dict[str, list[int]] = {}
        for i, request in enumerate(batch):
            groups.setdefault(parse_language(request.text), []).append(i)
        for language, indices in groups.items():
            completions = _transcribe(
                self.model,
                self.processor,
                self.device,
                [batch[i] for i in indices],
                language,
                batch[0].config,
            )
            for i, text in zip(indices, completions):
                results[i] = text
        return [results[i] for i in range(len(batch))]


def parse_language(text: str) -> str:
    """The ISO language code in a rendered instruction, validated.

    Args:
        text: What the solver rendered, e.g. ``"en"`` (surrounding whitespace and
            case are ignored).

    Returns:
        The code, lowercased.

    Raises:
        ValueError: If the text is not one of the model's 14 languages. Passing a
            language the model does not know would otherwise transcribe in the
            wrong language with no error.
    """
    code = text.strip().lower()
    if code not in LANGUAGES:
        raise ValueError(
            f"{FAMILY} needs its language as the instruction, e.g. 'en'; got {text!r}. "
            f"Supported: {', '.join(sorted(LANGUAGES))}. A benchmark whose samples carry their own "
            "prompt (MCIF, AIR-Bench) keeps it unless the task is run with -T instruction_override=true."
        )
    return code


def _transcribe(
    model,
    processor,
    device: str,
    batch: list[_Request],
    language: str,
    config: GenerateConfig,
) -> list[str]:
    """Run one same-language batch through the model.

    A module function so a test can pass a fake model and processor. The
    processor chunks long audio itself and reports the chunk-to-sample map;
    ``decode`` puts the chunks back together, so a completion is one string per
    request even when its audio was split.
    """
    import torch

    sampling_rate = processor.feature_extractor.sampling_rate
    check_audio(batch, sampling_rate, FAMILY)
    inputs = processor(
        [request.audio for request in batch],
        sampling_rate=sampling_rate,
        return_tensors="pt",
        language=language,
    )
    # A list of (sample index, chunk index | None), one per feature row: consumed
    # by decode, not by generate, so it stays off the device.
    chunk_index = inputs.pop("audio_chunk_index", None)
    inputs = {key: value.to(device) for key, value in inputs.items() if hasattr(value, "to")}
    inputs["input_features"] = inputs["input_features"].to(model.dtype)

    with torch.no_grad():
        generated = model.generate(**inputs, **_generate_kwargs(config))
    decoded = processor.decode(
        generated, skip_special_tokens=True, audio_chunk_index=chunk_index, language=language
    )
    if isinstance(decoded, str):  # a single request with no chunking decodes to one string
        decoded = [decoded]
    return [text.strip() for text in decoded]
