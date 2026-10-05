"""Whisper large-v3 (but it's only an) ASR/translation model.

Since Whisper is not a chat model (it transcribes, or, if``task=translate``, translates into English), this providers ignores the prompt text the solver renders and only processes the audio. Basically, the instruction in the launcher is there because the launcher requires one per task.

Because of this:
* Since the provider only sees the prompt and the audio (i.e., not the sample's meta), there is not per-sample langauge to pass.
Because of this, some clips may be assigned the wrong language, leading to incorrect transcriptions or translations.
* Direction is determined by the model argument, not per-sample metadata. This means that evaluating ASR and X->en ST requires two separate runs of the same checkpoint (``-M task=transcribe`` and ``-M task=translate``).
"""

from __future__ import annotations

import logging

from melteval.providers.base import _Request
from melteval.providers.hf import HFSpeechChatAPI, check_audio


logger = logging.getLogger(__name__)

FAMILY = "Whisper"
#: Whisper's encoder window
WINDOW_SECONDS = 30 # this is just the encoder window length in seconds; but Whisper can handle longer clips in a long-form mode
TASKS = ("transcribe", "translate")


class WhisperAPI(HFSpeechChatAPI):
    """``openai/whisper-large-v3`` through ``transformers``."""

    family = FAMILY
    repo = "openai/whisper-large-v3"
    revision = "06f233fe06e710322aca913c1bc4249a0d71fce1"

    def __init__(self, *args: object, task: str = "transcribe", **kwargs: object) -> None:
        """Load Whisper.

        Args:
            task: ``transcribe`` (the audio's own language) or ``translate``
                (into English).
            *args: See :class:`HFSpeechChatAPI`.
            **kwargs: See :class:`HFSpeechChatAPI`.

        Raises:
            ValueError: If *task* is neither.
        """
        if task not in TASKS:
            raise ValueError(f"Whisper task must be one of {TASKS}, got {task!r}.")
        super().__init__(*args, **kwargs)
        self.task = task

    def _load_classes(self) -> tuple[type, type]:
        """Whisper's model and processor classes."""
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        return WhisperForConditionalGeneration, WhisperProcessor

    def _generate_batch(self, batch: list[_Request]) -> list[str]:
        """Transcribe (or translate) one padded batch; the prompt text is not used."""
        if self.text_only:
            raise ValueError("Whisper has no text-only mode: it cannot act as a judge.")
        return _transcribe(self.model, self.processor, self.device, batch, task=self.task)


def _transcribe(model, processor, device: str, batch: list[_Request], task: str) -> list[str]:
    """Run one batch through Whisper. A module function so a test can pass fakes."""
    import torch

    sampling_rate = processor.feature_extractor.sampling_rate
    check_audio(batch, sampling_rate, FAMILY)
    audios = [r.audio for r in batch]

    long_form = any(r.duration > WINDOW_SECONDS for r in batch)
    if long_form:
        # Unpadded to the longest clip with an attention mask, so generate()
        # walks it window by window instead of cutting at 30 s.
        inputs = processor(
            audios,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        )
    else:
        # Pads every clip to the 30 s the encoder expects.
        inputs = processor(audios, sampling_rate=sampling_rate, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items() if hasattr(value, "to")}
    inputs["input_features"] = inputs["input_features"].to(model.dtype)

    # Greedy, like every other provider. Language is left to auto-detection.
    gen_kwargs: dict[str, object] = {"task": task, "do_sample": False}
    if long_form:
        gen_kwargs["return_timestamps"] = True
    with torch.no_grad():
        generated = model.generate(**inputs, **gen_kwargs)
    return [text.strip() for text in processor.batch_decode(generated, skip_special_tokens=True)]
