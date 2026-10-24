"""Gemma 4 E4B-it, addressed as ``--model melt/hf/gemma4``.

Loaded through plain ``transformers`` (``AutoModelForMultimodalLM``); the
shared batching, audio loading and ``generate()`` plumbing is in
:mod:`melteval.providers.hf`. What is specific to the model:

* **Audio goes after the text** in the user turn, as the model card says.
* **No clip-length check.** The card states a 30 s limit but the model does not
  raise on longer audio; its feature extractor truncates to 30 s without a
  word. The audio is passed as it is, and each clip over the limit is written
  to its sample's output metadata (``audio_over_model_limit_seconds``) and
  counted in a warning, so a bad WER on a long clip can be explained and the
  report can count them.
* **Translation output.** The card's speech-translation prompt asks for the
  transcript, a newline, ``<Target>: `` and the translation. When the prompt
  is that one, the provider keeps only the text after the marker (the raw
  output goes to ``raw_output`` in the metadata; a missing marker is flagged
  ``translation_marker_missing`` and the answer returned whole).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from melteval.providers.base import _generate_kwargs, _Request
from melteval.providers.hf import HFSpeechChatAPI, generate


logger = logging.getLogger(__name__)

FAMILY = "Gemma 4"
#: The model card's audio limit in seconds.
MAX_AUDIO_SECONDS = 30.0
#: The marker the card's translation prompt asks the model to print before the translation.
_MARKER = re.compile(r"then one newline, then output the string '([^']+?): '")


class Gemma4API(HFSpeechChatAPI):
    """Generate from Gemma 4 E4B-it (or a checkpoint of the same architecture)."""

    family = FAMILY
    repo = "google/gemma-4-E4B-it"
    revision = "ee0ef6023621cff504d758262d4e04895a5af4a2"

    def _load_classes(self) -> tuple[Any, Any]:
        """Gemma 4's model and processor classes."""
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        return AutoModelForMultimodalLM, AutoProcessor

    def _generate_batch(self, batch: list[_Request]) -> list[str]:
        """Run one padded batch: text first, then the audio, as the model card says."""
        mark_over_limit(batch, MAX_AUDIO_SECONDS, FAMILY)
        prompts = [render(self.processor, r.text) for r in batch]
        if not self._logged_prompt:
            logger.info("%s prompt as sent (first sample): %r", self.family, prompts[0])
            self._logged_prompt = True
        completions = generate(
            self.model,
            self.processor,
            self.device,
            batch,
            prompts,
            gen_kwargs=_generate_kwargs(batch[0].config),
            family=self.family,
            max_seconds=None,
        )
        return [extract_after_marker(c, r) for c, r in zip(completions, batch)]


def render(processor, instruction: str) -> str:
    """The instruction, then one audio, in the model's own chat template."""
    conversation = [
        {
            "role": "user",
            "content": [{"type": "text", "text": instruction}, {"type": "audio", "audio": "melteval://sample"}],
        }
    ]
    return processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)


def mark_over_limit(batch: list[_Request], limit: float, family: str) -> None:
    """Record, but do not cut or refuse, the clips longer than the model's stated limit."""
    over = [r for r in batch if r.duration > limit]
    for r in over:
        r.metadata["audio_over_model_limit_seconds"] = round(r.duration, 3)
    if over:
        logger.warning(
            "%s: %d clip(s) in this batch are over its stated %s s limit (longest %.1f s); passed as they are.",
            family,
            len(over),
            limit,
            max(r.duration for r in over),
        )


def extract_after_marker(completion: str, request: _Request) -> str:
    """Keep the text after ``<Target>: `` in a transcript-then-translation answer.

    The marker is read from the prompt (the card's translation instruction names it).
    When the model did not print it, the whole answer is returned and the sample is
    flagged ``translation_marker_missing`` rather than guessed at.
    """
    found = _MARKER.search(request.text)
    if found is None:
        return completion
    marker = f"{found.group(1)}: "
    head, sep, tail = completion.partition(marker)
    request.metadata["raw_output"] = completion
    if not sep:
        request.metadata["translation_marker_missing"] = True
        return completion
    return tail.strip()
