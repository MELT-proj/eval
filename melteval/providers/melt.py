"""In-process model provider for MELT checkpoints.

Registered as ``melt``, so a checkpoint is addressed as
``--model melt/<path-to-checkpoint>``.

What is specific to MELT is here: loading a ``MELTForCausalLM`` plus its
processor, and turning one padded batch into strings. Resolving the audio
locator off the event loop and collecting inspect's one-at-a-time
``generate()`` calls into batches are the same problem for every local speech
checkpoint, and live in :mod:`melteval.providers.base`.

Nothing heavy is imported at module scope: registering the provider must not
cost a torch import.
"""

from __future__ import annotations

import logging
from typing import Any

from inspect_ai.model import GenerateConfig, modelapi

from melteval.providers.base import DEFAULT_BATCH_SIZE, BatchedSpeechAPI, _Request


logger = logging.getLogger(__name__)


@modelapi(name="melt")
class MELTAPI(BatchedSpeechAPI):
    """Generate from a local MELT checkpoint."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        processor: str | None = None,
        device: str | None = None,
        dtype: str = "bfloat16",
        batch_size: int = DEFAULT_BATCH_SIZE,
        **model_args: Any,
    ) -> None:
        """Load a checkpoint.

        Args:
            model_name: Path to the checkpoint directory.
            base_url: Unused; a local model has no endpoint.
            api_key: Unused; a local model needs no credential.
            config: Default generation config.
            processor: Path to the processor, when it is not saved beside the
                model (as it is not when the run directory holds it instead).
            device: Torch device. Defaults to CUDA when available.
            dtype: Compute dtype for the model weights.
            batch_size: Samples per forward pass. Also the concurrency inspect
                is allowed, since more in flight than fit in a batch only adds
                queueing.
            **model_args: Forwarded to ``from_pretrained``.
        """
        super().__init__(model_name, base_url, api_key, config, batch_size=batch_size)

        import torch
        from melt.modeling import MELTForCausalLM, MELTProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        logger.info("Loading MELT checkpoint from %s onto %s", model_name, self.device)
        self.model = MELTForCausalLM.from_pretrained(
            model_name, dtype=torch_dtype, **model_args
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = MELTProcessor.from_pretrained(processor or model_name)

    def close(self) -> None:
        """Drop the model so the GPU memory is released."""
        self.model = None
        self.processor = None

    def _generate_batch(self, batch: list[_Request]) -> list[str]:
        """Run one padded batch through the model."""
        import torch

        inputs = self.processor(
            text=[r.text for r in batch],
            audio=_batched_audio(batch),
            sampling_rate=batch[0].sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if hasattr(value, "to")
        }
        # MELTProcessor always produces float32 features; a model loaded in a
        # lower-precision dtype (bfloat16 by default here) fails inside the
        # audio encoder's LayerNorm with "expected scalar type Float but found
        # BFloat16" unless the features are cast to match. Read the dtype off
        # the model itself rather than hardcoding it, so this keeps working
        # under `-M dtype=float16` or `=float32` too.
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(self.model.dtype)

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=inputs["input_ids"],
                input_features=inputs.get("input_features"),
                features_attention_mask=inputs.get("features_attention_mask"),
                # generate() raises without this when audio is present: merged
                # embeddings are left-padded and pad positions are otherwise
                # indistinguishable from real tokens.
                attention_mask=inputs.get("attention_mask"),
                **_generate_kwargs(batch[0].config),
            )

        # generate() delegates with inputs_embeds, so the decoder returns only
        # the newly generated tokens -- no prompt to strip.
        return [text.strip() for text in self.processor.batch_decode(generated, skip_special_tokens=True)]


def _batched_audio(batch: list[_Request]) -> list[list[Any]] | None:
    """Build the ``audio`` argument for a batched :class:`MELTProcessor` call.

    The processor's batched contract is list-of-lists aligned with ``text``:
    ``audio[i]`` is the (possibly empty) list of audio arrays for ``text[i]``'s
    audio tokens, not a flat list of arrays belonging to the batch as a whole.
    A flat list -- what this returned before the bug that motivated this
    function's extraction -- fails inside the processor with "audio[0] must be
    a list of audio arrays", found running a real batch end to end (no unit
    test exercised this shape before).

    Returns:
        ``None`` when no request in the batch carries audio (an all-text
        batch); otherwise one list per request, `[]` for a text-only request
        mixed into an otherwise-audio batch.
    """
    if not any(r.audio is not None for r in batch):
        return None
    return [[r.audio] if r.audio is not None else [] for r in batch]


def _generate_kwargs(config: GenerateConfig) -> dict[str, Any]:
    """Translate an inspect generate config into transformers kwargs.

    Greedy by default: an eval that samples is measuring the sampler as much as
    the model, and two runs would not be comparable.
    """
    kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_tokens or 256,
        "use_cache": True,
        "do_sample": config.temperature is not None and config.temperature > 0,
    }
    if config.temperature is not None and config.temperature > 0:
        kwargs["temperature"] = config.temperature
    if config.top_p is not None:
        kwargs["top_p"] = config.top_p
    if config.top_k is not None:
        kwargs["top_k"] = config.top_k
    if config.num_choices is not None and config.num_choices > 1:
        kwargs["num_return_sequences"] = config.num_choices
    return kwargs
