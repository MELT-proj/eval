"""What every HuggingFace chat-template speech LLM needs, whatever the model.

Baselines loaded through ``transformers`` (Qwen2-Audio, Qwen3-Omni, ...) share
one shape: the model ships a processor whose ``apply_chat_template`` renders the
conversation and places the model's own audio token, the processor expands that
token to one position per encoder frame, and ``generate()`` runs from
``input_ids`` so its output starts with the prompt. Everything in that sentence
is written once, here. A subclass says only what genuinely differs: which
classes to load, how long a clip the model can take in one piece, and how its
``generate()`` wants its arguments and returns its output.

The batch logic lives in module functions rather than on the class because
``@modelapi`` replaces a provider class with a factory function, so a class
cannot be instantiated around a fake model in a test.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from inspect_ai.model import GenerateConfig

from melteval.providers.base import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_WINDOW,
    BatchedSpeechAPI,
    _generate_kwargs,
    _Request,
)


logger = logging.getLogger(__name__)


class HFSpeechChatAPI(BatchedSpeechAPI):
    """A ``transformers`` speech LLM that applies its own chat template.

    Subclasses set :attr:`family` and implement :meth:`_load_classes`; the
    remaining hooks have defaults that fit a plain ``generate()``.
    """

    #: Human-readable model family, for logs and error messages.
    family: str = "HF speech model"
    #: HuggingFace Hub repository the model is loaded from by default.
    repo: str | None = None
    #: Hub commit loaded by default. Pinned so a number is always tied to the
    #: same weights (docs/baseline-providers.md lists them).
    revision: str | None = None

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        device: str | None = None,
        dtype: str = "bfloat16",
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_window: int = DEFAULT_BATCH_WINDOW,
        text_only: bool = False,
        truncate_long_audio: bool = False,
        path: str | None = None,
        revision: str | None = None,
        **model_args: Any,
    ) -> None:
        """Load the model and its processor.

        Args:
            model_name: The model's name in the log, e.g. ``hf/qwen2_audio``
                (see :func:`melteval.providers.router.create_model`). Also where
                it is loaded from when *path* is not given.
            base_url: Unused; a local model has no endpoint.
            api_key: Unused; a local model needs no credential.
            config: Default generation config.
            device: Torch device. Defaults to CUDA when available. Ignored
                with ``device_map``, where the weights place themselves.
            dtype: Compute dtype for the model weights.
            batch_size: Samples per forward pass.
            batch_window: How many batches' worth of requests to pool and sort
                by audio length before batching (see :class:`BatchedSpeechAPI`).
            text_only: Serve text-only requests, with no audio in the prompt
                -- for using the model as a judge (``--model-role
                grader="{model: melt/hf/qwen3_omni, model_args: {text_only:
                true}}"``). Off by default, and a request carrying audio is
                refused in this mode: in a speech eval, a sample that lost its
                audio would otherwise be answered from the text alone, with a
                plausible score.
            truncate_long_audio: Cut clips longer than the processor's window
                down to it instead of refusing the batch -- what the model does
                out of the box. Off by default. When on, each truncated sample
                says so in its output metadata (``audio_truncated_from_seconds``)
                and the run logs a warning with the count, so the choice is
                visible in the log rather than silent.
            path: Hub id or local directory to load the weights from.
            revision: Hub commit to load; ``None`` for a local directory.
            **model_args: Forwarded to ``from_pretrained``. ``attn_implementation``
                defaults to ``sdpa``, which needs nothing extra installed;
                ``device_map=auto`` shards a model too big for one GPU.
        """
        super().__init__(model_name, base_url, api_key, config, batch_size=batch_size, batch_window=batch_window)
        model_cls, processor_cls = self._load_classes()
        source = path or model_name
        logger.info("Loading %s from %s (revision %s)", self.family, source, revision or "-")
        self.model, self.processor, self.device = load(
            model_cls, processor_cls, source, device=device, dtype=dtype, revision=revision, **model_args
        )
        self._prepare_model()
        # -M/model_args values arrive YAML-parsed, but a quoted "false" would
        # be a truthy string; compare explicitly rather than trusting bool().
        self.text_only = _truthy(text_only)
        self.truncate_long_audio = _truthy(truncate_long_audio)
        self._logged_prompt = False

    def _load_classes(self) -> tuple[Any, Any]:
        """Return ``(model_class, processor_class)`` to load the checkpoint with."""
        raise NotImplementedError

    def _prepare_model(self) -> None:
        """Adjust the loaded model before the first batch (default: nothing)."""

    def _max_audio_seconds(self) -> float | None:
        """Longest clip the processor takes without truncating it (default: no limit)."""
        return None

    def _processor_kwargs(self) -> dict[str, Any]:
        """Extra keyword arguments for the processor's audio call (default: none)."""
        return {}

    def _gen_kwargs(self, config: GenerateConfig) -> dict[str, Any]:
        """Translate an inspect generate config into this model's ``generate()`` kwargs."""
        return _generate_kwargs(config)

    def _sequences(self, output: Any) -> Any:
        """Extract the token ids from what ``generate()`` returned (default: it is them)."""
        return output

    def close(self) -> None:
        """Drop the model so the GPU memory is released."""
        self.model = None
        self.processor = None

    def _generate_batch(self, batch: list[_Request]) -> list[str]:
        """Run one padded batch through the model."""
        prompts = [render(self.processor, r.text, with_audio=not self.text_only) for r in batch]
        if not self._logged_prompt:
            logger.info("%s prompt as sent (first sample): %r", self.family, prompts[0])
            self._logged_prompt = True
        return generate(
            self.model,
            self.processor,
            self.device,
            batch,
            prompts,
            gen_kwargs=self._gen_kwargs(batch[0].config),
            family=self.family,
            max_seconds=self._max_audio_seconds(),
            sequences=self._sequences,
            with_audio=not self.text_only,
            truncate=self.truncate_long_audio,
            processor_kwargs=self._processor_kwargs(),
        )


def _truthy(value: Any) -> bool:
    """A model arg as a bool. -M values arrive YAML-parsed, but a quoted
    "false" would be a truthy string, so compare explicitly."""
    return str(value).strip().lower() in ("true", "1", "yes")


def load(
    model_cls,
    processor_cls,
    model_name: str,
    device: str | None,
    dtype: str,
    revision: str | None = None,
    **model_args: Any,
):
    """Load a model and its processor, ready for batched generation.

    *revision* is a Hub commit. With ``HF_HUB_OFFLINE=1`` a commit hash is
    looked up in the cache directly, so a pinned revision loads offline once
    it has been downloaded.

    Returns:
        ``(model, processor, device)``, where *device* is where inputs go.
    """
    import torch

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    model_args.setdefault("attn_implementation", "sdpa")
    model = model_cls.from_pretrained(model_name, dtype=torch_dtype, revision=revision, **model_args)
    if "device_map" in model_args:
        device = str(model.device)
    else:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    model.eval()

    processor = processor_cls.from_pretrained(model_name, revision=revision)
    # Batched generation from a decoder-only model has to be left-padded:
    # with right padding, the pad tokens sit between the prompt and the first
    # generated token, and the one slice in `generate` that strips the prompt
    # would cut into it. Set explicitly rather than trusting whatever the
    # tokenizer config ships with.
    processor.tokenizer.padding_side = "left"
    return model, processor, device


def conversation(instruction: str, with_audio: bool = True) -> list[dict]:
    """One user turn: the audio first, then the instruction, as the Qwen model cards do.

    The audio entry only has to be recognised as audio by the template, which
    emits the model's placeholder for it; the waveform itself goes to the
    processor, so the locator here is never read. Without audio (a judge
    prompt), the turn is the text alone.
    """
    content = [{"type": "audio", "audio_url": "melteval://sample"}] if with_audio else []
    content.append({"type": "text", "text": instruction})
    return [{"role": "user", "content": content}]


def render(processor, instruction: str, with_audio: bool = True) -> str:
    """Wrap *instruction* (and one audio, unless *with_audio* is off) in the model's own chat template."""
    return processor.apply_chat_template(
        conversation(instruction, with_audio), add_generation_prompt=True, tokenize=False
    )


def generate(
    model,
    processor,
    device: str,
    batch: list[_Request],
    prompts: list[str],
    gen_kwargs: dict[str, Any],
    family: str,
    max_seconds: float | None = None,
    sequences: Callable[[Any], Any] = lambda output: output,
    with_audio: bool = True,
    truncate: bool = False,
    processor_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """Generate one completion per request from already-rendered *prompts*.

    Args:
        model: A loaded model.
        processor: Its processor, with the tokenizer left-padding.
        device: Where the inputs go.
        batch: The requests, whose audio goes to the processor.
        prompts: One chat-templated prompt per request, in order.
        gen_kwargs: Keyword arguments for ``model.generate``.
        family: Model family, for error messages.
        max_seconds: Longest clip the processor takes whole, if it truncates.
        sequences: Pulls the token ids out of ``generate()``'s return value.
        with_audio: Whether the prompts carry audio. Off only for a judge
            (see ``text_only`` on :class:`HFSpeechChatAPI`); then every
            request must carry none.
        truncate: Cut audio longer than *max_seconds* to it instead of
            refusing the batch; each cut request gets
            ``audio_truncated_from_seconds`` in its metadata.
        processor_kwargs: Extra keyword arguments for the processor call
            that carries the audio.

    Returns:
        The decoded new tokens for each request, in order.
    """
    import torch

    if with_audio:
        feature_extractor = processor.feature_extractor
        audios = [r.audio for r in batch]
        if truncate and max_seconds is not None:
            audios = _truncate(batch, max_seconds, family)
        check_audio(batch, feature_extractor.sampling_rate, None if truncate else max_seconds, family)
        inputs = processor(
            text=prompts,
            audio=audios,
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
            **(processor_kwargs or {}),
        )
    else:
        with_audio_requests = [i for i, r in enumerate(batch) if r.audio is not None]
        if with_audio_requests:
            raise ValueError(
                f"{len(with_audio_requests)} request(s) carry audio, but {family} was loaded with "
                "text_only=true (as a judge). Its prompt has no audio slot, so the audio would be "
                "dropped without a trace; load it without text_only to evaluate speech."
            )
        inputs = processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items() if hasattr(value, "to")}
    # The feature extractor produces float32; a bf16 model's audio encoder
    # needs its input in its own dtype (same failure as in providers/melt.py).
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(model.dtype)

    with torch.no_grad():
        generated = sequences(model.generate(**inputs, **gen_kwargs))

    # generate() runs from input_ids, so the output starts with the prompt.
    # Left padding puts every row's prompt in the same leading columns, so one
    # slice strips all of them.
    new_tokens = generated[:, inputs["input_ids"].shape[1] :]
    return [
        text.strip()
        for text in processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    ]


def _truncate(batch: list[_Request], max_seconds: float, family: str) -> list[Any]:
    """Cut each request's audio to *max_seconds*, recording which were cut and from how long."""
    audios = []
    for r in batch:
        if r.audio is not None and r.duration > max_seconds:
            r.metadata["audio_truncated_from_seconds"] = round(r.duration, 3)
            audios.append(r.audio[: int(max_seconds * r.sample_rate)])
        else:
            audios.append(r.audio)
    cut = [r.duration for r in batch if "audio_truncated_from_seconds" in r.metadata]
    if cut:
        logger.warning(
            "%s: truncated %d clip(s) to its %s s window (longest %.1f s); the tail is not transcribed "
            "but is still in the reference (truncate_long_audio=true).",
            family,
            len(cut),
            max_seconds,
            max(cut),
        )
    return audios


def check_audio(batch: list[_Request], sampling_rate: int, max_seconds: float | None, family: str) -> None:
    """Reject a batch the processor would otherwise accept and quietly get wrong.

    Raises:
        ValueError: If a request has no audio, is not at the feature
            extractor's sampling rate, or is longer than *max_seconds*. A
            processor that pads every clip to a fixed window cuts the rest
            off, so a 40 s clip is transcribed as its first 30 s and scored
            against the full reference, with nothing in the log to say so.
    """
    missing = [i for i, r in enumerate(batch) if r.audio is None]
    if missing:
        raise ValueError(
            f"{len(missing)} sample(s) in the batch carry no audio. {family}'s prompt has an "
            "audio placeholder for every sample, so a text-only sample has nothing to fill it with."
        )

    wrong_rate = {r.sample_rate for r in batch if r.sample_rate != sampling_rate}
    if wrong_rate:
        raise ValueError(
            f"Audio at {sorted(wrong_rate)} Hz, but {family}'s feature extractor expects "
            f"{sampling_rate} Hz. Readers resample on load; a mismatch here means the frozen set "
            "references audio a reader passed through untouched."
        )

    if max_seconds is None:
        return
    too_long = [r.duration for r in batch if r.duration > max_seconds]
    if too_long:
        raise ValueError(
            f"{len(too_long)} sample(s) longer than {family}'s {max_seconds} s window (longest "
            f"{max(too_long):.1f} s). The processor would silently truncate them to {max_seconds} s "
            "and the hypothesis would be scored against the full reference. Filter or segment "
            "such samples before evaluating this model."
        )
