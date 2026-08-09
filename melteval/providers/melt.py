"""In-process model provider for MELT checkpoints.

Registered as ``melt``, so a checkpoint is addressed as
``--model melt/<path-to-checkpoint>``.

Two things make this more than a thin wrapper around ``generate()``:

* **Audio resolution.** The provider is handed messages, never sample metadata,
  so the frozen set's audio locator travels inside the message as
  :class:`ContentData`. Resolving it (disk read plus decode) happens off the
  event loop, so several samples' audio can be fetched while the GPU works.
* **Batching.** inspect calls ``generate()`` once per sample, concurrently.
  Running a speech model one utterance at a time wastes most of a GPU, so
  requests are collected into batches by a worker thread — the same shape as
  inspect's own ``hf`` provider, scoped to the instance rather than to a module
  global.

Nothing heavy is imported at module scope: registering the provider must not
cost a torch import.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any

import anyio
from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from melteval.dataset import AUDIO_DATA_KEY


logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 8
#: How long the batch thread waits for more work before running a short batch.
BATCH_WAIT_SECONDS = 0.25


@dataclass
class _Request:
    """One sample waiting to be generated."""

    text: str
    audio: Any | None
    sample_rate: int
    config: GenerateConfig
    future: Future = field(default_factory=Future)


@modelapi(name="melt")
class MELTAPI(ModelAPI):
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
        super().__init__(model_name, base_url, api_key, [], config)

        import torch
        from melt.modeling import MELTForCausalLM, MELTProcessor

        self.batch_size = int(batch_size)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        logger.info("Loading MELT checkpoint from %s onto %s", model_name, self.device)
        self.model = MELTForCausalLM.from_pretrained(
            model_name, dtype=torch_dtype, **model_args
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.processor = MELTProcessor.from_pretrained(processor or model_name)

        self._queue: Queue[_Request] = Queue()
        self._thread: Thread | None = None
        self._thread_lock = Lock()

    def max_connections(self) -> int:
        """Cap in-flight requests at the batch size."""
        return self.batch_size

    def close(self) -> None:
        """Drop the model so the GPU memory is released."""
        self.model = None
        self.processor = None

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """Generate a completion for one sample.

        Args:
            input: Messages built by the solver: the rendered prompt plus the
                audio locator.
            tools: Unused; a speech model has no tool interface.
            tool_choice: Unused.
            config: Generation parameters for this call.

        Returns:
            The model's completion.

        Raises:
            ValueError: If the message carries no text.
        """
        text, locator = _extract(input)
        if not text:
            raise ValueError(
                "No text content in the message. The solver renders the prompt; "
                "generating without one would measure an empty input."
            )

        audio, sample_rate = None, 16000
        if locator is not None:
            audio, sample_rate = await anyio.to_thread.run_sync(_load_audio, locator)

        request = _Request(text=text, audio=audio, sample_rate=sample_rate, config=config)
        self._ensure_worker()
        self._queue.put(request)

        while True:
            if request.future.done():
                return request.future.result()
            await anyio.sleep(0.05)

    def _ensure_worker(self) -> None:
        """Start the batching thread on first use."""
        with self._thread_lock:
            if self._thread is None:
                self._thread = Thread(target=self._process_batches, daemon=True)
                self._thread.start()

    def _process_batches(self) -> None:
        """Collect queued requests into batches and run them.

        Waits briefly for a full batch rather than generating immediately, so a
        run is batched even though inspect hands requests over one at a time.
        A short batch at the tail of a run costs a fraction of a second.
        """
        while True:
            batch: list[_Request] = []
            deadline = None
            while len(batch) < self.batch_size:
                timeout = BATCH_WAIT_SECONDS if deadline is None else max(0.0, deadline - time.monotonic())
                try:
                    batch.append(self._queue.get(timeout=timeout or 0.01))
                    if deadline is None:
                        deadline = time.monotonic() + BATCH_WAIT_SECONDS
                except Empty:
                    break

            if not batch:
                continue

            try:
                for request, completion in zip(batch, self._generate_batch(batch)):
                    request.future.set_result(
                        ModelOutput.from_content(model=self.model_name, content=completion)
                    )
            except Exception as exc:  # noqa: BLE001 - every waiter must be released
                # A failure here would otherwise hang the run: the awaiting
                # coroutines poll a future that nobody ever completes.
                logger.exception("MELT batch of %d failed", len(batch))
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(exc)

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


def _extract(messages: list[ChatMessage]) -> tuple[str, dict | None]:
    """Pull the prompt text and audio locator out of the messages."""
    from inspect_ai.model import ContentData

    texts: list[str] = []
    locator: dict | None = None

    for message in messages:
        content = message.content
        if isinstance(content, str):
            texts.append(content)
            continue
        for item in content:
            if isinstance(item, ContentData) and AUDIO_DATA_KEY in item.data:
                locator = item.data[AUDIO_DATA_KEY]
            elif getattr(item, "type", None) == "text":
                texts.append(item.text)
            elif getattr(item, "type", None) == "audio":
                locator = {"kind": "file", "path": item.audio}

    return "".join(texts), locator


def _load_audio(locator: dict):
    """Resolve an audio locator to ``(samples, sample_rate)``."""
    from melteval.manifest import AudioLocator
    from melteval.readers.base import reader_for_locator

    audio_locator = AudioLocator.from_dict(locator)
    return reader_for_locator(audio_locator).load_audio(audio_locator)


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
