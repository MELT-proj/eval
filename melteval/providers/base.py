"""What every local speech checkpoint needs, whatever framework trained it.

A provider for a local speech model has two jobs that have nothing to do with
the model: pulling the prompt and the audio locator back out of the messages
inspect hands it, and turning inspect's one-sample-at-a-time ``generate()``
into batches a GPU is not embarrassed by. Batches are pooled several at a
time and sorted by audio length before slicing, so padding is bounded by
similar-length rows rather than whatever arrived in that order
(MELT-proj/eval#7). Both were written for the MELT provider first; a second
family (NeMo/SALM, via ``providers/smurf.py``) needs them identically, so
they live here rather than being copied.

What stays in the subclass is only what is genuinely framework-specific: how a
checkpoint is loaded, and how one padded batch becomes a list of strings.

Nothing heavy is imported at module scope: registering a provider must not cost
a torch import.
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
from inspect_ai.model import ChatMessage, GenerateConfig, ModelAPI, ModelOutput
from inspect_ai.tool import ToolChoice, ToolInfo

from melteval.dataset import AUDIO_DATA_KEY


logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 8
#: How many batches' worth of requests to pool before sorting by audio length.
#: Sorting a single batch cannot help -- it does not change that batch's
#: longest row -- so this has to look past one batch_size at a time
#: (MELT-proj/eval#7).
DEFAULT_BATCH_WINDOW = 4
#: How long the batch thread waits for more work before running a short batch.
BATCH_WAIT_SECONDS = 0.25


@dataclass
class _Request:
    """One sample waiting to be generated."""

    text: str
    audio: Any | None
    sample_rate: int
    config: GenerateConfig
    #: Audio length in seconds, the sort key for length-bucketed batching.
    #: Computed from the already-decoded audio, so it costs nothing extra --
    #: by the time a request exists, `generate()` has already loaded it.
    duration: float = 0.0
    future: Future = field(default_factory=Future)


class BatchedSpeechAPI(ModelAPI):
    """A local speech model that generates in batches.

    Subclasses load their own checkpoint in ``__init__`` and implement
    :meth:`_generate_batch`. Everything else -- message extraction, off-loop
    audio decoding, the batching worker and its failure handling -- is shared.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_window: int = DEFAULT_BATCH_WINDOW,
    ) -> None:
        """Set up the batching machinery.

        Args:
            model_name: Path to the checkpoint, as given after the provider prefix.
            base_url: Unused; a local model has no endpoint.
            api_key: Unused; a local model needs no credential.
            config: Default generation config.
            batch_size: Samples per forward pass.
            batch_window: How many batches' worth of requests to pool and
                sort by audio length before slicing into batches, so a batch's
                padding is bounded by rows of similar length rather than
                whatever arrived in that order (MELT-proj/eval#7). Also sets
                the concurrency inspect is allowed -- the pool can only be
                bigger than one batch if more than one batch's worth of
                requests is in flight at once.
        """
        super().__init__(model_name, base_url, api_key, [], config)
        self.batch_size = int(batch_size)
        self.batch_window = int(batch_window)
        self._queue: Queue[_Request] = Queue()
        self._thread: Thread | None = None
        self._thread_lock = Lock()

    def max_connections(self) -> int:
        """Cap in-flight requests at the pooling window, not one batch.

        Length-sorted batching (MELT-proj/eval#7) needs more than one batch's
        worth of requests queued to have anything to sort -- sorting a single
        batch does not change its longest row. So the concurrency limit has
        to admit the whole pool, not just ``batch_size``.
        """
        return self.batch_size * self.batch_window

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
        duration = len(audio) / sample_rate if audio is not None else 0.0

        request = _Request(
            text=text, audio=audio, sample_rate=sample_rate, config=config, duration=duration
        )
        self._ensure_worker()
        self._queue.put(request)

        while True:
            if request.future.done():
                return request.future.result()
            await anyio.sleep(0.05)

    def _generate_batch(self, batch: list[_Request]) -> list[str]:
        """Run one padded batch through the model.

        Args:
            batch: Requests to generate, all at once.

        Returns:
            One completion per request, in the same order.
        """
        raise NotImplementedError

    def _ensure_worker(self) -> None:
        """Start the batching thread on first use."""
        with self._thread_lock:
            if self._thread is None:
                self._thread = Thread(target=self._process_batches, daemon=True)
                self._thread.start()

    def _process_batches(self) -> None:
        """Pool queued requests, sort by audio length, and run them in batches.

        Waits briefly for a full pool rather than generating immediately, so a
        run is batched even though inspect hands requests over one at a time.
        The processor pads every row in a batch to that batch's longest audio,
        so an arrival-order batch costs its longest utterance times its width.
        Pooling several batches' worth of requests and sorting by duration
        before slicing keeps each resulting batch close to uniform-length
        instead (MELT-proj/eval#7). A short pool at the tail of a run costs a
        fraction of a second.
        """
        pool_size = self.batch_size * self.batch_window
        while True:
            pool: list[_Request] = []
            deadline = None
            while len(pool) < pool_size:
                timeout = (
                    BATCH_WAIT_SECONDS if deadline is None else max(0.0, deadline - time.monotonic())
                )
                try:
                    pool.append(self._queue.get(timeout=timeout or 0.01))
                    if deadline is None:
                        deadline = time.monotonic() + BATCH_WAIT_SECONDS
                except Empty:
                    break

            if not pool:
                continue

            pool.sort(key=lambda r: r.duration)
            for start in range(0, len(pool), self.batch_size):
                batch = pool[start : start + self.batch_size]
                try:
                    completions = self._generate_batch(batch)
                    if len(completions) != len(batch):
                        # zip() would pair them off in order and drop the tail, so
                        # every sample after the first extra completion would be
                        # scored against another sample's audio. The realistic
                        # cause is a generation config asking for more than one
                        # sequence per prompt.
                        raise ValueError(
                            f"{type(self).__name__} returned {len(completions)} completions for a "
                            f"batch of {len(batch)}. Generation must return exactly one completion "
                            "per request, in order."
                        )
                    for request, completion in zip(batch, completions):
                        request.future.set_result(
                            ModelOutput.from_content(model=self.model_name, content=completion)
                        )
                except Exception as exc:  # noqa: BLE001 - every waiter must be released
                    # A failure here would otherwise hang the run: the awaiting
                    # coroutines poll a future that nobody ever completes.
                    logger.exception("%s batch of %d failed", type(self).__name__, len(batch))
                    for request in batch:
                        if not request.future.done():
                            request.future.set_exception(exc)


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
