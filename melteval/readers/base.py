"""Source readers: the only part of the harness that knows about corpus formats.

A reader does two things, at two different times:

* at **freeze** time it turns one source config entry into
  :class:`~melteval.manifest.EvalRecord` objects, resolving where the reference
  text lives and assigning an :class:`~melteval.manifest.AudioLocator`;
* at **generation** time it turns a locator back into audio samples.

Splitting it this way is what keeps frozen sets cheap: the freeze pass touches
manifests only, and audio is decoded one batch at a time inside the provider.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers only
    import numpy as np

    from melteval.manifest import AudioLocator, EvalRecord


@dataclass
class SourceResult:
    """What one source contributed to a frozen set.

    The counts are not decoration: a source that reads 5000 cuts and keeps 0
    is the failure mode this harness exists to make visible, and it looks
    identical to success unless the drops are reported.

    Attributes:
        records: Records to include in the manifest.
        stats: Read/kept/dropped counts and total audio hours.
    """

    records: list[EvalRecord] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class Reader(Protocol):
    """The contract every source type implements."""

    #: Value of the source config's ``type`` key that selects this reader.
    source_type: str
    #: Value of ``AudioLocator.kind`` this reader knows how to resolve.
    locator_kind: str

    def freeze(self, source_cfg: dict, source_index: int) -> SourceResult:
        """Read one source into records, with counts of what was dropped.

        Args:
            source_cfg: One entry of the frozen-set spec's ``input_cfg``.
            source_index: Position of this source in the spec, used to build
                globally unique sample keys.

        Returns:
            Records with a resolved target and audio locator, alongside stats.
            Samples with no usable reference are dropped and counted rather
            than emitted with an empty target.
        """
        ...

    def load_audio(self, locator: AudioLocator) -> tuple[np.ndarray, int]:
        """Resolve *locator* to ``(samples, sample_rate)``.

        Returns:
            A 1-D float32 mono array and its sample rate.
        """
        ...


_READERS: dict[str, Reader] = {}
_LOADED = False
#: Guards _LOADED / _READERS against the provider's batching worker thread and
#: inspect's own concurrent sample execution both calling get_reader() /
#: reader_for_locator() for the first time at once. Without it, a second
#: thread can observe _LOADED == True before the first thread's imports (and
#: thus their register_reader() calls) have actually finished, and finds an
#: empty registry -- "No reader can resolve audio locator of kind 'shar'" on a
#: perfectly correct locator, and only under concurrency.
_LOAD_LOCK = threading.Lock()


def register_reader(reader: Reader) -> Reader:
    """Register *reader* under its ``source_type``."""
    _READERS[reader.source_type] = reader
    return reader


def get_reader(source_type: str) -> Reader:
    """Look up the reader for *source_type*.

    Raises:
        ValueError: If no reader is registered, listing what is available so a
            typo in a spec is obvious.
    """
    _load_builtin_readers()
    if source_type not in _READERS:
        raise ValueError(
            f"Unknown source type {source_type!r}. Available: {sorted(_READERS) or '(none)'}"
        )
    return _READERS[source_type]


def reader_for_locator(locator: AudioLocator) -> Reader:
    """Look up the reader that can resolve *locator*."""
    _load_builtin_readers()
    for reader in _READERS.values():
        if getattr(reader, "locator_kind", None) == locator.kind:
            return reader
    raise ValueError(f"No reader can resolve audio locator of kind {locator.kind!r}")


def _load_builtin_readers() -> None:
    """Import the shipped readers, tolerating absent optional dependencies.

    ``lhotse`` and ``datasets`` are extras, so a harness installed for one
    source type must not fail to start because the other is missing.

    Double-checked locking: the fast path (already loaded) stays lock-free,
    since this runs on every ``get_reader``/``reader_for_locator`` call and
    those happen once per generated sample.
    """
    global _LOADED
    if _LOADED:
        return
    with _LOAD_LOCK:
        if _LOADED:  # another thread finished loading while this one waited
            return
        for module in ("melteval.readers.shar", "melteval.readers.hf", "melteval.readers.mcif"):
            try:
                __import__(module)
            except ImportError:
                continue
        _LOADED = True
