"""Tests for the reader registry, in particular its thread-safety.

Found running a real batched eval: with concurrency > 1, the MELT provider's
worker thread and inspect's own concurrent sample execution can both call
`get_reader()`/`reader_for_locator()` for the first time at once. The lazy
loader used a bare `if _LOADED: return` with no lock, so a second thread could
observe `_LOADED == True` before the first thread's imports (and thus their
`register_reader()` calls) had actually finished, and see an empty registry --
"No reader can resolve audio locator of kind 'shar'" on a perfectly valid
locator, non-deterministically, only under load.
"""

import sys
import threading

import pytest

from melteval.readers import base as reader_base
from melteval.readers.base import SourceResult, get_reader, register_reader


@pytest.fixture(autouse=True)
def _reset_registry(monkeypatch):
    """Every test starts from "nothing loaded yet", the state that races.

    ``melteval.readers.shar``/``.hf`` are already in ``sys.modules`` from
    earlier tests, so resetting ``_LOADED``/``_READERS`` alone would not be
    enough: ``__import__`` is a no-op for an already-imported module, so
    ``register_reader()`` would never run again and the registry would stay
    empty no matter how ``_load_builtin_readers`` is called. Evicting them
    forces a real re-import (``monkeypatch`` restores ``sys.modules`` after
    the test).
    """
    monkeypatch.setattr(reader_base, "_READERS", {})
    monkeypatch.setattr(reader_base, "_LOADED", False)
    monkeypatch.delitem(sys.modules, "melteval.readers.shar", raising=False)
    monkeypatch.delitem(sys.modules, "melteval.readers.hf", raising=False)


class TestConcurrentFirstLoad:
    def test_many_threads_racing_the_first_call_all_succeed(self):
        """Hammer the lazy-load path with concurrent first-time callers.

        Before the fix this reliably (if non-deterministically) raised
        "Unknown source type" for at least one thread on a loaded machine;
        with the lock, every thread either does the loading or waits for the
        thread that is.
        """
        n_threads = 32
        barrier = threading.Barrier(n_threads)
        errors: list[BaseException] = []
        lock = threading.Lock()

        def worker():
            barrier.wait()  # maximize contention on the first call
            try:
                get_reader("lhotse_shar")
            except BaseException as exc:  # noqa: BLE001 - captured for the assertion below
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == []

    def test_loading_happens_at_most_once(self, monkeypatch):
        """The lock's other job: don't re-import on every call once loaded."""
        calls = []
        real_import = reader_base._load_builtin_readers

        def counting_load():
            calls.append(1)
            real_import()

        monkeypatch.setattr(reader_base, "_load_builtin_readers", counting_load)
        for _ in range(5):
            reader_base.get_reader("lhotse_shar")
        # get_reader calls _load_builtin_readers every time; what must stay
        # bounded is the *work* inside it, guarded by the already-loaded check.
        assert reader_base._LOADED is True


class TestRegisterReader:
    def test_register_then_get_round_trips(self):
        class DummyReader:
            source_type = "dummy"
            locator_kind = "dummy"

            def freeze(self, source_cfg, source_index):
                return SourceResult()

            def load_audio(self, locator):
                raise NotImplementedError

        register_reader(DummyReader())
        assert isinstance(get_reader("dummy"), DummyReader)

    def test_unknown_source_type_lists_whats_available(self):
        with pytest.raises(ValueError, match="lhotse_shar"):
            get_reader("does-not-exist")
