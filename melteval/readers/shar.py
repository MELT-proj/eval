"""Reader for local lhotse Shar corpora.

Freezing reads **cut manifests only**. Passing ``fields={"cuts": [...]}`` to
``CutSet.from_shar`` leaves each recording as a placeholder instead of inlining
its encoded bytes, which is the difference between 0.09 s and flat memory versus
pulling a whole test split into RAM. (``materialize_cuts_for_eval`` in the
training repo does the latter: its docstring says no audio is loaded, but under
lhotse 2.x the default read is eager. It is reused here for neither reason.)

Audio is addressed by **global cut index**, resolved through the indexed
reader's O(1) ``__getitem__``. Manifest iteration order and indexed order agree,
which the tests pin down — if they ever diverged, every sample would be scored
against the wrong audio silently.
"""

from __future__ import annotations

import os
import random
import threading
from pathlib import Path

from melteval.manifest import AudioLocator, EvalRecord
from melteval.readers.base import SourceResult, register_reader


def _require_melt():
    """Import the training package's lhotse helpers, or explain the extra.

    The reference-resolution and tag conventions live in the training repo and
    are imported rather than reimplemented: a second copy of the ``text_field``
    precedence rules would be a second thing to get wrong.
    """
    try:
        from melt.training.data.audio.lhotse.dataloader import shar_manifest_files
        from melt.training.data.audio.lhotse.helpers import (
            get_tags_from_cut,
            get_text_from_cut,
        )
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "Reading lhotse Shar sources needs the training package and lhotse: "
            "install melt-eval[shar]."
        ) from exc
    return shar_manifest_files, get_tags_from_cut, get_text_from_cut


def _get(cfg, key: str, default=None):
    """Read *key* from a dict or an OmegaConf node."""
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _nested_value(obj, path: str, default=None):
    """Resolve a dotted path over attributes and dict keys alike."""
    current = obj
    for part in path.split("."):
        if current is None:
            return default
        if hasattr(current, part):
            current = getattr(current, part)
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current if current is not None else default


class SharReader:
    """Reads lhotse Shar directories, one per source config entry."""

    source_type = "lhotse_shar"
    locator_kind = "shar"

    def freeze(self, source_cfg: dict, source_index: int) -> SourceResult:
        """Read a shar directory into records, counting what was dropped.

        Args:
            source_cfg: Source entry with ``shar_path`` and optional ``tags``,
                ``text_field``, ``source_text_field``, ``min_duration``,
                ``max_duration``, ``max_samples`` and ``seed``.
            source_index: Position in the spec, used for sample keys.

        Raises:
            FileNotFoundError: If ``shar_path`` does not exist.
            ValueError: If the entry uses the nested group form, or the
                directory holds no cut manifests.
        """
        shar_manifest_files, get_tags_from_cut, get_text_from_cut = _require_melt()
        from lhotse import CutSet

        _reject_group_form(source_cfg, source_index)

        shar_path = Path(os.path.expandvars(str(_get(source_cfg, "shar_path"))))
        if not shar_path.exists():
            raise FileNotFoundError(f"Shar path not found: {shar_path}")

        manifests = shar_manifest_files(shar_path)
        if not manifests:
            raise ValueError(
                f"No cut manifests (cuts.*.jsonl[.gz]) in {shar_path}. "
                "An empty result here usually means the path is wrong, not that the source is empty."
            )

        tags = dict(_get(source_cfg, "tags", {}) or {})
        # The leaf directory is almost always "test", so a bare name would label
        # every source identically and collapse the per-source breakdown.
        source_label = str(_get(source_cfg, "name", "") or "/".join(shar_path.parts[-3:]))
        text_field = str(_get(source_cfg, "text_field", "text"))
        source_text_field = _get(source_cfg, "source_text_field")
        min_duration = _get(source_cfg, "min_duration")
        max_duration = _get(source_cfg, "max_duration")
        max_samples = _get(source_cfg, "max_samples")
        seed = int(_get(source_cfg, "seed", 0) or 0)
        indexes_root = _get(source_cfg, "indexes_root")

        cuts = CutSet.from_shar(
            fields={"cuts": [str(p) for p in manifests]},
            shuffle_shards=False,
            split_for_dataloading=False,
        )

        candidates: list[EvalRecord] = []
        n_read = 0
        n_dropped_duration = 0
        n_dropped_no_reference = 0
        for index, cut in enumerate(cuts):
            n_read += 1
            if min_duration is not None and cut.duration < float(min_duration):
                n_dropped_duration += 1
                continue
            if max_duration is not None and cut.duration > float(max_duration):
                n_dropped_duration += 1
                continue

            # Tags must be attached before reading them back, because
            # get_tags_from_cut is the single source of truth for the
            # task/language conventions -- notably that `lang` is the *target*
            # language for st.
            if tags:
                cut.custom = {**(cut.custom or {}), **tags}
                cut.tags = tags

            target = get_text_from_cut(cut, effective_text_field(cut, text_field))
            if not target:
                n_dropped_no_reference += 1
                continue

            task, lang, src_lang, tgt_lang = get_tags_from_cut(cut)
            source_text = (
                _nested_value(cut, str(source_text_field)) if source_text_field else None
            )

            candidates.append(
                EvalRecord(
                    sample_key="",  # assigned below, after subsetting
                    task=task,
                    target=target,
                    audio=AudioLocator(
                        kind=self.locator_kind,
                        params={
                            "dir": str(shar_path),
                            "index": index,
                            **({"indexes_root": str(indexes_root)} if indexes_root else {}),
                        },
                    ),
                    lang=lang,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    dataset_id=str(tags.get("dataset_id", "") or ""),
                    source=source_label,
                    source_text=str(source_text) if source_text else None,
                    cut_id=str(cut.id),
                    duration=float(cut.duration),
                )
            )

        if max_samples is not None and len(candidates) > int(max_samples):
            # Seeded shuffle, mirroring materialize_cuts_for_eval, so the same
            # spec selects the same subset on any machine. Re-sorted afterwards
            # so reads stay in shard order at generation time.
            random.Random(seed).shuffle(candidates)
            candidates = candidates[: int(max_samples)]
            candidates.sort(key=lambda r: r.audio.params["index"])

        for ordinal, record in enumerate(candidates):
            record.sample_key = f"{source_index:02d}-{ordinal:06d}"

        return SourceResult(
            records=candidates,
            stats={
                "read": n_read,
                "kept": len(candidates),
                "dropped_duration": n_dropped_duration,
                "dropped_no_reference": n_dropped_no_reference,
                "hours": round(sum(r.duration or 0.0 for r in candidates) / 3600.0, 4),
            },
        )

    def load_audio(self, locator: AudioLocator):
        """Resolve a shar locator to ``(samples, sample_rate)``."""
        import numpy as np

        cut = self._cut_at(str(locator.params["dir"]), int(locator.params["index"]),
                           locator.params.get("indexes_root"))
        audio = cut.load_audio()
        if hasattr(audio, "numpy"):
            audio = audio.numpy()
        if audio.ndim > 1:
            audio = audio[0]
        return audio.astype(np.float32), int(cut.sampling_rate)

    def _cut_at(self, shar_dir: str, index: int, indexes_root=None):
        """Return the cut at global *index*, reusing one open reader per dir.

        The reader's ``__getitem__`` does a bare ``seek`` + ``read`` against one
        shared file handle with no locking of its own (lhotse's indexed shar
        readers assume single-threaded access -- see the comment on
        ``IndexedTarReader._fh`` about reopening per *process*, which says
        nothing about threads). melt-eval's own provider calls this from
        several threads at once, so the per-directory lock below is load
        bearing: without it, concurrent seeks interleave and a read lands at
        the wrong offset, which decodes as tar-header garbage
        (`InvalidHeaderError: bad checksum`, MELT-proj/eval#2).
        """
        reader, lock = _indexed_reader(shar_dir, str(indexes_root) if indexes_root else None)
        with lock:
            return reader[index]


def effective_text_field(cut, source_default: str) -> str:
    """Return the field holding *cut*'s reference text.

    Precedence is the same one ``MELTMapDataset`` applies during training:
    a per-cut ``tags.text_field`` override wins over the source-level setting,
    which wins over plain supervision text. Ignoring the per-cut override is
    what makes the older inference script score CommonVoice against
    ``supervisions[0].text`` when the real reference is
    ``custom.metadata.sentence``.

    Args:
        cut: A lhotse cut, with tags already attached.
        source_default: ``text_field`` from the source config.

    Returns:
        A dotted field path suitable for ``get_text_from_cut``.
    """
    tags = getattr(cut, "tags", None)
    if isinstance(tags, dict) and tags.get("text_field"):
        return str(tags["text_field"])
    return source_default


def _reject_group_form(source_cfg, source_index: int) -> None:
    """Reject the nested ``input_cfg`` group form.

    Group ``tags:`` replace ``cut.tags`` wholesale and are applied after the
    children's, so a group carrying ``task: st`` silently clobbers a leaf's
    ``src_lang``/``tgt_lang``. Only the flat form is supported here, and saying
    so loudly beats producing quietly mislabelled records.
    """
    if _get(source_cfg, "input_cfg") is not None:
        raise ValueError(
            f"Source #{source_index} uses the nested group form (`input_cfg` inside a source). "
            "Group tags replace cut tags wholesale and are applied after the children's, "
            "clobbering src_lang/tgt_lang. Flatten the spec into one entry per shar_path."
        )


_READER_CACHE: dict[tuple[str, str | None], object] = {}
_READER_LOCKS: dict[tuple[str, str | None], threading.Lock] = {}
# Guards creation of cache/lock entries above, not reads through them -- held
# only for the brief get-or-create below, never across a `reader[index]` call.
_CACHE_LOCK = threading.Lock()


def _indexed_reader(shar_dir: str, indexes_root: str | None) -> tuple[object, threading.Lock]:
    """Open (once per directory) an indexed reader with O(1) random access.

    Returns the reader alongside a per-directory lock. The reader's
    ``__getitem__`` is not thread-safe (one shared, unsynchronized file handle
    per shard -- see ``_cut_at``), so every caller must hold the lock for the
    duration of a lookup; it is not enforced here because it must wrap the
    lookup itself, not construction.

    Raises:
        RuntimeError: If the directory has no ``.idx`` sidecars, since without
            them every lookup would rescan the shard.
    """
    key = (shar_dir, indexes_root)
    with _CACHE_LOCK:
        cached = _READER_CACHE.get(key)
        if cached is not None:
            return cached, _READER_LOCKS[key]

        from lhotse import CutSet

        kwargs = {"in_dir": shar_dir, "shuffle_shards": False, "split_for_dataloading": False}
        if indexes_root:
            kwargs["indexes_root"] = indexes_root
        reader = CutSet.from_shar(**kwargs).data

        # Exposed as a property on some lhotse builds and a method on others.
        access = getattr(reader, "has_constant_time_access", False)
        if not (access() if callable(access) else access):
            raise RuntimeError(
                f"{shar_dir} has no .idx sidecars, so audio lookup would rescan a shard per sample. "
                "Point at the indexed copy of the tree, or pass `indexes_root` for a plain tree "
                "whose sidecars live elsewhere."
            )
        _READER_CACHE[key] = reader
        _READER_LOCKS[key] = threading.Lock()
        return reader, _READER_LOCKS[key]


register_reader(SharReader())
