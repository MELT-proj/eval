"""Turn a frozen set -- or a spec read live -- into an inspect dataset.

The prompt is deliberately *not* built here. It depends on the checkpoint under
test — on the format keys in its ``training_config.yaml`` — while a frozen set
has to stay checkpoint-independent so the same samples can be put to several
models, including ones from other projects. The solver renders it.

Two ways in:

* :func:`frozen_dataset` reads a manifest that ``melteval freeze`` already
  wrote. This is the one to use for a corpus assembled out of local Shar
  trees, where "which samples?" is a question only the freeze pass can pin
  down and the answer has to survive being asked again months later.
* :func:`spec_dataset` runs the same readers at eval time and never writes a
  manifest. A published HuggingFace benchmark pinned to a revision is already
  immutable and already has a canonical sample set, so freezing it buys a copy
  of information the revision hash carries anyway. Sample keys come out
  identical either way (see :func:`melteval.freeze.read_sources`), so the same
  spec can be frozen later without invalidating a run made this way.
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai.dataset import MemoryDataset, Sample

from melteval.manifest import EvalRecord, read_manifest, resolve_frozen_set


#: Key under which the audio locator travels inside the user message.
#: A ``ModelAPI`` is handed messages only -- never the sample's metadata -- so
#: anything the provider needs at generation time has to be in the content.
AUDIO_DATA_KEY = "melteval_audio"


def frozen_dataset(
    path: str | Path,
    *,
    task: str | None = None,
    lang: str | None = None,
    dataset_id: str | None = None,
    limit: int | None = None,
) -> MemoryDataset:
    """Load a frozen set, optionally filtered.

    Args:
        path: Frozen-set directory, or the manifest inside it.
        task: Keep only samples with this task.
        lang: Keep only samples with this output language.
        dataset_id: Keep only samples from this corpus.
        limit: Keep at most this many samples, taken in manifest order so the
            subset stays deterministic.

    Returns:
        A dataset of samples carrying their audio locator and metadata.

    Raises:
        ValueError: If the filters select nothing, which is otherwise
            indistinguishable from a successful run over an empty set.
    """
    manifest = resolve_frozen_set(path)
    records = _filtered(
        read_manifest(manifest),
        task=task,
        lang=lang,
        dataset_id=dataset_id,
        limit=limit,
        origin=str(manifest),
    )

    return MemoryDataset(
        samples=[record_to_sample(r) for r in records],
        name=Path(manifest).parent.name,
        location=str(manifest),
    )


def spec_dataset(
    path: str | Path,
    *,
    task: str | None = None,
    lang: str | None = None,
    dataset_id: str | None = None,
    limit: int | None = None,
) -> MemoryDataset:
    """Read a source spec at eval time, with no frozen set in between.

    The filters do double duty here. They select records, as they do for a
    frozen set — but they are also applied to each **source config** first, so
    a source whose tags already rule it out is never read at all. That is not
    an optimisation detail for a spec covering a benchmark like AIR-Bench,
    where reading a source means having its split resolved and materialised:
    without it, scoring 50 samples of one split would pull all seven.

    Args:
        path: Path to a source spec YAML — the same format
            :func:`melteval.freeze.freeze` takes.
        task: Keep only samples with this task.
        lang: Keep only samples with this output language.
        dataset_id: Keep only samples from this corpus.
        limit: Keep at most this many samples, in spec order.

    Returns:
        A dataset of samples carrying their audio locator and metadata.

    Raises:
        ValueError: If the filters select nothing.
    """
    from melteval.freeze import load_spec, read_sources

    def include(source_cfg: dict) -> bool:
        return _source_may_match(source_cfg, task=task, lang=lang, dataset_id=dataset_id)

    spec = load_spec(path)
    if not any(include(cfg) for cfg in (spec.get("input_cfg") or [])):
        # read_sources() would otherwise report "no source yielded a usable
        # sample", which reads as a broken corpus rather than a filter that
        # matched nothing -- usually a typo in `-T dataset_id=`.
        _raise_no_matches(str(path), task=task, lang=lang, dataset_id=dataset_id)

    records, _ = read_sources(spec, include=include)
    records = _filtered(
        records, task=task, lang=lang, dataset_id=dataset_id, limit=limit, origin=str(path)
    )

    return MemoryDataset(
        samples=[record_to_sample(r) for r in records],
        name=str(spec.get("name") or Path(path).stem),
        location=str(path),
    )


def _filtered(
    records: list[EvalRecord],
    *,
    task: str | None,
    lang: str | None,
    dataset_id: str | None,
    limit: int | None,
    origin: str,
) -> list[EvalRecord]:
    """Apply the task/lang/dataset_id/limit filters, refusing an empty result.

    Raises:
        ValueError: If the filters select nothing, which is otherwise
            indistinguishable from a successful run over an empty set.
    """
    if task is not None:
        records = [r for r in records if r.task == task]
    if lang is not None:
        records = [r for r in records if r.lang == lang]
    if dataset_id is not None:
        records = [r for r in records if r.dataset_id == dataset_id]
    if limit is not None:
        records = records[:limit]

    if not records:
        _raise_no_matches(origin, task=task, lang=lang, dataset_id=dataset_id)
    return records


def _raise_no_matches(
    origin: str, *, task: str | None, lang: str | None, dataset_id: str | None
) -> None:
    """Refuse to run over an empty selection.

    Raises:
        ValueError: Always. An eval over zero samples still reports a score,
            so an empty selection has to stop the run rather than produce one.
    """
    raise ValueError(
        f"No samples in {origin} match task={task!r} lang={lang!r} "
        f"dataset_id={dataset_id!r}. An eval over zero samples reports a score, "
        "so this is an error rather than an empty run."
    )


def _source_may_match(
    source_cfg: dict, *, task: str | None, lang: str | None, dataset_id: str | None
) -> bool:
    """Whether a source could contribute a record surviving these filters.

    Decided from the source's declared ``tags`` alone, so it costs nothing.
    A source is skipped only when it **declares** a tag that contradicts a
    filter; an absent tag means "unknown", and unknown always reads. Getting
    this backwards would silently drop samples from a corpus that simply did
    not label itself, which is the failure mode this harness is built to avoid.

    ``lang`` is checked against ``tags.lang`` *or* ``tags.tgt_lang``: readers
    collapse an ST record's ``lang`` to its target language, so a source
    tagged only ``tgt_lang: de`` does match ``lang="de"``.
    """
    tags = dict(source_cfg.get("tags") or {})

    if task is not None and tags.get("task") not in (None, task):
        return False
    if dataset_id is not None and tags.get("dataset_id") not in (None, dataset_id):
        return False
    if lang is not None:
        declared = {tags[key] for key in ("lang", "tgt_lang") if tags.get(key)}
        if declared and lang not in declared:
            return False
    return True


def record_to_sample(record: EvalRecord) -> Sample:
    """Convert one frozen record to an inspect sample."""
    return Sample(
        # Placeholder: the solver replaces the message list with the rendered
        # prompt. Keeping something readable here means the log's input column
        # still says what the sample *is* when a run fails before generation.
        input=record.instruction or f"[{record.task} {record.lang} {record.duration or 0:.1f}s]",
        target=record.target,
        choices=record.choices,
        id=record.sample_key,
        metadata={
            "task": record.task,
            "lang": record.lang,
            "src_lang": record.src_lang,
            "tgt_lang": record.tgt_lang,
            "dataset_id": record.dataset_id,
            "source": record.source,
            "source_text": record.source_text,
            "instruction": record.instruction,
            "cut_id": record.cut_id,
            "duration": record.duration,
            "audio": record.audio.to_dict(),
            **record.extra,
        },
    )
