"""Turn a frozen set into an inspect dataset.

The prompt is deliberately *not* built here. It depends on the checkpoint under
test — on the format keys in its ``training_config.yaml`` — while a frozen set
has to stay checkpoint-independent so the same samples can be put to several
models, including ones from other projects. The solver renders it.
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
    records = read_manifest(manifest)

    if task is not None:
        records = [r for r in records if r.task == task]
    if lang is not None:
        records = [r for r in records if r.lang == lang]
    if dataset_id is not None:
        records = [r for r in records if r.dataset_id == dataset_id]
    if limit is not None:
        records = records[:limit]

    if not records:
        raise ValueError(
            f"No samples in {manifest} match task={task!r} lang={lang!r} "
            f"dataset_id={dataset_id!r}. An eval over zero samples reports a score, "
            "so this is an error rather than an empty run."
        )

    return MemoryDataset(
        samples=[record_to_sample(r) for r in records],
        name=Path(manifest).parent.name,
        location=str(manifest),
    )


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
        },
    )
