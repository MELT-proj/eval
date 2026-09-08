"""Turn a source spec into a frozen evaluation set.

A frozen set is a ``manifest.jsonl`` plus a ``frozen_set.json`` summary. It
records *what* is being evaluated and *where the audio is*, not the audio
itself, so two runs of the same spec evaluate the same samples and the set
costs megabytes.

Spec format::

    name: asr-test-v1
    seed: 0
    input_cfg:
      - type: lhotse_shar
        shar_path: ${LOCAL_DATASETS_DIR}/fleurs/de_de/test
        text_field: custom.pnc_text
        tags: {task: asr, lang: de, dataset_id: fleurs}

``input_cfg`` deliberately mirrors the training config's ``validation_ds`` so an
eval set can be written by copying the block that trained the checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from melteval.manifest import (
    MANIFEST_NAME,
    SUMMARY_NAME,
    EvalRecord,
    write_manifest,
)
from melteval.readers.base import get_reader


logger = logging.getLogger(__name__)


def load_spec(path: str | Path) -> dict:
    """Load a frozen-set spec, expanding environment variables in it.

    ``${LOCAL_DATASETS_DIR}`` and ``${oc.env:LOCAL_DATASETS_DIR}`` are both
    accepted, the latter because specs get copied out of training configs.

    Args:
        path: Path to the spec YAML.

    Returns:
        The spec as a plain dict.
    """
    import yaml

    text = Path(path).read_text(encoding="utf-8")
    text = text.replace("${oc.env:", "${")
    return yaml.safe_load(os.path.expandvars(text))


def spec_hash(spec: dict) -> str:
    """Return a short, stable hash of *spec* for run comparability."""
    payload = json.dumps(spec, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def read_sources(
    spec: dict, *, include: Callable[[dict], bool] | None = None
) -> tuple[list[EvalRecord], list[dict]]:
    """Run every source in *spec* through its reader.

    Shared by :func:`freeze` and by the live path in
    :func:`melteval.dataset.spec_dataset`, so a spec read without being frozen
    produces byte-identical records to the same spec frozen first.

    Args:
        spec: Parsed spec with an ``input_cfg`` list.
        include: Optional predicate on a source config. Sources it rejects are
            skipped **without renumbering the rest**: ``source_index`` feeds
            ``sample_key``, so a skipped source must not shift the keys of the
            ones after it, or a live run and a frozen run of the same spec
            would disagree on which sample is which.

    Returns:
        The records, and one stats dict per source that was read.

    Raises:
        ValueError: If the spec has no sources, or if no source that was read
            yielded a single usable sample.
    """
    sources = spec.get("input_cfg") or []
    if not sources:
        raise ValueError("Spec has no `input_cfg` sources.")

    default_seed = int(spec.get("seed", 0) or 0)

    records: list[EvalRecord] = []
    per_source: list[dict] = []

    for source_index, source_cfg in enumerate(sources):
        source_cfg = dict(source_cfg)
        source_cfg.setdefault("seed", default_seed)
        source_type = str(source_cfg.get("type", "lhotse_shar"))
        label = str(source_cfg.get("name") or source_cfg.get("shar_path") or source_type)

        if include is not None and not include(source_cfg):
            logger.info("source %d (%s): skipped, excluded by filter", source_index, label)
            continue

        reader = get_reader(source_type)
        result = reader.freeze(source_cfg, source_index)
        records.extend(result.records)

        per_source.append({"index": source_index, "source": label, "type": source_type, **result.stats})
        logger.info(
            "source %d (%s): kept %s of %s cuts, %.2f h",
            source_index, label, result.stats.get("kept"), result.stats.get("read"),
            result.stats.get("hours", 0.0),
        )
        if not result.records:
            logger.warning(
                "Source %d (%s) contributed no samples. Check the path and text_field "
                "before trusting any metric computed over this set.",
                source_index, label,
            )

    if not records:
        raise ValueError(
            "No source yielded a usable sample. This is almost always a wrong path or a "
            "text_field that resolves to nothing, not a genuinely empty corpus."
        )

    _assert_unique_keys(records)
    return records, per_source


def freeze(spec: dict, out_dir: str | Path) -> dict:
    """Build a frozen set from *spec* into *out_dir*.

    Args:
        spec: Parsed spec with an ``input_cfg`` list.
        out_dir: Directory to write ``manifest.jsonl`` and ``frozen_set.json``.

    Returns:
        The summary that was written.

    Raises:
        ValueError: If the spec has no sources, or if no source yielded a
            single usable sample.
    """
    records, per_source = read_sources(spec)
    default_seed = int(spec.get("seed", 0) or 0)

    out_dir = Path(out_dir)
    write_manifest(out_dir / MANIFEST_NAME, records)

    summary = {
        "name": spec.get("name", out_dir.name),
        "spec_hash": spec_hash(spec),
        "spec": spec,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": default_seed,
        "total_samples": len(records),
        "total_hours": round(sum(r.duration or 0.0 for r in records) / 3600.0, 4),
        "tasks": _count_by(records, "task"),
        "languages": _count_by(records, "lang"),
        "sources": per_source,
        "versions": _versions(),
    }
    (out_dir / SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Froze %d samples (%.2f h) into %s", len(records), summary["total_hours"], out_dir
    )
    return summary


def _assert_unique_keys(records: list[EvalRecord]) -> None:
    """Fail if two records share a sample key.

    Sample keys are assigned rather than taken from the corpus precisely so this
    cannot happen, so a collision means the assignment scheme broke — worth an
    exception rather than a silently merged eval log.
    """
    seen: set[str] = set()
    for record in records:
        if record.sample_key in seen:
            raise ValueError(f"Duplicate sample_key {record.sample_key!r} in frozen set.")
        seen.add(record.sample_key)


def _count_by(records: list[EvalRecord], attr: str) -> dict[str, int]:
    """Count records grouped by an attribute, for the summary."""
    counts: dict[str, int] = {}
    for record in records:
        key = str(getattr(record, attr) or "")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _versions() -> dict[str, str]:
    """Record reader versions, since manifests outlive the code that wrote them."""
    versions: dict[str, str] = {}
    for name in ("lhotse", "inspect_ai", "datasets"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            continue
    return versions
