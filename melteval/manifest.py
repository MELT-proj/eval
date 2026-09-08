"""The frozen-set manifest: one JSON record per evaluation sample.

A frozen set is the boundary between *reading corpora* and *running an eval*.
Everything upstream of it (lhotse, HuggingFace ``datasets``, per-corpus
conventions about where the reference text lives) is resolved once, at freeze
time; everything downstream reads only this schema.

Audio is referenced, not copied. A record carries an :class:`AudioLocator`
pointing back at the source, so a frozen set costs megabytes rather than the
tens of GB a materialised copy would (see ``docs/frozen-sets.md``). The
trade-off is that a manifest is only valid while its sources are intact, which
is why ``frozen_set.json`` records the source paths and a spec hash.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path


MANIFEST_NAME = "manifest.jsonl"
SUMMARY_NAME = "frozen_set.json"


@dataclass(frozen=True)
class AudioLocator:
    """Where a sample's audio lives, resolved lazily at generation time.

    Attributes:
        kind: Reader that understands this locator (``"shar"``, ``"hf"``,
            ``"file"``).
        params: Reader-specific addressing. A ``shar`` locator carries
            ``dir`` and the cut's global ``index``; a ``file`` locator carries
            ``path``. Kept open-ended so a new reader does not force a schema
            migration.
    """

    kind: str
    params: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialise to the form stored in the manifest."""
        return {"kind": self.kind, **self.params}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> AudioLocator:
        """Rebuild from a manifest record."""
        params = {k: v for k, v in data.items() if k != "kind"}
        return cls(kind=str(data["kind"]), params=params)


@dataclass
class EvalRecord:
    """One evaluation sample, independent of any model or framework.

    ``target`` is the thing being scored against, whatever the task: a
    transcript for ASR, a translation for ST, an answer (or list of acceptable
    answers) for question answering.

    Attributes:
        sample_key: Unique, stable identity for the sample. Assigned by the
            freezer as ``<source_index>-<ordinal>`` rather than taken from the
            corpus, because FLEURS cut IDs are not unique within a language or
            across languages (MELT-proj/training#54) and would silently merge
            samples in an eval log.
        task: Task identifier (``asr``, ``st``, ``speechqe``, or any benchmark's
            own label — the schema does not restrict it).
        target: Reference output, or a list of acceptable references.
        audio: Where to find the audio.
        lang: Language of the *expected output*. Follows the training repo's
            convention, where for ``st`` this is the target language.
        src_lang: Source language, when the task distinguishes one.
        tgt_lang: Target language, when the task distinguishes one.
        dataset_id: Corpus identifier, for per-corpus breakdowns.
        source: Human-readable source label (usually the shar subpath).
        source_text: Transcript of the audio when the corpus carries one
            alongside a translation. Reference-based COMET needs it; ST samples
            without it cannot be COMET-scored.
        instruction: Per-sample prompt, for benchmarks that ship their own
            rather than drawing from the training prompt pool.
        choices: Answer options for multiple-choice tasks.
        cut_id: The corpus's own identifier, kept for provenance only.
        duration: Audio duration in seconds. The only reliable budgeting unit —
            ``custom.num_tokens`` is absent from most sources
            (MELT-proj/training#59).
        extra: Reader-specific metadata that does not fit the fields above,
            carried through to the sample unchanged (see
            ``dataset.record_to_sample``). Exists so a reader whose corpus
            needs something no other reader does -- MCIF's chunk-to-reference
            grouping, for one -- does not force a schema migration on every
            other reader.
    """

    sample_key: str
    task: str
    target: str | list[str]
    audio: AudioLocator
    lang: str = ""
    src_lang: str = ""
    tgt_lang: str = ""
    dataset_id: str = ""
    source: str = ""
    source_text: str | None = None
    instruction: str | None = None
    choices: list[str] | None = None
    cut_id: str | None = None
    duration: float | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialise to a single manifest line."""
        data = asdict(self)
        data["audio"] = self.audio.to_dict()
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> EvalRecord:
        """Rebuild from a parsed manifest line."""
        data = dict(data)
        data["audio"] = AudioLocator.from_dict(data["audio"])
        return cls(**data)


def write_manifest(path: Path, records: list[EvalRecord]) -> None:
    """Write *records* to *path* as JSON lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(record.to_json() + "\n" for record in records)


def read_manifest(path: Path) -> list[EvalRecord]:
    """Read a manifest written by :func:`write_manifest`."""
    return list(iter_manifest(path))


def iter_manifest(path: Path) -> Iterator[EvalRecord]:
    """Stream records from a manifest file, skipping blank lines."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield EvalRecord.from_dict(json.loads(line))


def resolve_frozen_set(path: str | Path) -> Path:
    """Return the manifest file for *path*, which may be a set dir or the file.

    Args:
        path: A frozen-set directory, or the manifest inside one.

    Raises:
        FileNotFoundError: If no manifest is found.
    """
    path = Path(path)
    manifest = path / MANIFEST_NAME if path.is_dir() else path
    if not manifest.exists():
        raise FileNotFoundError(f"No frozen-set manifest at {manifest}")
    return manifest
