"""Reader for RedVox (https://huggingface.co/datasets/FBK-MT/RedVox).

RedVox ships a gated HuggingFace repo with two pieces per language:

* ``{lang}/metadata.jsonl``: the rows, including the harmful ``user_text``,
  the speech ``transcript``, the vulnerability type, and the audio type;
* ``{lang}/audio/*.wav``: the audio files, referenced by a repo-relative path
  string, not an embedded ``Audio`` feature.

The dataset has two input modalities. For ``speech`` rows the audio contains
the harmful/stereotypical statement and ``user_text`` is the question about
it (for example, "Is that true?"). For ``silence``/``noise_a``/``noise_b``
rows the request is in ``user_text`` and the audio is background sound.

Audio is referenced lazily: a ``redvox`` locator carries the repo, revision,
and ``{lang}/{audio}`` path, and :meth:`RedVoxReader.load_audio` downloads the
WAV on demand. Nothing here copies audio into a frozen set or into git.

The official judge query is language- and modality-dependent: for speech it
is ``transcript + " " + user_text``; for text it is ``user_text``. That string
is stored as ``extra["text_prompt"]`` for the scorer.
"""

from __future__ import annotations

import json
import math

from huggingface_hub import hf_hub_download

from melteval.manifest import AudioLocator, EvalRecord
from melteval.readers.base import SourceResult, register_reader


TARGET_SAMPLE_RATE = 16000
AUDIO_TYPES = ("speech", "silence", "noise_a", "noise_b")
LOCATOR_KIND = "redvox"

#: Parsed metadata per (repo, language, revision). A spec has one source per
#: audio type, and each source filters the same language file, so a process
#: that reads several sources should not parse the JSONL several times.
_METADATA_CACHE: dict[tuple[str, str, str], list[dict]] = {}


def _get(cfg, key, default=None):
    """Read *key* from a dict or an OmegaConf node."""
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _clean(value):
    """Turn JSON ``NaN``/missing into ``None``; leave strings/numbers alone."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _rows(repo: str, name: str, revision: str) -> list[dict]:
    """Load and cache one language's metadata JSONL."""
    key = (repo, name, revision)
    cached = _METADATA_CACHE.get(key)
    if cached is not None:
        return cached
    path = hf_hub_download(repo, f"{name}/metadata.jsonl", repo_type="dataset", revision=revision)
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    _METADATA_CACHE[key] = rows
    return rows


def _read_audio(path: str):
    """Read one WAV, split out so tests do not need ``soundfile``."""
    import soundfile as sf

    return sf.read(path, dtype="float32", always_2d=False)


def _resample(data, orig_sr: int, target_sr: int):
    """Resample with librosa, split out so tests do not need its numba import."""
    import librosa

    return librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)


class RedVoxReader:
    """Read one RedVox language/audio-type source into records."""

    source_type = "redvox"
    locator_kind = LOCATOR_KIND

    def freeze(self, source_cfg: dict, source_index: int) -> SourceResult:
        """Read one source into records.

        Args:
            source_cfg: Source entry with ``repo``, ``name`` (language) and
                ``revision`` required, plus ``audio_type`` to filter the
                language split and optional ``tags``/``split``.
            source_index: Position of this source in the spec, used for sample
                keys.

        Returns:
            Records whose ``instruction`` is ``user_text``, whose
            ``extra["text_prompt"]`` is the official judge query, and whose
            audio is a lazy ``redvox`` locator.
        """
        repo = _get(source_cfg, "repo")
        name = _get(source_cfg, "name")
        revision = _get(source_cfg, "revision")
        audio_type = _get(source_cfg, "audio_type")
        if not repo:
            raise ValueError(f"Source #{source_index}: redvox requires `repo`.")
        if not name:
            raise ValueError(f"Source #{source_index}: redvox requires `name` (the language config).")
        if not revision:
            raise ValueError(
                f"Source #{source_index} ({repo}): `revision` is required. Without it the "
                "gated dataset can change under an eval."
            )
        audio_type = str(audio_type or "")
        if audio_type not in AUDIO_TYPES:
            raise ValueError(
                f"Source #{source_index} ({repo}/{name}): audio_type must be one of "
                f"{AUDIO_TYPES}, got {audio_type!r}."
            )

        split = str(_get(source_cfg, "split", "test"))
        tags = dict(_get(source_cfg, "tags", {}) or {})
        source_label = f"redvox@{split}/{name}/{audio_type}"

        candidates: list[EvalRecord] = []
        n_read = 0
        n_dropped_audio_type = 0
        n_dropped_blank = 0

        for index, row in enumerate(_rows(str(repo), str(name), str(revision))):
            n_read += 1
            if str(row.get("audio_type") or "") != audio_type:
                n_dropped_audio_type += 1
                continue

            user_text = str(row.get("user_text") or "").strip()
            if not user_text:
                n_dropped_blank += 1
                continue

            transcript = _clean(row.get("transcript"))
            transcript = str(transcript).strip() if transcript is not None else None
            modality = str(
                row.get("modality") or ("speech" if audio_type == "speech" else "text")
            ).strip()
            judge_query = (
                f"{transcript} {user_text}".strip()
                if modality == "speech" and transcript
                else user_text
            )

            audio = row.get("audio")
            if not audio:
                raise ValueError(
                    f"Source #{source_index} ({repo}/{name}): row {row.get('ID')!r} has no `audio` path."
                )

            row_id = row.get("ID")
            candidates.append(
                EvalRecord(
                    sample_key="",  # assigned below
                    task=str(tags.get("task", "redvox")),
                    target="",
                    audio=AudioLocator(
                        kind=LOCATOR_KIND,
                        params={
                            "repo": str(repo),
                            "revision": str(revision),
                            "path": f"{name}/{audio}",
                            "lang": str(name),
                            "audio_type": audio_type,
                            "index": index,
                        },
                    ),
                    lang=str(row.get("lang") or name),
                    dataset_id=str(tags.get("dataset_id") or f"redvox-{name}-{audio_type}"),
                    source=source_label,
                    instruction=user_text,
                    cut_id=f"{name}-{row_id}",
                    duration=None,
                    extra={
                        "text_prompt": judge_query,
                        "vulnerability_type": str(row.get("vulnerability_type") or ""),
                        "audio_type": audio_type,
                        "modality": modality,
                        "transcript": transcript,
                        "user_text": user_text,
                        "original_id": _clean(row.get("original_id")),
                    },
                )
            )

        for ordinal, record in enumerate(candidates):
            record.sample_key = f"{source_index:02d}-{ordinal:06d}"

        return SourceResult(
            records=candidates,
            stats={
                "read": n_read,
                "kept": len(candidates),
                "dropped_audio_type": n_dropped_audio_type,
                "dropped_blank_user_text": n_dropped_blank,
                "hours": 0.0,
            },
        )

    def load_audio(self, locator: AudioLocator):
        """Download, decode, and resample one RedVox WAV to 16 kHz mono."""
        import numpy as np

        params = locator.params
        path = hf_hub_download(
            str(params["repo"]),
            str(params["path"]),
            repo_type="dataset",
            revision=str(params["revision"]),
        )
        data, sample_rate = _read_audio(path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = np.asarray(data, dtype=np.float32)
        if sample_rate != TARGET_SAMPLE_RATE:
            data = _resample(data, sample_rate, TARGET_SAMPLE_RATE)
            sample_rate = TARGET_SAMPLE_RATE
        return data, sample_rate


register_reader(RedVoxReader())
