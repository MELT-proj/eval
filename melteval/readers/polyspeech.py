"""Reader for PolySpeech-100 (https://huggingface.co/datasets/youngseng/PolySpeech-100-v1).

A spoken, Belebele-style multiple-choice reading-comprehension benchmark: a
passage, a question and four options per sample, each of them available both as
audio and as text. The upstream release ships 110 per-language parquet files
that are *file archives* (one row per wav/txt), not a table, so this reader does
not read the upstream repo. It reads a **local reconstruction** -- a directory
with a ``manifest.jsonl`` plus the extracted audio -- built once by a data
engineering pass (see ``docs/polyspeech.md`` for its layout and the quirks that
pass found). That keeps the eval path free of the archive format.

Two variants, selected per source with ``variant``:

* ``pa_qa`` -- the passage **and the question** are audio; the options are text.
  The audio is the passage followed by the spoken question, and the prompt
  carries a fixed scaffold sentence instead of the question, so the question
  text never reaches the model.
* ``pa_qt`` -- the passage is audio; the **question and the options are text**.

The options are always text in the prompt, never audio. Both variants are the
``audio_mcq`` task: the reference is the correct option's text and
:func:`melteval.scorers.mcq_scorer` resolves the model's answer to an option, so
no judge model is involved.

The framework carries one :class:`~melteval.manifest.AudioLocator` per sample,
so the locator here describes a *single waveform assembled at load time*:
the passage's sentence-level segments joined by short silences (and, for
``pa_qa``, the question after a longer one). Source files are mixed 16/24/44.1/48
kHz mono (some are MP3 inside a ``.wav`` container), so every segment is decoded
and resampled to 16 kHz on its own before joining -- a batch that mixed sample
rates would corrupt every sample in it but one (see ``readers/hf.py``).

Samples in a language are keyed by the **directory-derived** ``index``/``qid``
(kept in ``extra``), not by the ``Index``/``QID`` lines inside upstream's
``info.txt``, which are wrong in seven languages. Use them to join questions
across languages; 17 of the 24 languages cover the same 801 questions, the
other seven a different 801 (docs/polyspeech.md).
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from melteval.manifest import AudioLocator, EvalRecord
from melteval.readers.base import SourceResult, register_reader
from melteval.readers.hf import DEFAULT_CHOICE_INSTRUCTION_TEMPLATE, _render_instruction


logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000

MANIFEST_NAME = "manifest.jsonl"

VARIANTS = ("pa_qa", "pa_qt")

#: Pause between a passage's consecutive sentence segments, in seconds.
DEFAULT_SEGMENT_GAP_S = 0.25
#: Pause between the end of the passage and the spoken question (``pa_qa``).
DEFAULT_QUESTION_GAP_S = 1.0

#: Stands in for the question in the ``pa_qa`` prompt. The real question is in
#: the audio only; showing it as well would turn the variant into ``pa_qt``.
PA_QA_QUESTION = (
    "You will hear a passage followed by a spoken question about it. "
    "Choose the correct answer to the question."
)

#: Upstream language code (``<iso639-3>_<script>``) -> the 2-letter code the rest
#: of this repo uses for ``lang``. Covers the official EU languages present
#: upstream plus Ukrainian and Russian; Croatian and Irish do not exist upstream.
LANG_CODES = {
    "bul_Cyrl": "bg", "ces_Latn": "cs", "dan_Latn": "da", "nld_Latn": "nl", "eng_Latn": "en",
    "est_Latn": "et", "fin_Latn": "fi", "fra_Latn": "fr", "deu_Latn": "de", "ell_Grek": "el",
    "hun_Latn": "hu", "ita_Latn": "it", "lvs_Latn": "lv", "lit_Latn": "lt", "mlt_Latn": "mt",
    "pol_Latn": "pl", "por_Latn": "pt", "ron_Latn": "ro", "slk_Latn": "sk", "slv_Latn": "sl",
    "spa_Latn": "es", "swe_Latn": "sv", "ukr_Cyrl": "uk", "rus_Cyrl": "ru",
}  # fmt: skip


def _get(cfg, key: str, default=None):
    """Read *key* from a dict or an OmegaConf node."""
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


_MANIFEST_CACHE: dict[str, dict[str, list[dict]]] = {}
#: ``freeze`` runs once per source, and a spec has one source per language: parse
#: the (tens of MB) manifest once per process, not once per source.
_CACHE_LOCK = threading.Lock()


def _rows_by_language(root: Path) -> dict[str, list[dict]]:
    """Return the reconstruction's manifest rows grouped by upstream language code."""
    key = str(root)
    cached = _MANIFEST_CACHE.get(key)
    if cached is not None:
        return cached
    with _CACHE_LOCK:
        if key in _MANIFEST_CACHE:
            return _MANIFEST_CACHE[key]
        by_lang: dict[str, list[dict]] = {}
        with open(root / MANIFEST_NAME, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    by_lang.setdefault(row["lang"], []).append(row)
        _MANIFEST_CACHE[key] = by_lang
        return by_lang


def _check_revision(root: Path, revision: str) -> None:
    """Refuse a reconstruction that does not say it was built from *revision*.

    The reconstruction's README records the upstream commit it came from. A
    spec pins a revision for the same reason every other source here does:
    without it, nothing says which release a number was computed on.

    Raises:
        ValueError: If the README is missing or does not mention *revision*.
    """
    readme = root / "README.md"
    if not readme.exists() or revision not in readme.read_text(encoding="utf-8"):
        raise ValueError(
            f"{root} does not record upstream revision {revision} (looked in README.md). "
            "Either the spec's `revision` is stale or this is a reconstruction of another release."
        )


class PolySpeechReader:
    """Reads one (language, variant) slice of a local PolySpeech-100 reconstruction."""

    source_type = "polyspeech"
    locator_kind = "polyspeech"

    def freeze(self, source_cfg: dict, source_index: int) -> SourceResult:
        """Yield a multiple-choice record per sample of one language.

        Args:
            source_cfg: Source entry with ``root`` (reconstruction directory;
                ``${ENV}`` is expanded by the spec loader), ``revision`` (the
                upstream commit, checked against the reconstruction's README),
                ``language`` (upstream code such as ``deu_Latn``), ``variant``
                (``pa_qa`` or ``pa_qt``), optional ``instruction_template``
                (needs ``{question}`` and ``{choices}``), ``segment_gap_s``,
                ``question_gap_s``, ``max_samples``, ``seed`` and ``tags``
                (``lang`` defaults to the 2-letter code, ``dataset_id`` to
                ``polyspeech-<variant>-<lang>``).
            source_index: Position in the spec, used for sample keys.

        Raises:
            ValueError: If a required key is missing or invalid, the root is
                not a reconstruction, or the language is not in it.
        """
        root_cfg = str(_get(source_cfg, "root") or "")
        if not root_cfg:
            raise ValueError(f"Source #{source_index}: polyspeech requires `root`.")
        if "${" in root_cfg:
            raise ValueError(
                f"Source #{source_index}: `root` still contains an unexpanded variable ({root_cfg!r}). "
                "Export it before running (see docs/polyspeech.md)."
            )
        root = Path(root_cfg)
        if not (root / MANIFEST_NAME).exists():
            raise ValueError(f"Source #{source_index}: no {MANIFEST_NAME} under {root}.")

        revision = _get(source_cfg, "revision")
        if not revision:
            raise ValueError(
                f"Source #{source_index} ({root}): `revision` is required. Without it nothing "
                "records which upstream release the numbers were computed on."
            )
        _check_revision(root, str(revision))

        language = str(_get(source_cfg, "language") or "")
        variant = str(_get(source_cfg, "variant") or "")
        if variant not in VARIANTS:
            raise ValueError(f"Source #{source_index}: `variant` must be one of {VARIANTS}, got {variant!r}.")
        by_lang = _rows_by_language(root)
        if language not in by_lang:
            raise ValueError(
                f"Source #{source_index}: language {language!r} is not in {root}. Available: {sorted(by_lang)}."
            )

        template = str(_get(source_cfg, "instruction_template") or DEFAULT_CHOICE_INSTRUCTION_TEMPLATE)
        segment_gap_s = float(_get(source_cfg, "segment_gap_s", DEFAULT_SEGMENT_GAP_S))
        question_gap_s = float(_get(source_cfg, "question_gap_s", DEFAULT_QUESTION_GAP_S))
        max_samples = _get(source_cfg, "max_samples")
        seed = int(_get(source_cfg, "seed", 0) or 0)
        tags = dict(_get(source_cfg, "tags", {}) or {})
        lang = str(tags.get("lang") or LANG_CODES.get(language, language))
        dataset_id = str(tags.get("dataset_id") or f"polyspeech-{variant}-{lang}")

        rows = by_lang[language]
        first = rows[0]["passage_audio"][0]["path"]
        if not (root / first).exists():
            raise ValueError(f"Source #{source_index}: {root / first} is missing; is `root` the whole reconstruction?")

        candidates: list[EvalRecord] = []
        dropped = {"no_audio": 0, "bad_options": 0, "bad_answer": 0, "duplicate_options": 0, "no_question_text": 0}

        for row in rows:
            options = [str(o).strip() for o in row["options"]]
            answer = int(row["correct_answer"])
            if len(options) != 4 or not all(options):
                dropped["bad_options"] += 1
                continue
            if len({o.casefold() for o in options}) != len(options):
                # The reference is the option's *text*: two equal texts make the answer ambiguous.
                dropped["duplicate_options"] += 1
                continue
            if not 1 <= answer <= len(options):
                dropped["bad_answer"] += 1
                continue
            question_text = str(row["question"]).strip()
            if variant == "pa_qt" and not question_text:
                dropped["no_question_text"] += 1
                continue
            passage = [e["path"] for e in row["passage_audio"]]
            question_audio = [e["path"] for e in row["question_audio"]]
            if not passage or (variant == "pa_qa" and not question_audio):
                dropped["no_audio"] += 1
                continue

            passage_s = sum(e["duration_s"] for e in row["passage_audio"]) + segment_gap_s * (len(passage) - 1)
            duration = passage_s
            params: dict[str, object] = {
                "root": str(root),
                "passage": passage,
                "segment_gap_s": segment_gap_s,
            }
            if variant == "pa_qa":
                params["question"] = question_audio
                params["question_gap_s"] = question_gap_s
                duration += question_gap_s + sum(e["duration_s"] for e in row["question_audio"])
                duration += segment_gap_s * (len(question_audio) - 1)
                shown_question = PA_QA_QUESTION
            else:
                shown_question = question_text

            candidates.append(
                EvalRecord(
                    sample_key="",  # assigned below, after subsetting
                    task="audio_mcq",
                    target=options[answer - 1],
                    audio=AudioLocator(kind=self.locator_kind, params=params),
                    lang=lang,
                    dataset_id=dataset_id,
                    source=f"polyspeech-100@{str(revision)[:10]}/{variant}/{language}",
                    instruction=_render_instruction(template, shown_question, options),
                    choices=options,
                    cut_id=str(row["id"]),
                    duration=round(duration, 3),
                    extra={
                        # What the model is shown as "the question" (the scaffold for pa_qa)...
                        "question": shown_question,
                        # ...and the real one, for analysis only: never sent in pa_qa.
                        "question_text": question_text,
                        "variant": variant,
                        "upstream_lang": language,
                        "index": int(row["index"]),
                        "qid": int(row["qid"]),
                    },
                )
            )

        if max_samples is not None and len(candidates) > int(max_samples):
            import random as _random

            order = {id(r): i for i, r in enumerate(candidates)}
            _random.Random(seed).shuffle(candidates)
            candidates = sorted(candidates[: int(max_samples)], key=lambda r: order[id(r)])

        for ordinal, record in enumerate(candidates):
            record.sample_key = f"{source_index:02d}-{ordinal:06d}"

        return SourceResult(
            records=candidates,
            stats={
                "read": len(rows),
                "kept": len(candidates),
                **{f"dropped_{k}": v for k, v in dropped.items()},
                "hours": round(sum(r.duration or 0.0 for r in candidates) / 3600.0, 4),
            },
        )

    def load_audio(self, locator: AudioLocator):
        """Decode, resample and join a sample's segments into one waveform.

        Returns:
            A 1-D float32 mono array at 16 kHz: the passage segments separated
            by ``segment_gap_s`` of silence, then (``pa_qa``) ``question_gap_s``
            of silence and the question segments.

        Raises:
            RuntimeError: If a segment cannot be decoded.
        """
        import numpy as np

        params = locator.params
        root = Path(str(params["root"]))
        segment_gap = np.zeros(int(round(float(params.get("segment_gap_s", DEFAULT_SEGMENT_GAP_S)) * TARGET_SAMPLE_RATE)), np.float32)

        def join(paths: list[str]) -> np.ndarray:
            parts: list[np.ndarray] = []
            for i, rel in enumerate(paths):
                if i:
                    parts.append(segment_gap)
                parts.append(_load_segment(root / str(rel)))
            return np.concatenate(parts)

        audio = join(list(params["passage"]))
        if params.get("question"):
            question_gap = np.zeros(
                int(round(float(params.get("question_gap_s", DEFAULT_QUESTION_GAP_S)) * TARGET_SAMPLE_RATE)), np.float32
            )
            audio = np.concatenate([audio, question_gap, join(list(params["question"]))])
        return audio, TARGET_SAMPLE_RATE


def _load_segment(path: Path):
    """Decode one file to mono float32 at :data:`TARGET_SAMPLE_RATE`."""
    import numpy as np
    import soundfile as sf

    try:
        data, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except Exception as exc:
        raise RuntimeError(f"Could not decode audio {path}.") from exc
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sample_rate != TARGET_SAMPLE_RATE:
        import librosa

        data = librosa.resample(data, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
    return data


register_reader(PolySpeechReader())
