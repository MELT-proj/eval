"""Reader for MCIF (https://huggingface.co/datasets/FBK-MT/MCIF).

MCIF does not fit :mod:`melteval.readers.hf`'s contract: the reference for a
sample is not a column on that sample's own row. It lives in a separate
``MCIF.{track}.{lang}.ref.xml.gz`` file, keyed by an ``id`` that groups one or
more parquet rows into a single reference unit -- one row for QA/summarisation,
but many consecutive rows for ASR/translation in the ``short`` track, where a
whole talk's transcript is the reference and the parquet has already cut that
talk into short segments the model transcribes one at a time. Scoring has to
re-assemble the group's hypotheses in order before comparing to the one
reference; see ``melteval.scorers.chunked_asr_scorer`` /
``chunked_st_scorer``, and ``EvalRecord.extra`` (``group_id``, ``group_order``,
``group_size``), which is what carries the grouping through to the scorer.

The ``long`` track uses the same reference format, but every group there has
exactly one member: each row is already a whole talk, not a segment of one. So
the same reader and the same grouped scorers cover both tracks; ``long`` simply
degenerates to plain per-sample WER/BLEU, over audio that runs to the length of
a real conference talk rather than a short clip.

Each reference sample also carries which macro-task it is (``ASR``, ``TRANS``,
``QA``, ``SUM``) and, for QA, ``qa_type``/``qa_origin``. ASR and TRANS route to
the grouped scorers above; QA and SUM are both open-ended free text with no
lexical metric that means anything (the paper's own metric, BERTScore, needs a
model this reader has no business loading -- see
``melteval.scorers.chat_scorer``'s docstring for the same argument), so both
route to the existing ``audio_chat`` task and its judge-model scorer.

Audio and video are plain string paths into the same HF dataset repo, not a
``datasets`` ``Audio`` feature -- MCIF ships raw files rather than embedding
them in the parquet -- so this reader resolves them with
``huggingface_hub.hf_hub_download`` directly rather than through the
``datasets`` audio-decode path ``readers/hf.py`` uses. Video is never read:
this harness evaluates speech models, and every MCIF sample already has an
audio path standing in for it.

The scorers here (via ``melteval.registry``) are fast, dependency-light
approximations, on purpose: nothing in this package's default scoring path
loads a neural metric next to the model under evaluation (see
``melteval/rescore.py``). To reproduce the paper's own metrics -- WER against
the exact normalizer it uses, COMET, BERTScore -- as a **separate, post-hoc**
`inspect score` step, see ``melteval/mcif_scoring.py``, which calls straight
into the official ``mcif`` package instead of reimplementing any of it.
"""

from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET

from melteval.manifest import AudioLocator, EvalRecord
from melteval.prompt import escape_literal
from melteval.readers.base import SourceResult, register_reader
from melteval.readers.hf import DEFAULT_INSTRUCTION_TEMPLATE


TARGET_SAMPLE_RATE = 16000

#: Reference file's own macro-task label -> the melteval task it becomes.
#: ASR/TRANS need the grouped corpus scorers because a reference can span many
#: rows; QA/SUM are always one row per reference and go through the same
#: judge-graded path as AIR-Bench Chat.
TASK_KIND_TO_TASK = {
    "ASR": "chunked_asr",
    "TRANS": "chunked_st",
    "QA": "audio_chat",
    "SUM": "audio_chat",
}


def _get(cfg, key: str, default=None):
    """Read *key* from a dict or an OmegaConf node."""
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def download_reference_gz(repo: str, revision: str, track: str, lang: str) -> str:
    """Download ``MCIF.<track>.<lang>.ref.xml.gz`` and return its local path.

    Shared with :mod:`melteval.mcif_scoring`, which re-downloads the same
    file (``huggingface_hub`` caches it, so this costs nothing the second
    time) to score a finished run against the official reference rather than
    whatever a frozen set happened to carry.

    The file is left gzipped: :func:`xml.etree.ElementTree.parse` accepts a
    file object as readily as a path, so callers open it with :mod:`gzip`
    rather than this function decompressing it to a temporary file.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        filename=f"MCIF.{track}.{lang}.ref.xml.gz",
    )


class MCIFReader:
    """Reads one (track, prompt style, target language) slice of MCIF."""

    source_type = "mcif"
    locator_kind = "mcif"

    def freeze(self, source_cfg: dict, source_index: int) -> SourceResult:
        """Join the parquet split to its reference file and yield records.

        Args:
            source_cfg: Source entry with ``repo`` (default ``FBK-MT/MCIF``),
                ``revision``, ``track`` (``short``/``long``), ``prompt_type``
                (``fixed``/``mixed``), ``lang`` (target language: selects both
                the ``prompt_<lang>`` column and which reference file to
                read), optional ``id_column`` (default ``id``), optional
                ``modality`` (default ``audio``; a reference sample without a
                ``<{modality}_path>`` is dropped), and ``tags``.
            source_index: Position in the spec, used for sample keys.

        Raises:
            ValueError: If ``revision`` or ``lang`` is missing, ``track`` is
                not ``short``/``long``, or the reference file has no task
                block for the requested ``(track, lang)``.
        """
        from datasets import load_dataset

        repo = str(_get(source_cfg, "repo", "FBK-MT/MCIF"))
        revision = _get(source_cfg, "revision")
        if not revision:
            raise ValueError(
                f"Source #{source_index} ({repo}): `revision` is required. Without it a "
                "frozen set silently stops being reproducible the moment the dataset is "
                "updated upstream."
            )
        track = str(_get(source_cfg, "track", "short"))
        if track not in ("short", "long"):
            raise ValueError(f"Source #{source_index} ({repo}): `track` must be short/long, got {track!r}.")
        prompt_type = str(_get(source_cfg, "prompt_type", "fixed"))
        lang = str(_get(source_cfg, "lang") or "")
        if not lang:
            raise ValueError(
                f"Source #{source_index} ({repo}): `lang` is required -- it selects both the "
                "target-language prompt column and which reference file to read."
            )
        id_column = str(_get(source_cfg, "id_column", "id"))
        modality = str(_get(source_cfg, "modality", "audio") or "")
        tags = dict(_get(source_cfg, "tags", {}) or {})

        config_name = f"{track}_{prompt_type}prompt"
        prompt_column = f"prompt_{lang}"

        dataset = load_dataset(repo, name=config_name, split="test", revision=revision)
        id_to_index = {str(v): i for i, v in enumerate(dataset[id_column])}

        ref_path = download_reference_gz(repo, revision, track, lang)
        with gzip.open(ref_path, "rb") as fh:
            root = ET.parse(fh).getroot()

        task_node = None
        available = []
        for node in root.iter("task"):
            if node.attrib.get("track") == track and node.attrib.get("text_lang") == lang:
                task_node = node
                break
            available.append((node.attrib.get("track"), node.attrib.get("text_lang")))
        if task_node is None:
            raise ValueError(
                f"Source #{source_index} ({repo}): no task track={track!r} text_lang={lang!r} "
                f"in the reference file. Available: {available}."
            )

        source_label = f"{repo}@{config_name}/{lang}"
        candidates: list[EvalRecord] = []
        n_read = 0
        n_dropped_unknown_task = 0
        n_dropped_no_modality = 0
        n_dropped_no_reference = 0
        n_dropped_missing_row = 0
        n_dropped_no_instruction = 0
        task_counts: dict[str, int] = {}

        for sample_el in task_node.iter("sample"):
            n_read += 1
            task_kind = sample_el.attrib.get("task", "")
            melteval_task = TASK_KIND_TO_TASK.get(task_kind)
            if melteval_task is None:
                n_dropped_unknown_task += 1
                continue

            if modality and sample_el.find(f"{modality}_path") is None:
                n_dropped_no_modality += 1
                continue

            reference_el = sample_el.find("reference")
            reference = (reference_el.text or "").strip() if reference_el is not None else ""
            if not reference:
                n_dropped_no_reference += 1
                continue

            iid = sample_el.attrib.get("iid", "")
            raw_ids = [rid.strip() for rid in sample_el.attrib.get("id", "").split(",") if rid.strip()]
            qa_type = sample_el.attrib.get("qa_type")
            qa_origin = sample_el.attrib.get("qa_origin")

            for order, raw_id in enumerate(raw_ids):
                index = id_to_index.get(raw_id)
                if index is None:
                    n_dropped_missing_row += 1
                    continue
                row = dataset[index]

                prompt_text = row.get(prompt_column)
                if not prompt_text or not str(prompt_text).strip():
                    n_dropped_no_instruction += 1
                    continue

                audio_path = row.get("audio")
                if not audio_path:
                    n_dropped_no_modality += 1
                    continue

                instruction = DEFAULT_INSTRUCTION_TEMPLATE.replace(
                    "{question}", escape_literal(str(prompt_text).strip())
                )

                extra: dict[str, object] = {
                    "group_id": iid,
                    "group_order": order,
                    "group_size": len(raw_ids),
                    "task_kind": task_kind,
                    # So a post-hoc official scorer (melteval/mcif_scoring.py)
                    # can re-fetch the exact reference file that generated
                    # this sample straight from the log, without needing any
                    # of it passed in again as `-S` arguments.
                    "repo": repo,
                    "revision": revision,
                    "track": track,
                }
                if qa_type:
                    extra["qa_type"] = qa_type
                if qa_origin:
                    extra["qa_origin"] = qa_origin

                candidates.append(
                    EvalRecord(
                        sample_key="",  # assigned below, after subsetting
                        task=melteval_task,
                        target=reference,
                        audio=AudioLocator(
                            kind=self.locator_kind,
                            params={"repo": repo, "revision": revision, "path": str(audio_path)},
                        ),
                        lang=lang,
                        dataset_id=str(tags.get("dataset_id") or f"mcif-{track}-{prompt_type}-{lang}"),
                        source=source_label,
                        instruction=instruction,
                        cut_id=raw_id,
                        extra=extra,
                    )
                )
                task_counts[task_kind] = task_counts.get(task_kind, 0) + 1

        for ordinal, record in enumerate(candidates):
            record.sample_key = f"{source_index:02d}-{ordinal:06d}"

        return SourceResult(
            records=candidates,
            stats={
                "read": n_read,
                "kept": len(candidates),
                "dropped_unknown_task": n_dropped_unknown_task,
                "dropped_no_modality": n_dropped_no_modality,
                "dropped_no_reference": n_dropped_no_reference,
                "dropped_missing_row": n_dropped_missing_row,
                "dropped_no_instruction": n_dropped_no_instruction,
                **{f"task_{kind.lower()}": count for kind, count in task_counts.items()},
            },
        )

    def load_audio(self, locator: AudioLocator):
        """Download and decode a locator's raw audio file.

        Raises:
            RuntimeError: If ``soundfile`` cannot open the downloaded file.
        """
        import numpy as np
        import soundfile as sf
        from huggingface_hub import hf_hub_download

        params = locator.params
        local_path = hf_hub_download(
            repo_id=str(params["repo"]),
            repo_type="dataset",
            revision=str(params["revision"]),
            filename=str(params["path"]),
        )

        try:
            data, sample_rate = sf.read(local_path, dtype="float32", always_2d=False)
        except Exception as exc:
            raise RuntimeError(f"Could not decode audio for {locator}.") from exc

        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)

        if sample_rate != TARGET_SAMPLE_RATE:
            import librosa

            data = librosa.resample(data, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
            sample_rate = TARGET_SAMPLE_RATE

        return data, sample_rate


register_reader(MCIFReader())
