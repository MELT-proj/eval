"""Reader for remote HuggingFace ``datasets`` test splits.

Freezing reads **metadata only**, the same contract the shar reader keeps:
the audio column is cast to ``Audio(decode=False)`` before iterating, so a
freeze pass never pulls audio bytes through a codec — only the reference text
and (when the dataset provides one) a duration column are touched.

Audio is decoded independently of ``datasets``' own ``Audio(decode=True)``
path, with ``soundfile`` directly from the raw bytes. The installed
``datasets`` release requires ``torchcodec`` to decode through its own
pipeline, which is not otherwise a dependency of this project; reading the
raw bytes with ``soundfile`` (already a `shar` dependency) avoids adding it.
This means formats ``soundfile`` cannot open — MP3 without a suitable
`libsndfile`, for one — are not readable here; re-export such a split to
wav/flac first, or use a different reader.

Because the audio format is opaque at freeze time, per-sample duration is
only recorded when the dataset itself provides a duration column — computing
it would mean decoding, which is exactly the cost this pass exists to avoid.

Benchmarks reached this way often ship their own prompt per sample rather
than drawing one from the training template pool -- AIR-Bench asks a different
question of every clip, and half of them are multiple choice. So a source can
name an ``instruction_column`` (and, for multiple choice, ``choices_columns``),
and the reader composes them into the record's ``instruction`` through an
``instruction_template``. Corpus text is brace-escaped on the way in, because
the instruction is itself a template downstream -- see
``prompt.escape_literal``.

Unlike shar corpora, which are already 16 kHz mono, an arbitrary HF dataset
is not, and the MELT provider batches by borrowing one sample rate for the
whole batch (see ``providers/melt.py``). So audio here is resampled to 16 kHz
mono at read time, not left native — a batch that silently mixed sample rates
would corrupt every sample in it but for the one whose rate was actually used.
"""

from __future__ import annotations

import io
import threading

from melteval.manifest import AudioLocator, EvalRecord
from melteval.prompt import escape_literal
from melteval.readers.base import SourceResult, register_reader


TARGET_SAMPLE_RATE = 16000

#: Labels offered for multiple-choice options, in ``choices_columns`` order.
CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#: Composed when a source names an ``instruction_column`` but no template. The
#: audio token has to be in there: the processor expands it into encoder
#: frames, and a prompt without one is a text-only prompt about nothing.
DEFAULT_INSTRUCTION_TEMPLATE = "{audio_token}\n{question}"

#: Used instead when the source also names ``choices_columns``. The closing
#: line is what makes the answer parseable at all -- without it a model
#: paraphrases, and :func:`melteval.scorers.mcq_scorer` has to fall back to
#: matching option text.
DEFAULT_CHOICE_INSTRUCTION_TEMPLATE = (
    "{audio_token}\n{question}\n{choices}\nAnswer with the letter of the correct option."
)


def _get(cfg, key: str, default=None):
    """Read *key* from a dict or an OmegaConf node."""
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _render_instruction(template: str, question: str, choices: list[str]) -> str:
    """Compose one sample's prompt from *template*, *question* and *choices*.

    Placeholders are substituted by literal replacement rather than
    ``str.format``, and only ``{question}`` and ``{choices}`` are touched.
    Anything else the spec author wrote -- ``{audio_token}`` above all -- has
    to reach :func:`melteval.prompt.render_user_prompt` still spelled as a
    placeholder, which a ``format`` call here would have already consumed.

    Args:
        template: The source's ``instruction_template``.
        question: The row's question text, verbatim from the corpus.
        choices: Non-blank options, in ``choices_columns`` order. Empty for a
            free-form benchmark.

    Returns:
        The rendered instruction, with corpus text brace-escaped.

    Raises:
        ValueError: If the template drops the question, drops options that
            exist, or the source offers more options than there are labels.
            All three produce a prompt that reads fine and measures the wrong
            thing: a model asked to choose between options it was never shown
            scores at chance, and nothing in the log says why.
    """
    if "{question}" not in template:
        raise ValueError(
            f"instruction_template has no {{question}} placeholder: {template!r}. Every "
            "sample would then get the same prompt, and the benchmark's own questions "
            "would never reach the model."
        )
    if choices and "{choices}" not in template:
        raise ValueError(
            f"instruction_template has no {{choices}} placeholder: {template!r}, but this "
            "source sets choices_columns. The options would be scored against but never "
            "shown, which is not the benchmark."
        )
    if len(choices) > len(CHOICE_LABELS):
        raise ValueError(
            f"{len(choices)} options, but only {len(CHOICE_LABELS)} labels to show them "
            "under. Score this source with something other than option labels."
        )

    rendered_choices = "\n".join(
        f"{CHOICE_LABELS[position]}. {choice}" for position, choice in enumerate(choices)
    )
    return (
        template.replace("{question}", escape_literal(question))
        .replace("{choices}", escape_literal(rendered_choices))
        .strip()
    )


class HFReader:
    """Reads HuggingFace ``datasets`` splits, one per source config entry."""

    source_type = "hf_dataset"
    locator_kind = "hf"

    def freeze(self, source_cfg: dict, source_index: int) -> SourceResult:
        """Yield a record per row with a usable reference.

        Args:
            source_cfg: Source entry with ``repo``, ``revision``, ``split``,
                optional ``name`` (the dataset's own "config" argument),
                ``audio_column``, ``text_column``, ``id_column``,
                ``source_text_column``, ``instruction_column``,
                ``choices_columns``, ``instruction_template``,
                ``min_duration``, ``max_duration``, ``max_samples``, ``seed``,
                and ``tags``.
            source_index: Position in the spec, used for sample keys.

        Raises:
            ValueError: If ``repo`` or ``revision`` is missing. A revision is
                required, not merely accepted, because an unpinned HF dataset
                can change under a frozen set without anyone noticing —
                exactly what pinning is supposed to prevent.
        """
        from datasets import Audio, load_dataset

        repo = _get(source_cfg, "repo")
        if not repo:
            raise ValueError(f"Source #{source_index}: hf_dataset requires `repo`.")
        revision = _get(source_cfg, "revision")
        if not revision:
            raise ValueError(
                f"Source #{source_index} ({repo}): `revision` is required. Without it a frozen "
                "set silently stops being reproducible the moment the dataset is updated upstream."
            )

        name = _get(source_cfg, "name")
        split = str(_get(source_cfg, "split", "test"))
        audio_column = str(_get(source_cfg, "audio_column", "audio"))
        text_column = str(_get(source_cfg, "text_column", "text"))
        id_column = _get(source_cfg, "id_column")
        source_text_column = _get(source_cfg, "source_text_column")
        instruction_column = _get(source_cfg, "instruction_column")
        choices_columns = [str(c) for c in (_get(source_cfg, "choices_columns") or [])]
        instruction_template = str(
            _get(source_cfg, "instruction_template")
            or (DEFAULT_CHOICE_INSTRUCTION_TEMPLATE if choices_columns else DEFAULT_INSTRUCTION_TEMPLATE)
        )
        duration_column = str(_get(source_cfg, "duration_column", "duration"))
        min_duration = _get(source_cfg, "min_duration")
        max_duration = _get(source_cfg, "max_duration")
        max_samples = _get(source_cfg, "max_samples")
        seed = int(_get(source_cfg, "seed", 0) or 0)
        tags = dict(_get(source_cfg, "tags", {}) or {})

        dataset = load_dataset(repo, name=name, split=split, revision=revision)
        dataset = dataset.cast_column(audio_column, Audio(decode=False))

        source_label = f"{repo}@{split}" + (f"/{name}" if name else "")
        candidates: list[EvalRecord] = []
        n_read = 0
        n_dropped_no_reference = 0
        n_dropped_duration = 0
        n_dropped_no_instruction = 0

        for index, row in enumerate(dataset):
            n_read += 1

            target = row.get(text_column)
            if not target or not str(target).strip():
                n_dropped_no_reference += 1
                continue

            duration = row.get(duration_column)
            duration = float(duration) if duration is not None else None
            if duration is not None:
                if min_duration is not None and duration < float(min_duration):
                    n_dropped_duration += 1
                    continue
                if max_duration is not None and duration > float(max_duration):
                    n_dropped_duration += 1
                    continue

            instruction: str | None = None
            choices: list[str] | None = None
            if instruction_column:
                question = row.get(instruction_column)
                if not question or not str(question).strip():
                    # The record would fall back to the training template pool
                    # for a task that has none, and blow up mid-generation
                    # rather than here. Drop it and say so, like a missing
                    # reference.
                    n_dropped_no_instruction += 1
                    continue
                choices = [
                    str(row[column]).strip()
                    for column in choices_columns
                    if str(row.get(column) or "").strip()
                ]
                instruction = _render_instruction(instruction_template, str(question).strip(), choices)
                choices = choices or None

            row_id = str(row[id_column]) if id_column else str(index)
            source_text = row.get(source_text_column) if source_text_column else None

            candidates.append(
                EvalRecord(
                    sample_key="",  # assigned below, after subsetting
                    task=str(tags.get("task", "")),
                    target=str(target).strip(),
                    audio=AudioLocator(
                        kind=self.locator_kind,
                        params={
                            "repo": repo,
                            "revision": revision,
                            "split": split,
                            "index": index,
                            "audio_column": audio_column,
                            **({"name": name} if name else {}),
                        },
                    ),
                    lang=str(tags.get("lang", "")),
                    src_lang=str(tags.get("src_lang", "")),
                    tgt_lang=str(tags.get("tgt_lang", "")),
                    dataset_id=str(tags.get("dataset_id", "") or repo),
                    source=source_label,
                    source_text=str(source_text).strip() if source_text else None,
                    instruction=instruction,
                    choices=choices,
                    cut_id=row_id,
                    duration=duration,
                )
            )

        if max_samples is not None and len(candidates) > int(max_samples):
            import random as _random

            _random.Random(seed).shuffle(candidates)
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
                "dropped_no_instruction": n_dropped_no_instruction,
                "hours": round(sum(r.duration or 0.0 for r in candidates) / 3600.0, 4),
            },
        )

    def load_audio(self, locator: AudioLocator):
        """Resolve an hf locator to ``(samples, sample_rate)``.

        Raises:
            RuntimeError: If the audio column has neither embedded bytes nor a
                resolvable path, or ``soundfile`` cannot open the format.
        """
        import numpy as np
        import soundfile as sf

        params = locator.params
        dataset = _cached_dataset(
            repo=str(params["repo"]),
            name=params.get("name"),
            split=str(params["split"]),
            revision=str(params["revision"]),
            audio_column=str(params.get("audio_column", "audio")),
        )
        row = dataset[int(params["index"])]
        field = row[str(params.get("audio_column", "audio"))]

        try:
            if field.get("bytes"):
                data, sample_rate = sf.read(io.BytesIO(field["bytes"]), dtype="float32", always_2d=False)
            elif field.get("path"):
                data, sample_rate = sf.read(field["path"], dtype="float32", always_2d=False)
            else:
                raise RuntimeError(f"Audio locator has neither bytes nor a path: {locator}")
        except Exception as exc:
            raise RuntimeError(
                f"Could not decode audio for {locator}. If the source encodes audio as MP3, this "
                "usually means soundfile's libsndfile build cannot read it -- re-export the split "
                "to wav/flac, or read it with a reader that has an MP3 decoder."
            ) from exc

        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)

        if sample_rate != TARGET_SAMPLE_RATE:
            import librosa

            data = librosa.resample(data, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
            sample_rate = TARGET_SAMPLE_RATE

        return data, sample_rate


_DATASET_CACHE: dict[tuple, object] = {}
#: Guards the cache against the provider resolving several samples' audio at
#: once. ``load_audio`` runs on a worker thread per in-flight sample (see
#: ``providers/melt.py``), so an unguarded check-then-insert has every thread
#: in the first batch load and build the same split independently -- the same
#: shape of bug the shar reader hit for real (MELT-proj/eval#2), minus the
#: corruption, since two Datasets over the same arrow files do not share a
#: file handle.
_CACHE_LOCK = threading.Lock()


def _cached_dataset(repo: str, name: str | None, split: str, revision: str, audio_column: str):
    """Load (once per source) the dataset a locator's index is relative to."""
    key = (repo, name, split, revision)
    cached = _DATASET_CACHE.get(key)
    if cached is not None:
        return cached

    from datasets import Audio, load_dataset

    with _CACHE_LOCK:
        if key in _DATASET_CACHE:  # another thread loaded it while this one waited
            return _DATASET_CACHE[key]
        dataset = load_dataset(repo, name=name, split=split, revision=revision)
        dataset = dataset.cast_column(audio_column, Audio(decode=False))
        _DATASET_CACHE[key] = dataset
        return dataset


register_reader(HFReader())
