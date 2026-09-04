#!/usr/bin/env python3
"""Convert the HF split a melteval freeze config points at into a Lhotse CutSet.

Usage:
    python scripts/hf_to_lhotse.py --config configs/librispeech-hf-smoke.yaml \
        -o runs/crosscheck-smurf-cuts
    # -> runs/crosscheck-smurf-cuts/cuts.jsonl.gz  (+ runs/crosscheck-smurf-cuts/audio/*.wav)
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

TARGET_SAMPLE_RATE = 16000  # keep in step with melteval/readers/hf.py


def _first_source(config_path: Path) -> dict:
    """Return the first ``input_cfg`` entry of a melteval freeze config."""
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    sources = config.get("input_cfg") or []
    if not sources:
        raise ValueError(f"{config_path} has no input_cfg entries.")
    source = sources[0]
    if str(source.get("type", "hf_dataset")) != "hf_dataset":
        raise ValueError(
            f"{config_path} input_cfg[0] is type={source.get('type')!r}; this script only "
            "converts hf_dataset sources."
        )
    if len(sources) > 1:
        print(f"note: {config_path} has {len(sources)} sources; converting only the first.")
    return source


def _decode_audio(field: dict):
    """Decode one HF audio cell to a mono 16 kHz float32 array.

    Mirrors ``HFReader.load_audio`` in ``melteval/readers/hf.py``.
    """
    import librosa
    import numpy as np
    import soundfile as sf

    if field.get("bytes"):
        data, sample_rate = sf.read(io.BytesIO(field["bytes"]), dtype="float32", always_2d=False)
    elif field.get("path"):
        data, sample_rate = sf.read(field["path"], dtype="float32", always_2d=False)
    else:
        raise RuntimeError(f"Audio cell has neither bytes nor a path: {field!r}")

    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)

    if sample_rate != TARGET_SAMPLE_RATE:
        data = librosa.resample(data, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
        sample_rate = TARGET_SAMPLE_RATE
    return data, sample_rate


def convert(config_path: Path, out_dir: Path) -> Path:
    """Write ``<out_dir>/cuts.jsonl.gz`` from the config's HF split.

    Returns:
        The path to the written manifest.
    """
    from datasets import Audio, load_dataset
    from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
    import soundfile as sf

    source = _first_source(config_path)
    repo = source["repo"]
    revision = source.get("revision")
    if not revision:
        raise ValueError(f"{config_path}: the hf_dataset source needs a pinned `revision`.")
    name = source.get("name")
    split = str(source.get("split", "test"))
    audio_column = str(source.get("audio_column", "audio"))
    text_column = str(source.get("text_column", "text"))
    id_column = source.get("id_column")
    lang = str((source.get("tags") or {}).get("lang", "en"))

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(repo, name=name, split=split, revision=revision)
    dataset = dataset.cast_column(audio_column, Audio(decode=False))

    cuts: list = []
    n_read = 0
    n_dropped_no_reference = 0
    for index, row in enumerate(dataset):
        n_read += 1
        target = row.get(text_column)
        if not target or not str(target).strip():
            n_dropped_no_reference += 1
            continue

        # Same id the melteval hf reader records as EvalRecord.cut_id.
        uid = str(row[id_column]) if id_column else str(index)

        data, sample_rate = _decode_audio(row[audio_column])
        wav_path = (audio_dir / f"{uid.replace('/', '_')}.wav").resolve()
        sf.write(str(wav_path), data, sample_rate, subtype="FLOAT")

        recording = Recording.from_file(wav_path, recording_id=uid)
        supervision = SupervisionSegment(
            id=uid,
            recording_id=uid,
            start=0.0,
            duration=recording.duration,
            channel=0,
            text=str(target).strip(),
            language=lang,
        )
        cuts.append(
            MonoCut(
                id=uid,
                start=0.0,
                duration=recording.duration,
                channel=0,
                recording=recording,
                supervisions=[supervision],
            )
        )

    if not cuts:
        raise RuntimeError(f"No usable rows in {repo}@{split} (read {n_read}).")

    manifest_path = out_dir / "cuts.jsonl.gz"
    CutSet.from_cuts(cuts).to_file(str(manifest_path))

    ids = [c.id for c in cuts]
    preview = ids if len(ids) <= 6 else ids[:3] + ["..."] + ids[-3:]
    print(f"read {n_read} rows, dropped {n_dropped_no_reference} without a reference, wrote {len(cuts)} cuts")
    print(f"manifest: {manifest_path}")
    print(f"ids: {preview}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="melteval freeze config whose first input_cfg entry is an hf_dataset source.",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        required=True,
        type=Path,
        help="Directory for cuts.jsonl.gz and the extracted 16 kHz wavs.",
    )
    args = parser.parse_args()
    convert(args.config, args.out_dir)


if __name__ == "__main__":
    main()
