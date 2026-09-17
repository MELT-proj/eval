#!/usr/bin/env python3
"""
Calculate corpus WER/CER for one or more hypothesis (i.e., system output) sets against a frozen set.

It supports multiple hypothesis sources and applies a specified text normalizer to both references and hypotheses before scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from melteval.manifest import MANIFEST_NAME, read_manifest


def _get_normalizer(name: str):
    """
    Get a text normalizer function by name.

    Args:
        name: The name of the normalizer. Options are "none", "lower", or any name supported by melteval.scorers.get_normalizer.

    Returns:
        A function that takes a string and returns the normalized string.
    """
    if name == "none":
        return lambda text: text
    if name == "lower":
        import re

        _punct = re.compile(r"[^\w\s]", flags=re.UNICODE)
        _ws = re.compile(r"\s+")
        return lambda text: _ws.sub(" ", _punct.sub(" ", (text or "").lower())).strip()
    from melteval.scorers import get_normalizer

    return get_normalizer(name)


def _load_references(frozen_set: Path) -> dict[str, str]:
    """Map join key -> reference text. Key is cut_id, else sample_key."""
    refs: dict[str, str] = {}
    for rec in read_manifest(frozen_set / MANIFEST_NAME):
        key = str(rec.cut_id if rec.cut_id is not None else rec.sample_key)
        target = rec.target[0] if isinstance(rec.target, list) else rec.target
        refs[key] = target or ""
    return refs


def _load_hyps(path: Path) -> dict[str, str]:
    """Map join key -> hypothesis, from a .eval log or an inference .jsonl."""
    if path.suffix == ".eval":
        from inspect_ai.log import read_eval_log

        log = read_eval_log(str(path), resolve_attachments=True)
        out: dict[str, str] = {}
        for sample in log.samples or []:
            if getattr(sample, "error", None) is not None:
                continue
            key = str((sample.metadata or {}).get("cut_id") or sample.id)
            out[key] = (sample.output.completion if sample.output else "") or ""
        return out

    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out[str(rec["id"])] = rec.get("hypothesis", "") or ""
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--frozen-set", required=True, type=Path, help="Frozen set directory.")
    parser.add_argument(
        "--normalizer",
        default="lower",
        help="Normalizer applied to both sides: lower (default, no `melt`), "
        "basic | english (match `inspect eval`, need `melt`), or none.",
    )
    parser.add_argument(
        "sources",
        nargs="+",
        metavar="label=path",
        help="One or more hypothesis sets, e.g. melteval=run.eval upstream=upstream.jsonl",
    )
    args = parser.parse_args()

    import jiwer

    normalize = _get_normalizer(args.normalizer)
    refs = _load_references(args.frozen_set)
    if not refs:
        raise SystemExit(f"no references in {args.frozen_set}/{MANIFEST_NAME}")

    print(f"frozen set:  {args.frozen_set}  ({len(refs)} samples)")
    print(f"normalizer:  {args.normalizer}")
    print()
    print(f"{'system':<16} {'n':>5} {'WER':>9} {'CER':>9}")
    print("-" * 42)

    for spec in args.sources:
        if "=" not in spec:
            raise SystemExit(f"source must be label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        hyps = _load_hyps(Path(path))

        keys = [k for k in refs if k in hyps and normalize(refs[k]).strip()]
        missing = len(refs) - len(keys)
        r = [normalize(refs[k]) for k in keys]
        h = [normalize(hyps[k]) for k in keys]

        words = jiwer.process_words(r, h)
        chars = jiwer.process_characters(r, h)
        note = f"  ({missing} unscored)" if missing else ""
        print(f"{label:<16} {len(keys):>5} {words.wer:>9.4f} {chars.cer:>9.4f}{note}")


if __name__ == "__main__":
    main()
