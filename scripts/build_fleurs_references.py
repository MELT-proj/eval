"""Build FLEURS reference-text JSONs for any target locale, for fleurs24-st-x*-*.yaml.

Writes `{sentence_id: raw_transcription}` to configs/refs/fleurs-<lang>-reference-
{dev,test}.json, like build_fleurs_en_reference.py, but from the per-locale
`data/<locale>/{dev,test}.tsv` metadata files of the HF `google/fleurs` repo
instead of `load_dataset`: same `raw_transcription` column, no audio archives
downloaded, no dataset script. Needs outbound internet (nyx has it, artemis
and MN5 do not). One-time build step; the JSON files are committed.

Checked 2026-10-10: for en_us this reproduces the committed
fleurs-en-reference-{dev,test}.json byte for byte (150 and 350 ids).

    python scripts/build_fleurs_references.py de_de fr_fr es_419 it_it nl_nl pt_br
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REFS = Path(__file__).parent.parent / "configs/refs"
SPLITS = {"dev": "dev", "test": "test"}  # TSV split name -> output suffix


def iso_code(locale: str) -> str:
    """FLEURS locale (``es_419``, ``pt_br``) -> the ISO 639-1 code the configs tag."""
    return locale.split("_", 1)[0]


def read_reference_map(tsv_path: str, locale: str, split: str) -> dict[str, str]:
    """Read one FLEURS TSV (id, file, raw_transcription, ...) into ``{id: raw_transcription}``.

    Raises:
        ValueError: If one sentence id carries two different raw transcriptions
            across its recordings (never seen so far; the references would then
            need a majority vote, as build_fleurs_en_reference.py notes).
    """
    csv.field_size_limit(sys.maxsize)
    ref_map: dict[str, str] = {}
    with open(tsv_path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            key, text = str(row[0]), row[2]
            existing = ref_map.get(key)
            if existing is not None and existing != text:
                raise ValueError(
                    f"{locale} {split}: id {key} has two raw_transcription values "
                    f"({existing!r} vs {text!r})"
                )
            ref_map[key] = text
    return ref_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("locales", nargs="+", help="FLEURS locales, e.g. de_de es_419")
    args = parser.parse_args()
    for locale in args.locales:
        for split, suffix in SPLITS.items():
            tsv = hf_hub_download("google/fleurs", f"data/{locale}/{split}.tsv", repo_type="dataset")
            ref_map = read_reference_map(tsv, locale, split)
            out_path = REFS / f"fleurs-{iso_code(locale)}-reference-{suffix}.json"
            out_path.write_text(
                json.dumps(ref_map, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"{locale} {split}: {len(ref_map)} ids -> {out_path}")


if __name__ == "__main__":
    main()
