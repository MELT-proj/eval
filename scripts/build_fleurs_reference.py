"""Build the FLEURS reference-text JSONs that the FLEURS ST recipes join in by sentence id.

    python scripts/build_fleurs_reference.py                           # en_us test + validation (X->en)
    python scripts/build_fleurs_reference.py de_de fr_fr --splits test   # targets for en->X

Writes `{sentence_id: raw_transcription}` per FLEURS locale to
`configs/refs/fleurs-<lang>-reference-<test|dev>.json` (`de_de` -> `de`).
FLEURS reads the same sentences in every language, so one id is one sentence
everywhere: X->en recipes join the English text onto foreign audio, en->X ones
join the target language's text onto English audio. A one-time step: the JSONs
are committed and nothing here runs at freeze time.

Reads only `data/<locale>/<split>.tsv` from `google/fleurs` at a pinned revision
(metadata, not audio). Needs internet: mollete and nyx have it, artemis and MN5
do not.

Uses `raw_transcription` (the corpus's own cased, punctuated text), not the shar
`custom.pnc_text`, which differs across an id's duplicate recordings for 57/350
en_us test ids. An id with two different values is an error.
"""

import argparse
import csv
import json
from pathlib import Path

from huggingface_hub import hf_hub_download # type: ignore


REPO = "google/fleurs"
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
REFS = Path(__file__).resolve().parent.parent / "configs" / "refs"
#: The TSVs call the validation split `dev`; the JSONs are named `dev` too.
SPLIT_FILE = {"test": "test", "validation": "dev"}
SPLIT_NAME = {"test": "test", "validation": "dev"}
#: TSV columns: id, file name, raw_transcription, transcription, characters, num_samples, gender.
ID, RAW = 0, 2


def build(locale: str, split: str) -> dict[str, str]:
    """``{sentence_id: raw_transcription}`` for one FLEURS locale and split."""
    path = hf_hub_download(REPO, f"data/{locale}/{SPLIT_FILE[split]}.tsv", repo_type="dataset", revision=REVISION)
    ref_map: dict[str, str] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            key, text = row[ID], row[RAW]
            existing = ref_map.get(key)
            if existing is not None and existing != text:
                raise ValueError(
                    f"{locale} {split}: id {key} has two different raw_transcription values "
                    f"({existing!r} vs {text!r}); there is no tie-break rule for this yet."
                )
            ref_map[key] = text
    return ref_map


def main(argv: list[str] | None = None) -> None:
    """Write one JSON per (locale, split)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("locales", nargs="*", default=["en_us"], help="FLEURS locales (default: en_us).")
    parser.add_argument("--splits", nargs="+", default=["test", "validation"], choices=list(SPLIT_FILE))
    args = parser.parse_args(argv)

    for locale in args.locales:
        lang = locale.split("_")[0]
        for split in args.splits:
            ref_map = build(locale, split)
            out = REFS / f"fleurs-{lang}-reference-{SPLIT_NAME[split]}.json"
            out.write_text(json.dumps(ref_map, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            print(f"{locale} {split}: {len(ref_map)} ids -> {out.relative_to(REFS.parent.parent)}")


if __name__ == "__main__":
    main()
