"""Build the FLEURS en_us reference-text JSON that fleurs24-st-xen-*.yaml join in.

Fetches `google/fleurs` en_us `test` and `validation` from the Hub (needs
outbound internet -- nyx has it, artemis and MN5 do not) and writes
`{sentence_id: raw_transcription}` maps to configs/refs/. One-time build
step; the JSON files are committed, this script does not run at freeze time.

Uses `raw_transcription` (the corpus's own cased, punctuated transcription),
not the shar tree's `custom.pnc_text`: the PNC pass truecases each recording
of a sentence separately, so `pnc_text` differs across an id's duplicate
recordings for 57/350 test ids (05-language-ladder.md Sec 3.1, training
repo) while `raw_transcription` is one value per id straight from the
corpus. Re-run and re-verify (see the assertion below) if FLEURS' HF
metadata ever changes.
"""

import json
from pathlib import Path

from datasets import load_dataset

OUT = {
    "test": Path(__file__).parent.parent / "configs/refs/fleurs-en-reference-test.json",
    "validation": Path(__file__).parent.parent / "configs/refs/fleurs-en-reference-dev.json",
}


def main() -> None:
    for split, out_path in OUT.items():
        ds = load_dataset("google/fleurs", "en_us", split=split).select_columns(
            ["id", "raw_transcription"]
        )
        ref_map: dict[str, str] = {}
        for row in ds:
            key, text = str(row["id"]), row["raw_transcription"]
            existing = ref_map.get(key)
            if existing is not None and existing != text:
                raise ValueError(
                    f"id {key} has two different raw_transcription values in {split} "
                    f"({existing!r} vs {text!r}) -- the majority-vote fallback in "
                    "05-language-ladder.md Sec 3.1 was never implemented because this "
                    "never happened; it needs to now."
                )
            ref_map[key] = text
        out_path.write_text(
            json.dumps(ref_map, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"{split}: {len(ref_map)} ids -> {out_path}")


if __name__ == "__main__":
    main()
