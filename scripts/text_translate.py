"""Text-only translation baseline on the FLEURS test sets: the source transcript in, the translation out.

For a frozen ST set the manifest holds `source_text` (the reference transcript of the clip) and `target`
(the reference translation). A text-only model translates `source_text`; the output is scored with
melteval.st_metrics against `target`, per direction, cased and lowercased. Greedy decoding, one chat turn,
the instruction before the content (the same wording as format-repeatfirst.yaml).

Usage: python scripts/text_translate.py --model Qwen/Qwen3.5-2B --set /path/to/frozen-set --out DIR [--limit N]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


LANG_NAMES = {
    "bg": "Bulgarian",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "ga": "Irish",
    "hr": "Croatian",
    "hu": "Hungarian",
    "it": "Italian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mt": "Maltese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sv": "Swedish",
}
INSTRUCTION = "Translate the following content into {tgt_lang}, and write nothing else.\n\n{text}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--set", dest="frozen_set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--limit", type=int, default=0, help="clips per pair (smoke tests)")
    args = ap.parse_args()

    import sacrebleu  # noqa: F401  (fail early, before the model loads)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from melteval.spm import require_spm_model
    from melteval.st_metrics import _score_group

    require_spm_model("flores200")

    pairs: dict[str, list[dict]] = defaultdict(list)
    with open(Path(args.frozen_set) / "manifest.jsonl") as manifest:
        for line in manifest:
            rec = json.loads(line)
            pairs[rec["dataset_id"]].append(rec)
    if args.limit:
        pairs = {k: v[: args.limit] for k, v in pairs.items()}

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    over_cap = 0
    for dataset_id in sorted(pairs):
        recs = pairs[dataset_id]
        tgt = recs[0]["tgt_lang"]
        prompts = []
        for r in recs:
            msg = [{"role": "user", "content": INSTRUCTION.format(tgt_lang=LANG_NAMES[tgt], text=r["source_text"])}]
            prompts.append(
                tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            )
        order = sorted(range(len(recs)), key=lambda i: len(prompts[i]))
        hyps = [""] * len(recs)
        for s in range(0, len(order), args.batch_size):
            idx = order[s : s + args.batch_size]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(
                "cuda"
            )
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
            for i, g in zip(idx, gen[:, enc["input_ids"].shape[1] :], strict=True):
                if (g != tok.pad_token_id).sum().item() >= args.max_new_tokens:
                    over_cap += 1
                hyps[i] = tok.decode(g, skip_special_tokens=True).strip()
        refs = [r["target"] for r in recs]
        entry = {
            "dataset_id": dataset_id,
            "src_lang": recs[0]["src_lang"],
            "tgt_lang": tgt,
            "n": len(recs),
            "n_empty": sum(not h for h in hyps),
        }
        for key, lc in (("cased", False), ("lc", True)):
            values, sigs = _score_group(hyps, refs, {tgt}, "flores200", lc)
            entry[key] = values
            entry[f"{key}_signatures"] = sigs
        results.append(entry)
        with open(out_dir / f"{dataset_id}.hyp.jsonl", "w") as f:
            for r, h in zip(recs, hyps, strict=True):
                f.write(
                    json.dumps({"sample_key": r["sample_key"], "hyp": h, "ref": r["target"]}, ensure_ascii=False)
                    + "\n"
                )
        print(dataset_id, entry["n"], round(entry["cased"]["chrfpp"], 1), round(entry["lc"]["chrfpp"], 1), flush=True)
    summary = {
        "model": args.model,
        "set": args.frozen_set,
        "prompt": INSTRUCTION,
        "greedy": True,
        "max_new_tokens": args.max_new_tokens,
        "n_hit_token_cap": over_cap,
        "pairs": results,
    }
    with open(out_dir / "scores.json", "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
