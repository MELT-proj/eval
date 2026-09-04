#!/usr/bin/env python3
"""Compare melteval SMURF hypotheses against fbk_speechllm.inference output.

Usage:
    python scripts/compare_crosscheck.py \
        --eval-log logs/<run>.eval \
        --upstream runs/crosscheck-smurf/upstream.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from pathlib import Path

#: hyp-vs-hyp corpus WER at or below this, with no systematic flag, is a pass.
WER_PASS_THRESHOLD = 0.02


class _Palette:
    """ANSI colours, or empty strings when colour is off."""

    def __init__(self, enabled: bool) -> None:
        codes = {
            "bold": "1",
            "dim": "2",
            "red": "31",
            "green": "32",
            "yellow": "33",
            "cyan": "36",
            "reset": "0",
        }
        for name, code in codes.items():
            setattr(self, name, f"\033[{code}m" if enabled else "")

    def paint(self, text: str, *names: str) -> str:
        if not self.reset:
            return text
        return "".join(getattr(self, n) for n in names) + text + self.reset


def _want_colour(choice: str) -> bool:
    if choice == "always":
        return True
    if choice == "never":
        return False
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _norm(text: str) -> str:
    """Lowercase and collapse whitespace, for the 'normalized match' count."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _word_diff(upstream: str, melteval: str, pal: _Palette) -> str:
    """Render a word-level diff: upstream is the baseline, melteval the change.

    Words only in upstream are shown in red (dropped by melteval); words only in
    melteval in green (added); shared words plain.
    """
    a, b = upstream.split(), melteval.split()
    out: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            out.extend(a[i1:i2])
        else:
            if i1 != i2:
                out.append(pal.paint("-" + " ".join(a[i1:i2]), "red"))
            if j1 != j2:
                out.append(pal.paint("+" + " ".join(b[j1:j2]), "green"))
    return " ".join(out)


def _read_melteval(log_path: Path) -> list[tuple[str, str]]:
    """Return [(key, hypothesis), ...] from an inspect .eval log, in log order."""
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(log_path), resolve_attachments=True)
    rows: list[tuple[str, str]] = []
    n_error = 0
    for sample in log.samples or []:
        if getattr(sample, "error", None) is not None:
            n_error += 1
            continue
        key = str((sample.metadata or {}).get("cut_id") or sample.id)
        hypothesis = sample.output.completion if sample.output else ""
        rows.append((key, hypothesis or ""))
    if n_error:
        print(f"warning: {n_error} melteval sample(s) errored and are excluded.")
    return rows


def _read_upstream(jsonl_path: Path) -> list[tuple[str, str]]:
    """Return [(id, hypothesis), ...] from fbk_speechllm.inference output, in file order."""
    rows: list[tuple[str, str]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        rows.append((str(record["id"]), record.get("hypothesis", "") or ""))
    return rows


def _pair_up(
    melteval: list[tuple[str, str]], upstream: list[tuple[str, str]]
) -> list[tuple[str, str, str]]:
    """Join the two hypothesis sets. Keyed when the keys line up, else positional.

    Returns:
        [(key, hyp_melteval, hyp_upstream), ...].
    """
    m_map = dict(melteval)
    u_map = dict(upstream)
    shared = set(m_map) & set(u_map)
    if len(shared) == len(melteval) == len(upstream) and len(m_map) == len(melteval):
        return [(k, m_map[k], u_map[k]) for k, _ in melteval]

    print(
        f"warning: keys do not line up (melteval={len(melteval)}, upstream={len(upstream)}, "
        f"shared={len(shared)}); falling back to positional pairing."
    )
    if len(melteval) != len(upstream):
        raise SystemExit(
            f"cannot pair positionally: {len(melteval)} melteval rows vs {len(upstream)} upstream rows."
        )
    return [(m[0], m[1], u[1]) for m, u in zip(melteval, upstream)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-log", required=True, type=Path, help="The .eval log from step 4.")
    parser.add_argument("--upstream", required=True, type=Path, help="upstream.jsonl from step 3.")
    parser.add_argument("--max-print", type=int, default=15, help="How many differing pairs to show.")
    parser.add_argument(
        "--instruction",
        default="Transcribe this English audio:",
        help="Instruction text, for the 'echoed into the answer' check.",
    )
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    args = parser.parse_args()

    import jiwer

    pal = _Palette(_want_colour(args.color))

    pairs = _pair_up(_read_melteval(args.eval_log), _read_upstream(args.upstream))
    if not pairs:
        raise SystemExit("no pairs to compare.")

    exact = sum(1 for _, m, u in pairs if m.strip() == u.strip())
    normalized = sum(1 for _, m, u in pairs if _norm(m) == _norm(u))

    # Upstream is the "reference" side, so this reads as drift of melteval away from it. jiwer rejects an empty reference (same edge melteval's own asr_scorer excludes), so drop those pairs from the corpus figure.
    scored = [(m, u) for _, m, u in pairs if u.strip()]
    n_empty_ref = len(pairs) - len(scored)
    if scored:
        words = jiwer.process_words([u for _, u in scored], [m for m, _ in scored])
        chars = jiwer.process_characters([u for _, u in scored], [m for m, _ in scored])
        corpus_wer, corpus_cer = words.wer, chars.cer
    else:
        corpus_wer = corpus_cer = 0.0

    instr = _norm(args.instruction)
    truncations = 0
    echoes = 0
    length_outliers = 0
    per_pair: list[tuple[float, str, str, str]] = []
    for key, m, u in pairs:
        nm, nu = _norm(m), _norm(u)
        # An empty side is its own signal (counted as n_empty_ref / a missing
        # completion), not a truncation or a length blow-up -- skip it here so
        # the systematic flags stay specific.
        if nm and nu:
            if nm != nu and (nm.startswith(nu) or nu.startswith(nm)):
                truncations += 1
            if abs(len(nm) - len(nu)) / max(len(nm), len(nu)) > 0.5:
                length_outliers += 1
        if instr and nm and instr in nm and instr not in nu:
            echoes += 1
        pair_wer = jiwer.wer(u, m or "") if u.strip() else 0.0
        per_pair.append((pair_wer, key, m, u))

    def _count(label: str, n: int) -> str:
        colour = "green" if n == 0 else ("yellow" if n < len(pairs) else "red")
        return f"  {label} {pal.paint(str(n), colour, 'bold')}"

    wer_colour = "green" if corpus_wer <= WER_PASS_THRESHOLD else "red"

    print()
    print(pal.paint("── cross-check: melteval vs fbk_speechllm.inference ──", "bold", "cyan"))
    print(f"pairs compared:          {pal.paint(str(len(pairs)), 'bold')}")
    print(f"exact match:             {pal.paint(str(exact), 'bold')}  ({exact / len(pairs):.1%})")
    print(f"match after normalizing: {pal.paint(str(normalized), 'bold')}  ({normalized / len(pairs):.1%})")
    if n_empty_ref:
        print(f"excluded (empty upstream hyp): {pal.paint(str(n_empty_ref), 'yellow')}")
    print(pal.paint(
        "  (hyp-vs-hyp = the two systems' outputs against each other, no reference "
        "transcript; 0 = identical. An implementation-equivalence check, not accuracy.)",
        "dim",
    ))
    print(f"hyp-vs-hyp WER:           {pal.paint(f'{corpus_wer:.4f}', wer_colour, 'bold')}")
    print(f"hyp-vs-hyp CER:           {corpus_cer:.4f}")
    print()
    print("systematic-pattern flags:")
    print(_count("one hyp a prefix of the other (truncation):", truncations))
    print(_count("instruction echoed into the answer:        ", echoes))
    print(_count("length differs by >50%:                    ", length_outliers))

    differing = sorted((p for p in per_pair if p[0] > 0), key=lambda p: -p[0])
    if differing:
        print()
        print(f"top {min(args.max_print, len(differing))} differing pairs (upstream baseline, "
              f"{pal.paint('-dropped', 'red')} {pal.paint('+added', 'green')} by melteval):")
        for pair_wer, key, m, u in differing[: args.max_print]:
            print(f"  {pal.paint(f'[{key}]', 'cyan')} wer={pal.paint(f'{pair_wer:.3f}', 'bold')}")
            print(f"    {_word_diff(u, m, pal)}")

    systematic = truncations or echoes or length_outliers
    passed = not systematic and corpus_wer <= WER_PASS_THRESHOLD
    print()
    if passed:
        print(pal.paint(
            f"PASS: no systematic difference, hyp-vs-hyp WER {corpus_wer:.4f} <= {WER_PASS_THRESHOLD}.",
            "green", "bold",
        ))
        sys.exit(0)
    print(pal.paint(
        f"FAIL: systematic flags set ({bool(systematic)}) or hyp-vs-hyp WER {corpus_wer:.4f} "
        f"> {WER_PASS_THRESHOLD}. Inspect the pairs above before approving the branch.",
        "red", "bold",
    ))
    sys.exit(1)


if __name__ == "__main__":
    main()
