"""Command line entry point: everything that is not ``inspect eval`` itself.

Running an eval is ``inspect eval`` — the harness registers its tasks, provider
and scorers with inspect, so there is no wrapper to maintain. What lives here is
the work either side of it: building frozen sets, and inspecting them.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a subcommand."""
    parser = argparse.ArgumentParser(prog="melteval", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug-level logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser(
        "freeze", help="Build a frozen evaluation set from a source spec."
    )
    freeze_parser.add_argument("spec", help="Path to the frozen-set spec YAML.")
    freeze_parser.add_argument("-o", "--out", required=True, help="Output directory.")
    freeze_parser.set_defaults(func=_cmd_freeze)

    show_parser = subparsers.add_parser("show", help="Summarise an existing frozen set.")
    show_parser.add_argument("frozen_set", help="Frozen-set directory.")
    show_parser.set_defaults(func=_cmd_show)

    rescore_parser = subparsers.add_parser(
        "rescore",
        help="Extract (src, mt, ref) triples from a completed eval log for COMET/MetricX.",
    )
    rescore_parser.add_argument("log", help="Path to a .eval log.")
    rescore_parser.add_argument("-o", "--out", required=True, help="Output JSONL path.")
    rescore_parser.set_defaults(func=_cmd_rescore)

    text_prior_parser = subparsers.add_parser(
        "text-prior",
        help="Teacher-forced NLL/BPC of a frozen set's references under a bare backbone "
        "(02-backbones.md §3) -- not an inspect task, GPU work, run via sbatch.",
    )
    text_prior_parser.add_argument("frozen_set", help="Frozen-set directory.")
    text_prior_parser.add_argument("--model", required=True, help="HF hub id or local path.")
    text_prior_parser.add_argument(
        "--chat-template-config",
        required=True,
        choices=("llama3", "chatml"),
        help="Assistant-turn boundary markers for label masking (chat_templates.py).",
    )
    text_prior_parser.add_argument(
        "--chat-template-from",
        default=None,
        help="Load the tokenizer/chat template from here instead (a base checkpoint "
        "that ships no chat template of its own).",
    )
    text_prior_parser.add_argument(
        "--task", default=None, choices=("asr", "st"), help="Restrict to one task."
    )
    text_prior_parser.add_argument("--lang", default=None, help="Restrict to one language.")
    text_prior_parser.add_argument(
        "--limit", type=int, default=None, help="Cap samples per language."
    )
    text_prior_parser.add_argument("--device", default="cuda")
    text_prior_parser.add_argument("--dtype", default="bfloat16")
    text_prior_parser.add_argument("-o", "--out", required=True, help="Output JSON path.")
    text_prior_parser.set_defaults(func=_cmd_text_prior)

    args = parser.parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return int(args.func(args))


def _cmd_freeze(args: argparse.Namespace) -> int:
    """Build a frozen set and print its summary."""
    from melteval.freeze import freeze, load_spec

    summary = freeze(load_spec(args.spec), args.out)
    print(json.dumps(_headline(summary), indent=2, ensure_ascii=False))
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    """Print the summary of an existing frozen set."""
    from melteval.manifest import SUMMARY_NAME

    summary_path = Path(args.frozen_set) / SUMMARY_NAME
    if not summary_path.exists():
        print(f"No {SUMMARY_NAME} in {args.frozen_set}", file=sys.stderr)
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(_headline(summary), indent=2, ensure_ascii=False))
    return 0


def _cmd_rescore(args: argparse.Namespace) -> int:
    """Extract rescoring triples from a log and print how many were usable."""
    from melteval.rescore import extract_triples, write_triples

    triples, stats = extract_triples(args.log)
    write_triples(triples, args.out)
    print(json.dumps(stats, indent=2))
    return 0


def _cmd_text_prior(args: argparse.Namespace) -> int:
    """Score one backbone's text-only prior against a frozen set."""
    from melteval.text_prior import run, write_results

    results = run(
        frozen_set=args.frozen_set,
        model_id=args.model,
        chat_template_config=args.chat_template_config,
        chat_template_from=args.chat_template_from,
        task=args.task,
        lang=args.lang,
        limit=args.limit,
        device=args.device,
        dtype=args.dtype,
    )
    write_results(results, args.out)
    print(json.dumps(results["overall"], indent=2))
    return 0


def _headline(summary: dict) -> dict:
    """Trim a summary to the parts worth reading in a terminal."""
    return {
        key: summary[key]
        for key in ("name", "spec_hash", "total_samples", "total_hours", "tasks", "languages")
        if key in summary
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
