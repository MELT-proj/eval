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


def _headline(summary: dict) -> dict:
    """Trim a summary to the parts worth reading in a terminal."""
    return {
        key: summary[key]
        for key in ("name", "spec_hash", "total_samples", "total_hours", "tasks", "languages")
        if key in summary
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
