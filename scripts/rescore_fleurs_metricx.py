"""Backfill MetricX sources into the FLEURS ST logs of the speechval campaigns.

Why: the artemis FLEURS frozen sets were built before the reader could resolve
``supervisions.0.text``, so their records carry no ``source_text`` and every
MetricX score in those logs is reference-only — incomparable with CoVoST2 and
Europarl, which are scored on full ``(src, mt, ref)`` triples. This driver reads
the rebuilt sets, refuses to continue unless every sample can be given a source,
snapshots the old values, and rewrites each log in place with ``inspect score``.

It is not part of the eval path. It submits GPU work through
``infra/runners/submit_score.sh`` (see AGENTS.md) and reads the campaign configs
only to find their output directories and frozen sets.

    python scripts/rescore_fleurs_metricx.py --site artemis --dry-run
    python scripts/rescore_fleurs_metricx.py --site artemis --wait

Run it with the melteval venv's Python: it imports ``scripts/evaluate.py``,
which needs PyYAML.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
for _entry in (str(REPO_ROOT / "scripts"), str(REPO_ROOT)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import evaluate  # noqa: E402
from evaluate import run_command, wait_for_jobs  # noqa: E402

from melteval.manifest import read_manifest, resolve_frozen_set  # noqa: E402


#: The two FLEURS ST benchmarks this backfill touches.
FLEURS_BENCHMARKS = ("fleurs24-st-xen-dev", "fleurs24-st-enx-dev")

#: Campaigns to process when ``--campaign`` is not given.
DEFAULT_CAMPAIGNS = (
    "configs/eval/speechval-fondue.yaml",
    "configs/eval/speechval-canary.yaml",
    "configs/eval/speechval-qwen3-omni.yaml",
)

#: The scorers every log in scope records. ``inspect score`` with no ``--scorer`` rebuilds them all;
#: an unexpected third scorer would be re-run too, so the driver refuses instead.
EXPECTED_SCORERS = {"st_scorer", "metricx_scorer"}

#: Metric the new scorer adds. A log that has it has been through this driver.
NEW_METRIC = "metricx_source_coverage"

#: Must match ``melteval.rescore.SOURCE_MANIFEST_ENV`` (a test pins the two together).
SOURCE_MANIFEST_ENV = "MELTEVAL_SOURCE_MANIFEST"

COVERAGE_FIELDS = ("campaign", "model", "benchmark", "slice", "samples", "with_source", "coverage")
BEFORE_FIELDS = (
    "campaign", "model", "benchmark", "slice",
    "metricx", "metricx_coverage", "corpus_bleu", "corpus_spbleu", "corpus_chrf",
)  # fmt: skip
RESCORE_FIELDS = (
    "campaign", "model", "benchmark", "slice",
    "metricx_before", "metricx_after", "delta", "bleu_before", "bleu_after", "source_coverage",
)  # fmt: skip
BEFORE_NUMERIC = ("metricx", "metricx_coverage", "corpus_bleu", "corpus_spbleu", "corpus_chrf")


@dataclass(frozen=True)
class LogRef:
    """One current log of one (campaign, model, benchmark, slice)."""

    campaign: str
    model: str
    benchmark: str
    slice: str
    path: Path

    @property
    def key(self) -> str:
        return f"{self.campaign}:{self.model}:{self.benchmark}:{self.slice}"


class CoverageError(RuntimeError):
    """Some sample of some slice cannot be given a source text."""

    def __init__(self, rows: list[dict], message: str):
        super().__init__(message)
        self.rows = rows


class VerificationError(RuntimeError):
    """A rescored log does not look like the driver's work."""


def discover(output_dir: Path, campaign: str, benchmarks: set[str]) -> list[LogRef]:
    """Every current log under ``<output_dir>/logs/<model>/<benchmark>/<slice>/`` for *benchmarks*."""
    roots = sorted((output_dir / "logs").glob("*/*/*"))
    refs: list[LogRef] = []
    for slice_dir in roots:
        benchmark, model = slice_dir.parent.name, slice_dir.parent.parent.name
        if benchmark not in benchmarks or not slice_dir.is_dir():
            continue
        logs = sorted(slice_dir.glob("*.json"))
        if len(logs) > 1:
            raise SystemExit(f"{slice_dir}: {len(logs)} current logs, expected one: {[p.name for p in logs]}")
        if logs:
            refs.append(LogRef(campaign, model, benchmark, slice_dir.name, logs[0]))
    return refs


def load_manifest_index(path: str | Path) -> dict[tuple[str, str], str]:
    """``(dataset_id, sample_key) -> source_text`` for a frozen set.

    The dataset id is part of the key because FLEURS sample keys repeat across the X->en and en->X
    sets: a key-only lookup finds a record for every sample and scores it against the wrong language.
    """
    manifest = resolve_frozen_set(path)
    return {
        (record.dataset_id, record.sample_key): (record.source_text or "")
        for record in read_manifest(manifest)
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(log: dict) -> dict[str, dict[str, float | None]]:
    """``results.scores`` as ``{scorer: {metric: value}}``."""
    return {
        score.get("name"): {name: metric.get("value") for name, metric in (score.get("metrics") or {}).items()}
        for score in (log.get("results") or {}).get("scores") or []
    }


def _scorer_names(log: dict) -> set[str]:
    return {scorer.get("name") for scorer in (log.get("eval") or {}).get("scorers") or []}


def already_rescored(path: Path) -> bool:
    """Whether *path* already carries this driver's new metric."""
    return NEW_METRIC in (_metrics(_load(path)).get("metricx_scorer") or {})


def coverage_rows(refs: list[LogRef], manifests: dict[str, Path]) -> list[dict]:
    """Per-slice source coverage; raises :class:`CoverageError` if any slice is below 100%."""
    indexes: dict[str, dict[tuple[str, str], str]] = {}
    rows: list[dict] = []
    for ref in refs:
        index = indexes.get(ref.benchmark)
        if index is None:
            index = indexes[ref.benchmark] = load_manifest_index(manifests[ref.benchmark])
        samples = _load(ref.path).get("samples") or []
        with_source = 0
        for sample in samples:
            metadata = sample.get("metadata") or {}
            if index.get((str(metadata.get("dataset_id") or ""), str(sample.get("id")))):
                with_source += 1
        rows.append(
            {
                "campaign": ref.campaign,
                "model": ref.model,
                "benchmark": ref.benchmark,
                "slice": ref.slice,
                "samples": len(samples),
                "with_source": with_source,
                "coverage": with_source / len(samples) if samples else 0.0,
            }
        )
    missing = [row for row in rows if row["coverage"] < 1.0]
    if missing:
        detail = ", ".join(
            f"{row['model']} {row['benchmark']}/{row['slice']}: {row['with_source']}/{row['samples']}"
            for row in missing
        )
        raise CoverageError(rows, f"source coverage below 100%: {detail}")
    return rows


def snapshot(refs: list[LogRef]) -> tuple[list[dict], list[dict]]:
    """The old per-slice values and per-sample triples, so the rewrite can be audited."""
    rows: list[dict] = []
    triples: list[dict] = []
    for ref in refs:
        log = _load(ref.path)
        metrics = _metrics(log)
        metricx = metrics.get("metricx_scorer") or {}
        st = metrics.get("st_scorer") or {}
        rows.append(
            {
                "campaign": ref.campaign,
                "model": ref.model,
                "benchmark": ref.benchmark,
                "slice": ref.slice,
                "metricx": metricx.get("metricx"),
                "metricx_coverage": metricx.get("metricx_coverage"),
                "corpus_bleu": st.get("corpus_bleu"),
                "corpus_spbleu": st.get("corpus_spbleu"),
                "corpus_chrf": st.get("corpus_chrf"),
            }
        )
        for sample in log.get("samples") or []:
            score = (sample.get("scores") or {}).get("metricx_scorer") or {}
            triple = (score.get("metadata") or {}).get("metricx_triple")
            if triple:
                triples.append(
                    {
                        "campaign": ref.campaign,
                        "model": ref.model,
                        "benchmark": ref.benchmark,
                        "slice": ref.slice,
                        "sample_id": sample.get("id"),
                        **triple,
                    }
                )
    return rows, triples


def verify(refs: list[LogRef], before: list[dict]) -> list[dict]:
    """Check every rescored log against the snapshot; returns the old/new report rows.

    Raises:
        VerificationError: with every failure listed, not just the first.
    """
    snapshots = {(row["campaign"], row["model"], row["benchmark"], row["slice"]): row for row in before}
    rows: list[dict] = []
    failures: list[str] = []
    for ref in refs:
        log = _load(ref.path)
        metrics = _metrics(log)
        label = f"{ref.model} {ref.benchmark}/{ref.slice}"
        if set(metrics) != EXPECTED_SCORERS:
            failures.append(f"{label}: results.scores is {sorted(metrics)}, expected {sorted(EXPECTED_SCORERS)}")
            continue
        metricx = metrics["metricx_scorer"]
        coverage = metricx.get(NEW_METRIC)
        if coverage != 1.0:
            failures.append(f"{label}: source coverage {coverage!r}, expected 1.0")
        snapshot_row = snapshots.get((ref.campaign, ref.model, ref.benchmark, ref.slice))
        if snapshot_row is None:
            failures.append(f"{label}: no row in metricx-before.csv")
            continue
        for name in ("corpus_bleu", "corpus_spbleu", "corpus_chrf"):
            expected = snapshot_row.get(name)
            if expected is None:
                continue
            actual = (metrics.get("st_scorer") or {}).get(name)
            if actual != expected:
                failures.append(f"{label}: {name} changed from {expected!r} to {actual!r}")
        values: list[float] = []
        for sample in log.get("samples") or []:
            completion = (sample.get("output") or {}).get("completion") or ""
            if not completion.strip():
                continue
            score = ((sample.get("scores") or {}).get("metricx_scorer") or {}).get("value")
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not 0.0 <= score <= 25.0:
                failures.append(f"{label}: sample {sample.get('id')} has no per-sample MetricX ({score!r})")
                continue
            values.append(float(score))
        after = metricx.get("metricx")
        if values and after is not None and abs(sum(values) / len(values) - after) > 1e-9:
            failures.append(f"{label}: metricx {after!r} is not the mean of its per-sample scores")
        rows.append(
            {
                "campaign": ref.campaign,
                "model": ref.model,
                "benchmark": ref.benchmark,
                "slice": ref.slice,
                "metricx_before": snapshot_row.get("metricx"),
                "metricx_after": after,
                "delta": (
                    after - snapshot_row["metricx"]
                    if after is not None and snapshot_row.get("metricx") is not None
                    else None
                ),
                "bleu_before": snapshot_row.get("corpus_bleu"),
                "bleu_after": (metrics.get("st_scorer") or {}).get("corpus_bleu"),
                "source_coverage": coverage,
            }
        )
    if failures:
        raise VerificationError("\n".join(failures))
    return rows


def submit(groups: list[list[LogRef]], site: str, manifests: dict[str, Path], venv: str) -> list[str]:
    """Submit one scoring job per group; returns the SLURM job ids."""
    job_ids: list[str] = []
    for group in groups:
        first = group[0]
        command, env = command_for(
            site,
            [ref.path for ref in group],
            manifests[first.benchmark],
            venv,
            f"melteval:rescore:{first.model}:{first.benchmark}",
            first.path.parent / "score-slurm-%j.out",
        )
        code, out = run_command(command, env)
        if code:
            raise SystemExit(f"submission failed for {first.model} {first.benchmark}: {out}")
        found = re.findall(r"Submitted batch job (\d+)", out)
        job_ids += found
        print(f"submitted {len(group)} log(s) for {first.model} {first.benchmark}: {', '.join(found) or 'no job id'}")
    return job_ids


def command_for(
    site: str, logs: list[Path], manifest: Path, venv: str, tag: str, slurm_out: Path
) -> tuple[list[str], dict[str, str]]:
    """The ``submit_score.sh`` argv and environment for one group of logs.

    No ``--scorer``: inspect then rebuilds every scorer the log recorded, so ``st_scorer`` and
    ``metricx_scorer`` are recomputed together and the log cannot lose its BLEU.
    """
    argv = [
        "infra/runners/submit_score.sh", site, *[str(path) for path in logs], "--",
        "--model", "mockllm/model", "--action", "overwrite",
    ]  # fmt: skip
    env = {
        "VENV_PATH": venv,
        SOURCE_MANIFEST_ENV: str(manifest),
        "MELT_JOB_TAG": tag,
        "MELT_SLURM_OUT": str(slurm_out),
    }
    return argv, env


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def _read_csv(path: Path, numeric: tuple[str, ...] = ()) -> list[dict]:
    """Rows of *path* (empty if missing); *numeric* fields come back as floats, empties as None."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for field in numeric:
            if field in row:
                row[field] = float(row[field]) if row[field] not in ("", None) else None
    return rows


def _fleurs_benchmarks(config: dict) -> dict[str, Path]:
    """``{benchmark: frozen-set dir}`` for the FLEURS ST benchmarks of *config*."""
    found: dict[str, Path] = {}
    for bench in config["benchmarks"]:
        name = bench.get("name")
        if name not in FLEURS_BENCHMARKS:
            continue
        if not bench.get("frozen_set"):
            raise SystemExit(f"{name}: has no frozen_set, only a live spec; cannot resolve a source manifest")
        path, problem = evaluate.resolve_path(bench["frozen_set"], f"{name} frozen set")
        if path is None:
            raise SystemExit(problem)
        found[name] = Path(path)
    return found


def _groups(refs: list[LogRef]) -> list[list[LogRef]]:
    """One group per (campaign, model, benchmark): one SLURM job, the model loaded per log."""
    grouped: dict[tuple[str, str, str], list[LogRef]] = {}
    for ref in refs:
        grouped.setdefault((ref.campaign, ref.model, ref.benchmark), []).append(ref)
    return [grouped[key] for key in sorted(grouped)]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", default="artemis", help="site file whose scheduler args and venvs to use")
    parser.add_argument("--campaign", action="append", default=None, help=f"campaign config (default: {DEFAULT_CAMPAIGNS})")
    parser.add_argument("--out", default=None, help="override each config's output_dir")
    parser.add_argument("--dry-run", action="store_true", help="write the reports and print the jobs; submit nothing")
    parser.add_argument("--max-logs", type=int, default=0, help="cap the logs handled (one smoke slice)")
    parser.add_argument("--wait", action="store_true", help="wait for the scoring jobs, then verify")
    parser.add_argument("--verify-only", action="store_true", help="verify the existing logs against the snapshot")
    parser.add_argument("--force", action="store_true", help="rescore logs that already have the new metric")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    evaluate.load_site_env(args.site)

    for campaign_path in args.campaign or DEFAULT_CAMPAIGNS:
        config = evaluate.load_config(Path(campaign_path), args.out)
        campaign = str(config.get("name") or Path(campaign_path).stem)
        benchmarks = _fleurs_benchmarks(config)
        if not benchmarks:
            continue
        output_dir: Path = config["_output_dir"]
        results = output_dir / "results"
        all_refs = discover(output_dir, campaign, set(benchmarks))
        if not all_refs:
            continue

        snapshot_path = results / "metricx-before.csv"
        if args.verify_only:
            before = _read_csv(snapshot_path, BEFORE_NUMERIC)
            if not before:
                raise SystemExit(f"{campaign_path}: no {snapshot_path}; run the backfill first")
            rows = verify(all_refs, before)
            _write_csv(results / "metricx-rescore.csv", rows, RESCORE_FIELDS)
            print(f"{campaign_path}: verified {len(rows)} log(s)")
            continue

        refs = [ref for ref in all_refs if args.force or not already_rescored(ref.path)]
        if args.max_logs:
            refs = refs[: args.max_logs]
        if not refs:
            print(f"{campaign_path}: nothing to score")
            continue

        try:
            rows = coverage_rows(refs, benchmarks)
        except CoverageError as error:
            _write_csv(results / "fleurs-source-coverage.csv", error.rows, COVERAGE_FIELDS)
            raise SystemExit(str(error)) from error
        _write_csv(results / "fleurs-source-coverage.csv", rows, COVERAGE_FIELDS)

        # Keep the first value seen: a partial run followed by a full one must not overwrite a
        # not-yet-rescored log's old numbers with new ones.
        before = _read_csv(snapshot_path, BEFORE_NUMERIC)
        known = {(row["campaign"], row["model"], row["benchmark"], row["slice"]) for row in before}
        fresh, triples = snapshot(
            [ref for ref in all_refs if (ref.campaign, ref.model, ref.benchmark, ref.slice) not in known]
        )
        if fresh:
            _write_csv(snapshot_path, before + fresh, BEFORE_FIELDS)
        if triples:
            _append_jsonl(results / "metricx-before.jsonl", triples)
        print(f"{campaign_path}: {len(refs)} log(s) to score, source coverage 100%")

        venv = _post_venv(config)
        groups = _groups(refs)
        if args.dry_run:
            for group in groups:
                first = group[0]
                command, env = command_for(
                    args.site,
                    [ref.path for ref in group],
                    benchmarks[first.benchmark],
                    venv,
                    f"melteval:rescore:{first.model}:{first.benchmark}",
                    first.path.parent / "score-slurm-%j.out",
                )
                print(f"{SOURCE_MANIFEST_ENV}={env[SOURCE_MANIFEST_ENV]} {' '.join(command)}")
            continue

        job_ids = submit(groups, args.site, benchmarks, venv)
        if args.wait:
            wait_for_jobs(job_ids)
            rescored = verify(all_refs, _read_csv(snapshot_path, BEFORE_NUMERIC))
            _write_csv(results / "metricx-rescore.csv", rescored, RESCORE_FIELDS)
            print(f"{campaign_path}: verified {len(rescored)} log(s)")


def _post_venv(config: dict) -> str:
    """The metrics venv the FLEURS benchmarks score in (they all name the same ``post_venv``)."""
    for bench in config["benchmarks"]:
        if bench.get("name") in FLEURS_BENCHMARKS and bench.get("post_venv"):
            venv = evaluate.expand(bench["post_venv"])
            if venv:
                return venv
    raise SystemExit("the FLEURS benchmarks name no usable post_venv (is METRICS_VENV_PATH unset?)")


if __name__ == "__main__":
    main()
