"""Watch a folder of SMURF checkpoints: evaluate each new one, keep track, and list the ones not worth keeping.

    python scripts/watch_checkpoints.py run     CONFIG [--interval 300 | --once]   # the loop
    python scripts/watch_checkpoints.py status  CONFIG                             # the ledger, as a table
    python scripts/watch_checkpoints.py enqueue CONFIG STEP                        # evaluate this one whatever every_n says
    python scripts/watch_checkpoints.py candidates CONFIG                          # what could be deleted (never deletes)
    python scripts/watch_checkpoints.py rerun-baselines CONFIG                     # evaluate the baselines again
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import ckpt_watch as cw  # noqa: E402  (next to this script)


#: Seconds after a submission during which a job with no log yet is assumed still starting.
SUBMIT_GRACE = 600


def load_evaluate():
    """``scripts/evaluate.py`` as a module: its planning code says which jobs a config means and how they stand."""
    spec = importlib.util.spec_from_file_location("evaluate", REPO_ROOT / "scripts" / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate"] = module
    spec.loader.exec_module(module)
    return module


def expand(value: str) -> str:
    """Expand ``${VAR}`` in a path; an unset variable is an error, not a folder called ``${VAR}``."""
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise SystemExit(f"{value!r} uses an environment variable that is not set (site file or shell).")
    return expanded


class Watch:
    """A watch config, its ledger and the paths derived from them."""

    def __init__(self, path: Path):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.site = raw["site"]
        # The site file defines ${EVAL_ROOT} and friends, so it is loaded before any path is expanded.
        self.evaluate = load_evaluate()
        try:
            self.evaluate.load_site_env(self.site)
        except subprocess.CalledProcessError:
            raise SystemExit(
                f"infra/sites/{self.site}.sh failed to load. On JUPITER: export JUPITER_ACCOUNT=<project> and "
                f"run `jutil env activate -p <project>` first (it defines $PROJECT and $SCRATCH)."
            ) from None
        self.ckpt_dir = Path(expand(raw["ckpt_dir"])).expanduser()
        self.pattern = raw.get("pattern", "step=*.ckpt")
        self.every_n = int(raw.get("every_n", 1))
        self.settle = float(raw.get("settle_minutes", 10)) * 60
        self.max_in_flight = int(raw.get("max_in_flight", 2))
        self.output_dir = Path(expand(raw["output_dir"])).expanduser().resolve()
        self.template = yaml.safe_load((REPO_ROOT / raw["eval_config"]).read_text(encoding="utf-8"))
        # Optional: fixed models evaluated once, on the same benchmarks, into the same report.
        self.baselines_config = raw.get("baselines_config")
        self.baselines_models = list(raw.get("baselines_models") or [])
        r = raw.get("retention") or {}
        best = r.get("keep_best") or {}
        self.retention = cw.Retention(
            keep_last=int(r.get("keep_last", 2)),
            keep_best_k=int(best.get("k", 0)),
            metric=best.get("metric"),
            mode=best.get("mode", "min"),
            keep_every=int(r.get("keep_every", 0)),
        )
        self.state_path = self.output_dir / "watch_state.json"
        self.state = cw.load_state(self.state_path)

    def save(self) -> None:
        cw.save_state(self.state, self.state_path)

    def metrics(self) -> dict[int, float]:
        """The retention metric by step. Warns if it is set but nothing matches it (a misspelt name)."""
        if not self.retention.metric:
            return {}
        values = cw.read_metrics(self.output_dir / "results" / "results_long.csv", self.retention.metric)
        if not values and any(e.status == cw.DONE for e in self.state.entries.values()):
            print(
                f"WARNING: no value for retention metric {self.retention.metric!r} in results/results_long.csv; "
                f"keep_best is doing nothing. Check the name with: grep -i <benchmark> results/results_long.csv"
            )
        return values

    def config_for(self, entry: cw.Entry) -> Path:
        """Write the evaluate.py config for this checkpoint; return its path."""
        config = cw.step_config(self.template, entry.step, Path(entry.path), self.output_dir)
        path = self.output_dir / "configs" / f"{cw.model_name(entry.step)}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path


    def baselines_path(self) -> Path:
        """Write the evaluate.py config for the baselines; return its path."""
        source = yaml.safe_load((REPO_ROOT / self.baselines_config).read_text(encoding="utf-8"))
        config = cw.baselines_config(self.template, source["models"], self.baselines_models, self.output_dir)
        path = self.output_dir / "configs" / "baselines.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path

    def has_baselines(self) -> bool:
        return bool(self.baselines_config and self.baselines_models)


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)



def discover(watch: Watch) -> None:
    """Add every newly finished checkpoint to the ledger, and decide whether it is evaluated."""
    for ckpt in cw.find_checkpoints(watch.ckpt_dir, watch.pattern):
        if ckpt.step in watch.state.entries:
            continue
        if not cw.is_complete(ckpt, watch.settle):
            log(f"step {ckpt.step}: still being written, waiting")
            continue
        seq = watch.state.next_seq
        selected = cw.is_selected(seq, watch.every_n)
        watch.state.entries[ckpt.step] = cw.Entry(
            ckpt.step, str(ckpt.path.resolve()), seq, cw.SELECTED if selected else cw.SKIPPED, now()
        )
        log(f"step {ckpt.step}: new ({'evaluate' if selected else 'skipped by every_n'})")


def submit(watch: Watch) -> None:
    """Start the jobs of selected checkpoints, at most ``max_in_flight`` checkpoints at a time."""
    entries = watch.state.entries.values()
    in_flight = sum(e.status == cw.SUBMITTED for e in entries)
    for entry in sorted((e for e in entries if e.status == cw.SELECTED), key=lambda e: e.step):
        if in_flight >= watch.max_in_flight:
            break
        if not Path(entry.path).exists():
            entry.status, entry.note = cw.FAILED, "checkpoint file is gone"
            continue
        config = watch.config_for(entry)
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate.py"),
            "run",
            str(config),
            "--site",
            watch.site,
        ]
        log(f"step {entry.step}: submitting")
        code = subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
        entry.status, entry.note = (cw.SUBMITTED, "") if code == 0 else (cw.FAILED, f"evaluate.py run exited {code}")
        entry.submitted_at = time.time()
        in_flight += code == 0


def submit_baselines(watch: Watch) -> None:
    """Start the baselines' jobs, once. A finished slice is never rerun; a failed one waits for ``rerun-baselines``."""
    base = watch.state.baselines
    if not watch.has_baselines() or base.status:
        return
    command = [
        sys.executable, str(REPO_ROOT / "scripts" / "evaluate.py"), "run", str(watch.baselines_path()),
        "--site", watch.site,
    ]  # fmt: skip
    log(f"baselines ({', '.join(watch.baselines_models)}): submitting")
    code = subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
    base.status, base.note = ("submitted", "") if code == 0 else (cw.FAILED, f"evaluate.py run exited {code}")
    base.submitted_at = time.time()


def settle(states: list[str], submitted_at: float) -> str | None:
    """``done``/``failed`` once the jobs of a submission have ended, ``None`` while they are still going."""
    if all(s == "done" for s in states):
        return cw.DONE
    if "queued" not in states and ("missing" not in states or time.time() - submitted_at > SUBMIT_GRACE):
        return cw.FAILED
    return None


def refresh_baselines(watch: Watch, evaluate) -> bool:
    """Like ``refresh``, for the baselines; return whether they finished."""
    base = watch.state.baselines
    if base.status != "submitted":
        return False
    jobs, _ = evaluate.plan(evaluate.load_config(watch.baselines_path(), None))
    queued = evaluate.queued_tags()
    states = [evaluate.job_state(job, queued)[0] for job in jobs]
    outcome = settle(states, base.submitted_at) if jobs else cw.FAILED
    if outcome == cw.FAILED:
        base.note = f"jobs failed: {', '.join(sorted(set(states) - {'done'})) or 'none planned'} (see slurm-*.out)"
        log(f"baselines: FAILED ({base.note})")
    elif outcome == cw.DONE:
        log("baselines: evaluation done")
    if outcome:
        base.status = outcome
    return outcome == cw.DONE


def refresh(watch: Watch, evaluate) -> bool:
    """Look at the jobs of submitted checkpoints; return whether any finished (the report needs rebuilding)."""
    changed = False
    queued = evaluate.queued_tags()
    for entry in watch.state.entries.values():
        if entry.status != cw.SUBMITTED:
            continue
        jobs, _ = evaluate.plan(evaluate.load_config(watch.config_for(entry), None))
        states = [evaluate.job_state(job, queued)[0] for job in jobs]
        if not jobs:
            entry.status, entry.note = cw.FAILED, "no job could be planned (see evaluate.py plan)"
        else:
            outcome = settle(states, entry.submitted_at)
            if outcome == cw.DONE:
                entry.status, changed = cw.DONE, True
                log(f"step {entry.step}: evaluation done")
            elif outcome == cw.FAILED:
                # A job that just left the queue may not have its log yet, hence the
                # grace period (in settle) for the ones with no log at all.
                entry.status = cw.FAILED
                entry.note = (
                    f"jobs failed: {', '.join(sorted(set(states) - {'done'}))} (see slurm-*.out in the slice folders)"
                )
                log(f"step {entry.step}: FAILED ({entry.note})")
    return changed


def report(watch: Watch) -> None:
    """Rebuild <output_dir>/results (one column per checkpoint)."""
    config = next(iter(sorted((watch.output_dir / "configs").glob("*.yaml"))), None)
    if config is None:
        return
    code = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "evaluate.py"), "report", str(config)], cwd=REPO_ROOT, check=False
    ).returncode
    if code:
        log("WARNING: the report could not be built")


def cycle(watch: Watch, evaluate) -> None:
    """One pass: discover new checkpoints, look at running jobs, submit what is waiting."""
    discover(watch)
    submit_baselines(watch)
    # Both run: `or` would skip the baselines' check whenever a checkpoint finished.
    finished = [refresh(watch, evaluate), refresh_baselines(watch, evaluate)]
    if any(finished):
        report(watch)
    submit(watch)
    watch.save()


def cmd_run(watch: Watch, args) -> None:
    """The loop (``--once`` for a single pass)."""
    evaluate = watch.evaluate
    watch.output_dir.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            cycle(watch, evaluate)
        except Exception as exc:  # noqa: BLE001 - a transient error (squeue, filesystem) must not end the watch
            log(f"ERROR in this pass, will retry: {exc!r}")
        if args.once:
            return
        time.sleep(args.interval)



def cmd_status(watch: Watch, args) -> None:
    """Print the ledger, with the retention metric and what pruning would do."""
    metrics = watch.metrics()
    keep, delete = cw.plan_retention(watch.state, watch.retention, metrics)
    print(f"{'step':>8}  {'status':<10} {'metric':>10}  keep?")
    for step, entry in sorted(watch.state.entries.items()):
        value = f"{metrics[step]:.4f}" if step in metrics else "-"
        verdict = "" if entry.status == cw.DELETED else (keep.get(step) or "candidate")
        print(f"{step:>8}  {entry.status:<10} {value:>10}  {verdict}  {entry.note}")
    if watch.has_baselines():
        base = watch.state.baselines
        print(f"\nbaselines ({', '.join(watch.baselines_models)}): {base.status or 'pending'}  {base.note}")
    print(f"\n{len(keep)} kept, {len(delete)} deletion candidate(s). Ledger: {watch.state_path}")


def cmd_enqueue(watch: Watch, args) -> None:
    """Evaluate one checkpoint whatever every_n says (e.g. the last one of a run)."""
    entry = watch.state.entries.get(args.step)
    if entry is None:
        found = {c.step: c for c in cw.find_checkpoints(watch.ckpt_dir, watch.pattern)}
        if args.step not in found:
            raise SystemExit(f"No checkpoint for step {args.step} in {watch.ckpt_dir}.")
        seq = watch.state.next_seq
        entry = cw.Entry(args.step, str(found[args.step].path.resolve()), seq, cw.SKIPPED, first_seen=now())
        watch.state.entries[args.step] = entry
    if entry.status in (cw.SKIPPED, cw.FAILED):
        entry.status, entry.note = cw.SELECTED, ""
    watch.save()
    print(f"step {args.step}: {entry.status}; the loop will pick it up.")


def cmd_rerun_baselines(watch: Watch, args) -> None:
    """Evaluate the baselines again: slices that already have a successful log are left alone."""
    watch.state.baselines = cw.Baselines()
    watch.save()
    print("baselines: pending; the loop will submit what has no successful log.")


def cmd_candidates(watch: Watch, args) -> None:
    """List the checkpoints the retention policy would not keep. Never deletes anything.

    Writes ``<output_dir>/delete-candidates.txt`` with one ``rm`` line per
    checkpoint, to be run by hand after looking. Files already gone from disk
    are just marked as deleted in the ledger.
    """
    gone = [e for e in watch.state.entries.values() if e.status != cw.DELETED and not Path(e.path).exists()]
    for entry in gone:
        entry.status, entry.note = cw.DELETED, f"file gone {now()}"
    if gone:
        watch.save()
    keep, delete = cw.plan_retention(watch.state, watch.retention, watch.metrics())
    root = watch.ckpt_dir.resolve()
    lines, total = [], 0
    for step in delete:
        path = Path(watch.state.entries[step].path)
        if root not in path.resolve().parents:
            print(f"ignored step {step}: {path} is outside {root}")
            continue
        size = path.stat().st_size
        total += size
        print(f"step {step:>8}  {size / 1e9:6.1f} GB  {path}")
        lines.append(f"rm -- '{path}'")
    out = watch.output_dir / "delete-candidates.txt"
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"\n{len(lines)} candidate(s), {total / 1e9:.1f} GB. Kept {len(keep)}: "
          + ", ".join(f"{s} ({why})" for s, why in sorted(keep.items())))  # fmt: skip
    print(f"Nothing was deleted. Commands to run yourself, once you have looked: {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status", "enqueue", "candidates", "rerun-baselines"):
        p = sub.add_parser(name)
        p.add_argument("config", type=Path)
        if name == "run":
            p.add_argument("--interval", type=int, default=300, help="Seconds between passes.")
            p.add_argument("--once", action="store_true", help="One pass, then exit.")
        if name == "enqueue":
            p.add_argument("step", type=int)
    args = parser.parse_args(argv)
    commands = {
        "run": cmd_run,
        "status": cmd_status,
        "enqueue": cmd_enqueue,
        "candidates": cmd_candidates,
        "rerun-baselines": cmd_rerun_baselines,
    }
    commands[args.command](Watch(args.config), args)


if __name__ == "__main__":
    main()
