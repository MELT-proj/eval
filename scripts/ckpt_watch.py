"""Logic of the checkpoint watcher: what to evaluate, what to keep.

A checkpoint is a Lightning ``.ckpt`` file whose name carries the training step
(``step=4000-last.ckpt``). The watcher keeps one JSON state file, a ledger with
one entry per checkpoint it has seen, so that restarting it never evaluates
anything twice.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


STEP_RE = re.compile(r"step=(\d+)")

#: Ledger statuses. ``skipped`` was seen but not chosen by ``every_n``.
SELECTED, SKIPPED, SUBMITTED, DONE, FAILED, DELETED = "selected", "skipped", "submitted", "done", "failed", "deleted"


@dataclass
class Checkpoint:
    """A checkpoint file on disk."""

    step: int
    path: Path
    mtime: float


@dataclass
class Entry:
    """One line of the ledger."""

    step: int
    path: str
    seq: int  #: Order of discovery (0, 1, 2, ...): what ``every_n`` counts.
    status: str
    first_seen: str = ""
    submitted_at: float = 0.0  #: Unix time of the last submission.
    note: str = ""


@dataclass
class Retention:
    """What to keep when pruning."""

    keep_last: int = 2
    keep_best_k: int = 0
    metric: str | None = None  #: ``benchmark/slice/metric``, as in results_long.csv.
    mode: str = "min"  #: ``min`` (WER) or ``max`` (accuracy, BLEU).
    keep_every: int = 0  #: Keep every checkpoint whose step is a multiple of this (0 = none).


@dataclass
class State:
    """The ledger: every checkpoint seen, by step."""

    entries: dict[int, Entry] = field(default_factory=dict)

    @property
    def next_seq(self) -> int:
        return max((e.seq for e in self.entries.values()), default=-1) + 1



def step_of(name: str) -> int | None:
    """The training step in a checkpoint's file name, or ``None``."""
    match = STEP_RE.search(name)
    return int(match.group(1)) if match else None


def find_checkpoints(ckpt_dir: str | Path, pattern: str = "step=*.ckpt") -> list[Checkpoint]:
    """Files in *ckpt_dir* matching *pattern* that carry a step, oldest step first."""
    found = []
    for path in Path(ckpt_dir).glob(pattern):
        step = step_of(path.name)
        if step is None or not path.is_file():
            continue
        found.append(Checkpoint(step, path, path.stat().st_mtime))
    return sorted(found, key=lambda c: c.step)


def is_complete(ckpt: Checkpoint, settle_seconds: float, now: float | None = None) -> bool:
    """A checkpoint counts as fully written once nothing has touched it for *settle_seconds*.

    Lightning writes the file in place, so a checkpoint being saved has a recent
    modification time and a size that is still growing.
    """
    now = time.time() if now is None else now
    return ckpt.path.stat().st_size > 0 and now - ckpt.mtime >= settle_seconds


def is_selected(seq: int, every_n: int) -> bool:
    """Whether the seq-th checkpoint discovered is evaluated: the first, then every ``every_n``-th."""
    return seq % max(every_n, 1) == 0


def model_name(step: int) -> str:
    """Name of a checkpoint's row in the report (zero-padded so it sorts by step)."""
    return f"step-{step:07d}"


def step_config(template: dict, step: int, path: Path, output_dir: Path) -> dict:
    """The ``scripts/evaluate.py`` config that evaluates one checkpoint.

    The template's first model is the prototype (its ``args``, ``instruction``
    and ``venv`` carry over); only its name and checkpoint change. Every step
    shares *output_dir*, so the report has one column per checkpoint.
    """
    config = {k: v for k, v in template.items() if k not in ("models", "benchmarks")}
    config["output_dir"] = str(output_dir)
    config["models"] = [{**template["models"][0], "name": model_name(step), "model": f"smurf/{path}"}]
    config["benchmarks"] = template["benchmarks"]
    return config



def load_state(path: Path) -> State:
    """Read the ledger; an absent file is an empty ledger."""
    if not path.exists():
        return State()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return State({int(step): Entry(**entry) for step, entry in raw["entries"].items()})


def save_state(state: State, path: Path) -> None:
    """Write the ledger atomically, so a crash cannot leave half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"entries": {str(s): asdict(e) for s, e in sorted(state.entries.items())}}, indent=1),
        encoding="utf-8",
    )
    os.replace(tmp, path)



def read_metrics(results_long: Path, metric: str) -> dict[int, float]:
    """Value of *metric* (``benchmark/slice/metric``) for each checkpoint in ``results_long.csv``, by step.

    A checkpoint with no such row, or a value that is not a number yet
    (``pending``), is left out.
    """
    if not results_long.exists():
        return {}
    benchmark, slice_name, name = metric.split("/", 2)
    values = {}
    with results_long.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            step = step_of(row["model"].replace("step-", "step="))
            if step is None or (row["benchmark"], row["slice"], row["metric"]) != (benchmark, slice_name, name):
                continue
            try:
                values[step] = float(row["value"])
            except ValueError:
                continue
    return values



def plan_retention(state: State, retention: Retention, metrics: dict[int, float]) -> tuple[dict[int, str], list[int]]:
    """Decide which checkpoints are worth keeping, and which are candidates for deletion.

    This only decides: nothing is deleted here or anywhere in the watcher.

    A checkpoint is kept if it is one of:

    * the newest one, whatever happened to it (training resumes from it);
    * one of the last ``keep_last``;
    * one of the ``keep_best_k`` with the best value of the metric;
    * a multiple of ``keep_every``;
    * not finished: selected, submitted or failed (a failed one may be retried).

    Everything else (finished being evaluated, or never selected) is a deletion
    candidate.

    Returns:
        ``({step: why kept}, [deletion candidate steps])``, over checkpoints still on disk.
    """
    present = sorted(s for s, e in state.entries.items() if e.status != DELETED)
    keep: dict[int, str] = {}
    if present:
        keep[present[-1]] = "newest"
    for step in present[-retention.keep_last :] if retention.keep_last > 0 else []:
        keep.setdefault(step, "last")
    if retention.keep_every:
        for step in present:
            if step % retention.keep_every == 0:
                keep.setdefault(step, "milestone")
    if retention.keep_best_k and retention.metric:
        scored = [s for s in present if s in metrics]
        scored.sort(key=lambda s: metrics[s], reverse=retention.mode == "max")
        for step in scored[: retention.keep_best_k]:
            keep.setdefault(step, "best")
    for step in present:
        if state.entries[step].status in (SELECTED, SUBMITTED, FAILED):
            keep.setdefault(step, state.entries[step].status)
    return keep, [s for s in present if s not in keep]
