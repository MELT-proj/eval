"""Run a pack of eval slices on one GPU, loading the model only once.

    python -m melteval.pack MANIFEST [--worker-id ID]

Started by ``infra/run_eval_pack.sbatch``, one process per GPU, all on the same
manifest (written by ``scripts/evaluate.py run`` on sites that pack slices).
Each worker takes the next unclaimed slice, runs it, and repeats until none are
left, so one long slice does not hold up the other GPUs.

Each slice runs the same ``inspect eval`` as ``infra/run_eval.sbatch`` (same
task, arguments and log directory), but inside this process. inspect caches the
loaded model by name and arguments (``get_model()``) and ``eval()`` never closes
it (inspect_ai 0.3.254), so the first slice loads the checkpoint and the rest
reuse it. For this to work, ``scripts/evaluate.py`` only packs slices with
identical model arguments (``-M`` and flags like ``--max-tokens``); different
ones would load a second copy onto the same GPU.

A slice that fails, or is cut off by the time limit, is skipped. Its log (or
missing log) makes the next ``evaluate.py run`` retry it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

from melteval.providers import family_of


#: The task every slice runs, relative to the repo root (as in infra/run_eval.sbatch).
TASK = "melteval/tasks.py@speech"


def inspect_argv(model: str, entry: dict) -> list[str]:
    """The ``inspect`` command line for one manifest entry, as ``infra/run_eval.sbatch`` builds it.

    Keep the two in step: the evaluation-set argument, the default
    ``prompt_style`` from the model's family, and ``--fail-on-error`` for
    off-the-shelf baselines.
    """
    eval_set = entry["eval_set"]
    if Path(eval_set).is_dir():
        eval_set_arg = ["-T", f"frozen_set={eval_set}"]
    elif Path(eval_set).is_file() and eval_set.endswith((".yaml", ".yml")):
        eval_set_arg = ["-T", f"spec={eval_set}"]
    else:
        raise FileNotFoundError(f"evaluation set not found (or not a recognised kind): {eval_set}")

    args = [str(a) for a in entry["args"]]
    family = family_of(model)
    if family is None:
        raise ValueError(f"{model!r} is not a melt/..., melt/hf/... or smurf/... model")
    joined = f" {' '.join(args)} "
    if "prompt_style=" not in joined:
        args += ["-T", f"prompt_style={family}"]
    if family == "hf" and "fail-on-error" not in joined:
        args += ["--fail-on-error", os.environ.get("FAIL_ON_ERROR", "0.05")]
    return ["eval", TASK, "--model", model, "--log-dir", entry["log_dir"], *eval_set_arg, *args]


def claim(claims_dir: Path, index: int) -> bool:
    """Atomically claim slice *index* for this worker; ``False`` if another one has it."""
    try:
        os.close(os.open(claims_dir / str(index), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


@contextlib.contextmanager
def output_to(path: Path):
    """Send this process's stdout and stderr (Python and native) to *path* while the block runs."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with path.open("a", encoding="utf-8") as handle:
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def run_slice(argv: list[str]) -> int:
    """Run ``inspect <argv>`` in this process; return its exit code (raises on a crash)."""
    from inspect_ai._cli.main import inspect as inspect_cli  # type: ignore # the `inspect` console script's own group

    result = inspect_cli.main(args=argv, prog_name="inspect", standalone_mode=False)
    return result if isinstance(result, int) else 0


def log_status(log_dir: Path, since: float) -> str:
    """Status of the newest inspect log written to *log_dir* after *since* (wall clock), or ``"no log"``.

    The CLI exits 0 for a run that ended in ``error`` too; the log is what says.
    """
    from inspect_ai.log import read_eval_log # type: ignore

    logs = [p for p in [*log_dir.glob("*.eval"), *log_dir.glob("*.json")] if p.stat().st_mtime >= since]
    if not logs:
        return "no log"
    newest = max(logs, key=lambda p: p.stat().st_mtime)
    try:
        return read_eval_log(str(newest), header_only=True).status
    except Exception as exc:  # noqa: BLE001 - a half-written log is just "not done"
        return f"unreadable log ({exc})"


def run(manifest_path: Path, worker_id: str) -> int:
    """Claim and run slices from the manifest until none is left; return how many failed."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model = manifest["model"]
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    claims_dir = manifest_path.with_suffix("").with_name(f"{manifest['pack_id']}.{job_id}.claims")
    claims_dir.mkdir(parents=True, exist_ok=True)

    def say(message: str) -> None:
        print(f"[pack {worker_id}] {time.strftime('%H:%M:%S')} {message}", flush=True)

    say(f"{model}, {len(manifest['slices'])} slice(s), GPU {os.environ.get('CUDA_VISIBLE_DEVICES', '?')}")
    failed = 0
    for index, entry in enumerate(manifest["slices"]):
        if not claim(claims_dir, index):
            continue
        log_dir = Path(entry["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        out = log_dir / f"pack-{job_id}-{worker_id}.out"
        say(f"start {entry['label']} -> {out}")
        started, wall = time.monotonic(), time.time()
        try:
            with output_to(out):
                argv = inspect_argv(model, entry)
                print(f"[pack {worker_id}] inspect {' '.join(argv)}", flush=True)
                code = run_slice(argv)
        except BaseException as exc:  # noqa: BLE001 - one bad slice must not end the pack
            if isinstance(exc, KeyboardInterrupt):
                raise
            code = 1
            with out.open("a", encoding="utf-8") as handle:
                traceback.print_exc(file=handle)
        minutes = (time.monotonic() - started) / 60
        status = log_status(log_dir, wall) if code == 0 else f"exit {code}"
        if status != "success":
            failed += 1
            say(f"FAILED {entry['label']} ({status}, {minutes:.1f} min; see {out})")
        else:
            say(f"done   {entry['label']} ({minutes:.1f} min)")
    say(f"no slices left; {failed} failed")
    return failed


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run this worker's share of the pack."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path, help="Pack manifest written by scripts/evaluate.py run.")
    parser.add_argument("--worker-id", default="w0", help="Name of this worker in messages and output files.")
    args = parser.parse_args(argv)
    sys.exit(1 if run(args.manifest, args.worker_id) else 0)


if __name__ == "__main__":
    main()
