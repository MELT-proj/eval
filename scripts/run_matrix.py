"""Submit every (model, benchmark slice) of a matrix config, one SLURM job each.

    python scripts/run_matrix.py configs/matrix/baselines-v1.yaml --site bocconi --dry-run
    python scripts/run_matrix.py configs/matrix/baselines-v1.yaml --site bocconi [--models qwen3-omni-30b-a3b-instruct]
    python scripts/run_matrix.py configs/matrix/baselines-v1.yaml --site bocconi prefetch
    python scripts/run_matrix.py configs/matrix/baselines-v1.yaml --site bocconi score [--dry-run] [--models ...]

See ``configs/matrix/baselines-v1.yaml`` for the config format.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent

#: Tasks graded only by a judge model, generated now and scored later.
JUDGE_TASKS = {"audio_chat"}

#: Providers that apply their own chat template, so the solver passes them a
#: bare instruction (``-T instruction=...``). MELT takes its prompt from the
#: checkpoint's training config instead and rejects an instruction argument.
INSTRUCTION_PROVIDERS = {"smurf", "qwen2_audio", "qwen3_omni"}

_UNSET_VAR = re.compile(r"\$\{?\w+\}?")


@dataclass
class Job:
    """One (model, benchmark slice) evaluation."""

    model: str
    provider: str
    checkpoint: str
    benchmark: str
    slice: str
    task: str
    eval_set: str
    log_dir: Path
    args: list[str]
    env: dict[str, str] = field(default_factory=dict)
    #: How the log is graded after generation (judge-only tasks): the
    #: benchmark's ``score`` block, with the slice's ``scorers`` over it.
    score: dict | None = None


def expand(value: str | dict | None, site: str) -> str | None:
    """Expand ``~`` and ``${VAR}`` in *value* (or its entry for *site*).

    Returns:
        The expanded string, or ``None`` when there is no value for this site
        or it names an environment variable that is not set.
    """
    if isinstance(value, dict):
        value = value.get(site)
    if value is None:
        return None
    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    return None if _UNSET_VAR.search(expanded) else expanded


def resolve_path(value: str | dict | None, site: str, what: str) -> tuple[str | None, str | None]:
    """Resolve a path for *site*, globbing a ``*`` to exactly one match.

    A relative path is taken from the repo root, not the working directory. A
    pattern ending in ``/`` matches directories only (so ``checkpoints/melt/*/``
    ignores a stray archive next to the run folder).

    Returns:
        ``(path, None)`` when it exists, else ``(None, reason)``.
    """
    path = expand(value, site)
    if path is None:
        return None, f"no {what} for site {site!r} (unset variable in {value!r}?)"
    if not Path(path).is_absolute():
        path = f"{REPO_ROOT}/{path}"  # not REPO_ROOT / path: that drops a trailing "/"
    if "*" in path:
        matches = sorted(glob.glob(path))
        if len(matches) != 1:
            return None, f"{what} {path!r} matches {len(matches)} paths, expected exactly one"
        path = matches[0].rstrip("/")
    if not Path(path).exists():
        return None, f"{what} {path} does not exist on this site"
    return path, None


def resolve_log_root(matrix: dict, site: str) -> str:
    """Where this site's logs go: the matrix's ``log_root``, else ``$OUTPUT_DIR/<name>``."""
    log_root = expand(matrix.get("log_root"), site)
    if log_root is None:
        output_dir = os.environ.get("OUTPUT_DIR")
        if not output_dir:
            raise SystemExit(f"No log_root for site {site!r} in the matrix and OUTPUT_DIR is not set.")
        log_root = str(Path(output_dir) / matrix["name"])
    return log_root


def slice_name(slice_cfg: dict) -> str:
    """A slice's directory and metadata name: its task, and dataset_id if any."""
    dataset_id = slice_cfg.get("dataset_id")
    return f"{slice_cfg['task']}-{dataset_id}" if dataset_id else slice_cfg["task"]


def plan(
    matrix: dict,
    site: str,
    models: list[str] | None = None,
    benchmarks: list[str] | None = None,
) -> tuple[list[Job], list[str]]:
    """Expand the matrix into jobs for *site*.

    Returns:
        The jobs to consider, and a line per (model or benchmark or slice)
        skipped, with why.
    """
    jobs: list[Job] = []
    skipped: list[str] = []

    log_root = resolve_log_root(matrix, site)

    for model in matrix["models"]:
        if models and model["name"] not in models:
            continue
        checkpoint, reason = resolve_path(model["checkpoint"], site, "checkpoint")
        if checkpoint is None:
            skipped.append(f"model {model['name']}: {reason}")
            continue
        env = {"MELTEVAL_PROVIDER": model["provider"]}
        if "venv" in model:
            venv = expand(model["venv"], site)
            if venv is None or not Path(venv).exists():
                skipped.append(f"model {model['name']}: venv {model['venv']!r} not found on this site")
                continue
            env["VENV_PATH"] = venv

        for bench in matrix["benchmarks"]:
            if benchmarks and bench["name"] not in benchmarks:
                continue
            if "spec" in bench:
                eval_set, reason = resolve_path(bench["spec"], site, "spec")
            else:
                eval_set, reason = resolve_path(bench.get("frozen_set"), site, "frozen set")
            if eval_set is None:
                skipped.append(f"{model['name']} x {bench['name']}: {reason}")
                continue

            for slice_cfg in bench["slices"]:
                name = slice_name(slice_cfg)
                task = slice_cfg["task"]
                dataset_id = slice_cfg.get("dataset_id")
                label = f"{model['name']} x {bench['name']}/{name}"

                if dataset_id and any(fnmatch.fnmatch(dataset_id, p) for p in model.get("skip", [])):
                    skipped.append(f"{label}: excluded by the model's skip list")
                    continue

                args = ["-T", f"task_filter={task}"]
                if dataset_id:
                    args += ["-T", f"dataset_id={dataset_id}"]

                if bench.get("needs_instruction") and model["provider"] in INSTRUCTION_PROVIDERS:
                    instruction = (model.get("instruction") or {}).get(task)
                    if instruction is None:
                        skipped.append(f"{label}: no instruction for task {task!r} in the model config")
                        continue
                    # -T values are parsed as YAML; a JSON string is a quoted YAML string.
                    args += ["-T", f"instruction={json.dumps(instruction)}"]

                args += list((model.get("task_args") or {}).get(task, []))
                args += list(model.get("args", []))
                args += [str(a) for a in slice_cfg.get("args", [])]
                if task in JUDGE_TASKS:
                    args.append("--no-score")
                args += [
                    "--metadata", f"melteval_model={model['name']}",
                    "--metadata", f"melteval_benchmark={bench['name']}",
                    "--metadata", f"melteval_slice={name}",
                    "--metadata", f"melteval_task={task}",
                    "--metadata", f"melteval_matrix={matrix['name']}",
                ]  # fmt: skip

                log_dir = Path(log_root) / model["name"] / bench["name"] / name
                job_env = {**env, "OUTPUT_DIR": str(log_dir)}
                if "time" in slice_cfg:
                    job_env["MELT_TIME"] = str(slice_cfg["time"])
                score = dict(bench.get("score") or {})
                if "scorers" in slice_cfg:
                    score["scorers"] = slice_cfg["scorers"]
                jobs.append(
                    Job(
                        model=model["name"],
                        provider=model["provider"],
                        checkpoint=checkpoint,
                        benchmark=bench["name"],
                        slice=name,
                        task=task,
                        eval_set=eval_set,
                        log_dir=log_dir,
                        args=args,
                        env=job_env,
                        score=score or None,
                    )
                )
    return jobs, skipped


def successful_logs(log_dir: Path) -> list[Path]:
    """Logs in *log_dir* whose run finished (header-only reads)."""
    from inspect_ai.log import read_eval_log

    found = []
    for path in sorted([*log_dir.glob("*.eval"), *log_dir.glob("*.json")]):
        try:
            if read_eval_log(str(path), header_only=True).status == "success":
                found.append(path)
        except Exception as exc:  # noqa: BLE001 - a half-written log is just "not done"
            print(f"  ! unreadable log {path}: {exc}", file=sys.stderr)
    return found


def wait_for_room() -> None:
    """Block until the user has fewer than ``$MAX_QUEUED`` jobs in the queue.

    A cluster's QOS usually caps how many jobs one user may have submitted at
    once (Bocconi: 30), and sbatch past it fails outright -- a matrix is bigger
    than that. The site file sets MAX_QUEUED; unset means no limit.
    """
    import time

    limit = os.environ.get("MAX_QUEUED")
    if not limit:
        return
    user = os.environ.get("USER", "")
    announced = False
    while True:
        queued = subprocess.run(
            ["squeue", "-h", "-u", user], capture_output=True, text=True, check=False
        ).stdout.count("\n")
        if queued < int(limit):
            return
        if not announced:
            print(f"  … {queued} job(s) queued (limit {limit}); waiting for room", flush=True)
            announced = True
        time.sleep(60)


def job_tag(*parts: str) -> str:
    """The SLURM ``--comment`` a job is submitted with, naming what it evaluates."""
    return "melteval:" + ":".join(parts)


def queued_tags() -> set[str]:
    """Tags of this user's jobs still pending or running (``squeue -o %k``)."""
    result = subprocess.run(
        ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%k"], capture_output=True, text=True, check=False
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip().startswith("melteval:")}


def submit(job: Job, site: str, dry_run: bool) -> None:
    """Submit *job* through submit_eval.sh (or print what would be)."""
    command = ["infra/runners/submit_eval.sh", site, job.checkpoint, job.eval_set, *job.args]
    env_str = " ".join(f"{k}={v}" for k, v in job.env.items())
    print(f"  $ {env_str} {' '.join(command)}", flush=True)
    if dry_run:
        return
    wait_for_room()
    job.log_dir.mkdir(parents=True, exist_ok=True)
    tag = job_tag(job.model, job.benchmark, job.slice)
    subprocess.run(command, cwd=REPO_ROOT, env={**os.environ, **job.env, "MELT_JOB_TAG": tag}, check=True)


def cmd_run(args: argparse.Namespace, matrix: dict) -> None:
    """Submit every job without a successful log yet."""
    jobs, skipped = plan(matrix, args.site, args.models, args.benchmarks)
    for line in skipped:
        print(f"skip  {line}")

    submitted = done = in_queue = 0
    queued = set() if args.dry_run else queued_tags()
    for job in jobs:
        if successful_logs(job.log_dir):
            done += 1
            print(f"done  {job.model} x {job.benchmark}/{job.slice}")
            continue
        if job_tag(job.model, job.benchmark, job.slice) in queued:
            in_queue += 1
            print(f"queued  {job.model} x {job.benchmark}/{job.slice} (already pending or running)")
            continue
        print(f"{'plan' if args.dry_run else 'submit'}  {job.model} x {job.benchmark}/{job.slice}", flush=True)
        submit(job, args.site, args.dry_run)
        submitted += 1

    verb = "would submit" if args.dry_run else "submitted"
    print(f"\n{verb} {submitted} job(s); {done} already done; {in_queue} already queued; {len(skipped)} skipped.")


#: Scorer for a judged slice when its benchmark names a judge but no scorers.
DEFAULT_JUDGE_SCORER = "melteval/scorers.py@chat_scorer"

#: Providers that can serve a judge prompt (``text_only`` in providers/hf.py).
TEXT_ONLY_PROVIDERS = {"qwen2_audio", "qwen3_omni"}


@dataclass
class ScoreJob:
    """One ``inspect score`` pass over some logs, in one SLURM job."""

    label: str
    logs: list[Path]
    args: list[str]
    venv: str | None = None


def resolve_judge(judge: str, matrix: dict, site: str) -> tuple[list[str], str | None]:
    """Turn a benchmark's ``judge`` into ``--model-role`` arguments and a venv.

    *judge* is either the name of a model in the matrix -- loaded text-only,
    from its own checkpoint and venv -- or a ``provider/model`` string passed to
    inspect as is (an API judge).

    Raises:
        SystemExit: If a named judge cannot be resolved on this site.
    """
    by_name = {m["name"]: m for m in matrix["models"]}
    if judge not in by_name:
        return ["--model-role", f"grader={judge}"], None

    model = by_name[judge]
    if model["provider"] not in TEXT_ONLY_PROVIDERS:
        raise SystemExit(
            f"Judge {judge!r} uses provider {model['provider']!r}, which cannot serve a text-only "
            f"prompt. Use one of: {sorted(TEXT_ONLY_PROVIDERS)}, or an API model as provider/model."
        )
    checkpoint, reason = resolve_path(model["checkpoint"], site, "judge checkpoint")
    if checkpoint is None:
        raise SystemExit(f"Judge {judge!r}: {reason}")
    role = {"model": f"{model['provider']}/{checkpoint}", "model_args": {"text_only": True}}
    return ["--model-role", f"grader={json.dumps(role)}"], expand(model.get("venv"), site)


def _has_scorer(log: Path, scorer: str) -> bool:
    """Whether *log* already carries results from *scorer* (``file.py@name`` or a registry name)."""
    from inspect_ai.log import read_eval_log

    results = read_eval_log(str(log), header_only=True).results
    wanted = scorer.split("@")[-1].split("/")[-1]
    return bool(results) and any(s.name.split("/")[-1] == wanted for s in results.scores)


def plan_scoring(matrix: dict, jobs: list[Job], site: str, rescore: bool = False) -> tuple[list[ScoreJob], list[str]]:
    """Group the finished judge-only logs into scoring jobs.

    One job per (model under test, benchmark, scorer), so a judge is loaded
    once per group rather than once per slice, and a failure is scoped.

    Returns:
        The scoring jobs, and a line per log already scored or skipped.
    """
    groups: dict[tuple, ScoreJob] = {}
    notes: list[str] = []
    for job in jobs:
        if job.task not in JUDGE_TASKS:
            continue
        if not job.score:
            notes.append(f"{job.model} x {job.benchmark}/{job.slice}: no `score` block for this benchmark")
            continue
        scorers = job.score.get("scorers", [DEFAULT_JUDGE_SCORER] if "judge" in job.score else [])
        pending = []
        for log in successful_logs(job.log_dir):
            for scorer in scorers:
                if _has_scorer(log, scorer) and not rescore:
                    notes.append(f"scored  {log.name} ({scorer})")
                else:
                    pending.append((log, scorer))
        if not pending:
            continue

        # Resolved only once there is something to score, so a judge not yet
        # downloaded does not block a dry run over slices still generating.
        if "judge" in job.score:
            extra, venv = resolve_judge(job.score["judge"], matrix, site)
        else:
            extra, venv = [], expand(job.score.get("venv"), site)
            if job.score.get("venv") and (venv is None or not Path(venv).exists()):
                label = f"{job.model} x {job.benchmark}/{job.slice}"
                notes.append(f"{label}: scoring venv {job.score['venv']!r} not found")
                continue

        for log, scorer in pending:
            key = (job.model, job.benchmark, scorer, venv, tuple(extra))
            if key not in groups:
                # --model mockllm/model: `inspect score` rebuilds the log's model
                # even though scoring never generates, and our providers load
                # their weights on construction. A judge comes in through the
                # grader role, not this.
                args = [
                    "--model", "mockllm/model",
                    "--scorer", scorer,
                    "--action", "overwrite" if rescore else "append",
                    *extra,
                ]  # fmt: skip
                groups[key] = ScoreJob(f"{job.model} x {job.benchmark} [{scorer}]", [], args, venv)
            groups[key].logs.append(log)
    return list(groups.values()), notes


def cmd_score(args: argparse.Namespace, matrix: dict) -> None:
    """Submit the scoring of judge-only slices (audio_chat) through SLURM."""
    jobs, _ = plan(matrix, args.site, args.models, args.benchmarks)
    score_jobs, notes = plan_scoring(matrix, jobs, args.site, rescore=args.rescore)
    for line in notes:
        print(line)
    queued = set() if args.dry_run else queued_tags()
    for sj in score_jobs:
        command = ["infra/runners/submit_score.sh", args.site, *map(str, sj.logs), "--", *sj.args]
        env = {"VENV_PATH": sj.venv} if sj.venv else {}
        env["MELT_JOB_TAG"] = job_tag("score", sj.label)
        if env["MELT_JOB_TAG"] in queued:
            print(f"queued  {sj.label} (already pending or running)")
            continue
        print(f"{'plan' if args.dry_run else 'submit'}  {sj.label}: {len(sj.logs)} log(s)")
        print(f"  $ {' '.join(f'{k}={v}' for k, v in env.items())} {' '.join(command)}")
        if not args.dry_run:
            wait_for_room()
            subprocess.run(command, cwd=REPO_ROOT, env={**os.environ, **env}, check=True)
    verb = "would submit" if args.dry_run else "submitted"
    print(f"\n{verb} {len(score_jobs)} scoring job(s).")


def cmd_prefetch(args: argparse.Namespace, matrix: dict) -> None:
    """Download every spec slice the matrix reads, into ``HF_HOME``, from here.

    Reads each ``(spec, dataset_id)`` through :func:`melteval.dataset.spec_dataset`
    -- the same code path a job takes -- so exactly what a job will ask for is
    cached, and nothing else. CPU and network only: run it on the login node,
    once, before submitting, rather than letting every job download the same
    split at the same time.
    """
    from melteval.dataset import spec_dataset

    print(f"HF_HOME={os.environ.get('HF_HOME', '(unset: ~/.cache/huggingface)')}")
    for bench in matrix["benchmarks"]:
        if args.benchmarks and bench["name"] not in args.benchmarks:
            continue
        if "spec" not in bench:
            print(f"skip  {bench['name']}: a frozen set, nothing to download")
            continue
        spec = f"{REPO_ROOT}/{bench['spec']}"
        dataset_ids = list(dict.fromkeys(s.get("dataset_id") for s in bench["slices"]))
        for dataset_id in dataset_ids:
            label = f"{bench['name']}/{dataset_id or '(all)'}"
            if args.dry_run:
                print(f"plan  {label}")
                continue
            print(f"fetch {label} ...", flush=True)
            dataset = spec_dataset(spec, dataset_id=dataset_id)
            print(f"      {len(dataset)} samples")


COMMANDS = ("score", "prefetch", "log-root")


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch."""
    # The filters are accepted both before and after a subcommand. Defaults are
    # SUPPRESS on the copies so a subcommand does not reset what was given
    # before it.
    filters = argparse.ArgumentParser(add_help=False)
    filters.add_argument("--models", nargs="+", default=argparse.SUPPRESS, help="Only these models (by name).")
    filters.add_argument("--benchmarks", nargs="+", default=argparse.SUPPRESS, help="Only these benchmarks.")
    filters.add_argument(
        "--dry-run", action="store_true", default=argparse.SUPPRESS, help="Print the commands; submit nothing."
    )

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[filters]
    )
    parser.add_argument("matrix", type=Path, help="Matrix config YAML.")
    parser.add_argument("--site", required=True, help="Site file under infra/sites/ to submit through.")
    sub = parser.add_subparsers(dest="command")
    score = sub.add_parser(
        "score", parents=[filters], help="Grade audio_chat logs as each benchmark's `score` block says."
    )
    score.add_argument("--rescore", action="store_true", help="Re-grade logs that already have scores.")
    sub.add_parser("prefetch", parents=[filters], help="Download the spec slices the matrix reads into HF_HOME.")
    sub.add_parser("log-root", parents=[filters], help="Print where this site's logs go (for scripts).")
    args = parser.parse_args(argv)
    for name, default in (("models", None), ("benchmarks", None), ("dry_run", False), ("rescore", False)):
        if not hasattr(args, name):
            setattr(args, name, default)

    # `--models a b score` makes argparse read `score` as a model name and run
    # generation instead of scoring. Refuse it rather than submit the wrong jobs.
    swallowed = [n for n in (args.models or []) + (args.benchmarks or []) if n in COMMANDS]
    if swallowed:
        parser.error(
            f"{swallowed[0]!r} was read as a model/benchmark name. Put the command before the filters: "
            f"run_matrix.py <matrix> --site <site> {swallowed[0]} --models ..."
        )

    matrix = yaml.safe_load(args.matrix.read_text(encoding="utf-8"))
    commands = {
        "score": cmd_score,
        "prefetch": cmd_prefetch,
        "log-root": lambda args, matrix: print(resolve_log_root(matrix, args.site)),
    }
    commands.get(args.command, cmd_run)(args, matrix)


if __name__ == "__main__":
    main()
