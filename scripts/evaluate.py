"""Evaluate a set of models on a set of benchmarks, and collect everything in one folder.

    python scripts/evaluate.py plan     CONFIG [--site SITE] [-v]     # what would run; runs nothing
    python scripts/evaluate.py run      CONFIG (--site SITE | --local) [--wait] [--limit N]
    python scripts/evaluate.py status   CONFIG [--site SITE]          # done / queued / failed / missing
    python scripts/evaluate.py score    CONFIG (--site SITE | --local) [--rescore]   # judge for audio_chat
    python scripts/evaluate.py report   CONFIG                        # CSV + Excel into <output_dir>/results
    python scripts/evaluate.py prefetch CONFIG [--site SITE]          # datasets + baseline weights, from a login node

Every command also takes --out DIR (instead of the config's output_dir) and
--models/--benchmarks NAME... to restrict to some of them.

One job is one (model, benchmark slice). ``run`` only starts jobs that have no
successful log yet, so running it again after a failure retries exactly what is
missing. The config format and the output folder are described in
docs/running-evaluations.md; configs/eval/baselines-v1.yaml is an example.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent

#: Tasks graded by a judge model: generated with --no-score, graded by ``score``.
JUDGE_TASKS = {"audio_chat"}

#: The scorer ``score`` applies when a benchmark names a judge but no scorers.
DEFAULT_JUDGE_SCORER = "melteval/scorers.py@chat_scorer"

#: Environment variable holding the key of an API judge's provider, checked
#: before scoring starts so a missing key fails at once rather than per log.
API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

_UNSET_VAR = re.compile(r"\$\{?\w+\}?")


# --------------------------------------------------------------------------- config


def load_site_env(site: str) -> None:
    """Load a site file's exports (HF_HOME, venv paths, MAX_QUEUED, ...) into this process.

    The config's ``${VAR}`` paths resolve against them. Values already exported
    in the shell win, since every site file writes ``${VAR:-default}``.
    """
    site_file = REPO_ROOT / "infra" / "sites" / f"{site}.sh"
    if not site_file.exists():
        raise SystemExit(f"Unknown site {site!r}: no {site_file}.")
    out = subprocess.run(
        ["bash", "-c", f'source "{site_file}" >/dev/null && env -0'],
        capture_output=True,
        check=True,
    ).stdout.decode()
    for entry in out.split("\0"):
        key, sep, value = entry.partition("=")
        if sep and key.isidentifier():
            os.environ[key] = value


def expand(value) -> str | None:
    """Expand ``~`` and ``${VAR}``; ``None`` if a variable is not set."""
    if value is None:
        return None
    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    return None if _UNSET_VAR.search(expanded) else expanded


def resolve_path(value, what: str) -> tuple[str | None, str | None]:
    """Resolve a path: relative means relative to the repo root, a ``*`` must match exactly one path.

    A pattern ending in ``/`` matches directories only.

    Returns:
        ``(absolute path, None)``, or ``(None, why not)``.
    """
    path = expand(value)
    if path is None:
        return None, f"{what} {value!r} uses an environment variable that is not set"
    if not Path(path).is_absolute():
        path = f"{REPO_ROOT}/{path}"  # not REPO_ROOT / path: that drops a trailing "/"
    if "*" in path:
        matches = sorted(glob.glob(path))
        if len(matches) != 1:
            return None, f"{what} {path!r} matches {len(matches)} paths, expected exactly one"
        path = matches[0]
    path = path.rstrip("/")
    if not Path(path).exists():
        return None, f"{what} {path} does not exist on this machine"
    return path, None


def resolve_model(model: str) -> tuple[str | None, str | None]:
    """Turn a config's ``model:`` into the inspect model name a job runs.

    ``melt/<checkpoint>`` and ``smurf/<checkpoint>`` get an absolute path;
    ``melt/hf/<key>`` must be a known baseline; anything else is refused.

    Returns:
        ``(model name, None)``, or ``(None, why not)``.
    """
    from melteval.providers import family_of

    family = family_of(model)
    api, _, name = model.partition("/")
    if family == "hf":
        from melteval.providers.router import HF_MODELS

        key = name.removeprefix("hf/")
        if key not in HF_MODELS:
            return None, f"unknown baseline {model!r} (known: {', '.join(f'melt/hf/{k}' for k in HF_MODELS)})"
        return model, None
    if family in ("melt", "smurf"):
        path, reason = resolve_path(name, "checkpoint")
        return (f"{api}/{path}", None) if path else (None, reason)
    if family == "vllm":
        return None, "melt/vllm/... is not implemented"
    return None, f"{model!r} is not a melt/..., melt/hf/... or smurf/... model"


def load_config(path: Path, out: str | None) -> dict:
    """Read a config and settle its output directory (``--out`` wins over ``output_dir``)."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    output_dir = expand(out or config.get("output_dir"))
    if not output_dir:
        raise SystemExit(
            f"No output directory: pass --out, or set output_dir in {path} "
            f"(and export the variables it uses: {config.get('output_dir')!r})."
        )
    config["_output_dir"] = Path(output_dir).resolve()
    config["_path"] = path
    return config


def slice_name(slice_cfg: dict) -> str:
    """A slice's folder name: its task, plus its dataset_id if any."""
    dataset_id = slice_cfg.get("dataset_id")
    return f"{slice_cfg['task']}-{dataset_id}" if dataset_id else slice_cfg["task"]


@dataclass
class Job:
    """One (model, benchmark slice): one ``inspect eval``."""

    model_name: str  #: The name in the config (a folder, a row of the report).
    model: str  #: The inspect model name, e.g. melt/hf/qwen2_audio.
    benchmark: str
    slice: str
    task: str
    eval_set: str
    log_dir: Path
    args: list[str]
    env: dict[str, str] = field(default_factory=dict)
    judge: dict | None = None  #: The benchmark's judge settings, for audio_chat slices.

    @property
    def label(self) -> str:
        return f"{self.model_name} x {self.benchmark}/{self.slice}"

    @property
    def tag(self) -> str:
        """The SLURM --comment the job is submitted with, to find it in the queue."""
        return f"melteval:{self.model_name}:{self.benchmark}:{self.slice}"


def plan(
    config: dict, models: list[str] | None = None, benchmarks: list[str] | None = None
) -> tuple[list[Job], list[str]]:
    """Expand the config into jobs.

    Returns:
        The jobs, and one line per (model, benchmark or slice) that cannot run
        on this machine or is excluded, with why.
    """
    from melteval.providers import family_of

    jobs: list[Job] = []
    skipped: list[str] = []
    log_root = config["_output_dir"] / "logs"
    log_format = config.get("log_format", "json")

    for model_cfg in config["models"]:
        name = model_cfg["name"]
        if models and name not in models:
            continue
        model, reason = resolve_model(model_cfg["model"])
        if model is None:
            skipped.append(f"model {name}: {reason}")
            continue
        family = family_of(model)
        env = {}
        if "venv" in model_cfg:
            venv = expand(model_cfg["venv"])
            if venv is None or not Path(venv).exists():
                skipped.append(f"model {name}: venv {model_cfg['venv']!r} not found on this machine")
                continue
            env["VENV_PATH"] = venv

        for bench in config["benchmarks"]:
            if benchmarks and bench["name"] not in benchmarks:
                continue
            if "spec" in bench:
                eval_set, reason = resolve_path(bench["spec"], "spec")
            else:
                eval_set, reason = resolve_path(bench.get("frozen_set"), "frozen set")
            if eval_set is None:
                skipped.append(f"{name} x {bench['name']}: {reason}")
                continue

            for slice_cfg in bench["slices"]:
                task, dataset_id, sname = slice_cfg["task"], slice_cfg.get("dataset_id"), slice_name(slice_cfg)
                label = f"{name} x {bench['name']}/{sname}"
                if dataset_id and any(fnmatch.fnmatch(dataset_id, p) for p in model_cfg.get("skip", [])):
                    skipped.append(f"{label}: excluded by the model's skip list")
                    continue

                args = ["-T", f"task_filter={task}"]
                if dataset_id:
                    args += ["-T", f"dataset_id={dataset_id}"]
                # A benchmark whose samples carry no instruction of their own
                # needs one from the model entry -- except for MELT, whose
                # prompt comes from the checkpoint's training config.
                if bench.get("needs_instruction") and family in ("hf", "smurf"):
                    instruction = (model_cfg.get("instruction") or {}).get(task)
                    if instruction is None:
                        skipped.append(f"{label}: no instruction for task {task!r} in the model entry")
                        continue
                    # -T values are parsed as YAML; a JSON string is a quoted YAML string.
                    args += ["-T", f"instruction={json.dumps(instruction)}"]

                args += [str(a) for a in model_cfg.get("args", [])]
                args += [str(a) for a in slice_cfg.get("args", [])]
                if task in JUDGE_TASKS:
                    args.append("--no-score")
                args += ["--log-format", log_format]
                args += [
                    "--metadata", f"melteval_model={name}",
                    "--metadata", f"melteval_benchmark={bench['name']}",
                    "--metadata", f"melteval_slice={sname}",
                    "--metadata", f"melteval_task={task}",
                    "--metadata", f"melteval_campaign={config['name']}",
                ]  # fmt: skip

                log_dir = log_root / name / bench["name"] / sname
                job_env = {**env, "OUTPUT_DIR": str(log_dir)}
                if "time" in slice_cfg:
                    job_env["MELT_TIME"] = str(slice_cfg["time"])
                judge = {k: bench[k] for k in ("judge", "judge_args", "judge_venv", "scorers") if k in bench}
                jobs.append(
                    Job(name, model, bench["name"], sname, task, eval_set, log_dir, args, job_env, judge or None)
                )
    return jobs, skipped


def read_logs(log_dir: Path) -> list[tuple[Path, object]]:
    """``(path, header)`` of every inspect log in *log_dir*, oldest first."""
    from inspect_ai.log import read_eval_log

    found = []
    for path in sorted([*log_dir.glob("*.eval"), *log_dir.glob("*.json")], key=lambda p: p.stat().st_mtime):
        try:
            found.append((path, read_eval_log(str(path), header_only=True)))
        except Exception as exc:  # noqa: BLE001 - a half-written log is just "not done"
            print(f"  ! unreadable log {path}: {exc}", file=sys.stderr)
    return found


def successful_logs(log_dir: Path) -> list[Path]:
    """Logs in *log_dir* whose run finished."""
    return [path for path, log in read_logs(log_dir) if log.status == "success"]


def squeue_available() -> bool:
    return shutil.which("squeue") is not None


def queued_tags() -> set[str]:
    """Tags of this user's SLURM jobs still pending or running."""
    if not squeue_available():
        return set()
    result = subprocess.run(
        ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%k"], capture_output=True, text=True, check=False
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip().startswith("melteval:")}


def job_state(job: Job, queued: set[str]) -> tuple[str, str]:
    """``(state, detail)``: done, queued, failed or missing."""
    logs = read_logs(job.log_dir) if job.log_dir.exists() else []
    if any(log.status == "success" for _, log in logs):
        return "done", ""
    if job.tag in queued:
        return "queued", ""
    if logs:
        path, log = logs[-1]
        message = str(getattr(log.error, "message", "") or log.status).strip().splitlines()
        return "failed", f"{log.status}: {message[-1] if message else ''} ({path.name})"
    return "missing", ""


def wait_for_room() -> None:
    """Block while this user has ``$MAX_QUEUED`` or more SLURM jobs (a QOS cap; the site file sets it)."""
    limit = os.environ.get("MAX_QUEUED")
    if not limit or not squeue_available():
        return
    announced = False
    while True:
        queued = subprocess.run(
            ["squeue", "-h", "-u", os.environ.get("USER", "")], capture_output=True, text=True, check=False
        ).stdout.count("\n")
        if queued < int(limit):
            return
        if not announced:
            print(f"  ... {queued} job(s) queued (limit {limit}); waiting for room", flush=True)
            announced = True
        time.sleep(60)


def wait_for_jobs(job_ids: list[str], poll: int = 120) -> None:
    """Block until every SLURM job in *job_ids* has left the queue."""
    if not job_ids:
        return
    print(f"\nwaiting for {len(job_ids)} job(s): {','.join(job_ids)}", flush=True)
    while True:
        out = subprocess.run(
            ["squeue", "-h", "-j", ",".join(job_ids)], capture_output=True, text=True, check=False
        ).stdout
        left = len(out.strip().splitlines())
        if left == 0:
            return
        print(f"  {datetime.now():%H:%M:%S}  {left} job(s) still queued or running", flush=True)
        time.sleep(poll)


def record(config: dict, entry: dict) -> None:
    """Append one line to ``<output_dir>/submissions.jsonl``: what was started, when, how."""
    path = config["_output_dir"] / "submissions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **entry}) + "\n")


def run_command(command: list[str], env: dict[str, str], log_file: Path | None = None) -> tuple[int, str]:
    """Run *command* from the repo root, echoing its output (and copying it to *log_file*).

    Returns:
        The exit code and the whole output.
    """
    lines = []
    handle = log_file.open("a", encoding="utf-8") if log_file else None
    process = subprocess.Popen(
        command, cwd=REPO_ROOT, env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
        if handle:
            handle.write(line)
    if handle:
        handle.close()
    return process.wait(), "".join(lines)


def current_venv() -> str:
    """The activate script of the venv running this script: the default for a local job."""
    return str(Path(sys.prefix) / "bin" / "activate")


def start(job: Job, site: str | None, config: dict) -> str | None:
    """Start *job*: sbatch through the site (returns the job id), or run it here and now.

    Both go through infra/run_eval.sbatch; on SLURM it is submitted, locally it
    is just run (its #SBATCH lines are comments to bash).
    """
    job.log_dir.mkdir(parents=True, exist_ok=True)
    if site:
        command = ["infra/runners/submit_eval.sh", site, job.model, job.eval_set, *job.args]
        env = {**job.env, "MELT_JOB_TAG": job.tag, "MELT_SLURM_OUT": str(job.log_dir / "slurm-%j.out")}
        wait_for_room()
        code, out = run_command(command, env)
        job_id = next(iter(re.findall(r"Submitted batch job (\d+)", out)), None)
        record(config, {"runner": f"slurm:{site}", "job": job.label, "model": job.model, "command": command,
                        "slurm_job_id": job_id, "exit_code": code})  # fmt: skip
        return job_id
    command = ["bash", "infra/run_eval.sbatch", job.model, job.eval_set, *job.args]
    env = {"VENV_PATH": current_venv(), **job.env}
    log_file = job.log_dir / f"local-{datetime.now():%Y%m%d-%H%M%S}.out"
    code, _ = run_command(command, env, log_file)
    record(config, {"runner": "local", "job": job.label, "model": job.model, "command": command,
                    "exit_code": code, "output": str(log_file)})  # fmt: skip
    return None


def snapshot_config(config: dict) -> None:
    """Keep a copy of the config in the output folder, so the folder says what it holds."""
    out = config["_output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config["_path"], out / "config.yaml")


def cmd_plan(args, config) -> None:
    """Print every job, its state, and the command that would start it."""
    jobs, skipped = plan(config, args.models, args.benchmarks)
    for line in skipped:
        print(f"skip     {line}")
    queued = queued_tags()
    for job in jobs:
        state, detail = job_state(job, queued)
        print(f"{state:<8} {job.label}  {detail}")
        if args.verbose and state in ("missing", "failed"):
            print(f"         {job.model} {job.eval_set} {' '.join(job.args)}")
    print(f"\noutput folder: {config['_output_dir']}")


def cmd_status(args, config) -> None:
    """Count jobs by state, and list the ones that need attention."""
    jobs, skipped = plan(config, args.models, args.benchmarks)
    queued = queued_tags()
    counts: dict[str, int] = {}
    for job in jobs:
        state, detail = job_state(job, queued)
        counts[state] = counts.get(state, 0) + 1
        if state != "done":
            print(f"{state:<8} {job.label}  {detail}")
    print("\n" + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) + f"; {len(skipped)} skipped (see plan)")
    print(f"output folder: {config['_output_dir']}")


def cmd_run(args, config) -> None:
    """Start every job that has no successful log and is not already queued."""
    if args.limit:
        # A cut-down run must never count as done for the real one: it goes
        # to a folder of its own.
        config["_output_dir"] = config["_output_dir"] / f"smoke-limit{args.limit}"
    jobs, skipped = plan(config, args.models, args.benchmarks)
    for line in skipped:
        print(f"skip     {line}")
    snapshot_config(config)

    queued = queued_tags()
    job_ids, started = [], 0
    for job in jobs:
        state, _ = job_state(job, queued)
        if state in ("done", "queued"):
            print(f"{state:<8} {job.label}")
            continue
        if args.limit:
            job.args += ["-T", f"limit={args.limit}"]
        print(f"\n==== start {job.label}", flush=True)
        job_id = start(job, args.site, config)
        started += 1
        if job_id:
            job_ids.append(job_id)

    print(f"\nstarted {started} job(s); output folder: {config['_output_dir']}")
    if args.site and args.wait:
        wait_for_jobs(job_ids)
    if args.local or args.wait:
        cmd_status(args, config)
        cmd_report(args, config)


def resolve_judge(judge: dict) -> tuple[list[str], str | None, str | None]:
    """A benchmark's judge settings as ``--model-role`` arguments.

    Returns:
        The arguments, whether the judge is ``"local"`` (a model this harness
        loads, so GPU work) or ``"api"``, and the venv to score in (``None``:
        the default one).

    Raises:
        SystemExit: If the judge cannot grade text, or its API key is missing.
    """
    from melteval.providers import family_of

    model = judge["judge"]
    model_args = dict(judge.get("judge_args") or {})
    family = family_of(model)
    if family == "hf":
        kind = "local"
        model_args["text_only"] = True
    elif family is None:
        kind = "api"
        key = API_KEYS.get(model.split("/")[0])
        if key and not os.environ.get(key):
            raise SystemExit(f"Judge {model}: {key} is not set in this shell.")
    else:
        raise SystemExit(
            f"Judge {model}: a {family!r} model cannot grade text-only prompts. "
            "Use melt/hf/<model> (loaded text-only) or an API model such as openai/gpt-4o."
        )
    role = {"model": model, "model_args": model_args} if model_args else model
    grader = json.dumps(role) if isinstance(role, dict) else role
    venv = expand(judge.get("judge_venv")) if judge.get("judge_venv") else None
    return ["--model-role", f"grader={grader}"], kind, venv


def has_scorer(log: Path, scorer: str) -> bool:
    """Whether *log* already carries results from *scorer*."""
    from inspect_ai.log import read_eval_log

    results = read_eval_log(str(log), header_only=True).results
    wanted = scorer.split("@")[-1].split("/")[-1]
    return bool(results) and any(s.name.split("/")[-1] == wanted for s in results.scores)


def cmd_score(args, config) -> None:
    """Grade the finished judge-only slices (audio_chat) with their benchmark's judge.

    One job per (model under test, benchmark, scorer), so the judge is loaded
    once per group. An API judge needs no GPU and is run right here; a local
    judge is GPU work and goes through SLURM (or runs here with --local).
    """
    jobs, _ = plan(config, args.models, args.benchmarks)
    groups: dict[tuple, dict] = {}
    for job in jobs:
        if job.task not in JUDGE_TASKS:
            continue
        if not job.judge or "judge" not in job.judge:
            print(f"skip     {job.label}: its benchmark names no judge")
            continue
        extra, kind, venv = resolve_judge(job.judge)
        for scorer in job.judge.get("scorers", [DEFAULT_JUDGE_SCORER]):
            for log in successful_logs(job.log_dir) if job.log_dir.exists() else []:
                if has_scorer(log, scorer) and not args.rescore:
                    print(f"scored   {job.label} ({scorer})")
                    continue
                key = (job.model_name, job.benchmark, scorer)
                group = groups.setdefault(
                    key, {"label": f"{job.model_name} x {job.benchmark} [{scorer}]", "logs": [], "kind": kind,
                          "venv": venv, "log_dir": job.log_dir.parent,
                          # --model mockllm/model: `inspect score` rebuilds the log's own
                          # model otherwise, and our providers load weights on construction.
                          "args": ["--model", "mockllm/model", "--scorer", scorer,
                                   "--action", "overwrite" if args.rescore else "append", *extra]},
                )  # fmt: skip
                group["logs"].append(str(log))

    job_ids = []
    for group in groups.values():
        print(f"\n==== score {group['label']}: {len(group['logs'])} log(s) ({group['kind']} judge)", flush=True)
        env = {"VENV_PATH": group["venv"]} if group["venv"] else {}
        if group["kind"] == "local" and args.site:
            env |= {"MELT_JOB_TAG": f"melteval:score:{group['label']}",
                    "MELT_SLURM_OUT": str(group["log_dir"] / "score-slurm-%j.out")}  # fmt: skip
            command = ["infra/runners/submit_score.sh", args.site, *group["logs"], "--", *group["args"]]
            wait_for_room()
        else:
            env.setdefault("VENV_PATH", current_venv())
            command = ["bash", "infra/run_score.sbatch", *group["logs"], "--", *group["args"]]
        code, out = run_command(command, env)
        job_ids += re.findall(r"Submitted batch job (\d+)", out)
        record(config, {"runner": "score", "job": group["label"], "command": command, "exit_code": code})
    print(f"\n{len(groups)} scoring job(s) started.")
    if args.site and args.wait:
        wait_for_jobs(job_ids)
        cmd_report(args, config)


def cmd_report(args, config) -> None:
    """Write the CSV/Excel summary of every successful log into ``<output_dir>/results``."""
    logs, results = config["_output_dir"] / "logs", config["_output_dir"] / "results"
    if not logs.exists():
        print(f"no logs yet under {logs}")
        return
    command = [sys.executable, "projects/baselines/report.py", "--log-root", str(logs), "--out", str(results)]
    code, _ = run_command(command, {})
    if code:
        print("WARNING: the report could not be built (see above).")


def cmd_prefetch(args, config) -> None:
    """Download what the jobs will read -- benchmark splits and baseline weights -- into HF_HOME.

    Run it once from a login node (compute nodes may have no internet). It
    reads each (spec, dataset_id) through the same code a job uses, so exactly
    what a job will ask for is cached.
    """
    from huggingface_hub import snapshot_download

    from melteval.dataset import spec_dataset
    from melteval.providers import family_of
    from melteval.providers.router import HF_MODELS

    print(f"HF_HOME={os.environ.get('HF_HOME', '(unset: ~/.cache/huggingface)')}")
    wanted = [m["model"] for m in config["models"] if not args.models or m["name"] in args.models]
    wanted += [b["judge"] for b in config["benchmarks"] if "judge" in b]
    for model in dict.fromkeys(wanted):
        if family_of(model) != "hf":
            continue
        cls = HF_MODELS[model.removeprefix("melt/hf/")]
        print(f"fetch {model}: {cls.repo}@{cls.revision} ...", flush=True)
        snapshot_download(cls.repo, revision=cls.revision)

    for bench in config["benchmarks"]:
        if args.benchmarks and bench["name"] not in args.benchmarks:
            continue
        if "spec" not in bench:
            print(f"skip  {bench['name']}: a frozen set, nothing to download")
            continue
        spec, reason = resolve_path(bench["spec"], "spec")
        if spec is None:
            print(f"skip  {bench['name']}: {reason}")
            continue
        for dataset_id in dict.fromkeys(s.get("dataset_id") for s in bench["slices"]):
            print(f"fetch {bench['name']}/{dataset_id or '(all)'} ...", flush=True)
            print(f"      {len(spec_dataset(spec, dataset_id=dataset_id))} samples")


COMMANDS = {
    "plan": cmd_plan,
    "run": cmd_run,
    "status": cmd_status,
    "score": cmd_score,
    "report": cmd_report,
    "prefetch": cmd_prefetch,
}


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("config", type=Path, help="Evaluation config YAML (e.g. configs/eval/baselines-v1.yaml).")
    parser.add_argument("--out", help="Output folder, instead of the config's output_dir.")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--site", help="Submit through SLURM with infra/sites/<site>.sh (also loads its variables).")
    where.add_argument("--local", action="store_true", help="Run the jobs one after another on this machine.")
    parser.add_argument("--models", nargs="+", help="Only these models (by name).")
    parser.add_argument("--benchmarks", nargs="+", help="Only these benchmarks (by name).")
    parser.add_argument("--wait", action="store_true", help="run/score on SLURM: wait for the jobs, then report.")
    parser.add_argument("--limit", type=int, help="run: at most N samples per job (a smoke test, in its own folder).")
    parser.add_argument("--rescore", action="store_true", help="score: grade logs that already have scores again.")
    parser.add_argument("-v", "--verbose", action="store_true", help="plan: also print each job's arguments.")
    args = parser.parse_args(argv)

    if args.command in ("run", "score") and not (args.site or args.local):
        parser.error(f"{args.command} needs --site <site> (SLURM) or --local (this machine).")
    if args.site:
        load_site_env(args.site)
    if not (REPO_ROOT / "melteval").is_dir():
        raise SystemExit("scripts/evaluate.py must stay inside the melt-eval repo.")
    sys.path.insert(0, str(REPO_ROOT))

    COMMANDS[args.command](args, load_config(args.config, args.out))


if __name__ == "__main__":
    main()
