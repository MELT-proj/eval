"""Evaluate a set of models on a set of benchmarks, and collect everything in one folder.

    python scripts/evaluate.py plan     CONFIG [--site SITE] [-v]     # what would run; runs nothing
    python scripts/evaluate.py run      CONFIG (--site SITE | --local) [--wait] [--limit N]
    python scripts/evaluate.py status   CONFIG [--site SITE]          # done / queued / failed / missing
    python scripts/evaluate.py score    CONFIG (--site SITE | --local) [--rescore]   # post-hoc: judges, BERTScore, COMET
    python scripts/evaluate.py report   CONFIG                        # CSV + Excel into <output_dir>/results
    python scripts/evaluate.py prefetch CONFIG [--site SITE]          # datasets + baseline weights, from a login node

Every command also takes --out DIR (instead of the config's output_dir) and
--models/--benchmarks NAME... to restrict to some of them.

One job is one (model, benchmark slice). ``run`` only starts jobs that have no
successful log yet, so running it again after a failure retries exactly what is
missing. On a site that sets ``MELT_PACK_GPUS`` (JUPITER: whole 4-GPU nodes), or
with a ``pack:`` block naming ``gpus``, the pending jobs go to SLURM as packs
instead: one job per node, one worker per GPU, the model loaded once per worker
(see "Packs" below). The config format and the output folder are described in
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

#: SLURM --comment of a pack job, followed by its manifest's absolute path.
PACK_PREFIX = "melteval-pack:"

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
    #: Judge-free scorers run after generation by ``score`` (MCIF's official
    #: metrics, COMET): the slice's ``post_scorers``, in the benchmark's ``post_venv``.
    post_scorers: list[str] = field(default_factory=list)
    post_venv: str | None = None

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

                # Optional per-model MCQ prompt template ({question}, {options}), e.g. DeSTA2.5's.
                mcq_instruction = model_cfg.get("mcq_instruction")
                if mcq_instruction and task == "audio_mcq" and family == "hf":
                    args += ["-T", f"mcq_instruction={json.dumps(mcq_instruction)}"]

                args += [str(a) for a in model_cfg.get("args", [])]
                # Optional per-task system prompts (`system:` map, `default` as fallback), e.g. DeSTA2.5's.
                # Passed as a model argument so inspect records it in the log.
                system = model_cfg.get("system") or {}
                system_prompt = system.get(task, system.get("default"))
                if system_prompt is not None:
                    args += ["-M", f"system_prompt={json.dumps(system_prompt)}"]
                args += [str(a) for a in slice_cfg.get("args", [])]
                if task in JUDGE_TASKS:
                    args.append("--no-score")
                post_scorers = [str(s) for s in slice_cfg.get("post_scorers", [])]
                if post_scorers:
                    # So the report can show a missing post-hoc metric as pending.
                    args += ["--metadata", f"melteval_post_scorers={','.join(post_scorers)}"]
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
                    Job(name, model, bench["name"], sname, task, eval_set, log_dir, args, job_env, judge or None,
                        post_scorers, bench.get("post_venv"))
                )  # fmt: skip
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
    """Tags of this user's SLURM jobs still pending or running.

    A pack is queued under one comment, ``melteval-pack:<manifest>``; it stands
    for the tag of every slice its manifest lists.
    """
    if not squeue_available():
        return set()
    result = subprocess.run(
        ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%k"], capture_output=True, text=True, check=False
    )
    return expand_tags(line.strip() for line in result.stdout.splitlines())


def expand_tags(comments) -> set[str]:
    """Slice tags from SLURM comments: a slice job's own tag, or every slice of a pack's manifest."""
    tags = set()
    for comment in comments:
        if comment.startswith(PACK_PREFIX):
            manifest = Path(comment.removeprefix(PACK_PREFIX))
            try:
                tags.update(entry["tag"] for entry in json.loads(manifest.read_text(encoding="utf-8"))["slices"])
            except (OSError, ValueError, KeyError) as exc:
                print(f"  ! unreadable pack manifest {manifest}: {exc}", file=sys.stderr)
        elif comment.startswith("melteval:"):
            tags.add(comment)
    return tags


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


# --------------------------------------------------------------------------- packs
#
# A site that only hands out whole nodes (JUPITER: 4 GH200 each) would bill a
# 4-GPU node for every 1-GPU slice job. A pack is one SLURM job holding many
# slices of one model: infra/run_eval_pack.sbatch starts one worker per GPU,
# each loads the model once and takes slices from the pack until none are left
# (melteval/pack.py). Every slice still writes its own log into its own
# log_dir, so status, report and the watcher see exactly what they see without
# packs, and an unfinished slice is simply missing on the next ``run``.


def pack_settings(config: dict) -> dict | None:
    """The config's ``pack:`` block, completed from the site; ``None`` when slices go one job each.

    ``gpus`` comes from the config or, failing that, ``$MELT_PACK_GPUS`` (the
    site file sets it where nodes are whole); without either there are no
    packs. ``pack: false`` turns them off whatever the site says.
    """
    raw = config.get("pack")
    if raw is False:
        return None
    raw = dict(raw or {})
    gpus = int(raw.get("gpus") or os.environ.get("MELT_PACK_GPUS") or 0)
    if gpus < 1:
        return None
    return {"gpus": gpus, "time": raw.get("time"), "max_slices": int(raw.get("max_slices") or 0)}


#: Arguments that only shape the task or the log, never the model inspect resolves.
_TASK_ONLY_ARGS = {"-T": 1, "--metadata": 1, "--log-format": 1, "--no-score": 0}


def model_key(job: Job) -> tuple[str, ...]:
    """The arguments of *job* that reach ``get_model()``: ``-M`` and generation flags like ``--max-tokens``.

    inspect memoizes a model on these, so two slices that differ here would
    load a second copy of the checkpoint onto the same GPU: they go in
    different packs.
    """
    key, i = [], 0
    while i < len(job.args):
        width = _TASK_ONLY_ARGS.get(job.args[i])
        if width is None:
            key.append(job.args[i])
            i += 1
        else:
            i += 1 + width
    return tuple(key)


def _seconds(value: str | None) -> int:
    """``HH:MM:SS`` (or ``MM:SS``, or ``D-HH:MM:SS``) in seconds; 0 if unset."""
    if not value:
        return 0
    days, _, clock = str(value).rpartition("-")
    seconds = 0
    for part in clock.split(":"):
        seconds = seconds * 60 + int(part)
    return seconds + int(days or 0) * 86400


def make_packs(jobs: list[Job], max_slices: int = 0) -> list[list[Job]]:
    """Group *jobs* into packs: one model, one venv, one set of model arguments each.

    Longest slices first (by their ``time:``), so a worker that draws one does
    not leave the others idle at the end; a group bigger than *max_slices* is
    dealt round-robin into as many packs as it needs, which keeps the long
    slices spread across them.
    """
    groups: dict[tuple, list[Job]] = {}
    for job in jobs:
        groups.setdefault((job.model, job.env.get("VENV_PATH"), model_key(job)), []).append(job)
    packs = []
    for group in groups.values():
        group = sorted(group, key=lambda job: -_seconds(job.env.get("MELT_TIME")))
        count = -(-len(group) // max_slices) if max_slices else 1
        packs += [group[k::count] for k in range(count)]
    return packs


def write_pack(pack: list[Job], config: dict) -> Path:
    """Write the manifest melteval/pack.py reads: what each slice runs, and where its log goes."""
    first = pack[0]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    pack_id = re.sub(r"[^\w.-]", "_", f"{first.model_name}-{stamp}")
    path = config["_output_dir"] / "packs" / f"{pack_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "pack_id": pack_id,
        "model": first.model,
        "slices": [
            {"label": job.label, "tag": job.tag, "eval_set": job.eval_set, "log_dir": str(job.log_dir),
             "args": job.args}
            for job in pack
        ],
    }  # fmt: skip
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return path


def start_pack(pack: list[Job], site: str, settings: dict, config: dict) -> str | None:
    """Submit *pack* as one whole-node SLURM job; return its job id."""
    for job in pack:
        job.log_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_pack(pack, config)
    command = ["infra/runners/submit_pack.sh", site, str(manifest)]
    env = {
        "MELT_GPUS": str(settings["gpus"]),
        "MELT_JOB_TAG": f"{PACK_PREFIX}{manifest}",
        "MELT_SLURM_OUT": str(manifest.with_suffix("")) + "-slurm-%j.out",
    }
    if pack[0].env.get("VENV_PATH"):
        env["VENV_PATH"] = pack[0].env["VENV_PATH"]
    # Never shorter than the pack's longest slice, whatever `pack: time` says.
    times = [settings["time"], *(job.env.get("MELT_TIME") for job in pack)]
    longest = max((t for t in times if t), key=_seconds, default=None)
    if longest:
        env["MELT_TIME"] = str(longest)
    wait_for_room()
    code, out = run_command(command, env)
    job_id = next(iter(re.findall(r"Submitted batch job (\d+)", out)), None)
    record(config, {"runner": f"slurm:{site}", "pack": str(manifest), "jobs": [job.label for job in pack],
                    "model": pack[0].model, "command": command, "slurm_job_id": job_id, "exit_code": code})  # fmt: skip
    return job_id


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
    pending = []
    for job in jobs:
        state, detail = job_state(job, queued)
        print(f"{state:<8} {job.label}  {detail}")
        if args.verbose and state in ("missing", "failed"):
            print(f"         {job.model} {job.eval_set} {' '.join(job.args)}")
        if state in ("missing", "failed"):
            pending.append(job)
    settings = pack_settings(config) if args.site else None
    if settings and pending:
        packs = make_packs(pending, settings["max_slices"])
        print(f"\n`run` would submit {len(pending)} slice(s) as {len(packs)} pack(s), "
              f"{settings['gpus']} GPU(s) each:")  # fmt: skip
        for pack in packs:
            print(f"  pack of {len(pack)}: {pack[0].model_name}  ({', '.join(job.slice for job in pack)})")
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
    pending = []
    for job in jobs:
        state, _ = job_state(job, queued)
        if state in ("done", "queued"):
            print(f"{state:<8} {job.label}")
            continue
        if args.limit:
            job.args += ["-T", f"limit={args.limit}"]
        pending.append(job)

    job_ids, started = [], 0
    settings = pack_settings(config) if args.site else None
    if settings:
        for pack in make_packs(pending, settings["max_slices"]):
            print(f"\n==== start pack of {len(pack)} slice(s) on {settings['gpus']} GPU(s): {pack[0].model_name}",
                  flush=True)  # fmt: skip
            for job in pack:
                print(f"     {job.label}")
            job_id = start_pack(pack, args.site, settings, config)
            started += len(pack)
            if job_id:
                job_ids.append(job_id)
    else:
        for job in pending:
            print(f"\n==== start {job.label}", flush=True)
            job_id = start(job, args.site, config)
            started += 1
            if job_id:
                job_ids.append(job_id)

    in_jobs = f" in {len(job_ids)} SLURM job(s)" if args.site else ""
    print(f"\nstarted {started} slice(s){in_jobs}; output folder: {config['_output_dir']}")
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


@dataclass
class PendingScore:
    """One finished log still missing one post-hoc scorer."""

    job: Job
    scorer: str
    log: Path
    extra: list[str]  #: ``--model-role grader=...`` for a judge, nothing otherwise.
    kind: str  #: ``api`` (runs here) or ``local`` (loads a model: GPU work, SLURM).
    venv: str | None  #: Activate script to score in; ``None``: this script's own.

    @property
    def group(self) -> tuple[str, str, str]:
        """What one scoring job covers: a model's logs on one benchmark, for one scorer."""
        return self.job.model_name, self.job.benchmark, self.scorer

    @property
    def tag(self) -> str:
        """The SLURM --comment of the scoring job for this group."""
        return f"melteval:score:{self.job.model_name}:{self.job.benchmark}:{self.scorer.split('@')[-1]}"


def post_scorers_of(job: Job) -> tuple[list[tuple[str, list[str], str, str | None]], str | None]:
    """The post-hoc scorers *job*'s logs need, as ``(scorer, extra args, kind, venv)``.

    Returns:
        The scorers, and why some could not be resolved (``None`` if all could).
    """
    out = []
    if job.task in JUDGE_TASKS and job.judge and "judge" in job.judge:
        extra, kind, venv = resolve_judge(job.judge)
        out += [(s, extra, kind, venv) for s in job.judge.get("scorers", [DEFAULT_JUDGE_SCORER])]
    if job.post_scorers:
        venv = expand(job.post_venv) if job.post_venv else None
        if job.post_venv and (venv is None or not Path(venv).exists()):
            return out, f"post_venv {job.post_venv!r} not found on this machine"
        # BERTScore and COMET load a model of their own: GPU work.
        out += [(s, [], "local", venv) for s in job.post_scorers]
    return out, None


def pending_scores(jobs: list[Job], rescore: bool = False, verbose: bool = False) -> list[PendingScore]:
    """Every (successful log, post-hoc scorer) of *jobs* not scored yet (all of them with *rescore*)."""
    pending = []
    for job in jobs:
        if job.task in JUDGE_TASKS and not (job.judge and "judge" in job.judge) and not job.post_scorers:
            if verbose:
                print(f"skip     {job.label}: its benchmark names no judge and the slice no post_scorers")
            continue
        scorers, problem = post_scorers_of(job)
        if problem and verbose:
            print(f"skip     {job.label}: {problem}")
        for scorer, extra, kind, venv in scorers:
            for log in successful_logs(job.log_dir) if job.log_dir.exists() else []:
                if has_scorer(log, scorer) and not rescore:
                    if verbose:
                        print(f"scored   {job.label} ({scorer})")
                    continue
                pending.append(PendingScore(job, scorer, log, extra, kind, venv))
    return pending


def cmd_score(args, config) -> None:
    """Run the post-hoc scorers on finished logs: judges (audio_chat), MCIF's official metrics, COMET.

    One job per (model under test, benchmark, scorer), so a judge or metric
    model is loaded once per group. An API judge needs no GPU and is run right
    here; anything that loads a model is GPU work and goes through SLURM (or
    runs here with --local). A group whose scoring job is still queued is not
    submitted again, so this can be called repeatedly (the watcher does).
    """
    jobs, _ = plan(config, args.models, args.benchmarks)
    queued = queued_tags()
    groups: dict[tuple, dict] = {}
    for item in pending_scores(jobs, args.rescore, verbose=True):
        if item.tag in queued:
            print(f"queued   {item.job.label} ({item.scorer})")
            continue
        group = groups.setdefault(
            item.group, {"label": f"{item.job.model_name} x {item.job.benchmark} [{item.scorer}]", "logs": [],
                         "kind": item.kind, "venv": item.venv, "tag": item.tag, "log_dir": item.job.log_dir.parent,
                         # --model mockllm/model: `inspect score` rebuilds the log's own
                         # model otherwise, and our providers load weights on construction.
                         "args": ["--model", "mockllm/model", "--scorer", item.scorer,
                                  "--action", "overwrite" if args.rescore else "append", *item.extra]},
        )  # fmt: skip
        group["logs"].append(str(item.log))

    job_ids = []
    for group in groups.values():
        print(f"\n==== score {group['label']}: {len(group['logs'])} log(s) ({group['kind']})", flush=True)
        env = {"VENV_PATH": group["venv"]} if group["venv"] else {}
        if group["kind"] == "local" and args.site:
            env |= {"MELT_JOB_TAG": group["tag"], "MELT_SLURM_OUT": str(group["log_dir"] / "score-slurm-%j.out")}
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
