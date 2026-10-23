"""Tests for ``scripts/evaluate.py`` and the report it builds (``projects/baselines/report.py``)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluate = _load("evaluate", "scripts/evaluate.py")


def _config(tmp_path, **overrides):
    """A config over real specs in the repo, a fake MELT checkpoint and a fake SMURF one."""
    (tmp_path / "runs" / "melt-run").mkdir(parents=True)
    (tmp_path / "smurf.ckpt").touch()
    config = {
        "name": "test",
        "_output_dir": tmp_path / "out",
        "models": [
            {
                "name": "qwen",
                "model": "melt/hf/qwen3_omni",
                "instruction": {"asr": "Transcribe this audio."},
                "skip": ["mcif-long-*"],
            },
            {
                "name": "desta",
                "model": "melt/hf/desta25",
                "instruction": {"asr": "Transcribe this audio."},
                "system": {"default": "Focus.", "asr": "Focus. Respond directly."},
                "mcq_instruction": "{question} {options}",
            },
            {"name": "melt", "model": f"melt/{tmp_path}/runs/*/"},
            {"name": "smurf", "model": f"smurf/{tmp_path}/smurf.ckpt", "instruction": {"asr": "Transcribe: "}},
            {"name": "unset", "model": "melt/${MELTEVAL_TEST_UNSET_CKPT}"},
        ],
        "benchmarks": [
            {
                "name": "air-bench",
                "spec": "configs/hf/air-bench.yaml",
                "judge": "melt/hf/qwen3_omni",
                "slices": [
                    {"task": "audio_mcq", "dataset_id": "air-bench-foundation-speech"},
                    {"task": "audio_chat", "dataset_id": "air-bench-chat-speech"},
                ],
            },
            {
                "name": "mcif",
                "spec": "configs/hf/mcif.yaml",
                "slices": [{"task": "chunked_asr", "dataset_id": "mcif-long-fixed-en"}],
            },
            {
                "name": "librispeech",
                "spec": "configs/hf/librispeech-hf-test-clean.yaml",
                "needs_instruction": True,
                "slices": [
                    {"task": "asr", "dataset_id": "librispeech-test-clean", "time": "02:00:00"},
                    {"task": "st"},
                ],
            },
        ],
    }
    config.update(overrides)
    return config


def _jobs(tmp_path, **overrides):
    jobs, skipped = evaluate.plan(_config(tmp_path, **overrides))
    return {(job.model_name, job.slice): job for job in jobs}, skipped


class TestResolveModel:
    def test_a_baseline_keeps_its_name(self):
        assert evaluate.resolve_model("melt/hf/qwen2_audio") == ("melt/hf/qwen2_audio", None)

    def test_a_checkpoint_becomes_absolute_and_a_glob_must_match_one_path(self, tmp_path):
        """inspect chdirs into melteval/ while building the task, so a relative path would point elsewhere."""
        (tmp_path / "a").mkdir()
        assert evaluate.resolve_model(f"melt/{tmp_path}/*/") == (f"melt/{tmp_path}/a", None)
        (tmp_path / "b").mkdir()
        model, reason = evaluate.resolve_model(f"melt/{tmp_path}/*/")
        assert model is None and "matches 2 paths" in reason

    @pytest.mark.parametrize(
        ("model", "why"),
        [
            ("melt/hf/nonexistent", "unknown baseline"),
            ("melt/vllm/some/model", "not implemented"),
            ("qwen2_audio//snapshots/abc", "not a melt/"),
            ("melt/${MELTEVAL_TEST_UNSET_CKPT}", "not set"),
        ],
    )
    def test_what_cannot_run_says_why(self, model, why):
        resolved, reason = evaluate.resolve_model(model)
        assert resolved is None and why in reason


class TestPlan:
    def test_one_job_per_model_and_slice_that_can_run(self, tmp_path):
        jobs, skipped = _jobs(tmp_path)
        assert sorted(jobs) == [
            ("desta", "asr-librispeech-test-clean"),
            ("desta", "audio_chat-air-bench-chat-speech"),
            ("desta", "audio_mcq-air-bench-foundation-speech"),
            ("desta", "chunked_asr-mcif-long-fixed-en"),
            ("melt", "asr-librispeech-test-clean"),
            ("melt", "audio_chat-air-bench-chat-speech"),
            ("melt", "audio_mcq-air-bench-foundation-speech"),
            ("melt", "chunked_asr-mcif-long-fixed-en"),
            ("melt", "st"),
            ("qwen", "asr-librispeech-test-clean"),
            ("qwen", "audio_chat-air-bench-chat-speech"),
            ("qwen", "audio_mcq-air-bench-foundation-speech"),
            ("smurf", "asr-librispeech-test-clean"),
            ("smurf", "audio_chat-air-bench-chat-speech"),
            ("smurf", "audio_mcq-air-bench-foundation-speech"),
            ("smurf", "chunked_asr-mcif-long-fixed-en"),
        ]
        assert any(line.startswith("model unset:") for line in skipped)
        assert any("qwen x mcif/chunked_asr-mcif-long-fixed-en" in line and "skip list" in line for line in skipped)
        assert any("qwen x librispeech/st" in line and "no instruction" in line for line in skipped)

    def test_the_instruction_is_a_yaml_quoted_string_and_melt_gets_none(self, tmp_path):
        """inspect parses -T values as YAML; MELT takes its prompt from the checkpoint's training config."""
        jobs, _ = _jobs(tmp_path)
        assert 'instruction="Transcribe this audio."' in jobs["qwen", "asr-librispeech-test-clean"].args
        assert not any(a.startswith("instruction=") for a in jobs["melt", "asr-librispeech-test-clean"].args)
        assert not any(
            a.startswith("instruction=") for a in jobs["qwen", "audio_mcq-air-bench-foundation-speech"].args
        )

    def test_a_system_prompt_is_a_model_argument_per_task_with_a_default(self, tmp_path):
        """Only a model entry with `system:` gets one; a task not listed falls back to `default`."""
        jobs, _ = _jobs(tmp_path)
        assert 'system_prompt="Focus. Respond directly."' in jobs["desta", "asr-librispeech-test-clean"].args
        assert 'system_prompt="Focus."' in jobs["desta", "audio_mcq-air-bench-foundation-speech"].args
        for key in (("qwen", "asr-librispeech-test-clean"), ("melt", "asr-librispeech-test-clean")):
            assert not any(a.startswith("system_prompt=") for a in jobs[key].args)

    def test_an_mcq_instruction_goes_only_to_the_multiple_choice_slices_of_its_model(self, tmp_path):
        jobs, _ = _jobs(tmp_path)
        mcq = jobs["desta", "audio_mcq-air-bench-foundation-speech"]
        assert 'mcq_instruction="{question} {options}"' in mcq.args
        for key in (("desta", "asr-librispeech-test-clean"), ("qwen", "audio_mcq-air-bench-foundation-speech")):
            assert not any(a.startswith("mcq_instruction=") for a in jobs[key].args)

    def test_judge_tasks_are_generated_unscored(self, tmp_path):
        jobs, _ = _jobs(tmp_path)
        assert "--no-score" in jobs["qwen", "audio_chat-air-bench-chat-speech"].args
        assert "--no-score" not in jobs["qwen", "audio_mcq-air-bench-foundation-speech"].args

    def test_each_job_logs_json_to_its_own_folder_and_names_itself(self, tmp_path):
        jobs, _ = _jobs(tmp_path)
        job = jobs["qwen", "asr-librispeech-test-clean"]
        assert job.log_dir == tmp_path / "out" / "logs" / "qwen" / "librispeech" / "asr-librispeech-test-clean"
        assert job.env == {"OUTPUT_DIR": str(job.log_dir), "MELT_TIME": "02:00:00"}
        assert job.model == "melt/hf/qwen3_omni"
        assert job.args[job.args.index("--log-format") :][:2] == ["--log-format", "json"]
        assert "melteval_model=qwen" in job.args and "melteval_campaign=test" in job.args

    def test_the_configs_skip_drops_slices_by_benchmark_name_or_dataset_id_for_every_model(self, tmp_path):
        jobs, skipped = _jobs(tmp_path, skip=["air-bench", "librispeech-test-*"])
        assert {s for _, s in jobs} == {"chunked_asr-mcif-long-fixed-en", "st"}
        assert any("smurf x air-bench/audio_chat-air-bench-chat-speech" in line and "config's skip" in line
                   for line in skipped)  # fmt: skip

    def test_a_models_skip_matches_a_benchmark_name_too(self, tmp_path):
        """A slice without a dataset_id (`{task: asr}`) can only be skipped by its benchmark's name."""
        config = _config(tmp_path)
        config["models"][0]["skip"] = ["librispeech"]
        jobs, _ = evaluate.plan(config)
        assert not any(j.model_name == "qwen" and j.benchmark == "librispeech" for j in jobs)
        assert any(j.model_name == "smurf" and j.benchmark == "librispeech" for j in jobs)


class TestSuite:
    """`benchmarks:` as the path of a suite file shared by several configs."""

    def _write(self, tmp_path, benchmarks):
        import yaml

        config_file = tmp_path / "c.yaml"
        config_file.write_text(yaml.safe_dump({"name": "t", "output_dir": str(tmp_path / "out"), "models": [],
                                               "benchmarks": benchmarks}))  # fmt: skip
        return config_file

    def test_a_suite_path_is_read_from_the_repo_root(self, tmp_path):
        config = evaluate.load_config(self._write(tmp_path, "configs/eval/suites/v1.yaml"), None)
        assert "voxpopuli" in [b["name"] for b in config["benchmarks"]]

    def test_an_inline_list_is_kept_as_written(self, tmp_path):
        config = evaluate.load_config(self._write(tmp_path, [{"name": "a", "slices": []}]), None)
        assert config["benchmarks"] == [{"name": "a", "slices": []}]

    def test_a_missing_suite_stops_with_its_path(self, tmp_path):
        with pytest.raises(SystemExit, match="nope.yaml"):
            evaluate.load_config(self._write(tmp_path, "configs/eval/suites/nope.yaml"), None)

    def test_the_snapshot_holds_the_expanded_suite(self, tmp_path):
        import yaml

        config = evaluate.load_config(self._write(tmp_path, "configs/eval/suites/v1.yaml"), None)
        evaluate.snapshot_config(config)
        snapshot = yaml.safe_load((tmp_path / "out" / "config.yaml").read_text())
        assert snapshot["benchmarks"] == config["benchmarks"]
        assert not any(k.startswith("_") for k in snapshot)

    @pytest.mark.parametrize("name", ["baselines-v1", "smurf-v1"])
    def test_the_real_configs_share_the_suite(self, name, tmp_path):
        config = evaluate.load_config(ROOT / "configs" / "eval" / f"{name}.yaml", str(tmp_path))
        suite = evaluate.load_benchmarks("configs/eval/suites/v1.yaml")
        assert config["benchmarks"] == suite


class TestResolveJudge:
    def test_a_local_baseline_is_loaded_text_only(self):
        args, kind, venv = evaluate.resolve_judge({"judge": "melt/hf/qwen3_omni", "judge_args": {"batch_size": 2}})
        assert kind == "local" and venv is None
        assert json.loads(args[1].removeprefix("grader=")) == {
            "model": "melt/hf/qwen3_omni",
            "model_args": {"batch_size": 2, "text_only": True},
        }

    def test_an_api_judge_is_passed_through(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert evaluate.resolve_judge({"judge": "openai/gpt-4o"}) == (
            ["--model-role", "grader=openai/gpt-4o"],
            "api",
            None,
        )

    def test_an_api_judge_without_its_key_stops_before_scoring(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
            evaluate.resolve_judge({"judge": "openai/gpt-4o"})

    @pytest.mark.parametrize("judge", ["smurf//m.ckpt", "melt//ckpt"])
    def test_a_judge_that_cannot_take_text_only_is_refused(self, judge):
        with pytest.raises(SystemExit, match="text-only"):
            evaluate.resolve_judge({"judge": judge})


class TestPostScorers:
    """Judge-free scorers run after generation (MCIF's official BERTScore, COMET)."""

    QA = "melteval/mcif_scoring.py@mcif_official_qa_scorer"

    def _config(self, tmp_path, venv="act"):
        (tmp_path / "act").touch()
        bench = {
            "name": "mcif",
            "spec": "configs/hf/mcif.yaml",
            "post_venv": str(tmp_path / venv),
            "slices": [
                {"task": "chunked_asr", "dataset_id": "mcif-short-fixed-en"},
                {"task": "audio_chat", "dataset_id": "mcif-short-fixed-en", "post_scorers": [self.QA]},
            ],
        }
        return _config(tmp_path, benchmarks=[bench])

    def test_a_slice_carries_its_post_scorers_and_names_them_in_the_log(self, tmp_path):
        jobs, _ = evaluate.plan(self._config(tmp_path))
        qa = next(j for j in jobs if j.model_name == "qwen" and j.task == "audio_chat")
        asr = next(j for j in jobs if j.model_name == "qwen" and j.task == "chunked_asr")
        assert qa.post_scorers == [self.QA] and asr.post_scorers == []
        assert "--no-score" in qa.args and "--no-score" not in asr.args
        assert f"melteval_post_scorers={self.QA}" in qa.args

    def test_skip_post_scorers_neither_runs_nor_declares_them(self, tmp_path):
        """Declared but never run would show as pending in the report forever."""
        config = self._config(tmp_path)
        config["skip_post_scorers"] = [self.QA]
        jobs, _ = evaluate.plan(config)
        qa = next(j for j in jobs if j.model_name == "qwen" and j.task == "audio_chat")
        assert qa.post_scorers == []
        assert not any(a.startswith("melteval_post_scorers=") for a in qa.args)

    def _score(self, tmp_path, monkeypatch, config, queued=frozenset()):
        from types import SimpleNamespace as NS

        logs = {}
        for job in evaluate.plan(config)[0]:
            job.log_dir.mkdir(parents=True)
            logs[job.log_dir] = [job.log_dir / "x.json"]
        monkeypatch.setattr(evaluate, "successful_logs", lambda d: logs.get(d, []))
        monkeypatch.setattr(evaluate, "has_scorer", lambda log, scorer: False)
        monkeypatch.setattr(evaluate, "queued_tags", lambda: set(queued))
        commands = []
        monkeypatch.setattr(evaluate, "run_command", lambda c, env: commands.append((c, env)) or (0, ""))
        args = NS(models=["qwen"], benchmarks=None, rescore=False, site="mollete", wait=False)
        evaluate.cmd_score(args, config)
        return commands

    def test_score_runs_them_without_a_judge_in_their_venv(self, tmp_path, monkeypatch):
        config = self._config(tmp_path)
        [(command, env)] = self._score(tmp_path, monkeypatch, config)
        assert command[:2] == ["infra/runners/submit_score.sh", "mollete"]
        assert command[command.index("--scorer") + 1] == self.QA
        assert "--model-role" not in command
        assert env["VENV_PATH"] == str(tmp_path / "act")

    def test_a_group_already_queued_is_not_submitted_again(self, tmp_path, monkeypatch):
        config = self._config(tmp_path)
        tag = "melteval:score:qwen:mcif:mcif_official_qa_scorer"
        assert self._score(tmp_path, monkeypatch, config, queued={tag}) == []

    def test_a_missing_venv_skips_rather_than_scoring_in_the_wrong_one(self, tmp_path, monkeypatch):
        jobs, _ = evaluate.plan(self._config(tmp_path, venv="nope"))
        qa = next(j for j in jobs if j.model_name == "qwen" and j.task == "audio_chat")
        scorers, problem = evaluate.post_scorers_of(qa)
        assert scorers == [] and "not found" in problem


class TestRun:
    def test_a_limited_run_goes_to_its_own_folder(self, tmp_path, monkeypatch):
        """A 5-sample log must never count as done for the real run."""
        config_file = tmp_path / "c.yaml"
        config = _config(tmp_path)
        config.pop("_output_dir")
        config["output_dir"] = str(tmp_path / "out")
        config["models"] = config["models"][:1]
        config["benchmarks"] = config["benchmarks"][:1]
        import yaml

        config_file.write_text(yaml.safe_dump(config))
        started = []
        monkeypatch.setattr(evaluate, "start", lambda job, site, config: started.append(job))
        monkeypatch.setattr(evaluate, "cmd_report", lambda args, config: None)

        evaluate.main(["run", str(config_file), "--local", "--limit", "5"])

        assert {job.log_dir.parts[-5] for job in started} == {"smoke-limit5"}
        assert all(job.args[-2:] == ["-T", "limit=5"] for job in started)
        assert (tmp_path / "out" / "smoke-limit5" / "config.yaml").exists()

    def test_run_needs_a_site_or_local(self, tmp_path):
        with pytest.raises(SystemExit):
            evaluate.main(["run", str(tmp_path / "c.yaml")])


class TestReport:
    """Logs from a real ``inspect_ai.eval()``, read back into the tables."""

    @pytest.fixture
    def report(self):
        pytest.importorskip("pandas")
        return _load("baselines_report", "projects/baselines/report.py")

    def _eval(self, log_dir, model, task, score=True, target="yes", **metadata):
        from inspect_ai import Task
        from inspect_ai import eval as inspect_eval
        from inspect_ai.dataset import MemoryDataset, Sample
        from inspect_ai.scorer import accuracy, includes

        scorer = includes() if score else None
        [log] = inspect_eval(
            Task(
                dataset=MemoryDataset([Sample(input="q", target=target, id="s-0")]),
                scorer=scorer,
                metrics=[accuracy()],
            ),
            model="mockllm/model",
            display="none",
            log_dir=str(log_dir),
            score=score,
            metadata={
                "melteval_model": model,
                "melteval_benchmark": "air-bench",
                "melteval_slice": f"{task}-x",
                "melteval_task": task,
                **metadata,
            },
        )
        assert log.status == "success"
        return log

    def test_models_become_rows_and_unscored_slices_show_pending(self, tmp_path, report):
        self._eval(tmp_path / "a", "model-a", "audio_chat", target="Default output")
        self._eval(tmp_path / "b", "model-b", "audio_chat", score=False)

        runs = report.read_runs([tmp_path])
        long = report.metrics_long(runs)
        table = report.summary_table(long)

        assert list(table.index) == ["model-a", "model-b"]
        assert table.loc["model-a", ("air-bench", "audio_chat-x", "accuracy")] == 1.0
        assert table.loc["model-b", ("air-bench", "audio_chat-x", "score")] == report.PENDING

    def test_a_declared_post_scorer_not_run_yet_shows_pending_beside_the_scores(self, tmp_path, report):
        self._eval(tmp_path / "a", "model-a", "audio_chat", target="Default output",
                   melteval_post_scorers="melteval/mcif_scoring.py@mcif_official_qa_scorer")  # fmt: skip
        table = report.summary_table(report.metrics_long(report.read_runs([tmp_path])))
        assert table.loc["model-a", ("air-bench", "audio_chat-x", "accuracy")] == 1.0
        assert table.loc["model-a", ("air-bench", "audio_chat-x", "mcif_official_qa_scorer")] == report.PENDING

    def test_the_newest_log_of_a_slice_wins(self, tmp_path, report):
        self._eval(tmp_path / "old", "model-a", "audio_chat", target="nope")
        self._eval(tmp_path / "new", "model-a", "audio_chat", target="Default output")

        long = report.metrics_long(report.read_runs([tmp_path]))
        assert report.summary_table(long).loc["model-a", ("air-bench", "audio_chat-x", "accuracy")] == 1.0

    def test_the_workbook_has_a_summary_a_sheet_per_benchmark_and_the_runs(self, tmp_path, report):
        openpyxl = pytest.importorskip("openpyxl")
        self._eval(tmp_path / "a", "model-a", "audio_chat", target="Default output")

        runs = report.read_runs([tmp_path / "a"])
        written = report.write(runs, report.metrics_long(runs), tmp_path / "out")

        assert openpyxl.load_workbook(written["results.xlsx"]).sheetnames == ["summary", "air-bench", "runs"]
        assert written["results_long.csv"].exists()
        assert "model-a" in written["summary.csv"].read_text()

    def test_skipped_samples_are_counted_for_a_local_hf_run_only(self, monkeypatch, report):
        """A clip past the model's window fails on its own; the runs sheet says how many."""
        from types import SimpleNamespace as NS

        samples = [
            NS(error=NS(message="Qwen2-Audio: audio of 31.6 s exceeds its 30 s window; sample not evaluated")),
            NS(error=NS(message="CUDA out of memory")),
            NS(error=None),
        ]
        monkeypatch.setattr(report, "read_eval_log_samples", lambda *a, **k: iter(samples))

        hf = NS(eval=NS(model="melt/hf/qwen2_audio"))
        other = NS(eval=NS(model="openai/gpt-4o"))
        assert report._count_skipped("log.eval", hf) == 1
        assert report._count_skipped("log.eval", other) is None


class TestWaitForRoom:
    """The QOS caps submitted jobs per user (Bocconi: 30); sbatch past it fails outright."""

    def _queue(self, monkeypatch, lengths):
        import subprocess
        import time
        from types import SimpleNamespace

        polls = iter(lengths)
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(stdout="job\n" * next(polls))

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(time, "sleep", lambda seconds: None)
        return calls

    def test_waits_until_the_queue_drops_below_the_limit(self, monkeypatch):
        monkeypatch.setenv("MAX_QUEUED", "28")
        calls = self._queue(monkeypatch, [30, 28, 27])
        evaluate.wait_for_room()
        assert len(calls) == 3

    def test_no_limit_set_means_no_polling(self, monkeypatch):
        monkeypatch.delenv("MAX_QUEUED", raising=False)
        calls = self._queue(monkeypatch, [])
        evaluate.wait_for_room()
        assert calls == []


class TestPacks:
    """Whole-node sites: one SLURM job per model (and model arguments), one worker per GPU."""

    def test_no_site_setting_means_no_packs(self, monkeypatch):
        monkeypatch.delenv("MELT_PACK_GPUS", raising=False)
        assert evaluate.pack_settings({"pack": {"time": "06:00:00"}}) is None

    def test_the_site_turns_packs_on_and_the_config_can_turn_them_off(self, monkeypatch):
        monkeypatch.setenv("MELT_PACK_GPUS", "4")
        assert evaluate.pack_settings({"pack": {"time": "06:00:00", "max_slices": 20}}) == {
            "gpus": 4, "time": "06:00:00", "max_slices": 20,
        }  # fmt: skip
        assert evaluate.pack_settings({}) == {"gpus": 4, "time": None, "max_slices": 0}
        assert evaluate.pack_settings({"pack": False}) is None
        monkeypatch.setenv("MELT_PACK_GPUS", "0")
        assert evaluate.pack_settings({"pack": {}}) is None
        assert evaluate.pack_settings({"pack": {"gpus": 2}})["gpus"] == 2

    def test_slices_that_would_load_a_different_model_go_in_different_packs(self, tmp_path):
        """inspect memoizes a model on -M and generation flags: mixing them would load two copies on one GPU."""
        jobs, _ = _jobs(tmp_path)
        desta = [job for (name, _), job in jobs.items() if name == "desta"]
        packs = evaluate.make_packs(desta)
        # DeSTA's system prompt is a -M argument and differs between asr and the rest.
        assert len(packs) == 2
        assert all(len({evaluate.model_key(job) for job in pack}) == 1 for pack in packs)

    def test_task_and_log_arguments_do_not_split_a_pack(self, tmp_path):
        jobs, _ = _jobs(tmp_path)
        smurf = [job for (name, _), job in jobs.items() if name == "smurf"]
        assert len(smurf) > 1
        assert len(evaluate.make_packs(smurf)) == 1
        assert "--max-tokens" in evaluate.model_key(
            evaluate.Job("m", "melt/hf/x", "b", "s", "asr", "e", tmp_path, ["-T", "a=1", "--max-tokens", "4096"])
        )

    def test_max_slices_splits_round_robin_longest_first(self, tmp_path):
        def job(name, time=None):
            env = {"MELT_TIME": time} if time else {}
            return evaluate.Job("m", "smurf/x.ckpt", "b", name, "asr", "e", tmp_path / name, [], env)

        jobs = [job("a"), job("b"), job("long1", "08:00:00"), job("c"), job("long2", "02:00:00")]
        packs = evaluate.make_packs(jobs, max_slices=3)
        assert [[j.slice for j in pack] for pack in packs] == [["long1", "a", "c"], ["long2", "b"]]

    def test_a_queued_pack_stands_for_every_slice_it_holds(self, tmp_path):
        jobs, _ = _jobs(tmp_path)
        smurf = [job for (name, _), job in jobs.items() if name == "smurf"]
        manifest = evaluate.write_pack(smurf, {"_output_dir": tmp_path / "out"})
        tags = evaluate.expand_tags([f"{evaluate.PACK_PREFIX}{manifest}", "melteval:other:b:s", "unrelated"])
        assert tags == {job.tag for job in smurf} | {"melteval:other:b:s"}
        assert all(evaluate.job_state(job, tags)[0] == "queued" for job in smurf)

    def test_a_pack_is_given_its_longest_slice_time(self, tmp_path, monkeypatch):
        jobs, _ = _jobs(tmp_path)
        smurf = [job for (name, _), job in jobs.items() if name == "smurf"]
        seen = {}

        def fake_run(command, env, log_file=None):
            seen.update(env, command=command)
            return 0, "Submitted batch job 42\n"

        monkeypatch.setattr(evaluate, "run_command", fake_run)
        monkeypatch.delenv("MAX_QUEUED", raising=False)
        config = {"_output_dir": tmp_path / "out"}
        settings = {"gpus": 4, "time": "01:00:00", "max_slices": 0}
        assert evaluate.start_pack(smurf, "jupiter", settings, config) == "42"
        assert seen["MELT_TIME"] == "02:00:00"  # librispeech's slice time beats the pack's 1 h
        assert seen["MELT_GPUS"] == "4"
        assert seen["command"][:2] == ["infra/runners/submit_pack.sh", "jupiter"]
        assert seen["MELT_JOB_TAG"] == f"{evaluate.PACK_PREFIX}{seen['command'][2]}"
