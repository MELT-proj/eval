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
                "spec": "configs/librispeech-hf-test-clean.yaml",
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
            ("melt/hf/whisper", "unknown baseline"),
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

    def _eval(self, log_dir, model, task, score=True, target="yes"):
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

    def test_truncated_samples_are_counted_for_a_run_that_could_truncate(self, tmp_path, report):
        """truncate_long_audio is an explicit choice; the runs sheet says how often it bit."""
        from inspect_ai import Task
        from inspect_ai import eval as inspect_eval
        from inspect_ai.dataset import MemoryDataset, Sample
        from inspect_ai.model import ModelOutput

        cut = ModelOutput.from_content(model="mockllm/model", content="x")
        cut.metadata = {"audio_truncated_from_seconds": 31.6}
        whole = ModelOutput.from_content(model="mockllm/model", content="y")
        inspect_eval(
            Task(dataset=MemoryDataset([Sample(input="q", target="x", id=f"s-{i}") for i in range(2)])),
            model="mockllm/model",
            model_args={"custom_outputs": [cut, whole], "truncate_long_audio": True},
            display="none",
            log_dir=str(tmp_path),
            max_connections=1,
            metadata={"melteval_model": "m", "melteval_benchmark": "b", "melteval_slice": "s", "melteval_task": "asr"},
        )

        [count] = report.read_runs([tmp_path])["truncated_samples"]
        assert count == 1


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
