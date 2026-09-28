"""Tests for the baseline matrix: ``scripts/run_matrix.py`` and ``projects/baselines/report.py``.
"""

from __future__ import annotations

import importlib.util
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


run_matrix = _load("run_matrix", "scripts/run_matrix.py")


def _matrix(tmp_path, **overrides):
    ckpt = tmp_path / "hf" / "snapshots" / "abc123"
    ckpt.mkdir(parents=True)
    matrix = {
        "name": "test-matrix",
        "log_root": str(tmp_path / "logs"),
        "models": [
            {
                "name": "qwen",
                "provider": "qwen3_omni",
                "checkpoint": str(tmp_path / "hf" / "snapshots" / "*"),
                "instruction": {"asr": "Transcribe this audio."},
                "skip": ["mcif-long-*"],
            },
            {"name": "melt", "provider": "melt", "checkpoint": "${MELTEVAL_TEST_UNSET_CKPT}"},
        ],
        "benchmarks": [
            {
                "name": "air-bench",
                "spec": "configs/hf/air-bench.yaml",
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
                "slices": [{"task": "asr", "dataset_id": "librispeech-test-clean"}, {"task": "st"}],
            },
        ],
    }
    matrix.update(overrides)
    return matrix


class TestPlan:
    def _plan(self, tmp_path, **overrides):
        return run_matrix.plan(_matrix(tmp_path, **overrides), "bocconi")

    def test_a_glob_checkpoint_resolves_to_its_one_snapshot(self, tmp_path):
        jobs, _ = self._plan(tmp_path)
        assert {job.checkpoint for job in jobs} == {str(tmp_path / "hf" / "snapshots" / "abc123")}

    def test_an_unset_checkpoint_skips_the_model_with_a_reason(self, tmp_path):
        jobs, skipped = self._plan(tmp_path)
        assert all(job.model == "qwen" for job in jobs)
        assert any("model melt" in line and "unset variable" in line for line in skipped)

    def test_skip_patterns_and_missing_instructions_are_reported(self, tmp_path):
        jobs, skipped = self._plan(tmp_path)
        assert [job.slice for job in jobs] == [
            "audio_mcq-air-bench-foundation-speech",
            "audio_chat-air-bench-chat-speech",
            "asr-librispeech-test-clean",
        ]
        assert any("mcif-long-fixed-en" in line and "skip list" in line for line in skipped)
        assert any("librispeech/st" in line and "no instruction" in line for line in skipped)

    def test_the_instruction_is_passed_as_a_yaml_quoted_string(self, tmp_path):
        """inspect parses -T values as YAML; an unquoted 'Transcribe: ' would be a mapping."""
        jobs, _ = self._plan(tmp_path)
        [asr] = [job for job in jobs if job.task == "asr"]
        assert 'instruction="Transcribe this audio."' in asr.args

    def test_benchmark_prompts_get_no_instruction_argument(self, tmp_path):
        jobs, _ = self._plan(tmp_path)
        [mcq] = [job for job in jobs if job.task == "audio_mcq"]
        assert not any(arg.startswith("instruction=") for arg in mcq.args)

    def test_judge_tasks_are_generated_unscored(self, tmp_path):
        jobs, _ = self._plan(tmp_path)
        by_task = {job.task: job for job in jobs}
        assert "--no-score" in by_task["audio_chat"].args
        assert "--no-score" not in by_task["audio_mcq"].args

    def test_each_job_logs_to_its_own_directory_and_names_itself(self, tmp_path):
        jobs, _ = self._plan(tmp_path)
        [mcq] = [job for job in jobs if job.task == "audio_mcq"]
        assert mcq.env["OUTPUT_DIR"] == str(tmp_path / "logs" / "qwen" / "air-bench" / mcq.slice)
        assert mcq.env["MELTEVAL_PROVIDER"] == "qwen3_omni"
        assert "melteval_model=qwen" in mcq.args
        assert "melteval_task=audio_mcq" in mcq.args



class TestScoringPlan:
    """How judge-only slices are graded after generation."""

    def _matrix(self, tmp_path):
        matrix = _matrix(tmp_path)
        matrix["benchmarks"][0]["score"] = {"judge": "qwen"}
        return matrix

    def test_a_judge_named_from_the_matrix_is_loaded_text_only(self, tmp_path):
        import json

        args, _ = run_matrix.resolve_judge("qwen", self._matrix(tmp_path), "bocconi")
        assert args[0] == "--model-role"
        role = json.loads(args[1].removeprefix("grader="))
        assert role["model"] == f"qwen3_omni/{tmp_path / 'hf' / 'snapshots' / 'abc123'}"
        assert role["model_args"] == {"text_only": True}

    def test_an_api_judge_is_passed_through(self, tmp_path):
        args, venv = run_matrix.resolve_judge("openai/gpt-4o", self._matrix(tmp_path), "bocconi")
        assert args == ["--model-role", "grader=openai/gpt-4o"] and venv is None

    def test_a_judge_that_cannot_take_text_only_is_refused(self, tmp_path):
        matrix = self._matrix(tmp_path)
        matrix["models"].append({"name": "smurf", "provider": "smurf", "checkpoint": str(tmp_path)})
        with pytest.raises(SystemExit, match="text-only"):
            run_matrix.resolve_judge("smurf", matrix, "bocconi")

    def test_the_benchmark_score_block_reaches_its_judged_slices(self, tmp_path):
        jobs, _ = run_matrix.plan(self._matrix(tmp_path), "bocconi")
        [chat] = [job for job in jobs if job.task == "audio_chat"]
        assert chat.score == {"judge": "qwen"}

    def test_a_slice_can_add_scorers_over_its_benchmark(self, tmp_path):
        """MCIF long carries SUM as well as QA; short has no SUM reference at all."""
        matrix = self._matrix(tmp_path)
        matrix["benchmarks"][0]["slices"][1]["scorers"] = ["a.py@qa", "a.py@sum"]
        jobs, _ = run_matrix.plan(matrix, "bocconi")
        [chat] = [job for job in jobs if job.task == "audio_chat"]
        assert chat.score == {"judge": "qwen", "scorers": ["a.py@qa", "a.py@sum"]}


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
        run_matrix.wait_for_room()
        assert len(calls) == 3

    def test_no_limit_set_means_no_polling(self, monkeypatch):
        monkeypatch.delenv("MAX_QUEUED", raising=False)
        calls = self._queue(monkeypatch, [])
        run_matrix.wait_for_room()
        assert calls == []
