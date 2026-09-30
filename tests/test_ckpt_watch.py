"""Tests for the checkpoint watcher's logic (``scripts/ckpt_watch.py``) and its config for evaluate.py."""

from __future__ import annotations

import csv
import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ckpt_watch", ROOT / "scripts" / "ckpt_watch.py")
cw = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cw
_spec.loader.exec_module(cw)


def _touch(path: Path, age_seconds: float = 3600, size: int = 10) -> Path:
    path.write_bytes(b"x" * size)
    mtime = path.stat().st_mtime - age_seconds
    os.utime(path, (mtime, mtime))
    return path


def _state(**steps: str) -> cw.State:
    """A ledger from ``step_100="done"``-style arguments, in the order given."""
    entries = {}
    for seq, (key, status) in enumerate(steps.items()):
        step = int(key.removeprefix("step_"))
        entries[step] = cw.Entry(step, f"/ckpt/step={step}.ckpt", seq, status)
    return cw.State(entries)


class TestDiscovery:
    def test_step_is_read_from_the_name(self):
        assert cw.step_of("step=2784-last.ckpt") == 2784
        assert cw.step_of("epoch=3.ckpt") is None

    def test_finds_only_checkpoints_with_a_step_sorted_by_step(self, tmp_path):
        _touch(tmp_path / "step=4000-last.ckpt")
        _touch(tmp_path / "step=900-last.ckpt")
        _touch(tmp_path / "notes.txt")
        assert [c.step for c in cw.find_checkpoints(tmp_path)] == [900, 4000]

    def test_a_file_still_being_written_is_not_complete(self, tmp_path):
        done = cw.find_checkpoints(_touch(tmp_path / "step=1.ckpt", age_seconds=3600).parent)[0]
        assert cw.is_complete(done, settle_seconds=600)
        fresh = cw.Checkpoint(
            2, _touch(tmp_path / "step=2.ckpt", age_seconds=5), os.stat(tmp_path / "step=2.ckpt").st_mtime
        )
        assert not cw.is_complete(fresh, settle_seconds=600)

    def test_an_empty_file_is_not_complete(self, tmp_path):
        empty = _touch(tmp_path / "step=3.ckpt", size=0)
        assert not cw.is_complete(cw.Checkpoint(3, empty, empty.stat().st_mtime), settle_seconds=0)

    def test_every_n_picks_the_first_then_every_nth(self):
        assert [cw.is_selected(seq, 3) for seq in range(7)] == [True, False, False, True, False, False, True]
        assert all(cw.is_selected(seq, 1) for seq in range(4))


class TestState:
    def test_round_trip_and_next_seq(self, tmp_path):
        state = _state(step_100=cw.DONE, step_200=cw.SKIPPED)
        cw.save_state(state, tmp_path / "s.json")
        loaded = cw.load_state(tmp_path / "s.json")
        assert loaded.entries == state.entries
        assert loaded.next_seq == 2

    def test_missing_file_is_an_empty_ledger(self, tmp_path):
        assert cw.load_state(tmp_path / "none.json").entries == {}


class TestMetrics:
    def test_reads_the_metric_by_step_and_ignores_pending(self, tmp_path):
        path = tmp_path / "results_long.csv"
        rows = [
            ("step-0000100", "lib", "asr-x", "corpus_wer_raw", "0.30"),
            ("step-0000200", "lib", "asr-x", "corpus_wer_raw", "0.20"),
            ("step-0000300", "lib", "asr-x", "corpus_wer_raw", "pending"),
            ("step-0000200", "lib", "asr-x", "preamble_rate", "0.9"),
        ]
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "benchmark", "slice", "metric", "value"])
            writer.writerows(rows)
        assert cw.read_metrics(path, "lib/asr-x/corpus_wer_raw") == {100: 0.30, 200: 0.20}


class TestRetention:
    def test_keeps_newest_last_best_and_milestones(self):
        state = _state(**{f"step_{s}": cw.DONE for s in (1000, 2000, 3000, 4000, 5000, 6000)})
        metrics = {1000: 0.5, 2000: 0.2, 3000: 0.4, 4000: 0.6, 5000: 0.3, 6000: 0.35}
        retention = cw.Retention(keep_last=1, keep_best_k=2, metric="m", mode="min", keep_every=3000)
        keep, delete = cw.plan_retention(state, retention, metrics)
        assert keep == {6000: "newest", 3000: "milestone", 2000: "best", 5000: "best"}
        assert delete == [1000, 4000]

    def test_max_mode_prefers_larger_values(self):
        state = _state(step_1=cw.DONE, step_2=cw.DONE, step_3=cw.DONE)
        keep, delete = cw.plan_retention(
            state, cw.Retention(keep_last=0, keep_best_k=1, metric="m", mode="max"), {1: 0.9, 2: 0.1, 3: 0.5}
        )
        assert set(keep) == {3, 1}  # newest + best
        assert delete == [2]

    def test_unfinished_checkpoints_are_never_deleted(self):
        state = _state(step_1=cw.SELECTED, step_2=cw.SUBMITTED, step_3=cw.FAILED, step_4=cw.DONE, step_5=cw.DONE)
        keep, delete = cw.plan_retention(state, cw.Retention(keep_last=1), {})
        assert {1, 2, 3, 5} <= set(keep)
        assert delete == [4]

    def test_a_never_evaluated_checkpoint_can_be_pruned(self):
        state = _state(step_1=cw.SKIPPED, step_2=cw.DONE, step_3=cw.DONE)
        _, delete = cw.plan_retention(state, cw.Retention(keep_last=1), {})
        assert delete == [1, 2]

    def test_already_deleted_ones_are_ignored(self):
        state = _state(step_1=cw.DELETED, step_2=cw.DONE)
        keep, delete = cw.plan_retention(state, cw.Retention(keep_last=1), {})
        assert keep == {2: "newest"} and delete == []

    def test_empty_ledger(self):
        assert cw.plan_retention(cw.State(), cw.Retention(), {}) == ({}, [])


class TestStepConfig:
    template = {
        "name": "smurf",
        "output_dir": "${EVAL_ROOT}/smurf",
        "models": [{"name": "smurf", "model": "smurf/x", "args": ["-M", "batch_size=8"], "instruction": {"asr": "T"}}],
        "benchmarks": [{"name": "a", "slices": []}, {"name": "b", "slices": []}],
    }

    def test_only_name_and_checkpoint_change(self, tmp_path):
        config = cw.step_config(self.template, 4000, Path("/c/step=4000.ckpt"), tmp_path)
        assert config["models"] == [
            {"name": "step-0004000", "model": "smurf//c/step=4000.ckpt", "args": ["-M", "batch_size=8"],
             "instruction": {"asr": "T"}}
        ]  # fmt: skip
        assert config["output_dir"] == str(tmp_path)
        assert [b["name"] for b in config["benchmarks"]] == ["a", "b"]

    def test_the_template_is_not_mutated(self, tmp_path):
        cw.step_config(self.template, 1, Path("/c/step=1.ckpt"), tmp_path)
        assert self.template["models"][0]["name"] == "smurf"
