"""Tests for ``melteval/pack.py``: the per-GPU worker of a whole-node pack."""

from __future__ import annotations

import json

from melteval import pack


def _entry(tmp_path, eval_set, args=()):
    return {"label": "l", "tag": "t", "eval_set": str(eval_set), "log_dir": str(tmp_path / "logs"), "args": list(args)}


class TestInspectArgv:
    """Must build the command infra/run_eval.sbatch builds for the same slice."""

    def test_a_directory_is_a_frozen_set_and_a_yaml_a_spec(self, tmp_path):
        (tmp_path / "frozen").mkdir()
        (tmp_path / "spec.yaml").touch()
        argv = pack.inspect_argv("smurf/x.ckpt", _entry(tmp_path, tmp_path / "frozen"))
        assert argv[:6] == ["eval", pack.TASK, "--model", "smurf/x.ckpt", "--log-dir", str(tmp_path / "logs")]
        assert ["-T", f"frozen_set={tmp_path / 'frozen'}"] == argv[6:8]
        argv = pack.inspect_argv("smurf/x.ckpt", _entry(tmp_path, tmp_path / "spec.yaml"))
        assert ["-T", f"spec={tmp_path / 'spec.yaml'}"] == argv[6:8]

    def test_prompt_style_follows_the_family_unless_given(self, tmp_path):
        (tmp_path / "s.yaml").touch()
        argv = pack.inspect_argv("smurf/x.ckpt", _entry(tmp_path, tmp_path / "s.yaml", ["-T", "task_filter=asr"]))
        assert argv[-2:] == ["-T", "prompt_style=smurf"]
        argv = pack.inspect_argv("smurf/x.ckpt", _entry(tmp_path, tmp_path / "s.yaml", ["-T", "prompt_style=melt"]))
        assert argv.count("-T") == 2 and "prompt_style=smurf" not in argv

    def test_baselines_tolerate_some_failed_samples(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FAIL_ON_ERROR", raising=False)
        (tmp_path / "s.yaml").touch()
        argv = pack.inspect_argv("melt/hf/qwen2_audio", _entry(tmp_path, tmp_path / "s.yaml"))
        assert argv[-4:] == ["-T", "prompt_style=hf", "--fail-on-error", "0.05"]
        assert "--fail-on-error" not in pack.inspect_argv("smurf/x.ckpt", _entry(tmp_path, tmp_path / "s.yaml"))


class TestRun:
    def test_each_slice_is_claimed_by_exactly_one_worker(self, tmp_path):
        claims = tmp_path / "claims"
        claims.mkdir()
        assert pack.claim(claims, 0) is True
        assert pack.claim(claims, 0) is False
        assert pack.claim(claims, 1) is True

    def test_a_failing_slice_does_not_stop_the_others(self, tmp_path, monkeypatch):
        (tmp_path / "s.yaml").touch()
        manifest = tmp_path / "p.json"
        slices = [_entry(tmp_path / str(i), tmp_path / "s.yaml") for i in range(3)]
        manifest.write_text(json.dumps({"pack_id": "p", "model": "smurf/x.ckpt", "slices": slices}))
        ran = []

        def fake_slice(argv):
            ran.append(argv)
            if len(ran) == 1:
                raise RuntimeError("CUDA out of memory")
            return 0

        monkeypatch.setattr(pack, "run_slice", fake_slice)
        monkeypatch.setattr(pack, "log_status", lambda log_dir, since: "success")
        monkeypatch.setenv("SLURM_JOB_ID", "7")
        assert pack.run(manifest, "gpu0") == 1
        assert len(ran) == 3
        # A second worker on the same job finds nothing left to claim.
        assert pack.run(manifest, "gpu1") == 0 and len(ran) == 3
        assert "CUDA out of memory" in (tmp_path / "0" / "logs" / "pack-7-gpu0.out").read_text()
