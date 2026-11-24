"""Tests for the RedVox prefetch helper."""

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_prefetch():
    spec = importlib.util.spec_from_file_location(
        "prefetch_redvox", ROOT / "scripts/prefetch_redvox.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_prefetch_uses_the_pinned_revision_and_audio_patterns(monkeypatch):
    prefetch = _load_prefetch()
    seen = {}

    def fake_snapshot(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "/cache/redvox"

    monkeypatch.setattr(prefetch, "snapshot_download", fake_snapshot)

    assert prefetch.main() == "/cache/redvox"
    assert seen["args"] == ("FBK-MT/RedVox",)
    assert seen["kwargs"]["repo_type"] == "dataset"
    assert seen["kwargs"]["revision"] == "fca218d440f6b8ed4508bea2eee6b4fb5a913e22"
    assert "*/audio/*.wav" in seen["kwargs"]["allow_patterns"]
