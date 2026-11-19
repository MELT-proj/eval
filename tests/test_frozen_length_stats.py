"""The duration table quoted next to the PolySpeech dev subset."""

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parent.parent / "scripts" / "frozen_length_stats.py"


def test_reports_pooled_and_per_dataset(tmp_path):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    records = [
        {"sample_key": "00-000000", "dataset_id": "polyspeech-pa_qt-en", "duration": 10.0},
        {"sample_key": "00-000001", "dataset_id": "polyspeech-pa_qt-en", "duration": 20.0},
        {"sample_key": "01-000000", "dataset_id": "polyspeech-pa_qt-de", "duration": 30.0},
        {"sample_key": "01-000001", "dataset_id": "polyspeech-pa_qt-de", "duration": 50.0},
        {"sample_key": "01-000002", "dataset_id": "polyspeech-pa_qt-de"},
    ]
    (frozen / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    out = tmp_path / "stats.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(frozen), "--json", str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "pooled" in result.stdout and "polyspeech-pa_qt-de" in result.stdout
    stats = json.loads(out.read_text())
    assert stats["pooled"]["n"] == 4
    assert stats["pooled"]["median"] == 25.0
    assert stats["pooled"]["p90"] == 50.0
    assert stats["pooled"]["over40"] == 0.25
    assert stats["by_dataset"]["polyspeech-pa_qt-en"]["n"] == 2
    assert stats["by_dataset"]["polyspeech-pa_qt-de"]["n"] == 2
