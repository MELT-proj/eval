# PolySpeech Dev Subset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 200-per-language PolySpeech `pa_qt` dev benchmark, freeze it, and run it on the Jupiter copy of the battery.

**Architecture:** A new spec (`configs/hf/polyspeech-100-pa_qt-200.yaml`) is the existing `pa_qt` spec with `max_samples: 200` and `seed: 0` on every source. A separate suite (`configs/eval/suites/speechval-polyspeech.yaml`) exposes it as 24 `audio_mcq` slices behind `${POLYSPEECH_PA_QT_SET}`, so `speechval-dev.yaml` and every artemis run stay untouched. The campaign runs on Jupiter against the checkpoints already there, using the shared venv with `PYTHONPATH` overrides; logs return to artemis by rsync.

**Tech Stack:** Python 3.12, pytest, ruff, `melteval` (inspect_ai extension), YAML specs, SLURM on Jupiter.

**Spec:** `docs/superpowers/specs/2026-11-18-polyspeech-dev-design.md`

## Global Constraints

- Python ≥3.10 builtins for types; **do not import from `typing`** (`X | None`, `list[str]`).
- `ruff` clean, line length 119. Add `uv`, not `pip`, where an install is needed.
- Tooling on the workstation: `PY=/mnt/scratch-artemis/giuseppe/venvs/melteval/bin/python`; use `$PY -m pytest` and `$PY -m ruff`.
- Never run GPU work outside SLURM, on any cluster. Jupiter jobs go through `infra/runners/submit_eval.sh` or `scripts/evaluate.py run`.
- Frozen sets reference audio; never copy it. The reconstruction is Jupiter-only and pinned to revision `5c9235323df63e175819d5bee46b82efc6965bbd`.
- Subset values are exact: **200 samples/language, `seed: 0`, 24 languages, 4,800 records**.
- Do not add PolySpeech to `configs/eval/suites/speechval-dev.yaml`. That suite is used on artemis, which has no reconstruction and no export.
- Copy the model stanza (instructions/system prompts) verbatim from `/mnt/scratch-artemis/giuseppe/smurf-sweep/speechval-smurf-sweep.yaml`. Prompt parity is a shipped-bug class; invent nothing here.
- Jupiter storage: `$PROJECT/$USER` and `$SCRATCH/$USER` only. Never `$HOME`.

## Review Focus

1. **A freeze made with a different seed, spec, or reconstruction revision silently produces a different 200 samples.** The run label and `frozen_set.json` must carry the seed and revision; the spec test pins `seed: 0`.
2. **A site without the reconstruction must fail loudly, not skip the benchmark.** The guarded site export plus the separate suite is what protects artemis; a test asserts `speechval-dev.yaml` has no PolySpeech entry.
3. **Passages exceeding the 40 s training window are the norm (~89%).** A low/at-chance accuracy with high `unresolved_rate` is a validity signal, not a model result; the pilot step checks both.
4. **A model may answer with option text instead of a letter.** `mcq_scorer` reports `unresolved_rate`; a high rate means the prompt/parse path, not the model, is broken.
5. **Seven languages (de et hu lt nl ro ru) use a different question set.** Cross-language means are not a parallel split; the docs must say to compare on the question overlap.

---

### Task 1: Define the dev spec, suite, site export, and docs

**Files:**
- Create: `configs/hf/polyspeech-100-pa_qt-200.yaml`
- Create: `configs/eval/suites/speechval-polyspeech.yaml`
- Modify: `infra/sites/jupiter.sh` (after the `speechval-dev` export block, ~line 60)
- Modify: `docs/speechval.md` (table + new section)
- Modify: `docs/polyspeech.md` (dev-subset section)
- Test: `tests/test_freeze.py` (`TestSpeechvalDevSpecs`, after `test_every_suite_slice_has_a_source`)

**Interfaces:**
- Produces: spec sources each carry `max_samples: 200` and `seed: 0`; suite benchmark `polyspeech-100-pa_qt-200` with 24 slices `{task: audio_mcq, dataset_id: polyspeech-pa_qt-<lang>}`; site exports `POLYSPEECH_ROOT` and `POLYSPEECH_PA_QT_SET`.

- [ ] **Step 1: Write the failing tests**

Add to `TestSpeechvalDevSpecs` in `tests/test_freeze.py`:

```python
    def test_polyspeech_dev_spec_is_the_full_spec_with_a_200_cap(self):
        full = load_spec(self.ROOT / "configs/hf/polyspeech-100-pa_qt.yaml")
        dev = load_spec(self.ROOT / "configs/hf/polyspeech-100-pa_qt-200.yaml")
        assert len(full["input_cfg"]) == len(dev["input_cfg"]) == 24
        assert [s["tags"]["dataset_id"] for s in full["input_cfg"]] == [
            s["tags"]["dataset_id"] for s in dev["input_cfg"]
        ]
        for source in dev["input_cfg"]:
            assert source["variant"] == "pa_qt"
            assert source["max_samples"] == 200
            assert source["seed"] == 0
        for full_source, dev_source in zip(full["input_cfg"], dev["input_cfg"]):
            def strip_cap(source: dict) -> dict:
                return {k: v for k, v in source.items() if k not in ("max_samples", "seed")}

            assert strip_cap(full_source) == strip_cap(dev_source)

    def test_polyspeech_suite_has_one_slice_per_language(self):
        import yaml

        dev = load_spec(self.ROOT / "configs/hf/polyspeech-100-pa_qt-200.yaml")
        suite = yaml.safe_load(
            (self.ROOT / "configs/eval/suites/speechval-polyspeech.yaml").read_text()
        )["benchmarks"]
        assert len(suite) == 1
        [bench] = suite
        assert bench["name"] == "polyspeech-100-pa_qt-200"
        assert bench["frozen_set"] == "${POLYSPEECH_PA_QT_SET}"
        assert len(bench["slices"]) == 24
        assert {s["task"] for s in bench["slices"]} == {"audio_mcq"}
        assert len({s["dataset_id"] for s in bench["slices"]}) == 24
        assert {s["dataset_id"] for s in bench["slices"]} == {
            source["tags"]["dataset_id"] for source in dev["input_cfg"]
        }

    def test_polyspeech_is_not_in_the_main_dev_suite(self):
        import yaml

        suite = yaml.safe_load((self.ROOT / "configs/eval/suites/speechval-dev.yaml").read_text())["benchmarks"]
        assert all("polyspeech" not in bench["name"].lower() for bench in suite)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/test_freeze.py -q -k polyspeech`
Expected: 2 failures (missing spec, missing suite); `test_polyspeech_is_not_in_the_main_dev_suite` passes as a guard on the current tree.

- [ ] **Step 3: Create the 200-per-language spec**

```bash
cp configs/hf/polyspeech-100-pa_qt.yaml configs/hf/polyspeech-100-pa_qt-200.yaml
$PY - <<'PY'
from pathlib import Path

path = Path("configs/hf/polyspeech-100-pa_qt-200.yaml")
text = path.read_text()
old_header = "name: polyspeech-100-pa_qt\n"
new_header = (
    "# Dev subset of the same reconstruction: 200 samples per language, seed 0, so the\n"
    "# arithmetic is affordable. Not comparable with full-set PolySpeech numbers.\n"
    "name: polyspeech-100-pa_qt-200\n"
)
assert old_header in text
text = text.replace(old_header, new_header, 1)
assert text.count("    variant: pa_qt\n") == 24
text = text.replace(
    "    variant: pa_qt\n",
    "    variant: pa_qt\n    max_samples: 200\n    seed: 0\n",
)
path.write_text(text)
PY
```

- [ ] **Step 4: Create the separate suite**

Create `configs/eval/suites/speechval-polyspeech.yaml`:

```yaml
# SQA/SLU half of the speechval-dev partition. Separate from speechval-dev.yaml
# on purpose: the reconstruction is Jupiter-only, so folding it into the main
# suite would break every artemis run on an unset ${POLYSPEECH_PA_QT_SET}.
# Paired with configs/hf/polyspeech-100-pa_qt-200.yaml (200 samples/language,
# seed 0). Scored by mcq_scorer (accuracy per dataset_id, unresolved_rate), no judge.

benchmarks:
  - name: polyspeech-100-pa_qt-200
    frozen_set: ${POLYSPEECH_PA_QT_SET}
    slices:
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-bg}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-cs}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-da}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-nl}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-en}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-et}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-fi}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-fr}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-de}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-el}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-hu}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-it}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-lv}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-lt}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-mt}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-pl}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-pt}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-ro}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-sk}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-sl}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-es}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-sv}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-uk}
      - {task: audio_mcq, dataset_id: polyspeech-pa_qt-ru}
```

- [ ] **Step 5: Export the two paths in the Jupiter site file**

Append after the `EUROPARL_ST_DEV_SET` block in `infra/sites/jupiter.sh`:

```bash
# --- PolySpeech (dev SQA/SLU); reconstruction and frozen set live on Jupiter ---
if [[ -d /e/data1/smurf4eu-data/speech/polyspeech-100 ]]; then
    export POLYSPEECH_ROOT="${POLYSPEECH_ROOT:-/e/data1/smurf4eu-data/speech/polyspeech-100}"
fi
_polyspeech_frozen="${POLYSPEECH_FROZEN_ROOT:-${PROJECT}/${USER}/frozen-dev/polyspeech-100-pa_qt-200}"
if [[ -d "$_polyspeech_frozen" ]]; then
    export POLYSPEECH_PA_QT_SET="${POLYSPEECH_PA_QT_SET:-$_polyspeech_frozen}"
fi
```

- [ ] **Step 6: Document the dev row**

In `docs/speechval.md`:
- Add a table row: `| PolySpeech-100 (SQA/SLU) | configs/hf/polyspeech-100-pa_qt-200.yaml | frozen | polyspeech-pa_qt-<lang> (24) | POLYSPEECH_PA_QT_SET |`
- Add a section stating: separate suite `configs/eval/suites/speechval-polyspeech.yaml`; 200 samples/language, seed 0, 4,800 records, ~107 h; a screening subset, not comparable with full PolySpeech; the 40 s training window vs the subset's measured duration distribution (fill the pooled n/median/p95/percent-over-40 from `scripts/frozen_length_stats.py` after the freeze); SMURF-only for now; seven languages use a different question set, so compare on the question overlap.

In `docs/polyspeech.md`, add a "Dev subset" paragraph pointing at the 200 spec and the separate suite, and repeat the "not comparable with full-set numbers" sentence.

- [ ] **Step 7: Run the tests and lint**

Run: `$PY -m pytest tests/test_freeze.py -q -k polyspeech && $PY -m ruff check tests/test_freeze.py`
Expected: 3 passed; ruff "All checks passed!".

- [ ] **Step 8: Commit**

```bash
git add configs/hf/polyspeech-100-pa_qt-200.yaml configs/eval/suites/speechval-polyspeech.yaml \
    infra/sites/jupiter.sh docs/speechval.md docs/polyspeech.md tests/test_freeze.py
git commit -m "speechval: PolySpeech 200/language dev spec and separate suite"
```

---

### Task 2: Frozen-set duration statistics

**Files:**
- Create: `scripts/frozen_length_stats.py`
- Test: `tests/test_frozen_length_stats.py`

**Interfaces:**
- Produces: `python scripts/frozen_length_stats.py <frozen-dir> [--json OUT]`; reads `<frozen-dir>/manifest.jsonl`; prints one row per `dataset_id` plus `pooled`; columns `n mean median p90 p95 min max over40`; `--json` writes `{"pooled": {...}, "by_dataset": {...}}`.
- Percentiles are nearest-rank: `sorted(values)[ceil(pct/100 * n) - 1]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_frozen_length_stats.py`:

```python
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
        capture_output=True, text=True, check=True,
    )
    assert "pooled" in result.stdout and "polyspeech-pa_qt-de" in result.stdout
    stats = json.loads(out.read_text())
    assert stats["pooled"]["n"] == 4
    assert stats["pooled"]["median"] == 25.0
    assert stats["pooled"]["p90"] == 50.0
    assert stats["pooled"]["over40"] == 0.25
    assert stats["by_dataset"]["polyspeech-pa_qt-en"]["n"] == 2
    assert stats["by_dataset"]["polyspeech-pa_qt-de"]["n"] == 2
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_frozen_length_stats.py -q`
Expected: FAIL (the script does not exist).

- [ ] **Step 3: Implement `scripts/frozen_length_stats.py`**

```python
#!/usr/bin/env python3
"""Duration distribution of a frozen set, pooled and per dataset_id.

Usage:
    python scripts/frozen_length_stats.py <frozen-dir> [--json OUT]

Reads ``<frozen-dir>/manifest.jsonl``, skips records without a duration, and
reports n, mean, median, p90, p95, min, max and the fraction above ``--over``
(default 40 s, the SMURF training window). Percentiles are nearest-rank.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


def load_durations(manifest: Path) -> list[tuple[str, float]]:
    """Return ``(dataset_id, duration_s)`` for every record that has a duration."""
    out: list[tuple[str, float]] = []
    with manifest.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            duration = record.get("duration")
            if duration is None:
                continue
            out.append((str(record.get("dataset_id") or ""), float(duration)))
    return out


def nearest_rank(values: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already sorted list."""
    rank = max(1, math.ceil(pct / 100.0 * len(values)))
    return values[rank - 1]


def summarize(values: list[float], over_s: float) -> dict[str, float]:
    """The summary row for one group of durations."""
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0,
                "min": 0.0, "max": 0.0, "over40": 0.0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 2),
        "median": round(statistics.median(ordered), 2),
        "p90": round(nearest_rank(ordered, 90), 2),
        "p95": round(nearest_rank(ordered, 95), 2),
        "min": round(ordered[0], 2),
        "max": round(ordered[-1], 2),
        "over40": round(sum(v > over_s for v in ordered) / len(ordered), 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frozen", type=Path, help="frozen-set directory containing manifest.jsonl")
    parser.add_argument("--json", type=Path, help="write the table as JSON here")
    parser.add_argument("--over", type=float, default=40.0, help="over-limit threshold in seconds")
    args = parser.parse_args(argv)

    rows = load_durations(args.frozen / "manifest.jsonl")
    if not rows:
        sys.exit(f"{args.frozen}: no durations in manifest.jsonl")

    by_dataset: dict[str, list[float]] = {}
    for dataset_id, duration in rows:
        by_dataset.setdefault(dataset_id, []).append(duration)
    table = {
        "over_s": args.over,
        "pooled": summarize([d for _, d in rows], args.over),
        "by_dataset": {k: summarize(v, args.over) for k, v in sorted(by_dataset.items())},
    }

    header = f"{'dataset':32s} {'n':>5s} {'mean':>7s} {'median':>7s} {'p90':>7s} {'p95':>7s} {'min':>6s} {'max':>7s} {'over40':>6s}"
    print(header)
    for name, row in [("pooled", table["pooled"]), *table["by_dataset"].items()]:
        print(f"{name:32s} {row['n']:5d} {row['mean']:7.2f} {row['median']:7.2f} {row['p90']:7.2f} "
              f"{row['p95']:7.2f} {row['min']:6.2f} {row['max']:7.2f} {row['over40']:6.2%}")
    if args.json:
        args.json.write_text(json.dumps(table, indent=1) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the test and lint**

Run: `$PY -m pytest tests/test_frozen_length_stats.py -q && $PY -m ruff check scripts/frozen_length_stats.py tests/test_frozen_length_stats.py`
Expected: 1 passed; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add scripts/frozen_length_stats.py tests/test_frozen_length_stats.py
git commit -m "scripts: duration distribution of a frozen set"
```

---

### Task 3: Sync Jupiter, build the campaign config, freeze the subset

**Files (on Jupiter, under `$PROJECT/$USER`):**
- `eval-polyspeech/` (the repo copy, synced from the workstation)
- `frozen-dev/polyspeech-100-pa_qt-200/` (created by `melteval freeze`)
- `frozen-dev/polyspeech-100-pa_qt-200/length_stats.json` (Task 2 script)
- `eval-polyspeech/configs/eval/speechval-polyspeech.yaml` (campaign config)

**Interfaces:**
- Consumes: the spec and suite from Task 1; `scripts/frozen_length_stats.py` from Task 2.
- Produces: `POLYSPEECH_PA_QT_SET` directory with exactly 4,800 records; a campaign config with 84 `smurf//` model entries and one benchmark (the suite); `length_stats.json` next to the frozen set.

- [ ] **Step 1: Sync the repo and verify the environment**

From the workstation:

```bash
rsync -a --exclude .git --exclude .venv --exclude __pycache__ \
  /home/giuseppe/melt-proj/melt-eval/ jupiter:/e/project1/e-ext-2025e01-100/attanasio1/eval-polyspeech/
```

On Jupiter (login node, no GPU):

```bash
cd /e/project1/e-ext-2025e01-100/attanasio1/eval-polyspeech
source /e/project1/e-ext-2025e01-100/attanasio1/jup_env.sh
cd /e/project1/e-ext-2025e01-100/attanasio1/eval-polyspeech   # jup_env.sh cd's away
export PYTHONPATH=$PWD:/e/project1/e-ext-2025e01-100/speech_stream/speechllm/src
/e/project1/e-ext-2025e01-100/calvo-bartolome1/venvs/smurf-eval/bin/python \
  -c "import fbk_speechllm.prompts, melteval; print(fbk_speechllm.prompts.__file__, melteval.__file__)"
```

Expected: both paths under `eval-polyspeech` and `speech_stream/speechllm/src`.

- [ ] **Step 2: Generate the 84-model campaign config**

Save this as `/tmp/make_polyspeech_config.py` and run it on the workstation with `$PY` (it reads the checkpoint roots over ssh and the stanza from the sweep's config):

```python
import re, subprocess, yaml
from pathlib import Path

P = "/e/project1/e-ext-2025e01-100"
ROOTS = [
    f"/p/scratch/e-ext-2025e01-100/speech_stream/eurollm_speech_16_nodes_run_1_asr_long",
    f"{P}/speech_stream/eurollm_speech_16_nodes_run_1_asr_long/checkpoints",
]
listing = subprocess.run(
    ["ssh", "jupiter", "ls -d " + " ".join(f"{r}/step=*.ckpt" for r in ROOTS)],
    capture_output=True, text=True, check=True,
).stdout.split()
steps = set()
for path in listing:
    m = re.search(r"/step=(\d+)\.ckpt$", path)
    if not m:
        continue
    step = int(m.group(1))
    if 102000 <= step <= 594000 and step % 6000 == 0:
        steps.add(step)
steps.add(600000)
assert len(steps) == 84, sorted(steps)

template = yaml.safe_load(Path("/mnt/scratch-artemis/giuseppe/smurf-sweep/speechval-smurf-sweep.yaml").read_text())
stanza = dict(template["models"][0])
stanza.pop("name"); stanza.pop("model")
stanza["venv"] = f"{P}/calvo-bartolome1/venvs/smurf-eval/bin/activate"

def source(step: int) -> str:
    p = f"/p/scratch/e-ext-2025e01-100/speech_stream/eurollm_speech_16_nodes_run_1_asr_long/step={step}.ckpt"
    e = f"{P}/speech_stream/eurollm_speech_16_nodes_run_1_asr_long/checkpoints/step={step}.ckpt"
    return e if step >= 300000 else p

models = []
for step in sorted(steps):
    entry = {"name": f"eurollm-speech-16n-step-{step:07d}", "model": "smurf/" + source(step), **stanza}
    models.append(entry)

config = {
    "name": "speechval-polyspeech",
    "output_dir": "${EVAL_ROOT}/speechval-polyspeech",
    "models": models,
    "benchmarks": f"{P}/attanasio1/eval-polyspeech/configs/eval/suites/speechval-polyspeech.yaml",
    "pack": {"time": "03:00:00", "max_slices": 24, "gpus": 4},
}
Path("/tmp/speechval-polyspeech.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
print(len(models), "models")
```

Expected: `84 models`. Then:

```bash
scp /tmp/speechval-polyspeech.yaml \
  jupiter:/e/project1/e-ext-2025e01-100/attanasio1/eval-polyspeech/configs/eval/speechval-polyspeech.yaml
```

- [ ] **Step 3: Freeze the subset and measure it**

```bash
cd /e/project1/e-ext-2025e01-100/attanasio1/eval-polyspeech
source /e/project1/e-ext-2025e01-100/attanasio1/jup_env.sh
cd /e/project1/e-ext-2025e01-100/attanasio1/eval-polyspeech
export PYTHONPATH=$PWD:/e/project1/e-ext-2025e01-100/speech_stream/speechllm/src
export POLYSPEECH_ROOT=/e/data1/smurf4eu-data/speech/polyspeech-100
F=/e/project1/e-ext-2025e01-100/attanasio1/frozen-dev/polyspeech-100-pa_qt-200
V=/e/project1/e-ext-2025e01-100/calvo-bartolome1/venvs/smurf-eval/bin
"$V/melteval" freeze configs/hf/polyspeech-100-pa_qt-200.yaml -o "$F"
"$V/python" scripts/frozen_length_stats.py "$F" --json "$F/length_stats.json"
```

(`$V/melteval` is the console script for `melteval.cli:main`; `$V/python` is the shared venv's interpreter. Do not use the login shell's bare `python`, which lacks the package's dependencies.)

- [ ] **Step 4: Verify the freeze**

```bash
/e/project1/e-ext-2025e01-100/calvo-bartolome1/venvs/smurf-eval/bin/python - "$F" <<'PY'
import json, sys
from pathlib import Path
d = json.loads((Path(sys.argv[1]) / "frozen_set.json").read_text())
assert d["total_samples"] == 4800, d["total_samples"]
assert len(d["sources"]) == 24, len(d["sources"])
assert {s["kept"] for s in d["sources"]} == {200}, sorted({s["kept"] for s in d["sources"]})
assert d["seed"] == 0, d["seed"]
print("ok", d["total_samples"], "samples,", round(d["total_hours"], 1), "h")
PY
```

Expected: `ok 4800 samples, ~107 h`. Record the pooled line and the per-language rows from `length_stats.json`; they go into `docs/speechval.md` in Task 5. If the freeze reveals a different total (e.g. a language with fewer than 200 valid rows), stop and report it before continuing.

---

### Task 4: Smoke run, then the two-checkpoint pilot

**Files:**
- Logs: `$SCRATCH/$USER/eval-runs/speechval-polyspeech/logs/...` (Jupiter)

**Interfaces:**
- Consumes: the frozen set and campaign config from Task 3.
- Produces: a 20-sample smoke log and two full 24-slice checkpoints; measured pack wall time and submit→start wait.

- [ ] **Step 1: Smoke one slice on 102k (1 GPU, 20 samples)**

```bash
cd /e/project1/e-ext-2025e01-100/attanasio1/eval-polyspeech
source /e/project1/e-ext-2025e01-100/attanasio1/jup_env.sh
cd /e/project1/e-ext-2025e01-100/attanasio1/eval-polyspeech
export PYTHONPATH=$PWD:/e/project1/e-ext-2025e01-100/speech_stream/speechllm/src
export POLYSPEECH_ROOT=/e/data1/smurf4eu-data/speech/polyspeech-100
export POLYSPEECH_PA_QT_SET=/e/project1/e-ext-2025e01-100/attanasio1/frozen-dev/polyspeech-100-pa_qt-200
export VENV_PATH=/e/project1/e-ext-2025e01-100/calvo-bartolome1/venvs/smurf-eval/bin/activate
export V=/e/project1/e-ext-2025e01-100/calvo-bartolome1/venvs/smurf-eval/bin
export MELT_TIME=00:30:00
infra/runners/submit_eval.sh jupiter \
  "smurf//p/scratch/e-ext-2025e01-100/speech_stream/eurollm_speech_16_nodes_run_1_asr_long/step=102000.ckpt" \
  "$POLYSPEECH_PA_QT_SET" -T task_filter=audio_mcq -T dataset_id=polyspeech-pa_qt-en -T limit=20 -M batch_size=4
```

Wait for the job, then read its log. Expected: `status: success`, an `accuracy_polyspeech-pa_qt-en` score, and `unresolved_rate` below 0.5. If the import or the prompt path fails, stop here — do not submit the pilot.

- [ ] **Step 2: Submit the pilot (2 checkpoints × 24 slices, 4-GPU packs)**

```bash
cd /e/project1/e-ext-2025e01-100/attanasio1/eval-polyspeech
# same env exports as Step 1
"$V/python" scripts/evaluate.py run configs/eval/speechval-polyspeech.yaml \
  --models eurollm-speech-16n-step-0102000 eurollm-speech-16n-step-0594000 \
  --site jupiter
```

Expected: two pack submissions, each 24 slices, `--gpus-per-node=4`, `--time=03:00:00`; the command exits after submitting (`MAX_QUEUED` is unset on Jupiter, so `run` does not block).

- [ ] **Step 3: Record the measurement**

Poll until both packs leave the queue (`"$V/python" scripts/evaluate.py status ... --site jupiter`), then:

- Pack wall time: first and last `[pack gpuN]` lines of the pack `slurm-*.out`.
- Queue wait: `sacct -j <jobid> -X -o JobID,Submit,Start,Elapsed`.
- Validity: for each of the 24 slices, `accuracy_polyspeech-pa_qt-*` vs 0.25 and `unresolved_rate`.

Write the four numbers (wall time per checkpoint, wait, macro accuracy, mean unresolved rate) into the campaign folder as `pilot.md`. If macro accuracy is at 0.25 with a high unresolved rate, stop and report — the passages are beyond the model, and the plan does not proceed to Task 5.

---

### Task 5: Scale, sync logs, and record the numbers

**Files:**
- Modify: `docs/speechval.md` (measured duration distribution and pilot numbers)
- Logs: `/mnt/scratch-artemis/giuseppe/melt-data/eval-runs/speechval-polyspeech/` (artemis)

**Interfaces:**
- Consumes: the pilot from Task 4; the campaign config from Task 3.
- Produces: one log per (checkpoint, language), a `results_long.csv`, and updated docs.

- [ ] **Step 1: Submit the remaining checkpoints**

Same environment and cwd as Task 4:

```bash
"$V/python" scripts/evaluate.py run configs/eval/speechval-polyspeech.yaml --site jupiter
```

The two pilot checkpoints are already complete and are skipped by the planner. Expected: 82 packs (2,016 slices minus the pilot's 48), 4 GPUs each. The command exits after submitting.

If the pilot's cost or validity argues for the smaller scope, pass `--models` with the 20 checkpoints that already have ST (the list is in the spec's "Open decision" section) instead of the full config.

- [ ] **Step 2: Confirm completeness**

```bash
"$V/python" scripts/evaluate.py status configs/eval/speechval-polyspeech.yaml --site jupiter
```

Expected: no `missing` or `failed` lines; `84 done` (or the chosen subset).

- [ ] **Step 3: Build the campaign table**

```bash
"$V/python" scripts/evaluate.py report configs/eval/speechval-polyspeech.yaml --site jupiter
```

Expected: `results/results_long.csv` with 84 models × 24 slices × accuracy rows.

- [ ] **Step 4: Sync the logs to artemis**

```bash
rsync -a jupiter:/e/scratch/e-ext-2025e01-100/attanasio1/eval-runs/speechval-polyspeech/ \
  /mnt/scratch-artemis/giuseppe/melt-data/eval-runs/speechval-polyspeech/
```

The worklog scans `speechval-*` and will ingest the runs; the matrix has no `audio_mcq` family yet, so visibility is the run list and inspector.

- [ ] **Step 5: Put the measured numbers in the docs**

In `docs/speechval.md`, fill the PolySpeech section with:
- the pooled duration table from `length_stats.json` (n, median, p95, percent over 40 s), and one line on the per-language range;
- the pilot's wall time per 4-GPU pack and queue wait;
- the macro accuracy and mean `unresolved_rate`, with the chance-level interpretation.

- [ ] **Step 6: Commit and run the suite**

```bash
git add docs/speechval.md
git commit -m "docs: PolySpeech dev subset length distribution and pilot numbers"
$PY -m pytest tests -q --deselect tests/test_evaluate.py::TestWaitForRoom::test_waits_until_the_queue_drops_below_the_limit
$PY -m ruff check .
```

Expected: the unit suite passes (environmental deselect as usual), ruff clean.
