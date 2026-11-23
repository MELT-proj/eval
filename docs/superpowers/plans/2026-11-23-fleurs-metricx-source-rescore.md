# FLEURS MetricX source backfill — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rescore MetricX with full `(src, mt, ref)` triples and per-sample values for all 381 FLEURS ST logs in the worklog, and make every future FLEURS eval carry the source.

**Architecture:** `metricx_scorer` becomes a per-sample scorer whose source comes from the sample metadata, falling back to a frozen-set manifest keyed by `(dataset_id, sample_key)`; a focused backfill driver re-runs each log's recorded scorers in one `inspect score --action overwrite` pass with the manifest in `MELTEVAL_SOURCE_MANIFEST`, gated by a CPU preflight that refuses any slice below 100% source coverage. The artemis FLEURS frozen sets are rebuilt from this checkout so future runs need none of this.

**Tech Stack:** Python 3.12, inspect_ai 0.3.254, transformers/torch (MetricX-24 hybrid XXL bf16), lhotse, SLURM, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-11-23-fleurs-metricx-source-rescore-design.md`

## Global Constraints

- Do **not** import from `typing`; use builtins (`X | None`, `list[str]`).
- `ruff`, line length 119. Type hints on signatures, docstrings on public functions/classes.
- Never run GPU work outside SLURM, even a smoke test. CPU-only work (freezing, tests, lint) is fine directly.
- Never write anything sizeable under `/mnt/home/giuseppe`.
- `inspect score --action overwrite` must never be run with a single `--scorer`: it deletes that job's other scorers. The backfill passes **no** `--scorer`.
- Any `melteval freeze` runs as `PYTHONPATH=. melteval freeze …` from the repo root; the venv's editable install points at another checkout.
- CoVoST2 and Europarl logs are not touched. SMURF/EuroLLM-speech rows are not touched.
- Frozen-set paths: `/mnt/scratch-artemis/giuseppe/frozen/speechval-dev/{fleurs24-st-enx-dev,fleurs24-st-xen-dev-full}`.
- Logs to rescore: `speechval-fondue` (7 models × 44), `speechval-canary` (44), `speechval-qwen3-omni` (29). Benchmark names: `fleurs24-st-xen-dev`, `fleurs24-st-enx-dev`.
- The metrics venv is `${METRICS_VENV_PATH}` = `/mnt/scratch-artemis/giuseppe/venvs/metrics-eval/bin/activate`.

## Review Focus

- A log joined against the *other* FLEURS set: `sample_key` is `<source_index>-<ordinal>` and collides, so a key-only lookup reports 100% coverage while scoring every sample against the wrong language. The lookup key and the preflight must both include `dataset_id` (Tasks 1, 3).
- `--action overwrite` dropping `st_scorer`: the new metricx value must not cost BLEU. The command shape and the verify step pin this (Task 4).
- Empty completions: `Score(value=nan)` must not poison the corpus mean or break the log round-trip, and `metricx_source_coverage` must still count the sample as sourced (Task 1).
- Re-freezing changing sample identity (`sample_key`/`audio`/`target`/`cut_id`), orphaning every existing log (Task 6).
- Re-running the backfill after a partial run: appending `metricx_scorer` a second time becomes `metricx_scorer-1`, not a replacement. Already-rescored logs must be skipped (Tasks 3, 4).

---

### Task 1: Per-sample MetricX with a source-manifest fallback

**Files:**
- Modify: `melteval/rescore.py` (metricx section, lines 191–298)
- Test: `tests/test_rescore.py` (replace `TestMetricxScorerThroughRealInspect`)

**Interfaces:**
- Consumes: `melteval.manifest.resolve_frozen_set`, `read_manifest`, `EvalRecord` (fields `dataset_id`, `sample_key`, `source_text`, `src_lang`, `tgt_lang`).
- Produces:
  - `SOURCE_MANIFEST_ENV = "MELTEVAL_SOURCE_MANIFEST"`
  - `@dataclass(frozen=True) class SourceManifest: path: Path; records: dict[tuple[str, str], EvalRecord]`
  - `load_source_manifest(path: str | Path) -> SourceManifest`
  - `metricx_scorer(source_manifest: str | None = None) -> Scorer`
  - metrics `metricx()`, `metricx_coverage()`, `metricx_source_coverage()`
  - `metricx_predict(inputs: list[str], batch_size: int = 16) -> list[float]` keeps its signature.

- [ ] **Step 1: Write the failing tests**

In `tests/test_rescore.py`, replace `TestMetricxScorerThroughRealInspect` with:

```python
class TestMetricxScorerThroughRealInspect:
    """``metricx_scorer`` under a real ``inspect_ai.eval()``, with the model replaced by a recorder."""

    def test_per_sample_values_mean_and_source_coverage(self, tmp_path, monkeypatch):
        from inspect_ai import Task
        from inspect_ai import eval as inspect_eval
        from inspect_ai.dataset import MemoryDataset, Sample
        import melteval.rescore as rescore

        seen = []
        monkeypatch.setattr(rescore, "metricx_predict", lambda inputs: seen.extend(inputs) or [2.0])
        dataset = MemoryDataset(
            [
                Sample(input="x", target="Pani Przewodnicząca.", id="a", metadata={"source_text": "Madame la Présidente."}),
                Sample(input="x", target="Guten Tag.", id="b", metadata={}),
                Sample(input="x", target="Tschüss.", id="c", metadata={"source_text": "Au revoir."}),
                Sample(input="x", target="Danke.", id="d", metadata={"source_text": "Merci."}),
            ]
        )
        contents = ("Pani Przewodnicząca.", "Guten Tag.", "Tschüss.", "")
        outputs = [ModelOutput.from_content(model="mockllm/model", content=c) for c in contents]
        [log] = inspect_eval(
            Task(dataset=dataset, scorer=rescore.metricx_scorer()),
            model="mockllm/model",
            model_args={"custom_outputs": outputs},
            display="none",
            log_dir=str(tmp_path),
        )
        assert log.status == "success"
        metrics = log.results.scores[0].metrics
        assert metrics["metricx"].value == pytest.approx(2.0)
        assert metrics["metricx_coverage"].value == pytest.approx(3 / 4)
        assert metrics["metricx_source_coverage"].value == pytest.approx(3 / 4)
        assert len(seen) == 3
        assert sorted(seen) == sorted(
            [
                "source: Madame la Présidente. candidate: Pani Przewodnicząca. reference: Pani Przewodnicząca.",
                "source:  candidate: Guten Tag. reference: Guten Tag.",
                "source: Au revoir. candidate: Tschüss. reference: Tschüss.",
            ]
        )
        values = {s.id: s.scores["metricx_scorer"].value for s in log.samples}
        assert values["a"] == pytest.approx(2.0)
        assert values["d"] == -1.0  # no completion: the out-of-range UNSCORED sentinel
```

Add a second class with the manifest fallback:

```python
class TestMetricxSourceManifest:
    def _manifest(self, tmp_path, records):
        from melteval.manifest import MANIFEST_NAME
        path = tmp_path / "set"
        path.mkdir()
        with open(path / MANIFEST_NAME, "w") as fh:
            for r in records:
                fh.write(r.to_json() + "\n")
        return str(path)

    def _record(self, key, dataset_id, source_text, src_lang="fr", tgt_lang="en"):
        from melteval.manifest import AudioLocator, EvalRecord
        return EvalRecord(
            sample_key=key, task="st", target="ref", audio=AudioLocator(kind="shar", params={}),
            lang=tgt_lang, src_lang=src_lang, tgt_lang=tgt_lang, dataset_id=dataset_id,
            source_text=source_text,
        )

    def _run(self, tmp_path, monkeypatch, scorer, samples):
        from inspect_ai import Task
        from inspect_ai import eval as inspect_eval
        from inspect_ai.dataset import MemoryDataset
        import melteval.rescore as rescore

        monkeypatch.setattr(rescore, "metricx_predict", lambda inputs: [1.0])
        outputs = [ModelOutput.from_content(model="mockllm/model", content="out") for _ in samples]
        [log] = inspect_eval(
            Task(dataset=MemoryDataset(samples), scorer=scorer),
            model="mockllm/model", model_args={"custom_outputs": outputs}, display="none",
            log_dir=str(tmp_path),
        )
        return log

    def test_manifest_fills_a_missing_source(self, tmp_path, monkeypatch):
        from inspect_ai.dataset import Sample
        import melteval.rescore as rescore

        manifest = self._manifest(tmp_path, [self._record("00-000000", "fleurs-fr_en-dev", "Bonjour.")])
        scorer = rescore.metricx_scorer(source_manifest=manifest)
        log = self._run(tmp_path, monkeypatch, scorer, [
            Sample(input="x", target="ref", id="00-000000",
                   metadata={"dataset_id": "fleurs-fr_en-dev", "src_lang": "fr", "tgt_lang": "en"}),
        ])
        assert log.results.scores[0].metrics["metricx_source_coverage"].value == 1.0
        meta = log.samples[0].scores["metricx_scorer"].metadata
        assert meta["metricx_triple"]["src"] == "Bonjour."
        assert meta["metricx_source"] == "manifest"

    def test_same_key_under_another_dataset_is_not_used(self, tmp_path, monkeypatch):
        from inspect_ai.dataset import Sample
        import melteval.rescore as rescore

        manifest = self._manifest(tmp_path, [self._record("00-000000", "fleurs-es_en-dev", "Hola.")])
        scorer = rescore.metricx_scorer(source_manifest=manifest)
        log = self._run(tmp_path, monkeypatch, scorer, [
            Sample(input="x", target="ref", id="00-000000",
                   metadata={"dataset_id": "fleurs-fr_en-dev", "src_lang": "fr", "tgt_lang": "en"}),
        ])
        assert log.results.scores[0].metrics["metricx_source_coverage"].value == 0.0

    def test_manifest_language_mismatch_fails_the_eval(self, tmp_path, monkeypatch):
        from inspect_ai.dataset import Sample
        import melteval.rescore as rescore

        manifest = self._manifest(tmp_path, [self._record("00-000000", "fleurs-fr_en-dev", "Bonjour.", src_lang="de")])
        scorer = rescore.metricx_scorer(source_manifest=manifest)
        log = self._run(tmp_path, monkeypatch, scorer, [
            Sample(input="x", target="ref", id="00-000000",
                   metadata={"dataset_id": "fleurs-fr_en-dev", "src_lang": "fr", "tgt_lang": "en"}),
        ])
        assert log.status == "error"
        assert "src_lang" in str(log.error.message)

    def test_env_var_is_the_fallback(self, tmp_path, monkeypatch):
        from inspect_ai.dataset import Sample
        import melteval.rescore as rescore

        manifest = self._manifest(tmp_path, [self._record("00-000000", "fleurs-fr_en-dev", "Bonjour.")])
        monkeypatch.setenv(rescore.SOURCE_MANIFEST_ENV, manifest)
        scorer = rescore.metricx_scorer()
        log = self._run(tmp_path, monkeypatch, scorer, [
            Sample(input="x", target="ref", id="00-000000",
                   metadata={"dataset_id": "fleurs-fr_en-dev", "src_lang": "fr", "tgt_lang": "en"}),
        ])
        assert log.results.scores[0].metrics["metricx_source_coverage"].value == 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_rescore.py -q -k "Metricx"`.
Expected: FAIL — `metricx_source_coverage` missing from the metrics dict / `SOURCE_MANIFEST_ENV` not defined / `TypeError: metricx_scorer() got an unexpected keyword argument`.

- [ ] **Step 3: Implement in `melteval/rescore.py`**

Add near the MetricX constants:

```python
SOURCE_MANIFEST_ENV = "MELTEVAL_SOURCE_MANIFEST"

@dataclass(frozen=True)
class SourceManifest:
    path: Path
    records: dict[tuple[str, str], EvalRecord]

def load_source_manifest(path: str | Path) -> SourceManifest
```

`load_source_manifest` resolves via `resolve_frozen_set(path)`, reads with `read_manifest`, and keys on `(record.dataset_id, record.sample_key)`.

Change `metricx_predict` so the tokenizer/model load once per process: module-level `_MODEL: tuple | None`, a `threading.Lock`, and a `_metricx_model()` helper that returns `(tokenizer, model, device, pad)`. Keep the body's length-sorted batching and the `logit 250089` read-out exactly as they are.

Replace `metricx()`/`metricx_coverage()`/`metricx_scorer()`:

```python
@metric
def metricx() -> Metric:
    """System-level MetricX-24 error: mean of the per-sample scores (nan skipped)."""

@metric
def metricx_coverage() -> Metric:
    """Fraction of samples that were scorable (non-empty completion)."""

@metric
def metricx_source_coverage() -> Metric:
    """Fraction of samples whose MetricX input had a non-empty source."""
```

`metricx()` annotates `calculate(scores: list[SampleScore]) -> float`, keeps `sample_score.score.value` in [0, 25], returns `nan` if none (a metric's own NaN return is not the sentinel — it is not a sample score). `metricx_coverage` counts `_metricx_triple(s) is not None`. `metricx_source_coverage` counts `bool((s.score.metadata or {}).get("metricx_source_ok"))`.

`metricx_scorer(source_manifest: str | None = None)`:

```python
@scorer(metrics=[metricx(), metricx_coverage(), metricx_source_coverage()])
def metricx_scorer(source_manifest: str | None = None) -> Scorer:
    """MetricX-24 hybrid XXL, per sample, from source, completion and target. ..."""
```

Read the manifest once, at factory time, from `source_manifest or os.environ.get(SOURCE_MANIFEST_ENV)`. In `score()`: resolve the source (metadata `source_text` if non-empty, else the manifest entry for `(metadata["dataset_id"], state.sample_id)`, else `""`); raise `ValueError` mentioning `src_lang` if a manifest hit's `(src_lang, tgt_lang)` differs from the sample's; score one input via `metricx_predict([...])[0]`; return `Score(value=rescore.UNSCORED, ...)` (`-1.0`) when the completion is empty, because inspect reads a root-level `nan` as its "unscored" sentinel and drops the sample from every metric. Metadata keys: `metricx_triple`, `metricx_source` (`"sample"|"manifest"|"manifest-miss"|"none"`), `metricx_source_ok` (bool), `metricx_manifest` (str or None).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_rescore.py -q`.
Expected: PASS (all, including the existing COMET/triple tests).

- [ ] **Step 5: Commit**

```bash
git add melteval/rescore.py tests/test_rescore.py
git commit -m "rescore: score MetricX per sample, with a frozen-set source fallback"
```

---

### Task 2: Make a silently-empty `source_text_field` visible at freeze time

**Files:**
- Modify: `melteval/freeze.py:102-128` (`read_sources`)
- Test: `tests/test_freeze.py`

**Interfaces:**
- Consumes: `SourceResult.records`, `EvalRecord.source_text`.
- Produces: `frozen_set.json`'s `sources[i]["source_text_missing"]: int` (present only when the source sets `source_text_field`), and one `logger.warning` when the field resolves to nothing for every record.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_freeze.py`:

```python
class TestSourceTextCoverage:
    def test_counts_missing_source_text_and_warns_when_all_are_empty(self, tmp_path, fake_reader, caplog):
        fake_reader.per_source = {0: [_record("a"), _record("b"), _record("c")]}
        spec = {"name": "t", "input_cfg": [
            {"type": "fake", "source_text_field": "supervisions.0.text"},
        ]}
        with caplog.at_level("WARNING"):
            summary = freeze(spec, tmp_path)
        assert summary["sources"][0]["source_text_missing"] == 3
        assert "supervisions.0.text" in caplog.text

    def test_no_key_without_a_source_text_field(self, tmp_path, fake_reader):
        fake_reader.per_source = {0: [_record("a")]}
        summary = freeze({"name": "t", "input_cfg": [{"type": "fake"}]}, tmp_path)
        assert "source_text_missing" not in summary["sources"][0]
```

`_record` and `fake_reader` are the existing helpers in that file; `_record` builds a record with no `source_text`.

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_freeze.py -q -k SourceTextCoverage`.
Expected: FAIL — `KeyError: 'source_text_missing'` / no warning.

- [ ] **Step 3: Implement in `read_sources`**

After `result = reader.freeze(...)`, build `stats = dict(result.stats)`; when `source_cfg.get("source_text_field")` is truthy, set `stats["source_text_missing"] = sum(1 for r in result.records if not r.source_text)` and, if `result.records` and the count equals `len(result.records)`, log a warning naming the source index, label and the field. Append `{"index": …, "source": label, "type": source_type, **stats}`. Do not raise: `custom.pnc_text` is legitimately absent for Irish in the test spec.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_freeze.py -q`.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add melteval/freeze.py tests/test_freeze.py
git commit -m "freeze: report how many records got no source_text"
```

---

### Task 3: Backfill driver — discovery, coverage gate, snapshot, dry run

**Files:**
- Create: `scripts/rescore_fleurs_metricx.py`
- Test: `tests/test_rescore_fleurs_metricx.py`

**Interfaces:**
- Consumes: `scripts/evaluate.py`'s `load_site_env`, `load_config`, `expand`, `resolve_path` (load the module the way `tests/test_evaluate.py` does); `melteval.manifest.resolve_frozen_set`, `read_manifest`.
- Produces (imported by Task 4):
  - `FLEURS_BENCHMARKS: tuple[str, str]`
  - `DEFAULT_CAMPAIGNS: tuple[str, ...]` = the three config paths
  - `@dataclass(frozen=True) class LogRef: campaign: str; model: str; benchmark: str; slice: str; path: Path` with `key` = `f"{campaign}:{model}:{benchmark}:{slice}"`
  - `discover(output_dir: Path, campaign: str, benchmarks: set[str]) -> list[LogRef]`
  - `load_manifest_index(path: Path) -> dict[tuple[str, str], str]`
  - `coverage_rows(refs: list[LogRef], manifests: dict[str, Path]) -> list[dict]` (keys: `campaign`, `model`, `benchmark`, `slice`, `samples`, `with_source`, `coverage`)
  - `class CoverageError(RuntimeError)`
  - `snapshot(refs) -> tuple[list[dict], list[dict]]` (per-slice row: `campaign`, `model`, `benchmark`, `slice`, `metricx`, `metricx_coverage`, `corpus_bleu`, `corpus_spbleu`, `corpus_chrf`; per-sample old triples)
  - `already_rescored(path: Path) -> bool`
  - `command_for(site: str, logs: list[Path], manifest: Path, venv: str, tag: str, slurm_out: Path) -> tuple[list[str], dict[str, str]]` (argv and the environment to submit with)
  - `main(argv: list[str] | None = None) -> None`

- [ ] **Step 1: Write the failing tests**

`tests/test_rescore_fleurs_metricx.py`. It imports the driver the way `tests/test_evaluate.py` does, then builds a campaign tree and tiny logs:

```python
ROOT = Path(__file__).resolve().parents[1]

def _load(name, relative):
    ...  # importlib.util, as in tests/test_evaluate.py

driver = _load("rescore_fleurs_metricx", "scripts/rescore_fleurs_metricx.py")


def _log(samples, metrics, scorer_names=("st_scorer", "metricx_scorer"), rescored=False):
    metricx_metrics = {
        "metricx": {"name": "metricx", "value": metrics["metricx"]},
        "metricx_coverage": {"name": "metricx_coverage", "value": 1.0},
    }
    if rescored:
        metricx_metrics["metricx_source_coverage"] = {"name": "metricx_source_coverage", "value": 1.0}
    return {
        "eval": {"scorers": [{"name": n, "options": {}} for n in scorer_names]},
        "results": {"scores": [
            {"name": "st_scorer", "metrics": {"corpus_bleu": {"name": "corpus_bleu", "value": metrics["bleu"]}}},
            {"name": "metricx_scorer", "metrics": metricx_metrics},
        ]},
        "samples": samples,
    }


@pytest.fixture
def campaign(tmp_path):
    """A campaign config + suite + one slice log + the frozen set it refers to."""
    set_dir = tmp_path / "set"
    set_dir.mkdir()
    (set_dir / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"sample_key": "00-000000", "task": "st", "target": "a", "audio": {"kind": "shar", "params": {}},
         "lang": "en", "src_lang": "fr", "tgt_lang": "en", "dataset_id": "fleurs-fr_en-dev",
         "source": "fleurs/fr_fr/validation", "source_text": "Bonjour.", "cut_id": "1"},
        {"sample_key": "00-000001", "task": "st", "target": "b", "audio": {"kind": "shar", "params": {}},
         "lang": "en", "src_lang": "fr", "tgt_lang": "en", "dataset_id": "fleurs-fr_en-dev",
         "source": "fleurs/fr_fr/validation", "source_text": "Salut.", "cut_id": "2"},
    ]) + "\n")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "benchmarks:\n"
        "  - name: fleurs24-st-xen-dev\n"
        f"    frozen_set: {set_dir}\n"
        "    slices:\n"
        "      - {task: st, dataset_id: fleurs-fr_en-dev}\n"
    )
    out = tmp_path / "out"
    log_dir = out / "logs" / "model-a" / "fleurs24-st-xen-dev" / "st-fleurs-fr_en-dev"
    log_dir.mkdir(parents=True)
    log = _log(
        samples=[
            {"id": "00-000000", "metadata": {"dataset_id": "fleurs-fr_en-dev"},
             "scores": {"metricx_scorer": {"value": 2.0, "metadata": {"metricx_triple": {"src": "", "mt": "a", "ref": "a"}}}}},
            {"id": "00-000001", "metadata": {"dataset_id": "fleurs-fr_en-dev"},
             "scores": {"metricx_scorer": {"value": 4.0, "metadata": {"metricx_triple": {"src": "", "mt": "b", "ref": "b"}}}}},
        ],
        metrics={"bleu": 10.0, "metricx": 3.0},
    )
    (log_dir / "2026-01-01T00-00-00_task.json").write_text(json.dumps(log))
    (log_dir / "2026-01-01T00-00-00_task.json.superseded").write_text("{}")
    config = tmp_path / "config.yaml"
    config.write_text(
        "name: speechval-test\n"
        f"output_dir: {out}\n"
        f"benchmarks: {suite}\n"
    )
    return {"config": config, "out": out, "log": log_dir / "2026-01-01T00-00-00_task.json", "set": set_dir}
```

Cases to cover:

- `test_discover_finds_only_current_logs`: one slice dir with `x.json` and `x.json.superseded`, another benchmark outside `FLEURS_BENCHMARKS`; asserts exactly one `LogRef`.
- `test_coverage_uses_dataset_id_as_well_as_sample_id`: manifest has `("fleurs-es_en-dev", "00-000000")` but the log sample is `fleurs-fr_en-dev`/`00-000000`; coverage is 0.0.
- `test_coverage_is_one_when_every_sample_resolves`: 2/2 → 1.0.
- `test_coverage_rows_raise_below_one`: `with pytest.raises(driver.CoverageError)`.
- `test_snapshot_records_old_metricx_and_triples`: keys and values.
- `test_dry_run_prints_one_command_per_group`: capsys contains `submit_score.sh`, the manifest path, and `--action overwrite`; contains no `--scorer`.
- `test_already_rescored_detects_the_new_metric`: `metricx_source_coverage` present → True.

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_rescore_fleurs_metricx.py -q`.
Expected: FAIL — module has no `discover` / import error.

- [ ] **Step 3: Implement the driver's read-only half**

Module docstring: what it backfills, why the manifest is needed, that it submits through `infra/runners/submit_score.sh`, and that it is not part of the eval path. CLI (argparse): `--site artemis` (default), `--campaign` (repeatable, default `DEFAULT_CAMPAIGNS`), `--dry-run`, `--max-logs N`, `--wait`, `--verify-only`, `--force`.

`discover`: walk `output_dir/logs/<model>/<benchmark>/<slice>/` for `*.json` that do not end in `.superseded`; skip benchmarks not in the requested set; sort by `(campaign, benchmark, model, slice)` and stamp `campaign` on every ref. Raise `SystemExit` if a slice dir has more than one current log.

`load_manifest_index`: `resolve_frozen_set(path)` then `read_manifest`, returning `{(r.dataset_id, r.sample_key): r.source_text or ""}`.

`coverage_rows`: for each ref, `json.loads(path.read_text())`, then for every sample count `manifests_index[(sample["metadata"].get("dataset_id"), sample["id"])]` non-empty. Missing metadata counts as missing. Raise `CoverageError` listing the offending slices after building all rows.

`snapshot`: per ref, read `results.scores` for `metricx_scorer` → `metricx`/`metricx_coverage`, and for `st_scorer` → `corpus_bleu`/`corpus_spbleu`/`corpus_chrf` (absent ones left empty, e.g. a no-spBLEU run); and every sample's `scores.metricx_scorer.metadata.metricx_triple` with its `sample_id`. Return two lists of dicts (CSV rows and JSONL rows).

`already_rescored`: any `results.scores[*].metrics` key `metricx_source_coverage`.

`command_for`: returns `(["infra/runners/submit_score.sh", site, *[str(p) for p in logs], "--", "--model", "mockllm/model", "--action", "overwrite"], env)` where `env` carries `VENV_PATH=<venv>`, `MELTEVAL_SOURCE_MANIFEST=<manifest>`, `MELT_JOB_TAG=<tag>` and `MELT_SLURM_OUT=<slurm_out>`. No `--scorer`, ever.

`main` for this task: load site env, load each campaign config, resolve `{benchmark: frozen_set path}` for `FLEURS_BENCHMARKS` (`resolve_path(bench["frozen_set"], …)`, abort if a benchmark uses `spec` instead), discover, drop `already_rescored` unless `--force`, apply `--max-logs`, run coverage, write `<output_dir>/results/fleurs-source-coverage.csv`, snapshot to `<output_dir>/results/metricx-before.{csv,jsonl}`, and for `--dry-run` print the command per `(campaign, model, benchmark)` group then return.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_rescore_fleurs_metricx.py -q`.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/rescore_fleurs_metricx.py tests/test_rescore_fleurs_metricx.py
git commit -m "scripts: FLEURS MetricX backfill preflight and snapshot"
```

---

### Task 4: Backfill driver — submission and post-run verification

**Files:**
- Modify: `scripts/rescore_fleurs_metricx.py`
- Test: `tests/test_rescore_fleurs_metricx.py`

**Interfaces:**
- Consumes: Task 3's `LogRef`, `load_manifest_index`, `snapshot`, `command_for`; `scripts/evaluate.py`'s `run_command` and `wait_for_jobs`.
- Produces:
  - `submit(groups: dict[tuple[str, str, str], list[LogRef]], site: str, manifests: dict[str, Path], venv: str) -> list[str]` (job ids)
  - `verify(refs: list[LogRef], before: list[dict]) -> list[dict]` (keys: `campaign`, `model`, `benchmark`, `slice`, `metricx_before`, `metricx_after`, `delta`, `bleu_before`, `bleu_after`, `source_coverage`)
  - `VerificationError`

- [ ] **Step 1: Write the failing tests**

```python
def _sample(sample_id, value=2.0, completion="out", src="Bonjour."):
    score = {"value": value, "metadata": {"metricx_triple": {"src": src, "mt": completion, "ref": completion}}}
    return {
        "id": sample_id,
        "metadata": {"dataset_id": "fleurs-fr_en-dev"},
        "output": {"completion": completion} if completion else None,
        "scores": {"metricx_scorer": score} if value is not None else {},
    }


def _before(campaign):
    return [{
        "campaign": "speechval-test", "model": "model-a", "benchmark": "fleurs24-st-xen-dev",
        "slice": "st-fleurs-fr_en-dev", "metricx": 3.0, "metricx_coverage": 1.0, "corpus_bleu": 10.0,
    }]


def test_verify_accepts_a_correct_log(campaign):
    ref = driver.LogRef("speechval-test", "model-a", "fleurs24-st-xen-dev", "st-fleurs-fr_en-dev", campaign["log"])
    log = _log(samples=[_sample("00-000000", 2.0), _sample("00-000001", 4.0)], metrics={"bleu": 10.0, "metricx": 3.0}, rescored=True)
    campaign["log"].write_text(json.dumps(log))
    rows = driver.verify([ref], _before(campaign))
    assert rows[0]["metricx_before"] == 3.0
    assert rows[0]["metricx_after"] == pytest.approx(3.0)
    assert rows[0]["source_coverage"] == 1.0
```

Then, each as its own test: `test_verify_rejects_a_log_without_per_sample_values` (one sample's `scores` empty → `pytest.raises(driver.VerificationError, match="00-000001")`), `test_verify_rejects_a_lost_bleu` (metrics without `corpus_bleu`), `test_verify_rejects_source_coverage_below_one` (`metricx_source_coverage` 0.5), `test_verify_rejects_a_mean_that_is_not_the_sample_mean` (`metricx` 9.0), and `test_submit_sets_the_manifest_env_and_no_scorer` (monkeypatch `driver.run_command`, capture `(command, env)`, assert `env["MELTEVAL_SOURCE_MANIFEST"] == str(campaign["set"])`, `env["VENV_PATH"] == "/venvs/metrics/activate"`, and `"--scorer" not in command`).

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_rescore_fleurs_metricx.py -q -k "verify or submit"`.
Expected: FAIL — `verify`/`submit` not defined.

- [ ] **Step 3: Implement**

`submit`: group by `(campaign, model, benchmark)`; per group call `run_command(*command_for(site, logs, manifest, venv, tag, slurm_out))` and capture job ids with the existing `Submitted batch job (\d+)` regex. Import `run_command` and `wait_for_jobs` into the driver's namespace (`from evaluate import run_command, wait_for_jobs`) so tests can monkeypatch `driver.run_command`.

`verify`: for each ref and its `before` row — the scorer set in `results.scores` is exactly `{st_scorer, metricx_scorer}`; `metricx_source_coverage == 1.0`; `corpus_bleu`, `corpus_spbleu` and `corpus_chrf` (where the snapshot has them) are bit-identical; every sample with a non-empty completion has a finite `scores.metricx_scorer.value`; `metricx` equals the mean of those values within `1e-9`. Collect all failures and raise once with every slice listed.

`main` continues: after the dry-run branch, if `--verify-only`, load `results/metricx-before.csv` from disk (error if absent), run `verify` over all refs and write `<output_dir>/results/metricx-rescore.csv`; otherwise `submit`, print the job ids, and if `--wait` call `wait_for_jobs` then verify and write the CSV.

- [ ] **Step 4: Run the tests and lint**

Run: `pytest tests/test_rescore_fleurs_metricx.py -q && ruff check scripts/rescore_fleurs_metricx.py melteval/rescore.py melteval/freeze.py`.
Expected: PASS, no lint errors.

- [ ] **Step 5: Commit**

```bash
git add scripts/rescore_fleurs_metricx.py tests/test_rescore_fleurs_metricx.py
git commit -m "scripts: submit and verify the FLEURS MetricX backfill"
```

---

### Task 5: Documentation

**Files:**
- Modify: `docs/speechval.md` (FLEURS frozen-set rows, a new "MetricX sources" subsection)
- Modify: `AGENTS.md` ("Things that are easy to get wrong here")

**Interfaces:** none.

- [ ] **Step 1: Document the backfill and the freeze rule**

In `docs/speechval.md`: state that FLEURS ST is scored with full triples, that the artemis sets were rebuilt on 2026-11-23 (same `spec_hash`, `source_text` 100%), the per-sample MetricX values and the `metricx_source_coverage` metric, and the `MELTEVAL_SOURCE_MANIFEST` backfill knob. Add the `--rescore` hazard: `scripts/evaluate.py score --rescore` passes one scorer to `--action overwrite`, which deletes a slice's other post-hoc scorers, so it must not be used on an ST benchmark until it is reworked. Change every freeze command in the doc to `PYTHONPATH=. melteval freeze …` and say why.

- [ ] **Step 2: Add the AGENTS bullet**

One bullet under "Things that are easy to get wrong here": the venv's editable install points at another checkout, so a bare `melteval freeze` can silently produce records the current reader would not; freeze with `PYTHONPATH=.` and check `source_text_missing` in `frozen_set.json`.

- [ ] **Step 3: Commit**

```bash
git add docs/speechval.md AGENTS.md
git commit -m "docs: FLEURS MetricX sources and the PYTHONPATH freeze rule"
```

---

### Task 6: Rebuild the artemis FLEURS frozen sets and settle Jupiter

**Files:** none in the repo. Writes under `/mnt/scratch-artemis/giuseppe/frozen/speechval-dev/`; on Jupiter, under `/e/data1/smurf4eu-data/speech/frozen-dev/` and `/e/project1/e-ext-2025e01-100/attanasio1/frozen-dev/`.

**Interfaces:** consumes Task 2's `source_text_missing`.

- [ ] **Step 1: Freeze to a side directory**

```bash
cd /home/giuseppe/melt-proj/melt-eval
SET=/mnt/scratch-artemis/giuseppe/frozen/speechval-dev
PYTHONPATH=. /mnt/scratch-artemis/giuseppe/venvs/melteval/bin/melteval freeze \
    configs/shar/fleurs24-st-enx-dev.yaml -o $SET/fleurs24-st-enx-dev.new
PYTHONPATH=. /mnt/scratch-artemis/giuseppe/venvs/melteval/bin/melteval freeze \
    configs/shar/fleurs24-st-xen-dev-full.yaml -o $SET/fleurs24-st-xen-dev-full.new
```

Expected: 8819 and 8261 records; `sources[*].source_text_missing` all 0.

- [ ] **Step 2: Verify record identity and source coverage on artemis**

Compare `.new/manifest.jsonl` against the current `manifest.jsonl` record by record on `sample_key`, `audio`, `target`, `cut_id`, and assert `source_text` is non-empty for every record and the count is unchanged. Any difference stops the work: the logs would be orphaned.

- [ ] **Step 3: Compare with Jupiter, by digest**

On artemis and on Jupiter (both `/e/data1/smurf4eu-data/speech/frozen-dev/` and `/e/project1/.../frozen-dev/`), compute a sha256 over the canonical JSON of every record's `(sample_key, audio, target, cut_id, source_text)`:

```bash
python3 - "$MANIFEST" <<'EOF'
import hashlib, json, sys
p = sys.argv[1]
h = hashlib.sha256()
n = src = 0
for line in open(p):
    r = json.loads(line); n += 1
    src += bool(r.get("source_text"))
    h.update(json.dumps([r["sample_key"], r["audio"], r["target"], r["cut_id"], r["source_text"]],
                        sort_keys=True, ensure_ascii=False).encode() + b"\n")
print(n, src, h.hexdigest()[:16])
EOF
```

Expected: the same `(count, with_source, digest)` triple as the artemis `.new` manifest, on both Jupiter trees. Jupiter was frozen from the same reader with the source field already in place, so the digests should already match; if they do, **do not rewrite Jupiter** — record the triples as the evidence that the rows are comparable.

- [ ] **Step 4: Only if a Jupiter digest differs, re-freeze it**

On Jupiter, in its repo copy (`/e/project1/e-ext-2025e01-100/attanasio1/eval-qwen3`), source the Jupiter site env and freeze into a side directory with `PYTHONPATH=.`, using `/e/project1/e-ext-2025e01-100/calvo-bartolome1/venvs/freeze-eval` (or `smurf-eval`). Verify Step 2's identity check against the existing manifest, then swap it in keeping `manifest.jsonl.prev`. Repeat for both trees that differ. Re-run the Step 3 digest and record it.

- [ ] **Step 5: Swap artemis in, keeping the old manifest**

For each set: `mv manifest.jsonl manifest.jsonl.prev`, copy `manifest.jsonl` and `frozen_set.json` from `.new`, then remove `.new`. Re-run Step 2's check against `.prev` to confirm only `source_text` moved.

- [ ] **Step 6: Record the evidence**

Record, for each of the three trees: count, with_source, digest; and whether Jupiter was rewritten or already identical. Both sets' digests must match before Task 7. No commit.

---

### Task 7: Run the backfill on artemis

**Files:** none in the repo. Writes logs and `results/*.csv` under the three campaign folders.

**Interfaces:** consumes Tasks 1–6.

- [ ] **Step 1: Preflight**

```bash
cd /home/giuseppe/melt-proj/melt-eval
python scripts/rescore_fleurs_metricx.py --site artemis --dry-run
```

Expected: `results/fleurs-source-coverage.csv` with 381 rows, every `coverage` 1.0; one printed command per `(campaign, model, benchmark)` group; exit 0. Any slice below 1.0 stops here.

- [ ] **Step 2: Smoke one log through SLURM**

```bash
python scripts/rescore_fleurs_metricx.py --site artemis --max-logs 1 --wait
```

Expected: one job, `metricx_source_coverage` 1.0 in that log, per-sample values present, BLEU identical to the snapshot, `metricx` = mean of the values. Record the wall time.

- [ ] **Step 3: Full run**

```bash
python scripts/rescore_fleurs_metricx.py --site artemis --wait
```

Expected: 18 jobs (the smoke's log is skipped, but its group still has its other logs), all COMPLETED, then `results/metricx-rescore.csv` written for the three campaigns with no verification failures.

- [ ] **Step 4: Rebuild the campaign reports**

`python scripts/evaluate.py report configs/eval/speechval-fondue.yaml`, and the same for `speechval-canary` and `speechval-qwen3-omni`. Expected: `results/summary.csv` etc. refreshed.

- [ ] **Step 5: Check the worklog**

Wait one cache interval, then confirm three rows in the matrix show the new MetricX with unchanged BLEU, and no FLEURS slice shows a missing metric. Report the before/after numbers for FLEURS X→en and en→X per model.
