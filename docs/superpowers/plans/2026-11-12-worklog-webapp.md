# SpeechEval Worklog Implementation Plan

> Moved during execution to `/home/giuseppe/melt-proj/speecheval-worklog`, a
> separate repo; the authoritative plan now lives there at
> `docs/plans/2026-11-12-worklog-webapp.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A read-only Flask dashboard that scans artemis inspect logs and renders the speechval results as a model × benchmark matrix, per-language heatmaps and a live run feed.

**Architecture:** A background scanner walks configured roots, extracts one row per inspect log, and caches it in SQLite keyed by `(path, mtime_ns, size)`. Flask serves one `/api/snapshot` payload of winning slice rows plus filter/KPI data; the page (vanilla JS, vendored ECharts + Tabulator) aggregates and renders client-side and polls for changes. Nothing imports `melteval`, nothing re-scores, nothing writes outside the SQLite cache.

**Tech Stack:** Python 3.10+, Flask 3.1, PyYAML, SQLite (stdlib), vanilla JS ES2020, ECharts 5.5, Tabulator 6.3, pytest.

**Spec:** `docs/superpowers/specs/2026-11-12-speechval-canary-and-worklog-design.md` (Part B)

## Global Constraints

- No import from `typing` in `projects/worklog/`; use `X | None`, `list[str]`, `dict[str, int]`.
- Line length 119, ruff-clean.
- No `melteval` import in `projects/worklog/`; the dashboard reads log files, not library internals.
- The only write is the SQLite cache, default `/mnt/scratch-artemis/giuseppe/worklog/worklog.sqlite` (`$WORKLOG_CACHE` overrides); never under `$HOME` unless explicitly passed.
- Server binds `127.0.0.1` by default; `--host 0.0.0.0` is always an explicit flag.
- No CDN at runtime: ECharts and Tabulator are vendored under `worklog/static/vendor/`.
- Read-only API: no endpoint accepts a write except `POST /api/refresh`, which only rescans.
- Smoke/duplicate semantics mirror `scripts/evaluate.py`: newest successful log per `(campaign, model, benchmark, slice)` wins; smoke folders are ignored by default.

## Review Focus

- **A log still being written** (SLURM job finishing): `json.load` fails or the file is truncated → it must be skipped and retried next scan, never cached as an error run.
- **Two successful logs for the same slice** (a rerun): newest wins in the matrix; the older stays in the feed. Reversing them silently publishes a stale number.
- **A declared post-scorer that has not run** (`melteval_post_scorers` present, no metric): shown as pending (`…`), never as missing or zero.
- **A slice with a region code** (`zh-CN`) or no `task_args` (older logs): direction parsed from the sample metadata or slice id; an unparseable one is labelled `unclassified` and still rendered, not dropped or crashed on.
- **Unrelated JSON in a scanned root** (configs, other tools' logs): ignored without counting as a parse error in the header warning.

Tests for each live in the task that owns the code (Tasks 2, 3, 4).

---

### Task 1: Scaffold, config loader and cache location

**Files:**
- Create: `projects/worklog/requirements.txt`
- Create: `projects/worklog/worklog/__init__.py` (empty)
- Create: `projects/worklog/worklog/config.py`
- Create: `projects/worklog/worklog.yaml`
- Create: `projects/worklog/tests/conftest.py`
- Test: `projects/worklog/tests/test_config.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Family(name: str, task: str, benchmarks: tuple[str, ...], direction: str | None = None)`
  - `@dataclass(frozen=True) Metric(label: str, lower: bool)`
  - `@dataclass(frozen=True) Config(roots, include, ignore, cache, port, host, interval_seconds, aggregate, families, metrics, headline)` with `headline_metric(task: str) -> Metric | None` and `metric_names(task: str) -> tuple[str, ...]`
  - `load_config(path: Path | None = None, overrides: dict | None = None) -> Config`

- [ ] **Step 1: Write the failing tests**

`tests/conftest.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

`tests/test_config.py` — assert defaults, YAML overrides, env expansion, and the
task→metric lookups:

```python
def test_defaults_load_shipped_yaml():
    cfg = load_config()
    assert cfg.port == 8765 and cfg.host == "127.0.0.1"
    assert cfg.aggregate == "macro"
    assert cfg.roots[0].name == "eval-runs"


def test_cache_expands_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKLOG_CACHE", str(tmp_path / "w.sqlite"))
    p = tmp_path / "c.yaml"
    p.write_text("roots: [/x]\\ncache: ${WORKLOG_CACHE}\\n")
    assert load_config(p).cache == tmp_path / "w.sqlite"


def test_metric_lookup(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "roots: [/x]\\n"
        "metrics:\\n  corpus_wer_raw: {label: WER, lower: true}\\n"
        "headline: {asr: corpus_wer_raw}\\n"
    )
    cfg = load_config(p)
    assert cfg.headline_metric("asr").label == "WER"
    assert cfg.headline_metric("st") is None
    assert cfg.metric_names("asr") == ("corpus_wer_raw",)
```

- [ ] **Step 2: Run to see them fail**

Run: `pytest projects/worklog/tests/test_config.py -v`
Expected: `ModuleNotFoundError: No module named 'worklog'`.

- [ ] **Step 3: Implement `config.py` and the shipped YAML**

`config.py`: dataclasses above; `load_config` reads YAML (default
`<package parent>/worklog.yaml`), applies `os.path.expandvars`/`expanduser` to
`roots`, `cache`, then `overrides`, validates `aggregate in {"macro","weighted"}`,
and builds `metrics: dict[str, Metric]`, `headline: dict[str, str]`,
`families: tuple[Family, ...]`. `Family.direction` must be `"xen"`, `"enx"` or
`None`; anything else raises `ValueError`.

`worklog.yaml` is the block from the design doc (roots, include, ignore, cache,
port, host, interval, aggregate, `families`, `metrics`, `headline`), with
`headline: {asr: corpus_wer_raw, st: corpus_bleu}`.

`requirements.txt`:

```
flask>=3.1,<4
pyyaml>=6
pytest>=8
```

- [ ] **Step 4: Run**

Run: `pytest projects/worklog/tests/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add projects/worklog
git commit -m "worklog: config loader and shipped defaults"
```

---

### Task 2: Log extraction

**Files:**
- Create: `projects/worklog/worklog/scanner.py`
- Create: `projects/worklog/tests/helpers.py`
- Test: `projects/worklog/tests/test_scanner.py`

**Interfaces:**
- Consumes: `Config` (Task 1)
- Produces:
  - `@dataclass(frozen=True) ScannedRun(path, mtime_ns, size, campaign, model, benchmark, slice, task, src, tgt, status, total_samples, completed_samples, created, finished, log_id, metrics, pending, row_id: int | None = None)`
  - `scan_run(path: Path) -> ScannedRun | None`
  - `iter_log_paths(config: Config) -> Iterator[Path]`
  - `parse_pair(dataset_id: str | None, slice_id: str | None, task: str, sample_meta: dict | None) -> tuple[str | None, str | None]`
  - `direction_of(src: str | None, tgt: str | None) -> str | None`
  - `model_family(model: str) -> str` (`baseline` / `melt` / `smurf` / `other`)

- [ ] **Step 1: Write the fixture helper and failing tests**

`tests/helpers.py`:

```python
def make_log(tmp_path, *, campaign="speechval-canary", model="canary-1b-v2",
             eval_model="melt/hf/canary", benchmark="fleurs24-st-xen-dev",
             slice_id="st-fleurs-de_en", task="st", status="success",
             metrics=None, post_scorers=(), dataset_id="fleurs-de_en",
             lang=None, src="de", tgt="en", samples=10, completed=10,
             created="2026-11-12T10:00:00+00:00",
             finished="2026-11-12T10:05:00+00:00") -> Path:
    """Write a minimal inspect-like log and return its path."""
```

It writes one JSON object with `version`, `status`, `eval.{task, model, created,
dataset:{samples}, task_args:{dataset_id, lang}, metadata:{melteval_*}}`,
`stats:{completed_at}`, `results:{completed_samples, scores:[{name:"st_scorer",
metrics: metrics or {}}]}`, and `samples: [{metadata: {task, lang: tgt,
src_lang: src, tgt_lang: tgt}}]` (omit `samples` when `status != "success"`).
Write it under `<tmp>/logs/<model>/<benchmark>/<slice_id>/<created>.json`.

Tests:

```python
def test_success_st_fields(tmp_path):
    p = make_log(tmp_path)
    run = scan_run(p)
    assert (run.campaign, run.model, run.benchmark, run.slice, run.task) == (
        "speechval-canary", "canary-1b-v2", "fleurs24-st-xen-dev", "st-fleurs-de_en", "st")
    assert (run.src, run.tgt) == ("de", "en")
    assert direction_of(run.src, run.tgt) == "xen"
    assert run.metrics == {"corpus_bleu": 24.5}
    assert run.status == "success" and run.finished == "2026-11-12T10:05:00+00:00"


def test_pending_post_scorer(tmp_path):
    p = make_log(tmp_path, post_scorers=("melteval/rescore.py@metricx_scorer",),
                 metrics={"corpus_bleu": 24.5})
    assert scan_run(p).pending == ("metricx_scorer",)


def test_error_log_has_no_samples_and_is_extracted(tmp_path):
    p = make_log(tmp_path, status="error", metrics={}, samples=0)
    run = scan_run(p)
    assert run.status == "error" and run.finished is None


def test_unrelated_json_is_ignored(tmp_path):
    p = tmp_path / "something.json"
    p.write_text('{"hello": "world"}')
    assert scan_run(p) is None


def test_half_written_file_is_ignored(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text('{"eval": {"task":')
    assert scan_run(p) is None


def test_region_codes_and_missing_task_args(tmp_path):
    assert parse_pair("covost2-en_zh-CN-dev", None, "st", None) == ("en", "zh-CN")
    assert parse_pair(None, "st-fleurs-de_en", "st", None) == ("de", "en")
    assert parse_pair(None, "asr-bg", "asr", {"lang": "bg"}) == (None, None)


def test_model_family():
    assert model_family("melt/hf/canary") == "baseline"
    assert model_family("melt/checkpoints/x") == "melt"
    assert model_family("smurf//e/x.ckpt") == "smurf"
```

- [ ] **Step 2: Run**

Run: `pytest projects/worklog/tests/test_scanner.py -v`
Expected: collection error (`No module named 'worklog.scanner'`).

- [ ] **Step 3: Implement `scanner.py`**

- `scan_run`: read + `json.load` inside `try/except (OSError, json.JSONDecodeError)`;
  require `eval.metadata.melteval_benchmark` (else `None`); read fields per the
  spec's table; `status = j.get("status") or "unknown"`; `finished =
  j["stats"].get("completed_at")`; flatten metrics with
  `f"{scorer}.{name}"` only when two scorers share a metric name (otherwise the
  bare name); `pending` = comma-split `melteval_post_scorers`, each reduced to
  its part after `@`, minus metrics already present.
- `parse_pair`: prefer the first sample's `metadata.{src_lang,tgt_lang}` (or
  `lang` for ASR), then regex on `dataset_id`, then on `slice_id`:
  `r"-([a-z]{2}(?:-[A-Z]{2})?)_([a-z]{2}(?:-[A-Z]{2})?)(?:-(?:dev|test))?$"`.
  For `task in {"asr", "chunked_asr"}` return `(None, None)`.
- `direction_of` → `"enx"` if `src == "en" != tgt`, `"xen"` if `tgt == "en" != src`, else `None`.
- `iter_log_paths`: for each root, each direct child directory matching any
  `include` glob; walk it for `*.json`; skip a path when any of its parts
  matches an `ignore` glob (use `fnmatch.fnmatch(part, pattern)` after
  stripping `**/` and `/**` from the pattern); dedupe and yield sorted.

- [ ] **Step 4: Run**

Run: `pytest projects/worklog/tests/test_scanner.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add projects/worklog/worklog/scanner.py projects/worklog/tests
git commit -m "worklog: extract runs from inspect logs"
```

---

### Task 3: SQLite cache and duplicate policy

**Files:**
- Create: `projects/worklog/worklog/store.py`
- Test: `projects/worklog/tests/test_store.py`

**Interfaces:**
- Consumes: `ScannedRun` (Task 2)
- Produces:
  - `class Store:`
    - `__init__(self, path: Path)`
    - `sync(self, runs: Iterable[ScannedRun]) -> str` — upsert by path, delete rows whose path is absent, return a content version
    - `version(self) -> str`
    - `winning(self) -> list[ScannedRun]` — newest `success` per `(campaign, model, benchmark, slice)`
    - `history(self, *, campaign=None, model=None, benchmark=None, status=None, offset=0, limit=50) -> list[ScannedRun]`
    - `get(self, run_id: int) -> ScannedRun | None` — `run_id` is the SQLite row id, exposed as `row_id` on returned `ScannedRun`s
    - `all(self) -> list[ScannedRun]`

- [ ] **Step 1: Write the failing tests**

```python
def test_sync_and_version_stable(tmp_path):
    s = Store(tmp_path / "w.sqlite")
    v1 = s.sync([run("a.json", created="2026-11-12T10:00:00", status="success")])
    assert s.sync([run("a.json", created="2026-11-12T10:00:00", status="success")]) == v1
    v2 = s.sync([run("a.json", created="2026-11-12T10:00:00", status="success", mtime_ns=2)])
    assert v2 != v1


def test_newest_success_wins_and_error_never_shadows(tmp_path):
    s = Store(tmp_path / "w.sqlite")
    s.sync([
        run("old.json", created="2026-11-12T09:00:00", status="success", metrics={"corpus_bleu": 20.0}),
        run("new.json", created="2026-11-12T10:00:00", status="success", metrics={"corpus_bleu": 24.0}),
        run("err.json", created="2026-11-12T11:00:00", status="error", metrics={}),
    ])
    win = s.winning()
    assert len(win) == 1 and win[0].metrics["corpus_bleu"] == 24.0
    assert [r.status for r in s.history(limit=10)] == ["error", "success", "success"]


def test_deleted_log_is_removed(tmp_path):
    s = Store(tmp_path / "w.sqlite")
    s.sync([run("a.json"), run("b.json", slice="asr-en", task="asr")])
    s.sync([run("a.json")])
    assert [r.path for r in s.all()] == ["a.json"]
```

The `run(...)` helper wraps `ScannedRun` with defaults.

- [ ] **Step 2: Run**

Run: `pytest projects/worklog/tests/test_store.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `store.py`**

Tables exactly as in the design (`runs`, `metrics`, `meta`). `sync` runs in one
transaction: upsert changed `(path, mtime_ns, size)` rows, rewrite their metrics,
delete rows not in the incoming path set, then set `meta.version` to
`sha1("|".join(sorted(f"{path}:{mtime}:{size}")))`, truncated to 12 chars.
`winning`/`history` select then group/sort in Python (a few hundred rows) via a
shared `_row_to_run`. `metrics` stores `is_pending`.

- [ ] **Step 4: Run**

Run: `pytest projects/worklog/tests/test_store.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add projects/worklog/worklog/store.py projects/worklog/tests/test_store.py
git commit -m "worklog: sqlite cache with evaluate.py duplicate policy"
```

---

### Task 4: Aggregation and snapshot

**Files:**
- Create: `projects/worklog/worklog/views.py`
- Test: `projects/worklog/tests/test_views.py`

**Interfaces:**
- Consumes: `Config`, `ScannedRun`
- Produces:
  - `match_family(families, benchmark, task, direction) -> Family | None`
  - `family_of(config, run) -> Family | None`
  - `aggregate(values: Sequence[tuple[float, int]], kind: str) -> float | None` — each item is `(value, samples)`
  - `build_snapshot(config: Config, runs: Sequence[ScannedRun], feed: Sequence[ScannedRun], version: str, now: str) -> dict`
  - Snapshot schema (the frontend contract):

```json
{"version": "a1b2c3", "generated_at": "…",
 "kpis": {"models": 1, "families": 8, "slices": 92, "pending": 0, "failed": 1, "warnings": 0},
 "filters": {"campaigns": ["…"], "models": ["…"], "families": ["…"], "languages": ["de", "…"]},
 "model_families": {"canary-1b-v2": "baseline"},
 "slices": [{"id": "canary-1b-v2|fleurs24-st-xen-dev|st-fleurs-de_en", "campaign": "…", "model": "…",
             "benchmark": "fleurs24-st-xen-dev", "slice": "st-fleurs-de_en", "family": "FLEURS X→EN",
             "task": "st", "src": "de", "tgt": "en", "direction": "xen", "status": "success",
             "samples": 862, "completed": 862, "created": "…", "finished": "…",
             "metrics": {"corpus_bleu": 24.5}, "pending": [], "path": "/abs/log.json"}],
 "matrix": {"asr": {"metric": "corpus_wer_raw", "lower": true, "cells": [{"model": "…", "family": "…", "value": null, "n": 0, "total": 0, "status": "empty"}]}, "st": {"…": "…"}},
 "runs": [{"id": 3, "campaign": "…", "model": "…", "slice": "…", "status": "error", "created": "…", "finished": null, "path": "…"}]}
```

- [ ] **Step 1: Write the failing tests**

```python
def test_europarl_split_by_direction(cfg, run):
    fam = family_of(cfg, run(benchmark="europarl-st-dev", task="st", src="fr", tgt="en"))
    assert fam.name == "Europarl X→EN"
    fam = family_of(cfg, run(benchmark="europarl-st-dev", task="st", src="en", tgt="de"))
    assert fam.name == "Europarl EN→X"


def test_unknown_benchmark_is_unclassified(cfg, run):
    assert family_of(cfg, run(benchmark="something-new")) is None


def test_macro_and_weighted():
    assert aggregate([(10.0, 1), (20.0, 3)], "macro") == 15.0
    assert aggregate([(10.0, 1), (20.0, 3)], "weighted") == 17.5
    assert aggregate([], "macro") is None


def test_matrix_cell_states(cfg, run):
    runs = [run(benchmark="fleurs24-st-xen-dev", slice="st-fleurs-de_en", src="de", tgt="en",
                metrics={"corpus_bleu": 20.0}),
            run(benchmark="fleurs24-st-xen-dev", slice="st-fleurs-fr_en", src="fr", tgt="en",
                metrics={"corpus_bleu": 30.0}, pending=("metricx_scorer",))]
    snap = build_snapshot(cfg, runs, [], "v1", "now")
    cell = [c for c in snap["matrix"]["st"]["cells"] if c["family"] == "FLEURS X→EN"][0]
    assert cell["value"] == 25.0 and cell["n"] == 2 and cell["total"] == 22
    assert snap["kpis"]["pending"] == 1


def test_pending_only_cell(cfg, run):
    runs = [run(metrics={}, pending=("metricx_scorer",))]
    snap = build_snapshot(cfg, runs, [], "v1", "now")
    cell = [c for c in snap["matrix"]["st"]["cells"] if c["family"] == "FLEURS X→EN"][0]
    assert cell["status"] == "pending" and cell["value"] is None
```

The `cfg` fixture loads the shipped config with roots pointed at `tmp_path`;
`run` and the helper are shared with Task 3's tests (move it into `helpers.py`).

- [ ] **Step 2: Run**

Run: `pytest projects/worklog/tests/test_views.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `views.py`**

- `match_family`: benchmark in `family.benchmarks` and (`family.direction is
  None or family.direction == direction`).
- `family_of`: `match_family(config.families, run.benchmark, run.task, direction_of(run.src, run.tgt))`.
- `aggregate`: `None` on empty; `macro` is the unweighted mean; `weighted` is
  `sum(value * samples) / sum(samples)`, with non-positive sample counts ignored
  but counted in `n`.
- `build_snapshot`: one slice row per winning run (family may be `None` →
  `family: "unclassified"`), matrix per task over every model × family, cells in
  `ok | partial | pending | empty` per B4b. `kpis.pending` counts slice rows with
  a non-empty `pending`; `failed` counts feed rows with status `error`; `warnings`
  comes from the app's scanner state (pass 0 here).

- [ ] **Step 4: Run**

Run: `pytest projects/worklog/tests/test_views.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add projects/worklog/worklog/views.py projects/worklog/tests
git commit -m "worklog: family matching and matrix aggregation"
```

---

### Task 5: Flask app and JSON API

**Files:**
- Create: `projects/worklog/worklog/app.py`
- Create: `projects/worklog/app.py` (entry)
- Test: `projects/worklog/tests/test_api.py`

**Interfaces:**
- Consumes: `Config`, `Store`, `build_snapshot`
- Produces:
  - `create_app(config: Config, store: Store) -> Flask`
  - Routes: `GET /`, `GET /api/snapshot?version=`, `GET /api/runs`, `GET /api/run/<int:run_id>`, `POST /api/refresh`, `GET /api/health`
  - `python app.py [--config PATH] [--root PATH] [--cache PATH] [--port INT] [--host HOST]` starts the scanner thread and the server

- [ ] **Step 1: Write the failing tests**

```python
@pytest.fixture
def client(cfg, store):
    return create_app(cfg, store).test_client()


def test_snapshot_shape(client, store):
    store.sync([run("a.json", metrics={"corpus_bleu": 24.5})])
    r = client.get("/api/snapshot")
    body = r.get_json()
    assert r.status_code == 200
    assert body["version"] and body["slices"][0]["metrics"]["corpus_bleu"] == 24.5
    assert set(body) == {"version", "generated_at", "kpis", "filters", "model_families",
                         "slices", "matrix", "runs"}


def test_version_match_is_204(client, store):
    store.sync([run("a.json")])
    v = client.get("/api/snapshot").get_json()["version"]
    assert client.get(f"/api/snapshot?version={v}").status_code == 204


def test_refresh_rescans_and_bumps_version(client, store, tmp_path):
    store.sync([run("a.json")])
    v = client.get("/api/snapshot").get_json()["version"]
    client.get("/api/refresh")  # create_app wires the scan callback
    assert client.get(f"/api/snapshot?version={v}").status_code in (204, 200)


def test_run_detail_and_missing(client, store):
    store.sync([run("a.json")])
    rid = store.all()[0].row_id
    assert client.get(f"/api/run/{rid}").status_code == 200
    assert client.get("/api/run/9999").status_code == 404
    assert client.get("/api/run/9999").get_json()["error"]


def test_health(client):
    assert client.get("/api/health").get_json()["ok"] is True
```

- [ ] **Step 2: Run**

Run: `pytest projects/worklog/tests/test_api.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`create_app` builds a snapshot from `store.winning()` + `store.history(limit=50)`
on each snapshot request (cheap: SQLite read), computes `generated_at` in UTC and
compares `version`. `/api/runs` paginates `store.history(**filters)`. `/api/run/<id>`
returns the stored row with metrics. `/api/refresh` calls the configured scan
callback (default no-op in tests; the real app passes the scanner's
`scan_once`), then returns `{"version": store.version()}`. `/` renders
`static/index.html`. `/api/health` returns `{"ok": true, "version": ...}`.

`app.py` (entry) loads config, syncs once before serving, starts a daemon thread
looping `store.sync(iter_log_paths(config))` every `interval_seconds` with a
`threading.Lock` shared with `/api/refresh`, then `app.run(host, port)`.
`store.sync` must be safe under concurrent reads (single connection per call).

- [ ] **Step 4: Run**

Run: `pytest projects/worklog/tests/test_api.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add projects/worklog/app.py projects/worklog/worklog/app.py projects/worklog/tests/test_api.py
git commit -m "worklog: flask app and json api"
```

---

### Task 6: Page shell, vendored libraries and the look

**Files:**
- Create: `projects/worklog/worklog/static/index.html`
- Create: `projects/worklog/worklog/static/style.css`
- Create: `projects/worklog/worklog/static/util.js`
- Create: `projects/worklog/worklog/static/vendor/{echarts.min.js,tabulator.min.js,tabulator.min.css}`
- Create: `projects/worklog/scripts/vendor.sh`
- Test: `projects/worklog/tests/test_static.py`

**Interfaces:**
- Produces: the DOM ids the later tasks bind to: `#kpis`, `#filters`,
  `#matrix`, `#heatmap`, `#feed`, `#refresh`, `#statusline`, `#modal`.
  `util.js` exports `fmt(value, digits)`, `cmp(a, b, lower)`, `speed(ms)`.

- [ ] **Step 1: Write the failing test**

```python
def test_index_serves_vendored_assets(client):
    r = client.get("/")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    for needle in ('id="matrix"', 'id="heatmap"', 'id="feed"', "/static/vendor/echarts.min.js"):
        assert needle in html
    for asset in ("echarts.min.js", "tabulator.min.js", "tabulator.min.css"):
        assert client.get(f"/static/vendor/{asset}").status_code == 200
```

- [ ] **Step 2: Run**

Run: `pytest projects/worklog/tests/test_static.py -v`
Expected: fails (no index, 404 assets).

- [ ] **Step 3: Vendor and build the shell**

`scripts/vendor.sh`:

```bash
#!/bin/bash
set -euo pipefail
D=$(dirname "$0")/../worklog/static/vendor
mkdir -p "$D"
curl -fsSL https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js -o "$D/echarts.min.js"
curl -fsSL https://cdn.jsdelivr.net/npm/tabulator-tables@6.3.0/dist/js/tabulator.min.js -o "$D/tabulator.min.js"
curl -fsSL https://cdn.jsdelivr.net/npm/tabulator-tables@6.3.0/dist/css/tabulator.min.css -o "$D/tabulator.min.css"
```

Design the dark palette in `style.css` with CSS custom properties (`--bg
#0f1216`, `--panel #161b22`, `--line #263041`, `--text #e6edf3`, `--muted
#8b98a9`, `--accent #4cc9f0`, `--good #3ddc97`, `--warn #f4b942`, `--bad #ff6b6b`).
`index.html`: header with title, `#statusline`, `#refresh` button, KPI strip
(`#kpis`); the filter row (`#filters`); the matrix panel; a two-column row with
heatmap and feed; `<dialog id="modal">`. Load vendor CSS/JS, then `util.js`,
`matrix.js` (Task 7), `charts.js`/`feed.js` (Task 8), `app.js` (Task 8) with
`defer`. Put placeholder text in the panels so the page is legible before data.

- [ ] **Step 4: Run**

Run: `pytest projects/worklog/tests/test_static.py -v`
Expected: 1 passed (run `bash projects/worklog/scripts/vendor.sh` first).

- [ ] **Step 5: Commit**

```bash
git add projects/worklog/scripts projects/worklog/worklog/static projects/worklog/tests/test_static.py
git commit -m "worklog: page shell, palette and vendored charts"
```

---

### Task 7: Matrix, filters and drill-down

**Files:**
- Create: `projects/worklog/worklog/static/matrix.js`
- Modify: `projects/worklog/worklog/static/index.html`, `style.css`

**Interfaces:**
- Consumes: `snapshot.slices`, `snapshot.matrix`, `snapshot.filters`, `snapshot.model_families`; `util.js`
- Produces: `window.Worklog.matrix.render(snapshot, state)` and
  `window.Worklog.matrix.drill(model, family)`; `state = {campaign, modelFamily,
  task, benchmark, language, metric}`; matrix emits `custom` events on cell click.

- [ ] **Step 1: Implement the matrix renderer**

Rows: models, grouped by `model_families` (`Baselines`, `MELT`, `SMURF`,
`Other`) with a group header row spanning the columns. Columns: the families of
the selected task, in config order. Cell: the aggregated value over the slice
rows matching `(model, family)` and the filters, formatted by the metric's label
(`WER %` with two decimals, BLEU/chrF one), tinted by rank within the column
(best = `--good` background at 18 % alpha, worst = `--bad`), `…` for pending, `–`
for empty, a `•` marker when the slice has a failed run in `snapshot.runs`. Best
value per column gets a `★` in the column header. Ties do not mark.

Clicking a cell opens `#modal` with a table (Tabulator) of that family's slices
for the model: language/direction, value, samples, status, log path (as text with
a copy button). The table sorts by value.

- [ ] **Step 2: Verify by hand against a fixture**

Run: `python -m http.server` is **not** the test. Start the app later (Task 9) and
check that a cell's value equals the mean of the drill-down rows for `FLEURS
X→EN` (22 languages, two of them 20.0 and 30.0 in the test fixture ⇒ 25.0).
Record the checked pair in the commit message.

- [ ] **Step 3: Commit**

```bash
git add projects/worklog/worklog/static
git commit -m "worklog: matrix with family columns and drill-down"
```

---

### Task 8: Heatmap, run feed, live refresh and URL state

**Files:**
- Create: `projects/worklog/worklog/static/charts.js`, `feed.js`, `app.js`
- Modify: `index.html`, `style.css`

**Interfaces:**
- Consumes: snapshot, `matrix.js`
- Produces: `window.Worklog.app` that owns `state`, fetches `/api/snapshot`,
  polls every 10 s, and re-renders all panels; state serialized to
  `location.hash` (`#task=st&benchmark=fleurs24-st-xen-dev&metric=corpus_bleu&language=de`).

- [ ] **Step 1: Heatmap**

`charts.js`: `render(el, snapshot, state)` builds an ECharts `heatmap` of models
(rows) × languages (columns) for the selected family and metric, values from the
slice rows, `visualMap` sequential (`#1d3557` → `#4cc9f0` → `#3ddc97` for
higher-is-better, inverted for WER). Tooltip: model, language, value, samples.
Resize with the window.

- [ ] **Step 2: Feed**

`feed.js`: `render(el, snapshot.runs)` builds a list: status chip (`success` /
`error` / `cancelled` / `queued`), campaign › model, slice, absolute finish time,
duration from `created`→`finished`, and the log path in a `title`. Errors first,
then newest. "Load more" calls `/api/runs?offset=`.

- [ ] **Step 3: App wiring**

`app.js`: on load fetch `/api/snapshot`, populate `#filters` (campaign, model
family, task, benchmark, language — each a `<select>` built from
`snapshot.filters`), restore state from the hash, render. On any filter change,
update the hash and re-render without a fetch. Poll `/api/snapshot?version=`:
`204` → nothing; `200` → replace snapshot, re-render, keep scroll. `#refresh`
POSTs `/api/refresh` then refetches. `#statusline` shows "last refresh HH:MM:SS ·
N slices · M warnings". A `warnings` count in `kpis` renders an amber chip.

- [ ] **Step 4: Verify**

Run the app (Task 9), hard-reload twice: filters survive the reload from the
hash; a screenshot in the PR description shows matrix, heatmap and feed with the
fixture data. Check the browser console is clean.

- [ ] **Step 5: Commit**

```bash
git add projects/worklog/worklog/static
git commit -m "worklog: heatmap, run feed and live refresh"
```

---

### Task 9: README, real-log check and deployment

**Files:**
- Create: `projects/worklog/README.md`
- Modify: `projects/worklog/worklog.yaml` (only if the real run reveals a needed root)

- [ ] **Step 1: Real-log check**

```bash
cd projects/worklog
WORKLOG_CACHE=/tmp/worklog-smoke.sqlite python app.py \
    --root /mnt/scratch-artemis/giuseppe/melt-data/eval-runs \
    --port 8765 &
sleep 3
curl -s localhost:8765/api/health
curl -s "localhost:8765/api/snapshot" | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); print(d['kpis']); print(sorted({s['benchmark'] for s in d['slices']}))"
kill %1
```

Expected: `{"ok": true, ...}` and the benchmarks of the Canary campaign (plus any
other `speechval-*` campaigns present). If `kpis.slices` is 0, the include glob
or the root is wrong — check `curl localhost:8765/api/runs | head`.

- [ ] **Step 2: Compare one number to the campaign report**

Pick one slice from `/api/snapshot` and compare its metric with the same
`(benchmark, slice, metric)` in
`${EVAL_ROOT}/speechval-canary/results/results_long.csv`. They must agree
exactly; a mismatch means scorer-name flattening picked the wrong entry.

- [ ] **Step 3: Write the README**

Cover: what it is, the read-only contract, `uv pip install -r requirements.txt`,
the start command (with `--root` and `--cache`), which config keys exist, how to
add a campaign root and a benchmark family, the no-auth warning, and the
`--host 0.0.0.0` + SSH-tunnel note. State that `results/*.csv` is *not* read.

- [ ] **Step 4: Full test run and lint**

```bash
pytest projects/worklog/tests -q
ruff check projects/worklog
```

Expected: all pass, no ruff findings.

- [ ] **Step 5: Commit**

```bash
git add projects/worklog/README.md projects/worklog/worklog.yaml
git commit -m "worklog: deployment notes and real-log verification"
```

---

## Self-review notes

- Spec Part B sections map to: B1→Tasks 1/5/6, B2→Task 2, B3→Task 3, B4/B4b→Task 4,
  B5→Tasks 6–8, B6→Task 8, B7→Task 1, B8→Tasks 2/4/9, B9→every task's test step,
  B10→Task 9.
- B4's "server-computed default matrix" lives in Task 4; the client re-aggregates
  after a filter or metric change with `util.js`'s `average(values, kind)`
  (macro or sample-weighted), which mirrors `views.aggregate` and is checked by
  eye against the drill-down in Task 7.
- The app's scan callback is injected in `create_app` so `/api/refresh` is
  testable without a thread.
