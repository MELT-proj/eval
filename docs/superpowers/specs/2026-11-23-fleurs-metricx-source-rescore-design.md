# FLEURS MetricX source backfill — design

2026-11-23. Status: proposed (awaiting approval).

## Intent and why

MetricX-24 hybrid XXL is reference-based: upstream's input is
`source: <src> candidate: <hyp> reference: <ref>`. CoVoST2 and Europarl are
scored that way. The FLEURS ST slices on artemis were scored with `src: ""`
(reference-only), because their frozen sets carry no `source_text`. That makes
the FLEURS column incomparable with the other two and the numbers themselves
unreliable.

The goal is one setup across all three ST benchmarks, per-sample scores so
bootstrap CIs can be computed, and a frozen-set path that gives every future
FLEURS eval the source without anyone remembering to do anything.

## What I found (recon, 2026-11-23)

**The configs are already correct.** `configs/shar/fleurs24-st-enx-dev.yaml`
and `fleurs24-st-xen-dev-full.yaml` have had
`source_text_field: supervisions.0.text` since `2cb6eb7` (2026-11-10).

**The frozen sets on artemis are stale, and the venv is why.** The artemis
venv's editable install of `melteval` resolves to
`/mnt/scratch-artemis/giuseppe/eval-mcif/melteval`, a different checkout whose
`readers/shar.py::_nested_value` has no list-index branch. `supervisions.0.text`
therefore resolved to nothing and every record was frozen with
`source_text: null` — silently, with the correct spec hash
(`e77d0be19482` / `968f8151b34e`). Freezing the same spec with the repo on
`PYTHONPATH` (`PYTHONPATH=. melteval freeze …`; the console script otherwise
wins over the cwd) yields the text. Evidence: all 8819 en→X and 8261 X→en
records are null.

**Jupiter is already fine.** Both `/e/data1/.../speech/frozen-dev` and
`/e/project1/.../frozen-dev` FLEURS manifests have 0 null `source_text`, and
the Qwen3-Omni logs (synced from Jupiter) show 0 of 363 samples with an empty
`metricx_triple.src`. So there is nothing to re-freeze there.

**Scope.** 381 current FLEURS logs across the worklog rows we own:

| row | logs | src state |
|---|---|---|
| 7 fondue checkpoints × 44 | 308 | empty, must rescore |
| canary-1b-v2 × 44 | 44 | empty, must rescore |
| qwen3-omni × (6 X→en + 23 en→X) | 29 | already full triples; rescore only to add per-sample values |

44 per artemis row, not 46: `ga` is skipped in both directions by those
campaigns. No slice dir has more than one current (non-`.superseded`) log.

**`--action overwrite` drops sibling scorers.** Verified on a throwaway copy:
scoring with only `metricx_scorer` leaves `results.scores` containing only
`metricx_scorer`, deleting `corpus_bleu`/`corpus_spbleu`. Re-running
`st_scorer` reproduces BLEU/chrF/spBLEU bit-identically (34.467871 / 59.891844
/ 40.823327 on one log).

**`--scorer` takes one scorer per invocation.** In the pinned
`inspect_ai==0.3.254` the CLI option is scalar (`_cli/score.py:68`;
`resolve_scorers` returns a one-element list, `_eval/score.py:531`), and `-S`
is splatted into that one scorer. A second `--scorer` silently replaces the
first. The way out is to pass **no** `--scorer`: inspect then rebuilds every
scorer recorded in `log.eval.scorers` (verified on a throwaway log — both
entries come back, with their own metric lists). `-S` is ignored on that path,
so the manifest reaches `metricx_scorer` through an environment variable
instead (see §1). This works only if both scorers are in inspect's registry:
the entry point imports `melteval`, whose `__init__` imported `melteval.scorers`
but not `melteval.rescore`, so `metricx_scorer` was not registered and the first
smoke job died on `LookupError: metricx_scorer was not found`. `__init__` now
imports both; the module is dependency-free at import time.

**A root-level `nan` in `Score.value` is inspect's "unscored" sentinel.** It
drops the sample from every metric (`_eval/task/results.py:400`), which would
take unscorable samples out of the coverage denominators. So an unscorable
sample carries `-1.0` — out of MetricX's [0, 25] range — and the metrics filter
it, instead of a fake 0 or a `nan`.

## Goals

1. Rescore MetricX with full `(src, mt, ref)` triples for all 381 logs, on
   artemis, through SLURM.
2. Store each sample's MetricX in the log; the corpus value is their mean.
3. Report source coverage per slice and refuse to score a slice below 100%.
4. Rebuild the artemis FLEURS frozen sets so future evals carry the source,
   and make a silently-empty `source_text_field` visible at freeze time.
5. Replace the old, reference-only values in place; leave BLEU/chrF/spBLEU
   exactly as they are.

## Non-goals

- CoVoST2 and Europarl: already full triples, not touched.
- SMURF/EuroLLM-speech rows: the owner will handle them separately.
- Re-freezing Jupiter's FLEURS sets: they already carry the source (verify and
  report only).
- Re-running generation, or changing any reference/hypothesis.
- Pushing the rescored qwen copies back to Jupiter.

## Anti-goals (this would make the work a failure)

- Any slice losing `corpus_bleu`, `corpus_spbleu` or `corpus_chrf` from its
  results.
- Any scored log whose samples still have an empty `src` when a source exists.
- A re-freeze that changes `sample_key`, `audio`, `target` or `cut_id`, which
  would orphan the existing logs.
- GPU work outside SLURM, or anything written under `/mnt/home`.

## Constraints

- `melteval` must be imported from this checkout for any freeze
  (`PYTHONPATH=.`), because the venv's editable install points elsewhere.
- `spec_hash` will not change on re-freeze (the spec is unchanged), so the
  rebuilt sets must be verified by comparing records, not by hash.
- `SACREBLEU` must be set when `st_scorer` is re-run (`infra/sites/artemis.sh`
  does it; `run_score.sbatch` gets it via `submit_score.sh`).
- 381 logs × ~290–394 samples ≈ 126k MetricX inputs. One GPU job can loop
  several logs, but the model loads once per `inspect score` process.
- Logs are large (fondue campaign 11 GB); no full copies of the old logs.
  The old metricx means and triples are snapshotted to a small CSV/JSONL.

## Approaches considered

**A. Per-sample scorer, native in the log (chosen).** `metricx_scorer` scores
each sample and returns `Score(value=<score>)`; the corpus metric is the mean
of those values. A `source_manifest` scorer argument supplies the text for
logs whose sample metadata predates the fix. Logs become self-describing, and
future runs get per-sample values with no extra plumbing. Cost: batch-of-1
inference with a cached model, roughly 60–90 s per ~330-sample log instead of
~30 s.

**B. Keep the batched metric, write a sidecar JSONL.** Fast, but per-sample
values live beside the log rather than in it, every consumer needs to learn
about the sidecar, and future runs only get it if a path is threaded through
the launcher.

**C. Patch the log JSON directly.** Fastest (one model load for everything),
but it is hand-surgery on inspect's log format and has to fake `results` and
events; rejected.

**D. `--action append` with the existing scorer name.** Inspect dedupes the
name (`metricx_scorer-1`), so results grow a second entry rather than being
replaced. Rejected.

**E. Two `inspect score` passes (overwrite with `st_scorer`, then append with
`metricx_scorer`).** Works, but doubles the rewrites, needs a second sbatch
script, and leaves a window per log where MetricX is absent. Rejected once
inspect's no-`--scorer` path turned out to rebuild both scorers in one go.

A was picked because the artifact the user wants to bootstrap over is the log
itself, and because "all future evals get it" is otherwise a config change
someone has to remember.

## Design

### 1. `melteval/rescore.py`

`metricx_scorer(source_manifest: str | None = None)`:

- `source_manifest` comes from the argument, or, when the argument is absent,
  from `MELTEVAL_SOURCE_MANIFEST`. The env fallback exists only because
  `inspect score` drops `-S` when it rebuilds scorers from the log header
  (§4); it is documented as a backfill knob, not the normal path. Either way
  the path used is recorded per sample (`metricx_manifest`), so a log says
  where its sources came from.
- Source resolution, per sample: `state.metadata["source_text"]` when non-empty;
  otherwise, when a manifest is in play, a lookup keyed by
  `(metadata["dataset_id"], state.sample_id)` from that frozen set's
  `manifest.jsonl`; otherwise `""`. The path used is recorded in the score
  metadata (`metricx_source`: `sample` | `manifest` | `none`).
  - The composite key is not optional. `sample_key` is
    `<source_index>-<ordinal>` and collides between the two FLEURS sets
    (`00-000000` is a different sentence in X→en and en→X), so a key-only join
    would pass every coverage check while scoring the wrong language's
    transcript. On a hit, also assert that the manifest record's `src_lang`
    and `tgt_lang` equal the sample's; a mismatch is a scorer error, not a
    silent `""`.
- Per-sample `Score(value=float)`, `-1.0` (`UNSCORED`) when the completion is
  empty, so inspect does not treat it as its NaN sentinel; the
  `metricx_triple` stays in `Score.metadata` as today.
- The model is loaded once per process behind a module-level cache and a lock
  (`inspect score` may call the scorer for several samples concurrently).

Metrics on the scorer:

- `metricx()` — mean over the per-sample values in [0, 25]. Same number as
  the old batched mean for the same triples.
- `metricx_coverage()` — fraction of samples with a value (unchanged meaning:
  the rest had no output).
- `metricx_source_coverage()` — fraction of samples with a non-empty resolved
  source, over **all** samples. This is the record that makes a regression
  visible, and it is what the 100% gate is about; an empty completion is the
  separate, pre-existing concern of `metricx_coverage` (one canary slice,
  `st-fleurs-sk_en-dev`, sits at 0.9972 and stays there).

`metricx_predict` keeps its signature (list in, list out, internal batching)
so the existing test and any future caller still work.

### 2. `melteval/freeze.py`

When a source sets `source_text_field`, count records whose `source_text` is
empty and add `source_text_missing` to that source's stats in
`frozen_set.json`. If every record is missing, log a warning naming the field.
Not an error: some specs legitimately point at an optional field (the FLEURS
test spec's `custom.pnc_text` is absent for Irish).

### 3. Rebuild the artemis FLEURS sets

`PYTHONPATH=. melteval freeze configs/shar/fleurs24-st-enx-dev.yaml -o
/mnt/scratch-artemis/giuseppe/frozen/speechval-dev/fleurs24-st-enx-dev`, and the
same for `fleurs24-st-xen-dev-full`. Keep the old `manifest.jsonl` as
`manifest.jsonl.prev`.

Acceptance check, before anything is rescored: the new manifest has the same
record count, and every record's `sample_key`, `audio`, `target` and `cut_id`
is identical to `.prev`; `source_text` is non-empty for 100% of records.

### 4. `scripts/rescore_fleurs_metricx.py`

A focused backfill driver, not a change to the eval path.

- Inputs: `--site artemis`, one or more `--campaign` configs (default:
  `speechval-fondue`, `speechval-canary`, `speechval-qwen3-omni`), `--dry-run`,
  `--wait`.
- Discovers current logs under `<output_dir>/logs/<model>/<benchmark>/<slice>/`,
  ignoring `*.json.superseded`.
- **Preflight (CPU only, no GPU):** joins each log's `(sample.dataset_id,
  sample.id)` to its benchmark's manifest and writes
  `<campaign>/results/fleurs-source-coverage.csv`
  (`model, benchmark, slice, samples, with_source, coverage`). Exits non-zero
  if any slice is below 1.0. Nothing is submitted in that case. The join is
  keyed on `dataset_id` as well as the id precisely because the two FLEURS
  sets share sample keys; a key-only join reports a clean 100% while pointing
  every sample at the wrong language.
- **Snapshot:** writes the old per-slice `metricx` mean and `metricx_coverage`
  to `<campaign>/results/metricx-before.csv`, plus the old triples for the
  affected slices to `<campaign>/results/metricx-before.jsonl` (small; lets the
  change be audited without copying 12 GB of logs).
- **Submit:** one SLURM job per `(campaign, model, benchmark)` (18 jobs) through
  `VENV_PATH=<metrics venv> infra/runners/submit_score.sh artemis <logs…> -- …`
  with `SBATCH` args from `infra/sites/artemis.sh` and, per job,
  `MELTEVAL_SOURCE_MANIFEST=<that benchmark's artemis frozen set>`. Each log is
  scored in **one** pass:

  ```
  inspect score <log> --model mockllm/model --action overwrite --overwrite
  ```

  No `--scorer`: inspect rebuilds both scorers from `log.eval.scorers`, so
  `st_scorer` and `metricx_scorer` are recomputed together and the log cannot
  end up with one of them missing. `--action overwrite` replaces `results`,
  which is what removes the stale reference-only `metricx` value. The driver
  asserts `log.eval.scorers` is exactly `{st_scorer, metricx_scorer}` for every
  log before submitting; anything else (a judge, a third scorer) would be
  re-run too, so it aborts instead.
  `VENV_PATH` is the `metrics-eval` venv that produced the existing MetricX
  scores (inspect 0.3.254, transformers 4.57.6, weights in `HF_HOME`), not the
  site default. The existing `run_score.sbatch` needs no change: it already
  loops logs and rewrites each in place.
- **Verify (after the jobs):** per log — every sample with a non-empty
  completion has a finite per-sample value; source coverage is 1.0; the new
  `corpus_bleu`/`corpus_spbleu`/`corpus_chrf` equal the snapshot exactly; the
  new `metricx` equals the mean of the stored per-sample values. Writes
  `<campaign>/results/metricx-rescore.csv` (old vs new per slice).
- Regenerates `results/` with `projects/baselines/report.py`.

The scorer argument path comes from `/infra/sites/artemis.sh`'s
`FLEURS24_ST_*_DEV_SET`, not from the log's `eval.dataset.location`: the qwen
logs were produced on Jupiter and name paths that do not exist on artemis, and
`inspect score` does not need the dataset to re-score.

### 5. The `evaluate.py --rescore` hazard (documented, not fixed here)

`scripts/evaluate.py score --rescore` passes only the scorer being re-run to
`--action overwrite`, which deletes that job's other post-hoc scorers — on an
ST slice, BLEU disappears. Making it safe is not a small fix under a scalar
`--scorer`: it becomes a per-log ordered multi-pass plan, and with no
`--scorer` at all it would also re-run generation-time metrics. That is a
launcher refactor with its own tests, and this backfill does not need it (it
uses its own driver and two explicit passes). It goes in the PR description
and in `docs/speechval.md` as a known hazard, and wants its own issue.

Decision: keep this project focused on the FLEURS rescore. Say so if you'd
rather have the launcher fixed in the same PR.

### 6. Docs

- `docs/speechval.md`: the FLEURS MetricX source backfill (what changed, the
  numbers), and the freeze recipe pinned to this checkout
  (`PYTHONPATH=. melteval freeze …`) with the reason.
- `AGENTS.md`, "Things that are easy to get wrong here": one bullet on the
  venv's editable install pointing at a different checkout, and what it cost.

## Verification

- Unit: per-sample scoring with a fake model through a real `inspect_ai.eval()`
  (per-sample values, mean, both coverage metrics, manifest fallback keyed by
  `dataset_id`, and a wrong-language manifest raising instead of scoring);
  freeze warning/stat; coverage-join helper.
- One real slice through SLURM before the batch: per-sample values present,
  mean matches the recomputed mean, BLEU unchanged, `metricx_source_coverage`
  = 1.0, wall time recorded.
- Full batch with the preflight gate; post-run verification over all 381 logs.
- Spot-check three rows in the worklog UI/DB: MetricX changed to the
  source-aware value, BLEU unchanged.

## Order of work

1. Code: rescore per-sample + manifest fallback + coverage metric; freeze
   guard; tests.
2. Rebuild the two artemis frozen sets; verify record identity and 100% source.
3. Preflight coverage over the 381 logs; stop if anything is under 100%.
4. Smoke one slice on SLURM with both passes; record throughput.
5. Submit and run the 18 jobs; verify; write the CSVs; rebuild reports.
6. Docs, then the usual PR.

## Left to the builder

- Whether per-sample throughput needs a small async batcher. Plan for the
  simple serialised call; only add batching if a real slice is more than ~4×
  slower than the old batched metric, and say so in the PR.
- Job partitioning if the a6000 queue is saturated (a100/h100/h200 overrides
  are one env var away, with their own QoS).
