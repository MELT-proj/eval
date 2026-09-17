"""Read a campaign's inspect logs into one tidy table.

A *campaign* is one `submit_campaign_mn5.sh` sweep: every checkpoint of one
training run, evaluated on the ASR frozen set (one log covering every corpus
and language at once) and on the ST frozen set (one log per translation
direction). This module turns a directory of those `.eval` files into a single
long-format frame with one row per
``(run, checkpoint, task, dataset_id, lang, metric)``.

**Metrics are recomputed from the per-sample records, not read off the log
header.** The header carries corpus WER/CER broken down by *language only*
(``melteval/scorers.py`` groups on ``lang``), while a single ASR log mixes four
corpora — so ``wer_it`` there pools voxpopuli, mls_sidon and cv22_sidon into one
number, and no per-corpus figure exists to read. Re-aggregating from the raw
counts each sample carries gives the per-``(corpus, language)`` cell the header
cannot, and it is the *same* arithmetic the corpus metric does (total errors
over total reference units — never the mean of per-sample rates). The
recomputation is checked against the header in :func:`header_metrics`; see
``report.py``, which asserts the two agree.

Reading goes through ``read_eval_log_sample_summaries``, which pulls only the
``summaries.json`` member of the log's zip. That member already carries each
sample's metadata *and* its score value, so a 490 MB ASR log parses in ~20 s
instead of the several minutes a full ``read_eval_log`` would spend inflating
160k individual sample records that this module never looks at.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log, read_eval_log_sample_summaries


logger = logging.getLogger(__name__)

#: `melteval` registers itself as an inspect provider, so a log's model name is
#: `melt/<checkpoint path>`. The path is what identifies the run.
MODEL_PREFIX = "melt/"

#: Trailing `-s<seed>` in a run name, e.g. `IFT-700both-...-s1337-8g`.
SEED_RE = re.compile(r"-s(\d+)(?:-|$)")

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")

#: Label used for the run root (the end-of-training weights), which sits
#: alongside the numbered `checkpoint-N/` directories rather than inside one.
FINAL = "final"

#: `dataset_id` of the pooled ASR cell (every corpus and language at once).
#: Upper case so it sorts ahead of the real corpora in a spreadsheet header.
OVERALL = "OVERALL"


# =============================================================================
# Log identity
# =============================================================================


@dataclass(frozen=True)
class LogInfo:
    """What a log's header says about which model ran which task."""

    path: str
    status: str
    run_name: str
    checkpoint: str
    step: int
    is_final: bool
    seed: int | None
    task_filter: str
    dataset_filter: str | None
    lang_filter: str | None
    n_samples: int
    created: str

    @property
    def model(self) -> str:
        """Display name: the run name, with `-ckptNNNN` for a checkpoint."""
        return self.run_name if self.is_final else f"{self.run_name}-ckpt{self.step:04d}"


def _parse_model_path(model: str) -> tuple[str, str, int, bool]:
    """Split an inspect model name into ``(run_name, checkpoint, step, is_final)``.

    The checkpoint path in the header is used rather than the log's position on
    disk: a campaign's log tree is a convention (one folder per checkpoint), but
    the model path is what the job actually loaded, so it stays right even for
    logs that were moved, re-nested, or written by a one-off run.

    A run root scores as ``final`` with the step of the last numbered
    checkpoint plus one median gap — an estimate, since nothing in the log
    records the true ``max_steps``. It only positions the point on the x-axis;
    :func:`build_frame` recomputes it once the whole run is known.
    """
    path = Path(model.removeprefix(MODEL_PREFIX))
    match = CHECKPOINT_RE.match(path.name)
    if match:
        return path.parent.name, path.name, int(match.group(1)), False
    return path.name, FINAL, -1, True


def read_log_info(path: Path) -> LogInfo:
    """Read one log's header (no sample data)."""
    header = read_eval_log(str(path), header_only=True)
    run_name, checkpoint, step, is_final = _parse_model_path(header.eval.model)
    seed = SEED_RE.search(run_name)
    args = header.eval.task_args or {}
    return LogInfo(
        path=str(path),
        status=str(header.status),
        run_name=run_name,
        checkpoint=checkpoint,
        step=step,
        is_final=is_final,
        seed=int(seed.group(1)) if seed else None,
        task_filter=str(args.get("task_filter") or ""),
        dataset_filter=args.get("dataset_id"),
        lang_filter=args.get("lang"),
        n_samples=header.eval.dataset.samples or 0,
        created=str(header.eval.created),
    )


def discover_logs(roots: str | Path | list[str | Path]) -> list[Path]:
    """Find every `.eval` file under *roots*, deduplicated and sorted."""
    if isinstance(roots, (str, Path)):
        roots = [roots]
    found: set[Path] = set()
    for root in roots:
        root = Path(root)
        if root.is_file():
            found.add(root.resolve())
        else:
            found.update(p.resolve() for p in root.rglob("*.eval"))
    return sorted(found)


# =============================================================================
# Per-log metric extraction
# =============================================================================


def _asr_cells(summaries) -> dict[tuple, dict]:
    """Sum ASR edit counts per ``(dataset_id, lang)``.

    Corpus WER is total word errors over total reference words, so the counts
    are summed and divided once at the end. Averaging the per-sample rates
    instead would weight a three-word utterance the same as a thirty-word one.
    """
    acc: dict[tuple, dict] = defaultdict(
        lambda: {"wer_errors": 0, "ref_words": 0, "cer_errors": 0, "ref_chars": 0, "n": 0, "seconds": 0.0}
    )
    for summary in summaries:
        meta = summary.metadata or {}
        score = next(iter(summary.scores.values()), None)
        if score is None or not isinstance(score.value, dict):
            continue
        cell = acc[(meta.get("dataset_id", ""), meta.get("lang", ""), "", "")]
        for key in ("wer_errors", "ref_words", "cer_errors", "ref_chars"):
            cell[key] += int(score.value.get(key, 0))
        cell["n"] += 1
        cell["seconds"] += float(meta.get("duration") or 0.0)

    out = {}
    for key, cell in acc.items():
        out[key] = {
            "n_samples": cell["n"],
            "hours": cell["seconds"] / 3600.0,
            "wer": _ratio(cell["wer_errors"], cell["ref_words"]),
            "cer": _ratio(cell["cer_errors"], cell["ref_chars"]),
            # The counts travel with the rate so any later regrouping -- across
            # corpora, across languages, across the several logs a split
            # campaign writes per checkpoint -- can sum them and divide once.
            # A weighted mean of the rates is not the same number, and the
            # difference is exactly the length bias corpus WER exists to avoid.
            "wer_counts": (cell["wer_errors"], cell["ref_words"]),
            "cer_counts": (cell["cer_errors"], cell["ref_chars"]),
        }
    return out


def _ratio(errors: int, total: int) -> float:
    return errors / total if total else float("nan")


def _st_cells(info: LogInfo, summaries) -> dict[tuple, dict]:
    """Corpus BLEU/chrF for an ST log, read off the log header.

    ST is the one place the sample summaries must *not* be trusted: inspect
    truncates long strings when it writes ``summaries.json`` (a hypothesis over
    ~1000 characters comes back ending in ``...``), and a checkpoint that has
    not learned to stop produces exactly those — degenerate repetition loops.
    Recomputing BLEU over the truncated text quietly moves the score (on the
    llama-1b final checkpoint's pt→en log, 5 of 4023 samples were clipped and
    BLEU came out 1.763 instead of 1.753) and it moves *more* the worse the
    checkpoint is, which is the opposite of a harmless bias. ASR is unaffected:
    its scorer stores integer edit counts, not text.

    So the numbers come from the header, where the real scorer already computed
    them over the untruncated output. That is exact and costs one small read
    instead of inflating tens of thousands of full sample records — and it is
    sound because a `melteval` ST run is always a single direction: the scorer
    raises on mixed target languages rather than pick one BLEU tokenizer for
    two. That invariant is asserted below rather than assumed.

    Raises:
        ValueError: If the log covers more than one direction, since the single
            header BLEU then belongs to no one cell.
    """
    acc: dict[tuple, dict] = defaultdict(lambda: {"n": 0, "seconds": 0.0})
    for summary in summaries:
        meta = summary.metadata or {}
        tgt = meta.get("tgt_lang") or meta.get("lang", "")
        cell = acc[(meta.get("dataset_id", ""), meta.get("lang", ""), meta.get("src_lang", ""), tgt)]
        cell["n"] += 1
        cell["seconds"] += float(meta.get("duration") or 0.0)

    if len(acc) > 1:
        raise ValueError(
            f"{info.path} mixes {len(acc)} translation directions ({sorted(acc)}). A corpus BLEU over "
            "several directions has no single correct tokenizer, so the header metric this reads cannot "
            "be attributed to any one of them. Re-run the eval one direction at a time (-T dataset_id=...)."
        )

    header = header_metrics(info)
    return {
        key: {
            "n_samples": cell["n"],
            "hours": cell["seconds"] / 3600.0,
            "bleu": header.get("corpus_bleu", float("nan")),
            "chrf": header.get("corpus_chrf", float("nan")),
        }
        for key, cell in acc.items()
    }


#: Which metrics each task contributes, in the order they should be shown.
TASK_METRICS = {"asr": ("wer", "cer"), "st": ("bleu", "chrf")}


def _corpus_of(dataset_id: str, direction: str) -> str:
    """Strip a trailing `-<direction>` from an ST corpus id (`covost2-pt_en`)."""
    suffix = f"-{direction}"
    return dataset_id[: -len(suffix)] if direction and dataset_id.endswith(suffix) else dataset_id


def extract_rows(info: LogInfo) -> list[dict]:
    """Turn one log into long-format rows, one per (cell, metric)."""
    summaries = read_eval_log_sample_summaries(info.path)
    if info.task_filter == "asr":
        cells = _asr_cells(summaries)
    elif info.task_filter == "st":
        cells = _st_cells(info, summaries)
    else:
        logger.warning("Skipping %s: unsupported task_filter %r", info.path, info.task_filter)
        return []

    rows = []
    for (dataset_id, lang, src_lang, tgt_lang), values in cells.items():
        direction = f"{src_lang}_{tgt_lang}" if src_lang and tgt_lang else ""
        base = {
            "run_name": info.run_name,
            "model": info.model,
            "seed": info.seed,
            "checkpoint": info.checkpoint,
            "step": info.step,
            "is_final": info.is_final,
            "task": info.task_filter,
            "dataset_id": dataset_id,
            # ST corpora are registered per direction (`covost2-pt_en`), so the
            # direction is already in the id. Split it back out to keep the
            # corpus comparable across directions and the column headers from
            # saying `pt_en` twice.
            "corpus": _corpus_of(dataset_id, direction),
            "lang": lang,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "direction": direction,
            "n_samples": values["n_samples"],
            "hours": values["hours"],
            "log_path": info.path,
        }
        for metric in TASK_METRICS[info.task_filter]:
            # `numerator`/`denominator` are the ratio metrics' raw counts, and
            # empty for BLEU/chrF -- which is not an omission: BLEU has no
            # denominator to pool over, which is why the ST summary is a macro
            # average and cannot be anything else.
            numerator, denominator = values.get(f"{metric}_counts", (float("nan"), float("nan")))
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "value": values[metric],
                    "numerator": numerator,
                    "denominator": denominator,
                }
            )
    return rows


def pool_asr(frame: pd.DataFrame) -> pd.DataFrame:
    """Add each model's pooled ``OVERALL`` ASR row, summed from the counts.

    Done over the whole frame rather than per log, so it comes out the same
    whether a checkpoint's ASR ran as one job over the entire frozen set or as
    one job per ``(corpus, language)`` — the latter being what the campaign
    submits, since those are the units the report is built from anyway.
    """
    asr = frame[(frame["task"] == "asr") & frame["denominator"].notna()]
    if asr.empty:
        return frame
    pooled = asr.groupby([*ID_COLUMNS, "task", "metric"], as_index=False).agg(
        numerator=("numerator", "sum"),
        denominator=("denominator", "sum"),
        n_samples=("n_samples", "sum"),
        hours=("hours", "sum"),
    )
    # n_samples/hours were summed once per metric, so each is counted once per
    # cell -- correct, because a cell contributes one row per metric.
    pooled["value"] = pooled["numerator"] / pooled["denominator"]
    pooled = pooled.assign(
        dataset_id=OVERALL, corpus=OVERALL, lang="all", src_lang="", tgt_lang="", direction="", log_path=""
    )
    return pd.concat([frame, pooled], ignore_index=True)


def header_metrics(info: LogInfo) -> dict[str, float]:
    """The log header's own corpus metrics, for cross-checking :func:`extract_rows`."""
    header = read_eval_log(info.path, header_only=True)
    if not header.results:
        return {}
    return {
        name: metric.value
        for score in header.results.scores
        for name, metric in score.metrics.items()
        if isinstance(metric.value, (int, float))
    }


# =============================================================================
# Whole-campaign frame
# =============================================================================


def _fingerprint(paths: list[Path]) -> str:
    """Hash the log set's (path, size, mtime) so a stale cache is never reused."""
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(f"{path}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _place_final(frame: pd.DataFrame) -> pd.DataFrame:
    """Give each run's ``final`` row an x-position past its last checkpoint.

    Nothing in an eval log records the training step the run root corresponds
    to, so it gets the last numbered checkpoint plus one median gap. That is a
    plotting position, not a fact about the run: read the point by its `final`
    label, not by its step. (In practice a run's last numbered checkpoint is
    often the end of training itself, in which case `final` simply repeats it —
    visible in the chart as a flat final segment.)
    """
    frame = frame.copy()
    for run, group in frame.groupby("run_name"):
        steps = sorted(group.loc[~group["is_final"], "step"].unique())
        if not steps:
            # Only the run root was evaluated: there is no scale to place it on.
            frame.loc[(frame["run_name"] == run) & frame["is_final"], "step"] = 0
            continue
        gaps = [b - a for a, b in zip(steps, steps[1:])]
        gap = int(pd.Series(gaps).median()) if gaps else max(steps[-1], 1)
        frame.loc[(frame["run_name"] == run) & frame["is_final"], "step"] = steps[-1] + gap
    return frame


def build_frame(
    roots: str | Path | list[str | Path],
    *,
    cache: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Extract every log under *roots* into one long-format frame.

    Args:
        roots: Campaign log directories (or individual `.eval` files).
        cache: Parquet file to read/write. Reused only when the log set's
            paths, sizes and mtimes are unchanged, since re-reading a full
            campaign costs a couple of minutes.
        refresh: Ignore any existing cache and re-extract.

    Returns:
        One row per ``(model, task, dataset_id, lang, metric)``, with a
        ``value`` column and the run/checkpoint columns to group by.
    """
    paths = discover_logs(roots)
    if not paths:
        raise FileNotFoundError(f"No .eval logs under {roots!r}.")

    stamp = Path(f"{cache}.json") if cache else None
    reusable = cache and not refresh and Path(cache).exists() and stamp and stamp.exists()
    if reusable and json.loads(stamp.read_text()).get("fingerprint") == _fingerprint(paths):
        logger.info("Reusing cached frame at %s", cache)
        return pd.read_parquet(cache)

    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for index, path in enumerate(paths, start=1):
        info = read_log_info(path)
        if info.status != "success":
            # A cancelled or errored run has scored only part of its set, so
            # its numbers are not comparable to a complete one -- drop it
            # loudly rather than quietly average an unknown subset.
            skipped.append((str(path), info.status))
            continue
        logger.info("[%d/%d] %s %s %s", index, len(paths), info.model, info.task_filter, path.name)
        rows.extend(extract_rows(info))

    if skipped:
        logger.warning("Skipped %d log(s) that did not finish: %s", len(skipped), skipped)
    if not rows:
        raise ValueError(f"No successful logs under {roots!r}.")

    frame = pool_asr(_place_final(pd.DataFrame(rows)))
    frame = frame.sort_values(["run_name", "step", "task", "dataset_id", "lang", "metric"]).reset_index(drop=True)

    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache, index=False)
        stamp.write_text(json.dumps({"fingerprint": _fingerprint(paths), "n_logs": len(paths)}, indent=2))
    return frame


def log_index(roots: str | Path | list[str | Path]) -> pd.DataFrame:
    """One row per discovered log: what it is and whether it finished."""
    return pd.DataFrame([asdict(read_log_info(p)) for p in discover_logs(roots)])


# =============================================================================
# Spreadsheet tables
# =============================================================================

#: Columns identifying the row's model, carried to the front of every table.
ID_COLUMNS = ["model", "run_name", "seed", "checkpoint", "step", "is_final"]


def _pivot(frame: pd.DataFrame, column_key: list[str]) -> pd.DataFrame:
    """Pivot long rows to one row per model, one column per metric/cell."""
    wide = frame.pivot_table(index=ID_COLUMNS, columns=["metric", *column_key], values="value", sort=False)
    # `wer/voxpopuli/it` rather than a MultiIndex header: a CSV has one header
    # row, and a flattened name survives the trip into a spreadsheet intact.
    wide.columns = ["/".join(str(part) for part in col if str(part)) for col in wide.columns]
    return wide[sorted(wide.columns)].reset_index().sort_values(["run_name", "step"]).reset_index(drop=True)


def asr_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per model; one column per ``{wer,cer}/<corpus>/<language>``.

    Also carries ``wer/OVERALL/all`` and ``cer/OVERALL/all``: the corpus rate
    over every cell pooled, summed from the counts rather than averaged across
    cells, so it matches what the eval log's own top-line metric reports.
    """
    asr = frame[frame["task"] == "asr"]
    if asr.empty:
        return pd.DataFrame(columns=ID_COLUMNS)
    return _pivot(asr, ["corpus", "lang"])


def st_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per model; one column per ``{bleu,chrf}/<corpus>/<direction>``.

    ``bleu/MACRO-AVG`` is the unweighted mean over directions. It is a summary
    of the six numbers next to it, not a corpus score: BLEU's brevity penalty
    is defined over one corpus, so there is no pooled BLEU across language
    pairs to compute.
    """
    st = frame[frame["task"] == "st"]
    if st.empty:
        return pd.DataFrame(columns=ID_COLUMNS)
    table = _pivot(st, ["corpus", "direction"])
    for metric in ("bleu", "chrf"):
        columns = [c for c in table.columns if c.startswith(f"{metric}/")]
        if columns:
            table[f"{metric}/MACRO-AVG"] = table[columns].mean(axis=1)
    return table
