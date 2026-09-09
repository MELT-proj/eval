"""Campaign report: two spreadsheet CSVs and the charts, from a tree of inspect logs.

Run it either way — the notebook and the script do the same work:

    marimo edit projects/melt/report.py          # interactive
    python projects/melt/report.py               # headless, writes the artifacts

Paths come from the environment (``MELT_LOG_ROOT``, ``MELT_REPORT_OUT``,
``MELT_REPORT_REFRESH``) so both entry points agree; in the notebook the same
values are editable in the form at the top.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import logging
    import os
    import sys
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    # `campaign` and `charts` sit next to this notebook. Depending on how it is
    # launched (script, `marimo edit`, `marimo run`) that directory is not
    # necessarily on the path already, and importing by location keeps the
    # notebook runnable from any working directory.
    _here = Path(globals().get("__file__", "report.py")).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    import campaign
    import charts

    # A cold extraction reads every log and takes minutes; without this the
    # headless run sits silent the whole time with no way to tell it apart from
    # a hang. Only set up if the host has not configured logging already.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    charts.set_style()
    return Path, campaign, charts, mo, os, pd


@app.cell
def _(mo):
    mo.md(
        """
        # MELT eval campaign report

        One row per evaluated model, for every `(corpus, language)` cell of the ASR
        set and every direction of the ST set.

        A *model* here is one checkpoint: the run name for the end-of-training
        weights, and `<run>-ckptNNNN` for an intermediate one. `seed` is carried as
        its own column so repeats of a recipe under different initialisations can be
        averaged later without re-deriving it from the run name.
        """
    )
    return


@app.cell
def _(mo, os):
    log_root = mo.ui.text(
        value=os.environ.get(
            "MELT_LOG_ROOT", "/mnt/scratch-nyx/giuseppe/melt/eval-logs/mn5-campaign-v1"
        ),
        label="Campaign log root",
        full_width=True,
    )
    out_dir = mo.ui.text(
        value=os.environ.get("MELT_REPORT_OUT", "report-out"),
        label="Output directory",
        full_width=True,
    )
    refresh = mo.ui.checkbox(
        value=os.environ.get("MELT_REPORT_REFRESH", "") not in ("", "0", "false"),
        label="Ignore the cache and re-read every log",
    )
    mo.vstack([log_root, out_dir, refresh])
    return log_root, out_dir, refresh


@app.cell
def _(Path, campaign, log_root, mo):
    index = campaign.log_index(log_root.value)
    unfinished = index[index["status"] != "success"]
    mo.md(
        f"""
        Found **{len(index)}** logs under `{Path(log_root.value)}` covering
        **{index["run_name"].nunique()}** run(s) and **{index["checkpoint"].nunique()}**
        checkpoint(s).
        {"" if unfinished.empty else f"⚠️ **{len(unfinished)}** did not finish and are excluded below."}
        """
    )
    return (index,)


@app.cell
def _(index):
    index[["run_name", "checkpoint", "seed", "task_filter", "dataset_filter", "status", "n_samples"]]
    return


@app.cell
def _(campaign, log_root, out_dir, refresh):
    frame = campaign.build_frame(
        log_root.value,
        cache=f"{out_dir.value}/frame.parquet",
        refresh=refresh.value,
    )
    frame
    return (frame,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Cross-check against the log headers

        The per-cell numbers above are re-aggregated from each sample's own raw
        counts, because the log header breaks ASR down by language only and a single
        ASR log mixes four corpora. That re-aggregation is only trustworthy if it
        reproduces the header where the two *do* overlap — the pooled `OVERALL` cell
        against `corpus_wer`/`corpus_cer`, and each ST direction against its
        `corpus_bleu`. Anything but a rounding-level difference here means the
        extraction has drifted from the scorer, and the tables below should not be
        used until it is explained.
        """
    )
    return


@app.cell
def _(campaign, frame, index, pd):
    # Only headers are re-read here (a small member of each log's zip), never
    # the sample records again -- the numbers being checked are the ones
    # already in `frame`.
    _rows = []
    for _row in index[index["status"] == "success"].to_dict("records"):
        _info = campaign.LogInfo(**_row)
        _header = campaign.header_metrics(_info)
        _mine = frame[frame["log_path"] == _info.path]
        for _metric, _key in (("wer", "corpus_wer"), ("cer", "corpus_cer"), ("bleu", "corpus_bleu"), ("chrf", "corpus_chrf")):
            _cells = _mine[_mine["metric"] == _metric]
            if _cells.empty or _key not in _header:
                continue
            # Pool this log's own cells the way the scorer does -- sum the
            # counts, divide once -- and compare to what the scorer wrote.
            # For ST there is one cell and no counts, so the value stands.
            if _cells["denominator"].notna().all():
                _ours = _cells["numerator"].sum() / _cells["denominator"].sum()
            else:
                _ours = float(_cells["value"].iloc[0])
            _rows.append(
                {
                    "model": _info.model,
                    "task": _info.task_filter,
                    "metric": _metric,
                    "cells": len(_cells),
                    "header": _header[_key],
                    "recomputed": _ours,
                    "abs_diff": abs(_header[_key] - _ours),
                }
            )
        # Every sample must land in exactly one cell: a grouping bug that
        # dropped or double-counted samples could still add up to the right
        # pooled rate, so the partition is checked separately from the value.
        _counted = int(_mine[_mine["metric"] == ("wer" if _info.task_filter == "asr" else "bleu")]["n_samples"].sum())
        assert _counted == _info.n_samples, f"{_info.path}: cells cover {_counted} samples, log has {_info.n_samples}"

    checks = pd.DataFrame(_rows)
    worst = float(checks["abs_diff"].max()) if not checks.empty else 0.0
    assert worst < 1e-9, f"Recomputed metrics disagree with the log headers by up to {worst}:\n{checks}"
    checks.sort_values("abs_diff", ascending=False).head(10)
    return (checks, worst)


@app.cell
def _(checks, mo, worst):
    mo.md(
        f"✅ **{len(checks)}** corpus metrics match their log header exactly "
        f"(largest absolute difference `{worst:.2e}`), and every log's samples are "
        "accounted for exactly once across its cells."
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## The two spreadsheet tables

        Column names are `<metric>/<corpus>/<language-or-direction>`, flattened to a
        single header row so they survive a paste into Google Sheets.

        * `wer/OVERALL/all` pools every ASR cell — summed from the counts, so it is
          the corpus rate over the whole set, not an average of the cells.
        * `bleu/MACRO-AVG` is the unweighted mean over ST directions. It summarises
          the six corpus scores beside it; it is not itself a corpus score, since
          BLEU's brevity penalty is defined over a single corpus.
        """
    )
    return


@app.cell
def _(Path, campaign, frame, out_dir):
    written = {}
    _dir = Path(out_dir.value)
    _dir.mkdir(parents=True, exist_ok=True)

    asr = campaign.asr_table(frame)
    st = campaign.st_table(frame)
    for _name, _table in (("asr_results.csv", asr), ("st_results.csv", st), ("metrics_long.csv", frame)):
        _path = _dir / _name
        _table.to_csv(_path, index=False)
        written[_name] = _path
    return asr, st, written


@app.cell
def _(asr):
    asr
    return


@app.cell
def _(st):
    st
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Charts

        Colour is the breakdown (language, or direction) on a warm plum→gold ramp;
        marker *shape* is the training run and the enlarged marker is the
        end-of-training checkpoint. Runs are what this campaign accumulates over
        time, so they get the cue that stays readable once several are overlaid.
        """
    )
    return


@app.cell
def _(charts, frame, out_dir):
    figures = charts.save_all(frame, out_dir.value)
    return (figures,)


@app.cell
def _(figures):
    figures["headline_by_checkpoint"]
    return


@app.cell
def _(figures):
    figures["asr_wer_by_checkpoint"]
    return


@app.cell
def _(figures):
    figures["st_bleu_by_checkpoint"]
    return


@app.cell
def _(figures):
    figures["asr_final_wer_heatmap"]
    return


@app.cell
def _(mo, figures, out_dir, written):
    mo.md(
        "### Written\n\n"
        + "\n".join(f"- `{p}`" for p in written.values())
        + "\n"
        + "\n".join(f"- `{out_dir.value}/{n}.png` (and `.pdf`)" for n in figures)
    )
    return


if __name__ == "__main__":
    app.run()
