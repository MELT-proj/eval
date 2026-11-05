# Copyright 2026 FBK
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""
Bar plots and tables for the SpeechEval results of a SMURF checkpoint against baselines.

Reads ``aggregated.xlsx`` (``projects/smurf4eu/aggregate.py``), keeps one
checkpoint (reported under a codename) plus the baselines, drops non-EU
languages (e.g. zh), and writes PDF + PNG charts and CSV tables with a focus
on multilingual EU performance:

* ASR: VoxPopuli (WER, lower is better)
* ST X->EN and EN->X: FLEURS (COMET-22 DA x100, higher is better)
* English-only extras: LibriSpeech clean/other (WER) and MCIF chunked ST (COMET-22 DA, official pipeline)

The per-language tabs of the workbook are the source: ``voxpopuli``,
``fleurs-st-xen``, ``fleurs-st-enx``, ``librispeech`` and ``mcif``.

Every chart is produced in three variants:

* ``all``: all models, each scored on the languages it supports; the summary also reports the
  mean over the languages common to all models.
* ``full_support``: only the models that support every language of the benchmark.
* ``common``: all models, restricted to the languages supported by all of them.

Qwen models support only a subset of languages on the speech side (source languages), so their
ASR and X->EN results are restricted to those languages; EN->X is reported in full.
Whisper large-v3 is a single model: ASR results come from transcribe mode, X->EN results from translate mode
(which only supports translation into English).
"""

import argparse
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl


mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch


OURS = "ours"
CHECKPOINT_MODEL = "eurollm-16n-run1-step-0207000"
OURS_CODENAME = "EuroSpeech-9B-26-v0.1"
EU_LANGS = frozenset(
    [
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fi",
        "fr",
        "ga",
        "hr",
        "hu",
        "it",
        "lt",
        "lv",
        "mt",
        "nl",
        "pl",
        "pt",
        "ro",
        "sk",
        "sl",
        "sv",
    ],
)
QWEN_SPEECH_LANGS = frozenset({"en", "de", "ru", "it", "fr", "es", "pt", "nl"})
VARIANTS = ("all", "full_support", "common")

MISSING = "–"  # noqa: RUF001
# The "translate" results come from the same Whisper-large-v3 model (run in translation mode).
RESULT_ALIASES = {"whisper-large-v3-translate": "whisper-large-v3"}
ASR, ST_XEN, ST_ENX = "asr_voxpopuli", "st_xen", "st_enx"
MCIF = "st_mcif"  # summary panel only: MCIF chunked ST (English audio into DE / IT)
TASK_TITLES = {
    ASR: "ASR - VoxPopuli (WER %, lower is better)",
    ST_XEN: "ST X→EN - FLEURS (COMET-22, higher is better)",
    ST_ENX: "ST EN→X - FLEURS (COMET-22, higher is better)",
    MCIF: "ST EN→X - MCIF chunked (COMET-22, higher is better)",
}
TASK_YLABELS = {ASR: "WER (%)", ST_XEN: "COMET-22", ST_ENX: "COMET-22", MCIF: "COMET-22"}
VARIANT_TITLES = {
    "all": "all models, each on its supported languages",
    "full_support": "only models supporting all languages",
    "common": "all models, on commonly supported languages",
}


@dataclass(frozen=True)
class ModelSpec:
    """
    Display and support information for one model.

    Attributes:
        name: Codename used in the charts.
        color: Hex color of the model.
        tasks: Tasks the model can perform.
        speech_langs: Supported source (speech) languages, or None if unrestricted.

    """

    name: str
    color: str
    tasks: frozenset[str]
    speech_langs: frozenset[str] | None = None


# Insertion order is the plotting order: fully open models (Apertus, ours) first, then open-weights ones.
# Ours is indigo; baselines use distinct hues so that no baseline resembles our color.
MODELS: dict[str, ModelSpec] = {
    # Its model card lists no audio languages, so none is excluded: a weak language shows as a low score.
    "apertus-1.5-8b": ModelSpec("Apertus-1.5-8B", "#e87ba4", frozenset({ASR, ST_XEN, ST_ENX})),
    OURS: ModelSpec(OURS_CODENAME, "#5B4CF5", frozenset({ASR, ST_XEN, ST_ENX})),
    "qwen2-audio-7b-instruct": ModelSpec(
        "Qwen2-Audio-7B",
        "#eb6834",
        frozenset({ASR, ST_XEN, ST_ENX}),
        QWEN_SPEECH_LANGS,
    ),
    "qwen3-omni-30b-a3b-instruct": ModelSpec(
        "Qwen3-Omni-30B-A3B",
        "#1baf7a",
        frozenset({ASR, ST_XEN, ST_ENX}),
        QWEN_SPEECH_LANGS,
    ),
    "whisper-large-v3": ModelSpec("Whisper-large-v3", "#eda100", frozenset({ASR, ST_XEN})),
    "gemma-4-e4b-it": ModelSpec("Gemma-4-E4B-IT", "#b07aa1", frozenset({ASR, ST_XEN, ST_ENX})),
    "phi-4-multimodal-instruct": ModelSpec(
        "Phi-4-multimodal",
        "#4e79a7",
        frozenset({ASR, ST_XEN, ST_ENX}),
        frozenset({"en", "de", "es", "fr", "it", "pt"}),
    ),
    "canary-1b-v2": ModelSpec(
        "Canary-1B-v2",
        "#9c755f",
        frozenset({ASR, ST_XEN, ST_ENX}),
        frozenset(EU_LANGS - {"ga"}),
    ),
}

# English-only benchmarks: label -> (sheet, row filter, metric, scale, ylabel).
ENGLISH_BENCHMARKS = {
    "LibriSpeech clean": ("librispeech", {"lang": "clean"}, "wer", 100.0, "WER (%)"),
    "LibriSpeech other": ("librispeech", {"lang": "other"}, "wer", 100.0, "WER (%)"),
}
# MCIF chunked ST: label -> same tuple. COMET-22 DA of MCIF's official pipeline (mwerSegmenter re-segmentation).
MCIF_BENCHMARKS = {
    "EN→DE": ("mcif", {"task": "st", "lang": "de"}, "comet", 100.0, "COMET-22"),
    "EN→IT": ("mcif", {"task": "st", "lang": "it"}, "comet", 100.0, "COMET-22"),
}


def model_key(model: str) -> str | None:
    """
    Map a model name of the aggregated workbook to a key of ``MODELS``.

    Args:
        model: Value of the ``model`` column.

    Returns:
        str | None: The key in ``MODELS``, or None if the row must be discarded.

    """
    if model == CHECKPOINT_MODEL:
        return OURS
    model = RESULT_ALIASES.get(model, model)
    return model if model in MODELS else None


def _tab(book: pd.ExcelFile, name: str, columns: list[str]) -> pd.DataFrame:
    """Read one per-dataset tab, or an empty frame with the expected columns."""
    if name not in book.sheet_names:
        return pd.DataFrame(columns=columns)
    return pd.read_excel(book, sheet_name=name)


def _with_key(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the plotting key and drop rows for models the charts do not show."""
    frame = frame.copy()
    frame["model"] = frame["model"].map(model_key)
    return frame[frame["model"].notna()]


def load_results(workbook: Path) -> pd.DataFrame:
    """
    Load the per-language tabs and return one tidy table for the multilingual tasks.

    Non-EU languages are dropped, missing/``pending`` values are dropped, and each model
    is restricted to the tasks and speech-side languages it supports.

    Args:
        workbook: Path to ``aggregated.xlsx``.

    Returns:
        pd.DataFrame: Columns ``task``, ``model``, ``lang``, ``value``.

    """
    book = pd.ExcelFile(workbook)
    asr = _tab(book, "voxpopuli", ["model", "lang", "wer"])
    st_xen = _tab(book, "fleurs-st-xen", ["model", "lang", "comet"])
    st_enx = _tab(book, "fleurs-st-enx", ["model", "lang", "comet"])
    parts = []
    for task, frame, metric, scale in (
        (ASR, asr, "wer", 100.0),
        (ST_XEN, st_xen, "comet", 100.0),
        (ST_ENX, st_enx, "comet", 100.0),
    ):
        if frame.empty:
            continue
        rows = _with_key(frame[["model", "lang", metric]].rename(columns={metric: "value"}))
        rows["value"] = pd.to_numeric(rows["value"], errors="coerce") * scale
        rows.insert(0, "task", task)
        parts.append(rows)
    tidy = pd.concat(parts, ignore_index=True)
    tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce")
    tidy = tidy[tidy["value"].notna() & tidy["lang"].isin(EU_LANGS)]
    keep = []
    for task, model, lang in zip(tidy["task"], tidy["model"], tidy["lang"], strict=True):
        spec = MODELS[model]
        speech_restricted = spec.speech_langs is not None and task in (ASR, ST_XEN)
        keep.append(task in spec.tasks and not (speech_restricted and lang not in spec.speech_langs))
    return tidy[keep].reset_index(drop=True)


def load_english_benchmarks(workbook: Path, benchmarks: dict = ENGLISH_BENCHMARKS) -> pd.DataFrame:
    """
    Load single-slice benchmarks (LibriSpeech by default, or ``MCIF_BENCHMARKS``).

    Args:
        workbook: Path to ``aggregated.xlsx``.
        benchmarks: Label -> (sheet, row filter, metric, scale, metric label).

    Returns:
        pd.DataFrame: Columns ``benchmark``, ``model``, ``value``, ``ylabel``.

    """
    book = pd.ExcelFile(workbook)
    out = []
    for label, (sheet, filters, metric, scale, ylabel) in benchmarks.items():
        frame = _tab(book, sheet, ["model", "lang", "task", metric])
        if frame.empty or metric not in frame:
            continue
        rows = frame
        for column, wanted in filters.items():
            rows = rows[rows[column] == wanted]
        rows = _with_key(rows[["model", metric]])
        rows["value"] = pd.to_numeric(rows[metric], errors="coerce")
        out.extend(
            {"benchmark": label, "model": key, "value": value * scale, "ylabel": ylabel}
            for key, value in zip(rows["model"], rows["value"], strict=True)
            if pd.notna(value)
        )
    if not out:
        msg = f"No results found for the benchmarks {sorted(benchmarks)} and the selected models."
        raise ValueError(msg)
    return pd.DataFrame(out, columns=["benchmark", "model", "value", "ylabel"])


def all_langs(tidy: pd.DataFrame, task: str) -> list[str]:
    """
    Languages of a task (union over models), English first for ASR, then alphabetical.

    Args:
        tidy: Output of ``load_results``.
        task: Task name.

    Returns:
        list[str]: Sorted language codes.

    """
    langs = sorted(tidy.loc[tidy["task"] == task, "lang"].unique())
    return sorted(langs, key=lambda lang: (lang != "en", lang))


def apply_variant(tidy: pd.DataFrame, task: str, variant: str) -> pd.DataFrame:
    """
    Restrict the rows of a task according to a variant.

    Args:
        tidy: Output of ``load_results``.
        task: Task name.
        variant: One of ``VARIANTS``.

    Returns:
        pd.DataFrame: Rows of ``task`` kept by the variant.

    """
    rows = tidy[tidy["task"] == task]
    langs = set(all_langs(tidy, task))
    by_model = {m: set(g["lang"]) for m, g in rows.groupby("model")}
    if variant == "full_support":
        full = {m for m, ls in by_model.items() if ls >= langs}
        return rows[rows["model"].isin(full)]
    if variant == "common":
        common = set.intersection(*by_model.values()) if by_model else set()
        return rows[rows["lang"].isin(common)]
    return rows


def group_scores(tidy: pd.DataFrame, task: str, variant: str) -> pd.DataFrame:
    """
    Aggregate scores (English / Multilingual / Multilingual common) per model.

    "Multilingual" is the mean over the non-English languages. "Multilingual (common langs)" is the
    mean over the non-English languages supported by every model present in the task.

    Args:
        tidy: Output of ``load_results``.
        task: Task name.
        variant: One of ``VARIANTS``.

    Returns:
        pd.DataFrame: Index = model key, columns = group labels. ``attrs['n']`` holds
        {(model, group): number of languages} and ``attrs['common']`` the sorted non-English
        languages supported by all models.

    """
    rows = apply_variant(tidy, task, variant)
    common = set.intersection(*[set(g["lang"]) for _, g in rows.groupby("model")]) if len(rows) else set()
    groups = {"English": rows["lang"] == "en"}
    multi = rows["lang"] != "en"
    if variant == "common":
        groups["Multilingual (common langs)"] = multi
    else:
        groups["Multilingual"] = multi
        if variant == "all":
            groups["Multilingual (common langs)"] = multi & rows["lang"].isin(common)
    table = {}
    counts = {}
    for name, mask in groups.items():
        sub = rows[mask]
        if sub.empty or (name == "English" and task != ASR):
            continue
        table[name] = sub.groupby("model")["value"].mean()
        for m, n in sub.groupby("model")["lang"].nunique().items():
            counts[m, name] = n
    out = pd.DataFrame(table)
    out.attrs["n"] = counts
    out.attrs["common"] = sorted(common - {"en"})
    return out.reindex([m for m in MODELS if m in out.index])


def style_axes(ax: Axes) -> None:
    """
    Apply a light, recessive style to an axes.

    Args:
        ax: Axes to style.

    """
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#b9b8b2")
    ax.tick_params(axis="both", length=0, colors="#52514e")
    ax.yaxis.grid(visible=True, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def _legend(fig: Figure, models: list[str], *, y: float = 0.0) -> None:
    """
    Add a figure-level legend for the given models.

    Args:
        fig: Figure to annotate.
        models: Model keys to list.
        y: Vertical anchor of the legend in figure coordinates.

    """
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODELS[m].color) for m in models]
    fig.legend(
        handles,
        [MODELS[m].name for m in models],
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=len(models),
        frameon=False,
        fontsize=10,
    )


def _save(fig: Figure, out_dir: Path, stem: str) -> None:
    """
    Save a figure as PDF and PNG.

    Args:
        fig: Figure to save.
        out_dir: Output directory.
        stem: File name without extension.

    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _bars(ax: Axes, values: pd.DataFrame, *, label_values: bool, slot: float) -> None:
    """
    Draw grouped bars (x = columns of ``values``, one bar per model row).

    All bars have the same width (``slot`` x-units per bar, 0.9 of it filled), so that panels
    drawn with the same x-unit length look alike whatever their number of models.

    Args:
        ax: Target axes.
        values: Rows = model keys, columns = x categories.
        label_values: Whether to print the value on top of each bar.
        slot: Width in x-units reserved to each bar (a group has width 1).

    """
    n = max(len(values), 1)
    x = np.arange(len(values.columns))
    for i, (model, row) in enumerate(values.iterrows()):
        pos = x + (i - (n - 1) / 2) * slot
        bars = ax.bar(pos, row.to_numpy(dtype=float), slot * 0.9, color=MODELS[model].color)
        if label_values:
            for rect, val in zip(bars, row, strict=True):
                if pd.notna(val):
                    ax.annotate(
                        f"{val:.1f}",
                        (rect.get_x() + rect.get_width() / 2, val),
                        ha="center",
                        va="bottom",
                        xytext=(0, 2),
                        textcoords="offset points",
                        fontsize=8,
                        color="#52514e",
                    )
    ax.set_xticks(x, list(values.columns))
    ax.set_xlim(-0.5, len(values.columns) - 0.5)
    ax.set_ylim(0, float(np.nanmax(values.to_numpy(dtype=float))) * 1.12)


def _round_tops(ax: Axes, radius_pt: float = 3.0) -> None:
    """
    Replace the bars of an axes with bars whose upper corners are slightly rounded.

    The axes geometry must be final (call after the layout is fixed). The radius is circular in
    display space and the bottom edge stays flat on the baseline.

    Args:
        ax: Axes whose bar rectangles are replaced.
        radius_pt: Corner radius in points.

    """
    fig = ax.figure
    pos = ax.get_position()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    sx = (x1 - x0) / (pos.width * fig.get_figwidth() * 72)  # x-units per point
    sy = (y1 - y0) / (pos.height * fig.get_figheight() * 72)  # y-units per point
    clip = plt.Rectangle((x0, 0), x1 - x0, y1, transform=ax.transData)
    for rect in list(ax.patches):
        height = rect.get_height()
        if not np.isfinite(height) or height <= 0:
            rect.remove()
            continue
        ry = radius_pt * sy
        patch = FancyBboxPatch(
            (rect.get_x(), -ry),
            rect.get_width(),
            height + ry,
            boxstyle=f"round,pad=0,rounding_size={radius_pt * sx}",
            mutation_aspect=sy / sx,
            facecolor=rect.get_facecolor(),
            edgecolor="none",
        )
        ax.add_patch(patch)
        patch.set_clip_path(clip)
        rect.remove()


SUMMARY_GROUP_IN = 2.1  # inches per bar group (same for every panel, so bars have the same width)


def group_note(scores: pd.DataFrame) -> str:
    """
    List the languages the "Multilingual (common langs)" group of a panel averages over.

    Args:
        scores: Output of ``group_scores`` for one task.

    Returns:
        str: Wrapped note ("" when the panel has no common-langs group).

    """
    if "Multilingual (common langs)" not in scores.columns:
        return ""
    langs = ", ".join(lang.upper() for lang in scores.attrs["common"])
    n_common = len(scores.attrs["common"])
    text = f"Multilingual (common langs): mean over the {n_common} non-English languages every model supports: {langs}"
    return textwrap.fill(text, width=38)


MCIF_NOTE = "MCIF chunked ST (short, fixed segmentation), English audio."


def mcif_scores(mcif: pd.DataFrame) -> pd.DataFrame:
    """
    Shape the MCIF chunked ST results as a summary panel (one bar group per target language).

    Args:
        mcif: Output of ``load_english_benchmarks`` with ``MCIF_BENCHMARKS``.

    Returns:
        pd.DataFrame: Index = model key, columns = target languages. ``attrs['n']`` is 1 everywhere (one slice per
        group) and ``attrs['common']`` is empty.

    """
    wide = mcif.pivot_table(index="model", columns="benchmark", values="value")
    wide = wide[[b for b in MCIF_BENCHMARKS if b in wide.columns]]
    wide = wide.reindex([m for m in MODELS if m in wide.index])
    wide.attrs["n"] = {(m, g): 1 for m in wide.index for g in wide.columns}
    wide.attrs["common"] = []
    return wide


def plot_summary(tidy: pd.DataFrame, mcif: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    Plot the summary (one panel per task) on the languages supported by all models, and return it as a table.

    ASR reports English and the multilingual mean over the common languages; the FLEURS ST panels the multilingual mean
    over the common languages; MCIF chunked ST its EN->DE and EN->IT scores. Panels have a fixed width per bar group
    and all bars share the same width.

    Args:
        tidy: Output of ``load_results``.
        mcif: MCIF chunked ST results (``load_english_benchmarks`` with ``MCIF_BENCHMARKS``).
        out_dir: Output directory.

    Returns:
        pd.DataFrame: Long table with columns ``task``, ``group``, ``model``, ``value``, ``n_langs``.

    """
    variant = "common"
    panels = {t: group_scores(tidy, t, variant) for t in (ASR, ST_XEN, ST_ENX)}
    panels[MCIF] = mcif_scores(mcif)
    panels = {t: g for t, g in panels.items() if not g.empty}
    slot = 0.8 / max(len(g) for g in panels.values())
    notes = {t: textwrap.fill(MCIF_NOTE, width=38) if t == MCIF else group_note(g) for t, g in panels.items()}
    left, gap, right, axes_h, top = 0.85, 0.95, 0.2, 2.5, 1.1
    bottom = 1.05 + 0.16 * max(note.count("\n") + 1 for note in notes.values())
    fig_w = left + right + sum(SUMMARY_GROUP_IN * len(g.columns) for g in panels.values()) + gap * (len(panels) - 1)
    fig_h = bottom + axes_h + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    records = []
    x_in = left
    for task, scores in panels.items():
        w_in = SUMMARY_GROUP_IN * len(scores.columns)
        ax = fig.add_axes((x_in / fig_w, bottom / fig_h, w_in / fig_w, axes_h / fig_h))
        x_in += w_in + gap
        _bars(ax, scores, label_values=True, slot=slot)
        style_axes(ax)
        title, _, hint = TASK_TITLES[task].partition(" (")
        ax.set_title(f"{title}\n({hint}", fontsize=11, loc="left", color="#1a1a19")
        ax.set_ylabel(TASK_YLABELS[task], color="#52514e")
        ax.tick_params(axis="x", labelsize=9)
        for group in scores.columns:
            for model, val in scores[group].dropna().items():
                records.append(
                    {
                        "task": task,
                        "group": group,
                        "model": MODELS[model].name,
                        "value": val,
                        "n_langs": scores.attrs["n"][model, group],
                    }
                )
        ax.set_xticklabels([t.get_text().replace(" (", "\n(") for t in ax.get_xticklabels()])
        ax.text(0, -0.26, notes[task], transform=ax.transAxes, va="top", fontsize=8.5, color="#52514e")
        _round_tops(ax)
    present = [m for m in MODELS if any(m in s.index for s in panels.values())]
    fig.suptitle(f"{OURS_CODENAME} vs. baselines - {VARIANT_TITLES[variant]}", fontsize=13, x=0.01, y=0.97, ha="left")
    _legend(fig, present, y=0.02)
    _save(fig, out_dir, "summary")
    return pd.DataFrame(records)


# Models left out of a heatmap (not of its table) so that the shading separates the remaining ones.
HEATMAP_EXCLUDE: dict[str, tuple[str, ...]] = {
    ST_XEN: ("qwen2-audio-7b-instruct",),
    ST_ENX: ("qwen2-audio-7b-instruct",),
}


def _column_goodness(wide: pd.DataFrame, task: str) -> pd.DataFrame:
    """
    Normalize each language column to [0, 1], 1 being the best model (lower WER, higher BLEU).

    Args:
        wide: Rows = models, columns = languages.
        task: Task name (WER is lower-is-better, BLEU higher-is-better).

    Returns:
        pd.DataFrame: Same shape as ``wide``; columns with a single value get 0.5.

    """
    lo, hi = wide.min(), wide.max()
    span = (hi - lo).replace(0, np.nan)
    scaled = (wide - lo) / span
    scaled = scaled.fillna(0.5).where(wide.notna())
    return 1 - scaled if task == ASR else scaled


def plot_per_language(tidy: pd.DataFrame, task: str, variant: str, out_dir: Path) -> pd.DataFrame:
    """
    Plot a model x language heatmap with the scores written in the cells.

    Cell color is normalized per language (darkest = best model on that language), the best
    value of each column is in bold and missing results are shown as a dash. Models in
    ``HEATMAP_EXCLUDE`` for the task are not drawn but stay in the returned table.

    Args:
        tidy: Output of ``load_results``.
        task: Task name.
        variant: One of ``VARIANTS``.
        out_dir: Output directory.

    Returns:
        pd.DataFrame: Wide table, rows = model codenames, columns = languages.

    """
    rows = apply_variant(tidy, task, variant)
    if rows.empty:
        return pd.DataFrame()
    wide = rows.pivot_table(index="model", columns="lang", values="value")
    wide = wide.reindex([m for m in MODELS if m in wide.index])
    wide = wide[[lang for lang in all_langs(tidy, task) if lang in wide.columns]]
    full = wide
    wide = wide.drop(index=list(HEATMAP_EXCLUDE.get(task, ())), errors="ignore")
    good = _column_goodness(wide, task)
    best = wide.min() if task == ASR else wide.max()
    cmap = mpl.colors.LinearSegmentedColormap.from_list("slate", ["#eef0f4", "#8e9bb3", "#2f3b55"])
    n_rows, n_cols = wide.shape
    fig, ax = plt.subplots(figsize=(min(16.0, 0.62 * n_cols + 3.2), 0.6 * n_rows + 1.8))
    for i, model in enumerate(wide.index):
        for j, lang in enumerate(wide.columns):
            val, g = wide.loc[model, lang], good.loc[model, lang]
            color = "#f6f5f2" if pd.isna(g) else cmap(g)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white", linewidth=2))
            if pd.isna(val):
                text, weight, ink = MISSING, "normal", "#9a9992"
            else:
                text = f"{val:.1f}"
                weight = "bold" if np.isclose(val, best[lang]) else "normal"
                ink = "white" if g > 0.5 else "#1a1a19"  # noqa: PLR2004
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=9, color=ink, fontweight=weight)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_xticks(np.arange(n_cols) + 0.5, [lang.upper() for lang in wide.columns])
    ax.set_yticks(np.arange(n_rows) + 0.5, [MODELS[m].name for m in wide.index])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, colors="#1a1a19")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(f"{TASK_TITLES[task]}\n{VARIANT_TITLES[variant]}", fontsize=11, loc="left", color="#1a1a19", pad=28)
    fig.text(
        0.01,
        0.0,
        "Cell shade is relative within each language column (darkest = best model); bold = best; dash = not available.",
        fontsize=8.5,
        color="#52514e",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, out_dir / variant, f"heatmap_{task}")
    full.index = [MODELS[m].name for m in full.index]
    return full


def plot_english(english: pd.DataFrame, out_dir: Path) -> None:
    """
    Plot the English-only ASR benchmarks (LibriSpeech clean/other).

    Args:
        english: Output of ``load_english_benchmarks``.
        out_dir: Output directory.

    """
    n_models = english["model"].nunique()
    fig, ax = plt.subplots(figsize=(6, 4.4))
    wide = english.pivot_table(index="model", columns="benchmark", values="value")
    wide = wide.reindex([m for m in MODELS if m in wide.index])
    wide = wide[[b for b in ENGLISH_BENCHMARKS if b in wide.columns]]
    _bars(ax, wide, label_values=True, slot=0.8 / n_models)
    style_axes(ax)
    ax.set_ylabel("WER (%)", color="#52514e")
    ax.set_title("English ASR (lower is better)", fontsize=11, loc="left", color="#1a1a19")
    present = [m for m in MODELS if m in set(english["model"])]
    fig.suptitle(f"{OURS_CODENAME} - English benchmarks", fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _round_tops(ax)
    _legend(fig, present, y=0.0)
    _save(fig, out_dir, "english_benchmarks")


def _write_markdown(wide: pd.DataFrame, path: Path) -> None:
    """
    Write a wide model x language table as a Markdown table (values rounded to 1 decimal).

    Args:
        wide: Rows = model names, columns = languages.
        path: Output file.

    """
    header = "| Model | " + " | ".join(lang.upper() for lang in wide.columns) + " |"
    lines = [header, "|---|" + "---:|" * len(wide.columns)]
    lines.extend(
        f"| {name} | " + " | ".join(MISSING if pd.isna(v) else f"{v:.1f}" for v in row) + " |"
        for name, row in wide.iterrows()
    )
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    """
    Generate all charts and tables.

    Args:
        argv: Command line arguments (defaults to ``sys.argv``).

    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--aggregate", type=Path, required=True, help="Path to aggregated.xlsx.")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("projects/smurf4eu/plotting/output"), help="Output directory."
    )
    args = parser.parse_args(argv)

    tidy = load_results(args.aggregate)
    english = load_english_benchmarks(args.aggregate)
    mcif = load_english_benchmarks(args.aggregate, MCIF_BENCHMARKS)
    tables = args.out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    plot_summary(tidy, mcif, args.out_dir).round(3).to_csv(tables / "summary.csv", index=False)
    for variant in VARIANTS:
        for task in (ASR, ST_XEN, ST_ENX):
            wide = plot_per_language(tidy, task, variant, args.out_dir)
            if not wide.empty:
                wide.round(2).to_csv(tables / f"per_language_{task}_{variant}.csv")
                _write_markdown(wide, tables / f"per_language_{task}_{variant}.md")
    plot_english(english, args.out_dir)
    english.assign(model=english["model"].map(lambda m: MODELS[m].name)).round(3).to_csv(
        tables / "english_benchmarks.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
