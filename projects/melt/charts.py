"""Charts for a campaign frame, in an autumn-sunset palette.

Every function takes the long-format frame :mod:`campaign` produces and returns
a matplotlib figure, so they work the same in a marimo cell, a script, or a
REPL.

Two conventions run through all of them, because the campaign has two axes that
grow independently and both have to stay readable:

* **hue is the thing being broken down** (language, or translation direction) —
  the sunset ramp runs plum → brick → amber → gold, warm throughout.
* **marker shape is the training run**, and marker size marks the end of
  training. Runs are what gets added over time (a second seed, a second model),
  and colour is already spent on language, so shape is what is left. Keeping
  shape on the run also means a chart of one run and a chart of four are read
  the same way.

Colour alone is never the only cue for a category: the palette is deliberately
narrow in hue (that is what makes it a sunset rather than a rainbow), so
languages are also separable by marker and by position on the facet grid.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure


#: Warm categorical ramp: deep plum through brick and ember to late gold.
#: Ordered so that adjacent entries stay distinguishable at line width, and
#: long enough for the ten languages the ASR campaign covers.
SUNSET: list[str] = [
    "#5B1E3B",  # plum
    "#8C2F39",  # oxblood
    "#C0453C",  # brick
    "#E2673A",  # ember
    "#F0913F",  # amber
    "#F6BE55",  # gold
    "#B8863C",  # ochre
    "#7E4A2E",  # umber
    "#A9524E",  # clay
    "#D9A566",  # sand
]

#: Continuous version of the same ramp, pale sky → dark plum, for heatmaps.
#: Dark reads as "worse" for an error rate, which is why WER heatmaps use it
#: as-is and score-like metrics (BLEU, chrF) use the reversed form.
SUNSET_CMAP = LinearSegmentedColormap.from_list(
    "sunset", ["#FDF3DE", "#F6BE55", "#F0913F", "#C0453C", "#5B1E3B"]
)
SUNSET_CMAP_R = SUNSET_CMAP.reversed()

#: Marker shapes cycled per training run.
RUN_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

#: Point size for an intermediate checkpoint vs. the end of training.
MARKER_SIZE = 7
FINAL_MARKER_SIZE = 15


def set_style() -> None:
    """Apply the report's look to matplotlib's defaults. Call once."""
    sns.set_theme(style="whitegrid", context="notebook")
    mpl.rcParams.update(
        {
            "figure.facecolor": "#FFFDF8",
            "axes.facecolor": "#FFFDF8",
            "axes.edgecolor": "#6D5B4B",
            "axes.labelcolor": "#3A2C22",
            "axes.titleweight": "semibold",
            "axes.titlecolor": "#3A2C22",
            "grid.color": "#E7DAC7",
            "grid.linewidth": 0.7,
            "text.color": "#3A2C22",
            "xtick.color": "#6D5B4B",
            "ytick.color": "#6D5B4B",
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
        }
    )


def palette_for(levels) -> dict:
    """Map each level to a sunset colour, wrapping if there are more than ten."""
    return {level: SUNSET[i % len(SUNSET)] for i, level in enumerate(levels)}


def markers_for(levels) -> dict:
    """Map each level (a training run) to a marker shape."""
    return {level: RUN_MARKERS[i % len(RUN_MARKERS)] for i, level in enumerate(levels)}


def _mark_final(ax, data: pd.DataFrame, x: str, y: str, hue: str, palette: dict) -> None:
    """Overdraw the end-of-training point at a larger size.

    ``final`` is the checkpoint anyone actually ships, and its step is an
    estimate rather than a recorded number (see ``campaign._place_final``), so
    it earns a visual marker of its own instead of blending into the line.
    """
    final = data[data["is_final"]]
    if final.empty:
        return
    for level, group in final.groupby(hue, observed=True):
        ax.scatter(
            group[x],
            group[y],
            s=FINAL_MARKER_SIZE**2,
            facecolor=palette.get(level, "#3A2C22"),
            edgecolor="#FFFDF8",
            linewidth=1.2,
            zorder=5,
        )


def _tidy_legend(grid: sns.FacetGrid, title: str | None = None) -> None:
    if grid.legend is not None and title:
        grid.legend.set_title(title)


# =============================================================================
# ASR
# =============================================================================


def asr_by_checkpoint(frame: pd.DataFrame, metric: str = "wer", col_wrap: int = 3) -> Figure:
    """WER (or CER) against training step, one panel per corpus, hue by language.

    The pooled ``OVERALL`` cell is left out: it sits on a different scale from
    the per-corpus panels (it mixes ten languages, several of which the model
    was never trained on) and would flatten every real curve next to it. It has
    its own chart in :func:`headline_by_checkpoint`.
    """
    data = frame[(frame["task"] == "asr") & (frame["metric"] == metric) & (frame["corpus"] != "OVERALL")]
    if data.empty:
        raise ValueError(f"No ASR rows with metric={metric!r}.")

    langs = sorted(data["lang"].unique())
    runs = sorted(data["run_name"].unique())
    palette = palette_for(langs)

    grid = sns.relplot(
        data=data,
        kind="line",
        x="step",
        y="value",
        hue="lang",
        hue_order=langs,
        palette=palette,
        style="run_name" if len(runs) > 1 else None,
        style_order=runs if len(runs) > 1 else None,
        # With one run there is no style dimension for seaborn to hang markers
        # on, so `markers=True` would silently draw none; pass the shape
        # directly instead, since a marker per checkpoint is the point.
        markers=markers_for(runs) if len(runs) > 1 else None,
        marker=None if len(runs) > 1 else RUN_MARKERS[0],
        dashes=len(runs) > 1,
        col="corpus",
        col_wrap=min(col_wrap, data["corpus"].nunique()),
        height=3.4,
        aspect=1.35,
        linewidth=1.8,
        markersize=MARKER_SIZE,
        facet_kws={"sharey": False},
    )
    for corpus, ax in grid.axes_dict.items():
        _mark_final(ax, data[data["corpus"] == corpus], "step", "value", "lang", palette)
    grid.set_titles("{col_name}")
    grid.set_axis_labels("training step", metric.upper())
    _tidy_legend(grid, "language")
    grid.figure.suptitle(f"ASR {metric.upper()} by checkpoint", y=1.02, fontsize=13, weight="semibold")
    return grid.figure


def asr_final_heatmap(frame: pd.DataFrame, metric: str = "wer") -> Figure:
    """Corpus × language grid of the final checkpoint's error rate, one row per run."""
    data = frame[
        (frame["task"] == "asr")
        & (frame["metric"] == metric)
        & (frame["is_final"])
        & (frame["corpus"] != "OVERALL")
    ]
    if data.empty:
        raise ValueError(f"No final-checkpoint ASR rows with metric={metric!r}.")

    runs = sorted(data["run_name"].unique())
    fig, axes = plt.subplots(
        len(runs), 1, figsize=(1.0 * data["lang"].nunique() + 4, 2.2 * len(runs) + 1.2), squeeze=False
    )
    for ax, run in zip(axes.flat, runs):
        pivot = (
            data[data["run_name"] == run]
            .pivot_table(index="corpus", columns="lang", values="value")
            .sort_index()
        )
        sns.heatmap(
            pivot,
            ax=ax,
            cmap=SUNSET_CMAP,
            annot=True,
            fmt=".2f",
            linewidths=0.6,
            linecolor="#FFFDF8",
            cbar_kws={"label": metric.upper()},
            # Anchored at 0 so colour means the same thing in every run's row;
            # 1.0 (every reference word wrong) is the point past which a cell
            # is not "worse", it is a language the model cannot do at all.
            vmin=0.0,
            vmax=max(1.0, float(pivot.max().max())),
        )
        ax.set_title(f"{run} — final checkpoint", loc="left")
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.tight_layout()
    return fig


# =============================================================================
# ST
# =============================================================================


def st_by_checkpoint(frame: pd.DataFrame, metric: str = "bleu") -> Figure:
    """BLEU (or chrF++) against training step, hue by translation direction."""
    data = frame[(frame["task"] == "st") & (frame["metric"] == metric)]
    if data.empty:
        raise ValueError(f"No ST rows with metric={metric!r}.")

    directions = sorted(data["direction"].unique())
    runs = sorted(data["run_name"].unique())
    palette = palette_for(directions)

    grid = sns.relplot(
        data=data,
        kind="line",
        x="step",
        y="value",
        hue="direction",
        hue_order=directions,
        palette=palette,
        style="run_name" if len(runs) > 1 else None,
        style_order=runs if len(runs) > 1 else None,
        # With one run there is no style dimension for seaborn to hang markers
        # on, so `markers=True` would silently draw none; pass the shape
        # directly instead, since a marker per checkpoint is the point.
        markers=markers_for(runs) if len(runs) > 1 else None,
        marker=None if len(runs) > 1 else RUN_MARKERS[0],
        dashes=len(runs) > 1,
        col="corpus" if data["corpus"].nunique() > 1 else None,
        height=4.0,
        aspect=1.5,
        linewidth=1.8,
        markersize=MARKER_SIZE,
    )
    for ax in grid.axes.flat:
        _mark_final(ax, data, "step", "value", "direction", palette)
    grid.set_axis_labels("training step", "chrF++" if metric == "chrf" else metric.upper())
    _tidy_legend(grid, "direction")
    label = "chrF++" if metric == "chrf" else metric.upper()
    grid.figure.suptitle(f"ST {label} by checkpoint", y=1.03, fontsize=13, weight="semibold")
    return grid.figure


# =============================================================================
# Cross-run headline
# =============================================================================


def headline_by_checkpoint(frame: pd.DataFrame) -> Figure:
    """One panel per task, hue by run: the chart to read when comparing runs.

    ASR is the pooled corpus WER over the whole set; ST is the unweighted mean
    BLEU over the directions, which is a summary of six corpus scores and not
    itself a corpus score (BLEU's brevity penalty is defined over one corpus).
    """
    asr = frame[(frame["task"] == "asr") & (frame["metric"] == "wer") & (frame["corpus"] == "OVERALL")]
    st = (
        frame[(frame["task"] == "st") & (frame["metric"] == "bleu")]
        .groupby(["run_name", "model", "step", "is_final"], as_index=False)["value"]
        .mean()
    )

    runs = sorted(set(asr["run_name"]) | set(st["run_name"]))
    palette = palette_for(runs)
    markers = markers_for(runs)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for ax, (data, label, better) in zip(
        axes, [(asr, "ASR — pooled WER", "lower is better"), (st, "ST — mean BLEU", "higher is better")]
    ):
        if data.empty:
            ax.set_visible(False)
            continue
        for run, group in data.groupby("run_name"):
            group = group.sort_values("step")
            ax.plot(
                group["step"],
                group["value"],
                color=palette[run],
                marker=markers[run],
                markersize=MARKER_SIZE,
                linewidth=2.0,
                label=run,
            )
            final = group[group["is_final"]]
            ax.scatter(
                final["step"],
                final["value"],
                s=FINAL_MARKER_SIZE**2,
                marker=markers[run],
                facecolor=palette[run],
                edgecolor="#FFFDF8",
                linewidth=1.4,
                zorder=5,
            )
        ax.set_title(f"{label}  ({better})", loc="left")
        ax.set_xlabel("training step")
        ax.set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Campaign headline: one point per checkpoint", fontsize=13, weight="semibold")
    fig.tight_layout()
    return fig


# =============================================================================
# Writing them out
# =============================================================================


def save_all(frame: pd.DataFrame, out_dir, formats=("png", "pdf")) -> dict[str, Figure]:
    """Render every chart into *out_dir* and return them by name."""
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    figures: dict[str, Figure] = {"headline_by_checkpoint": headline_by_checkpoint(frame)}
    if not frame[frame["task"] == "asr"].empty:
        figures["asr_wer_by_checkpoint"] = asr_by_checkpoint(frame, "wer")
        figures["asr_cer_by_checkpoint"] = asr_by_checkpoint(frame, "cer")
        figures["asr_final_wer_heatmap"] = asr_final_heatmap(frame, "wer")
    if not frame[frame["task"] == "st"].empty:
        figures["st_bleu_by_checkpoint"] = st_by_checkpoint(frame, "bleu")
        figures["st_chrf_by_checkpoint"] = st_by_checkpoint(frame, "chrf")

    for name, figure in figures.items():
        for suffix in formats:
            figure.savefig(out_dir / f"{name}.{suffix}")
    return figures
