"""MetricX-24 rescoring of finished speech-translation eval logs.

``melteval st-metrics`` gives lexical scores (BLEU, spBLEU, chrF++) from a log; this adds the neural
MetricX-24 metric from the same log, offline. It follows the repo's rule for neural MT metrics
(see ``melteval/rescore.py``): it runs from its own environment (``$VENVS/metricx``: torch + transformers
4.40, no ``inspect_ai``) and is never imported by ``melteval``. It therefore reads JSON logs with the standard
library; a ``.eval`` log is read through ``inspect_ai`` when that is importable.

MetricX-24 (Juraska et al., WMT 2024) is a regression model: it reads one string and returns an *error
score* in [0, 25], lower is better. The hybrid checkpoint serves both modes from the same weights:

* reference-based: ``source: {src} candidate: {hyp} reference: {ref}``  -> ``metricx_ref`` (primary);
* reference-free (QE): ``source: {src} candidate: {hyp}``              -> ``metricx_qe`` (secondary).

The input string, the 1536-token cap, the ``google/mt5-xl`` tokenizer, the removal of the final EOS token and
the read-out (one decoder step from token 0, logit 250089 = ``<extra_id_10>``, clamped to [0, 25]) are ported
from ``metricx24/predict.py`` and ``metricx24/models.py`` of ``google-research/metricx`` at the commit pinned
in :data:`UPSTREAM_COMMIT`. The checkpoint's ``MT5ForRegression`` head is an ``MT5ForConditionalGeneration``
whose output is read at one position, so the stock class is used and the regression class is not vendored.

Source, hypothesis and reference are read from the log itself: ``metadata.source_text`` (the transcript in
the source language, *not* the audio), ``output.completion`` and ``target``. A sample without a source
transcript (Irish in the FLEURS X->en sets has none) is scored reference-based with an empty source, which
the upstream README allows, and gets no QE score: reference-free scoring without a source is undefined.

Casing: the backbones write (almost) no capital letters while the references are cased, and MetricX is
case-sensitive. ``--casing both`` (default) scores every sample twice, as written and with source,
hypothesis and reference lowercased, so a case habit is not read as a translation error; the lowercased
columns carry the ``_lc`` suffix and are the ones comparable to the lowercased chrF++ in the existing tables.

Each sample is scored independently of its batch: inputs are sorted by length for speed, padded with an
attention mask, and scattered back to their original order, so the score does not depend on who shares the
batch beyond bf16 numerical noise (checked by the tests).

Usage, from the ``metricx`` venv on a GPU node::

    python projects/metricx/score.py --out-root OUT \\
        --run llama1bBaseT=path/to/log.json --run qwen35_2bBaseT=path/to/other.json

Outputs go to ``OUT/<name>/<dataset>/{scores.jsonl,summary.json,summary.csv}``; a run whose
``summary.json`` exists is skipped (``--force`` redoes it), so a rerun only scores what is new.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path


logger = logging.getLogger(__name__)

#: Checkpoint and tokenizer, pinned. ``google/metricx-24-hybrid-xxl-v2p6-bfloat16`` is the bf16 export of the
#: hybrid XXL model; its tokenizer is ``google/mt5-xl`` (upstream README), not a file in the model repo.
MODEL_ID = "google/metricx-24-hybrid-xxl-v2p6-bfloat16"
MODEL_REVISION = "3d9c6f7fd11b58d3eceeb68ee39fe7d2c761ee20"
TOKENIZER_ID = "google/mt5-xl"
TOKENIZER_REVISION = "63fc6450d80515b48e026b69ef2fbbd426433e84"
#: ``google-research/metricx`` commit the input format and read-out were ported from.
UPSTREAM_COMMIT = "fc4978eb064670f7cc33e93ea4f52d38396b8ae6"

#: Upstream's ``--max_input_length`` for MetricX-24; the README warns against raising it.
MAX_INPUT_LENGTH = 1536
#: ``<extra_id_10>``: the logit read as the score.
SCORE_TOKEN_ID = 250089
SCORE_MIN, SCORE_MAX = 0.0, 25.0

#: Language tiers of the existing ST tables (hours of transcribed audio per language in the MELT pools; see
#: ``projects/self-distill/eval/tables/eu24-dev-tables.md`` in the training repo). English is neither a source
#: nor a target there, so the tiers cover the 23 non-English languages.
TIERS = {
    "high": ["es", "fr", "de", "it", "nl", "pt"],
    "mid": ["pl", "hu", "fi", "cs", "ro", "sv"],
    "low": ["sk", "bg", "lv", "da", "lt", "el", "et", "hr", "sl", "ga", "mt"],
}

CASINGS = ("as-is", "lower", "both")
#: Per-sample columns, as written to ``scores.jsonl`` and ``summary.csv``.
METRIC_COLUMNS = ("metricx_ref", "metricx_qe", "metricx_ref_lc", "metricx_qe_lc")


@dataclasses.dataclass
class Segment:
    """One sample of an ST log, reduced to what MetricX reads."""

    id: str
    src_lang: str
    tgt_lang: str
    source: str
    hypothesis: str
    reference: str
    has_source: bool


def build_input(source: str, hypothesis: str, reference: str | None = None) -> str:
    """Return the MetricX-24 input string; reference-free when *reference* is ``None``."""
    text = f"source: {source} candidate: {hypothesis}"
    return text if reference is None else f"{text} reference: {reference}"


def pad_batch(batch: list[list[int]], pad_id: int = 0) -> tuple[list[list[int]], list[list[int]]]:
    """Right-pad token id lists to a common width; return ``(input_ids, attention_mask)``."""
    width = max(len(ids) for ids in batch)
    input_ids = [ids + [pad_id] * (width - len(ids)) for ids in batch]
    mask = [[1] * len(ids) + [0] * (width - len(ids)) for ids in batch]
    return input_ids, mask


def score_token_ids(
    token_ids: list[list[int]],
    predict: Callable[[list[list[int]]], list[float]],
    batch_size: int = 32,
    max_batch_tokens: int = 32768,
) -> list[float]:
    """Score token id lists through *predict*, longest first, and return the scores in input order.

    *predict* takes a batch of unpadded id lists and returns one score each. Sorting by length keeps padding
    small; the scatter back by index makes the output order independent of that sort. A batch holds at most
    *batch_size* inputs and *max_batch_tokens* padded tokens.
    """
    order = sorted(range(len(token_ids)), key=lambda i: (-len(token_ids[i]), i))
    scores = [0.0] * len(token_ids)
    start = 0
    while start < len(order):
        width = max(len(token_ids[order[start]]), 1)  # the longest of the chunk: the list is sorted descending
        chunk = order[start : start + max(1, min(batch_size, max_batch_tokens // width))]
        for i, score in zip(chunk, predict([token_ids[i] for i in chunk]), strict=True):
            scores[i] = float(score)
        start += len(chunk)
    return scores


class MetricXScorer:
    """The MetricX-24 hybrid checkpoint on one device: ``score(inputs)`` maps input strings to error scores."""

    def __init__(
        self,
        model_name_or_path: str = MODEL_ID,
        revision: str | None = MODEL_REVISION,
        device: str = "cuda",
        batch_size: int = 32,
    ) -> None:
        import torch
        import transformers

        self._torch = torch
        self.device = device
        self.batch_size = batch_size
        self.n_truncated = 0
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)
        self.model = transformers.MT5ForConditionalGeneration.from_pretrained(
            model_name_or_path, revision=revision, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        self.model.to(device).eval()

    def encode(self, text: str) -> list[int]:
        """Tokenize like upstream: truncate to :data:`MAX_INPUT_LENGTH`, then drop the final EOS token."""
        ids = self.tokenizer(text, max_length=MAX_INPUT_LENGTH, truncation=True, padding=False)["input_ids"]
        self.n_truncated += len(ids) >= MAX_INPUT_LENGTH
        return ids[:-1]

    def _predict(self, batch: list[list[int]]) -> list[float]:
        torch = self._torch
        input_ids, mask = pad_batch(batch, self.model.config.pad_token_id)
        decoder_input_ids = torch.zeros((len(batch), 1), dtype=torch.long, device=self.device)
        with torch.inference_mode():
            logits = self.model(
                input_ids=torch.tensor(input_ids, device=self.device),
                attention_mask=torch.tensor(mask, device=self.device),
                decoder_input_ids=decoder_input_ids,
                use_cache=False,
            ).logits
        # The bf16 logit is read as is, exactly as upstream does, before the cast and the clamp.
        return logits[:, 0, SCORE_TOKEN_ID].float().clamp(SCORE_MIN, SCORE_MAX).tolist()

    def score(self, inputs: list[str]) -> list[float]:
        """Score MetricX input strings; the result is parallel to *inputs*."""
        return score_token_ids([self.encode(text) for text in inputs], self._predict, batch_size=self.batch_size)


# --- reading logs ---------------------------------------------------------------------------------------


def _raw_samples(path: Path) -> tuple[list[dict], dict]:
    """Read a log into plain dicts (``id``, ``target``, ``output.completion``, ``metadata``, ``error``)."""
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as fh:
            log = json.load(fh)
        return log.get("samples") or [], {
            "task": log["eval"].get("task"),
            "dataset": log["eval"].get("dataset", {}).get("name"),
            "status": log.get("status"),
        }
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(path), resolve_attachments=True)
    samples = [
        {
            "id": s.id,
            "target": s.target,
            "output": {"completion": s.output.completion if s.output else ""},
            "metadata": s.metadata or {},
            "error": s.error,
        }
        for s in log.samples or []
    ]
    return samples, {"task": log.eval.task, "dataset": log.eval.dataset.name, "status": log.status}


def read_log(path: str | Path) -> tuple[list[Segment], dict]:
    """Read a finished ST log into :class:`Segment` records and a header.

    *path* is a ``.json`` or ``.eval`` log, or a directory holding exactly one of them. A sample that errored
    during the run is skipped and counted; an empty completion is kept and scored as an empty candidate.

    Returns:
        ``(segments, header)``; the header has ``log``, ``task``, ``dataset``, ``n_samples``, ``n_errors``,
        ``n_empty`` and ``n_source_missing``.

    Raises:
        ValueError: If the log did not finish with status ``success``, a directory holds no single log, a
            sample has several references, a sample has no ``src_lang``/``tgt_lang``, or ids repeat.
    """
    path = Path(path)
    if path.is_dir():
        found = sorted([*path.glob("*.json"), *path.glob("*.eval")])
        if len(found) != 1:
            raise ValueError(f"{path} holds {len(found)} logs; pass the one to score.")
        path = found[0]
    samples, header = _raw_samples(path)
    if header["status"] != "success":
        raise ValueError(f"{path} has status {header['status']!r}, not 'success'.")

    segments: list[Segment] = []
    n_errors = n_empty = 0
    for sample in samples:
        if sample.get("error"):
            n_errors += 1
            continue
        metadata = sample.get("metadata") or {}
        for field in ("src_lang", "tgt_lang"):
            if not metadata.get(field):
                raise ValueError(f"Sample {sample['id']!r} has no {field!r} in its metadata.")
        target = sample["target"]
        if not isinstance(target, str):
            if len(target) != 1:
                raise ValueError(f"Sample {sample['id']!r} has {len(target)} references; expected exactly one.")
            target = target[0]
        hypothesis = (sample.get("output") or {}).get("completion") or ""
        n_empty += not hypothesis.strip()
        source = metadata.get("source_text") or ""
        segments.append(
            Segment(
                id=str(sample["id"]),
                src_lang=metadata["src_lang"],
                tgt_lang=metadata["tgt_lang"],
                source=source,
                hypothesis=hypothesis,
                reference=target,
                has_source=bool(source.strip()),
            )
        )
    if len({s.id for s in segments}) != len(segments):
        raise ValueError(f"{path} has repeated sample ids.")
    header.update(
        log=str(path),
        n_samples=len(samples),
        n_errors=n_errors,
        n_empty=n_empty,
        n_source_missing=sum(not s.has_source for s in segments),
    )
    return segments, header


def language_axis(segments: list[Segment]) -> str:
    """The metadata field the per-language table groups by: ``src_lang`` for X->en, ``tgt_lang`` for en->X.

    Raises:
        ValueError: If neither field is constant (a log mixing several directions has no single axis).
    """
    if len({s.tgt_lang for s in segments}) == 1:
        return "src_lang"
    if len({s.src_lang for s in segments}) == 1:
        return "tgt_lang"
    raise ValueError("Neither src_lang nor tgt_lang is constant; the log mixes directions.")


# --- scoring --------------------------------------------------------------------------------------------


def score_segments(segments: list[Segment], scorer, casing: str = "both") -> list[dict]:
    """Score *segments*; one row per segment with the :data:`METRIC_COLUMNS` that *casing* asks for.

    *scorer* is anything with ``score(list[str]) -> list[float]``. QE is ``None`` for a segment with no source
    text; reference-based scoring uses the empty source instead. All inputs of one casing go through one call,
    so the scorer sorts the whole set by length at once.
    """
    if casing not in CASINGS:
        raise ValueError(f"casing must be one of {CASINGS}, got {casing!r}.")
    rows = [
        {"id": s.id, "src_lang": s.src_lang, "tgt_lang": s.tgt_lang, "source_missing": not s.has_source}
        for s in segments
    ]
    for suffix, fold in (("", str), ("_lc", str.lower)):
        if (suffix == "" and casing == "lower") or (suffix == "_lc" and casing == "as-is"):
            continue
        ref_inputs = [build_input(fold(s.source), fold(s.hypothesis), fold(s.reference)) for s in segments]
        with_source = [i for i, s in enumerate(segments) if s.has_source]
        qe_inputs = [build_input(fold(segments[i].source), fold(segments[i].hypothesis)) for i in with_source]
        ref_scores = scorer.score(ref_inputs)
        qe_scores = dict(zip(with_source, scorer.score(qe_inputs), strict=True))
        for i, row in enumerate(rows):
            row[f"metricx_ref{suffix}"] = ref_scores[i]
            row[f"metricx_qe{suffix}"] = qe_scores.get(i)
    return rows


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict], axis: str) -> dict:
    """Per-language means, the macro-mean over languages and the tier macro-means.

    Per language: the mean of each metric over the samples that have one (``n`` counts all samples, ``n_qe``
    those with a QE score). Macro: the plain mean over languages, as in the existing ST tables.
    ``macro_mean_with_source`` leaves out languages without source transcripts, so reference-based and QE
    macros cover the same languages.
    """
    columns = [c for c in METRIC_COLUMNS if any(c in r for r in rows)]
    by_lang: dict[str, list[dict]] = {}
    for row in rows:
        by_lang.setdefault(row[axis], []).append(row)

    languages = []
    for lang in sorted(by_lang):
        group = by_lang[lang]
        entry: dict = {"lang": lang, "n": len(group), "n_source_missing": sum(r["source_missing"] for r in group)}
        for column in columns:
            entry[column] = _mean([r[column] for r in group if r[column] is not None])
        languages.append(entry)

    def macro(entries: list[dict]) -> dict:
        out: dict = {"n_languages": len(entries)}
        for column in columns:
            out[column] = _mean([e[column] for e in entries if e[column] is not None])
        return out

    complete = [e for e in languages if not e["n_source_missing"]]
    present = {e["lang"]: e for e in languages}
    return {
        "axis": axis,
        "columns": columns,
        "languages": languages,
        "macro_mean": macro(languages),
        "macro_mean_with_source": macro(complete),
        "tiers": {name: macro([present[lang] for lang in langs if lang in present]) for name, langs in TIERS.items()},
    }


def write_outputs(out_dir: Path, rows: list[dict], summary: dict) -> None:
    """Write ``scores.jsonl`` (per sample), ``summary.json`` and ``summary.csv`` (per language and macro)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = summary["columns"]
    with open(out_dir / "scores.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["kind", "group", "n", *columns])
        for e in summary["languages"]:
            writer.writerow(["language", e["lang"], e["n"], *(_cell(e[c]) for c in columns)])
        for name, tier in summary["tiers"].items():
            writer.writerow(["tier", name, tier["n_languages"], *(_cell(tier[c]) for c in columns)])
        for name in ("macro_mean", "macro_mean_with_source"):
            writer.writerow(["macro", name, summary[name]["n_languages"], *(_cell(summary[name][c]) for c in columns)])
    # summary.json goes last: its presence marks a finished run.
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)


def _cell(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def provenance(header: dict, casing: str, scorer_stats: dict) -> dict:
    """What produced a summary: the log, the pinned model and tokenizer, and the input templates."""
    return {
        "log": header["log"],
        "task": header["task"],
        "dataset": header["dataset"],
        "n_samples": header["n_samples"],
        "n_errors_skipped": header["n_errors"],
        "n_empty_candidates": header["n_empty"],
        "n_source_missing": header["n_source_missing"],
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer": TOKENIZER_ID,
        "tokenizer_revision": TOKENIZER_REVISION,
        "upstream_commit": UPSTREAM_COMMIT,
        "max_input_length": MAX_INPUT_LENGTH,
        "template_ref": build_input("{src}", "{hyp}", "{ref}"),
        "template_qe": build_input("{src}", "{hyp}"),
        "casing": casing,
        "score_range": [SCORE_MIN, SCORE_MAX],
        **scorer_stats,
    }


def score_log(
    log_path: str | Path, out_dir: Path, scorer, casing: str = "both", limit: int | None = None, seed: int = 0
) -> dict:
    """Score one log and write its outputs to *out_dir*; return the summary.

    *limit* keeps a seeded random subset of that many samples (smoke runs); ``None`` scores them all.
    """
    segments, header = read_log(log_path)
    if limit is not None and limit < len(segments):
        segments = sorted(random.Random(seed).sample(segments, limit), key=lambda s: s.id)
    axis = language_axis(segments)
    started = time.time()
    n_truncated_before = getattr(scorer, "n_truncated", 0)
    rows = score_segments(segments, scorer, casing)
    summary = summarize(rows, axis)
    stats = {
        "n_scored": len(rows),
        "n_truncated_inputs": getattr(scorer, "n_truncated", 0) - n_truncated_before,
        "seconds": round(time.time() - started, 1),
    }
    summary = {"provenance": provenance(header, casing, stats), **summary}
    write_outputs(out_dir, rows, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Command line: load the model once, then score every ``--run`` log that has no summary yet."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", action="append", required=True, metavar="NAME=LOG", help="model name = log path")
    parser.add_argument("--out-root", required=True, type=Path, help="outputs go to <out-root>/<name>/<dataset>/")
    parser.add_argument("--casing", choices=CASINGS, default="both")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, help="score a seeded random subset of this many samples per log")
    parser.add_argument("--force", action="store_true", help="rescore runs whose summary.json exists")
    parser.add_argument("--model-path", default=MODEL_ID, help="checkpoint dir or hub id (default: the pinned one)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    todo = []
    for spec in args.run:
        name, _, log = spec.partition("=")
        if not name or not log:
            parser.error(f"--run expects NAME=LOG, got {spec!r}")
        _, header = read_log(log)
        out_dir = args.out_root / name / header["dataset"]
        if (out_dir / "summary.json").exists() and not args.force:
            logger.info("skip %s %s: already scored", name, header["dataset"])
            continue
        todo.append((name, log, out_dir))
    if not todo:
        return 0

    revision = MODEL_REVISION if args.model_path == MODEL_ID else None
    scorer = MetricXScorer(args.model_path, revision=revision, batch_size=args.batch_size)
    for name, log, out_dir in todo:
        summary = score_log(log, out_dir, scorer, args.casing, args.limit)
        macro = summary["macro_mean"]
        logger.info(
            "%s %s: n=%d ref=%s qe=%s ref_lc=%s qe_lc=%s (%.0f s)",
            name, summary["provenance"]["dataset"], summary["provenance"]["n_scored"],
            *(_cell(macro.get(c)) or "-" for c in METRIC_COLUMNS), summary["provenance"]["seconds"],
        )  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
