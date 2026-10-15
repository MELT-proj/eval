"""Lexical ST metrics computed from finished eval logs.

``st_scorer`` computes BLEU, spBLEU and chrF++ at the end of a run, pooled and
grouped by ``lang`` -- which on the FLEURS ST sets is the *target* language, so
the 23 sources of an X→en set all land in one ``en`` bucket. Some runs never
scored at all (``--no-score``, which is how a multi-target en→X set has to be
run), and some are scored under a different convention than the one a paper
used (cased vs lowercased, plain chrF vs chrF++). This module recomputes the
numbers from the log itself: the hypothesis is ``sample.output.completion`` and
the reference ``sample.target``, so it reads scored and unscored logs alike.

Per group (a direction, a source or target language, or everything) it reports

* ``bleu`` -- sacrebleu BLEU with the tokenizer :func:`melteval.scorers.corpus_bleu`
  routes to by target language (13a unless listed in ``BLEU_TOKENIZER_BY_LANG``).
  If a group has several target languages no single tokenizer is right, and
  ``bleu`` is ``None`` with the reason in ``bleu_note`` -- never a 13a number
  under a mixed-language group;
* ``spbleu`` -- BLEU over SentencePiece-tokenized text (``flores200`` or
  ``flores101``), language-independent;
* ``chrf`` -- plain chrF (word order 0, signature ``nw:0``);
* ``chrfpp`` -- chrF++ (word order 2, ``nw:2``), the value ``corpus_chrf``
  logs under its historical name.

Samples that errored during the run are skipped and counted. A sample with an
empty completion is *scored*, as an empty hypothesis -- a miss, not a gap --
and counted separately so a model that stopped answering is visible.

Like :mod:`melteval.rescore` this needs no model stack: it imports neither torch,
lhotse nor ``melt``, and inspect_ai and sacrebleu load only when it is called.

Usage::

    melteval st-metrics path/to/log.eval --group-by dataset_id --lowercase
    melteval st-metrics a.eval b.eval --group-by none --json > scores.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from melteval.spm import SPM_TOKENIZERS, require_spm_model


logger = logging.getLogger(__name__)

#: ``--group-by`` choices. ``none`` pools every scored sample into one group.
GROUP_BY = ("dataset_id", "src_lang", "tgt_lang", "none")

#: Metric columns, in display order.
METRICS = ("bleu", "spbleu", "chrf", "chrfpp")
_DISPLAY = {"bleu": "BLEU", "spbleu": "spBLEU", "chrf": "chrF", "chrfpp": "chrF++"}


def _reference(sample) -> str:
    """Return the one reference string of *sample*.

    Raises:
        ValueError: If the sample has more than one reference; ST sets carry
            exactly one, and silently picking the first would hide a log that
            is not what this module assumes.
    """
    target = sample.target
    if isinstance(target, str):
        return target
    target = list(target)
    if len(target) != 1:
        raise ValueError(f"Sample {sample.id!r} has {len(target)} references; expected exactly one.")
    return target[0]


def _tgt_lang(metadata: dict) -> str:
    """Target language of a sample, read the way ``corpus_bleu`` reads it."""
    return metadata.get("tgt_lang") or metadata.get("lang") or ""


def _group_keys(samples: list, group_by: str) -> tuple[list[str], str]:
    """Return one group key per sample, and the ``group_by`` actually applied.

    ``dataset_id`` falls back to ``tgt_lang`` when the *log* carries no
    ``dataset_id`` at all (older logs); a log where only some samples have it is
    an error, since the groups would mix two namespaces.

    Raises:
        ValueError: If *group_by* is unknown or a sample lacks the field.
    """
    if group_by not in GROUP_BY:
        raise ValueError(f"group_by must be one of {GROUP_BY}, got {group_by!r}.")
    if group_by == "none":
        return ["all"] * len(samples), group_by

    effective = group_by
    if group_by == "dataset_id":
        have = sum(1 for s in samples if (s.metadata or {}).get("dataset_id"))
        if have == 0:
            effective = "tgt_lang"
            logger.warning("No sample carries a dataset_id; grouping by tgt_lang instead.")
        elif have != len(samples):
            raise ValueError(f"{len(samples) - have} of {len(samples)} samples have no dataset_id.")

    keys = []
    for sample in samples:
        metadata = sample.metadata or {}
        key = _tgt_lang(metadata) if effective == "tgt_lang" else metadata.get(effective)
        if not key:
            raise ValueError(f"Sample {sample.id!r} has no {effective!r} in its metadata.")
        keys.append(key)
    return keys, effective


def _score_group(
    hypotheses: list[str], references: list[str], tgt_langs: set[str], spm: str, lowercase: bool
) -> tuple[dict, dict[str, str]]:
    """Compute the four metrics for one group; return them with their sacrebleu signatures."""
    import sacrebleu

    from melteval.scorers import BLEU_TOKENIZER_BY_LANG

    refs = [references]
    out: dict = {}
    signatures: dict[str, str] = {}

    langs = {lang.lower() for lang in tgt_langs if lang}
    if len(langs) > 1:
        out["bleu"] = None
        out["bleu_note"] = f"not defined: several target languages {sorted(langs)} have no single BLEU tokenizer"
    else:
        tokenizer = BLEU_TOKENIZER_BY_LANG.get(next(iter(langs), ""), "13a")
        bleu = sacrebleu.BLEU(tokenize=tokenizer, lowercase=lowercase)
        out["bleu"] = bleu.corpus_score(hypotheses, refs).score
        signatures["bleu"] = str(bleu.get_signature())

    spbleu = sacrebleu.BLEU(tokenize=spm, lowercase=lowercase)
    out["spbleu"] = spbleu.corpus_score(hypotheses, refs).score
    signatures["spbleu"] = str(spbleu.get_signature())

    for name, word_order in (("chrf", 0), ("chrfpp", 2)):
        chrf = sacrebleu.CHRF(word_order=word_order, lowercase=lowercase)
        out[name] = chrf.corpus_score(hypotheses, refs).score
        signatures[name] = str(chrf.get_signature())
    return out, signatures


def score_log(
    log_path: str | Path, group_by: str = "dataset_id", lowercase: bool = False, spm: str = "flores200"
) -> dict:
    """Score one finished ``.eval`` log with BLEU, spBLEU, chrF and chrF++.

    Args:
        log_path: A log written by ``inspect eval``, scored or not.
        group_by: ``dataset_id`` (default; falls back to ``tgt_lang`` for a log
            without it), ``src_lang``, ``tgt_lang``, or ``none`` to pool.
        lowercase: Pass ``lowercase=True`` to every sacrebleu metric (the
            signatures then read ``case:lc``), for models that write lowercase
            against cased references.
        spm: ``flores200`` (default) or ``flores101``, the spBLEU tokenizer.

    Returns:
        ``{"log", "task", "group_by", "lowercase", "spm", "n_samples", "n_scored",
        "n_errors", "n_empty", "signatures", "groups"}``. ``signatures`` maps each
        metric to the distinct sacrebleu signatures used (one, except ``bleu``
        when targets need different tokenizers). ``groups`` is a list of
        ``{"group", "n", "n_empty", "bleu", "spbleu", "chrf", "chrfpp"}`` sorted
        by group, plus ``bleu_note`` where ``bleu`` is ``None``.

    Raises:
        SPMModelError: If the SentencePiece model is not staged (checked before
            the log is read; nothing is downloaded).
        ValueError: On a multi-epoch log, a sample with several references, or
            missing group metadata.
    """
    from inspect_ai.log import read_eval_log

    if spm not in SPM_TOKENIZERS:
        raise ValueError(f"spm must be one of {SPM_TOKENIZERS}, got {spm!r}.")
    require_spm_model(spm)

    log = read_eval_log(str(log_path), resolve_attachments=True)
    samples = log.samples or []
    if any(s.epoch != 1 for s in samples):
        raise ValueError(f"{log_path} has several epochs; this command scores single-epoch logs only.")

    usable = [s for s in samples if getattr(s, "error", None) is None]
    n_errors = len(samples) - len(usable)
    keys, effective = _group_keys(usable, group_by)

    buckets: dict[str, dict] = {}
    for sample, key in zip(usable, keys, strict=True):
        hypothesis = sample.output.completion if sample.output else ""
        bucket = buckets.setdefault(key, {"hyp": [], "ref": [], "langs": set(), "empty": 0})
        bucket["hyp"].append(hypothesis)
        bucket["ref"].append(_reference(sample))
        bucket["langs"].add(_tgt_lang(sample.metadata or {}))
        bucket["empty"] += not hypothesis.strip()

    groups = []
    signatures: dict[str, list[str]] = {}
    for key in sorted(buckets):
        bucket = buckets[key]
        values, sigs = _score_group(bucket["hyp"], bucket["ref"], bucket["langs"], spm, lowercase)
        groups.append({"group": key, "n": len(bucket["hyp"]), "n_empty": bucket["empty"], **values})
        for metric, sig in sigs.items():
            if sig not in signatures.setdefault(metric, []):
                signatures[metric].append(sig)

    return {
        "log": str(log_path),
        "task": log.eval.task,
        "group_by": effective,
        "lowercase": lowercase,
        "spm": spm,
        "n_samples": len(samples),
        "n_scored": len(usable),
        "n_errors": n_errors,
        "n_empty": sum(g["n_empty"] for g in groups),
        "signatures": signatures,
        "groups": groups,
    }


def score_logs(
    log_paths: list[str | Path], group_by: str = "dataset_id", lowercase: bool = False, spm: str = "flores200"
) -> list[dict]:
    """:func:`score_log` over several logs, in order; one result per log."""
    return [score_log(path, group_by=group_by, lowercase=lowercase, spm=spm) for path in log_paths]


def format_table(result: dict) -> str:
    """Render one :func:`score_log` result as a readable plain-text table."""
    lines = [
        f"{result['log']}",
        f"  task={result['task']}  group_by={result['group_by']}  "
        f"{'lowercased' if result['lowercase'] else 'cased'}  spm={result['spm']}",
        f"  samples={result['n_samples']}  scored={result['n_scored']}  errors skipped={result['n_errors']}  "
        f"empty completions (scored as misses)={result['n_empty']}",
    ]
    for metric in METRICS:
        for signature in result["signatures"].get(metric, []):
            lines.append(f"  {_DISPLAY[metric]:<7} {signature}")

    width = max([len("group"), *(len(g["group"]) for g in result["groups"])])
    header = f"{'group':<{width}} {'n':>6} {'empty':>5} " + " ".join(f"{_DISPLAY[m]:>8}" for m in METRICS)
    lines += ["", header, "-" * len(header)]
    for g in result["groups"]:
        cells = " ".join(f"{g[m]:8.3f}" if g[m] is not None else f"{'n/a':>8}" for m in METRICS)
        lines.append(f"{g['group']:<{width}} {g['n']:>6} {g['n_empty']:>5} {cells}")
        if g.get("bleu_note"):
            lines.append(f"{'':<{width}}   BLEU {g['bleu_note']}")
    return "\n".join(lines)


def to_json(results: list[dict]) -> str:
    """Serialise :func:`score_logs` output as indented JSON."""
    return json.dumps(results, indent=2, ensure_ascii=False)
