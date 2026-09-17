"""MCIF's own metrics (WER, COMET, BERTScore), as a post-hoc `inspect score` step.

``melteval/readers/mcif.py``'s own scorers (``chunked_asr``/``chunked_st``/
``audio_chat``) are fast, dependency-light approximations, deliberately: they
run inline during generation, in the same process as the model under test, and
nothing in that path should load a second neural model next to it (see
``melteval/rescore.py`` for the same argument about COMET/MetricX). This
module is the other half -- the exact metrics the MCIF paper reports, computed
**after** generation, from a finished ``.eval`` log, by calling straight into
the official `mcif <https://github.com/hlt-mt/mcif>`_ package rather than
reimplementing any part of it. It is not registered anywhere in
``melteval.registry`` and never runs as part of ``inspect eval`` -- only via
``inspect score``, and only from wherever ``mcif-bench`` is installed.

Usage, from an environment with ``pip install mcif-bench`` (which pulls in
``jiwer``, ``whisper-normalizer``, ``bert-score`` and ``unbabel-comet``) and
this package installed alongside it (``pip install -e /path/to/melt-eval`` --
none of its own extras are needed, just ``inspect_ai`` and
``huggingface_hub``, both base dependencies)::

    # generate first, wherever the checkpoint lives -- --no-score if that
    # environment cannot also carry mcif-bench's dependencies
    infra/runners/submit_eval.sh artemis /path/to/checkpoint configs/hf/mcif.yaml \\
        -T task_filter=chunked_asr -T dataset_id=mcif-short-fixed-en --no-score

    # then, from the mcif-bench environment
    inspect score path/to/log.eval \\
        --scorer melteval/mcif_scoring.py@mcif_official_asr_scorer

No ``-S`` arguments are needed: every sample already carries which
``(repo, revision, track, lang)`` produced it (see
``melteval.readers.mcif.MCIFReader``), so the reference file this scores
against is resolved straight from the log, the same revision generation used.

**Every member of every reference group must be in the log.** The official
functions reassemble a group's completions themselves, by id, and KeyError on
a missing one -- caught here and re-raised with the missing id named, but not
otherwise recoverable. A partial run (``-T limit=``, a debug slice) cannot be
scored this way; generate the whole ``dataset_id`` first.

**Translation additionally needs the ``mwerSegmenter`` tool** the official
pipeline itself depends on (not a pip package -- see
``mcif.evaluation.MwerSegmenter``'s docstring for where to get it and set
``MWERSEGMENTER_ROOT``) and downloads a COMET checkpoint on first use.
BERTScore downloads a baseline-rescaling file per language on first use.
Neither happens for ASR, which needs only ``jiwer``.
"""

from __future__ import annotations

import gzip

from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, Value, metric, scorer
from inspect_ai.solver import TaskState

from melteval.readers.mcif import download_reference_gz


def _mcif_source(scores: list[SampleScore]) -> tuple[str, str, str, str]:
    """Return the single ``(repo, revision, track, lang)`` every sample must
    agree on -- together, the only thing that decides which reference file
    this run is scored against.

    Raises:
        ValueError: If the samples disagree (score one ``dataset_id`` at a
            time), or lack this metadata entirely -- meaning they were not
            produced by :class:`melteval.readers.mcif.MCIFReader`, so there is
            no reference file for this module to go and fetch.
    """
    keys = {
        (
            str((s.sample_metadata or {}).get("repo", "")),
            str((s.sample_metadata or {}).get("revision", "")),
            str((s.sample_metadata or {}).get("track", "")),
            str((s.sample_metadata or {}).get("lang", "")),
        )
        for s in scores
    }
    if len(keys) > 1:
        raise ValueError(
            f"Mixed MCIF sources in one scoring run: {sorted(keys)}. Score one (track, "
            "language) combination at a time -- e.g. generate with a single -T dataset_id=."
        )
    key = next(iter(keys), ("", "", "", ""))
    if not all(key):
        raise ValueError(
            "Samples have no MCIF source metadata (repo/revision/track/lang). This scorer "
            "only works on a log produced from melteval.readers.mcif.MCIFReader -- e.g. "
            "melteval/tasks.py@chunked_asr against configs/hf/mcif.yaml."
        )
    return key


def _hypo_dict(scores: list[SampleScore]) -> dict[str, str]:
    """``{raw MCIF id: completion}``, as the official scoring functions index
    by the corpus's own id, not melteval's ``sample_key``."""
    return {
        str((s.sample_metadata or {}).get("cut_id", s.sample_id)): (s.score.metadata or {}).get(
            "hypothesis", ""
        )
        for s in scores
    }


def _read_official_reference(repo: str, revision: str, track: str, lang: str):
    """Download the exact reference file that generated *scores* and parse it
    with the official reader, so grouping/qa_type/source-transcript extraction
    all match the paper's own pipeline byte for byte."""
    from mcif.evaluation import read_reference

    ref_gz = download_reference_gz(repo, revision, track, lang)
    with gzip.open(ref_gz, "rb") as fh:
        return read_reference(fh, track, lang, modality="audio")


def _reraise_missing_completion(exc: KeyError, macro_task: str) -> None:
    raise ValueError(
        f"Missing completion for sample id {exc}, needed by a {macro_task} reference group. "
        "Official scoring needs every member of every group present in the log -- a partial "
        "run (`-T limit=`, a debug slice) cannot be scored this way; generate the whole "
        "dataset_id first."
    ) from exc


@metric
def official_asr_wer() -> Metric:
    """Corpus WER via ``mcif.evaluation.score_asr`` -- the paper's own metric:
    ``jiwer.wer`` after the exact normalizer (Whisper's English one) the
    official pipeline uses, not melteval's own."""

    def calculate(scores: list[SampleScore]) -> Value:
        if not scores:
            return 0.0
        repo, revision, track, lang = _mcif_source(scores)
        ref_dict = _read_official_reference(repo, revision, track, lang)
        if "ASR" not in ref_dict:
            raise ValueError(
                f"No ASR reference for track={track!r} lang={lang!r}. ASR is only defined "
                "for the English target -- score this log with official_trans_comet instead."
            )
        from mcif.evaluation import score_asr

        try:
            return score_asr(_hypo_dict(scores), ref_dict, lang)
        except KeyError as exc:
            _reraise_missing_completion(exc, "ASR")

    return calculate


@metric
def official_trans_comet() -> Metric:
    """COMET via ``mcif.evaluation.score_st`` -- the paper's own metric, after
    the same ``mwerSegmenter`` resentence-alignment step the official
    evaluation performs. See this module's docstring for the external tool
    and checkpoint download this pulls in."""

    def calculate(scores: list[SampleScore]) -> Value:
        if not scores:
            return 0.0
        repo, revision, track, lang = _mcif_source(scores)
        ref_dict = _read_official_reference(repo, revision, track, lang)
        if "TRANS" not in ref_dict:
            raise ValueError(
                f"No TRANS reference for track={track!r} lang={lang!r}. TRANS is only "
                "defined for non-English targets -- score an English log with "
                "official_asr_wer instead."
            )
        from mcif.evaluation import score_st

        try:
            return score_st(_hypo_dict(scores), ref_dict, lang)
        except KeyError as exc:
            _reraise_missing_completion(exc, "TRANS")

    return calculate


@metric
def official_bertscore(macro_task: str) -> Metric:
    """BERTScore via ``mcif.evaluation``, for ``macro_task`` ``"QA"`` or
    ``"SUM"`` -- the paper's own metric for both."""

    def calculate(scores: list[SampleScore]) -> Value:
        if not scores:
            return 0.0
        repo, revision, track, lang = _mcif_source(scores)
        ref_dict = _read_official_reference(repo, revision, track, lang)
        if macro_task not in ref_dict:
            raise ValueError(
                f"No {macro_task} reference for track={track!r} lang={lang!r}. "
                f"Available: {sorted(ref_dict)}."
            )
        try:
            if macro_task == "QA":
                from mcif.evaluation import score_sqa

                score, _ = score_sqa(_hypo_dict(scores), ref_dict, lang, breakdown_qa_types=False)
            else:
                from mcif.evaluation import score_ssum

                score = score_ssum(_hypo_dict(scores), ref_dict, lang)
        except KeyError as exc:
            _reraise_missing_completion(exc, macro_task)
        return score

    return calculate


async def _stash_hypothesis(state: TaskState, target: Target) -> Score:
    """Shared body for every scorer in this module: the actual metric is
    computed once, over the whole run, by re-downloading the official
    reference and calling straight into ``mcif.evaluation`` -- nothing
    meaningful is computed per sample here."""
    hypothesis = state.output.completion
    return Score(value={}, answer=hypothesis, metadata={"hypothesis": hypothesis})


@scorer(metrics=[official_asr_wer()])
def mcif_official_asr_scorer() -> Scorer:
    """Reproduces the MCIF paper's ASR metric. See this module's docstring."""
    return _stash_hypothesis


@scorer(metrics=[official_trans_comet()])
def mcif_official_trans_scorer() -> Scorer:
    """Reproduces the MCIF paper's translation metric. See this module's docstring."""
    return _stash_hypothesis


@scorer(metrics=[official_bertscore("QA")])
def mcif_official_qa_scorer() -> Scorer:
    """Reproduces the MCIF paper's QA metric. See this module's docstring."""
    return _stash_hypothesis


@scorer(metrics=[official_bertscore("SUM")])
def mcif_official_sum_scorer() -> Scorer:
    """Reproduces the MCIF paper's summarisation metric. See this module's docstring."""
    return _stash_hypothesis
