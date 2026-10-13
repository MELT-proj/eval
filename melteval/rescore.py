"""Extract (src, mt, ref) triples from a completed eval log for rescoring.

(For COMET written straight into the log instead, see :func:`comet_scorer`
at the end of this module.)

Neural MT metrics (COMET, MetricX) live in their own venvs — `comet` and
`metricx` on artemis — because they need their own pinned torch/transformers
stack, and that stack has no business coexisting with the model being
evaluated. So this module does not score anything itself. It reads a finished
``.eval`` log, pulls out ``{sample_id, src, mt, ref}`` per ST sample, and
writes a JSONL sidecar next to nothing in particular — the scoring step is a
separate command, run from the other venv, documented below rather than
wired in as a subprocess call across environments neither should know about.

Usage::

    melteval rescore path/to/log.eval -o path/to/triples.jsonl

Then, from the ``comet`` venv::

    import json
    from comet import download_model, load_from_checkpoint

    samples = [json.loads(line) for line in open("triples.jsonl")]
    model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    output = model.predict(samples, batch_size=8, gpus=1)
    print(output.system_score)

``model.predict`` takes exactly this triple's keys (``src``, ``mt``, ``ref``);
the extra ``sample_id`` key is carried through only for joining scores back to
samples afterwards, not read by COMET itself.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, metric, scorer # type: ignore
from inspect_ai.solver import TaskState # type: ignore


logger = logging.getLogger(__name__)


def extract_triples(log_path: str | Path) -> tuple[list[dict], dict]:
    """Read a completed ``.eval`` log and build rescoring triples.

    Only samples with a ``source_text`` in their metadata are usable — that
    field is the transcript a reference-based MT metric needs as `src`, and it
    is only ever populated for sources whose frozen-set spec set
    ``source_text_field`` (see ``readers/shar.py``). Samples an eval run
    itself failed on are skipped too: a missing completion cannot be scored.

    Args:
        log_path: Path to a ``.eval`` log written by ``inspect eval``.

    Returns:
        ``(triples, stats)``. Each triple is
        ``{"sample_id", "src", "mt", "ref"}``. ``stats`` counts the log's
        total samples against how many were usable and why the rest were not,
        so a rescore run that silently scores zero samples is visible as
        exactly that rather than looking like a clean, empty success.
    """
    from inspect_ai.log import read_eval_log # type: ignore

    log = read_eval_log(str(log_path), resolve_attachments=True)
    samples = log.samples or []

    triples: list[dict] = []
    skipped_error = 0
    skipped_no_source_text = 0
    skipped_no_completion = 0

    for sample in samples:
        if getattr(sample, "error", None) is not None:
            skipped_error += 1
            continue

        source_text = (sample.metadata or {}).get("source_text")
        if not source_text:
            skipped_no_source_text += 1
            continue

        hypothesis = sample.output.completion if sample.output else ""
        if not hypothesis or not hypothesis.strip():
            skipped_no_completion += 1
            continue

        reference = sample.target if isinstance(sample.target, str) else next(iter(sample.target), "")

        triples.append(
            {
                "sample_id": sample.id,
                "src": source_text,
                "mt": hypothesis,
                "ref": reference,
            }
        )

    stats = {
        "total_samples": len(samples),
        "usable": len(triples),
        "skipped_error": skipped_error,
        "skipped_no_source_text": skipped_no_source_text,
        "skipped_no_completion": skipped_no_completion,
    }
    if not triples:
        logger.warning(
            "No rescorable samples in %s (%s). ST sources need `source_text_field` set at "
            "freeze time for this to produce anything.",
            log_path, stats,
        )
    return triples, stats


def write_triples(triples: list[dict], out_path: str | Path) -> None:
    """Write *triples* as JSON lines."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(triple, ensure_ascii=False) + "\n" for triple in triples)


# --------------------------------------------------------------------------- comet_scorer
#
# The same triples, scored by `inspect score` instead of through a JSONL sidecar,
# so COMET lands in the log next to BLEU and the report and watcher pick it up:
#
#     inspect score path/to/log.json --scorer melteval/rescore.py@comet_scorer --action append
#
# Run from the metrics venv (needs unbabel-comet), never as part of `inspect eval`.

#: COMET-22 DA, the ST metric of eval suite v1.
COMET_MODEL = "Unbabel/wmt22-comet-da"


def _triple(sample_score: SampleScore) -> dict | None:
    """The ``{src, mt, ref}`` stashed for one sample, or ``None`` if it could not be scored."""
    return (sample_score.score.metadata or {}).get("comet_triple")


@metric
def comet() -> Metric:
    """System-level COMET (mean of segment scores) over every sample with a source transcript."""

    def calculate(scores: list[SampleScore]) -> float:
        triples = [t for t in map(_triple, scores) if t]
        if not triples:
            return float("nan")
        import torch # type: ignore
        from comet import download_model, load_from_checkpoint # type: ignore

        model = load_from_checkpoint(download_model(COMET_MODEL))
        output = model.predict(triples, batch_size=16, gpus=1 if torch.cuda.is_available() else 0)
        return float(output.system_score)

    return calculate


@metric
def comet_coverage() -> Metric:
    """Fraction of samples COMET could score. Below 1.0: some had no source transcript or no output."""

    def calculate(scores: list[SampleScore]) -> float:
        return sum(_triple(s) is not None for s in scores) / len(scores) if scores else 0.0

    return calculate


@scorer(metrics=[comet(), comet_coverage()])
def comet_scorer() -> Scorer:
    """COMET-22 DA for speech translation, from each sample's ``source_text``, completion and target.

    Nothing is computed per sample: the triple goes into ``Score.metadata``
    (``Score.value`` is reduced as numbers -- see AGENTS.md) and :func:`comet`
    scores the whole run at once, loading the model a single time.
    """

    async def score(state: TaskState, target: Target) -> Score:
        source = (state.metadata or {}).get("source_text")
        hypothesis = state.output.completion if state.output else ""
        usable = bool(source) and bool(hypothesis and hypothesis.strip())
        triple = {"src": source, "mt": hypothesis, "ref": target.text} if usable else None
        return Score(value={}, answer=hypothesis, metadata={"comet_triple": triple})

    return score
