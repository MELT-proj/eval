"""Extract (src, mt, ref) triples from a completed eval log for rescoring.

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
    from inspect_ai.log import read_eval_log

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
