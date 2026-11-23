"""Extract (src, mt, ref) triples from a completed eval log for rescoring.

(For COMET / MetricX-24 written straight into the log instead, see :func:`comet_scorer` / :func:`metricx_scorer`
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
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, metric, scorer  # type: ignore
from inspect_ai.solver import TaskState  # type: ignore

from melteval.manifest import EvalRecord, read_manifest, resolve_frozen_set


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
    from inspect_ai.log import read_eval_log  # type: ignore

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
        import torch  # type: ignore
        from comet import download_model, load_from_checkpoint  # type: ignore

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


# --------------------------------------------------------------------------- metricx_scorer
#
# MetricX-24 (hybrid XXL), reference-based, as a post-hoc scorer like :func:`comet_scorer`:
#
#     inspect score path/to/log.json --scorer melteval/rescore.py@metricx_scorer --action append
#
# Run from the metrics venv (torch + transformers + sentencepiece), never as part of `inspect eval`. The input
# format, the 1536-token cap, the ``google/mt5-xl`` tokenizer, the dropped final EOS and the read-out (logit
# 250089 = ``<extra_id_10>`` after one decoder step from token 0, clamped to [0, 25]) follow upstream
# (``google-research/metricx`` at commit fc4978e, ``metricx24/predict.py`` and ``models.py``). The checkpoint's
# ``MT5ForRegression`` head is an ``MT5ForConditionalGeneration`` read at one position, so the stock class loads it.

#: bf16 export of ``google/metricx-24-hybrid-xxl-v2p6`` (same model, weights cast to bfloat16, which is the
#: precision it is run in), and the tokenizer upstream names for it. Pinned.
METRICX_MODEL = "google/metricx-24-hybrid-xxl-v2p6-bfloat16"
METRICX_REVISION = "3d9c6f7fd11b58d3eceeb68ee39fe7d2c761ee20"
METRICX_TOKENIZER = "google/mt5-xl"
METRICX_TOKENIZER_REVISION = "63fc6450d80515b48e026b69ef2fbbd426433e84"
METRICX_MAX_INPUT_LENGTH = 1536
METRICX_SCORE_TOKEN_ID = 250089

#: Environment variable naming a frozen set whose ``source_text`` should fill in for samples that have
#: none. ``inspect score`` drops ``-S`` when it rebuilds a log's scorers, so the backfill sets this
#: instead of passing ``source_manifest=`` to the scorer (docs/speechval.md).
SOURCE_MANIFEST_ENV = "MELTEVAL_SOURCE_MANIFEST"


@dataclass(frozen=True)
class SourceManifest:
    """A frozen set's records, indexed by ``(dataset_id, sample_key)``.

    The composite key is required: ``sample_key`` is ``<source_index>-<ordinal>`` and repeats across
    frozen sets (``00-000000`` is a different sentence in FLEURS X->en and en->X), so a key-only
    lookup would score every sample against the wrong language's transcript and report a number.
    """

    path: Path
    records: dict[tuple[str, str], EvalRecord]


def load_source_manifest(path: str | Path) -> SourceManifest:
    """Read a frozen set (its directory or manifest) into a MetricX source lookup."""
    manifest = resolve_frozen_set(path)
    return SourceManifest(
        path=manifest,
        records={(record.dataset_id, record.sample_key): record for record in read_manifest(manifest)},
    )


def _resolve_source(state: TaskState, manifest: SourceManifest | None) -> tuple[str, str, bool]:
    """The source text for *state*: ``(text, origin, has_source)``."""
    metadata = state.metadata or {}
    source = metadata.get("source_text")
    if source:
        return str(source), "sample", True
    if manifest is None:
        return "", "none", False
    key = (str(metadata.get("dataset_id") or ""), str(state.sample_id))
    record = manifest.records.get(key)
    if record is None:
        return "", "manifest-miss", False
    if (record.src_lang, record.tgt_lang) != (metadata.get("src_lang"), metadata.get("tgt_lang")):
        raise ValueError(
            f"source manifest {manifest.path} has {key} as src_lang={record.src_lang!r}, "
            f"tgt_lang={record.tgt_lang!r}, but the sample is src_lang={metadata.get('src_lang')!r}, "
            f"tgt_lang={metadata.get('tgt_lang')!r}: wrong frozen set for this log?"
        )
    text = record.source_text or ""
    return text, "manifest", bool(text)


def metricx_input(source: str, hypothesis: str, reference: str) -> str:
    """The reference-based MetricX-24 input string."""
    return f"source: {source} candidate: {hypothesis} reference: {reference}"


#: The tokenizer, model, device and pad id, loaded once per process on first use. ``inspect score`` calls
#: the scorer for many samples, and loading 13B parameters per sample would dominate everything else.
_MODEL_STATE: tuple | None = None
#: Serialises model use. Re-entrant so :func:`_load_metricx_model` can be called under it.
_METRICX_LOCK = threading.RLock()


def _load_metricx_model() -> tuple:
    """Return ``(tokenizer, model, device, pad)``, loading the checkpoint on first use."""
    global _MODEL_STATE
    if _MODEL_STATE is None:
        import torch  # type: ignore
        import transformers  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            METRICX_TOKENIZER, revision=METRICX_TOKENIZER_REVISION
        )
        model = transformers.MT5ForConditionalGeneration.from_pretrained(
            METRICX_MODEL, revision=METRICX_REVISION, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
        model.to(device).eval()
        _MODEL_STATE = (tokenizer, model, device, model.config.pad_token_id)
    return _MODEL_STATE


def metricx_predict(inputs: list[str], batch_size: int = 16) -> list[float]:
    """MetricX-24 error scores in [0, 25] (lower is better), parallel to *inputs*; the model is loaded once."""
    import torch  # type: ignore

    with _METRICX_LOCK:
        tokenizer, model, device, pad = _load_metricx_model()
        ids = [
            tokenizer(t, max_length=METRICX_MAX_INPUT_LENGTH, truncation=True, padding=False)["input_ids"][:-1]
            for t in inputs
        ]
        scores = [0.0] * len(ids)
        # By length: little padding, order restored below.
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            width = max(len(ids[i]) for i in chunk)
            input_ids = torch.tensor([ids[i] + [pad] * (width - len(ids[i])) for i in chunk], device=device)
            mask = torch.tensor([[1] * len(ids[i]) + [0] * (width - len(ids[i])) for i in chunk], device=device)
            with torch.inference_mode():
                logits = model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    decoder_input_ids=torch.zeros((len(chunk), 1), dtype=torch.long, device=device),
                    use_cache=False,
                ).logits
            for i, value in zip(chunk, logits[:, 0, METRICX_SCORE_TOKEN_ID].float().clamp(0.0, 25.0).tolist()):
                scores[i] = value
        return scores


def _metricx_triple(sample_score: SampleScore) -> dict | None:
    """The ``{src, mt, ref}`` stashed by :func:`metricx_scorer`, or ``None`` if the sample had no output."""
    return (sample_score.score.metadata or {}).get("metricx_triple")


#: ``Score.value`` for a sample with nothing to score. Out of MetricX's [0, 25] range so it cannot be
#: mistaken for a real score. Not ``nan``: inspect reads a root-level float NaN as its "unscored" sentinel
#: and drops the sample from every metric, which would take unscorable samples out of the coverage
#: denominators (and report a canary slice with one empty completion as 100% covered).
UNSCORED = -1.0


def _scored(value) -> bool:
    """Whether *value* is a real MetricX score (a number in [0, 25])."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 25.0


@metric
def metricx() -> Metric:
    """System-level MetricX-24 error: the mean of the per-sample scores (0-25, lower is better).

    Unscorable samples (empty completion) carry :data:`UNSCORED` and are left out, exactly as the old
    batched metric left out samples without a triple.
    """

    def calculate(scores: list[SampleScore]) -> float:
        values = [float(s.score.value) for s in scores if _scored(s.score.value)]
        return sum(values) / len(values) if values else float("nan")

    return calculate


@metric
def metricx_coverage() -> Metric:
    """Fraction of samples MetricX could score (the rest had no output)."""

    def calculate(scores: list[SampleScore]) -> float:
        return sum(_metricx_triple(s) is not None for s in scores) / len(scores) if scores else 0.0

    return calculate


@metric
def metricx_source_coverage() -> Metric:
    """Fraction of samples whose MetricX input had a non-empty source.

    Counted over all samples, including ones with no completion: an empty output is
    :func:`metricx_coverage`'s concern, a missing source is this one's.
    """

    def calculate(scores: list[SampleScore]) -> float:
        if not scores:
            return 0.0
        return sum(bool((s.score.metadata or {}).get("metricx_source_ok")) for s in scores) / len(scores)

    return calculate


@scorer(metrics=[metricx(), metricx_coverage(), metricx_source_coverage()])
def metricx_scorer(source_manifest: str | None = None) -> Scorer:
    """MetricX-24 hybrid XXL, reference-based, scored per sample from source, completion and target.

    The source is the sample's ``source_text`` when it has one. ``source_manifest`` — or, failing
    that, ``$MELTEVAL_SOURCE_MANIFEST`` — names a frozen set to look it up in, keyed by
    ``(dataset_id, sample_key)``; that is how logs frozen before the source field existed are scored
    with full triples. The per-sample score lands in ``Score.value`` (:data:`UNSCORED` when there is no
    output) and the triple stays in ``Score.metadata``, so :func:`metricx` can take the mean without a
    second model pass.
    """
    manifest_path = source_manifest or os.environ.get(SOURCE_MANIFEST_ENV)
    manifest = load_source_manifest(manifest_path) if manifest_path else None

    async def score(state: TaskState, target: Target) -> Score:
        source, origin, has_source = _resolve_source(state, manifest)
        hypothesis = state.output.completion if state.output else ""
        usable = bool(hypothesis and hypothesis.strip())
        triple = None
        value = UNSCORED
        if usable:
            triple = {"src": source, "mt": hypothesis, "ref": target.text}
            value = metricx_predict([metricx_input(source, hypothesis, target.text)])[0]
        return Score(
            value=value,
            answer=hypothesis,
            metadata={
                "metricx_triple": triple,
                "metricx_source": origin,
                "metricx_source_ok": has_source,
                "metricx_manifest": str(manifest.path) if manifest else None,
            },
        )

    return score
