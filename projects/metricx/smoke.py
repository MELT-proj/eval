"""Smoke check of the MetricX port on ~50 real samples (one backbone, one language), on a GPU.

Reports, as JSON on stdout: the score range and spread, a perfect copy of the reference (should be near 0), a
shuffled hypothesis (should be clearly worse), the correlation of reference-based and QE scores, the effect of
batch size and order, and, with ``--upstream``, the agreement with the official ``MT5ForRegression`` and
``predict.get_dataset`` code of a ``google-research/metricx`` checkout (run after our own scoring, on the same
inputs, one sample at a time and unpadded, which is how upstream runs).

Usage (metricx venv, GPU)::

    python projects/metricx/smoke.py LOG --lang de --n 50 [--upstream PATH/TO/metricx]
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

import score as mx  # noqa: E402


def pearson(a: list[float], b: list[float]) -> float:
    """Pearson correlation of two equally long lists."""
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    return cov / (sum((x - ma) ** 2 for x in a) ** 0.5 * sum((y - mb) ** 2 for y in b) ** 0.5)


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman correlation: Pearson on average ranks."""

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return out

    return pearson(ranks(a), ranks(b))


def stats(values: list[float]) -> dict:
    """Min, max, mean, standard deviation and number of distinct values."""
    mean = sum(values) / len(values)
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "std": (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5,
        "distinct": len(set(values)),
    }


def upstream_scores(upstream: Path, inputs: list[dict], qe: bool, model_path: str) -> list[float]:
    """Score *inputs* with upstream's own tokenization and ``MT5ForRegression``, one sample per forward pass."""
    import torch
    import transformers

    sys.path.insert(0, str(upstream))
    from metricx24 import models, predict

    tokenizer = transformers.AutoTokenizer.from_pretrained(mx.TOKENIZER_ID, revision=mx.TOKENIZER_REVISION)
    model = models.MT5ForRegression.from_pretrained(model_path, revision=mx.MODEL_REVISION, torch_dtype="auto")
    model.to("cuda").eval()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "in.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in inputs))
        ds = predict.get_dataset(str(path), tokenizer, mx.MAX_INPUT_LENGTH, torch.device("cuda"), qe)["test"]
        out = []
        with torch.inference_mode():
            for example in ds:
                result = model(
                    input_ids=example["input_ids"].unsqueeze(0), attention_mask=example["attention_mask"].unsqueeze(0)
                )
                out.append(float(result.predictions[0]))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> int:
    """Run the checks and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("log")
    parser.add_argument("--lang", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--upstream", type=Path, help="google-research/metricx checkout to cross-check against")
    parser.add_argument("--model-path", default=mx.MODEL_ID)
    args = parser.parse_args()

    segments, header = mx.read_log(args.log)
    axis = mx.language_axis(segments)
    segments = [s for s in segments if getattr(s, axis) == args.lang and s.has_source][: args.n]
    rng = random.Random(0)
    scorer = mx.MetricXScorer(args.model_path, revision=mx.MODEL_REVISION if args.model_path == mx.MODEL_ID else None)

    def ref_inputs(hyps: list[str]) -> list[str]:
        return [mx.build_input(s.source, h, s.reference) for s, h in zip(segments, hyps, strict=True)]

    def qe_inputs(hyps: list[str]) -> list[str]:
        return [mx.build_input(s.source, h) for s, h in zip(segments, hyps, strict=True)]

    hyps = [s.hypothesis for s in segments]
    shuffled = hyps[:]
    while any(a == b for a, b in zip(shuffled, hyps, strict=True)):
        rng.shuffle(shuffled)
    refs = [s.reference for s in segments]

    real_ref, real_qe = scorer.score(ref_inputs(hyps)), scorer.score(qe_inputs(hyps))
    copy_ref, copy_qe = scorer.score(ref_inputs(refs)), scorer.score(qe_inputs(refs))
    shuf_ref, shuf_qe = scorer.score(ref_inputs(shuffled)), scorer.score(qe_inputs(shuffled))
    lc_ref = scorer.score([t.lower() for t in ref_inputs(hyps)])
    # A copy should score ~0 whatever the source says; if it does not, the source may disagree with the reference.
    copy_nosrc = scorer.score([mx.build_input("", r, r) for r in refs])
    worst = max(range(len(segments)), key=lambda i: copy_ref[i])

    inputs = ref_inputs(hyps)
    scorer.batch_size = 1
    single = scorer.score(inputs)
    scorer.batch_size = 32
    reversed_scores = scorer.score(inputs[::-1])[::-1]
    batch_diff = max(abs(a - b) for a, b in zip(single, real_ref, strict=True))
    order_diff = max(abs(a - b) for a, b in zip(reversed_scores, real_ref, strict=True))

    report = {
        "log": args.log,
        "dataset": header["dataset"],
        "lang": args.lang,
        "n": len(segments),
        "real_ref": stats(real_ref),
        "real_qe": stats(real_qe),
        "real_ref_lowercased": stats(lc_ref),
        "copy_of_reference_ref": stats(copy_ref),
        "copy_of_reference_qe": stats(copy_qe),
        "copy_of_reference_ref_empty_source": stats(copy_nosrc),
        "worst_copy": {
            "score": copy_ref[worst],
            "score_empty_source": copy_nosrc[worst],
            "source": segments[worst].source,
            "reference": segments[worst].reference,
        },
        "shuffled_hypothesis_ref": stats(shuf_ref),
        "shuffled_hypothesis_qe": stats(shuf_qe),
        "shuffled_worse_than_real_in": sum(s > r for s, r in zip(shuf_ref, real_ref, strict=True)),
        "ref_vs_qe": {"pearson": pearson(real_ref, real_qe), "spearman": spearman(real_ref, real_qe)},
        "all_in_range": all(mx.SCORE_MIN <= v <= mx.SCORE_MAX for v in real_ref + real_qe + copy_ref + shuf_ref),
        "batch_1_vs_32_max_abs_diff": batch_diff,
        "reversed_order_max_abs_diff": order_diff,
        "n_truncated_inputs": scorer.n_truncated,
    }

    if args.upstream:
        del scorer
        gc.collect()
        sub = slice(0, 20)
        rows = [{"source": s.source, "hypothesis": s.hypothesis, "reference": s.reference} for s in segments[sub]]
        for mode, ours in (("ref", real_ref[sub]), ("qe", real_qe[sub])):
            theirs = upstream_scores(args.upstream, rows, mode == "qe", args.model_path)
            diffs = [abs(a - b) for a, b in zip(ours, theirs, strict=True)]
            report[f"upstream_{mode}"] = {
                "n": len(diffs),
                "max_abs_diff": max(diffs),
                "mean_abs_diff": sum(diffs) / len(diffs),
            }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
