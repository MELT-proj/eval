"""Tests for the MetricX rescoring of ST logs (``projects/metricx/score.py``).

The formatting, joining, batching and aggregation logic runs against a fake scorer and a mock model, on a four
sentence log, with no torch. One ``hub`` test loads the real checkpoint (pinned, from the hub cache) and needs a
GPU; the rest are plain unit tests.

``projects/`` is not a package, so the module is loaded by path.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest


def _load():
    path = Path(__file__).resolve().parents[1] / "projects" / "metricx" / "score.py"
    spec = importlib.util.spec_from_file_location("metricx_score", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mx = _load()

#: id, src_lang, source transcript, reference, hypothesis. The Irish sample has no source transcript.
_ROWS = [
    ("00-000000", "es", "La ONU espera un fondo.", "The U.N. hopes for a fund.", "the un hopes for a fund."),
    ("00-000001", "es", "Hola a todos.", "Hello everyone.", "hello all."),
    ("01-000000", "de", "Guten Tag, wie geht es?", "Good day, how are you?", "good day how are you"),
    ("02-000000", "ga", "", "The weather is nice.", "the weather is nice."),
]


def _log(direction: str = "xen", rows=_ROWS, status: str = "success", extra: list[dict] | None = None) -> dict:
    samples = []
    for sample_id, lang, source, reference, hypothesis in rows:
        if direction == "xen":
            src_lang, tgt_lang, target, text = lang, "en", reference, source
        else:  # the English transcript is the source, the reference is in the target language
            src_lang, tgt_lang, target, text = "en", lang, f"[{lang}] {reference}", reference
        metadata = {"src_lang": src_lang, "tgt_lang": tgt_lang, "dataset_id": f"fleurs-{src_lang}_{tgt_lang}"}
        if text:
            metadata["source_text"] = text
        samples.append(
            {
                "id": sample_id,
                "target": target,
                "output": {"completion": hypothesis},
                "metadata": metadata,
                "error": None,
            }
        )
    return {
        "status": status,
        "eval": {"task": "speech-st", "dataset": {"name": f"fleurs24-st-{direction}-dev"}},
        "samples": samples + (extra or []),
    }


@pytest.fixture
def xen_log(tmp_path) -> Path:
    path = tmp_path / "xen.json"
    path.write_text(json.dumps(_log("xen")))
    return path


class FakeScorer:
    """Scores a string by its length, and records every call so tests can read the inputs."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def score(self, inputs: list[str]) -> list[float]:
        self.calls.append(list(inputs))
        return [float(len(text) % 25) for text in inputs]


def test_input_templates_match_upstream():
    assert mx.build_input("S", "H", "R") == "source: S candidate: H reference: R"
    assert mx.build_input("S", "H") == "source: S candidate: H"
    assert mx.build_input("", "H", "R") == "source:  candidate: H reference: R"  # empty source, as upstream allows


def test_read_log_takes_fields_from_the_log(xen_log):
    segments, header = mx.read_log(xen_log)
    first = segments[0]
    assert (first.id, first.src_lang, first.tgt_lang) == ("00-000000", "es", "en")
    assert first.source == "La ONU espera un fondo."  # the transcript, from metadata.source_text
    assert first.reference == "The U.N. hopes for a fund."  # sample.target
    assert first.hypothesis == "the un hopes for a fund."  # output.completion
    assert [s.has_source for s in segments] == [True, True, True, False]
    assert (header["n_samples"], header["n_source_missing"], header["dataset"]) == (4, 1, "fleurs24-st-xen-dev")


def test_read_log_skips_errors_and_rejects_unfinished_logs(tmp_path):
    errored = {"id": "03-000000", "target": "x", "output": {}, "metadata": {}, "error": {"message": "boom"}}
    path = tmp_path / "log.json"
    path.write_text(json.dumps(_log(extra=[errored])))
    segments, header = mx.read_log(path)
    assert len(segments) == 4 and header["n_errors"] == 1

    path.write_text(json.dumps(_log(status="error")))
    with pytest.raises(ValueError, match="status 'error'"):
        mx.read_log(path)


def test_read_log_accepts_a_directory_with_one_log(tmp_path, xen_log):
    assert len(mx.read_log(xen_log.parent)[0]) == 4
    (tmp_path / "second.json").write_text(json.dumps(_log()))
    with pytest.raises(ValueError, match="holds 2 logs"):
        mx.read_log(tmp_path)


def test_language_axis_follows_the_constant_side(tmp_path):
    path = tmp_path / "enx.json"
    path.write_text(json.dumps(_log("enx")))
    segments, _ = mx.read_log(path)
    assert mx.language_axis(segments) == "tgt_lang"
    assert segments[0].source == "The U.N. hopes for a fund."  # en->X: the English transcript is the source
    assert segments[0].reference.startswith("[es] ")

    xen_segments, _ = mx.read_log(_write(tmp_path, "xen2.json", _log("xen")))
    assert mx.language_axis(xen_segments) == "src_lang"
    with pytest.raises(ValueError, match="mixes directions"):
        mx.language_axis([*segments, *xen_segments])


def _write(tmp_path: Path, name: str, log: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(log))
    return path


def test_score_segments_builds_ref_and_qe_inputs_and_skips_qe_without_source(xen_log):
    segments, _ = mx.read_log(xen_log)
    scorer = FakeScorer()
    rows = mx.score_segments(segments, scorer, casing="as-is")
    ref_inputs, qe_inputs = scorer.calls
    assert ref_inputs[0] == (
        "source: La ONU espera un fondo. candidate: the un hopes for a fund. reference: The U.N. hopes for a fund."
    )
    assert ref_inputs[3] == "source:  candidate: the weather is nice. reference: The weather is nice."
    assert qe_inputs[0] == "source: La ONU espera un fondo. candidate: the un hopes for a fund."
    assert len(qe_inputs) == 3  # no QE input for the Irish sample
    assert [r["metricx_qe"] is None for r in rows] == [False, False, False, True]
    assert all(r["metricx_ref"] is not None for r in rows)
    assert not any(key.endswith("_lc") for key in rows[0])


def test_lowercase_casing_lowercases_source_hypothesis_and_reference(xen_log):
    segments, _ = mx.read_log(xen_log)
    scorer = FakeScorer()
    rows = mx.score_segments(segments, scorer, casing="both")
    as_is_ref, as_is_qe, lc_ref, lc_qe = scorer.calls
    assert "The U.N." in as_is_ref[0] and "La ONU" in as_is_ref[0]
    assert lc_ref[0] == (
        "source: la onu espera un fondo. candidate: the un hopes for a fund. reference: the u.n. hopes for a fund."
    )
    assert lc_qe[0] == "source: la onu espera un fondo. candidate: the un hopes for a fund."
    assert {"metricx_ref", "metricx_qe", "metricx_ref_lc", "metricx_qe_lc"} <= set(rows[0])
    with pytest.raises(ValueError, match="casing"):
        mx.score_segments(segments, scorer, casing="title")


def test_summary_groups_by_language_and_averages_languages_not_samples(xen_log):
    segments, _ = mx.read_log(xen_log)
    rows = mx.score_segments(segments, FakeScorer(), casing="as-is")
    summary = mx.summarize(rows, "src_lang")
    by_lang = {e["lang"]: e for e in summary["languages"]}
    assert set(by_lang) == {"es", "de", "ga"} and by_lang["es"]["n"] == 2
    assert by_lang["es"]["metricx_ref"] == pytest.approx((rows[0]["metricx_ref"] + rows[1]["metricx_ref"]) / 2)
    assert by_lang["ga"]["metricx_qe"] is None and by_lang["ga"]["n_source_missing"] == 1
    macro = summary["macro_mean"]["metricx_ref"]
    assert macro == pytest.approx(sum(by_lang[lang]["metricx_ref"] for lang in by_lang) / 3)  # not the sample mean
    # Languages with no source are left out of the "with source" macro, so ref and QE macros cover the same set.
    assert summary["macro_mean_with_source"]["n_languages"] == 2
    assert summary["macro_mean"]["metricx_qe"] == pytest.approx(summary["macro_mean_with_source"]["metricx_qe"])
    # Tiers follow the existing ST tables: es/de are high-resource, ga is low.
    assert summary["tiers"]["high"]["n_languages"] == 2 and summary["tiers"]["low"]["n_languages"] == 1
    assert summary["tiers"]["mid"]["metricx_ref"] is None


def test_score_log_writes_scores_summary_json_and_csv(tmp_path, xen_log):
    out_dir = tmp_path / "out" / "m" / "fleurs24-st-xen-dev"
    summary = mx.score_log(xen_log, out_dir, FakeScorer(), casing="as-is")
    rows = [json.loads(line) for line in (out_dir / "scores.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["00-000000", "00-000001", "01-000000", "02-000000"]
    assert rows[3]["metricx_qe"] is None and rows[3]["source_missing"] is True
    on_disk = json.loads((out_dir / "summary.json").read_text())
    assert on_disk["provenance"]["model_revision"] == mx.MODEL_REVISION == summary["provenance"]["model_revision"]
    assert on_disk["provenance"]["template_ref"] == "source: {src} candidate: {hyp} reference: {ref}"
    table = list(csv.DictReader((out_dir / "summary.csv").open()))
    assert [r["group"] for r in table if r["kind"] == "language"] == ["de", "es", "ga"]
    assert {r["kind"] for r in table} == {"language", "tier", "macro"}
    assert next(r for r in table if r["group"] == "ga")["metricx_qe"] == ""


def test_limit_takes_a_seeded_subset(tmp_path, xen_log):
    first = mx.score_log(xen_log, tmp_path / "a", FakeScorer(), casing="as-is", limit=2, seed=1)
    again = mx.score_log(xen_log, tmp_path / "b", FakeScorer(), casing="as-is", limit=2, seed=1)
    assert first["provenance"]["n_scored"] == 2
    assert (tmp_path / "a" / "scores.jsonl").read_text() == (tmp_path / "b" / "scores.jsonl").read_text()
    assert again["provenance"]["n_scored"] == 2


# --- batching ---------------------------------------------------------------------------------------------


def test_pad_batch_pads_right_with_a_matching_mask():
    ids, mask = mx.pad_batch([[5, 6, 7], [8]], pad_id=0)
    assert ids == [[5, 6, 7], [8, 0, 0]] and mask == [[1, 1, 1], [1, 0, 0]]


def _mask_aware_predict(calls: list[list[list[int]]]):
    """A mock model that reads a batch through ``pad_batch`` and attention masking, as the real one does.

    Its score depends only on a row's own tokens; if the padding leaked into it, the score would change with
    the batch the row was put in.
    """

    def predict(batch: list[list[int]]) -> list[float]:
        calls.append(batch)
        input_ids, mask = mx.pad_batch(batch, pad_id=0)
        return [
            sum(i * (k + 1) for k, (i, m) in enumerate(zip(row, row_mask, strict=True)) if m) / 7.0
            for row, row_mask in zip(input_ids, mask, strict=True)
        ]

    return predict


def test_scores_do_not_depend_on_batch_composition_or_order():
    rng = random.Random(0)
    token_ids = [[rng.randrange(1, 100) for _ in range(rng.randrange(1, 40))] for _ in range(50)]
    alone = mx.score_token_ids(token_ids, _mask_aware_predict([]), batch_size=1)
    batched = mx.score_token_ids(token_ids, _mask_aware_predict([]), batch_size=16)
    permutation = list(range(50))
    rng.shuffle(permutation)
    shuffled = mx.score_token_ids([token_ids[i] for i in permutation], _mask_aware_predict([]), batch_size=7)
    assert batched == alone
    assert [shuffled[permutation.index(i)] for i in range(50)] == batched  # the input order never matters


def test_batches_are_length_sorted_and_respect_both_limits():
    calls: list[list[list[int]]] = []
    token_ids = [[1] * n for n in (3, 30, 5, 30, 4, 29, 2)]
    scores = mx.score_token_ids(token_ids, _mask_aware_predict(calls), batch_size=3, max_batch_tokens=70)
    assert all(len(batch) <= 3 for batch in calls)
    assert all(max(map(len, batch)) * len(batch) <= 70 or len(batch) == 1 for batch in calls)
    flat = [len(ids) for batch in calls for ids in batch]
    assert flat == sorted(flat, reverse=True)  # longest first, so padding stays small
    assert len(scores) == 7 and sorted(flat) == sorted(map(len, token_ids))


# --- the real model ---------------------------------------------------------------------------------------


@pytest.mark.hub
def test_real_model_smoke():
    """The pinned checkpoint on a GPU: range, copy ~ 0, shuffled worse, batch-size independence."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentencepiece")
    if not torch.cuda.is_available():
        pytest.skip("MetricX-24 XXL needs a GPU")
    scorer = mx.MetricXScorer()

    pairs = [
        ("Der Himmel ist heute blau.", "The sky is blue today."),
        (
            "Ich habe gestern einen langen Brief an meine Schwester geschrieben.",
            "I wrote a long letter to my sister yesterday.",
        ),
        ("Die Regierung hat neue Regeln für den Verkehr angekündigt.", "The government announced new traffic rules."),
        ("Wir treffen uns morgen um acht Uhr am Bahnhof.", "We are meeting at the station at eight o'clock tomorrow."),
    ]
    copies = [mx.build_input(src, ref, ref) for src, ref in pairs]
    wrong = [mx.build_input(src, pairs[(i + 2) % 4][1], ref) for i, (src, ref) in enumerate(pairs)]

    copy_scores, wrong_scores = scorer.score(copies), scorer.score(wrong)
    assert all(mx.SCORE_MIN <= s <= mx.SCORE_MAX for s in copy_scores + wrong_scores)
    assert max(copy_scores) < 2.0
    assert min(wrong_scores) > max(copy_scores) + 3.0

    scorer.batch_size = 1
    singly = scorer.score(copies + wrong)
    scorer.batch_size = 8
    together = scorer.score(copies + wrong)
    assert singly == pytest.approx(together, abs=0.25)  # bf16 padding noise only
