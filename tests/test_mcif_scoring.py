"""Tests for the post-hoc, official-metric MCIF scorers.

These never import the real `mcif` package: `mcif.evaluation`'s functions are
monkeypatched with fakes, so the tests check the wiring this module is
actually responsible for -- which reference file gets fetched, how the hypo
dict is built, how a missing group member is reported -- without needing
`mcif-bench` (and, transitively, `torch`/`comet`/`bert-score`) installed just
to run the test suite. See `tests/test_readers_mcif.py` for the reader half.
"""

from __future__ import annotations

import gzip
import sys
import types

import pytest
from inspect_ai.scorer import SampleScore, Score

from melteval.mcif_scoring import (
    _hypo_dict,
    _mcif_source,
    official_asr_wer,
    official_bertscore,
    official_trans_comet,
)


def _score(hypothesis: str, **metadata) -> SampleScore:
    metadata.setdefault("repo", "FBK-MT/MCIF")
    metadata.setdefault("revision", "deadbeef")
    metadata.setdefault("track", "short")
    metadata.setdefault("lang", "en")
    metadata.setdefault("cut_id", metadata.get("sample_id", "0"))
    sample_id = metadata.pop("sample_id", metadata["cut_id"])
    return SampleScore(
        score=Score(value={}, metadata={"hypothesis": hypothesis}),
        sample_metadata=metadata,
        sample_id=sample_id,
    )


class TestMcifSource:
    def test_reads_the_shared_source(self):
        scores = [_score("a"), _score("b")]
        assert _mcif_source(scores) == ("FBK-MT/MCIF", "deadbeef", "short", "en")

    def test_mixed_sources_raise(self):
        scores = [_score("a", track="short"), _score("b", track="long")]
        with pytest.raises(ValueError, match="Mixed MCIF sources"):
            _mcif_source(scores)

    def test_missing_source_metadata_raises(self):
        scores = [SampleScore(score=Score(value={}), sample_metadata={}, sample_id="0")]
        with pytest.raises(ValueError, match="no MCIF source metadata"):
            _mcif_source(scores)


class TestHypoDict:
    def test_keys_by_raw_cut_id_not_sample_key(self):
        scores = [_score("hello", cut_id="88"), _score("world", cut_id="89")]
        assert _hypo_dict(scores) == {"88": "hello", "89": "world"}

    def test_falls_back_to_sample_id_without_cut_id(self):
        s = SampleScore(
            score=Score(value={}, metadata={"hypothesis": "x"}),
            sample_metadata={},
            sample_id="00-0",
        )
        assert _hypo_dict([s]) == {"00-0": "x"}


@pytest.fixture
def fake_ref_gz(tmp_path):
    path = tmp_path / "ref.xml.gz"
    path.write_bytes(gzip.compress(b"<testset/>"))
    return str(path)


@pytest.fixture
def fake_mcif_evaluation(monkeypatch):
    """Install a fake `mcif`/`mcif.evaluation` so the metrics under test never
    need the real package. Returns the fake module to let a test set its
    `score_asr`/`score_st`/`score_sqa`/`score_ssum` return values."""
    mcif_pkg = types.ModuleType("mcif")
    evaluation = types.ModuleType("mcif.evaluation")
    mcif_pkg.evaluation = evaluation
    monkeypatch.setitem(sys.modules, "mcif", mcif_pkg)
    monkeypatch.setitem(sys.modules, "mcif.evaluation", evaluation)
    return evaluation


@pytest.fixture
def patched_download(monkeypatch, fake_ref_gz):
    calls = []

    def fake_download(repo, revision, track, lang):
        calls.append((repo, revision, track, lang))
        return fake_ref_gz

    monkeypatch.setattr("melteval.mcif_scoring.download_reference_gz", fake_download)
    return calls


class TestOfficialAsrWer:
    def test_downloads_the_reference_the_samples_point_at(
        self, fake_mcif_evaluation, patched_download
    ):
        fake_mcif_evaluation.read_reference = lambda fh, track, lang, modality=None: {"ASR": {}}
        fake_mcif_evaluation.score_asr = lambda hypo, ref, lang: 0.25

        scores = [_score("hello", track="short", lang="en")]
        assert official_asr_wer()(scores) == pytest.approx(0.25)
        assert patched_download == [("FBK-MT/MCIF", "deadbeef", "short", "en")]

    def test_empty_corpus_is_zero(self):
        assert official_asr_wer()([]) == 0.0

    def test_missing_asr_bucket_is_a_clear_error(self, fake_mcif_evaluation, patched_download):
        fake_mcif_evaluation.read_reference = lambda fh, track, lang, modality=None: {"TRANS": {}}
        with pytest.raises(ValueError, match="No ASR reference"):
            official_asr_wer()([_score("hello", lang="de")])

    def test_a_missing_group_member_is_reported_not_a_bare_keyerror(
        self, fake_mcif_evaluation, patched_download
    ):
        def raising_score_asr(hypo, ref, lang):
            raise KeyError("122")

        fake_mcif_evaluation.read_reference = lambda fh, track, lang, modality=None: {"ASR": {}}
        fake_mcif_evaluation.score_asr = raising_score_asr

        with pytest.raises(ValueError, match="Missing completion for sample id"):
            official_asr_wer()([_score("hello")])


class TestOfficialTransComet:
    def test_calls_score_st(self, fake_mcif_evaluation, patched_download):
        fake_mcif_evaluation.read_reference = lambda fh, track, lang, modality=None: {"TRANS": {}}
        fake_mcif_evaluation.score_st = lambda hypo, ref, lang: 0.81

        scores = [_score("Ich gehe.", track="short", lang="de")]
        assert official_trans_comet()(scores) == pytest.approx(0.81)

    def test_missing_trans_bucket_is_a_clear_error(self, fake_mcif_evaluation, patched_download):
        fake_mcif_evaluation.read_reference = lambda fh, track, lang, modality=None: {"ASR": {}}
        with pytest.raises(ValueError, match="No TRANS reference"):
            official_trans_comet()([_score("hello", lang="en")])


class TestOfficialBertscore:
    def test_qa_calls_score_sqa(self, fake_mcif_evaluation, patched_download):
        fake_mcif_evaluation.read_reference = lambda fh, track, lang, modality=None: {"QA": {}}
        fake_mcif_evaluation.score_sqa = lambda hypo, ref, lang, breakdown_qa_types: (0.7, None)

        assert official_bertscore("QA")([_score("the answer")]) == pytest.approx(0.7)

    def test_sum_calls_score_ssum(self, fake_mcif_evaluation, patched_download):
        fake_mcif_evaluation.read_reference = lambda fh, track, lang, modality=None: {"SUM": {}}
        fake_mcif_evaluation.score_ssum = lambda hypo, ref, lang: 0.6

        assert official_bertscore("SUM")([_score("a summary")]) == pytest.approx(0.6)

    def test_missing_bucket_lists_whats_available(self, fake_mcif_evaluation, patched_download):
        fake_mcif_evaluation.read_reference = lambda fh, track, lang, modality=None: {"QA": {}}
        with pytest.raises(ValueError, match="No SUM reference"):
            official_bertscore("SUM")([_score("x")])
