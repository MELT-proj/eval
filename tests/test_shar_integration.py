"""Integration tests against the real Shar corpora.

Skipped unless ``MELTEVAL_SHAR_ROOT`` points at a Shar tree. These cover the
things that cannot be faked: that manifest order and indexed random-access order
agree, and that each corpus's reference really does come out of the field its
config names.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from melteval.freeze import freeze
from melteval.manifest import MANIFEST_NAME, read_manifest


SHAR_ROOT = os.environ.get("MELTEVAL_SHAR_ROOT")

pytestmark = pytest.mark.skipif(
    not SHAR_ROOT or not Path(SHAR_ROOT).exists(),
    reason="Set MELTEVAL_SHAR_ROOT to a Shar tree to run integration tests.",
)


def _spec(subpath: str, **source):
    source.setdefault("type", "lhotse_shar")
    source["shar_path"] = f"{SHAR_ROOT}/{subpath}"
    return {"name": "it", "seed": 0, "input_cfg": [source]}


@pytest.fixture(scope="module")
def fleurs_set(tmp_path_factory):
    out = tmp_path_factory.mktemp("fleurs")
    freeze(
        _spec("fleurs/de_de/test", max_samples=8, text_field="custom.pnc_text",
              tags={"task": "asr", "lang": "de", "dataset_id": "fleurs"}),
        out,
    )
    return out


class TestLocatorResolution:
    def test_locator_resolves_to_the_recorded_cut(self, fleurs_set: Path):
        """The load-bearing invariant: index i really is the cut frozen at i.

        Manifest iteration and indexed random access are two different code
        paths through lhotse. If their orders ever diverged, every sample would
        be scored against someone else's audio and nothing would raise.
        """
        from melteval.readers.base import reader_for_locator

        for record in read_manifest(fleurs_set / MANIFEST_NAME):
            reader = reader_for_locator(record.audio)
            cut = reader._cut_at(record.audio.params["dir"], record.audio.params["index"])
            assert str(cut.id) == record.cut_id

    def test_audio_duration_matches_the_manifest(self, fleurs_set: Path):
        from melteval.readers.base import reader_for_locator

        for record in read_manifest(fleurs_set / MANIFEST_NAME):
            reader = reader_for_locator(record.audio)
            audio, sample_rate = reader.load_audio(record.audio)
            assert audio.ndim == 1
            assert abs(len(audio) / sample_rate - record.duration) < 0.05


class TestReferenceResolution:
    """Each corpus keeps its reference somewhere different."""

    def test_librispeech_reads_pnc_text(self, tmp_path: Path):
        """pnc_text is punctuated and cased; supervision text is neither."""
        freeze(
            _spec("librispeech/clean/test", max_samples=5, text_field="custom.pnc_text",
                  tags={"task": "asr", "lang": "en"}),
            tmp_path,
        )
        targets = [r.target for r in read_manifest(tmp_path / MANIFEST_NAME)]
        assert any(t != t.upper() and any(c in t for c in ".,?!") for t in targets)

    def test_commonvoice_reads_the_sentence_field(self, tmp_path: Path):
        freeze(
            _spec("cv22_sidon/de/test", max_samples=5,
                  text_field="custom.metadata.sentence",
                  tags={"task": "asr", "lang": "de"}),
            tmp_path,
        )
        records = read_manifest(tmp_path / MANIFEST_NAME)
        assert len(records) == 5
        assert all(r.target.strip() for r in records)

    def test_covost_keeps_translation_and_source_transcript_apart(self, tmp_path: Path):
        """For ST the reference is the translation; the transcript is the source."""
        freeze(
            _spec("covost2/ar_en/test", max_samples=5, source_text_field="custom.sentence",
                  tags={"task": "st", "src_lang": "ar", "tgt_lang": "en"}),
            tmp_path,
        )
        for record in read_manifest(tmp_path / MANIFEST_NAME):
            assert record.task == "st"
            # get_tags_from_cut reports the *target* language as `lang` for st.
            assert record.lang == "en"
            assert record.source_text and record.source_text != record.target


class TestConcurrentAccess:
    """Regression test for MELT-proj/eval#2.

    MELTAPI.generate() resolves audio for up to ``batch_size`` samples at once
    via a thread pool, so several threads hit the same cached indexed reader
    concurrently. That reader's ``__getitem__`` does a bare seek+read against
    one shared file handle with no locking of its own -- lhotse assumes
    single-threaded access. Without a lock in ``_cut_at``, concurrent lookups
    interleave and a read lands at the wrong offset, decoding as tar-header
    garbage (``InvalidHeaderError: bad checksum``).
    """

    def test_concurrent_cut_at_calls_do_not_corrupt_reads(self, tmp_path: Path):
        out = tmp_path / "librispeech"
        freeze(
            _spec("librispeech/clean/test", max_samples=64,
                  text_field="custom.pnc_text", tags={"task": "asr", "lang": "en"}),
            out,
        )
        records = read_manifest(out / MANIFEST_NAME)
        assert len(records) == 64

        from melteval.readers.base import reader_for_locator

        reader = reader_for_locator(records[0].audio)
        shar_dir = records[0].audio.params["dir"]

        # Every thread hammers the same directory's cached reader, several
        # times over, so a real race has many chances to show up.
        def resolve(record):
            cut = reader._cut_at(shar_dir, record.audio.params["index"])
            return record.cut_id, str(cut.id)

        jobs = records * 4
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(resolve, jobs))

        for expected_id, actual_id in results:
            assert actual_id == expected_id


class TestSubsetting:
    def test_same_spec_selects_the_same_subset(self, tmp_path: Path):
        """Two runs of one spec must be comparable, so the subset must be fixed."""
        first, second = tmp_path / "a", tmp_path / "b"
        spec = _spec("fleurs/de_de/test", max_samples=10, seed=7, tags={"task": "asr", "lang": "de"})
        freeze(spec, first)
        freeze(spec, second)
        assert [r.cut_id for r in read_manifest(first / MANIFEST_NAME)] == [
            r.cut_id for r in read_manifest(second / MANIFEST_NAME)
        ]

    def test_records_are_ordered_by_index_for_sequential_reads(self, tmp_path: Path):
        """Random access costs ~18 ms a sample against ~2 ms sequential."""
        freeze(
            _spec("fleurs/de_de/test", max_samples=10, tags={"task": "asr", "lang": "de"}),
            tmp_path,
        )
        indices = [r.audio.params["index"] for r in read_manifest(tmp_path / MANIFEST_NAME)]
        assert indices == sorted(indices)
