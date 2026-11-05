"""Tests for post-hoc ST metrics from finished eval logs.

One test writes a real ``.eval`` log with a real ``inspect_ai.eval()`` (no scorer
at all, like a ``--no-score`` run) and reads it back; the rest patch
``read_eval_log`` the way ``test_rescore.py`` does, to cover errors and empty
completions that are awkward to produce for real.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from inspect_ai.model import ModelOutput

from melteval.spm import SPMModelError
from melteval.st_metrics import format_table, score_log, score_logs, to_json


PAIRS = [
    # dataset_id, tgt_lang, reference, hypothesis
    ("fleurs-de_en", "en", "I am going home now.", "I go home now."),
    ("fleurs-de_en", "en", "The weather is nice today.", "The weather is nice."),
    ("fleurs-es_en", "en", "The cat eats fish every day.", "The cat eats fish."),
    ("fleurs-en_de", "de", "Ich gehe nach Hause.", "Ich gehe heim."),
    ("fleurs-en_de", "de", "Guten Tag, wie geht es Ihnen?", "Guten Tag, wie geht es dir?"),
]


class _Eval:
    task = "speech-st"


class _Sample:
    def __init__(self, id, dataset_id, tgt_lang, reference, completion, error=None, epoch=1):
        self.id = id
        self.epoch = epoch
        self.target = reference
        self.output = ModelOutput.from_content(model="mockllm/model", content=completion)
        self.metadata = {"dataset_id": dataset_id, "tgt_lang": tgt_lang, "src_lang": "xx", "lang": tgt_lang}
        self.error = error


class _Log:
    eval = _Eval()

    def __init__(self, samples):
        self.samples = samples


@pytest.fixture
def patched_log(monkeypatch):
    import inspect_ai.log as log_module

    holder = {}
    monkeypatch.setattr(log_module, "read_eval_log", lambda path, resolve_attachments=False: holder["log"])

    def _set(samples):
        holder["log"] = _Log(samples)

    return _set


def _samples(pairs=PAIRS):
    return [_Sample(f"{i:02d}", d, t, r, h) for i, (d, t, r, h) in enumerate(pairs)]


def _by_group(result):
    return {g["group"]: g for g in result["groups"]}


class TestScoreLog:
    def test_matches_sacrebleu_called_directly(self, spm_dir, patched_log):
        import sacrebleu

        patched_log(_samples())
        result = score_log("x.eval", group_by="dataset_id")
        group = _by_group(result)["fleurs-de_en"]

        hyps = [h for d, _, _, h in PAIRS if d == "fleurs-de_en"]
        refs = [[r for d, _, r, _ in PAIRS if d == "fleurs-de_en"]]
        assert group["n"] == 2
        assert group["bleu"] == pytest.approx(sacrebleu.BLEU().corpus_score(hyps, refs).score)
        assert group["spbleu"] == pytest.approx(sacrebleu.BLEU(tokenize="flores200").corpus_score(hyps, refs).score)
        assert group["chrf"] == pytest.approx(sacrebleu.CHRF(word_order=0).corpus_score(hyps, refs).score)
        assert group["chrfpp"] == pytest.approx(sacrebleu.CHRF(word_order=2).corpus_score(hyps, refs).score)

    def test_signatures_are_reported_once_per_log(self, spm_dir, patched_log):
        patched_log(_samples())
        signatures = score_log("x.eval")["signatures"]
        assert "tok:flores200" in signatures["spbleu"][0]
        assert "nw:0" in signatures["chrf"][0] and "nw:2" in signatures["chrfpp"][0]
        assert all("case:mixed" in sig for sigs in signatures.values() for sig in sigs)
        # en and de both route to 13a: one BLEU signature for the whole log.
        assert len(signatures["bleu"]) == 1 and "tok:13a" in signatures["bleu"][0]

    def test_lowercase_reaches_every_metric_and_signature(self, spm_dir, patched_log):
        reference = "Hello World, this is a longer test."
        patched_log(_samples([("d", "en", reference, reference.lower())]))
        cased = score_log("x.eval", lowercase=False)
        lowered = score_log("x.eval", lowercase=True)
        for metric in ("bleu", "spbleu", "chrf", "chrfpp"):
            assert lowered["groups"][0][metric] > 99, metric
            assert cased["groups"][0][metric] < lowered["groups"][0][metric] - 1, metric
        assert all("case:lc" in sig for sigs in lowered["signatures"].values() for sig in sigs)
        assert all("case:mixed" in sig for sigs in cased["signatures"].values() for sig in sigs)

    def test_empty_completion_is_scored_as_a_miss_not_dropped(self, spm_dir, patched_log):
        patched_log(
            [
                _Sample("a", "d", "en", "the cat sat", "the cat sat"),
                _Sample("b", "d", "en", "the dog ran", ""),
            ]
        )
        result = score_log("x.eval")
        assert (result["n_scored"], result["n_empty"], result["n_errors"]) == (2, 1, 0)
        assert result["groups"][0]["n"] == 2
        assert result["groups"][0]["chrfpp"] < 100

    def test_errored_samples_are_skipped_and_counted(self, spm_dir, patched_log):
        patched_log(
            [
                _Sample("a", "d", "en", "the cat sat", "the cat sat"),
                _Sample("b", "d", "en", "the dog ran", "", error=object()),
            ]
        )
        result = score_log("x.eval")
        assert (result["n_samples"], result["n_scored"], result["n_errors"], result["n_empty"]) == (2, 1, 1, 0)
        assert result["groups"][0]["chrfpp"] == pytest.approx(100.0)

    def test_pooling_mixed_targets_leaves_bleu_undefined_but_scores_the_rest(self, spm_dir, patched_log):
        patched_log(_samples())
        pooled = score_log("x.eval", group_by="none")["groups"][0]
        assert pooled["n"] == len(PAIRS)
        assert pooled["bleu"] is None and "several target languages" in pooled["bleu_note"]
        assert all(isinstance(pooled[m], float) for m in ("spbleu", "chrf", "chrfpp"))

    def test_group_by_src_and_tgt_lang(self, spm_dir, patched_log):
        patched_log(_samples())
        assert set(_by_group(score_log("x.eval", group_by="tgt_lang"))) == {"en", "de"}
        assert set(_by_group(score_log("x.eval", group_by="src_lang"))) == {"xx"}

    def test_log_without_dataset_id_falls_back_to_tgt_lang(self, spm_dir, patched_log):
        samples = _samples()
        for sample in samples:
            del sample.metadata["dataset_id"]
        patched_log(samples)
        result = score_log("x.eval")
        assert result["group_by"] == "tgt_lang"
        assert set(_by_group(result)) == {"en", "de"}

    def test_partial_dataset_id_is_an_error_not_a_guess(self, spm_dir, patched_log):
        samples = _samples()
        del samples[0].metadata["dataset_id"]
        patched_log(samples)
        with pytest.raises(ValueError, match="no dataset_id"):
            score_log("x.eval")

    def test_multi_epoch_logs_are_refused(self, spm_dir, patched_log):
        patched_log([_Sample("a", "d", "en", "x", "x", epoch=2)])
        with pytest.raises(ValueError, match="epochs"):
            score_log("x.eval")

    def test_missing_model_fails_before_reading_the_log(self, no_spm_dir):
        with pytest.raises(SPMModelError):
            score_log("/nonexistent/log.eval")

    def test_flores101_changes_the_signature(self, spm_dir, patched_log):
        patched_log(_samples())
        assert "tok:flores101" in score_log("x.eval", spm="flores101")["signatures"]["spbleu"][0]


class TestOutput:
    def test_table_shows_counts_signatures_and_every_metric(self, spm_dir, patched_log):
        patched_log(_samples())
        table = format_table(score_log("x.eval"))
        for expected in ("fleurs-de_en", "BLEU", "spBLEU", "chrF ", "chrF++", "tok:flores200", "errors skipped=0"):
            assert expected in table

    def test_table_says_n_a_for_an_undefined_bleu(self, spm_dir, patched_log):
        patched_log(_samples())
        assert "n/a" in format_table(score_log("x.eval", group_by="none"))

    def test_json_round_trips(self, spm_dir, patched_log):
        patched_log(_samples())
        [parsed] = json.loads(to_json(score_logs(["x.eval"])))
        assert {g["group"] for g in parsed["groups"]} == {"fleurs-de_en", "fleurs-es_en", "fleurs-en_de"}


class TestThroughARealLog:
    def test_reads_a_log_written_by_a_real_unscored_run(self, spm_dir, tmp_path):
        from inspect_ai import Task
        from inspect_ai import eval as inspect_eval
        from inspect_ai.dataset import MemoryDataset, Sample

        samples = [
            Sample(
                input="translate",
                target=ref,
                id=f"k{i}",
                metadata={"dataset_id": dataset_id, "tgt_lang": lang, "src_lang": "xx", "lang": lang},
            )
            for i, (dataset_id, lang, ref, _) in enumerate(PAIRS)
        ]
        outputs = [ModelOutput.from_content(model="mockllm/model", content=hyp) for *_, hyp in PAIRS]
        [log] = inspect_eval(
            Task(dataset=MemoryDataset(samples=samples)),
            model="mockllm/model",
            model_args={"custom_outputs": outputs},
            display="none",
            log_dir=str(tmp_path),
        )
        assert log.results is None or not log.results.scores  # no scorer ran

        [path] = tmp_path.glob("*.eval")
        result = score_log(path)
        assert result["n_scored"] == len(PAIRS) and result["n_errors"] == 0
        assert _by_group(result)["fleurs-en_de"]["n"] == 2
        assert _by_group(result)["fleurs-en_de"]["spbleu"] > 0


def test_importing_the_module_pulls_in_no_model_stack():
    code = (
        "import sys, melteval.st_metrics, melteval.cli\n"
        "bad = [m for m in ('torch', 'lhotse', 'melt', 'transformers') if m in sys.modules]\n"
        "assert not bad, bad"
    )
    repo = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-c", code], check=True, cwd=repo, env={**os.environ, "PYTHONPATH": str(repo)})
