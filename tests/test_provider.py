"""
Tests for the model providers' message handling and config translation, that is, extracting prompts and audio locators from messages and converting GenerateConfig into generation kwargs.
"""

from dataclasses import dataclass, field

import pytest
from inspect_ai.model import ChatMessageUser, ContentAudio, ContentData, ContentText, GenerateConfig

from melteval.dataset import AUDIO_DATA_KEY
from melteval.providers.base import _extract
from melteval.providers.melt import _batched_audio, _generate_kwargs
from melteval.providers.smurf import _collate_audio, _generation_kwargs, _plan_device_map, _with_audio_tag


class TestExtract:
    def test_plain_string_content(self):
        text, locator = _extract([ChatMessageUser(content="hello")])
        assert text == "hello"
        assert locator is None

    def test_text_and_locator_content(self):
        locator = {"kind": "shar", "dir": "/d", "index": 3}
        message = ChatMessageUser(
            content=[ContentText(text="transcribe this"), ContentData(data={AUDIO_DATA_KEY: locator})]
        )
        text, found = _extract([message])
        assert text == "transcribe this"
        assert found == locator

    def test_content_audio_becomes_a_file_locator(self):
        """--materialize-audio produces ContentAudio directly; the provider
        must resolve it the same way as a shar/hf locator."""
        message = ChatMessageUser(
            content=[ContentText(text="x"), ContentAudio(audio="/tmp/a.wav", format="wav")]
        )
        text, locator = _extract([message])
        assert text == "x"
        assert locator == {"kind": "file", "path": "/tmp/a.wav"}

    def test_no_audio_content_yields_no_locator(self):
        text, locator = _extract([ChatMessageUser(content=[ContentText(text="x")])])
        assert text == "x"
        assert locator is None

    def test_multiple_messages_concatenate_text(self):
        text, _ = _extract(
            [ChatMessageUser(content="a"), ChatMessageUser(content="b")]
        )
        assert text == "ab"


@dataclass
class _FakeRequest:
    """Stands in for `_Request` -- only `.audio` matters to `_batched_audio`."""

    audio: object = None
    text: str = ""
    sample_rate: int = 16000
    config: object = None
    future: object = field(default=None)


class TestBatchedAudio:
    """Regression coverage for the shape bug found running a real batch:
    MELTProcessor wants one list per text sample, not a flat list of arrays."""

    def test_all_text_batch_is_none(self):
        """None, not an empty list -- MELTProcessor treats `audio=[]` as a
        (mismatched-length) batch of zero, not as "no audio at all"."""
        batch = [_FakeRequest(audio=None), _FakeRequest(audio=None)]
        assert _batched_audio(batch) is None

    def test_each_sample_gets_its_own_singleton_list(self):
        batch = [_FakeRequest(audio="a"), _FakeRequest(audio="b")]
        assert _batched_audio(batch) == [["a"], ["b"]]

    def test_text_only_sample_in_a_mixed_batch_gets_an_empty_list(self):
        """Length must still match `text` even for the sample with no audio
        token -- that's what "aligned with text" means."""
        batch = [_FakeRequest(audio="a"), _FakeRequest(audio=None)]
        assert _batched_audio(batch) == [["a"], []]

    def test_single_sample_batch(self):
        assert _batched_audio([_FakeRequest(audio="only")]) == [["only"]]


class TestGenerateKwargs:
    def test_defaults_to_greedy(self):
        """Sampling by default would measure the sampler as much as the model."""
        kwargs = _generate_kwargs(GenerateConfig())
        assert kwargs["do_sample"] is False
        assert "temperature" not in kwargs

    def test_zero_temperature_stays_greedy(self):
        kwargs = _generate_kwargs(GenerateConfig(temperature=0.0))
        assert kwargs["do_sample"] is False

    def test_positive_temperature_enables_sampling(self):
        kwargs = _generate_kwargs(GenerateConfig(temperature=0.7))
        assert kwargs["do_sample"] is True
        assert kwargs["temperature"] == 0.7

    def test_max_tokens_falls_back_to_256(self):
        assert _generate_kwargs(GenerateConfig())["max_new_tokens"] == 256

    def test_max_tokens_is_forwarded(self):
        assert _generate_kwargs(GenerateConfig(max_tokens=64))["max_new_tokens"] == 64

    def test_top_p_and_top_k_are_forwarded_when_set(self):
        kwargs = _generate_kwargs(GenerateConfig(top_p=0.9, top_k=40))
        assert kwargs["top_p"] == 0.9
        assert kwargs["top_k"] == 40

    def test_num_choices_above_one_becomes_num_return_sequences(self):
        kwargs = _generate_kwargs(GenerateConfig(num_choices=4))
        assert kwargs["num_return_sequences"] == 4

    def test_single_choice_is_not_forwarded(self):
        assert "num_return_sequences" not in _generate_kwargs(GenerateConfig(num_choices=1))


class TestBatchWorker:
    """The shared worker, which both providers depend on for correctness of
    *which* completion belongs to which sample."""

    def _api(self, generate_batch):
        from melteval.providers.base import BatchedSpeechAPI

        class _FakeAPI(BatchedSpeechAPI):
            def _generate_batch(self, batch):
                return generate_batch(batch)

        return _FakeAPI("fake", batch_size=2)

    def _submit(self, api, count):
        from melteval.providers.base import _Request

        requests = [
            _Request(text=f"p{i}", audio=None, sample_rate=16000, config=GenerateConfig())
            for i in range(count)
        ]
        api._ensure_worker()
        for request in requests:
            api._queue.put(request)
        return [r.future for r in requests]

    def test_completions_are_returned_in_request_order(self):
        api = self._api(lambda batch: [r.text.upper() for r in batch])
        futures = self._submit(api, 2)
        assert [f.result(timeout=10).completion for f in futures] == ["P0", "P1"]

    def test_a_short_completion_list_fails_the_batch_instead_of_misaligning(self):
        """zip() would pair the completions off in order and drop the tail --
        scoring later samples against another sample's audio, silently."""
        api = self._api(lambda batch: ["only one"])
        futures = self._submit(api, 2)
        for future in futures:
            with pytest.raises(ValueError, match="exactly one completion"):
                future.result(timeout=10)

    def test_a_failing_forward_pass_releases_every_waiter(self):
        """Otherwise the run hangs: the coroutines poll a future nobody completes."""

        def boom(batch):
            raise RuntimeError("CUDA out of memory")

        api = self._api(boom)
        futures = self._submit(api, 2)
        for future in futures:
            with pytest.raises(RuntimeError, match="CUDA out of memory"):
                future.result(timeout=10)


class TestSmurfAudioTag:
    """The placeholder is the checkpoint's, so the provider attaches it."""

    TAG = "<|audioplaceholder|>"

    def test_suffix_matches_how_smurf_builds_its_conversations(self):
        """Instruction turn first, audio turn second -- so the tag lands last."""
        assert _with_audio_tag("Transcribe: ", self.TAG, "suffix") == f"Transcribe: {self.TAG}"

    def test_prefix_puts_the_audio_first(self):
        assert _with_audio_tag("Transcribe: ", self.TAG, "prefix") == f"{self.TAG}Transcribe: "

    def test_a_prompt_that_already_positions_the_tag_is_left_alone(self):
        """A benchmark shipping its own prompt keeps control of the position,
        and must not end up with two placeholders for one audio."""
        prompt = f"Given {self.TAG}, answer the question."
        assert _with_audio_tag(prompt, self.TAG, "suffix") == prompt


class TestSmurfGenerationKwargs:
    def test_defaults_to_greedy(self):
        kwargs = _generation_kwargs(GenerateConfig())
        assert kwargs["do_sample"] is False
        assert "temperature" not in kwargs

    def test_beams_are_only_set_when_wider_than_one(self):
        """GenerationConfig(num_beams=1) is the default; setting it explicitly
        alongside do_sample=False is noise in the log."""
        assert "num_beams" not in _generation_kwargs(GenerateConfig(), num_beams=1)
        assert _generation_kwargs(GenerateConfig(), num_beams=5)["num_beams"] == 5

    def test_max_tokens_falls_back_to_256(self):
        assert _generation_kwargs(GenerateConfig())["max_new_tokens"] == 256

    def test_sampling_params_are_forwarded(self):
        kwargs = _generation_kwargs(GenerateConfig(temperature=0.7, top_p=0.9, top_k=40))
        assert kwargs["do_sample"] is True
        assert (kwargs["temperature"], kwargs["top_p"], kwargs["top_k"]) == (0.7, 0.9, 40)


class TestSmurfCollateAudio:
    """SALM takes a zero-padded (B, T) waveform batch plus true lengths."""

    def test_pads_to_the_longest_and_reports_true_lengths(self):
        torch = pytest.importorskip("torch")
        batch = [_FakeRequest(audio=[0.5, 0.5, 0.5]), _FakeRequest(audio=[1.0])]

        audios, audio_lens = _collate_audio(batch, 16000)

        assert audios.shape == (2, 3)
        assert audios.dtype is torch.float32
        assert audio_lens.tolist() == [3, 1]
        # The pad must be silence, not a repeat: the encoder sees it either way,
        # and only the lengths tell it where the signal ends.
        assert audios[1, 1:].tolist() == [0.0, 0.0]

    def test_missing_audio_is_an_error(self):
        pytest.importorskip("torch")
        batch = [_FakeRequest(audio=[0.1]), _FakeRequest(audio=None)]
        with pytest.raises(ValueError, match="no audio"):
            _collate_audio(batch, 16000)

    def test_sample_rate_mismatch_is_an_error(self):
        """Resampling silently here would feed the encoder audio at the wrong
        speed and still produce fluent text."""
        pytest.importorskip("torch")
        batch = [_FakeRequest(audio=[0.1], sample_rate=8000)]
        with pytest.raises(ValueError, match="8000"):
            _collate_audio(batch, 16000)


class TestPlanDeviceMap:
    """Pure placement arithmetic for `device_map="auto"` -- no CUDA needed.

    Verified separately against a real checkpoint (a 9B-parameter LLM backbone
    plus a Conformer speech encoder) sharded across two 24 GB GPUs: this is
    the logic that made `generate()` stop crashing cross-device once the
    anchors (embed_tokens, perception, the LLM's head/norm/rotary) were forced
    onto the same GPU as decoder layer 0.
    """

    ANCHORS = ["embed_tokens", "perception", "llm.lm_head", "llm.model.norm"]

    def test_anchors_all_land_on_device_zero(self):
        device_map = _plan_device_map(self.ANCHORS, anchor_bytes=10, layer_bytes=[], budgets=[100])
        assert all(device_map[name] == 0 for name in self.ANCHORS)

    def test_layers_stay_on_device_zero_while_they_fit(self):
        device_map = _plan_device_map(
            self.ANCHORS, anchor_bytes=10, layer_bytes=[20, 20, 20], budgets=[100, 100]
        )
        assert [device_map[f"llm.model.layers.{i}"] for i in range(3)] == [0, 0, 0]

    def test_overflow_moves_to_the_next_device(self):
        """Anchors (10) + 3 layers of 20 leave only 30 free on device 0 -- the
        4th layer doesn't fit and spills to device 1."""
        device_map = _plan_device_map(
            self.ANCHORS, anchor_bytes=10, layer_bytes=[20, 20, 20, 20], budgets=[70, 100]
        )
        assert [device_map[f"llm.model.layers.{i}"] for i in range(4)] == [0, 0, 0, 1]

    def test_a_layer_bigger_than_every_budget_still_lands_somewhere(self):
        """No device ever refuses a layer outright -- packing is best-effort;
        an actual OOM at dispatch time is the real signal the model doesn't
        fit, not a silent wrong placement here. With more than one device it
        moves on to try the next one first, same as an ordinary overflow."""
        device_map = _plan_device_map(self.ANCHORS, anchor_bytes=10, layer_bytes=[500], budgets=[100, 100])
        assert device_map["llm.model.layers.0"] == 1

    def test_the_last_device_absorbs_everything_left_once_reached(self):
        """Once packing reaches the last device, later layers land there even
        over budget -- there is nowhere else to spill to."""
        device_map = _plan_device_map(
            self.ANCHORS, anchor_bytes=10, layer_bytes=[20, 20, 20, 20, 20], budgets=[30, 40]
        )
        assert [device_map[f"llm.model.layers.{i}"] for i in range(5)] == [0, 1, 1, 1, 1]

    def test_a_single_gpu_keeps_everything_on_device_zero(self):
        device_map = _plan_device_map(self.ANCHORS, anchor_bytes=10, layer_bytes=[20, 20, 20], budgets=[1])
        assert set(device_map.values()) == {0}
