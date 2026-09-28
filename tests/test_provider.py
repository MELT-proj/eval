"""Tests for the model providers' message handling and config translation.

That is, extracting prompts and audio locators from messages and converting
``GenerateConfig`` into generation kwargs.
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

    def test_a_requests_metadata_reaches_its_output_only(self):
        """How a provider records in the log what it did to one sample (e.g. truncated its audio)."""

        def generate(batch):
            batch[0].metadata["audio_truncated_from_seconds"] = 31.6
            return [r.text for r in batch]

        futures = self._submit(self._api(generate), 2)
        outputs = [f.result(timeout=10) for f in futures]
        assert outputs[0].metadata == {"audio_truncated_from_seconds": 31.6}
        assert outputs[1].metadata is None

    def test_a_requests_stop_reason_reaches_its_output(self):
        """How a completion cut at max_tokens is told apart in the log from one that ended on its own."""

        def generate(batch):
            batch[1].stop_reason = "max_tokens"
            return [r.text for r in batch]

        futures = self._submit(self._api(generate), 2)
        assert [f.result(timeout=10).stop_reason for f in futures] == ["stop", "max_tokens"]

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
        """Instruction turn first, audio turn second -- so the tag lands last,
        joined by a single space."""
        assert _with_audio_tag("Transcribe:", self.TAG, "suffix") == f"Transcribe: {self.TAG}"

    def test_prefix_puts_the_audio_first(self):
        assert _with_audio_tag("Transcribe:", self.TAG, "prefix") == f"{self.TAG} Transcribe:"

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


class TestMarkMaxTokens:
    """``mark_max_tokens``: which rows of a batch ran into ``max_new_tokens``."""

    EOS, PAD = 2, 0

    def _mark(self, rows, max_tokens):
        torch = pytest.importorskip("torch")
        from melteval.providers.base import _Request, mark_max_tokens

        batch = [
            _Request(text="p", audio=None, sample_rate=16000, config=GenerateConfig(max_tokens=max_tokens))
            for _ in rows
        ]
        mark_max_tokens(batch, torch.tensor(rows), {self.EOS, self.PAD})
        return [r.stop_reason for r in batch]

    def test_only_a_row_without_eos_or_padding_at_the_limit_was_cut(self):
        rows = [
            [5, 6, 7, 8],  # still going at the limit
            [5, 6, 7, self.EOS],  # ended on the very last token
            [5, self.EOS, self.PAD, self.PAD],  # ended early, padded to the batch
        ]
        assert self._mark(rows, max_tokens=4) == ["max_tokens", "stop", "stop"]

    def test_a_batch_shorter_than_the_limit_was_not_cut(self):
        assert self._mark([[5, 6, 7]], max_tokens=4) == ["stop"]

    def test_the_default_limit_applies_when_the_task_sets_none(self):
        from melteval.providers.base import DEFAULT_MAX_TOKENS

        assert self._mark([[5] * DEFAULT_MAX_TOKENS], max_tokens=None) == ["max_tokens"]


class TestStopIds:
    def test_special_tokens_and_the_models_eos_and_pad_are_all_included(self):
        import types

        from melteval.providers.base import stop_ids

        tokenizer = types.SimpleNamespace(all_special_ids=[1, 2])
        model = types.SimpleNamespace(generation_config=types.SimpleNamespace(eos_token_id=[3, 4], pad_token_id=5))
        assert stop_ids(tokenizer, model) == {1, 2, 3, 4, 5}


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


class _FakeQwenProcessor:
    """Enough of ``Qwen2AudioProcessor`` to drive ``_generate_batch`` without weights.

    The chat template tags the instruction so a test can see it was applied;
    the processor call encodes each prompt as ``len(text)`` ones, left-padded
    with zeros; decoding turns each generated token id back into a letter.
    """

    def __init__(self):
        import types

        self.feature_extractor = types.SimpleNamespace(sampling_rate=16000, chunk_length=30)
        self.tokenizer = types.SimpleNamespace(padding_side="right", all_special_ids=[0])
        self.calls: list[dict] = []
        self.conversations: list = []

    def apply_chat_template(self, conversation, add_generation_prompt, tokenize):
        assert add_generation_prompt and not tokenize
        self.conversations.append(conversation)
        [turn] = conversation
        text = next(c["text"] for c in turn["content"] if c["type"] == "text")
        return f"<tpl>{text}</tpl>"

    def __call__(self, text, return_tensors, padding, audio=None, sampling_rate=None, **kwargs):
        import torch

        self.calls.append({"text": text, "audio": audio, "sampling_rate": sampling_rate, "kwargs": kwargs})
        width = max(len(t) for t in text)
        input_ids = torch.tensor([[0] * (width - len(t)) + [1] * len(t) for t in text])
        return {"input_ids": input_ids, "input_features": torch.zeros(len(text), 2, dtype=torch.float32)}

    def batch_decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        return [" " + "".join(chr(ord("a") + int(i)) for i in row) + " " for row in ids]


class _FakeQwenModel:
    """Echoes the prompt followed by each row's index, as ``generate()`` from ``input_ids`` does."""

    def __init__(self):
        import torch

        self.dtype = torch.bfloat16
        self.seen: dict = {}

    def generate(self, input_ids, input_features=None, **kwargs):
        import torch

        self.seen = {"input_features": input_features, **kwargs}
        new = torch.arange(input_ids.shape[0]).unsqueeze(1)
        return torch.cat([input_ids, new], dim=1)


class TestQwen2Audio:
    """The provider's own logic, with a fake processor and model -- no weights."""

    def _run(self, batch):
        """Render and generate *batch* the way ``Qwen2AudioAPI._generate_batch`` does."""
        pytest.importorskip("torch")
        from melteval.providers.qwen2_audio import _generate, _render

        processor, model = _FakeQwenProcessor(), _FakeQwenModel()
        prompts = [_render(processor, r.text) for r in batch]
        return _generate(model, processor, "cpu", batch, prompts), processor, model

    def test_completions_are_the_new_tokens_only_in_request_order(self):
        """generate() here runs from input_ids, so the prompt comes back first
        and has to be cut off -- or every hypothesis starts with the chat template."""
        completions, _, _ = self._run([_QwenRequest(text="short"), _QwenRequest(text="a longer prompt")])

        assert completions == ["a", "b"]

    def test_the_models_template_wraps_the_instruction_with_one_audio(self):
        _, processor, _ = self._run([_QwenRequest(text="Transcribe this audio.")])

        [call] = processor.calls
        assert call["text"] == ["<tpl>Transcribe this audio.</tpl>"]
        assert len(call["audio"]) == 1
        [[turn]] = processor.conversations
        assert [c["type"] for c in turn["content"]] == ["audio", "text"]

    def test_features_are_cast_to_the_model_dtype(self):
        """A bf16 audio encoder fails on float32 features."""
        _, _, model = self._run([_QwenRequest()])

        assert model.seen["input_features"].dtype == model.dtype

    def test_generation_is_greedy_by_default(self):
        _, _, model = self._run([_QwenRequest()])

        assert model.seen["do_sample"] is False

    def test_audio_longer_than_the_window_is_rejected_not_truncated(self):
        """The processor pads to max_length and cuts the rest off: a 40 s clip
        would be transcribed as its first 30 s and scored as the whole thing."""
        with pytest.raises(ValueError, match="30 s window"):
            self._run([_QwenRequest(duration=10.0), _QwenRequest(duration=40.0)])

    def test_missing_audio_is_an_error(self):
        with pytest.raises(ValueError, match="no audio"):
            self._run([_QwenRequest(audio=None)])

    def test_a_completion_cut_at_max_tokens_is_marked(self):
        """The fake model emits one token per row: 0 (special, as eos) for the
        first, 1 for the second. At max_tokens=1 only the second was cut."""
        config = GenerateConfig(max_tokens=1)
        batch = [_QwenRequest(config=config), _QwenRequest(config=config)]
        self._run(batch)

        assert [r.stop_reason for r in batch] == ["stop", "max_tokens"]

    def test_nothing_is_marked_below_the_limit(self):
        batch = [_QwenRequest(), _QwenRequest()]
        self._run(batch)

        assert [r.stop_reason for r in batch] == ["stop", "stop"]

    def test_sample_rate_mismatch_is_an_error(self):
        with pytest.raises(ValueError, match="8000"):
            self._run([_QwenRequest(sample_rate=8000)])


class _FakeOmniModel(_FakeQwenModel):
    """Qwen3-Omni's ``generate()``: returns ``(token_ids, audio)`` and takes ``thinker_*`` kwargs."""

    def generate(self, input_ids, input_features=None, **kwargs):
        return super().generate(input_ids, input_features, **kwargs), None


class TestQwen3Omni:
    """Qwen3-Omni through the shared HF logic, with a fake processor and model -- no weights."""

    def _run(self, batch):
        pytest.importorskip("torch")
        from melteval.providers.qwen3_omni import _generate, _render

        processor, model = _FakeQwenProcessor(), _FakeOmniModel()
        prompts = [_render(processor, r.text) for r in batch]
        return _generate(model, processor, "cpu", batch, prompts), processor, model

    def test_the_token_ids_are_taken_from_the_tuple_and_the_prompt_stripped(self):
        completions, _, _ = self._run([_QwenRequest(text="short"), _QwenRequest(text="a longer prompt")])

        assert completions == ["a", "b"]

    def test_max_tokens_reaches_the_thinker(self):
        """A bare max_new_tokens loses to generate()'s own thinker_max_new_tokens=1024."""
        _, _, model = self._run([_QwenRequest(config=GenerateConfig(max_tokens=64))])

        assert model.seen["thinker_max_new_tokens"] == 64
        assert "max_new_tokens" not in model.seen
        assert model.seen["thinker_do_sample"] is False
        assert model.seen["return_audio"] is False

    def test_long_audio_is_not_rejected(self):
        """No fixed window: MCIF long talks go in whole."""
        completions, _, _ = self._run([_QwenRequest(duration=600.0)])

        assert completions == ["a"]

    def test_the_feature_extractor_is_told_not_to_truncate(self):
        """Its WhisperFeatureExtractor cuts to 30 s by default, silently (MCIF long, 2026-09-27)."""
        _, processor, _ = self._run([_QwenRequest(duration=600.0)])

        assert processor.calls[0]["kwargs"] == {"truncation": False}

    def test_missing_audio_is_an_error(self):
        with pytest.raises(ValueError, match="Qwen3-Omni.*no audio|no audio"):
            self._run([_QwenRequest(audio=None)])


class TestTextOnlyJudge:
    """``text_only``: an HF baseline serving a judge prompt, with no audio anywhere."""

    def _run(self, batch):
        pytest.importorskip("torch")
        from melteval.providers.hf import generate, render

        processor, model = _FakeQwenProcessor(), _FakeOmniModel()
        prompts = [render(processor, r.text, with_audio=False) for r in batch]
        completions = generate(
            model, processor, "cpu", batch, prompts, gen_kwargs={}, family="Qwen3-Omni",
            sequences=lambda output: output[0], with_audio=False,
        )
        return completions, processor

    def test_the_prompt_has_no_audio_slot_and_the_processor_gets_no_audio(self):
        completions, processor = self._run([_QwenRequest(text="Grade this answer.", audio=None)])

        assert completions == ["a"]
        [[turn]] = processor.conversations
        assert [c["type"] for c in turn["content"]] == ["text"]
        assert processor.calls[0]["audio"] is None

    def test_a_request_with_audio_is_refused(self):
        """A judge that silently dropped audio would hide a misrouted speech sample."""
        with pytest.raises(ValueError, match="text_only"):
            self._run([_QwenRequest()])


class TestTruncateLongAudio:
    """``truncate_long_audio``: an explicit opt-in to the processor's own cut, recorded per sample."""

    def _run(self, batch):
        pytest.importorskip("torch")
        from melteval.providers.hf import generate, render

        processor, model = _FakeQwenProcessor(), _FakeQwenModel()
        prompts = [render(processor, r.text) for r in batch]
        completions = generate(
            model, processor, "cpu", batch, prompts, gen_kwargs={}, family="Qwen2-Audio",
            max_seconds=30, truncate=True,
        )
        return completions, processor

    def _long_and_short(self):
        import numpy as np

        long = _QwenRequest(audio=np.zeros(40 * 16000, dtype=np.float32), duration=40.0)
        short = _QwenRequest(audio=np.zeros(5 * 16000, dtype=np.float32), duration=5.0)
        return long, short

    def test_a_long_clip_is_cut_to_the_window_and_the_rest_left_alone(self):
        long, short = self._long_and_short()
        completions, processor = self._run([long, short])

        assert completions == ["a", "b"]
        assert [len(a) for a in processor.calls[0]["audio"]] == [30 * 16000, 5 * 16000]

    def test_the_cut_is_recorded_on_that_sample_only(self):
        """So the log says which hypotheses cover only part of their reference."""
        long, short = self._long_and_short()
        self._run([long, short])

        assert long.metadata == {"audio_truncated_from_seconds": 40.0}
        assert short.metadata == {}

    def test_without_the_option_a_long_clip_is_still_refused(self):
        from melteval.providers.qwen2_audio import _generate, _render

        pytest.importorskip("torch")
        long, _ = self._long_and_short()
        processor, model = _FakeQwenProcessor(), _FakeQwenModel()
        with pytest.raises(ValueError, match="30 s window"):
            _generate(model, processor, "cpu", [long], [_render(processor, long.text)])


@dataclass
class _QwenRequest:
    """Stands in for `_Request`, with the duration the length check reads."""

    text: str = "Transcribe this audio."
    audio: object = field(default_factory=lambda: [0.0] * 4)
    sample_rate: int = 16000
    duration: float = 1.0
    config: GenerateConfig = field(default_factory=GenerateConfig)
    metadata: dict = field(default_factory=dict)
    stop_reason: str = "stop"
