"""Tests for the model providers' message handling and config translation.

That is, extracting prompts and audio locators from messages and converting
``GenerateConfig`` into generation kwargs.
"""

import sys
import types
from dataclasses import dataclass, field

import pytest
from inspect_ai.model import ChatMessageUser, ContentAudio, ContentData, ContentText, GenerateConfig

from melteval.dataset import AUDIO_DATA_KEY
from melteval.providers import smurf
from melteval.providers.base import _extract
from melteval.providers.melt import _batched_audio, _generate_kwargs
from melteval.providers.smurf import (
    _check_system_placement,
    _collate_audio,
    _generation_kwargs,
    _merge_config,
    _plan_device_map,
    _speechllm_class,
    _turns,
    _with_audio_tag,
)


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


class TestSmurfTurns:
    """A system prompt is a turn of its own, as SMURF's training dataloader builds it."""

    def test_without_a_system_prompt_the_conversation_is_one_user_turn(self):
        assert _turns("Transcribe: <tag>") == [{"role": "user", "content": "Transcribe: <tag>"}]

    def test_a_system_prompt_goes_first_and_is_not_merged_into_the_instruction(self):
        assert _turns("Transcribe: <tag>", "Respond directly.") == [
            {"role": "system", "content": "Respond directly."},
            {"role": "user", "content": "Transcribe: <tag>"},
        ]

    def test_an_empty_system_prompt_sends_no_system_turn(self):
        assert _turns("Transcribe: <tag>", "") == [{"role": "user", "content": "Transcribe: <tag>"}]

    def test_user_placement_opens_the_user_turn_with_the_system_prompt(self):
        """For a prompt_format with no system role, which would reject a system turn."""
        assert _turns("Transcribe: <tag>", "Respond directly.", "user") == [
            {"role": "user", "content": "Respond directly.\n\nTranscribe: <tag>"},
        ]

    def test_user_placement_without_a_system_prompt_leaves_the_turn_alone(self):
        assert _turns("Transcribe: <tag>", None, "user") == [{"role": "user", "content": "Transcribe: <tag>"}]


class TestSmurfSystemPlacement:
    """A system turn the checkpoint's prompt format cannot render is refused at load, not mid-eval."""

    QWEN = {"user": {}, "assistant": {}}
    EUROLLM = {"system": {}, "user": {}, "assistant": {}}

    def test_a_system_turn_on_a_format_without_a_system_role_is_refused(self):
        with pytest.raises(ValueError, match="system_placement=user"):
            _check_system_placement(self.QWEN, "qwen", "Respond directly.", "turn")

    @pytest.mark.parametrize(
        ("template", "system_prompt", "placement"),
        [
            (QWEN, "Respond directly.", "user"),
            (QWEN, None, "turn"),
            (EUROLLM, "Respond directly.", "turn"),
        ],
    )
    def test_what_the_format_can_render_passes(self, template, system_prompt, placement):
        _check_system_placement(template, "fmt", system_prompt, placement)


class TestSmurfSpeechllmClass:
    """The project's prompt formats are registered before loading, as fbk_speechllm.inference does."""

    @staticmethod
    def _fake_fbk(monkeypatch, *, with_prompts: bool) -> list[str]:
        calls: list[str] = []
        speech_llm = types.ModuleType("fbk_speechllm.models.speech_llm")
        speech_llm.SpeechLLM = type("SpeechLLM", (), {})
        utils = types.ModuleType("fbk_speechllm.utils")
        utils.register_custom_perception_modules = lambda: calls.append("perception")
        prompts = None
        if with_prompts:
            prompts = types.ModuleType("fbk_speechllm.prompts")
            prompts.register_custom_prompt_formats = lambda: calls.append("prompts")
        for name, module in [
            ("fbk_speechllm", types.ModuleType("fbk_speechllm")),
            ("fbk_speechllm.models", types.ModuleType("fbk_speechllm.models")),
            ("fbk_speechllm.models.speech_llm", speech_llm),
            ("fbk_speechllm.utils", utils),
            ("fbk_speechllm.prompts", prompts),  # None makes the import fail
        ]:
            monkeypatch.setitem(sys.modules, name, module)
        return calls

    def test_both_registries_are_filled_before_the_class_is_returned(self, monkeypatch):
        calls = self._fake_fbk(monkeypatch, with_prompts=True)
        assert _speechllm_class().__name__ == "SpeechLLM"
        assert sorted(calls) == ["perception", "prompts"]

    def test_an_fbk_speechllm_without_prompts_is_refused_rather_than_swapped_for_salm(self, monkeypatch):
        """SALM would load, and then fail on the first batch over an unknown prompt format."""
        self._fake_fbk(monkeypatch, with_prompts=False)
        with pytest.raises(ImportError, match="fbk_speechllm.prompts"):
            _speechllm_class()


class TestSmurfConfigOverride:
    """model_config_override is merged onto the checkpoint's stored config, not swapped in for it."""

    STORED = {
        "cfg": {
            "pretrained_llm": "utter-project/EuroLLM-9B-Instruct-2512",
            "prompt_format": "qwen",
            "freeze_params": ["^llm\\..+$", "^embed_tokens\\..+$"],
            "perception": {"output_dim": 4096, "modality_adapter": {"hidden_dim": 5120, "num_layers": 6}},
        }
    }

    def test_mappings_merge_and_everything_else_is_replaced(self):
        merged = _merge_config(
            self.STORED["cfg"],
            {"freeze_params": ["^llm\\..+$"], "perception": {"modality_adapter": {"hidden_dim": 7424}}},
        )
        assert merged["freeze_params"] == ["^llm\\..+$"]
        assert merged["perception"] == {"output_dim": 4096, "modality_adapter": {"hidden_dim": 7424, "num_layers": 6}}
        assert merged["prompt_format"] == "qwen"

    def test_only_what_the_override_names_changes(self, tmp_path, monkeypatch):
        torch = pytest.importorskip("torch")
        pytest.importorskip("lightning")
        ckpt = tmp_path / "step=1000.ckpt"
        torch.save({"state_dict": {}, "hyper_parameters": self.STORED}, ckpt)
        override = tmp_path / "override.yaml"
        override.write_text("cfg:\n  prompt_format: eurollm\n")

        loaded: dict = {}

        class FakeSpeechLLM:
            @classmethod
            def load_from_checkpoint(cls, path, **kwargs):
                loaded.update(kwargs)

        monkeypatch.setattr(smurf, "_speechllm_class", lambda: FakeSpeechLLM)
        smurf._load_speechllm(str(ckpt), model_config_override=str(override))

        expected = _merge_config(self.STORED["cfg"], {"prompt_format": "eurollm"})
        assert loaded["cfg"] == expected
        assert loaded["cfg"]["perception"]["modality_adapter"]["hidden_dim"] == 5120

    def test_a_checkpoint_without_stored_hparams_is_refused(self, tmp_path, monkeypatch):
        torch = pytest.importorskip("torch")
        pytest.importorskip("lightning")
        ckpt = tmp_path / "bare.ckpt"
        torch.save({"state_dict": {}}, ckpt)
        override = tmp_path / "override.yaml"
        override.write_text("cfg:\n  prompt_format: eurollm\n")
        monkeypatch.setattr(smurf, "_speechllm_class", lambda: object)
        with pytest.raises(ValueError, match="no hyperparameters"):
            smurf._load_speechllm(str(ckpt), model_config_override=str(override))


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


class _FakeQwenProcessor:
    """Enough of ``Qwen2AudioProcessor`` to drive ``_generate_batch`` without weights.

    The chat template tags the instruction so a test can see it was applied;
    the processor call encodes each prompt as ``len(text)`` ones, left-padded
    with zeros; decoding turns each generated token id back into a letter.
    """

    def __init__(self):
        import types

        self.feature_extractor = types.SimpleNamespace(sampling_rate=16000, chunk_length=30)
        self.tokenizer = types.SimpleNamespace(padding_side="right")
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

    def test_the_window_is_the_processors(self):
        """The refusal past it is tested in TestAudioPastTheWindow."""
        from melteval.providers.qwen2_audio import Qwen2AudioAPI

        api = object.__new__(Qwen2AudioAPI)  # bypass __init__, which loads a checkpoint
        api.processor = _FakeQwenProcessor()

        assert api._max_audio_seconds() == 30

    def test_missing_audio_is_an_error(self):
        with pytest.raises(ValueError, match="no audio"):
            self._run([_QwenRequest(audio=None)])

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


class TestAudioPastTheWindow:
    """A clip past the model's window fails that sample alone, loudly, before it is batched."""

    def _api(self, limit):
        from melteval.providers.hf import HFSpeechChatAPI

        # Bypass __init__ (it loads a checkpoint): _reject reads only these.
        api = object.__new__(HFSpeechChatAPI)
        api.family, api.text_only = "Qwen2-Audio", False
        api._max_audio_seconds = lambda: limit
        return api

    def test_a_clip_past_the_window_is_refused_with_its_length(self):
        reason = self._api(30)._reject(_QwenRequest(duration=40.0))
        assert "40.0 s" in reason and "30 s window" in reason

    def test_a_clip_within_the_window_or_a_model_without_one_is_served(self):
        assert self._api(30)._reject(_QwenRequest(duration=30.0)) is None
        assert self._api(None)._reject(_QwenRequest(duration=600.0)) is None

    def test_generate_fails_only_the_long_sample_and_logs_it(self, monkeypatch, caplog):
        import anyio
        import numpy as np
        from inspect_ai.model import ChatMessageUser, ContentData, ContentText

        from melteval.dataset import AUDIO_DATA_KEY
        from melteval.providers import base

        api = self._api(30)
        api._ensure_worker = lambda: None
        api._queue = base.Queue()
        monkeypatch.setattr(base, "_load_audio", lambda locator: (np.zeros(40 * 16000, dtype=np.float32), 16000))
        message = ChatMessageUser(
            content=[ContentText(text="Transcribe."), ContentData(data={AUDIO_DATA_KEY: {"path": "x"}})]
        )

        with pytest.raises(base.AudioTooLongError, match="40.0 s"):
            anyio.run(lambda: api.generate([message], [], "none", GenerateConfig()))
        assert api._queue.empty()  # never queued, so no batch is affected
        assert "SAMPLE SKIPPED" in caplog.text


class TestWhisper:
    """Whisper's batch logic with a fake processor and model -- no weights."""

    class _Processor:
        def __init__(self):
            import types

            self.feature_extractor = types.SimpleNamespace(sampling_rate=16000)
            self.calls: list[dict] = []

        def __call__(self, audios, sampling_rate, return_tensors, **kwargs):
            import torch

            self.calls.append({"n": len(audios), "kwargs": kwargs})
            return {"input_features": torch.zeros(len(audios), 2, dtype=torch.float32)}

        def batch_decode(self, ids, skip_special_tokens):
            assert skip_special_tokens
            return [f" text{int(i)} " for i in ids]

    class _Model:
        def __init__(self):
            import torch

            self.dtype = torch.bfloat16
            self.seen: dict = {}

        def generate(self, input_features, **kwargs):
            import torch

            self.seen = {"input_features": input_features, **kwargs}
            return torch.arange(input_features.shape[0])

    def _run(self, batch, task="transcribe"):
        pytest.importorskip("torch")
        from melteval.providers.whisper import _transcribe

        processor, model = self._Processor(), self._Model()
        return _transcribe(model, processor, "cpu", batch, task=task), processor, model

    def test_completions_come_back_in_order_and_stripped(self):
        completions, _, _ = self._run([_QwenRequest(), _QwenRequest()])
        assert completions == ["text0", "text1"]

    def test_the_task_is_passed_and_language_is_left_to_detection(self):
        _, _, model = self._run([_QwenRequest()], task="translate")
        assert model.seen["task"] == "translate"
        assert "language" not in model.seen
        assert model.seen["do_sample"] is False

    def test_short_audio_takes_the_plain_path(self):
        _, processor, model = self._run([_QwenRequest(duration=29.0)])
        assert processor.calls[0]["kwargs"] == {}
        assert "return_timestamps" not in model.seen

    def test_audio_past_30_s_is_decoded_long_form_not_cut(self):
        """The processor must not truncate, and generate() must walk the windows."""
        _, processor, model = self._run([_QwenRequest(duration=5.0), _QwenRequest(duration=95.0)])
        assert processor.calls[0]["kwargs"]["truncation"] is False
        assert processor.calls[0]["kwargs"]["return_attention_mask"] is True
        assert model.seen["return_timestamps"] is True

    def test_an_unknown_task_is_refused(self):
        from melteval.providers.whisper import WhisperAPI

        with pytest.raises(ValueError, match="task must be one of"):
            WhisperAPI("melt/hf/whisper", task="detect")


class TestDesta25:
    """DeSTA2.5's request handling with a fake model -- the real one needs the `desta` package."""

    class _Model:
        def __init__(self):
            self.calls: list[dict] = []

        def generate(self, messages, **kwargs):
            import types
            from pathlib import Path

            texts = []
            for system, turn in messages:  # a list of conversations: one batched call
                assert system["role"] == "system"
                # The WAV must exist while generate() runs: desta reads it from the path.
                assert Path(turn["audios"][0]["audio"]).exists()
                self.calls.append({"system": system["content"], "turn": turn, **kwargs})
                texts.append(f" answer{len(self.calls)} ")
            return types.SimpleNamespace(text=texts)

    def _run(self, batch):
        pytest.importorskip("soundfile")
        from melteval.providers.desta25 import _generate

        model = self._Model()
        return _generate(model, batch), model

    def _request(self, text="Transcribe this audio.", **kw):
        import numpy as np

        return _QwenRequest(text=text, audio=np.zeros(1600, dtype=np.float32), **kw)

    def test_a_batch_is_one_call_with_completions_in_order(self):
        completions, model = self._run([self._request("a"), self._request("b")])
        assert completions == ["answer1", "answer2"]
        assert [c["turn"]["content"] for c in model.calls] == ["<|AUDIO|>\na", "<|AUDIO|>\nb"]

    def test_every_request_carries_the_authors_default_system_prompt(self):
        from melteval.providers.desta25 import DEFAULT_SYSTEM_PROMPT

        _, model = self._run([self._request()])
        assert model.calls[0]["system"] == DEFAULT_SYSTEM_PROMPT == "Focus on the audio clips and instruction."

    def test_numpy_2_gets_the_sctypes_desta_reads(self, monkeypatch):
        """desta tests `samples.dtype in np.sctypes['int']`, removed in NumPy 2."""
        import numpy as np

        from melteval.providers.desta25 import _restore_numpy_sctypes

        monkeypatch.delattr(np, "sctypes", raising=False)
        _restore_numpy_sctypes()
        assert np.dtype("int16") in np.sctypes["int"]
        assert np.dtype("float32") in np.sctypes["float"]
        monkeypatch.delattr(np, "sctypes")

    def test_a_clip_past_30_seconds_is_refused_on_its_own(self):
        from melteval.providers.desta25 import Desta25API

        api = Desta25API.__new__(Desta25API)  # _reject reads nothing from __init__
        assert "30 s window" in api._reject(_QwenRequest(duration=40.0))
        assert api._reject(_QwenRequest(duration=30.0)) is None

    def test_missing_audio_is_an_error(self):
        pytest.importorskip("soundfile")
        from melteval.providers.desta25 import _generate

        with pytest.raises(ValueError, match="no audio"):
            _generate(self._Model(), [_QwenRequest(audio=None)])


@dataclass
class _QwenRequest:
    """Stands in for `_Request`, with the duration the length check reads."""

    text: str = "Transcribe this audio."
    audio: object = field(default_factory=lambda: [0.0] * 4)
    sample_rate: int = 16000
    duration: float = 1.0
    config: GenerateConfig = field(default_factory=GenerateConfig)
    metadata: dict = field(default_factory=dict)


class TestRouter:
    """``melt/<name>``: which class serves it, loaded from which weights -- no weights loaded here."""

    @pytest.fixture
    def built(self, monkeypatch):
        """Replace every class the router can build with a recorder of its arguments."""
        from melteval.providers import router

        calls = []

        def recorder(cls_name, repo=None, revision=None):
            class Recorder:
                def __init__(self, model_name, base_url, api_key, config, **kwargs):
                    calls.append((cls_name, model_name, kwargs))

            Recorder.repo, Recorder.revision = repo, revision
            return Recorder

        monkeypatch.setattr(router, "HF_MODELS", {"qwen2_audio": recorder("qwen2_audio", "Qwen/Q2", "abc123")})
        monkeypatch.setattr(router, "MELTAPI", recorder("melt"))
        return calls

    def test_hf_loads_the_pinned_revision_and_keeps_its_short_name(self, built):
        """The log says `melt/hf/qwen2_audio`; the weights come from repo@revision."""
        from melteval.providers.router import create_model

        create_model("hf/qwen2_audio", batch_size=4)
        assert built == [("qwen2_audio", "hf/qwen2_audio", {"path": "Qwen/Q2", "revision": "abc123", "batch_size": 4})]

    def test_revision_and_path_override_the_pin(self, built):
        from melteval.providers.router import create_model

        create_model("hf/qwen2_audio", revision="def456")
        create_model("hf/qwen2_audio", path="/local/finetune")
        assert [call[2] for call in built] == [
            {"path": "Qwen/Q2", "revision": "def456"},
            {"path": "/local/finetune", "revision": None},
        ]

    def test_anything_else_is_a_melt_checkpoint(self, built):
        from melteval.providers.router import create_model

        create_model("/abs/run/checkpoint-1000")
        assert built == [("melt", "/abs/run/checkpoint-1000", {})]

    def test_unknown_hf_model_and_vllm_are_errors(self, built):
        from melteval.providers.router import create_model

        with pytest.raises(ValueError, match="known: melt/hf/qwen2_audio"):
            create_model("hf/nonexistent")
        with pytest.raises(ValueError, match="not implemented"):
            create_model("vllm/some/model")

    def test_inspect_resolves_melt_to_the_router(self, built):
        """Through inspect's own model lookup, as `--model melt/hf/...` is."""
        from inspect_ai.model import get_model

        import melteval  # noqa: F401 -- registers the provider

        get_model("melt/hf/qwen2_audio", memoize=False)
        assert built[0][:2] == ("qwen2_audio", "hf/qwen2_audio")
