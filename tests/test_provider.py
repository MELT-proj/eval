"""Tests for the MELT model provider's message handling and config translation.

These exercise the parts that don't need a loaded model: extracting the prompt
and audio locator from a message, and translating an inspect ``GenerateConfig``
into ``generate()`` kwargs. The batching thread and the actual forward pass are
covered by the end-to-end smoke test against a real checkpoint instead, since
faking a transformers model would test the fake more than the code.
"""

from inspect_ai.model import ChatMessageUser, ContentAudio, ContentData, ContentText, GenerateConfig

from melteval.dataset import AUDIO_DATA_KEY
from melteval.providers.melt import _extract, _generate_kwargs


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
