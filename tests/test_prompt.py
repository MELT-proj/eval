"""Tests for prompt and chat-format parity with training.

The parity test is the point of this file. Everything else here supports it.
"""

import pytest

from melteval.prompt import (
    FormatSpec,
    apply_generation_format,
    find_training_config,
    load_format_spec,
    render_user_prompt,
    select_template,
)


melt = pytest.importorskip("melt.training.data.audio.lhotse.helpers")


def _spec(**overrides) -> FormatSpec:
    defaults = {
        "apply_chat_template": True,
        "prompt_template": None,
        "prompt_template_selection": "random",
        "chat_template_config": "chatml",
        "audio_token": "<|audio|>",
        "source": "test",
    }
    defaults.update(overrides)
    return FormatSpec(**defaults)


class ChatMLTokenizer:
    """A stand-in with ChatML semantics, so parity is tested without a download."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        rendered = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
        return rendered


class TestFormatParity:
    """Eval must send the same bytes training did, up to the assistant turn."""

    @pytest.mark.parametrize(
        ("task", "lang", "text"),
        [
            ("asr", "de", "Guten Tag."),
            ("asr", "en", "Hello there."),
            ("st", "en", "Hello, do you work here?"),
        ],
    )
    def test_eval_prompt_is_a_prefix_of_the_training_string(self, task, lang, text):
        """The operational meaning of "the same format".

        Training renders [user, assistant] with add_generation_prompt=False.
        Generation renders [user] with add_generation_prompt=True. The second
        must be exactly the first truncated at the assistant boundary; anything
        else is a distribution shift dressed up as a metric.
        """
        from melt.training.data.audio.lhotse.helpers import apply_chat_template_to_texts
        from melt.training.data.chat_templates import CHAT_TEMPLATE_CONFIGS

        tokenizer = ChatMLTokenizer()
        template = "{audio_token} Transcribe this audio in {lang}."
        spec = _spec(prompt_template_selection="custom", prompt_template=template)

        training = apply_chat_template_to_texts(
            texts=[text],
            tasks=[task],
            langs=[lang],
            tokenizer=tokenizer,
            audio_token=spec.audio_token,
            prompt_template=template,
            prompt_template_selection="custom",
        )[0]

        prompt = render_user_prompt(task=task, sample_key="00-000000", spec=spec, lang=lang)
        evaluation = apply_generation_format(prompt, spec, tokenizer)

        assert training.startswith(evaluation)
        boundary = CHAT_TEMPLATE_CONFIGS[spec.chat_template_config].assistant_start
        assert evaluation.endswith(boundary)

    def test_without_chat_template_the_prompt_is_sent_raw(self):
        spec = _spec(apply_chat_template=False, prompt_template_selection="custom",
                     prompt_template="{audio_token} Transcribe.")
        prompt = render_user_prompt(task="asr", sample_key="k", spec=spec, lang="en")
        assert apply_generation_format(prompt, spec, None) == prompt

    def test_missing_chat_template_support_raises(self):
        spec = _spec(prompt_template_selection="custom", prompt_template="{audio_token} x")
        with pytest.raises(ValueError, match="apply_chat_template"):
            apply_generation_format("x", spec, object())


class TestTemplateSelection:
    def test_selection_is_deterministic_in_the_sample_key(self):
        """Two runs of one frozen set must draw the same template per sample."""
        spec = _spec()
        first = [select_template("asr", f"00-{i:06d}", spec) for i in range(20)]
        second = [select_template("asr", f"00-{i:06d}", spec) for i in range(20)]
        assert first == second

    def test_selection_still_varies_across_samples(self):
        """Determinism must not collapse the pool to a single template."""
        spec = _spec()
        chosen = {select_template("asr", f"00-{i:06d}", spec) for i in range(40)}
        assert len(chosen) > 1

    def test_with_language_only_picks_language_templates(self):
        spec = _spec(prompt_template_selection="with_language")
        for i in range(20):
            assert "{lang}" in select_template("asr", f"00-{i:06d}", spec)

    def test_custom_string_template_wins(self):
        spec = _spec(prompt_template_selection="custom", prompt_template="{audio_token} Go.")
        assert select_template("asr", "k", spec) == "{audio_token} Go."

    def test_custom_dict_selects_by_task(self):
        spec = _spec(
            prompt_template_selection="custom",
            prompt_template={"asr": "{audio_token} A", "st": "{audio_token} S"},
        )
        assert select_template("st", "k", spec) == "{audio_token} S"

    def test_unknown_task_names_the_alternative(self):
        with pytest.raises(ValueError, match="instruction"):
            select_template("summarisation", "k", _spec())

    def test_unknown_selection_strategy_raises(self):
        with pytest.raises(ValueError, match="prompt_template_selection"):
            select_template("asr", "k", _spec(prompt_template_selection="best"))


class TestRendering:
    def test_audio_token_and_language_are_interpolated(self):
        spec = _spec(
            prompt_template_selection="custom",
            prompt_template="{audio_token} Transcribe this audio in {lang}.",
        )
        rendered = render_user_prompt(task="asr", sample_key="k", spec=spec, lang="de")
        assert rendered == "<|audio|> Transcribe this audio in German."

    def test_translation_directions_are_interpolated(self):
        spec = _spec(
            prompt_template_selection="custom",
            prompt_template="{audio_token} {src_lang} to {tgt_lang}.",
        )
        rendered = render_user_prompt(
            task="st", sample_key="k", spec=spec, lang="en", src_lang="ar", tgt_lang="en"
        )
        assert rendered == "<|audio|> Arabic to English."

    def test_per_sample_instruction_overrides_the_pool(self):
        """Benchmarks that ship their own prompts bypass the template pool."""
        rendered = render_user_prompt(
            task="summarisation",
            sample_key="k",
            spec=_spec(),
            instruction="{audio_token} Summarise the talk.",
        )
        assert rendered == "<|audio|> Summarise the talk."

    def test_unsupported_language_raises_as_in_training(self):
        spec = _spec(prompt_template_selection="custom", prompt_template="{audio_token} in {lang}")
        with pytest.raises(ValueError, match="Unsupported language"):
            render_user_prompt(task="asr", sample_key="k", spec=spec, lang="xx")

    def test_language_free_template_tolerates_an_unknown_code(self):
        """Only templates that actually interpolate {lang} need a known code."""
        spec = _spec(prompt_template_selection="custom", prompt_template="{audio_token} Transcribe.")
        assert render_user_prompt(task="asr", sample_key="k", spec=spec, lang="xx")


class TestFormatSpecLoading:
    def test_reads_the_four_format_keys_and_the_audio_token(self, tmp_path):
        (tmp_path / "training_config.yaml").write_text(
            "model:\n  decoder:\n    audio_token: <|aud|>\n"
            "data:\n"
            "  apply_chat_template: true\n"
            "  prompt_template_selection: with_language\n"
            "  chat_template_config: chatml\n"
        )
        spec = load_format_spec(tmp_path)
        assert spec.apply_chat_template is True
        assert spec.prompt_template_selection == "with_language"
        assert spec.audio_token == "<|aud|>"

    def test_found_from_a_checkpoint_subdirectory(self, tmp_path):
        """Checkpoints live under the run dir; the config lives at its root."""
        (tmp_path / "training_config.yaml").write_text("data: {}\n")
        checkpoint = tmp_path / "checkpoint-1200"
        checkpoint.mkdir()
        assert find_training_config(checkpoint) == tmp_path / "training_config.yaml"

    def test_missing_config_refuses_to_guess(self, tmp_path):
        """Silently defaulting the format is exactly how training#58 happened."""
        with pytest.raises(FileNotFoundError, match="--format-config"):
            load_format_spec(tmp_path)
