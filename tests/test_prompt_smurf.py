"""Tests for the SMURF prompt handling in the melteval package."""

import pytest

from melteval.prompt import load_smurf_prompt_spec, render_smurf_prompt


try:  # the language-name table lives in the training package
    from melt.training.data.audio.lhotse.helpers import LANGUAGE_ISO_TO_NAME  # noqa: F401

    HAS_MELT = True
except ImportError:  # pragma: no cover - depends on the environment under test
    HAS_MELT = False


ASR_INFERENCE_YAML = """
model_path: /models/model.ckpt
output_file: /out/generations.jsonl
data:
  test_ds:
    prompt_format: qwen
    token_equivalent_duration: 0.08
    input_cfg:
      - type: lhotse_as_conversation
        cuts_path: /data/test.jsonl.gz
        audio_locator_tag: "<|audioplaceholder|>"
        tags:
          context: "Transcribe this English audio: "
    batch_size: 4
generation:
  max_new_tokens: 128
"""


class TestLoadSmurfPromptSpec:
    """The instruction comes from the config SMURF itself prompts with."""

    def test_reads_the_context_tag_and_the_identifiers_beside_it(self, tmp_path):
        config = tmp_path / "asr_inference.yaml"
        config.write_text(ASR_INFERENCE_YAML, encoding="utf-8")

        spec = load_smurf_prompt_spec(config)

        assert spec.instruction == "Transcribe this English audio: "
        assert spec.prompt_format == "qwen"
        assert spec.audio_locator_tag == "<|audioplaceholder|>"
        assert spec.source == str(config)

    def test_finds_the_context_under_the_validation_ds_nesting(self, tmp_path):
        """Training configs nest input_cfg one level deeper than inference ones;
        both are legitimate places to read the trained prompt from."""
        config = tmp_path / "training.yaml"
        config.write_text(
            "data:\n"
            "  validation_ds:\n"
            "    datasets:\n"
            "      val_set_0:\n"
            "        input_cfg:\n"
            "          - type: lhotse_as_conversation\n"
            "            tags:\n"
            "              context: 'Translate into German: '\n",
            encoding="utf-8",
        )

        assert load_smurf_prompt_spec(config).instruction == "Translate into German: "

    def test_no_context_tag_is_a_valid_config(self, tmp_path):
        """MCIF-style inference carries the prompt per sample instead."""
        config = tmp_path / "mcif.yaml"
        config.write_text(
            "data:\n  test_ds:\n    prompt_format: qwen\n    input_cfg:\n      - type: lhotse_as_conversation\n",
            encoding="utf-8",
        )

        spec = load_smurf_prompt_spec(config)
        assert spec.instruction is None
        assert spec.prompt_format == "qwen"

    def test_two_different_contexts_is_an_error(self, tmp_path):
        """One run sends one prompt; picking one of two silently would put the
        result's prompt out of reach of anyone reading the log."""
        config = tmp_path / "mixed.yaml"
        config.write_text(
            "data:\n"
            "  test_ds:\n"
            "    input_cfg:\n"
            "      - tags: {context: 'Transcribe: '}\n"
            "      - tags: {context: 'Translate: '}\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="different context prompts"):
            load_smurf_prompt_spec(config)

    def test_missing_config_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_smurf_prompt_spec(tmp_path / "absent.yaml")


class TestRenderSmurfPrompt:
    def test_a_literal_instruction_passes_through(self):
        """SMURF's own configs write the language into the string; that case
        must not require any table or lookup."""
        text = "Transcribe this English audio: "
        assert render_smurf_prompt(text, task="asr", lang="en") == text

    def test_iso_placeholders_need_no_language_table(self):
        rendered = render_smurf_prompt(
            "{task}: {src_lang_code}->{tgt_lang_code}", task="st", src_lang="en", tgt_lang="de"
        )
        assert rendered == "st: en->de"

    def test_unknown_placeholder_is_an_error(self):
        """Better than a prompt with a literal {speaker} in it, which every
        sample would still be scored against."""
        with pytest.raises(ValueError, match="Unknown placeholder"):
            render_smurf_prompt("Transcribe {speaker}: ", task="asr", lang="en")

    @pytest.mark.skipif(not HAS_MELT, reason="needs the training package's language table")
    def test_language_name_matches_the_one_melt_prompts_use(self):
        assert render_smurf_prompt("Transcribe this {lang} audio: ", lang="en") == (
            "Transcribe this English audio: "
        )

    @pytest.mark.skipif(not HAS_MELT, reason="needs the training package's language table")
    def test_unsupported_language_code_is_an_error(self):
        with pytest.raises(ValueError, match="Unsupported language ISO code"):
            render_smurf_prompt("Transcribe this {lang} audio: ", lang="zz")

    @pytest.mark.skipif(HAS_MELT, reason="table available, so the fallback path is not reachable")
    def test_language_name_without_the_table_says_what_to_do_instead(self):
        with pytest.raises(ValueError, match="lang_code"):
            render_smurf_prompt("Transcribe this {lang} audio: ", lang="en")


class TestStripAudioToken:
    """AIR-Bench/MCIF instructions mark MELT's audio position; the bare-instruction path drops it."""

    def test_air_bench_default_template(self):
        from melteval.prompt import strip_audio_token

        assert strip_audio_token("{audio_token}\nWhat is the emotion?") == ("What is the emotion?", True)

    def test_placeholder_mid_text_keeps_the_surrounding_words(self):
        from melteval.prompt import strip_audio_token

        assert strip_audio_token("Listen: {audio_token}\nQ: why?") == ("Listen: Q: why?", True)

    def test_escaped_braces_are_not_the_placeholder(self):
        """Corpus text is brace-escaped on the way in; a question that literally
        mentions {audio_token} must survive rendering as text."""
        from melteval.prompt import strip_audio_token

        template = "{audio_token}\nWhat does {{audio_token}} mean?"
        stripped, removed = strip_audio_token(template)
        assert removed
        assert render_smurf_prompt(stripped) == "What does {audio_token} mean?"

    def test_a_template_without_it_is_untouched(self):
        from melteval.prompt import strip_audio_token

        assert strip_audio_token("Transcribe this English audio: ") == ("Transcribe this English audio: ", False)


class TestPromptStyleSelection:
    """`-T prompt_style=` is how a run picks its prompt path, so its mistakes
    have to be loud."""

    def test_unknown_style_is_an_error(self):
        from melteval.tasks import _prompt_solver

        with pytest.raises(ValueError, match="Unknown prompt_style"):
            _prompt_solver("nemo", None, None, None, None)

    def test_an_argument_from_the_other_style_is_rejected_not_ignored(self):
        """A dropped -T looks exactly like an honoured one in the log, and the
        whole point of naming a config is knowing which one shaped the result."""
        from melteval.tasks import _prompt_solver

        with pytest.raises(ValueError, match="format_config"):
            _prompt_solver("smurf", "/some/training_config.yaml", None, "Transcribe: ", None)

        with pytest.raises(ValueError, match="instruction"):
            _prompt_solver("melt", None, None, "Transcribe: ", None)

    def test_smurf_style_builds_the_smurf_solver(self):
        from melteval.tasks import _prompt_solver

        assert _prompt_solver("smurf", None, None, "Transcribe: ", None) is not None

    def test_two_sources_for_one_instruction_is_an_error(self, tmp_path):
        """One would win and the log would name the other."""
        from melteval.tasks import _prompt_solver

        config = tmp_path / "asr_inference.yaml"
        config.write_text(ASR_INFERENCE_YAML, encoding="utf-8")
        with pytest.raises(ValueError, match="not both"):
            _prompt_solver("smurf", None, None, "Transcribe: ", str(config))


class TestAsrNormalizerInASmurfOnlyEnvironment:
    """A SMURF environment has no ``melt-proj`` installed at all (see the module
    docstring). The WER/CER normalizers used to be imported from it, so the
    ``asr`` task with its default scorer crashed there before generation ever
    started. They are melteval's own copy now (:mod:`melteval.normalizers`);
    these pin that the default builds everywhere, that ``none`` stays
    available, and that it doesn't leak into the tasks it doesn't apply to.
    """

    @pytest.fixture
    def frozen(self, tmp_path):
        from melteval.manifest import MANIFEST_NAME, AudioLocator, EvalRecord, write_manifest

        write_manifest(
            tmp_path / MANIFEST_NAME,
            [
                EvalRecord(
                    sample_key="00-000000",
                    task="asr",
                    target="hello",
                    audio=AudioLocator(kind="file", params={"path": "/tmp/a.wav"}),
                    lang="en",
                    dataset_id="librispeech",
                    duration=2.0,
                )
            ],
        )
        return str(tmp_path)

    def test_normalizer_none_builds_the_task_without_melt(self, frozen):
        from melteval.tasks import asr

        task = asr(frozen, prompt_style="smurf", instruction="Transcribe: ", normalizer="none")
        assert task.name == "speech-asr"

    def test_the_default_normalizer_builds_without_melt(self, frozen):
        """A SMURF or baseline run scores with the same normalizer as a MELT
        run; `none` would make their WER incomparable."""
        from melteval.tasks import asr

        task = asr(frozen, prompt_style="smurf", instruction="Transcribe: ")
        assert task.name == "speech-asr"

    def test_the_task_carries_its_generation_budget(self, frozen):
        from melteval.registry import TASK_MAX_TOKENS
        from melteval.tasks import asr

        task = asr(frozen, prompt_style="smurf", instruction="Transcribe: ")
        assert task.config.max_tokens == TASK_MAX_TOKENS["asr"]

    def test_normalizer_is_rejected_for_st(self, frozen):
        from melteval.tasks import st

        with pytest.raises(ValueError, match="task_filter='asr'"):
            st(frozen, normalizer="none")

    def test_normalizer_conflicts_with_an_explicit_scorer(self, frozen):
        from inspect_ai.scorer import exact

        from melteval.tasks import asr

        with pytest.raises(ValueError, match="ignored when scorer is given"):
            asr(frozen, prompt_style="smurf", instruction="Transcribe: ", normalizer="none", scorer=exact())


class TestSmurfSolverThroughRealInspect:
    """Runs a real ``inspect_ai.eval()``.

    The solver's contract is about what reaches the model: one user message
    holding the bare instruction and the audio locator, with no chat template
    applied (the checkpoint applies its own). Asserting that on the message
    inspect actually delivers is the only version of the check that would have
    caught a double-wrapped prompt.
    """

    def _run(self, tmp_path, samples, **solver_kwargs):
        from inspect_ai import Task
        from inspect_ai import eval as inspect_eval

        from melteval.solver import smurf_prompt

        [log] = inspect_eval(
            Task(dataset=samples, solver=smurf_prompt(**solver_kwargs)),
            model="mockllm/model",
            display="none",
            log_dir=str(tmp_path),
        )
        return log

    def _sample(self, **metadata):
        from inspect_ai.dataset import MemoryDataset, Sample

        meta = {"task": "asr", "lang": "en", "audio": {"kind": "file", "path": "/tmp/a.wav"}}
        meta.update(metadata)
        return MemoryDataset(samples=[Sample(input="placeholder", target="hi", id="s-0", metadata=meta)])

    def test_the_message_is_the_bare_instruction_plus_the_audio(self, tmp_path):
        log = self._run(tmp_path, self._sample(), instruction="Transcribe this {lang_code} audio: ")

        assert log.status == "success"
        [message] = log.samples[0].messages[:1]
        text = "".join(c.text for c in message.content if getattr(c, "type", None) == "text")
        assert text == "Transcribe this en audio: "
        # No chat template, and no audio placeholder: both belong to the model.
        assert "<|im_start|>" not in text
        assert "<|audio" not in text
        assert any(getattr(c, "type", None) == "data" for c in message.content)

    def test_a_sample_carrying_its_own_instruction_wins(self, tmp_path):
        log = self._run(
            tmp_path,
            self._sample(instruction="Answer the question about the talk."),
            instruction="Transcribe: ",
        )

        assert log.status == "success"
        assert log.samples[0].store["melteval:prompt"] == "Answer the question about the talk."
        assert log.samples[0].store["melteval:format_spec"]["instruction_source"] == "sample"

    def test_no_instruction_anywhere_fails_the_run(self, tmp_path):
        """An empty user turn is a valid input to a speech LLM: it generates
        something plausible, and the score looks like a result."""
        log = self._run(tmp_path, self._sample())

        assert log.status == "error"
        assert "No instruction for sample" in str(log.error or log.samples[0].error)


class TestHFPromptStyle:
    """``prompt_style=hf`` (``melt/hf/<model>``) shares SMURF's bare-instruction path but not its config file."""

    def test_builds_with_an_instruction(self):
        from melteval.tasks import _prompt_solver

        assert _prompt_solver("hf", None, None, "Transcribe this audio.", None) is not None

    def test_melt_and_smurf_only_arguments_are_rejected(self):
        from melteval.tasks import _prompt_solver

        with pytest.raises(ValueError, match="format_config"):
            _prompt_solver("hf", "/some/training_config.yaml", None, None, None)
        with pytest.raises(ValueError, match="prompt_config.*'smurf'"):
            _prompt_solver("hf", None, None, None, "/some/asr_inference.yaml")

    def test_a_shared_argument_names_every_style_it_belongs_to(self):
        """`instruction` is both smurf's and hf's; naming only one would
        send the user to the wrong fix."""
        from melteval.tasks import _prompt_solver

        with pytest.raises(ValueError, match="'smurf' or 'hf'"):
            _prompt_solver("melt", None, None, "Transcribe: ", None)


class TestModelFamily:
    """Which prompt path a model name implies (melteval.providers)."""

    @pytest.mark.parametrize(
        ("model", "family"),
        [
            ("melt//abs/ckpt", "melt"),
            ("melt/relative/ckpt", "melt"),
            ("melt/hf/qwen2_audio", "hf"),
            ("melt/vllm/some/model", "vllm"),
            ("smurf//abs/model.ckpt", "smurf"),
            ("mockllm/model", None),
            ("openai/gpt-4o", None),
        ],
    )
    def test_family_of(self, model, family):
        from melteval.providers import family_of

        assert family_of(model) == family


class TestProviderGuard:
    """Crossing a model with another family's solver produces fluent, wrong text."""

    def _state(self, model):
        from types import SimpleNamespace

        api, _, name = model.partition("/")
        return SimpleNamespace(model=SimpleNamespace(api=api, name=name))

    @pytest.mark.parametrize(("model", "family"), [("melt//ckpt", "melt"), ("smurf//m.ckpt", "smurf")])
    def test_instruction_prompt_rejects_the_other_families(self, model, family):
        from melteval.solver import _INSTRUCTION_PROVIDERS, _require_provider

        with pytest.raises(ValueError, match=f"prompt_style={family}"):
            _require_provider(self._state(model), *_INSTRUCTION_PROVIDERS)

    @pytest.mark.parametrize("model", ["melt/hf/qwen2_audio", "melt/hf/qwen3_omni"])
    @pytest.mark.parametrize("expected", ["melt", "smurf"])
    def test_other_solvers_reject_the_hf_baselines(self, expected, model):
        """`melt/hf/...` shares the `melt` provider with MELT checkpoints, but not their prompt path."""
        from melteval.solver import _require_provider

        with pytest.raises(ValueError, match="prompt_style=hf"):
            _require_provider(self._state(model), expected)

    def test_a_matching_or_unknown_provider_passes(self):
        from melteval.solver import _INSTRUCTION_PROVIDERS, _require_provider

        _require_provider(self._state("melt/hf/qwen3_omni"), *_INSTRUCTION_PROVIDERS)
        _require_provider(self._state("melt//ckpt"), "melt")
        _require_provider(self._state("mockllm/model"), *_INSTRUCTION_PROVIDERS)


class TestInstructionSolverThroughRealInspect(TestSmurfSolverThroughRealInspect):
    """The same contract as SMURF's, run through :func:`instruction_prompt`."""

    def _run(self, tmp_path, samples, **solver_kwargs):
        from inspect_ai import Task
        from inspect_ai import eval as inspect_eval

        from melteval.solver import instruction_prompt

        [log] = inspect_eval(
            Task(dataset=samples, solver=instruction_prompt(**solver_kwargs)),
            model="mockllm/model",
            display="none",
            log_dir=str(tmp_path),
        )
        return log

    def test_the_log_records_the_instruction_and_its_source(self, tmp_path):
        log = self._run(tmp_path, self._sample(), instruction="Transcribe this audio.")

        spec = log.samples[0].store["melteval:format_spec"]
        assert spec["instruction"] == "Transcribe this audio."
        assert spec["instruction_source"] == "argument"
        assert spec["apply_chat_template"] is False

    def test_a_benchmark_instruction_with_the_audio_token_renders(self, tmp_path):
        """Every AIR-Bench/MCIF sample carries `{audio_token}`; without stripping
        it, render_smurf_prompt raised on all of them."""
        log = self._run(tmp_path, self._sample(instruction="{audio_token}\nWhat is the emotion?"))

        assert log.status == "success"
        store = log.samples[0].store
        assert store["melteval:prompt"] == "What is the emotion?"
        assert store["melteval:format_spec"]["audio_token_stripped"] is True
        # The template as the benchmark wrote it stays in the log.
        assert store["melteval:format_spec"]["instruction"] == "{audio_token}\nWhat is the emotion?"
