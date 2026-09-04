"""Prompt and chat-format parity with training.

A checkpoint must be evaluated in the format it was trained in. If training
wrapped each sample in a chat turn and evaluation sends a bare string, the
resulting number describes a distribution shift, not the model. The training
repo has already shipped that bug once, in the opposite direction
(MELT-proj/training#58, where eval read the formatting keys from one level too
deep and silently used their defaults).

So nothing here has a "sensible default". The four keys that decide the
sequence format are read from the run's ``training_config.yaml`` and, if that
file is missing, the harness refuses to guess.

The same rule applies to SMURF checkpoints, but the config that answers it is
a different one. There, the chat wrapping is applied by the model itself (a
NeMo ``PromptFormatter`` named in the checkpoint) and the only free variable is
the instruction text, which lives in the *data* config as the conversation's
``context`` tag. :func:`load_smurf_prompt_spec` reads it from there, for the
same reason: so the prompt comes from a file somebody wrote deliberately
rather than from a default nobody chose.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path


CONFIG_NAME = "training_config.yaml"

#: Keys at the ``data.`` level of a training config that decide the sequence
#: format. The same four the training loader inherits into its eval path.
FORMAT_KEYS = (
    "apply_chat_template",
    "prompt_template",
    "prompt_template_selection",
    "chat_template_config",
)


@dataclass(frozen=True)
class FormatSpec:
    """How a checkpoint expects its inputs to be formatted.

    Attributes:
        apply_chat_template: Whether samples were wrapped in a chat turn.
        prompt_template: Custom template, as a string or per-task mapping. Only
            consulted when *prompt_template_selection* is ``"custom"``.
        prompt_template_selection: ``random``, ``with_language`` or ``custom``.
        chat_template_config: Name of the boundary config (e.g. ``chatml``).
        audio_token: Placeholder the processor expands into encoder frames.
        source: Where this spec was read from, recorded in the eval log so a
            result can be traced back to the config that shaped it.
    """

    apply_chat_template: bool
    prompt_template: str | dict[str, str] | None
    prompt_template_selection: str
    chat_template_config: str
    audio_token: str
    source: str

    def to_dict(self) -> dict:
        """Serialise for the eval log."""
        return {
            "apply_chat_template": self.apply_chat_template,
            "prompt_template": self.prompt_template,
            "prompt_template_selection": self.prompt_template_selection,
            "chat_template_config": self.chat_template_config,
            "audio_token": self.audio_token,
            "source": self.source,
        }


def find_training_config(path: str | Path) -> Path:
    """Locate the ``training_config.yaml`` for a checkpoint or run directory.

    Checkpoints are written into ``<run_dir>/checkpoint-N`` while the config is
    written at the run root, so the parent directories are searched too.

    Args:
        path: A checkpoint directory, a run directory, or the config file.

    Returns:
        Path to the config file.

    Raises:
        FileNotFoundError: If no config is found, naming the override flag
            rather than falling back to defaults.
    """
    path = Path(path)
    if path.is_file():
        return path

    for candidate in (path, *path.parents[:2]):
        config = candidate / CONFIG_NAME
        if config.exists():
            return config

    raise FileNotFoundError(
        f"No {CONFIG_NAME} at or above {path}. Evaluation must use the format the "
        "checkpoint was trained in, and guessing it is how eval_loss silently stopped "
        "matching training loss before (MELT-proj/training#58). Pass --format-config "
        "to point at the config explicitly."
    )


def load_format_spec(path: str | Path) -> FormatSpec:
    """Read the formatting keys a checkpoint was trained under.

    Args:
        path: Checkpoint directory, run directory, or config file.

    Returns:
        The resolved spec.
    """
    import yaml

    config_path = find_training_config(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data = config.get("data") or {}
    decoder = (config.get("model") or {}).get("decoder") or {}

    return FormatSpec(
        apply_chat_template=bool(data.get("apply_chat_template", False)),
        prompt_template=data.get("prompt_template"),
        prompt_template_selection=str(data.get("prompt_template_selection", "random")),
        chat_template_config=str(data.get("chat_template_config", "chatml")),
        audio_token=str(decoder.get("audio_token", "<|audio|>")),
        source=str(config_path),
    )


def select_template(task: str, sample_key: str, spec: FormatSpec) -> str:
    """Pick the prompt template for one sample.

    Selection is deterministic in *sample_key*, so a re-run of the same frozen
    set draws the same template for the same sample. Training draws from the
    pool at random, per epoch, which is right for training and wrong for a
    measurement that has to be comparable between runs.

    Args:
        task: Task identifier.
        sample_key: The sample's stable key, used as the seed.
        spec: The checkpoint's format spec.

    Returns:
        The unformatted template string.

    Raises:
        ValueError: If the task has no template pool, or the selection strategy
            is unknown.
    """
    from melt.training.data.audio.lhotse.helpers import TASK_TEMPLATES, resolve_custom_template

    if spec.prompt_template_selection == "custom":
        return resolve_custom_template(spec.prompt_template, task)

    templates = TASK_TEMPLATES.get(task)
    if templates is None:
        raise ValueError(
            f"No prompt templates for task {task!r} (known: {sorted(TASK_TEMPLATES)}). "
            "Benchmarks that ship their own prompts should carry them per sample as "
            "`instruction` in the frozen set."
        )

    if spec.prompt_template_selection == "with_language":
        templates = [t for t in templates if "{lang}" in t]
        if not templates:
            raise ValueError(f"No templates with a {{lang}} placeholder for task {task!r}.")
    elif spec.prompt_template_selection != "random":
        raise ValueError(
            f"Unknown prompt_template_selection: {spec.prompt_template_selection!r}"
        )

    return random.Random(sample_key).choice(templates)


def render_user_prompt(
    task: str,
    sample_key: str,
    spec: FormatSpec,
    lang: str = "",
    src_lang: str = "",
    tgt_lang: str = "",
    instruction: str | None = None,
) -> str:
    """Render the user-turn text, before any chat template is applied.

    Args:
        task: Task identifier.
        sample_key: Stable sample key, seeding template selection.
        spec: The checkpoint's format spec.
        lang: Language of the expected output.
        src_lang: Source language, where the task has one.
        tgt_lang: Target language, where the task has one.
        instruction: A per-sample prompt that overrides the template pool.

    Returns:
        The formatted prompt, containing the audio token.

    Raises:
        ValueError: If *lang* is not a language the training code recognises.
            Deliberately as strict as training: a code training would reject
            must not quietly produce a different prompt here.
    """
    from melt.training.data.audio.lhotse.helpers import (
        LANGUAGE_ISO_TO_NAME,
        _resolve_language_name_safe,
    )

    template = instruction or select_template(task, sample_key, spec)

    language_name = LANGUAGE_ISO_TO_NAME.get((lang or "").lower())
    if language_name is None and "{lang}" in template:
        raise ValueError(
            f"Unsupported language ISO code {lang!r} for sample {sample_key}. "
            f"Expected one of: {', '.join(sorted(LANGUAGE_ISO_TO_NAME))}"
        )

    return template.format(
        audio_token=spec.audio_token,
        lang=language_name or "",
        src_lang=_resolve_language_name_safe(src_lang),
        tgt_lang=_resolve_language_name_safe(tgt_lang),
    )


def apply_generation_format(prompt: str, spec: FormatSpec, tokenizer) -> str:
    """Wrap *prompt* the way training wrapped it, minus the assistant turn.

    Training renders ``[user, assistant]`` with ``add_generation_prompt=False``;
    generation renders ``[user]`` with ``add_generation_prompt=True``. For a
    ChatML-style template the second is a strict prefix of the first, ending
    exactly at the assistant-turn boundary — which is the operational meaning of
    "the same format", and what ``tests/test_prompt.py`` asserts.

    Args:
        prompt: The rendered user-turn text.
        spec: The checkpoint's format spec.
        tokenizer: Tokenizer providing ``apply_chat_template``.

    Returns:
        The string to hand to the processor.

    Raises:
        ValueError: If the chat template is required but unavailable.
    """
    if not spec.apply_chat_template:
        return prompt

    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "The checkpoint was trained with apply_chat_template=true but its tokenizer "
            "has no apply_chat_template(). Evaluating it as raw text would measure a "
            "format the model never saw."
        )

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# --- SMURF ------------------------------------------------------------------
#
# A SMURF checkpoint is a NeMo SALM. Three of the four MELT format keys have no
# counterpart there: the chat template is applied by the model's own
# ``PromptFormatter``, and there is no template pool to select from. What is
# left is the instruction text, which SMURF puts in the data config as the
# conversation's ``context`` tag -- one fixed string per data source, not one
# per sample.


@dataclass(frozen=True)
class SmurfPromptSpec:
    """The instruction a SMURF checkpoint expects, and where it came from.

    Attributes:
        instruction: The instruction template, before per-sample placeholders
            are filled. ``None`` when the config carried no ``context`` tag,
            which is a valid SMURF setup (the conversation is then audio-only,
            with the task implied by training).
        prompt_format: Name of the NeMo ``PromptFormatter`` the config names.
            Recorded only: the model applies its own, read from its weights'
            config, and a disagreement between the two is worth seeing in the
            log rather than resolving silently here.
        audio_locator_tag: The placeholder named by the config, recorded for
            the same reason. The provider uses the checkpoint's own.
        source: Where this spec was read from.
    """

    instruction: str | None
    prompt_format: str | None
    audio_locator_tag: str | None
    source: str

    def to_dict(self) -> dict:
        """Serialise for the eval log."""
        return {
            "provider": "smurf",
            "instruction": self.instruction,
            "prompt_format": self.prompt_format,
            "audio_locator_tag": self.audio_locator_tag,
            "source": self.source,
        }


def load_smurf_prompt_spec(path: str | Path) -> SmurfPromptSpec:
    """Read the instruction context out of a SMURF data or inference config.

    Handles every shape those configs come in -- ``data.test_ds.input_cfg``,
    ``data.validation_ds.datasets.<name>.input_cfg``, a bare ``input_cfg`` --
    by scanning for the keys rather than walking a fixed path, because the
    inference configs and the training configs nest them differently and both
    are legitimate sources.

    Args:
        path: Path to the YAML.

    Returns:
        The instruction and the identifiers worth recording alongside it.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the config carries more than one distinct ``context``,
            which would make "the prompt this eval used" a per-source detail
            that a single run cannot honour.
    """
    import yaml

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"No SMURF config at {config_path}.")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    contexts = _collect_tag_values(config, "context")
    if len(contexts) > 1:
        raise ValueError(
            f"{config_path} defines {len(contexts)} different context prompts "
            f"({contexts!r}). One eval run sends one prompt, so pick the source you mean "
            "(or pass the instruction explicitly)."
        )

    prompt_formats = _collect_values(config, "prompt_format")
    audio_tags = _collect_values(config, "audio_locator_tag")

    return SmurfPromptSpec(
        instruction=contexts[0] if contexts else None,
        prompt_format=prompt_formats[0] if len(prompt_formats) == 1 else None,
        audio_locator_tag=audio_tags[0] if len(audio_tags) == 1 else None,
        source=str(config_path),
    )


def render_smurf_prompt(
    instruction: str,
    task: str = "",
    lang: str = "",
    src_lang: str = "",
    tgt_lang: str = "",
) -> str:
    """Fill a SMURF instruction template for one sample.

    Templates are usually literal in SMURF's own configs ("Transcribe this
    English audio: "), so a template with no placeholder is the common case and
    passes through untouched. The placeholders exist so that one frozen set
    covering several languages does not need one config per language:

    * ``{lang}``, ``{src_lang}``, ``{tgt_lang}`` — the language *name*
      ("English"), resolved with the training package's table so that a name
      here and a name in a MELT prompt are the same string.
    * ``{lang_code}``, ``{src_lang_code}``, ``{tgt_lang_code}`` — the ISO code,
      which needs no table and so works in a SMURF-only environment.
    * ``{task}`` — the frozen set's task id.

    Args:
        instruction: The template.
        task: Task identifier.
        lang: Language of the expected output.
        src_lang: Source language, where the task has one.
        tgt_lang: Target language, where the task has one.

    Returns:
        The instruction to send, before the audio placeholder is attached (the
        provider does that, using the checkpoint's own tag).

    Raises:
        ValueError: If the template names a placeholder that does not exist, or
            a language name cannot be resolved. Both are better as errors than
            as a prompt with a literal ``{lang}`` in it, which no model was
            trained on and which every sample would still score against.
    """
    if not isinstance(instruction, str):
        raise ValueError(
            f"SMURF instruction must be a string, got {type(instruction).__name__}: {instruction!r}. "
            "If this came from -T instruction=..., inspect eval parses that value as YAML, and a "
            "colon followed by a space (as in 'Transcribe this English audio: ') is read as a "
            "mapping instead of text. Wrap it in literal double quotes so YAML sees a quoted "
            'string, e.g. -T instruction="\\"Transcribe this English audio: \\""  (see '
            "docs/replication_notes.md)."
        )
    values = {
        "task": task,
        "lang_code": lang,
        "src_lang_code": src_lang,
        "tgt_lang_code": tgt_lang,
    }
    for key, code in (("lang", lang), ("src_lang", src_lang), ("tgt_lang", tgt_lang)):
        if "{" + key + "}" in instruction:
            values[key] = _language_name(code, key)

    try:
        return instruction.format_map(values)
    except KeyError as exc:
        known = ("task", "lang", "src_lang", "tgt_lang", "lang_code", "src_lang_code", "tgt_lang_code")
        raise ValueError(
            f"Unknown placeholder {exc} in the SMURF instruction {instruction!r}. "
            f"Available: {', '.join('{' + name + '}' for name in known)}."
        ) from exc


def _language_name(code: str, key: str) -> str:
    """Resolve an ISO code to the language name the training code uses.

    Raises:
        ValueError: If the table is unavailable or the code is not in it. The
            table lives in the training package, which a SMURF-only
            environment has no reason to install -- hence the suggestion to use
            the ``_code`` placeholder or a literal instruction instead.
    """
    try:
        from melt.training.data.audio.lhotse.helpers import LANGUAGE_ISO_TO_NAME
    except ImportError as exc:
        raise ValueError(
            f"The instruction uses {{{key}}}, which needs the training package's language table "
            f"(melt-proj is not importable: {exc}). In a SMURF-only environment use "
            f"{{{key}_code}} for the ISO code, or write the language into the instruction."
        ) from exc

    name = LANGUAGE_ISO_TO_NAME.get((code or "").lower())
    if name is None:
        raise ValueError(
            f"Unsupported language ISO code {code!r} for {{{key}}}. "
            f"Expected one of: {', '.join(sorted(LANGUAGE_ISO_TO_NAME))}"
        )
    return name


def _collect_values(node, key: str) -> list:
    """Collect the distinct values of *key* anywhere in a nested config."""
    found: list = []

    def walk(value) -> None:
        if isinstance(value, dict):
            for name, child in value.items():
                if name == key and not isinstance(child, (dict, list)):
                    if child not in found:
                        found.append(child)
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(node)
    return found


def _collect_tag_values(node, key: str) -> list:
    """Collect the distinct values of ``tags.<key>`` anywhere in a nested config.

    Scoped to ``tags`` mappings so that a coincidental key of the same name
    elsewhere in the config is not mistaken for a prompt.
    """
    found: list = []

    def walk(value) -> None:
        if isinstance(value, dict):
            tags = value.get("tags")
            if isinstance(tags, dict) and key in tags and tags[key] not in found:
                found.append(tags[key])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(node)
    return found
