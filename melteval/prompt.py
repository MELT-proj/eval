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
