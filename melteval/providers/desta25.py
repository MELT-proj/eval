"""DeSTA2.5-Audio as a baseline.

Unlike the Qwen baselines, DeSTA2.5-Audio is not a ``transformers`` model with
remote code: the Hub repository holds only the connector weights, and the model
class (``DeSTA25AudioModel``) lives in the ``desta`` package
(https://github.com/kehanlu/DeSTA2.5-Audio, ``pip install -e .``), which also
pulls its Llama-3.1-8B-Instruct backbone and a Whisper encoder from the Hub.
That package is imported lazily and is not a dependency of melteval: install it
in its own venv (docs/baseline-providers.md), since it pins its own
``transformers``/``torch``.

Important things to take into account from such package:

* **Audio is a file path.** Each request's audio is written to a temporary WAV.
* **A batch is one call.** ``generate()`` takes a list of conversations, so
  :meth:`Desta25API._generate_batch` sends the whole batch at once.
* **The prompt is the bare instruction.** The model applies the chat template
  and expands ``<|AUDIO|>`` itself (the ``hf`` prompt path, as for Qwen).
* **Every request carries a system message**, as the authors' tips require:
  without one the model describes the audio instead of transcribing it. It comes
  from ``-M system_prompt=...`` (the launcher fills it from the model entry's
  ``system:`` map, per task); otherwise :data:`DEFAULT_SYSTEM_PROMPT` is used.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from inspect_ai.model import GenerateConfig

from melteval.providers.base import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BATCH_WINDOW,
    BatchedSpeechAPI,
    _Request,
)


logger = logging.getLogger(__name__)

FAMILY = "DeSTA2.5-Audio"
AUDIO_PLACEHOLDER = "<|AUDIO|>"

#: The authors' default system prompt (DeSTA2.5-Audio README and
#: ``docs/evaluation_tips.md``). They recommend a different one for ASR
#: ("... Respond directly without any other words") and for multiple choice
#: ("... Choose one of the options without any explanation."); those belong in
#: the campaign config, per task.
DEFAULT_SYSTEM_PROMPT = "Focus on the audio clips and instruction."

#: ``generate()`` runs ``WhisperProcessor`` with its defaults, for the audio
#: features and for the internal transcription alike, and that pads every clip
#: to 30 s and truncates anything longer without a warning.
MAX_AUDIO_SECONDS = 30.0


class Desta25API(BatchedSpeechAPI):
    """``DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B`` through the ``desta`` package."""

    family = FAMILY
    repo = "DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B"
    revision = "1427f8725d35a0c9da0c7c54b1ffcc157a5948af"

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        device: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_window: int = DEFAULT_BATCH_WINDOW,
        path: str | None = None,
        revision: str | None = None,
        system_prompt: str | None = None,
        **model_args: object,
    ) -> None:
        """Load the model.

        Args:
            model_name: The model's name in the log, e.g. ``hf/desta25``.
            base_url: Unused; a local model has no endpoint.
            api_key: Unused; a local model needs no credential.
            config: Default generation config.
            device: Torch device. Defaults to CUDA when available.
            batch_size: Requests generated together in one ``generate()`` call.
            batch_window: See :class:`BatchedSpeechAPI`.
            path: Hub id or local directory to load the weights from.
            revision: Hub commit to load; ``None`` for a local directory. The
                Llama and Whisper backbones that ``desta`` pulls in are not pinned by it.
            system_prompt: The system message sent with every request. ``None``
                means :data:`DEFAULT_SYSTEM_PROMPT`.
            **model_args: Forwarded to ``DeSTA25AudioModel.from_pretrained``.

        Raises:
            ImportError: If the ``desta`` package is not installed.
        """
        super().__init__(model_name, base_url, api_key, config, batch_size=batch_size, batch_window=batch_window)
        _restore_numpy_sctypes()
        try:
            from desta import DeSTA25AudioModel
        except ImportError as exc:
            raise ImportError(
                "DeSTA2.5-Audio needs the `desta` package (git clone https://github.com/kehanlu/"
                "DeSTA2.5-Audio && pip install -e .), ideally in its own venv: see docs/baseline-providers.md."
            ) from exc
        import torch

        source = path or self.repo
        logger.info("Loading %s from %s (revision %s)", FAMILY, source, revision or "-")
        if revision and not os.path.isdir(source):
            # `DeSTA25AudioModel.from_pretrained` ignores `revision=` (it calls
            # hf_hub_download without one), so resolve the commit here and hand
            # it the local directory, which it does load from.
            from huggingface_hub import snapshot_download

            source = snapshot_download(source, revision=revision)
        self.model = DeSTA25AudioModel.from_pretrained(source, **model_args)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._logged_prompt = False

    def close(self) -> None:
        """Drop the model so the GPU memory is released."""
        self.model = None

    def _reject(self, request: _Request) -> str | None:
        """Return why a clip past 30 s cannot be run, or ``None``; that sample fails alone (as for Qwen2-Audio)."""
        if request.duration <= MAX_AUDIO_SECONDS:
            return None
        return (
            f"{FAMILY}: audio of {request.duration:.1f} s exceeds its {MAX_AUDIO_SECONDS:g} s window; "
            "sample not evaluated (it would be scored against the full reference from a truncated input)."
        )

    def _generate_batch(self, batch: list[_Request]) -> list[str]:
        """Generate the whole batch in one ``desta`` call."""
        if not self._logged_prompt:
            sent = _messages(batch[0].text, "<wav>", self.system_prompt)
            logger.info("%s prompt as sent (first sample): %r", FAMILY, [(m["role"], m["content"]) for m in sent])
            self._logged_prompt = True
        return _generate(self.model, batch, self.system_prompt)


def _restore_numpy_sctypes() -> None:
    """Give NumPy 2 back the ``np.sctypes`` that ``desta`` still reads.

    ``desta/utils/audio.py`` tests a WAV's dtype against ``np.sctypes['int']``
    and ``['float']``, which NumPy 2 removed; every sample then fails with an
    ``AttributeError``. The rest of the venv (scipy, scikit-learn, pandas)
    needs NumPy >= 2, so pinning NumPy < 2 breaks ``import transformers``.
    The values are NumPy 1's.
    """
    import numpy as np

    if not hasattr(np, "sctypes"):
        np.sctypes = {
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [bool, object, bytes, str, np.void],
        }


def _messages(instruction: str, audio_path: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> list[dict]:
    """The system message, then one user turn in ``desta``'s format: audio placeholder, then instruction."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"{AUDIO_PLACEHOLDER}\n{instruction}",
            "audios": [{"audio": audio_path, "text": None}],
        },
    ]


def _generate(model, batch: list[_Request], system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> list[str]:
    """One completion per request, from a single batched ``generate()`` call.

    A module function so a test can pass a fake model. Each request's audio is
    written to its own temporary WAV, since ``desta`` reads paths. They go under
    ``$TMPDIR`` (node-local scratch on a cluster), never the repo or ``$HOME``,
    and are deleted with the directory when the batch is done. Decoding follows
    the first request's config, as for the other ``hf`` providers.

    Raises:
        ValueError: If a request carries no audio.
    """
    import soundfile as sf

    missing = [i for i, r in enumerate(batch) if r.audio is None]
    if missing:
        raise ValueError(
            f"{len(missing)} sample(s) in the batch carry no audio. {FAMILY}'s prompt has an "
            "audio placeholder, so a text-only sample has nothing to fill it with."
        )

    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        conversations = []
        for i, request in enumerate(batch):
            wav = Path(tmp) / f"{i}.wav"
            sf.write(wav, request.audio, request.sample_rate)
            conversations.append(_messages(request.text, str(wav), system_prompt))
        output = model.generate(messages=conversations, **_decoding(batch[0].config))
    return [text.strip() for text in output.text]


def _decoding(config: GenerateConfig) -> dict[str, object]:
    """``desta``'s decoding arguments: greedy unless the task asks to sample."""
    from melteval.providers.base import DEFAULT_MAX_TOKENS

    sample = config.temperature is not None and config.temperature > 0
    return {
        "do_sample": sample,
        "top_p": config.top_p if config.top_p is not None else 1.0,
        "temperature": config.temperature if sample else 1.0,
        "max_new_tokens": config.max_tokens or DEFAULT_MAX_TOKENS,
    }
