"""NVIDIA canary-1b-v2, addressed as ``--model melt/hf/canary``.

canary is a NeMo AED model (``EncDecMultiTaskModel``), not a transformers one,
so it needs a venv with NeMo (see docs/baseline-providers.md). It is not a chat
model either: ``transcribe()`` takes audio, a source language and a target
language, and the task follows from the pair (equal: ASR; different: speech
translation, which the model supports to and from English only).

The instruction the task renders is therefore not a prompt but a three-field
spec, ``<task>|<source code>|<target code>``, written in the config as
``"{task}|{src_lang_code}|{tgt_lang_code}"`` (for ASR the target is ignored and
the source is the sample's language). A language the model does not cover raises
instead of returning a transcript in another language.

The card gives no clip-length limit beyond a recommended 5-25 s; NeMo chunks
longer audio itself, so clips are passed as they are.
"""

from __future__ import annotations

import logging
from typing import Any

from melteval.providers.base import BatchedSpeechAPI, _Request


logger = logging.getLogger(__name__)

FAMILY = "canary-1b-v2"
#: Languages canary-1b-v2 transcribes (ISO 639-1), from the model card.
LANGUAGES = frozenset(
    "bg hr cs da nl en et fi fr de el hu it lv lt mt pl pt ro sk sl es sv ru uk".split()
)


class CanaryAPI(BatchedSpeechAPI):
    """Generate from canary-1b-v2 through NeMo's ``transcribe()``."""

    family = FAMILY
    repo = "nvidia/canary-1b-v2"
    revision = "d455706339a6b32e1aa40f82c713a482a0c938e2"

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: Any = None,
        path: str | None = None,
        revision: str | None = None,
        device: str | None = None,
        batch_size: int = 8,
        batch_window: int = 4,
        pnc: str = "yes",
        **_unused: Any,
    ) -> None:
        """Load the model from the Hub cache (offline once staged) or from *path*, a ``.nemo`` file."""
        from inspect_ai.model import GenerateConfig

        super().__init__(model_name, base_url, api_key, config or GenerateConfig(), batch_size, batch_window)
        import torch
        from huggingface_hub import hf_hub_download
        from nemo.collections.asr.models import EncDecMultiTaskModel

        nemo_file = path if path and path.endswith(".nemo") else hf_hub_download(
            path or self.repo, "canary-1b-v2.nemo", revision=revision
        )
        logger.info("Loading %s from %s", FAMILY, nemo_file)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EncDecMultiTaskModel.restore_from(nemo_file, map_location=self.device).eval()
        self.pnc = pnc

    def close(self) -> None:
        """Drop the model so the GPU memory is released."""
        self.model = None

    def _generate_batch(self, batch: list[_Request]) -> list[str]:
        """Transcribe or translate one batch; requests are grouped by language pair, as NeMo takes one pair per call."""
        results: dict[int, str] = {}
        groups: dict[tuple[str, str], list[int]] = {}
        for i, request in enumerate(batch):
            groups.setdefault(parse_spec(request.text), []).append(i)
        for (source, target), indices in groups.items():
            if request_missing := [i for i in indices if batch[i].audio is None]:
                raise ValueError(f"{len(request_missing)} sample(s) carry no audio; {FAMILY} needs one per request.")
            outputs = self.model.transcribe(
                [batch[i].audio for i in indices],
                batch_size=len(indices),
                source_lang=source,
                target_lang=target,
                pnc=self.pnc,
                verbose=False,
            )
            for i, out in zip(indices, outputs):
                results[i] = (getattr(out, "text", out) or "").strip()
        return [results[i] for i in range(len(batch))]


def parse_spec(text: str) -> tuple[str, str]:
    """``<task>|<source>|<target>`` as ``(source, target)`` codes; ASR makes target equal source.

    Raises:
        ValueError: If the spec is malformed, or a language is not one canary covers, or a
            translation involves no English side.
    """
    parts = [p.strip() for p in text.split("|")]
    if len(parts) != 3 or parts[0] not in ("asr", "st"):
        raise ValueError(
            f"{FAMILY} instruction must be '<asr|st>|<source code>|<target code>' "
            f"(config: \"{{task}}|{{src_lang_code}}|{{tgt_lang_code}}\"), got {text!r}."
        )
    task, source, target = parts
    if task == "asr":
        target = source
    for code in (source, target):
        if code not in LANGUAGES:
            raise ValueError(f"{FAMILY} does not cover language {code!r}; run it on the languages it supports only.")
    if source != target and "en" not in (source, target):
        raise ValueError(f"{FAMILY} translates to and from English only, got {source}->{target}.")
    return source, target
