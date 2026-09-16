"""Text-only prior: how much of a language a backbone already knows before
any audio training touches it.

Teacher-forced NLL/BPC of a reference string under a bare
``AutoModelForCausalLM`` — never a ``MELTForCausalLM``, no ``<|audio|>``
token, no MELT vocab extension, since the whole point is the prior
*before* any MELT-specific change (`02-backbones.md` §3.1). Cannot go
through ``inspect eval``: melt-eval's Task/Solver/Scorer triad samples a
completion and scores the sampled text; there is no logprob-of-a-given-
continuation path anywhere in it (`02-backbones.md` §3.2). This mirrors
``no_audio_floor.py`` (training repo, `projects/ablation-campaign/`) but
without the MELT wrapper it scores inside.

Two numbers per sample, per `02-backbones.md` §3.3-§3.4:

- **raw**: ``tokenizer(reference)`` alone, no chat template, no
  instruction, every token scored -- a pure language-model prior.
- **conditioned**: ``[user, assistant]`` rendered through the same
  chat-template machinery training uses (``apply_chat_template_to_texts``,
  the training repo's own helper -- melt-eval already depends on
  ``melt-proj``), assistant turn masked in via
  ``mask_non_assistant_tokens``. The instruction is the first (canonical)
  entry of the matching ``TASK_TEMPLATES`` pool, pinned rather than
  randomly drawn, with the audio-token placeholder stripped (these
  tokenizers never learned one).

One process at a time, no padding: correctness over throughput for a
measurement run once per backbone on a few thousand short samples.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from melteval.manifest import EvalRecord, read_manifest, resolve_frozen_set


#: First (canonical) entry of each TASK_TEMPLATES pool (training repo,
#: melt/training/data/audio/lhotse/helpers.py), pinned so every backbone is
#: scored against the same fixed wording instead of a randomly drawn one.
INSTRUCTION_TEMPLATES: dict[str, str] = {
    "asr": "{audio_token} Transcribe this audio in {lang}.",
    "st": "{audio_token} Translate this audio to {lang}.",
}


@dataclass
class _Totals:
    """Accumulated nats/tokens/chars/words for one (language, mode) cell."""

    nats: float = 0.0
    tokens: int = 0
    chars: int = 0
    words: int = 0
    samples: int = 0

    def add(self, nats: float, tokens: int, reference: str) -> None:
        self.nats += nats
        self.tokens += tokens
        self.chars += len(reference)
        self.words += len(reference.split())
        self.samples += 1

    def summary(self) -> dict:
        nats_per_token = self.nats / self.tokens if self.tokens else 0.0
        bits_per_token = nats_per_token / math.log(2)
        bits_per_char = (self.nats / math.log(2)) / self.chars if self.chars else 0.0
        tokens_per_word = self.tokens / self.words if self.words else 0.0
        tokens_per_char = self.tokens / self.chars if self.chars else 0.0
        return {
            "num_samples": self.samples,
            "num_target_tokens": self.tokens,
            "nats_per_token": nats_per_token,
            "bits_per_token": bits_per_token,
            "bits_per_char": bits_per_char,
            "tokens_per_word": tokens_per_word,
            "tokens_per_char": tokens_per_char,
        }


def _target_text(record: EvalRecord) -> str:
    """The reference string to score -- ASR transcript or ST reference."""
    target = record.target
    return target if isinstance(target, str) else target[0]


def load_backbone(model_id: str, chat_template_from: str | None, device: str, dtype: str):
    """Load a bare causal LM and its tokenizer -- never ``MELTForCausalLM``.

    Args:
        model_id: HF hub id or local path of the checkpoint to score.
        chat_template_from: Load the tokenizer (and so the chat template)
            from here instead, for a base checkpoint that ships none of its
            own (`02-backbones.md` §3.1's per-backbone table).
        device: Torch device.
        dtype: Compute dtype name (e.g. ``bfloat16``).

    Returns:
        ``(model, tokenizer)``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(chat_template_from or model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype))
    model.to(device)
    model.eval()
    return model, tokenizer


def _score(model, tokenizer, input_ids, labels, device) -> tuple[float, int]:
    """One forward pass; returns (total nats, valid target-token count)."""
    import torch

    input_ids = input_ids.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
    valid = int((labels[..., 1:] != -100).sum().item())
    return float(out.loss.item()) * valid, valid


def score_raw(model, tokenizer, device, records: list[EvalRecord]) -> _Totals:
    """Unconditioned NLL/BPC: the reference string alone, BOS included.

    No chat template, no instruction -- a pure language-model prior,
    isolated from instruction-following / template-familiarity effects
    (`02-backbones.md` §3.3).
    """
    totals = _Totals()
    for record in records:
        reference = _target_text(record)
        ids = tokenizer(reference, return_tensors="pt")["input_ids"]
        if ids.shape[1] < 2:
            continue  # nothing to shift-predict
        nats, valid = _score(model, tokenizer, ids, ids.clone(), device)
        totals.add(nats, valid, reference)
    return totals


def score_conditioned(
    model, tokenizer, device, records: list[EvalRecord], chat_template_config: str, task: str
) -> _Totals:
    """Conditioned NLL/BPC: `[user, assistant]` rendered, assistant masked in.

    Reuses the training repo's own chat-template + masking machinery
    (`apply_chat_template_to_texts`, `mask_non_assistant_tokens`) --
    the same functions `no_audio_floor.py` already validated -- so the
    render is byte-for-byte what training would have produced, minus the
    audio token these backbones never learned.
    """
    from melt.training.data.audio.lhotse.helpers import (
        apply_chat_template_to_texts,
        mask_non_assistant_tokens,
    )
    from melt.training.data.chat_templates import get_chat_template_config

    ct_cfg = get_chat_template_config(chat_template_config)
    assistant_start_ids = tokenizer.encode(ct_cfg.assistant_start, add_special_tokens=False)
    assistant_end_ids = tokenizer.encode(ct_cfg.assistant_end, add_special_tokens=False)
    prompt_template = {"asr": INSTRUCTION_TEMPLATES["asr"], "st": INSTRUCTION_TEMPLATES["st"]}

    totals = _Totals()
    for record in records:
        reference = _target_text(record)
        [full_text] = apply_chat_template_to_texts(
            texts=[reference],
            tasks=[task],
            langs=[record.lang],
            tokenizer=tokenizer,
            audio_token="",
            prompt_template=prompt_template,
            prompt_template_selection="custom",
            src_langs=[record.src_lang],
            tgt_langs=[record.tgt_lang],
        )
        # apply_chat_template's own Jinja template already renders BOS where
        # the model expects one; add_special_tokens=True here would insert a
        # second one on top of it.
        ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        if ids.shape[1] < 2:
            continue
        labels = mask_non_assistant_tokens(ids.clone(), assistant_start_ids, assistant_end_ids)
        nats, valid = _score(model, tokenizer, ids, labels, device)
        if valid == 0:
            continue  # assistant-span markers not found in this render; skip rather than silently score nothing
        totals.add(nats, valid, reference)
    return totals


def run(
    frozen_set: str,
    model_id: str,
    chat_template_config: str,
    chat_template_from: str | None = None,
    task: str | None = None,
    lang: str | None = None,
    limit: int | None = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> dict:
    """Score one backbone against one frozen set. Returns the results dict
    also written to disk by the CLI."""
    manifest = resolve_frozen_set(frozen_set)
    records = read_manifest(manifest)
    if task is not None:
        records = [r for r in records if r.task == task]
    if lang is not None:
        records = [r for r in records if r.lang == lang]
    if not records:
        raise ValueError(f"No samples in {manifest} match task={task!r} lang={lang!r}")

    by_lang: dict[str, list[EvalRecord]] = {}
    for r in records:
        by_lang.setdefault(r.lang, []).append(r)
    if limit is not None:
        by_lang = {lang: recs[:limit] for lang, recs in by_lang.items()}

    model, tokenizer = load_backbone(model_id, chat_template_from, device, dtype)

    per_language: dict[str, dict] = {}
    overall_raw = _Totals()
    overall_cond = _Totals()
    for lang, recs in sorted(by_lang.items()):
        raw = score_raw(model, tokenizer, device, recs)
        cond = score_conditioned(model, tokenizer, device, recs, chat_template_config, recs[0].task)
        per_language[lang] = {"raw": raw.summary(), "conditioned": cond.summary()}
        overall_raw.nats += raw.nats
        overall_raw.tokens += raw.tokens
        overall_raw.chars += raw.chars
        overall_raw.words += raw.words
        overall_raw.samples += raw.samples
        overall_cond.nats += cond.nats
        overall_cond.tokens += cond.tokens
        overall_cond.chars += cond.chars
        overall_cond.words += cond.words
        overall_cond.samples += cond.samples

    return {
        "model": model_id,
        "chat_template_from": chat_template_from,
        "chat_template_config": chat_template_config,
        "frozen_set": str(manifest),
        "task": task,
        "languages": per_language,
        "overall": {"raw": overall_raw.summary(), "conditioned": overall_cond.summary()},
    }


def write_results(results: dict, out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
