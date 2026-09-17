"""In-process model provider for SMURF (NeMo/SALM) checkpoints.

Registered as ``smurf``, so a checkpoint is addressed as
``--model smurf/<path-to-model.ckpt>``.

SMURF trains with `NVIDIA NeMo Speech <https://github.com/NVIDIA-NeMo/Speech>`_:
the model is ``fbk_speechllm.models.speech_llm.SpeechLLM``, a thin subclass of
NeMo's ``SALM``, and its documented inference path
(``python -m fbk_speechllm.inference``) drives it through a Lhotse
``DataModule`` over a conversation manifest. That path is the wrong shape here
-- a frozen set already decides which samples are evaluated, and re-deriving
them from a NeMo data config would put the sample list back under the model's
control. So this provider uses SALM's own lower-level generation API instead,
which takes the prompt as chat turns and the audio as tensors:

    model.generate(prompts=[[{"role": "user", "content": f"…{tag}"}]],
                   audios=…, audio_lens=…, generation_config=…)

Two consequences worth knowing:

* **The chat template is the model's, not ours.** Given turns rather than token
  ids, SALM formats them with the ``PromptFormatter`` named by its own
  ``cfg.prompt_format`` (``qwen`` for the current runs). So the solver must
  hand over the bare instruction text -- see :func:`melteval.solver.smurf_prompt`
  -- and must *not* apply a chat template of its own, or the prompt gets
  wrapped twice.
* **The audio placeholder is the model's too.** SALM splices encoder frames
  into the positions of ``cfg.audio_locator_tag`` (``<|audioplaceholder|>``),
  read off the loaded checkpoint here rather than configured anywhere else, so
  it cannot drift from the weights it belongs to.

Nothing heavy is imported at module scope: registering the provider must not
cost a torch or NeMo import.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from inspect_ai.model import GenerateConfig, modelapi

from melteval.providers.base import DEFAULT_BATCH_SIZE, BatchedSpeechAPI, _Request


logger = logging.getLogger(__name__)

#: Where the audio placeholder goes when the rendered prompt does not already
#: contain it. ``suffix`` matches how SMURF's training data is built: the
#: conversation is a text turn holding the instruction followed by the audio
#: turn, so the placeholder lands at the end of the user message.
AUDIO_PLACEMENTS = ("suffix", "prefix")


@modelapi(name="smurf")
class SmurfAPI(BatchedSpeechAPI):
    """Generate from a local SMURF/NeMo SpeechLLM checkpoint."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        model_config_override: str | None = None,
        device: str | None = None,
        device_map: str | None = None,
        dtype: str = "bfloat16",
        map_location: str | None = None,
        device_map_reserve: float = 0.35,
        batch_size: int = DEFAULT_BATCH_SIZE,
        audio_placement: str = "suffix",
        num_beams: int = 1,
        use_model_defaults: bool | None = False,
        enable_thinking: bool | None = None,
        **model_args: Any,
    ) -> None:
        """Load a checkpoint.

        Args:
            model_name: Path to a Lightning ``.ckpt``, or to a directory saved
                with ``save_pretrained`` (NeMo's ``HFHubMixin`` layout).
            base_url: Unused; a local model has no endpoint.
            api_key: Unused; a local model needs no credential.
            config: Default generation config.
            model_config_override: Path to a YAML holding the model config to
                use in place of the one stored in the checkpoint -- the
                ``hparams.yaml`` shape that NeMo's ``exp_manager`` writes. Same
                key as SMURF's own inference config.
            device: Torch device. Defaults to CUDA when available. Mutually
                exclusive with device_map.
            device_map: ``"auto"`` shards the checkpoint across every visible
                CUDA device instead of loading it onto one. Mutually exclusive with device.
            dtype: Compute dtype the weights are cast to.
            map_location: Where to materialise the checkpoint while loading.
                Left unset by default, matching ``fbk_speechllm.inference``,
                which loads the weights straight back onto the device they
                were saved from.
            device_map_reserve: Fraction of each GPU's free memory left
                unassigned to weights when *device_map* is ``"auto"``, for
                activations and the KV cache during generation.
            batch_size: Samples per forward pass. Also the concurrency inspect
                is allowed, since more in flight than fit in a batch only adds
                queueing.
            audio_placement: Where to insert the model's audio placeholder when
                the rendered prompt does not already contain it -- ``suffix``
                (instruction, then audio; how SMURF's data is built) or
                ``prefix``.
            num_beams: Beam width. 1 (greedy) is what the SMURF inference
                configs use.
            use_model_defaults: Forwarded to HuggingFace ``generate``. ``False``
                mirrors SMURF's inference configs, which pin decoding to what
                the config says rather than to whatever the LLM ships with.
            enable_thinking: Prompt-formatter hint for formats with a thinking
                mode. Left unset by default, matching SMURF's inference path.
            **model_args: Forwarded to the checkpoint loader.

        Raises:
            ValueError: If *audio_placement* is not one of :data:`AUDIO_PLACEMENTS`,
                if *device* and *device_map* are both given, if *device_map* is
                given and is not ``"auto"``, or if it's given alongside a
                *map_location* other than ``"cpu"``.
        """
        super().__init__(model_name, base_url, api_key, config, batch_size=batch_size)

        if audio_placement not in AUDIO_PLACEMENTS:
            raise ValueError(
                f"audio_placement must be one of {AUDIO_PLACEMENTS}, got {audio_placement!r}."
            )
        if device is not None and device_map is not None:
            raise ValueError(
                "device and device_map are mutually exclusive: device pins the whole model on "
                "one device, device_map shards it across several."
            )
        if device_map is not None and device_map != "auto":
            raise ValueError(f'device_map only supports "auto", got {device_map!r}.')
        if device_map is not None and map_location not in (None, "cpu"):
            raise ValueError(
                f"device_map=\"auto\" loads the checkpoint on CPU before sharding it across GPUs; "
                f"map_location must be \"cpu\" or left unset, not {map_location!r}."
            )

        import torch

        self.audio_placement = audio_placement
        self.num_beams = int(num_beams)
        self.use_model_defaults = use_model_defaults
        self.enable_thinking = enable_thinking

        logger.info(
            "Loading SMURF checkpoint from %s (%s)",
            model_name,
            "sharded across every visible GPU" if device_map else f"onto {device or 'cuda if available'}",
        )
        self.model = _load_speechllm(
            model_name,
            model_config_override=model_config_override,
            map_location="cpu" if device_map else map_location,
            **model_args,
        )
        target_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        if device_map:
            # Shrink to bf16 first, still on CPU (cheap: no GPU involved), so
            # _shard_across_devices measures and moves the small copy instead
            # of the full-precision one.
            self.model = self.model.to(dtype=target_dtype)
            self.model = _shard_across_devices(self.model, reserve_fraction=device_map_reserve)
            self.device = self.model.device
        else:
            # One call, not `.to(dtype).to(device)`: two calls would put the
            # full-precision weights on the GPU first and only shrink them
            # there, so a checkpoint that just fits at the target dtype OOMs
            # before it ever gets that small.
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device=self.device, dtype=target_dtype)
        self.model.eval()

        self.audio_locator_tag = self.model.audio_locator_tag
        logger.info(
            "SMURF checkpoint loaded: prompt_format=%s audio_locator_tag=%s sampling_rate=%d",
            self.model.cfg.get("prompt_format"),
            self.audio_locator_tag,
            self.model.sampling_rate,
        )

    def close(self) -> None:
        """Drop the model so the GPU memory is released."""
        self.model = None

    def _generate_batch(self, batch: list[_Request]) -> list[str]:
        """Run one padded batch through the model."""
        import torch

        audios, audio_lens = _collate_audio(batch, self.model.sampling_rate)
        prompts = [
            [{"role": "user", "content": _with_audio_tag(r.text, self.audio_locator_tag, self.audio_placement)}]
            for r in batch
        ]

        generate_kwargs: dict[str, Any] = {}
        if self.use_model_defaults is not None:
            generate_kwargs["use_model_defaults"] = self.use_model_defaults
        if self.enable_thinking is not None:
            generate_kwargs["enable_thinking"] = self.enable_thinking

        with torch.no_grad():
            answer_ids = self.model.generate(
                prompts=prompts,
                audios=audios.to(self.model.device, non_blocking=True),
                audio_lens=audio_lens.to(self.model.device, non_blocking=True),
                generation_config=self._generation_config(batch[0].config),
                **generate_kwargs,
            )

        # Audio is always present here, so SALM generates from ``inputs_embeds``
        # and HuggingFace returns only the new tokens -- no prompt to strip.
        return [
            self.model.tokenizer.ids_to_text(ids, remove_special_tokens=True).strip()
            for ids in answer_ids.cpu()
        ]

    def _generation_config(self, config: GenerateConfig):
        """Build the ``GenerationConfig`` for one batch.

        The three special-token ids are not optional: SALM generates from
        embeddings, so HuggingFace cannot infer them from the input, and
        without ``eos_token_id`` every sample runs to ``max_new_tokens``.
        """
        from transformers import GenerationConfig as HFGenerationConfig

        return HFGenerationConfig(
            bos_token_id=self.model.text_bos_id,
            eos_token_id=self.model.text_eos_id,
            pad_token_id=self.model.text_pad_id,
            **_generation_kwargs(config, num_beams=self.num_beams),
        )


# These modules must all land on the same GPU as the LLM's decoder layer 0. This is due to how HF-s generate() picks devices for its own
_ANCHOR_MODULES = ("embed_tokens", "perception", "llm.lm_head", "llm.model.norm", "llm.model.rotary_emb")


def _plan_device_map(
    anchor_names: list[str], anchor_bytes: int, layer_bytes: list[int], budgets: list[int]
) -> dict[str, int]:
    """Decide which GPU every anchor module and every decoder layer goes on.

    Pure arithmetic: we are just looking at the sizes of modules and the available budgets to decide placement, without touching any actual GPUs.

    Args:
        anchor_names: Dotted attribute paths of the modules that must all
            share one device (:data:`_ANCHOR_MODULES`, filtered to what the
            loaded model actually has).
        anchor_bytes: Combined size of those modules, in bytes.
        layer_bytes: Size of each decoder layer, in order, in bytes.
        budgets: Bytes available for weights on each GPU, in device order --
            already net of whatever headroom the caller wants reserved.

    Returns:
        A ``dispatch_model``-shaped device map: every anchor name and every
        ``llm.model.layers.{i}`` mapped to a GPU index.
    """
    device_map = {name: 0 for name in anchor_names}
    device, budget_left = 0, budgets[0] - anchor_bytes
    last_device = len(budgets) - 1
    for i, size in enumerate(layer_bytes):
        if size > budget_left and device < last_device:
            device += 1
            budget_left = budgets[device]
        device_map[f"llm.model.layers.{i}"] = device
        budget_left -= size
    return device_map


def _shard_across_devices(model, reserve_fraction: float = 0.35):
    """Split a CPU-resident, already-cast SpeechLLM across every visible GPU.

    NeMo's SALM has no ``device_map="auto"`` of its own, so this builds the
    equivalent by hand, in five steps:

    1. Weigh every anchor module and every decoder layer.
    2. Check how much memory is actually *free* on each GPU right now, minus
       a reserved cushion for activations and the KV cache during generation.
    3. Hand those weights and budgets to :func:`_plan_device_map`, which
       decides a GPU for every piece.
    4. Move everything there for real with ``accelerate.dispatch_model``.
    5. Patch ``.device`` by hand: ``dispatch_model`` moves each submodule
       directly rather than through the model's own ``.to()``, so the
       ``.device`` that Lightning normally keeps up to date via that override
       is left stale, still pointing at wherever the model was before the
       move.

    Args:
        model: The loaded SpeechLLM/SALM, on CPU and already at its final
            compute dtype -- sharding measures module sizes to plan the split,
            so it has to weigh what will actually occupy the GPUs.
        reserve_fraction: Share of each GPU's *currently free* memory left
            unassigned to weights, for activations and the KV cache during
            generation. Free, not total, so this plays fair with anything
            else already resident on a shared box.

    Returns:
        The model, dispatched across ``torch.cuda.device_count()`` GPUs (see
        step 5 above for the ``.device`` patch).

    Raises:
        RuntimeError: If no CUDA device is visible.
        AttributeError: If the loaded model's LLM is not shaped like a
            standard HuggingFace decoder-only model (``model.llm.model.layers``).
    """
    import torch
    from accelerate import dispatch_model

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError('device_map="auto" needs at least one visible CUDA device to shard across.')

    def module_bytes(m) -> int:
        """How many bytes *m*'s parameters and buffers take up right now."""
        return sum(p.numel() * p.element_size() for p in m.parameters()) + sum(
            b.numel() * b.element_size() for b in m.buffers()
        )

    def get(dotted: str):
        """Resolve a dotted name like "llm.model.norm" to the actual submodule."""
        m = model
        for part in dotted.split("."):
            m = getattr(m, part)
        return m

    # Step 1: weigh the pieces (skip an anchor the model doesn't have -- some
    # checkpoints may not carry every one of _ANCHOR_MODULES).
    anchors = [name for name in _ANCHOR_MODULES if hasattr(model, name.split(".")[0])]
    anchor_bytes = sum(module_bytes(get(name)) for name in anchors)
    layer_bytes = [module_bytes(layer) for layer in model.llm.model.layers]
    # Step 2: how much room is actually free on each GPU right now.
    budgets = [int(torch.cuda.mem_get_info(i)[0] * (1 - reserve_fraction)) for i in range(n_gpus)]

    # Step 3: get the plan, step 4: carry it out for real.
    device_map = _plan_device_map(anchors, anchor_bytes, layer_bytes, budgets)
    model = dispatch_model(model, device_map=device_map)
    # Step 5: fix the stale `.device` -- point it at wherever the anchors
    # (and so layer 0, always grouped with them) actually ended up.
    model._device = next(get(anchors[0]).parameters()).device
    return model


def _load_speechllm(
    model_name: str,
    model_config_override: str | None = None,
    map_location: str | None = None,
    **model_args: Any,
):
    """Load a SMURF checkpoint the way ``fbk_speechllm.inference`` does.

    ``register_custom_perception_modules()`` has to run first: the training
    configs name ``fbk_speechllm.modules.perception.AudioPerceptionModule`` as
    a Hydra ``_target_``, and NeMo refuses to instantiate a target whose prefix
    is not allowlisted. Without it the load fails inside NeMo with a message
    about the target rather than about the missing registration.

    Args:
        model_name: Path to a ``.ckpt`` file or to a ``save_pretrained`` directory.
        model_config_override: Optional ``hparams.yaml``-shaped config that
            replaces the one stored in the checkpoint.
        map_location: Passed through to Lightning's checkpoint loader.
        **model_args: Forwarded to the loader.

    Returns:
        The loaded model, still on its load device and in its stored dtype.

    Raises:
        ImportError: If neither ``fbk_speechllm`` nor NeMo's ``SALM`` can be
            imported, naming what the environment is missing.
        FileNotFoundError: If *model_name* does not exist.
        ValueError: If a config override is given for a checkpoint that has no
            stored config to override.
    """
    model_class = _speechllm_class()

    path = Path(model_name)
    if not path.exists():
        raise FileNotFoundError(
            f"No SMURF checkpoint at {model_name}. Expected a Lightning .ckpt file, or a "
            "directory written by save_pretrained()."
        )

    if path.is_dir():
        # NeMo's HFHubMixin layout (config + weights), rather than a training
        # checkpoint. There is no Lightning state to restore, and the model
        # config comes from the directory, so an override has nothing to apply to.
        if model_config_override:
            raise ValueError(
                "model_config_override applies to a Lightning .ckpt, whose stored config it "
                f"replaces. {model_name} is a save_pretrained() directory, which carries its own "
                "config; edit that instead."
            )
        return model_class.from_pretrained(model_name, **model_args)

    override_args: dict[str, Any] = {}
    if model_config_override:
        from lightning.pytorch.core.saving import load_hparams_from_yaml

        # Read as a plain dict, not OmegaConf: SALM asserts its config is a
        # dict so that Lightning can serialise the hyperparameters back out.
        override_args = load_hparams_from_yaml(model_config_override, use_omegaconf=False)
        logger.info("Overriding the checkpoint's model config with %s", model_config_override)

    if map_location is not None:
        model_args["map_location"] = map_location
    return model_class.load_from_checkpoint(model_name, **override_args, **model_args)


def _speechllm_class():
    """Return the model class to load SMURF checkpoints with.

    Prefers ``fbk_speechllm``'s ``SpeechLLM``, which is what wrote them. NeMo's
    ``SALM`` is accepted as a fallback because the subclass adds only training
    -side behaviour (checkpoint saving, extra logging) -- nothing generation
    reads -- so a checkpoint whose config does not reference the project's own
    perception module still evaluates without the private package installed.
    """
    try:
        from fbk_speechllm.models.speech_llm import SpeechLLM
        from fbk_speechllm.utils import register_custom_perception_modules

        register_custom_perception_modules()
        return SpeechLLM
    except ImportError as fbk_error:
        try:
            from nemo.collections.speechlm2.models.salm import SALM
        except ImportError as nemo_error:
            raise ImportError(
                "Loading a SMURF checkpoint needs fbk_speechllm (preferred) or, at minimum, "
                "nemo_toolkit's speechlm2 collection. Neither imported: "
                f"fbk_speechllm -> {fbk_error}; nemo -> {nemo_error}. The SMURF stack does not "
                "co-install with melt-proj (different torch/transformers pins), so this provider "
                "expects its own environment."
            ) from nemo_error

        logger.warning(
            "fbk_speechllm is not installed (%s); falling back to NeMo's SALM. A checkpoint whose "
            "config names fbk_speechllm.modules.* will fail to load.",
            fbk_error,
        )
        return SALM


def _with_audio_tag(text: str, tag: str, placement: str) -> str:
    """Ensure the prompt contains the model's audio placeholder exactly once.

    The solver renders the instruction; where the audio goes is a property of
    the checkpoint, so it is applied here. A prompt that already carries the
    tag is left alone -- that is how a per-sample instruction from a benchmark
    keeps control of the position.
    """
    if tag in text:
        return text
    return f"{text} {tag}" if placement == "suffix" else f"{tag} {text}"


def _collate_audio(batch: list[_Request], sampling_rate: int):
    """Pad a batch's waveforms into ``(audios, audio_lens)`` tensors.

    Args:
        batch: The requests to collate.
        sampling_rate: The rate the model's feature extractor expects.

    Returns:
        A float32 ``(B, T)`` tensor of zero-padded waveforms and an int64
        ``(B,)`` tensor of true lengths in samples.

    Raises:
        ValueError: If a request has no audio, or its audio is not at the
            model's sampling rate. Both are silent-wrong-answer failures
            otherwise: a missing waveform would leave the placeholder token
            unexpanded, and a rate mismatch would feed the encoder a signal
            with the wrong duration and pitch and still produce fluent text.
    """
    import torch

    missing = [i for i, r in enumerate(batch) if r.audio is None]
    if missing:
        raise ValueError(
            f"{len(missing)} sample(s) in the batch carry no audio. The SMURF provider generates "
            "from a prompt whose audio placeholder is expanded by the speech encoder, so a "
            "text-only sample has nothing to expand it with."
        )

    wrong_rate = {r.sample_rate for r in batch if r.sample_rate != sampling_rate}
    if wrong_rate:
        raise ValueError(
            f"Audio at {sorted(wrong_rate)} Hz, but the checkpoint's feature extractor expects "
            f"{sampling_rate} Hz. Readers resample on load; a mismatch here means the frozen set "
            "references audio a reader passed through untouched."
        )

    waveforms = [torch.as_tensor(r.audio, dtype=torch.float32).reshape(-1) for r in batch]
    audio_lens = torch.tensor([w.numel() for w in waveforms], dtype=torch.int64)
    audios = torch.zeros(len(waveforms), int(audio_lens.max()), dtype=torch.float32)
    for row, waveform in enumerate(waveforms):
        audios[row, : waveform.numel()] = waveform
    return audios, audio_lens


def _generation_kwargs(config: GenerateConfig, num_beams: int = 1) -> dict[str, Any]:
    """Translate an inspect generate config into ``GenerationConfig`` fields.

    Greedy by default, for the same reason as the MELT provider: an eval that
    samples is measuring the sampler as much as the model, and two runs would
    not be comparable.
    """
    kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_tokens or 256,
        "do_sample": config.temperature is not None and config.temperature > 0,
    }
    if num_beams > 1:
        kwargs["num_beams"] = num_beams
    if config.temperature is not None and config.temperature > 0:
        kwargs["temperature"] = config.temperature
    if config.top_p is not None:
        kwargs["top_p"] = config.top_p
    if config.top_k is not None:
        kwargs["top_k"] = config.top_k
    if config.num_choices is not None and config.num_choices > 1:
        kwargs["num_return_sequences"] = config.num_choices
    return kwargs
