"""melt-eval: downstream evaluation for MELT speech models on inspect_ai.

Importing this package registers its tasks, solvers, scorers and model
providers with inspect's registry. It is wired to the ``inspect_ai``
entry-point group, whose loader does exactly one thing on discovery --
``import melteval`` -- so every registration side effect (``@task``,
``@scorer``, ``@modelapi``, …) has to happen by the time *this* module finishes
executing, however deep in the package it lives. A ``@modelapi``-decorated
class in a submodule nobody imports never runs its decorator, and ``--model
melt/<checkpoint>`` fails with "Model API melt not recognized" despite the file
existing right there in ``providers/``. Hence the otherwise-unused imports of
``providers.melt`` and ``providers.smurf`` below.

Nothing heavy is imported at module scope: both providers defer torch,
transformers and NeMo into their methods, so registering them costs nothing
until a model is actually loaded. This matters more than it looks: the two
model stacks do not co-install, so a SMURF environment must be able to import
this package without MELT's dependencies present, and vice versa.
"""

from melteval.dataset import frozen_dataset
from melteval.manifest import AudioLocator, EvalRecord
from melteval.prompt import FormatSpec, SmurfPromptSpec, load_format_spec, load_smurf_prompt_spec
from melteval.providers.melt import MELTAPI
from melteval.providers.smurf import SmurfAPI
from melteval.scorers import asr_scorer, corpus_bleu, corpus_cer, corpus_chrf, corpus_wer, st_scorer
from melteval.solver import smurf_prompt, speech_prompt
from melteval.tasks import asr, speech, st


__version__ = "0.1.0"

__all__ = [
    "MELTAPI",
    "AudioLocator",
    "EvalRecord",
    "FormatSpec",
    "SmurfAPI",
    "SmurfPromptSpec",
    "asr",
    "asr_scorer",
    "corpus_bleu",
    "corpus_cer",
    "corpus_chrf",
    "corpus_wer",
    "frozen_dataset",
    "load_format_spec",
    "load_smurf_prompt_spec",
    "smurf_prompt",
    "speech",
    "speech_prompt",
    "st",
    "st_scorer",
]
