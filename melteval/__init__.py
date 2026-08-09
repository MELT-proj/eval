"""melt-eval: downstream evaluation for MELT speech models on inspect_ai.

Importing this package registers its tasks, solver, scorers and model providers
with inspect's registry. It is wired to the ``inspect_ai`` entry-point group, so
``inspect eval melteval/speech --model melt/<checkpoint>`` resolves without an
explicit import.

Nothing heavy is imported at module scope: the MELT provider defers torch and
transformers into its methods, so registering a provider costs nothing until a
model is actually loaded.
"""

from melteval.dataset import frozen_dataset
from melteval.manifest import AudioLocator, EvalRecord
from melteval.prompt import FormatSpec, load_format_spec
from melteval.solver import speech_prompt
from melteval.tasks import speech


__version__ = "0.1.0"

__all__ = [
    "AudioLocator",
    "EvalRecord",
    "FormatSpec",
    "frozen_dataset",
    "load_format_spec",
    "speech",
    "speech_prompt",
]
