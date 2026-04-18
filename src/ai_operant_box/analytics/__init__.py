"""Post-session analytics: cumulative records and scallop-model fitter.

The submodules import ``matplotlib`` and ``scipy`` lazily inside their
functions, so the analytics subpackage can be imported without those
optional plotting / fitting dependencies installed.
"""

from ai_operant_box.analytics.model_fit import ModelFitAnalyzer, scallop_model
from ai_operant_box.analytics.record import (
    CumulativeRecorder,
    LiveVisualizer,
    MultiSessionRecorder,
)

__all__ = [
    "CumulativeRecorder",
    "LiveVisualizer",
    "ModelFitAnalyzer",
    "MultiSessionRecorder",
    "scallop_model",
]
