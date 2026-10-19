"""Post-session analytics: cumulative records and scallop-model fitter.

The submodules import ``matplotlib`` and ``scipy`` lazily inside their
functions, so the analytics subpackage can be imported without those
optional plotting / fitting dependencies installed.
"""

from ai_operant_box.analytics.fi_measures import FISummary, summarize_fi
from ai_operant_box.analytics.matching import (
    ChoiceTotals,
    GeneralizedMatchingFit,
    choice_totals,
    fit_generalized_matching,
)
from ai_operant_box.analytics.model_fit import ModelFitAnalyzer, scallop_model
from ai_operant_box.analytics.record import (
    CumulativeRecorder,
    LiveVisualizer,
    MultiSessionRecorder,
)

__all__ = [
    "ChoiceTotals",
    "CumulativeRecorder",
    "FISummary",
    "GeneralizedMatchingFit",
    "LiveVisualizer",
    "ModelFitAnalyzer",
    "MultiSessionRecorder",
    "choice_totals",
    "fit_generalized_matching",
    "scallop_model",
    "summarize_fi",
]
