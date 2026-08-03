from typing import Any

from mirt._core import sigmoid
from mirt.utils.batch import BatchFitResult, fit_model_grid, fit_models
from mirt.utils.calibration import (
    CalibrationResult,
    EquatingResult,
    equate,
    fixed_calib,
    transform_theta,
)
from mirt.utils.classical import TraditionalStats, item_fit_chisq, traditional
from mirt.utils.clinical import RCI, RCIResult, clinical_significance
from mirt.utils.confidence import PLCI, PLCIResult, delta_method, score_CI
from mirt.utils.cv import (
    AICScorer,
    BICScorer,
    CVResult,
    KFold,
    LeaveOneOut,
    LogLikelihoodScorer,
    Scorer,
    Splitter,
    StratifiedKFold,
    cross_validate,
)
from mirt.utils.data import validate_responses
from mirt.utils.dataframe import set_dataframe_backend
from mirt.utils.empirical import (
    RMSD_DIF,
    DIFEffectSize,
    EmpiricalPlotData,
    empirical_ES,
    empirical_plot,
    empirical_rmsea,
    mantel_haenszel,
    weighted_RMSD_DIF,
)
from mirt.utils.extraction import (
    ItemParameters,
    ModelValues,
    coef,
    extract_item,
    itemplot_data,
    mod2values,
)
from mirt.utils.information import (
    areainfo,
    expected_score,
    expected_test_score,
    gen_difficulty,
    iteminfo,
    probtrace,
    testinfo,
    theta_for_score,
)
from mirt.utils.multidimensional import (
    MDIFF,
    MDISC,
    composite_score_weights,
    direction_cosines,
)
from mirt.utils.predictions import (
    FixedEffects,
    RandomEffects,
    conditional_effects,
    fixef,
    predict_mixed,
    randef,
    shrinkage_estimates,
)
from mirt.utils.reliability import empirical_rxx, marginal_rxx, sem
from mirt.utils.residuals import LD_X2, Q3, ResidualResult, residuals
from mirt.utils.rotation import (
    apply_rotation_to_model,
    get_rotated_loadings,
    oblimin,
    promax,
    rotate_loadings,
    varimax,
)
from mirt.utils.sampling import (
    ParameterSamples,
    draw_parameters,
    posterior_summary,
    sample_expected_scores,
)
from mirt.utils.simulation import simdata
from mirt.utils.statistical_tests import (
    LagrangeTestResult,
    WaldTestResult,
    lagrange,
    likelihood_ratio,
    wald,
)
from mirt.utils.transform import (
    collapse_table,
    expand_table,
    key2binary,
    likert2int,
    poly2dich,
    recode_responses,
    reverse_score,
)

__all__ = [
    "sigmoid",
    "simdata",
    "validate_responses",
    "set_dataframe_backend",
    "rotate_loadings",
    "varimax",
    "promax",
    "oblimin",
    "apply_rotation_to_model",
    "get_rotated_loadings",
    "cross_validate",
    "CVResult",
    "Splitter",
    "KFold",
    "StratifiedKFold",
    "LeaveOneOut",
    "Scorer",
    "LogLikelihoodScorer",
    "AICScorer",
    "BICScorer",
    "fit_models",
    "fit_model_grid",
    "BatchFitResult",
    "testinfo",
    "iteminfo",
    "areainfo",
    "probtrace",
    "expected_score",
    "expected_test_score",
    "gen_difficulty",
    "theta_for_score",
    "marginal_rxx",
    "empirical_rxx",
    "sem",
    "cv_select_lambda",
    "information_criteria_path",
    "RegularizationCVResult",
    "traditional",
    "TraditionalStats",
    "item_fit_chisq",
    "wald",
    "lagrange",
    "likelihood_ratio",
    "WaldTestResult",
    "LagrangeTestResult",
    "mod2values",
    "extract_item",
    "coef",
    "itemplot_data",
    "ModelValues",
    "ItemParameters",
    "MDIFF",
    "MDISC",
    "direction_cosines",
    "composite_score_weights",
    "empirical_ES",
    "empirical_plot",
    "empirical_rmsea",
    "mantel_haenszel",
    "RMSD_DIF",
    "weighted_RMSD_DIF",
    "DIFEffectSize",
    "EmpiricalPlotData",
    "RCI",
    "RCIResult",
    "clinical_significance",
    "residuals",
    "ResidualResult",
    "Q3",
    "LD_X2",
    "fixed_calib",
    "equate",
    "transform_theta",
    "CalibrationResult",
    "EquatingResult",
    "PLCI",
    "PLCIResult",
    "score_CI",
    "delta_method",
    "key2binary",
    "poly2dich",
    "reverse_score",
    "expand_table",
    "collapse_table",
    "recode_responses",
    "likert2int",
    "draw_parameters",
    "ParameterSamples",
    "posterior_summary",
    "sample_expected_scores",
    "randef",
    "fixef",
    "predict_mixed",
    "conditional_effects",
    "shrinkage_estimates",
    "RandomEffects",
    "FixedEffects",
]

_LAZY_EXPORTS = {
    "RegularizationCVResult": "mirt.utils.regularization_cv",
    "cv_select_lambda": "mirt.utils.regularization_cv",
    "information_criteria_path": "mirt.utils.regularization_cv",
}


def __getattr__(name: str) -> Any:
    """Load estimation-backed utilities without creating import cycles."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy utility exports in interactive discovery."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
