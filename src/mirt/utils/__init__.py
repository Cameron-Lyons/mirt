"""Public utility API with on-demand imports.

Utility implementations and numerical dependencies remain deferred until a
public symbol is accessed.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_IMPORTS = {
    "sigmoid": "mirt._core",
    "simdata": "mirt.utils.simulation",
    "validate_responses": "mirt.utils.data",
    "set_dataframe_backend": "mirt.utils.dataframe",
    "get_dataframe_backend": "mirt.utils.dataframe",
    "rotate_loadings": "mirt.utils.rotation",
    "varimax": "mirt.utils.rotation",
    "promax": "mirt.utils.rotation",
    "oblimin": "mirt.utils.rotation",
    "apply_rotation_to_model": "mirt.utils.rotation",
    "get_rotated_loadings": "mirt.utils.rotation",
    "cross_validate": "mirt.utils.cv",
    "CVResult": "mirt.utils.cv",
    "Splitter": "mirt.utils.cv",
    "GroupKFold": "mirt.utils.cv",
    "KFold": "mirt.utils.cv",
    "StratifiedGroupKFold": "mirt.utils.cv",
    "StratifiedKFold": "mirt.utils.cv",
    "LeaveOneOut": "mirt.utils.cv",
    "Scorer": "mirt.utils.cv",
    "LogLikelihoodScorer": "mirt.utils.cv",
    "AICScorer": "mirt.utils.cv",
    "BICScorer": "mirt.utils.cv",
    "fit_models": "mirt.utils.batch",
    "fit_model_grid": "mirt.utils.batch",
    "BatchFitResult": "mirt.utils.batch",
    "GridFitResult": "mirt.utils.batch",
    "testinfo": "mirt.utils.information",
    "iteminfo": "mirt.utils.information",
    "areainfo": "mirt.utils.information",
    "probtrace": "mirt.utils.information",
    "expected_score": "mirt.utils.information",
    "expected_test_score": "mirt.utils.information",
    "gen_difficulty": "mirt.utils.information",
    "theta_for_score": "mirt.utils.information",
    "information_intervals": "mirt.utils.information",
    "marginal_rxx": "mirt.utils.reliability",
    "empirical_rxx": "mirt.utils.reliability",
    "sem": "mirt.utils.reliability",
    "cv_select_lambda": "mirt.utils.regularization_cv",
    "information_criteria_path": "mirt.utils.regularization_cv",
    "RegularizationCVResult": "mirt.utils.regularization_cv",
    "traditional": "mirt.utils.classical",
    "TraditionalStats": "mirt.utils.classical",
    "item_fit_chisq": "mirt.utils.classical",
    "itemstats": "mirt.utils.classical",
    "ItemStats": "mirt.utils.classical",
    "itemstats_to_dataframe": "mirt.utils.classical",
    "wald": "mirt.utils.statistical_tests",
    "lagrange": "mirt.utils.statistical_tests",
    "likelihood_ratio": "mirt.utils.statistical_tests",
    "WaldTestResult": "mirt.utils.statistical_tests",
    "LagrangeTestResult": "mirt.utils.statistical_tests",
    "mod2values": "mirt.utils.extraction",
    "extract_item": "mirt.utils.extraction",
    "coef": "mirt.utils.extraction",
    "estfun": "mirt.utils.extraction",
    "estfun_summary": "mirt.utils.extraction",
    "itemplot_data": "mirt.utils.extraction",
    "ModelValues": "mirt.utils.extraction",
    "ItemParameters": "mirt.utils.extraction",
    "MDIFF": "mirt.utils.multidimensional",
    "MDISC": "mirt.utils.multidimensional",
    "direction_cosines": "mirt.utils.multidimensional",
    "composite_score_weights": "mirt.utils.multidimensional",
    "empirical_ES": "mirt.utils.empirical",
    "empirical_plot": "mirt.utils.empirical",
    "empirical_rmsea": "mirt.utils.empirical",
    "mantel_haenszel": "mirt.utils.empirical",
    "RMSD_DIF": "mirt.utils.empirical",
    "weighted_RMSD_DIF": "mirt.utils.empirical",
    "DIFEffectSize": "mirt.utils.empirical",
    "EmpiricalPlotData": "mirt.utils.empirical",
    "itemGAM": "mirt.utils.empirical",
    "ItemGAMResult": "mirt.utils.empirical",
    "RCI": "mirt.utils.clinical",
    "RCIResult": "mirt.utils.clinical",
    "clinical_significance": "mirt.utils.clinical",
    "residuals": "mirt.utils.residuals",
    "ResidualResult": "mirt.utils.residuals",
    "Q3": "mirt.utils.residuals",
    "LD_X2": "mirt.utils.residuals",
    "fixed_calib": "mirt.utils.calibration",
    "equate": "mirt.utils.calibration",
    "transform_theta": "mirt.utils.calibration",
    "CalibrationResult": "mirt.utils.calibration",
    "EquatingResult": "mirt.utils.calibration",
    "PLCI": "mirt.utils.confidence",
    "PLCIResult": "mirt.utils.confidence",
    "score_CI": "mirt.utils.confidence",
    "delta_method": "mirt.utils.confidence",
    "CollapsedData": "mirt.utils.collapse",
    "collapse_patterns": "mirt.utils.collapse",
    "collapse_with_groups": "mirt.utils.collapse",
    "compute_pattern_likelihood": "mirt.utils.collapse",
    "weighted_sum_from_collapsed": "mirt.utils.collapse",
    "key2binary": "mirt.utils.transform",
    "poly2dich": "mirt.utils.transform",
    "reverse_score": "mirt.utils.transform",
    "expand_table": "mirt.utils.transform",
    "collapse_table": "mirt.utils.transform",
    "recode_responses": "mirt.utils.transform",
    "likert2int": "mirt.utils.transform",
    "draw_parameters": "mirt.utils.sampling",
    "ParameterSamples": "mirt.utils.sampling",
    "posterior_summary": "mirt.utils.sampling",
    "sample_expected_scores": "mirt.utils.sampling",
    "missing_patterns": "mirt.utils.imputation",
    "MissingPatternResult": "mirt.utils.imputation",
    "randef": "mirt.utils.predictions",
    "fixef": "mirt.utils.predictions",
    "predict_mixed": "mirt.utils.predictions",
    "conditional_effects": "mirt.utils.predictions",
    "shrinkage_estimates": "mirt.utils.predictions",
    "RandomEffects": "mirt.utils.predictions",
    "FixedEffects": "mirt.utils.predictions",
    "gen_random_pars": "mirt.utils.starting",
    "calc_null": "mirt.utils.starting",
    "multi_start_fit": "mirt.utils.starting",
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str) -> Any:
    """Resolve and cache a public utility symbol on first access."""
    try:
        module_name = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return loaded attributes together with deferred public symbols."""
    return sorted(set(globals()) | set(__all__))
