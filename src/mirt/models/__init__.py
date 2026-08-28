"""Public model API with on-demand imports.

Importing this namespace keeps numerical dependencies and individual model
modules deferred until one of their public symbols is accessed.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_IMPORTS = {
    "BaseItemModel": "mirt.models.base",
    "DichotomousItemModel": "mirt.models.base",
    "PolytomousItemModel": "mirt.models.base",
    "OneParameterLogistic": "mirt.models.dichotomous",
    "TwoParameterLogistic": "mirt.models.dichotomous",
    "ThreeParameterLogistic": "mirt.models.dichotomous",
    "FourParameterLogistic": "mirt.models.dichotomous",
    "FiveParameterLogistic": "mirt.models.dichotomous",
    "Rasch": "mirt.models.dichotomous",
    "ComplementaryLogLog": "mirt.models.dichotomous",
    "NegativeLogLog": "mirt.models.dichotomous",
    "LLTM": "mirt.models.explanatory",
    "LLTMResult": "mirt.models.explanatory",
    "RaschLLTM": "mirt.models.explanatory",
    "LatentRegressionModel": "mirt.models.explanatory",
    "LatentRegressionResult": "mirt.models.explanatory",
    "ExplanatoryIRT": "mirt.models.explanatory",
    "ExplanatoryIRTResult": "mirt.models.explanatory",
    "TreeNode": "mirt.models.irtree",
    "IRTreeSpec": "mirt.models.irtree",
    "IRTreeModel": "mirt.models.irtree",
    "Facet": "mirt.models.mfrm",
    "ManyFacetRaschModel": "mirt.models.mfrm",
    "MFRMResult": "mirt.models.mfrm",
    "PolytomousMFRM": "mirt.models.mfrm",
    "MultilevelIRTModel": "mirt.models.multilevel",
    "MultilevelIRTResult": "mirt.models.multilevel",
    "ThreeLevelIRTModel": "mirt.models.multilevel",
    "CrossedRandomEffectsModel": "mirt.models.multilevel",
    "RandomEffectSpec": "mirt.models.multilevel",
    "NestedHierarchy": "mirt.models.multilevel",
    "GradedResponseModel": "mirt.models.polytomous",
    "GeneralizedPartialCredit": "mirt.models.polytomous",
    "PartialCreditModel": "mirt.models.polytomous",
    "RatingScaleModel": "mirt.models.polytomous",
    "NominalResponseModel": "mirt.models.polytomous",
    "SequentialResponseModel": "mirt.models.sequential",
    "ContinuationRatioModel": "mirt.models.sequential",
    "AdjacentCategoryModel": "mirt.models.sequential",
    "TwoPLNestedLogit": "mirt.models.nested",
    "ThreePLNestedLogit": "mirt.models.nested",
    "FourPLNestedLogit": "mirt.models.nested",
    "MultidimensionalModel": "mirt.models.multidimensional",
    "BifactorModel": "mirt.models.bifactor",
    "GDINA": "mirt.models.cdm_advanced",
    "HigherOrderCDM": "mirt.models.cdm_advanced",
    "AttributeHierarchy": "mirt.models.cdm_advanced",
    "fit_gdina": "mirt.models.cdm_advanced",
    "PartiallyCompensatoryModel": "mirt.models.compensatory",
    "NoncompensatoryModel": "mirt.models.compensatory",
    "DisjunctiveModel": "mirt.models.compensatory",
    "MonotonicSplineModel": "mirt.models.nonparametric",
    "MonotonicPolynomialModel": "mirt.models.nonparametric",
    "KernelSmoothingModel": "mirt.models.nonparametric",
    "TestletModel": "mirt.models.testlet",
    "BifactorTestletModel": "mirt.models.testlet",
    "RandomTestletEffectsModel": "mirt.models.testlet",
    "create_testlet_structure": "mirt.models.testlet",
    "compute_testlet_q3": "mirt.models.testlet",
    "CustomItemModel": "mirt.models.custom",
    "CustomGroupModel": "mirt.models.custom",
    "ItemTypeSpec": "mirt.models.custom",
    "GroupSpec": "mirt.models.custom",
    "create_item_type": "mirt.models.custom",
    "create_group": "mirt.models.custom",
    "createGroup": "mirt.models.custom",
    "get_standard_item_type": "mirt.models.custom",
    "list_standard_item_types": "mirt.models.custom",
    "IsingModel": "mirt.models.network",
    "GaussianGraphicalModel": "mirt.models.network",
    "fit_ising": "mirt.models.network",
    "fit_ggm": "mirt.models.network",
    "compare_networks": "mirt.models.network",
    "BKTModel": "mirt.models.dynamic",
    "BKTResult": "mirt.models.dynamic",
    "BKTStepResult": "mirt.models.dynamic",
    "BKTBatchStepResult": "mirt.models.dynamic",
    "BKTForecastResult": "mirt.models.dynamic",
    "BKTBatchForecastResult": "mirt.models.dynamic",
    "BKTMasteryTargetResult": "mirt.models.dynamic",
    "BKTBatchMasteryTargetResult": "mirt.models.dynamic",
    "BKTPredictiveResult": "mirt.models.dynamic",
    "BKTBatchPredictiveResult": "mirt.models.dynamic",
    "BKTSkillRankingResult": "mirt.models.dynamic",
    "BKTBatchSkillRankingResult": "mirt.models.dynamic",
    "LongitudinalIRTModel": "mirt.models.dynamic",
    "LongitudinalResult": "mirt.models.dynamic",
    "StateSpaceIRT": "mirt.models.state_space",
    "StateSpaceStepResult": "mirt.models.state_space",
    "StateSpaceBatchStepResult": "mirt.models.state_space",
    "StateSpacePredictiveResult": "mirt.models.state_space",
    "StateSpaceBatchPredictiveResult": "mirt.models.state_space",
    "StateSpaceForecastResult": "mirt.models.state_space",
    "StateSpaceBatchForecastResult": "mirt.models.state_space",
    "PiecewiseGrowthModel": "mirt.models.dynamic",
    "NonlinearGrowthModel": "mirt.models.dynamic",
    "GrowthMixtureModel": "mirt.models.dynamic",
    "GrowthMixtureResult": "mirt.models.dynamic",
    "ResponseTimeModel": "mirt.models.response_time",
    "ResponseTimeResult": "mirt.models.response_time",
}

__all__ = list(_LAZY_IMPORTS)


def __getattr__(name: str) -> Any:
    """Resolve and cache a public model symbol on first access."""
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
