from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from mirt._version import __version__
from mirt.diagnostics.comparison import (
    anova_irt,
    compare_models,
    information_criteria,
    vuong_test,
)
from mirt.diagnostics.drf import compute_drf, compute_item_drf, reliability_invariance
from mirt.diagnostics.dtf import compute_dtf
from mirt.diagnostics.modelfit import compute_fit_indices, compute_m2
from mirt.diagnostics.sibtest import sibtest, sibtest_items
from mirt.estimation.em import EMEstimator
from mirt.estimation.mcmc import GibbsSampler, MCMCResult, MHRMEstimator
from mirt.estimation.mixed import LLTM, MixedEffectsFitResult, MixedEffectsIRT
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.models.base import BaseItemModel
from mirt.models.bifactor import BifactorModel
from mirt.models.cdm import DINA, DINO, BaseCDM, fit_cdm
from mirt.models.dichotomous import (
    FourParameterLogistic,
    OneParameterLogistic,
    Rasch,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.mixture import MixtureIRT, fit_mixture_irt
from mirt.models.multidimensional import MultidimensionalModel
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
)
from mirt.models.testlet import TestletModel, create_testlet_structure
from mirt.models.unfolding import (
    GeneralizedGradedUnfolding,
    HyperbolicCosineModel,
    IdealPointModel,
)
from mirt.models.zeroinflated import HurdleIRT, ZeroInflated2PL, ZeroInflated3PL
from mirt.results.fit_result import FitResult
from mirt.results.score_result import ScoreResult
from mirt.scoring import fscores
from mirt.utils.bootstrap import bootstrap_ci, bootstrap_se, parametric_bootstrap
from mirt.utils.data import validate_responses
from mirt.utils.dataframe import set_dataframe_backend
from mirt.utils.datasets import list_datasets, load_dataset
from mirt.utils.imputation import analyze_missing, impute_responses, listwise_deletion
from mirt.utils.plausible import (
    combine_plausible_values,
    generate_plausible_values,
    plausible_value_regression,
    plausible_value_statistics,
)
from mirt.utils.simulation import generate_item_parameters, simdata

try:
    from mirt.plotting import (  # noqa: F401
        plot_ability_distribution,
        plot_dif,
        plot_expected_score,
        plot_icc,
        plot_information,
        plot_itemfit,
        plot_person_item_map,
        plot_se,
    )

    _HAS_PLOTTING = True
except ImportError:
    _HAS_PLOTTING = False


def fit_mirt(
    data: NDArray[np.int_],
    model: Literal["1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM"] = "2PL",
    n_factors: int = 1,
    n_categories: int | None = None,
    estimation: Literal["EM"] = "EM",
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    item_names: list[str] | None = None,
    use_rust: bool = True,
) -> FitResult:
    from mirt._rust_backend import RUST_AVAILABLE, em_fit_2pl

    data = np.asarray(data)

    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got {data.ndim}D")

    n_persons, n_items = data.shape

    if item_names is None:
        item_names = [f"Item_{i + 1}" for i in range(n_items)]

    is_polytomous = model in ("GRM", "GPCM", "PCM", "NRM")

    if is_polytomous:
        if n_categories is None:
            n_categories = int(data[data >= 0].max()) + 1
        if n_categories < 2:
            raise ValueError("n_categories must be at least 2")

    if (
        use_rust
        and RUST_AVAILABLE
        and model == "2PL"
        and n_factors == 1
        and estimation == "EM"
    ):
        discrimination, difficulty, log_likelihood, n_iterations, converged = (
            em_fit_2pl(data, n_quadpts=n_quadpts, max_iter=max_iter, tol=tol)
        )

        irt_model = TwoParameterLogistic(
            n_items=n_items, n_factors=n_factors, item_names=item_names
        )
        irt_model._parameters = {
            "discrimination": np.asarray(discrimination),
            "difficulty": np.asarray(difficulty),
        }
        irt_model._is_fitted = True

        n_params = 2 * n_items
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + np.log(n_persons) * n_params

        return FitResult(
            model=irt_model,
            log_likelihood=log_likelihood,
            n_iterations=n_iterations,
            converged=converged,
            standard_errors={
                "discrimination": np.full(n_items, np.nan),
                "difficulty": np.full(n_items, np.nan),
            },
            aic=aic,
            bic=bic,
            n_observations=n_persons,
            n_parameters=n_params,
        )

    if model == "1PL":
        irt_model = OneParameterLogistic(n_items=n_items, item_names=item_names)
    elif model == "2PL":
        irt_model = TwoParameterLogistic(
            n_items=n_items, n_factors=n_factors, item_names=item_names
        )
    elif model == "3PL":
        irt_model = ThreeParameterLogistic(n_items=n_items, item_names=item_names)
    elif model == "4PL":
        irt_model = FourParameterLogistic(n_items=n_items, item_names=item_names)
    elif model == "GRM":
        assert n_categories is not None
        irt_model = GradedResponseModel(
            n_items=n_items,
            n_categories=n_categories,
            n_factors=n_factors,
            item_names=item_names,
        )
    elif model == "GPCM":
        assert n_categories is not None
        irt_model = GeneralizedPartialCredit(
            n_items=n_items,
            n_categories=n_categories,
            n_factors=n_factors,
            item_names=item_names,
        )
    elif model == "PCM":
        assert n_categories is not None
        irt_model = PartialCreditModel(
            n_items=n_items,
            n_categories=n_categories,
            item_names=item_names,
        )
    elif model == "NRM":
        assert n_categories is not None
        irt_model = NominalResponseModel(
            n_items=n_items,
            n_categories=n_categories,
            n_factors=n_factors,
            item_names=item_names,
        )
    else:
        raise ValueError(f"Unknown model: {model}")

    if estimation == "EM":
        estimator = EMEstimator(
            n_quadpts=n_quadpts,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown estimation method: {estimation}")

    result = estimator.fit(irt_model, data)

    return result


def itemfit(
    result: FitResult,
    responses: NDArray[np.int_] | None = None,
    statistics: list[str] | None = None,
) -> Any:
    from mirt.diagnostics.itemfit import compute_itemfit
    from mirt.utils.dataframe import create_dataframe

    if statistics is None:
        statistics = ["infit", "outfit"]

    fit_stats = compute_itemfit(result.model, responses, statistics)

    return create_dataframe(fit_stats, index=result.model.item_names, index_name="item")


def personfit(
    result: FitResult,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
    statistics: list[str] | None = None,
) -> Any:
    from mirt.diagnostics.personfit import compute_personfit
    from mirt.utils.dataframe import create_dataframe

    if statistics is None:
        statistics = ["infit", "outfit", "Zh"]

    if theta is None:
        score_result = fscores(result, responses, method="EAP")
        theta = score_result.theta

    fit_stats = compute_personfit(result.model, responses, theta, statistics)

    return create_dataframe(fit_stats, index_name="person")


def dif(
    data: NDArray[np.int_],
    groups: NDArray[np.int_] | NDArray[np.str_],
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM"] = "2PL",
    method: Literal["likelihood_ratio", "wald", "lord", "raju"] = "likelihood_ratio",
    n_categories: int | None = None,
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    focal_group: str | int | None = None,
) -> Any:
    """Compute Differential Item Functioning (DIF) statistics.

    DIF analysis tests whether items function differently across groups
    after controlling for ability level.

    Args:
        data: Response matrix (n_persons x n_items).
        groups: Group membership array (n_persons,). Must have exactly 2 groups.
        model: IRT model type.
        method: DIF detection method:
            - 'likelihood_ratio': Likelihood ratio test (recommended)
            - 'wald': Wald test on parameter differences
            - 'lord': Lord's chi-square test
            - 'raju': Raju's area measures
        n_categories: Number of categories for polytomous models.
        n_quadpts: Number of quadrature points for EM.
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance.
        focal_group: Which group to use as focal (default: second unique group).

    Returns:
        DataFrame with DIF statistics for each item:
            - statistic: Test statistic
            - p_value: P-value
            - effect_size: Effect size measure
            - classification: ETS classification (A/B/C)
    """
    from mirt.diagnostics.dif import compute_dif
    from mirt.utils.dataframe import create_dataframe

    dif_results = compute_dif(
        data=data,
        groups=groups,
        model=model,
        method=method,
        n_categories=n_categories,
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        focal_group=focal_group,
    )

    return create_dataframe(dif_results, index_name="item")


__all__ = [
    "__version__",
    "fit_mirt",
    "fscores",
    "simdata",
    "itemfit",
    "personfit",
    "dif",
    "OneParameterLogistic",
    "TwoParameterLogistic",
    "ThreeParameterLogistic",
    "FourParameterLogistic",
    "Rasch",
    "GradedResponseModel",
    "GeneralizedPartialCredit",
    "PartialCreditModel",
    "NominalResponseModel",
    "MultidimensionalModel",
    "BifactorModel",
    "BaseCDM",
    "DINA",
    "DINO",
    "fit_cdm",
    "TestletModel",
    "create_testlet_structure",
    "ZeroInflated2PL",
    "ZeroInflated3PL",
    "HurdleIRT",
    "GeneralizedGradedUnfolding",
    "IdealPointModel",
    "HyperbolicCosineModel",
    "MixtureIRT",
    "fit_mixture_irt",
    "BaseItemModel",
    "EMEstimator",
    "GaussHermiteQuadrature",
    "MHRMEstimator",
    "GibbsSampler",
    "MCMCResult",
    "MixedEffectsIRT",
    "LLTM",
    "MixedEffectsFitResult",
    "FitResult",
    "ScoreResult",
    "compute_m2",
    "compute_fit_indices",
    "anova_irt",
    "compare_models",
    "vuong_test",
    "information_criteria",
    "compute_dtf",
    "compute_drf",
    "compute_item_drf",
    "reliability_invariance",
    "sibtest",
    "sibtest_items",
    "generate_item_parameters",
    "validate_responses",
    "set_dataframe_backend",
    "load_dataset",
    "list_datasets",
    "bootstrap_se",
    "bootstrap_ci",
    "parametric_bootstrap",
    "impute_responses",
    "analyze_missing",
    "listwise_deletion",
    "generate_plausible_values",
    "combine_plausible_values",
    "plausible_value_regression",
    "plausible_value_statistics",
]

if _HAS_PLOTTING:
    __all__.extend(
        [
            "plot_icc",
            "plot_information",
            "plot_ability_distribution",
            "plot_itemfit",
            "plot_person_item_map",
            "plot_dif",
            "plot_expected_score",
            "plot_se",
        ]
    )
