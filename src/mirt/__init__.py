from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from mirt._gpu_backend import (
    GPU_AVAILABLE,
    get_gpu_device_name,
    get_gpu_memory_info,
    is_gpu_available,  # noqa: F401
    is_torch_available,
)
from mirt._rust_backend import RUST_AVAILABLE, is_rust_available  # noqa: F401
from mirt._version import __version__
from mirt.cat import CATEngine, CATResult, CATState
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
from mirt.equating import (
    AnchorDiagnostics,
    ChainLinkingResult,
    DriftResult,
    LinkingConstants,
    LinkingFitStatistics,
    LinkingResult,
    PolytomousLinkingResult,
    ProcrustesResult,
    ScoreEquatingResult,
    TimePointModel,
    accumulate_constants,
    bootstrap_linking_se,
    chain_link,
    chain_linking_summary,
    compare_linking_methods,
    compute_linking_fit,
    compute_mirt_linking_fit,
    compute_see,
    concurrent_link,
    delta_method_se,
    detect_drift,
    detect_longitudinal_drift,
    equipercentile_equating,
    factor_congruence_coefficient,
    link,
    link_gpcm,
    link_grm,
    link_mirt,
    link_nrm,
    linking_summary,
    lord_wingersky_recursion,
    match_factors,
    mirt_linking_summary,
    oblique_procrustes_rotation,
    observed_score_equating,
    orthogonal_procrustes_rotation,
    parameter_recovery_summary,
    purify_anchors,
    score_equating_summary,
    score_to_theta,
    signed_area_difference,
    target_rotation,
    theta_to_score,
    transform_mirt_parameters,
    transform_parameters,
    transform_polytomous_parameters,
    transform_theta_to_reference,
    transform_to_reference,
    true_score_equating,
)
from mirt.estimation.bl import BLEstimator
from mirt.estimation.em import EMEstimator
from mirt.estimation.latent_density import EmpiricalHistogramWoods
from mirt.estimation.mcmc import GibbsSampler, MCMCResult, MHRMEstimator
from mirt.estimation.mixed import LLTM, MixedEffectsFitResult, MixedEffectsIRT
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.se_methods import SEMethod, compute_se
from mirt.exceptions import (
    MirtConvergenceError,
    MirtDataError,
    MirtError,
    MirtEstimationError,
    MirtModelError,
    MirtValidationError,
)
from mirt.models.base import BaseItemModel
from mirt.models.bifactor import BifactorModel
from mirt.models.cdm import DINA, DINO, BaseCDM, fit_cdm
from mirt.models.custom import (
    CustomGroupModel,
    CustomItemModel,
    GroupSpec,
    ItemTypeSpec,
    create_item_type,
    createGroup,
    get_standard_item_type,
    list_standard_item_types,
)
from mirt.models.dichotomous import (
    FourParameterLogistic,
    OneParameterLogistic,
    Rasch,
    ThreeParameterLogistic,
    ThreeParameterLogisticUpper,
    TwoParameterLogistic,
    UnipolarLogLogistic,
)
from mirt.models.mixture import MixtureIRT, fit_mixture_irt
from mirt.models.multidimensional import MultidimensionalModel
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedRatingScaleModel,
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
from mirt.multigroup import (
    GroupLatentDistribution,
    InvarianceSpec,
    InvarianceTestResult,
    MultigroupEMEstimator,
    MultigroupFitResult,
    MultigroupLatentDensity,
    MultigroupModel,
    ParameterLink,
    compare_invariance,
    fit_multigroup,
    invariance_lrt,
    test_invariance_hierarchy,
)
from mirt.results.fit_result import FitResult
from mirt.results.score_result import ScoreResult
from mirt.scoring import fscores
from mirt.utils.batch import BatchFitResult, fit_models
from mirt.utils.bootstrap import bootstrap_ci, bootstrap_se, parametric_bootstrap
from mirt.utils.calibration import equate, fixed_calib
from mirt.utils.classical import ItemStats, TraditionalStats, itemstats, traditional
from mirt.utils.clinical import RCI, clinical_significance
from mirt.utils.collapse import CollapsedData, collapse_patterns
from mirt.utils.confidence import PLCI, score_CI
from mirt.utils.cv import CVResult, KFold, LeaveOneOut, StratifiedKFold, cross_validate
from mirt.utils.data import validate_responses
from mirt.utils.dataframe import set_dataframe_backend
from mirt.utils.datasets import list_datasets, load_dataset
from mirt.utils.empirical import (
    RMSD_DIF,
    ItemGAMResult,
    empirical_ES,
    empirical_plot,
    itemGAM,
)
from mirt.utils.extraction import coef, estfun, estfun_summary, extract_item, mod2values
from mirt.utils.imputation import (
    MIResult,
    analyze_missing,
    averageMI,
    impute_responses,
    listwise_deletion,
)
from mirt.utils.information import (
    areainfo,
    gen_difficulty,
    iteminfo,
    probtrace,
    testinfo,
)
from mirt.utils.multidimensional import MDIFF, MDISC
from mirt.utils.plausible import (
    combine_plausible_values,
    generate_plausible_values,
    plausible_value_regression,
    plausible_value_statistics,
)
from mirt.utils.predictions import fixef, randef
from mirt.utils.reliability import empirical_rxx, marginal_rxx, sem
from mirt.utils.residuals import Q3, residuals
from mirt.utils.sampling import draw_parameters
from mirt.utils.simulation import generate_item_parameters, simdata
from mirt.utils.starting import calc_null, gen_random_pars, multi_start_fit
from mirt.utils.statistical_tests import lagrange, wald
from mirt.utils.transform import (
    expand_table,
    key2binary,
    poly2dich,
    reverse_score,
)

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

try:
    from mirt.reports import (  # noqa: F401
        DIFAnalysisReport,
        FullDiagnosticReport,
        ItemAnalysisReport,
        ModelFitReport,
        generate_report,
    )

    _HAS_REPORTS = True
except ImportError:
    _HAS_REPORTS = False


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
    """Fit an Item Response Theory model to response data.

    This is the main function for estimating IRT model parameters using
    the EM algorithm with marginal maximum likelihood estimation.

    Parameters
    ----------
    data : ndarray of shape (n_persons, n_items)
        Response matrix. Missing responses should be coded as -1.
        For dichotomous models, responses should be 0 or 1.
        For polytomous models, responses should be 0, 1, ..., n_categories-1.
    model : {"1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM"}, default="2PL"
        IRT model to fit:

        - "1PL": One-parameter logistic (Rasch-like with common discrimination)
        - "2PL": Two-parameter logistic
        - "3PL": Three-parameter logistic (with guessing)
        - "4PL": Four-parameter logistic (with guessing and slipping)
        - "GRM": Graded Response Model (polytomous)
        - "GPCM": Generalized Partial Credit Model (polytomous)
        - "PCM": Partial Credit Model (polytomous)
        - "NRM": Nominal Response Model (polytomous)

    n_factors : int, default=1
        Number of latent factors for multidimensional models.
    n_categories : int, optional
        Number of response categories for polytomous models.
        If None, inferred from data.
    estimation : {"EM"}, default="EM"
        Estimation method. Currently only EM is supported.
    n_quadpts : int, default=21
        Number of quadrature points for numerical integration.
    max_iter : int, default=500
        Maximum number of EM iterations.
    tol : float, default=1e-4
        Convergence tolerance for parameter change.
    verbose : bool, default=False
        Print iteration progress.
    item_names : list of str, optional
        Names for each item. If None, items are named Item_1, Item_2, etc.
    use_rust : bool, default=True
        Use high-performance Rust backend if available.

    Returns
    -------
    FitResult
        Object containing:

        - model: The fitted IRT model with estimated parameters
        - log_likelihood: Final marginal log-likelihood
        - n_iterations: Number of EM iterations
        - converged: Whether convergence was achieved
        - standard_errors: Parameter standard errors
        - aic, bic: Information criteria

    Raises
    ------
    ValueError
        If data is not 2D or model type is unknown.

    Examples
    --------
    >>> from mirt import fit_mirt, simdata
    >>> # Simulate some response data
    >>> data = simdata(n_persons=500, n_items=20)
    >>> # Fit a 2PL model
    >>> result = fit_mirt(data, model="2PL")
    >>> print(f"Log-likelihood: {result.log_likelihood:.2f}")
    >>> print(result.model.parameters)
    """
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
    """Compute item fit statistics for a fitted IRT model.

    Item fit statistics assess how well individual items conform to the
    assumed IRT model. Poor-fitting items may indicate violations of
    model assumptions or problematic item content.

    Parameters
    ----------
    result : FitResult
        A fitted IRT model result from fit_mirt().
    responses : ndarray of shape (n_persons, n_items), optional
        Response data used for fit calculation. If None, uses the data
        from model fitting.
    statistics : list of str, optional
        Fit statistics to compute. Options include:

        - "infit": Information-weighted mean square (sensitive to
          unexpected responses near ability level)
        - "outfit": Unweighted mean square (sensitive to outliers)
        - "S_X2": Orlando-Thissen S-X2 statistic

        Default is ["infit", "outfit"].

    Returns
    -------
    DataFrame
        Item fit statistics with items as rows and statistics as columns.
        Includes fit statistic values and standardized z-scores.

    Examples
    --------
    >>> from mirt import fit_mirt, itemfit, simdata
    >>> data = simdata(n_persons=500, n_items=20)
    >>> result = fit_mirt(data)
    >>> fit_stats = itemfit(result)
    >>> # Flag items with infit > 1.2 or < 0.8
    >>> print(fit_stats[(fit_stats['infit'] > 1.2) | (fit_stats['infit'] < 0.8)])
    """
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
    """Compute person fit statistics to detect aberrant response patterns.

    Person fit statistics identify individuals whose response patterns
    are inconsistent with the IRT model, which may indicate careless
    responding, cheating, or other forms of aberrant behavior.

    Parameters
    ----------
    result : FitResult
        A fitted IRT model result from fit_mirt().
    responses : ndarray of shape (n_persons, n_items)
        Response matrix. Missing responses should be coded as -1.
    theta : ndarray of shape (n_persons,) or (n_persons, n_factors), optional
        Ability estimates. If None, computed using EAP scoring.
    statistics : list of str, optional
        Person fit statistics to compute. Options include:

        - "infit": Information-weighted mean square
        - "outfit": Unweighted mean square
        - "Zh": Standardized log-likelihood (Drasgow et al.)
        - "lz": Log-likelihood z-score

        Default is ["infit", "outfit", "Zh"].

    Returns
    -------
    DataFrame
        Person fit statistics with persons as rows and statistics as columns.

    Notes
    -----
    - Zh values below -2 may indicate aberrant responding
    - Infit/outfit values should be close to 1.0 (range 0.7-1.3 acceptable)
    - High outfit indicates unexpected responses to easy/hard items
    - High infit indicates inconsistent responses near ability level

    Examples
    --------
    >>> from mirt import fit_mirt, personfit, simdata
    >>> data = simdata(n_persons=500, n_items=20)
    >>> result = fit_mirt(data)
    >>> pfit = personfit(result, data)
    >>> # Flag potentially aberrant responders
    >>> aberrant = pfit[pfit['Zh'] < -2]
    >>> print(f"Flagged {len(aberrant)} aberrant responders")
    """
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
    "MirtError",
    "MirtValidationError",
    "MirtEstimationError",
    "MirtConvergenceError",
    "MirtModelError",
    "MirtDataError",
    "CATEngine",
    "CATResult",
    "CATState",
    "OneParameterLogistic",
    "TwoParameterLogistic",
    "ThreeParameterLogistic",
    "FourParameterLogistic",
    "ThreeParameterLogisticUpper",
    "UnipolarLogLogistic",
    "Rasch",
    "GradedResponseModel",
    "GradedRatingScaleModel",
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
    "BLEstimator",
    "GaussHermiteQuadrature",
    "EmpiricalHistogramWoods",
    "compute_se",
    "SEMethod",
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
    "cross_validate",
    "CVResult",
    "KFold",
    "StratifiedKFold",
    "LeaveOneOut",
    "fit_models",
    "BatchFitResult",
    "testinfo",
    "iteminfo",
    "areainfo",
    "probtrace",
    "gen_difficulty",
    "marginal_rxx",
    "empirical_rxx",
    "sem",
    "traditional",
    "TraditionalStats",
    "wald",
    "lagrange",
    "mod2values",
    "extract_item",
    "coef",
    "MDIFF",
    "MDISC",
    "empirical_ES",
    "empirical_plot",
    "RMSD_DIF",
    "RCI",
    "clinical_significance",
    "residuals",
    "Q3",
    "fixed_calib",
    "equate",
    "PLCI",
    "score_CI",
    "key2binary",
    "poly2dich",
    "reverse_score",
    "expand_table",
    "draw_parameters",
    "collapse_patterns",
    "CollapsedData",
    "gen_random_pars",
    "calc_null",
    "multi_start_fit",
    "randef",
    "fixef",
    "itemGAM",
    "ItemGAMResult",
    "averageMI",
    "MIResult",
    "itemstats",
    "ItemStats",
    "estfun",
    "estfun_summary",
    "createGroup",
    "GroupSpec",
    "CustomGroupModel",
    "CustomItemModel",
    "ItemTypeSpec",
    "create_item_type",
    "get_standard_item_type",
    "list_standard_item_types",
    "fit_multigroup",
    "compare_invariance",
    "test_invariance_hierarchy",
    "MultigroupFitResult",
    "MultigroupModel",
    "MultigroupEMEstimator",
    "MultigroupLatentDensity",
    "GroupLatentDistribution",
    "InvarianceSpec",
    "InvarianceTestResult",
    "ParameterLink",
    "invariance_lrt",
    "link",
    "transform_parameters",
    "LinkingConstants",
    "LinkingResult",
    "LinkingFitStatistics",
    "AnchorDiagnostics",
    "detect_drift",
    "purify_anchors",
    "signed_area_difference",
    "DriftResult",
    "bootstrap_linking_se",
    "delta_method_se",
    "compute_linking_fit",
    "linking_summary",
    "compare_linking_methods",
    "parameter_recovery_summary",
    "link_grm",
    "link_gpcm",
    "link_nrm",
    "transform_polytomous_parameters",
    "PolytomousLinkingResult",
    "link_mirt",
    "orthogonal_procrustes_rotation",
    "oblique_procrustes_rotation",
    "transform_mirt_parameters",
    "factor_congruence_coefficient",
    "match_factors",
    "compute_mirt_linking_fit",
    "target_rotation",
    "mirt_linking_summary",
    "ProcrustesResult",
    "true_score_equating",
    "observed_score_equating",
    "lord_wingersky_recursion",
    "equipercentile_equating",
    "score_to_theta",
    "theta_to_score",
    "score_equating_summary",
    "compute_see",
    "ScoreEquatingResult",
    "chain_link",
    "accumulate_constants",
    "transform_to_reference",
    "transform_theta_to_reference",
    "concurrent_link",
    "chain_linking_summary",
    "detect_longitudinal_drift",
    "ChainLinkingResult",
    "TimePointModel",
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

__all__.extend(
    [
        "GPU_AVAILABLE",
        "RUST_AVAILABLE",
        "is_gpu_available",
        "is_rust_available",
        "is_torch_available",
        "get_gpu_device_name",
        "get_gpu_memory_info",
        "get_backend_info",
        "set_backend",
        "get_backend",
    ]
)


_CURRENT_BACKEND: Literal["auto", "gpu", "rust", "numpy"] = "auto"


def set_backend(backend: Literal["auto", "gpu", "rust", "numpy"]) -> None:
    """Set the computational backend for IRT operations.

    Parameters
    ----------
    backend : {"auto", "gpu", "rust", "numpy"}
        Backend to use:

        - "auto": Automatically select the fastest available backend.
          Priority: GPU > Rust > NumPy
        - "gpu": Use PyTorch GPU acceleration (requires torch with CUDA).
        - "rust": Use Rust backend (requires compiled extension).
        - "numpy": Use pure NumPy implementation (always available).

    Raises
    ------
    ValueError
        If requested backend is not available.

    Examples
    --------
    >>> import mirt
    >>> mirt.set_backend("gpu")  # Use GPU if available
    >>> mirt.set_backend("auto")  # Auto-select best backend
    """
    global _CURRENT_BACKEND

    if backend not in ("auto", "gpu", "rust", "numpy"):
        raise ValueError(
            f"Invalid backend '{backend}'. Must be one of: 'auto', 'gpu', 'rust', 'numpy'"
        )

    if backend == "gpu" and not GPU_AVAILABLE:
        raise ValueError(
            "GPU backend requested but not available. "
            "Install PyTorch with CUDA support: pip install torch"
        )

    if backend == "rust" and not RUST_AVAILABLE:
        raise ValueError(
            "Rust backend requested but not available. "
            "Ensure the package was installed with Rust extension."
        )

    _CURRENT_BACKEND = backend


def get_backend() -> Literal["auto", "gpu", "rust", "numpy"]:
    """Get the currently configured backend.

    Returns
    -------
    str
        Current backend setting.
    """
    return _CURRENT_BACKEND


def get_backend_info() -> dict[str, Any]:
    """Get information about available computational backends.

    Returns
    -------
    dict
        Dictionary containing:

        - current_backend: Currently selected backend
        - effective_backend: Backend that will actually be used (for "auto")
        - gpu_available: Whether GPU acceleration is available
        - gpu_device: Name of GPU device (if available)
        - gpu_memory: GPU memory info (if available)
        - rust_available: Whether Rust backend is available
        - torch_available: Whether PyTorch is installed

    Examples
    --------
    >>> import mirt
    >>> info = mirt.get_backend_info()
    >>> print(f"Using: {info['effective_backend']}")
    >>> if info['gpu_available']:
    ...     print(f"GPU: {info['gpu_device']}")
    """
    effective = _CURRENT_BACKEND
    if effective == "auto":
        if GPU_AVAILABLE:
            effective = "gpu"
        elif RUST_AVAILABLE:
            effective = "rust"
        else:
            effective = "numpy"

    return {
        "current_backend": _CURRENT_BACKEND,
        "effective_backend": effective,
        "gpu_available": GPU_AVAILABLE,
        "gpu_device": get_gpu_device_name(),
        "gpu_memory": get_gpu_memory_info(),
        "rust_available": RUST_AVAILABLE,
        "torch_available": is_torch_available(),
    }
