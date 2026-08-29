from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Literal

from mirt._api_registry import MODULE_EXPORTS, build_all_exports, build_lazy_imports
from mirt._version import __version__

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from mirt.estimation.mcmc import MCMCResult
    from mirt.results.fit_result import FitResult


def fit_mirt(
    data: NDArray[np.int_],
    model: Literal["1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM"] = "2PL",
    n_factors: int = 1,
    n_categories: int | None = None,
    estimation: Literal["EM", "MHRM", "MCMC", "Gibbs"] = "EM",
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    item_names: list[str] | None = None,
    use_rust: bool = True,
) -> FitResult:
    """Fit an Item Response Theory model to response data.

    This is the main function for estimating IRT model parameters.
    Default estimation uses the EM algorithm with marginal maximum
    likelihood; MHRM and Gibbs/MCMC are also available.

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
    estimation : {"EM", "MHRM", "MCMC", "Gibbs"}, default="EM"
        Estimation method. "MCMC" and "Gibbs" are aliases for Gibbs sampling;
        results are returned as a FitResult with posterior-mean parameters and
        chain standard deviations as standard errors.
    n_quadpts : int, default=21
        Number of quadrature points for numerical integration (EM only).
    max_iter : int, default=500
        Maximum number of EM iterations (EM) or MHRM cycles / MCMC iterations
        depending on method.
    tol : float, default=1e-4
        Convergence tolerance for parameter change (EM).
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
    MirtDataError
        If data is not 2D.
    MirtValidationError
        If model type or estimation method is unknown, or polytomous
        category count is invalid.
    MirtModelError
        If the requested model cannot be constructed.

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
    import numpy as np

    from mirt._backend_config import should_use_rust
    from mirt._rust_backend import (
        compute_item_se_parallel,
        e_step_complete,
        em_fit_2pl,
    )
    from mirt.estimation.em import EMEstimator
    from mirt.estimation.mcmc import GibbsSampler, MHRMEstimator
    from mirt.estimation.quadrature import GaussHermiteQuadrature
    from mirt.exceptions import MirtDataError, MirtModelError, MirtValidationError
    from mirt.models.dichotomous import (
        FourParameterLogistic,
        OneParameterLogistic,
        ThreeParameterLogistic,
        TwoParameterLogistic,
    )
    from mirt.models.polytomous import (
        GeneralizedPartialCredit,
        GradedResponseModel,
        NominalResponseModel,
        PartialCreditModel,
    )
    from mirt.results.fit_result import FitResult
    from mirt.typing import EstimationMethod
    from mirt.utils.data import validate_responses

    supported_models = ("1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM")
    if model not in supported_models:
        raise MirtModelError(f"Unknown model: {model}", model_type=str(model))

    data = validate_responses(data)

    n_persons, n_items = data.shape

    if item_names is None:
        item_names = [f"Item_{i + 1}" for i in range(n_items)]

    is_polytomous = model in ("GRM", "GPCM", "PCM", "NRM")

    if is_polytomous:
        if n_categories is None:
            observed = data[data >= 0]
            if observed.size == 0:
                raise MirtValidationError(
                    "n_categories is required when all responses are missing",
                    parameter="n_categories",
                    expected=">= 2",
                )
            n_categories = int(observed.max()) + 1
        if n_categories < 2:
            raise MirtValidationError(
                "n_categories must be at least 2",
                parameter="n_categories",
                value=n_categories,
                expected=">= 2",
            )
        if np.any(data[data >= 0] >= n_categories):
            raise MirtDataError(
                "polytomous response codes must be below n_categories",
                n_persons=n_persons,
                n_items=n_items,
            )
    elif np.any(data[data >= 0] > 1):
        raise MirtDataError(
            "dichotomous responses must be coded as 0 or 1",
            n_persons=n_persons,
            n_items=n_items,
        )

    estimation_method: EstimationMethod = estimation

    if (
        should_use_rust(use_rust)
        and model == "2PL"
        and n_factors == 1
        and estimation_method == "EM"
    ):
        discrimination, difficulty, log_likelihood, n_iterations, converged = (
            em_fit_2pl(data, n_quadpts=n_quadpts, max_iter=max_iter, tol=tol)
        )

        irt_model = TwoParameterLogistic(
            n_items=n_items, n_factors=n_factors, item_names=item_names
        )
        discrimination = np.asarray(discrimination)
        difficulty = np.asarray(difficulty)
        irt_model._parameters = {
            "discrimination": discrimination,
            "difficulty": difficulty,
        }
        irt_model._is_fitted = True

        quad = GaussHermiteQuadrature(n_points=n_quadpts, n_dimensions=1)
        posterior_weights, _ = e_step_complete(
            data,
            quad.nodes.ravel(),
            quad.weights.ravel(),
            discrimination,
            difficulty,
        )
        se_a, se_b = compute_item_se_parallel(
            data,
            posterior_weights,
            quad.nodes.ravel(),
            discrimination,
            difficulty,
        )

        n_params = 2 * n_items
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + np.log(n_persons) * n_params

        return FitResult(
            model=irt_model,
            log_likelihood=log_likelihood,
            n_iterations=n_iterations,
            converged=converged,
            standard_errors={
                "discrimination": np.asarray(se_a),
                "difficulty": np.asarray(se_b),
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
    if estimation_method == "EM":
        estimator = EMEstimator(
            n_quadpts=n_quadpts,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            use_rust=use_rust,
        )
        return estimator.fit(irt_model, data)

    if estimation_method == "MHRM":
        return MHRMEstimator(
            n_cycles=max_iter,
            verbose=verbose,
            use_rust=use_rust,
        ).fit(irt_model, data)

    if estimation_method in ("MCMC", "Gibbs"):
        burnin = min(1000, max(max_iter // 5, 1))
        n_iter = max(max_iter, burnin + 10)
        mcmc = GibbsSampler(
            n_iter=n_iter,
            burnin=burnin,
            verbose=verbose,
            use_rust=use_rust,
        ).fit(irt_model, data)
        return _mcmc_result_to_fit_result(mcmc, n_persons)

    raise MirtValidationError(
        f"Unknown estimation method: {estimation}",
        parameter="estimation",
        value=estimation,
        expected="EM, MHRM, MCMC, or Gibbs",
    )


def _mcmc_result_to_fit_result(mcmc: MCMCResult, n_persons: int) -> FitResult:
    """Adapt MCMCResult to FitResult for a uniform fit_mirt return type."""
    import numpy as np

    from mirt.results.fit_result import FitResult

    model = mcmc.model
    n_params = model.n_parameters
    standard_errors: dict[str, NDArray[np.float64]] = {}
    for name, chain in mcmc.chains.items():
        if name in ("theta", "log_likelihood"):
            continue
        arr = np.asarray(chain, dtype=np.float64)
        if arr.ndim >= 2:
            standard_errors[name] = np.std(arr, axis=0, ddof=1)
        else:
            standard_errors[name] = np.array([float(np.std(arr, ddof=1))])

    for name, values in model.parameters.items():
        if name not in standard_errors:
            standard_errors[name] = np.full(values.shape, np.nan)

    aic = -2 * mcmc.log_likelihood + 2 * n_params
    bic = -2 * mcmc.log_likelihood + np.log(max(n_persons, 1)) * n_params
    converged = bool(mcmc.rhat) and all(r < 1.1 for r in mcmc.rhat.values())

    return FitResult(
        model=model,
        log_likelihood=mcmc.log_likelihood,
        n_iterations=mcmc.n_iterations,
        converged=converged,
        standard_errors=standard_errors,
        aic=aic,
        bic=bic,
        n_observations=n_persons,
        n_parameters=n_params,
    )


def itemfit(
    result: FitResult,
    responses: NDArray[np.int_] | None = None,
    statistics: list[str] | None = None,
    n_groups: int = 10,
    p_adjust: Literal["bonferroni", "holm", "fdr_bh", "none"] = "none",
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
    n_groups : int
        Number of observed-score groups used for S-X2. Must be at least 2.
        Default is 10.
    p_adjust : {"bonferroni", "holm", "fdr_bh", "none"}, default="none"
        Multiple-testing adjustment across item-level S-X2 p-values. When an
        adjustment is requested, the result includes a
        ``p_value_adjusted`` column while retaining the raw ``p_value``.

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
    >>> fit_stats = itemfit(result, data)
    >>> # Flag items with infit > 1.2 or < 0.8
    >>> print(fit_stats[(fit_stats['infit'] > 1.2) | (fit_stats['infit'] < 0.8)])
    """
    from mirt.diagnostics.itemfit import compute_itemfit
    from mirt.utils.dataframe import create_dataframe

    if statistics is None:
        statistics = ["infit", "outfit"]

    fit_stats = compute_itemfit(
        result.model,
        responses,
        statistics,
        n_groups=n_groups,
        p_adjust=p_adjust,
    )

    return create_dataframe(fit_stats, index=result.model.item_names, index_name="item")


def personfit(
    result: FitResult,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
    statistics: list[str] | None = None,
    *,
    p_adjust: Literal["none", "bonferroni", "holm", "fdr_bh"] | None = None,
    alpha: float = 0.05,
    alternative: Literal["lower", "two-sided", "upper"] = "lower",
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
    p_adjust : {"none", "bonferroni", "holm", "fdr_bh"}, optional
        Enable person-fit p-values and flags, optionally correcting across
        respondents. ``None`` keeps the default output unchanged; ``"none"``
        enables significance output without multiplicity correction.
    alpha : float, default=0.05
        Significance threshold for the ``aberrant`` column when ``p_adjust`` is
        enabled.
    alternative : {"lower", "two-sided", "upper"}, default="lower"
        Normal-tail alternative for standardized log-likelihood scores.

    Returns
    -------
    DataFrame
        Person fit statistics with persons as rows and statistics as columns.
        When ``p_adjust`` is supplied, raw and adjusted p-values plus an
        ``aberrant`` flag are included.

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
    >>> pfit = personfit(result, data, p_adjust="holm")
    >>> # Flag potentially aberrant responders with family-wise control
    >>> aberrant = pfit[pfit['aberrant']]
    >>> print(f"Flagged {len(aberrant)} aberrant responders")
    """
    from mirt.diagnostics.personfit import compute_personfit
    from mirt.scoring import fscores
    from mirt.utils.dataframe import create_dataframe

    if statistics is None:
        statistics = ["infit", "outfit", "Zh"]

    if theta is None:
        score_result = fscores(result, responses, method="EAP")
        theta = score_result.theta

    fit_stats = compute_personfit(
        result.model,
        responses,
        theta,
        statistics,
        p_adjust=p_adjust,
        alpha=alpha,
        alternative=alternative,
    )

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
    p_adjust: Literal["none", "bonferroni", "holm", "fdr_bh"] = "none",
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
        p_adjust: Multiple-testing adjustment across items. Default 'none'.

    Returns:
        DataFrame with DIF statistics for each item:
            - statistic: Test statistic
            - p_value: P-value
            - p_value_adjusted: Multiplicity-adjusted P-value
            - effect_size: Effect size measure
            - classification: ETS classification using adjusted P-values
            - adjustment: Multiple-testing method
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
        p_adjust=p_adjust,
    )

    return create_dataframe(dif_results, index_name="item")


__all__ = build_all_exports()
_MODULE_EXPORTS = MODULE_EXPORTS
_LAZY_IMPORTS = build_lazy_imports()


def __getattr__(name: str) -> Any:
    if name in _MODULE_EXPORTS:
        module = importlib.import_module(_MODULE_EXPORTS[name])
        globals()[name] = module
        return module

    if name in _LAZY_IMPORTS:
        module_name, symbol_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, symbol_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module 'mirt' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
