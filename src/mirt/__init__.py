from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from mirt._api_registry import MODULE_EXPORTS, build_all_exports, build_lazy_imports
from mirt._version import __version__

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray

    from mirt.estimation._em_context import EMFitContext
    from mirt.estimation._shared_step import EqualityConstraints
    from mirt.estimation.mcmc import MCMCResult
    from mirt.estimation.priors import Prior, PriorSpecification
    from mirt.model_syntax import ModelSpec
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult

    _ItemFamily = Literal["1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM"]


def fit_mirt(
    data: NDArray[np.int_] | Any,
    model: _ItemFamily | Sequence[_ItemFamily] = "2PL",
    n_factors: int = 1,
    n_categories: int | Sequence[int] | None = None,
    estimation: Literal["EM", "MHRM", "MCMC", "Gibbs"] = "EM",
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    verbose: bool = False,
    item_names: list[str] | None = None,
    use_rust: bool = True,
    compute_standard_errors: bool = True,
    start_values: Mapping[str, ArrayLike] | None = None,
    fixed: Mapping[str, ArrayLike] | None = None,
    priors: PriorSpecification | Mapping[str, Prior] | None = None,
    se_method: Literal[
        "auto", "oakes", "crossprod", "sandwich", "complete_data"
    ] = "auto",
    spec: ModelSpec | str | None = None,
    accelerate: Literal["none", "squarem"] = "none",
    constraints: EqualityConstraints | None = None,
) -> FitResult:
    """Fit an Item Response Theory model to response data.

    This is the main function for estimating IRT model parameters.
    Default estimation uses the EM algorithm with marginal maximum
    likelihood; MHRM and Gibbs/MCMC are also available.

    Parameters
    ----------
    data : ndarray or DataFrame of shape (n_persons, n_items)
        Response matrix. Missing responses are coded as any negative value or
        ``NaN`` (including pandas/polars nulls).
        For dichotomous models, responses should be 0 or 1.
        For polytomous models, responses should be 0, 1, ..., n_categories-1.
    model : str or sequence of str, default="2PL"
        IRT model to fit:

        - "1PL": One-parameter logistic (Rasch-like with common discrimination)
        - "2PL": Two-parameter logistic
        - "3PL": Three-parameter logistic (with guessing)
        - "4PL": Four-parameter logistic (with guessing and slipping)
        - "GRM": Graded Response Model (polytomous)
        - "GPCM": Generalized Partial Credit Model (polytomous)
        - "PCM": Partial Credit Model (polytomous)
        - "NRM": Nominal Response Model (polytomous)

        A sequence names one family per item for a mixed-format test, for
        example ``["3PL"] * 20 + ["GRM"] * 5``. It fits a
        :class:`~mirt.models.mixed_format.MixedItemModel` by
        :class:`~mirt.estimation.mixed_format_em.MixedFormatEMEstimator`,
        whose parameters are qualified by family, such as ``"3PL.guessing"``,
        in ``start_values``, ``fixed`` and the results. Mixed formats require
        ``estimation="EM"``; a sequence naming one family fits that family.

    n_factors : int, default=1
        Number of latent factors. Only "2PL", "GRM", "GPCM" and "NRM" support
        more than one factor; other families raise ``MirtModelError``.
    n_categories : int or sequence of int, optional
        Category count for all polytomous items, or one count per item.
        If None, each item's count is inferred from its largest observed code,
        with a minimum of two. Wholly unobserved items require explicit counts.
        For a mixed-format test, a sequence gives 2 for dichotomous items.
    estimation : {"EM", "MHRM", "MCMC", "Gibbs"}, default="EM"
        Estimation method. "MCMC" and "Gibbs" are aliases for Gibbs sampling;
        results are returned as a FitResult with posterior-mean parameters and
        chain standard deviations as standard errors. MHRM standard errors
        are observed-information errors for the families that
        ``se_method="auto"`` gives ``"oakes"`` under EM, and otherwise the
        spread of the Robbins-Monro iterates (``se_method="mhrm_iterate_sd"``,
        see :class:`~mirt.estimation.mcmc.MHRMEstimator`).
    n_quadpts : int, default=21
        Number of quadrature points for numerical integration (EM, and the
        observed-information standard errors of MHRM).
    max_iter : int, default=500
        Maximum number of EM iterations (EM) or MHRM cycles / MCMC iterations
        depending on method.
    tol : float, default=1e-4
        Convergence tolerance for parameter change (EM).
    verbose : bool, default=False
        Print iteration progress.
    item_names : list of str, optional
        Names for each item. If None, unique DataFrame column names are used
        when available; otherwise items are named Item_1, Item_2, etc.
    use_rust : bool, default=True
        Use high-performance Rust backend if available.
    compute_standard_errors : bool, default=True
        Compute parameter standard errors. Set to ``False`` for parameter-only
        fits such as bootstrap replicates, where skipping inference reduces
        repeated work. The result then contains an empty standard-error mapping.
    start_values : mapping of str to array_like, optional
        Starting values by stored parameter name, for example
        ``{"discrimination": a0}``. Parameters left out start from the family
        defaults. Values also set the coordinates held fixed by ``fixed``.
    fixed : mapping of str to bool or array of bool, optional
        Coordinates to hold at their starting values, by parameter name.
        ``True`` fixes a coordinate; a scalar applies to the whole parameter.
        Supported by EM; MHRM and Gibbs sampling raise ``MirtValidationError``.
    priors : PriorSpecification or mapping of str to Prior, optional
        Item-parameter priors for Bayes modal estimation with EM, for example
        ``{"guessing": BetaPrior(5, 17)}``. See ``EMEstimator(item_priors=...)``.
        The result reports ``log_posterior``, and standard errors add the
        prior's negative second derivative to the information, so they reflect
        the curvature of the log-posterior. Starting values, fixed coordinates
        and priors bypass the native 2PL EM fast path, and ``start_values``
        runs MHRM and Gibbs sampling with the NumPy samplers.
    se_method : {"auto", "oakes", "crossprod", "sandwich", "complete_data"}, \
default="auto"
        Standard-error estimator for EM fits (see :class:`EMEstimator`).
        ``"oakes"`` uses the observed information of the marginal likelihood,
        ``"crossprod"`` the outer product of person scores, ``"sandwich"``
        their robust combination, and ``"complete_data"`` itemwise
        complete-data curvature, which understates uncertainty. ``"auto"``
        selects ``"oakes"`` for unidimensional 1PL-4PL, GRM, GPCM and PCM fits
        and ``"complete_data"`` otherwise. The matrix methods also store the
        parameter covariance in ``FitResult.vcov``, and ``FitResult.se_method``
        records the estimator used. Other estimation methods accept only
        ``"auto"``.
    spec : ModelSpec or str, optional
        Confirmatory structure from :func:`mirt_model`, or model syntax that
        is parsed against the item names. Items load only on their factors,
        ``COV`` correlations are estimated and reported as
        ``FitResult.latent_covariance``, and ``FIXED``, ``START``,
        ``PRIOR`` and ``CONSTRAIN`` act like ``fixed``, ``start_values``,
        ``priors`` and ``constraints``, which can still be combined with them
        (priors only from one source). Multiple factors are supported for
        "2PL" (fitted as a slope-intercept :class:`MultidimensionalModel`),
        "GRM" and "GPCM"; one factor for every family. Requires EM estimation.
        A per-item ``model`` sequence must name one family; mixed families
        raise ``MirtValidationError``.
    accelerate : {"none", "squarem"}, default="none"
        EM acceleration (see :class:`EMEstimator`). ``"squarem"`` extrapolates
        consecutive EM steps by SQUAREM, which usually needs far fewer
        iterations on slowly converging fits. It runs the generic EM loop, so
        a unidimensional 2PL fit skips the native full-EM fast path. Other
        estimation methods accept only ``"none"``.
    constraints : sequence, optional
        Equality constraints across items for EM, like ``CONSTRAIN`` in R's
        ``mirt.model``. Each entry ties one stored parameter of several items,
        written as ``{"parameter": "discrimination", "items": [0, 1, 2]}`` or
        ``("discrimination", [0, 1, 2])``; items are zero-based positions or
        item names. For array parameters, ``"column"`` (a third tuple
        element) ties one coordinate per item, for example
        ``("thresholds", [0, 1], 2)``; without it whole rows are tied.
        Equal slopes for every item give a 2PL the Rasch structure with an
        estimated common slope. Each group counts as one parameter, tied
        coordinates share their estimate and standard error, and the
        constrained fit skips the native 2PL fast path (see
        :class:`EMEstimator`). Mixed-format models and other estimation
        methods raise an error.

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
        If data is not 2D or contains invalid response codes.
    MirtValidationError
        If ``n_factors`` is not a positive integer, the estimation method is
        unknown, a polytomous category count is invalid, per-item families
        do not name every item or are used without EM, ``accelerate`` is
        unknown or requested for a method other than EM, or ``constraints``
        are malformed, tie fixed or unknown coordinates, or are used without
        EM.
    MirtModelError
        If the model type is unknown or does not support ``n_factors``, or
        ``constraints`` are given for per-item model families.
    NotImplementedError
        If a ``CONSTRAIN`` group in ``spec`` equates different parameters.

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
    from mirt.backends.rust.estimation import _em_fit_2pl_prepared
    from mirt.estimation._em_context import EMFitContext
    from mirt.estimation._item_priors import validate_item_priors
    from mirt.estimation._refit import RefitRecipe
    from mirt.estimation._shared_step import validate_equality_constraints
    from mirt.estimation.base import _apply_starting_values, _free_masks_from_fixed
    from mirt.estimation.em import _ACCELERATIONS, EMEstimator
    from mirt.estimation.mcmc import GibbsSampler, MHRMEstimator
    from mirt.estimation.standard_errors import validate_se_method
    from mirt.exceptions import MirtValidationError
    from mirt.models._factory import (
        build_item_model,
        build_mixed_item_model,
        validate_item_types,
        validate_n_factors,
    )

    if not isinstance(compute_standard_errors, (bool, np.bool_)):
        raise MirtValidationError(
            "compute_standard_errors must be a boolean",
            parameter="compute_standard_errors",
            value=compute_standard_errors,
            expected="bool",
        )
    compute_standard_errors = bool(compute_standard_errors)
    se_method = validate_se_method(se_method)
    if se_method != "auto" and estimation != "EM":
        raise MirtValidationError(
            "se_method applies only to EM estimation",
            parameter="se_method",
            value=se_method,
            expected="'auto' for MHRM, MCMC and Gibbs",
        )
    if not isinstance(accelerate, str) or accelerate not in _ACCELERATIONS:
        raise MirtValidationError(
            "accelerate must be 'none' or 'squarem'",
            parameter="accelerate",
            value=accelerate,
            expected="'none' or 'squarem'",
        )
    if accelerate != "none" and estimation != "EM":
        raise MirtValidationError(
            "accelerate applies only to EM estimation",
            parameter="accelerate",
            value=accelerate,
            expected="'none' for MHRM, MCMC and Gibbs",
        )
    equality = validate_equality_constraints(constraints)
    if equality and estimation != "EM":
        raise MirtValidationError(
            "constraints apply only to EM estimation",
            parameter="constraints",
            value=estimation,
            expected="EM",
        )
    from mirt.results.fit_result import FitResult
    from mirt.typing import EstimationMethod
    from mirt.utils.data import response_column_names, validate_responses

    # Reject unknown families and factor counts before reading the data.
    validate_item_types(model)
    n_factors = validate_n_factors(n_factors)

    if item_names is None:
        item_names = response_column_names(data)
    data = validate_responses(data)
    if spec is not None:
        from mirt.model_syntax import _fit_spec

        return _fit_spec(
            data,
            spec,
            model=model,
            n_factors=n_factors,
            n_categories=n_categories,
            estimation=estimation,
            n_quadpts=n_quadpts,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            item_names=item_names,
            use_rust=use_rust,
            compute_standard_errors=compute_standard_errors,
            start_values=start_values,
            fixed=fixed,
            priors=priors,
            se_method=se_method,
            accelerate=accelerate,
            constraints=equality,
        )

    n_persons, n_items = data.shape

    if item_names is None:
        item_names = [f"Item_{i + 1}" for i in range(n_items)]

    item_types = validate_item_types(model, n_items)
    if not isinstance(item_types, str) and len(set(item_types)) == 1:
        item_types = item_types[0]
    if isinstance(item_types, str):
        irt_model = build_item_model(
            item_types,
            n_items,
            n_factors=n_factors,
            n_categories=n_categories,
            item_names=item_names,
            responses=data,
        )
    elif estimation != "EM":
        raise MirtValidationError(
            "per-item model families require estimation='EM'",
            parameter="estimation",
            value=estimation,
            expected="EM",
        )
    else:
        irt_model = build_mixed_item_model(
            item_types,
            n_factors=n_factors,
            n_categories=n_categories,
            item_names=item_names,
            responses=data,
        )

    estimation_method: EstimationMethod = estimation

    item_priors = validate_item_priors(priors)
    if item_priors is not None and estimation_method in ("MHRM", "MCMC", "Gibbs"):
        raise MirtValidationError(
            "priors are supported only with estimation='EM'",
            parameter="priors",
            value=estimation,
            expected="EM",
        )
    if fixed is not None:
        irt_model.set_free_parameter_masks(_free_masks_from_fixed(irt_model, fixed))
    if start_values is not None and estimation_method != "EM":
        # The native samplers do not accept starting values.
        _apply_starting_values(irt_model, start_values)
        use_rust = False
    # The native 2PL path starts from its own values, ignores masks, priors and
    # constraints, and runs plain EM.
    customized = (
        start_values is not None
        or bool(irt_model._free_parameter_restrictions)
        or item_priors is not None
        or accelerate != "none"
        or bool(equality)
    )

    if (
        should_use_rust(use_rust)
        and item_types == "2PL"
        and n_factors == 1
        and estimation_method == "EM"
        and not customized
    ):
        with EMFitContext(data, compress=True, native=True) as context:
            discrimination, difficulty, log_likelihood, n_iterations, converged = (
                _em_fit_2pl_prepared(
                    context.responses, n_quadpts, max_iter, tol, context.frequencies
                )
            )

            discrimination = np.asarray(discrimination)
            difficulty = np.asarray(difficulty)
            irt_model._parameters = {
                "discrimination": discrimination,
                "difficulty": difficulty,
            }
            irt_model._is_fitted = True

            standard_errors: dict[str, NDArray[np.float64]] = {}
            used_se_method = covariance = None
            if compute_standard_errors:
                standard_errors, used_se_method, covariance = _native_2pl_errors(
                    irt_model, context, n_quadpts, se_method
                )

            n_params = 2 * n_items
            aic = -2 * log_likelihood + 2 * n_params
            bic = -2 * log_likelihood + np.log(n_persons) * n_params

            return FitResult(
                model=irt_model,
                log_likelihood=log_likelihood,
                n_iterations=n_iterations,
                converged=converged,
                standard_errors=standard_errors,
                aic=aic,
                bic=bic,
                n_observations=n_persons,
                n_parameters=n_params,
                se_method=used_se_method,
                vcov=covariance,
                # Refits use the equivalent EM settings.
                refit_recipe=RefitRecipe(
                    EMEstimator,
                    {
                        "n_quadpts": n_quadpts,
                        "max_iter": max_iter,
                        "tol": tol,
                        "use_rust": use_rust,
                        "se_method": se_method,
                    },
                ),
            )

    if estimation_method == "EM":
        estimator_class = EMEstimator
        if not isinstance(item_types, str):
            from mirt.estimation.mixed_format_em import MixedFormatEMEstimator

            estimator_class = MixedFormatEMEstimator
        estimator = estimator_class(
            n_quadpts=n_quadpts,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            use_rust=use_rust,
            compute_standard_errors=compute_standard_errors,
            item_priors=item_priors,
            se_method=se_method,
            accelerate=accelerate,
            constraints=equality,
        )
        if start_values is None:
            return estimator.fit(irt_model, data)
        return estimator.fit(irt_model, data, start=start_values)

    if estimation_method == "MHRM":
        return MHRMEstimator(
            n_cycles=max_iter,
            burnin=min(500, max(max_iter // 4, 1)),
            verbose=verbose,
            use_rust=use_rust,
            compute_standard_errors=compute_standard_errors,
            n_quadpts=n_quadpts,
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
        result = _mcmc_result_to_fit_result(mcmc, n_persons)
        if not compute_standard_errors:
            result.standard_errors = {}
        return result

    raise MirtValidationError(
        f"Unknown estimation method: {estimation}",
        parameter="estimation",
        value=estimation,
        expected="EM, MHRM, MCMC, or Gibbs",
    )


def _native_2pl_errors(
    model: BaseItemModel,
    context: EMFitContext,
    n_quadpts: int,
    se_method: str,
) -> tuple[dict[str, NDArray[np.float64]], str, NDArray[np.float64] | None]:
    """Standard errors for the native 2PL fit, matching ``EMEstimator``.

    Returns the errors, the method used, and the free-parameter covariance
    (``None`` for complete-data curvature).
    """
    from mirt.constants import PROB_EPSILON
    from mirt.estimation._item_information import item_standard_errors
    from mirt.estimation.base import _parameter_bounds
    from mirt.estimation.em import _weight_posterior
    from mirt.estimation.quadrature import GaussHermiteQuadrature
    from mirt.estimation.standard_errors import (
        _posterior_from_model,
        estimate_covariance,
    )

    quadrature = GaussHermiteQuadrature(n_points=n_quadpts, n_dimensions=1)
    if se_method == "complete_data":
        posterior = _posterior_from_model(model, context.responses, quadrature)
        errors = item_standard_errors(
            model,
            context.responses,
            _weight_posterior(posterior, context.frequencies),
            quadrature.nodes,
            PROB_EPSILON,
            context=context,
        )
        assert errors is not None
        return errors, "complete_data", None
    method = "oakes" if se_method == "auto" else se_method
    estimate = estimate_covariance(
        model,
        context.responses,
        quadrature,
        quadrature.weights,
        method,
        frequencies=context.frequencies,
        bounds=lambda name: _parameter_bounds(model, name),
    )
    return estimate.standard_errors, method, estimate.covariance


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
    responses: ArrayLike | None = None,
    statistics: list[str] | None = None,
    n_groups: int | None = None,
    p_adjust: Literal["bonferroni", "holm", "fdr_bh", "none"] = "none",
    *,
    min_expected: float = 1.0,
    n_quadpts: int = 41,
    quadrature_points: NDArray[np.float64] | None = None,
    quadrature_weights: NDArray[np.float64] | None = None,
    item_parameter_counts: NDArray[np.int_] | None = None,
    na_rm: bool = False,
    n_plausible: int = 100,
    seed: int | None = None,
    prior_mean: NDArray[np.float64] | None = None,
    prior_cov: NDArray[np.float64] | None = None,
    theta: NDArray[np.float64] | None = None,
    constraints: Sequence[Any] | None = None,
) -> Any:
    """Compute item fit statistics for a fitted IRT model.

    Item fit statistics assess how well individual items conform to the
    assumed IRT model. Poor-fitting items may indicate violations of
    model assumptions or problematic item content.

    Parameters
    ----------
    result : FitResult
        A fitted IRT model result from fit_mirt().
    responses : array-like of shape (n_persons, n_items), optional
        Response data used for fit calculation. Required for all statistics.
        Negative codes, ``NaN`` and the nulls of nullable DataFrame columns
        denote missing responses, as in :func:`fit_mirt`.
    statistics : list of str, optional
        Fit statistics to compute. Options include:

        - "infit": Information-weighted mean square (sensitive to
          unexpected responses near ability level)
        - "outfit": Unweighted mean square (sensitive to outliers)
        - "z_infit", "z_outfit": Wilson-Hilferty standardized mean squares,
          approximately standard normal only with ``theta`` independent of
          these responses; with the default EAP abilities they are biased
          toward overfit and descriptive (a warning is issued)
        - "S_X2": Orlando-Thissen S-X2 statistic
        - "X2", "G2": Bock/Yen chi-square and likelihood-ratio statistics
          over ability groups (approximate p-values, liberal on short tests)
        - "PV_Q1": Chalmers-Ng plausible-value Q1 statistic

        Default is ["infit", "outfit"]. Unknown names raise
        ``MirtValidationError``.
    n_groups : int, optional
        Number of ability groups for X2, G2 and PV_Q1 (default 10).
        Deprecated and ignored by S-X2, which conditions on exact total scores.
    p_adjust : {"bonferroni", "holm", "fdr_bh", "none"}, default="none"
        Multiple-testing adjustment across item-level chi-square p-values. When
        an adjustment is requested, the result includes a
        ``p_value_adjusted`` column (``<name>_p_adjusted`` for X2, G2 and
        PV_Q1) while retaining the raw p-values.
    min_expected : float, default=1.0
        Minimum expected S-X2 cell count for adjacent score/category pooling.
        Zero disables sparse-cell pooling.
    n_quadpts : int, default=41
        Gauss-Hermite quadrature points per model factor for S-X2.
    quadrature_points, quadrature_weights : ndarray, optional
        Explicit latent grid and nonnegative probability masses for S-X2.
        Supply both to test a different fitted latent distribution.
    item_parameter_counts : ndarray, optional
        Estimated parameter counts per item for S-X2 degrees of freedom.
        Defaults to the model's free parameter masks; supply zeros when item
        parameters are externally known, or counts for shared parameters.
    na_rm : bool, default=False
        Exclude incomplete persons from S-X2. Otherwise S-X2 requires complete
        responses. Mean-square statistics always use available responses.
    n_plausible : int, default=100
        Plausible-value draws for PV_Q1.
    seed : int, optional
        Seed for the PV_Q1 plausible-value draws.
    prior_mean : ndarray of shape (n_factors,), optional
        Mean of the normal latent population. Defaults to
        ``result.latent_mean`` when the fit has one, and to zero otherwise.
    prior_cov : ndarray of shape (n_factors, n_factors), optional
        Covariance of the normal latent population, which S-X2 integrates
        over and which is the prior of the EAP abilities behind the other
        statistics. Defaults to ``result.latent_covariance`` when the fit
        estimated one, and to the identity otherwise.
    theta : ndarray of shape (n_persons,) or (n_persons, n_factors), optional
        Abilities for the mean-square and X2/G2 statistics, such as estimates
        from an independent calibration. EAP scores of ``responses`` by
        default. See :func:`mirt.diagnostics.compute_itemfit`.
    constraints : sequence, optional
        The ``constraints`` of a ``fit_mirt`` fit. A group of ``k`` tied
        coordinates counts as one parameter, ``1/k`` per item, in the
        chi-square degrees of freedom.

    Returns
    -------
    DataFrame
        Item fit statistics with items as rows and statistics as columns.
        S-X2 includes ``df`` and ``p_value``; X2, G2 and PV_Q1 include
        ``<name>_df`` and ``<name>_p``. An unestimable chi-square test
        has ``p_value=NaN``; nonpositive degrees of freedom are reported as zero.

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
    if prior_mean is None:
        prior_mean = getattr(result, "latent_mean", None)
    if prior_cov is None:
        prior_cov = getattr(result, "latent_covariance", None)

    fit_stats = compute_itemfit(
        result.model,
        responses,
        statistics,
        theta=theta,
        n_groups=n_groups,
        p_adjust=p_adjust,
        min_expected=min_expected,
        n_quadpts=n_quadpts,
        quadrature_points=quadrature_points,
        quadrature_weights=quadrature_weights,
        item_parameter_counts=item_parameter_counts,
        na_rm=na_rm,
        n_plausible=n_plausible,
        seed=seed,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        constraints=constraints,
    )

    return create_dataframe(fit_stats, index=result.model.item_names, index_name="item")


def personfit(
    result: FitResult,
    responses: ArrayLike,
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
    responses : array-like of shape (n_persons, n_items)
        Response matrix. Negative codes, ``NaN`` and the nulls of nullable
        DataFrame columns denote missing responses, as in :func:`fit_mirt`.
    theta : ndarray of shape (n_persons,) or (n_persons, n_factors), optional
        Ability estimates. If None, computed using EAP scoring.
    statistics : list of str, optional
        Person fit statistics to compute. Options include:

        - "infit": Information-weighted mean square
        - "outfit": Unweighted mean square
        - "z_infit", "z_outfit": Wilson-Hilferty standardized mean squares
        - "Zh": Standardized log-likelihood (Drasgow et al.)
        - "lz": Log-likelihood z-score

        Default is ["infit", "outfit", "Zh"]. Unknown names raise
        ``MirtValidationError``.
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
    from mirt.diagnostics.itemfit import _validate_statistics
    from mirt.diagnostics.personfit import _PERSONFIT_STATISTICS, compute_personfit
    from mirt.scoring import fscores
    from mirt.utils.dataframe import create_dataframe

    statistics = _validate_statistics(
        statistics, _PERSONFIT_STATISTICS, default=("infit", "outfit", "Zh")
    )

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
    *,
    anchors: Sequence[int | str] | None = None,
    scheme: Literal["drop", "add", "drop_sequential", "add_sequential"] = "drop",
    n_jobs: int = 1,
) -> Any:
    """Compute Differential Item Functioning (DIF) statistics.

    DIF analysis tests whether items function differently across groups
    after controlling for ability level. Groups are compared on a common
    latent scale, so group impact is not reported as DIF.

    The default likelihood-ratio test refits a multiple-group model once per
    tested item: one to two seconds for 30 binary 2PL items and 1,000
    persons per group, and several times longer per item for 3PL and
    polytomous items. ``n_jobs=-1`` parallelizes the refits; ``method="wald"``
    and :func:`mirt.diagnostics.compute_grdif` are fast screens. See
    :func:`mirt.diagnostics.compute_dif` for the methods.

    Args:
        data: Response matrix (n_persons x n_items).
        groups: Group membership array (n_persons,). Must have exactly 2 groups.
        model: IRT model type.
        method: DIF detection method:

            - 'likelihood_ratio': Nested multiple-group LR test (one baseline
              fit plus one refit per tested item)
            - 'wald': Wald test on linked parameter differences
            - 'lord': Lord's chi-square test, an alias of 'wald'
            - 'raju': Raju's area measures between linked curves (no p-value)
        n_categories: Number of categories for polytomous models.
        n_quadpts: Number of quadrature points for EM.
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance.
        focal_group: Which group to use as focal (default: second unique group).
        p_adjust: Multiple-testing adjustment across items. Default 'none',
            as in :func:`mirt.multigroup.multigroup_dif` and R's
            ``mirt::DIF``.
        anchors: Items assumed free of DIF, by index or name; they are not
            tested. They anchor the likelihood-ratio models or define the
            linking for the other methods.
        scheme: Likelihood-ratio scheme: 'drop', 'add', 'drop_sequential' or
            'add_sequential'. The 'add' schemes require anchors.
        n_jobs: Worker processes for likelihood-ratio refits.

    Returns:
        DataFrame with DIF statistics for each item:
            - statistic: Test statistic
            - df: Degrees of freedom
            - p_value: P-value
            - p_value_adjusted: Multiplicity-adjusted P-value
            - effect_size: Focal-minus-reference location difference
            - classification: ETS classification using adjusted P-values
            - adjustment: Multiple-testing method
            - tested: Whether the item was tested
            - converged: Whether the underlying fits converged
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
        anchors=anchors,
        scheme=scheme,
        n_jobs=n_jobs,
    )
    metadata = {"method", "anchors", "linking_constants"}
    columns = {
        name: values for name, values in dif_results.items() if name not in metadata
    }
    return create_dataframe(columns, index_name="item")


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
