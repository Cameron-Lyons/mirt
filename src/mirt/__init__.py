from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from mirt._api_registry import MODULE_EXPORTS, build_all_exports, build_lazy_imports
from mirt._version import __version__

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike, NDArray

    from mirt.estimation._shared_step import EqualityConstraints
    from mirt.estimation.mcmc import MCMCResult
    from mirt.estimation.priors import Prior, PriorSpecification
    from mirt.model_syntax import ModelSpec
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult

    _ItemFamily = Literal[
        "1PL", "Rasch", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM"
    ]


def fit_mirt(
    data: NDArray[np.int_] | Any,
    model: _ItemFamily | Sequence[_ItemFamily] = "2PL",
    n_factors: int = 1,
    n_categories: int | Sequence[int] | None = None,
    estimation: Literal["EM", "MHRM", "MCMC", "Gibbs"] = "EM",
    n_quadpts: int | None = None,
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
    quadrature: Literal["auto", "rectangular", "gauss-hermite"] = "auto",
    seed: int | None = None,
    use_gpu: bool | Literal["auto"] = "auto",
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

        - "1PL": One-parameter logistic with one discrimination common to
          all items, estimated by EM; equivalent to R mirt's
          ``itemtype="Rasch"``, whose estimated latent standard deviation is
          the common slope (see below); "Rasch" is accepted as an alias
        - "2PL": Two-parameter logistic
        - "3PL": Three-parameter logistic (with guessing)
        - "4PL": Four-parameter logistic (with guessing and slipping)
        - "GRM": Graded Response Model (polytomous)
        - "GPCM": Generalized Partial Credit Model (polytomous)
        - "PCM": Partial Credit Model (polytomous) with a common
          discrimination, estimated by EM like that of "1PL"
        - "NRM": Nominal Response Model (polytomous)

        A sequence names one family per item for a mixed-format test, for
        example ``["3PL"] * 20 + ["GRM"] * 5``. It fits a
        :class:`~mirt.models.mixed_format.MixedItemModel` by
        :class:`~mirt.estimation.mixed_format_em.MixedFormatEMEstimator`,
        whose parameters are qualified by family, such as ``"3PL.guessing"``,
        in ``start_values``, ``fixed`` and the results. Mixed formats require
        ``estimation="EM"``; a sequence naming one family fits that family.

        "1PL" and "PCM" fits under EM estimate the common discrimination
        ``a`` as one parameter, so a 1PL has ``n_items + 1`` parameters and
        the likelihood of R's Rasch model, which instead fixes the slopes at
        1 and estimates the latent variance ``a**2``; R's difficulties are
        ``a`` times these. ``fixed={"discrimination": True}`` holds ``a`` at
        its starting value (1 by default), the Rasch model for a
        standard-normal trait. Other estimation methods raise
        ``MirtValidationError`` for "1PL" and "PCM"; ``spec`` fits and the
        1PL and PCM items of a mixed-format test hold ``a`` at 1, and with
        ``spec``, ``COV = F*F`` estimates the latent variance instead.

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
        are observed-information errors for unidimensional 2PL-4PL, GRM and
        GPCM fits, and otherwise the spread of the Robbins-Monro
        iterates (``se_method="mhrm_iterate_sd"``, see
        :class:`~mirt.estimation.mcmc.MHRMEstimator`).
    n_quadpts : int, optional
        Quadrature points per latent dimension, at least 5, for EM and the
        observed-information standard errors of MHRM. The default depends on
        the number of factors, as in R mirt: 61 for one factor, 31 for two,
        15 for three, 9 for four, 7 for five and 5 for six. Larger models need
        an explicit value or ``estimation="MHRM"``, and grids whose EM
        arrays would exceed about a gigabyte warn.
    max_iter : int, default=500
        Maximum number of EM iterations (EM) or MHRM cycles / MCMC iterations
        depending on method. Must be a positive integer.
    tol : float, default=1e-4
        Convergence tolerance of EM: iteration stops when the marginal
        log-likelihood (log-posterior with ``priors``) changes by less than
        ``tol`` in absolute value between consecutive iterations. Must be
        finite and positive.
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
        Supported by EM, MHRM and Gibbs sampling; masked models run the NumPy
        samplers.
    priors : PriorSpecification or mapping of str to Prior, optional
        Item-parameter priors for Bayes modal estimation with EM, for example
        ``{"guessing": BetaPrior(5, 17)}``. See ``EMEstimator(item_priors=...)``.
        With ``estimation="MCMC"`` or ``"Gibbs"`` they replace the sampler's
        default priors (see :class:`GibbsSampler`); MHRM rejects them.
        The result reports ``log_posterior``, and standard errors add the
        prior's negative second derivative to the information, so they reflect
        the curvature of the log-posterior. ``start_values`` runs MHRM with
        the NumPy sampler instead of the native burn-in.
    se_method : {"auto", "oakes", "crossprod", "sandwich", "complete_data"}, \
default="auto"
        Standard-error estimator for EM fits (see :class:`EMEstimator`).
        ``"oakes"`` uses the observed information of the marginal likelihood,
        ``"crossprod"`` the outer product of person scores, ``"sandwich"``
        their robust combination, and ``"complete_data"`` itemwise
        complete-data curvature, which understates uncertainty, often two-
        to fivefold for "NRM" and multidimensional fits. ``"auto"`` selects
        ``"oakes"`` for every family above, with any number of factors and
        with ``spec``, and for mixed-format tests of them. Exploratory fits
        with ``n_factors >= 2`` are rotationally unidentified, so the upper
        triangle of the first ``n_factors - 1`` items' slopes is held at its
        estimates and gets ``NaN`` standard errors. The matrix methods also
        store the parameter covariance in ``FitResult.vcov``, and
        ``FitResult.se_method`` records the estimator used. Other estimation
        methods accept only ``"auto"``.
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
        iterations on slowly converging fits. With ``"none"``, EM takes the
        guarded parameter-expanded (PX-EM) steps of :class:`EMEstimator`
        where they apply; ``EMEstimator(px_em=False)`` fits by plain EM.
        Other estimation methods accept only ``"none"``. Mixed-format models
        and ``spec`` fits with ``COV`` terms, whose latent covariance is
        estimated, run plain EM instead and warn.
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
        coordinates share their estimate and standard error (see
        :class:`EMEstimator`). Mixed-format models and other estimation
        methods raise an error.
    quadrature : {"auto", "rectangular", "gauss-hermite"}, default="auto"
        Quadrature rule of EM (see :class:`EMEstimator`). ``"rectangular"``
        is R mirt's grid of equally spaced nodes on ``[-6, 6]``, which
        integrates the peaked likelihoods of long or discriminating tests
        accurately; ``"auto"`` uses it for 15 or more points per dimension,
        including the defaults for up to three factors, and Gauss-Hermite
        nodes for coarser grids.
    seed : int, optional
        Seed of the stochastic estimators (``"MHRM"``, ``"MCMC"`` and
        ``"Gibbs"``), so their fits can be reproduced. EM is deterministic and
        ignores it.
    use_gpu : bool or "auto", default="auto"
        Whether EM evaluates unidimensional likelihoods on a GPU when one is
        available (see :class:`EMEstimator`). ``"auto"`` follows
        :func:`set_backend`: the ``"auto"`` and ``"gpu"`` preferences use an
        available GPU, ``"rust"`` and ``"numpy"`` do not. ``True`` and
        ``False`` override the preference. The other estimation methods do
        not use a GPU and ignore it.

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
        unknown or requested for a method other than EM, ``constraints``
        are malformed, tie fixed or unknown coordinates, or are used without
        EM, ``n_quadpts`` is not an integer of at least 5 (or the default
        grid of more than six factors is too large), ``quadrature`` is
        unknown, ``max_iter`` is not a positive integer, ``tol`` is not
        finite and positive, ``use_gpu`` is neither a boolean nor "auto", or
        ``model`` is "1PL" or "PCM" with an estimation method other than EM.
    MirtModelError
        If the model type is unknown or does not support ``n_factors``, or
        ``constraints`` are given for per-item model families.
    NotImplementedError
        If a ``CONSTRAIN`` group in ``spec`` equates different parameters.

    Warns
    -----
    RuntimeWarning
        If ``spec`` describes a "2PL" bifactor structure, which
        :func:`bfactor` fits much faster on two-dimensional grids.
    UserWarning
        If ``accelerate="squarem"`` falls back to plain EM (see
        ``accelerate``).

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

    from mirt._backend_config import validate_use_gpu
    from mirt.estimation._item_priors import validate_item_priors
    from mirt.estimation._shared_step import validate_equality_constraints
    from mirt.estimation.base import (
        _apply_starting_values,
        _free_masks_from_fixed,
        _validate_convergence_controls,
    )
    from mirt.estimation.em import _ACCELERATIONS, EMEstimator
    from mirt.estimation.quadrature import (
        resolve_quadpts,
        validate_quadpts,
        validate_quadrature_rule,
    )
    from mirt.estimation.standard_errors import validate_se_method
    from mirt.exceptions import MirtValidationError
    from mirt.models._factory import (
        build_item_model,
        build_mixed_item_model,
        validate_item_types,
        validate_n_factors,
    )

    # Every estimation path validates the integration and convergence
    # controls alike.
    max_iter, tol = _validate_convergence_controls(max_iter, tol)
    if n_quadpts is not None:
        n_quadpts = validate_quadpts(n_quadpts)
    quadrature = validate_quadrature_rule(quadrature)
    use_gpu = validate_use_gpu(use_gpu)
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

    # Reject unknown families and factor counts before reading the data, and
    # resolve the "Rasch" alias to "1PL".
    model = validate_item_types(model)
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
            quadrature=quadrature,
            use_gpu=use_gpu,
        )

    n_persons, n_items = data.shape

    if item_names is None:
        item_names = [f"Item_{i + 1}" for i in range(n_items)]

    item_types = validate_item_types(model, n_items)
    if not isinstance(item_types, str) and len(set(item_types)) == 1:
        item_types = item_types[0]
    if item_types in ("1PL", "PCM") and estimation != "EM":
        # The samplers cannot tie slopes, and a unit-slope Rasch model under
        # the same name would be a different model from the EM fit.
        raise MirtValidationError(
            f"model={item_types!r} estimates one discrimination common to all "
            "items, which only estimation='EM' supports; fit it with EM, or "
            "fit the unit-slope Rasch model with the estimator directly, for "
            "example MHRMEstimator().fit(OneParameterLogistic(n_items), data)",
            parameter="estimation",
            value=estimation,
            expected="EM",
        )
    if isinstance(item_types, str):
        irt_model = build_item_model(
            item_types,
            n_items,
            n_factors=n_factors,
            n_categories=n_categories,
            item_names=item_names,
            responses=data,
            # EM estimates the common slope of the Rasch families.
            estimate_discrimination=item_types in ("1PL", "PCM"),
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
    if item_priors is not None and estimation_method == "MHRM":
        raise MirtValidationError(
            "priors are supported only with estimation='EM', 'MCMC' or 'Gibbs'",
            parameter="priors",
            value=estimation,
            expected="EM, MCMC, or Gibbs",
        )
    if fixed is not None:
        masks = _free_masks_from_fixed(irt_model, fixed)
        if getattr(irt_model, "estimate_discrimination", False) and not np.any(
            masks["discrimination"]
        ):
            # A common slope held at 1 is the family's own unit-slope model,
            # which keeps the native M-steps; another value stays a mask.
            start = {} if start_values is None else start_values
            if np.all(np.asarray(start.get("discrimination", 1.0)) == 1.0):
                irt_model = build_item_model(
                    item_types,
                    n_items,
                    n_categories=(
                        irt_model.n_categories if irt_model.is_polytomous else None
                    ),
                    item_names=item_names,
                )
                masks = {
                    name: mask
                    for name, mask in masks.items()
                    if name != "discrimination"
                }
        irt_model.set_free_parameter_masks(masks)
    if start_values is not None and estimation_method != "EM":
        # The native MHRM burn-in starts from its own values.
        _apply_starting_values(irt_model, start_values)
        use_rust = False
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
            use_gpu=use_gpu,
            use_rust=use_rust,
            compute_standard_errors=compute_standard_errors,
            item_priors=item_priors,
            se_method=se_method,
            accelerate=accelerate,
            constraints=equality,
            quadrature=quadrature,
        )
        fitted = (
            estimator.fit(irt_model, data)
            if start_values is None
            else estimator.fit(irt_model, data, start=start_values)
        )
        _warn_slopes_on_lower_bound(getattr(fitted, "model", None))
        return fitted

    # The samplers' module imports scipy.stats, which EM fits do not need.
    from mirt.estimation.mcmc import GibbsSampler, MHRMEstimator

    if estimation_method == "MHRM":
        return MHRMEstimator(
            n_cycles=max_iter,
            burnin=min(500, max(max_iter // 4, 1)),
            verbose=verbose,
            use_rust=use_rust,
            seed=seed,
            compute_standard_errors=compute_standard_errors,
            # Observed-information errors, for unidimensional families only.
            n_quadpts=resolve_quadpts(n_quadpts, min(n_factors, 6)),
        ).fit(irt_model, data)

    if estimation_method in ("MCMC", "Gibbs"):
        burnin = min(1000, max(max_iter // 5, 1))
        n_iter = max(max_iter, burnin + 10)
        mcmc = GibbsSampler(
            n_iter=n_iter,
            burnin=burnin,
            verbose=verbose,
            use_rust=use_rust,
            priors=priors,
            seed=seed,
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


def _warn_slopes_on_lower_bound(model: BaseItemModel | None) -> None:
    """Warn when unidimensional slope estimates end on their positive lower bound.

    Such an item, typically reverse-keyed or miskeyed, discriminates
    negatively; R mirt estimates a negative slope, which this package's
    unidimensional models do not allow.
    """
    import warnings

    import numpy as np

    from mirt.estimation.base import _parameter_bounds
    from mirt.models.base import BaseItemModel

    if (
        not isinstance(model, BaseItemModel)
        or model.n_factors != 1
        or "discrimination" not in model.parameters
    ):
        return
    free = np.asarray(model.free_parameter_masks["discrimination"], dtype=np.bool_)
    low, _ = _parameter_bounds(model, "discrimination")
    if low <= 0.0 or not np.any(free):
        return
    slopes = np.asarray(model.parameters["discrimination"], dtype=np.float64)
    pinned = np.flatnonzero(free.reshape(slopes.shape) & (slopes <= low + 1e-6))
    if pinned.size == 0:
        return
    names = ", ".join(str(model.item_names[item]) for item in pinned[:10])
    warnings.warn(
        f"discrimination of {names} ended on the lower bound {low:g}; such an "
        "item discriminates negatively (reverse-keyed or miskeyed). Recode it "
        "(for example 1 - x, or max - x for polytomous items) and refit; R mirt "
        "would report a negative slope",
        UserWarning,
        stacklevel=3,
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
    converged = mcmc.converged

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
    n_quadpts: int | None = None,
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
        - "Zh": Item-level standardized log-likelihood (Drasgow, Levine &
          Williams, 1985), as in R's ``itemfit(fit_stats = "Zh")``
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
    n_quadpts : int, optional
        Gauss-Hermite quadrature points per model factor for S-X2. ``None``
        (default) uses 41 for one or two factors and fewer for more factors
        (21 for three, 9 for four, 7 for five), keeping the grid at no more
        than 100,000 points.
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
        Abilities for the mean-square, Zh and X2/G2 statistics, such as
        estimates from an independent calibration. EAP scores of
        ``responses`` by default. See :func:`mirt.diagnostics.compute_itemfit`.
    constraints : sequence, optional
        Equality constraints of the fit. A group of ``k`` tied coordinates
        counts as one parameter, ``1/k`` per item, in the chi-square degrees
        of freedom. Defaults to the constraints ``result`` was fitted with
        (``fit_mirt(constraints=...)`` or ``CONSTRAIN``); pass ``[]`` to count
        every free coordinate separately.

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
    from mirt.diagnostics._utils import resolve_fit_constraints
    from mirt.diagnostics.itemfit import compute_itemfit
    from mirt.utils.dataframe import create_dataframe

    if statistics is None:
        statistics = ["infit", "outfit"]
    if prior_mean is None:
        prior_mean = getattr(result, "latent_mean", None)
    if prior_cov is None:
        prior_cov = getattr(result, "latent_covariance", None)
    constraints = resolve_fit_constraints(result, constraints)

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
    n_quadpts: int | None = None,
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
        n_quadpts: Quadrature points per dimension for EM; ``None`` uses the
            default grid of :func:`fit_mirt` (61 rectangular points).
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance.
        focal_group: Which group to use as focal (default: second unique group).
        p_adjust: Multiple-testing adjustment across items. Default 'none',
            as in :func:`mirt.multigroup.multigroup_dif` and R's
            ``mirt::DIF``.
        anchors: Items assumed free of DIF, by 0-based index or by name
            (DataFrame column names, otherwise ``Item_1``, ``Item_2``, ...
            as in :func:`fit_mirt` and :func:`itemfit`); they are not
            tested. They anchor the likelihood-ratio models or define the
            linking for the other methods.
        scheme: Likelihood-ratio scheme: 'drop', 'add', 'drop_sequential' or
            'add_sequential'. The 'add' schemes require anchors.
        n_jobs: Worker processes for likelihood-ratio refits.

    Returns:
        DataFrame with one row per item, labelled by item name like
        :func:`itemfit` (the ``item`` index for pandas, the first column
        for polars), with DIF statistics:

        - statistic: Test statistic
        - df: Degrees of freedom
        - p_value: P-value
        - p_value_adjusted: Multiplicity-adjusted P-value
        - effect_size: Focal-minus-reference location difference
        - ets_effect: MH D-DIF (binary items) or SMD/SD (polytomous
          items) behind the ETS class
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
    metadata = {"item_names", "method", "anchors", "linking_constants", "ets_scale"}
    columns = {
        name: values for name, values in dif_results.items() if name not in metadata
    }
    return create_dataframe(
        columns, index=dif_results.get("item_names"), index_name="item"
    )


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
