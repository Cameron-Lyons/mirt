from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Iterable, Sequence
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import xlog1py, xlogy

from mirt._backend_config import should_use_rust
from mirt._gpu_backend import (
    compute_log_likelihoods_2pl_gpu,
    compute_log_likelihoods_3pl_gpu,
    compute_log_likelihoods_gpcm_gpu,
    compute_log_likelihoods_grm_gpu,
    is_gpu_available,
)
from mirt._model_defaults import uses_builtin_model_hooks
from mirt.backends.rust._helpers import RUST_AVAILABLE
from mirt.backends.rust.estimation import em_iteration_3pl
from mirt.constants import PROB_EPSILON
from mirt.estimation._em_context import EMFitContext
from mirt.estimation._graded_order import (
    GradedOrder,
    graded_order,
    graded_threshold_constraint,
)
from mirt.estimation._item_priors import (
    ItemPriorPenalty,
    ItemPriors,
    resolve_item_priors,
    validate_item_priors,
)
from mirt.estimation._posterior import normalize_log_posterior
from mirt.estimation._shared_step import (
    EqualityConstraints,
    TiedCoordinates,
    validate_equality_constraints,
)
from mirt.estimation.base import (
    BaseEstimator,
    StartValues,
    _apply_starting_values,
    _free_shared_parameters,
    _parameter_bounds,
    _validate_start,
)
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.se_methods import _compute_item_se_curvature
from mirt.estimation.standard_errors import StandardErrorMethod, validate_se_method
from mirt.exceptions import MirtValidationError

if TYPE_CHECKING:
    from mirt.estimation.latent_density import LatentDensity
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult

# Relative function-change tolerance for precise M-steps: the native graded
# optimizer, analytic dichotomous items, shared and tied-item steps, and every
# item optimizer under SQUAREM. Item objectives are large sums, so looser
# values can stop with parameter errors near 1e-3, and the resulting noisy
# M-steps slow EM and defeat extrapolation.
_PRECISE_ITEM_FTOL = 1e-10
# Product quadrature grids beyond this many nodes trigger a warning.
_LARGE_GRID_NODES = 1_000_000
_ACCELERATIONS = ("none", "squarem")
# Latent densities that ``EMEstimator(latent_density=...)`` builds by name,
# and those of them defined for several factors.
_DENSITY_NAMES = (
    "gaussian",
    "normal",
    "empirical",
    "histogram",
    "eh",
    "ehw",
    "empiricalhist_woods",
    "davidian",
    "mixture",
)
_MULTIVARIATE_DENSITY_NAMES = ("gaussian", "normal", "empirical", "histogram", "eh")
# SQUAREM grows the maximum step length by this factor after a successful
# step at the maximum and shrinks it after a rejected one.
_SQUAREM_STEP_FACTOR = 4.0
# Itemwise optimization methods that the batched Newton M-step stands in for.
_ITEM_OPTIMIZATION_HOOKS = (
    "_optimize_item",
    "_optimize_item_return",
    "_optimize_item_params",
    "_get_item_params_and_bounds",
    "_set_item_params",
)


def _weight_posterior(
    posterior: NDArray[np.float64], frequencies: NDArray[np.float64] | None
) -> NDArray[np.float64]:
    """Scale compressed-pattern posteriors by their pattern frequencies."""
    return posterior if frequencies is None else posterior * frequencies[:, None]


def _warn_large_grid(model: BaseItemModel, n_quadpts: int) -> None:
    """Warn before building a product quadrature grid of over a million nodes."""
    # Python integers keep NumPy integer point counts from overflowing.
    n_nodes = int(n_quadpts) ** int(model.n_factors)
    if n_nodes <= _LARGE_GRID_NODES:
        return
    from mirt.models.bifactor import BifactorModel

    advice = (
        "mirt.bfactor or BifactorEMEstimator integrates bifactor models exactly "
        "on two-dimensional grids"
        if isinstance(model, BifactorModel)
        else "reduce n_quadpts or the number of factors"
    )
    warnings.warn(
        f"EM quadrature over {model.n_factors} factors with {n_quadpts} points "
        f"each needs {n_nodes:,} nodes, which is slow and memory intensive; "
        f"{advice}",
        RuntimeWarning,
        stacklevel=3,
    )


def _add_prior_curvature(
    errors: NDArray[np.float64], curvature: NDArray[np.float64] | None
) -> NDArray[np.float64]:
    """Add a prior's negative curvature to diagonal curvature standard errors.

    Coordinates without likelihood curvature (``NaN`` errors) keep only the
    prior's.
    """
    if curvature is None or not np.any(curvature):
        return errors
    with np.errstate(divide="ignore", invalid="ignore"):
        information = np.where(np.isfinite(errors) & (errors > 0), errors**-2.0, 0.0)
        total = information + curvature
        combined = np.where(total > 0, 1.0 / np.sqrt(total), np.nan)
    return np.where(curvature != 0, combined, errors)


def _ordered_step(
    order: GradedOrder,
    objective: Callable[[NDArray[np.float64]], Any],
    bounds: list[tuple[float, float]],
    start: NDArray[np.float64],
    proposal: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return an ordered graded M-step that does not raise the item objective.

    SLSQP meets its constraints only up to its tolerance, so thresholds tied
    by an empty category can end up slightly closer than the ordering gap.
    Such a proposal is projected onto the ordered box, like the native graded
    M-step's iterates, and kept unless its objective exceeds that of the
    start, which is projected as well.
    """
    lower, upper = np.asarray(bounds, dtype=np.float64).reshape(-1, 2).T
    projected = None
    if np.all(np.isfinite(proposal)):
        projected = order.project(proposal, lower, upper)
        if np.array_equal(projected, proposal):
            return projected
    start = order.project(start, lower, upper)
    if projected is None:
        return start

    def value(point: NDArray[np.float64]) -> float:
        result = objective(point)
        return float(result[0] if isinstance(result, tuple) else result)

    return projected if value(projected) <= value(start) else start


def _family_item_layout(
    model: BaseItemModel, item_idx: int
) -> tuple[
    BaseItemModel, NDArray[np.float64], NDArray[np.bool_], list[tuple[float, float]]
]:
    """Return one item's coordinates in its family's free layout.

    Returns a shallow view of ``model`` without user free-parameter masks,
    the item's values in that layout, which of them the masks leave free, and
    a box that also covers the fixed values, which may lie outside the
    optimizer bounds.
    """
    view = copy.copy(model)
    view._free_parameter_restrictions = {}
    family_masks = view.free_parameter_masks
    masks = model.free_parameter_masks
    values: list[float] = []
    free: list[bool] = []
    bounds: list[tuple[float, float]] = []
    for name, array in model.parameters.items():
        if not model._item_indexed(name):
            continue
        family = np.asarray(family_masks[name][item_idx], dtype=np.bool_).reshape(-1)
        row = np.asarray(model._canonical_parameter_values(name, array)[item_idx])
        item_free = np.asarray(masks[name][item_idx], dtype=np.bool_).reshape(-1)
        low, high = _parameter_bounds(model, name)
        for value, is_free in zip(
            row.reshape(-1)[family], item_free[family], strict=True
        ):
            values.append(float(value))
            free.append(bool(is_free))
            bounds.append(
                (low, high) if is_free else (min(low, value), max(high, value))
            )
    return (
        view,
        np.asarray(values, dtype=np.float64),
        np.asarray(free, dtype=np.bool_),
        bounds,
    )


def _fix_coordinates(
    objective: Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]],
    values: NDArray[np.float64],
    free: NDArray[np.bool_],
) -> Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]:
    """Restrict an objective and its gradient to the ``free`` coordinates."""

    def restricted(params: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        point = values.copy()
        point[free] = params
        value, gradient = objective(point)
        return value, gradient[free]

    return restricted


class EMEstimator(BaseEstimator):
    """Marginal maximum likelihood estimation by the EM algorithm.

    Each M-step optimizes the item parameters item by item. Parameters shared
    by all items, such as the common thresholds of the rating scale models,
    are then optimized jointly with the item parameters held fixed, which
    keeps every iteration a monotone conditional maximization step.

    Parameters
    ----------
    n_quadpts : int, default=21
        Gauss-Hermite quadrature points per latent dimension.
    max_iter : int, default=500
        Maximum number of M-steps.
    tol : float, default=1e-4
        Convergence tolerance for the change in marginal log-likelihood
        between consecutive EM iterates.
    verbose : bool, default=False
        Print the log-likelihood at each recorded iterate.
    latent_density : LatentDensity or str, optional
        Latent density specification. Defaults to a fixed Gaussian density
        with the prior mean and covariance passed to :meth:`fit`, which
        ``"gaussian"`` (or ``"normal"``) also selects. ``"empirical"`` (or
        ``"histogram"``/``"eh"``) estimates a histogram over the quadrature
        nodes, for any number of factors. The univariate ``"davidian"``
        (Davidian curve of degree 4), ``"mixture"`` (two Gaussian components)
        and ``"ehw"`` (Woods' extrapolated histogram) require one factor.
        Each fit builds a new density from a name, while a
        :class:`~mirt.estimation.latent_density.LatentDensity` instance is
        used, and updated, as given.
    prob_epsilon : float, default=1e-10
        Probability clipping bound used by item objectives.
    item_optim_maxiter : int, default=50
        Iteration limit for itemwise numerical optimizers.
    item_optim_ftol : float, default=1e-6
        Relative function-change tolerance for itemwise numerical optimizers.
        Built-in dichotomous items with analytic gradients (1PL-4PL and the
        logistic items of multidimensional and bifactor models) and the
        native graded response M-step use at most ``1e-10``, because looser
        relative criteria make their EM iterations jitter and stall short of
        the optimum. Built-in 1PL and 2PL items without parameter
        restrictions are solved jointly by Newton's method to a tight
        tolerance. They use the numerical optimizer only as a fallback, for
        example when an estimate reaches a parameter bound.
    se_step_size : float, default=1e-5
        Finite-difference step for itemwise complete-data standard errors of
        custom item models, and for marginal-likelihood differences of models
        without exact item derivatives.
    n_jobs : int, default=1
        Worker threads for itemwise optimization. ``-1`` uses every CPU.
    use_gpu : bool or "auto", default="auto"
        Whether to use a GPU likelihood backend when one is available.
    use_rust : bool, default=True
        Whether to use native kernels when they preserve the semantics.
    compute_standard_errors : bool, default=True
        Whether to compute item parameter standard errors after fitting.
    se_method : {"auto", "oakes", "crossprod", "sandwich", "complete_data"}, \
default="auto"
        Standard-error estimator. ``"oakes"`` inverts the observed
        information of the marginal likelihood, computed exactly by Louis's
        (1982) identity, and also stores the full parameter covariance in
        ``FitResult.vcov``. ``"crossprod"`` inverts the outer product of the
        marginal person scores and ``"sandwich"`` combines both into a
        misspecification-robust covariance. ``"complete_data"`` is itemwise
        complete-data curvature, which omits the information lost to the
        unobserved latent trait and understates uncertainty. ``"auto"`` uses
        ``"oakes"`` for unidimensional built-in 1PL-4PL, GRM, GPCM, PCM, RSM
        and GRSM models and ``"complete_data"`` otherwise;
        ``FitResult.se_method`` records the estimator used. The latent
        density is treated as fixed, and coordinates at an optimizer bound
        (for example a guessing parameter of 0) are held fixed with ``NaN``
        standard errors. The
        matrix methods cost O(N Q P^2) for N response patterns, Q nodes and
        P parameters; models outside the built-in item families fall back to
        O(P^2) marginal likelihood evaluations.
    accelerate : {"none", "squarem"}, default="none"
        EM acceleration. ``"squarem"`` applies SQUAREM extrapolation (Varadhan
        and Roland, 2008) with step-length adaptation, projection onto the
        item parameter bounds and a monotone fallback to the plain EM step.
        Extrapolation needs precise M-steps, so item optimizers then use a
        relative tolerance of at most ``1e-10``.
        SQUAREM uses the generic E- and M-steps and so bypasses the fused
        native 3PL iteration. A latent density other than a fixed Gaussian
        falls back to plain EM with a warning. With acceleration,
        ``max_iter`` still bounds the number of M-steps, ``n_iterations``
        counts E-steps, which includes the evaluation of each extrapolated
        point, and ``convergence_history`` records the log-likelihood at
        accepted iterates only.
    item_priors : PriorSpecification or mapping of str to Prior, optional
        Priors on item parameters for Bayes modal (MAP) estimation. A mapping
        assigns a :class:`~mirt.estimation.priors.Prior` to stored per-item
        parameters by name, for example ``{"guessing": BetaPrior(5, 17)}``. A
        ``PriorSpecification`` contributes its discrimination, difficulty,
        guessing and upper priors to parameters the model has. Each M-step then
        maximizes the expected log-likelihood plus the log-prior of the free
        coordinates, and convergence is judged on the log-posterior, which is
        recorded in ``convergence_history`` and reported as
        ``FitResult.log_posterior``. ``log_likelihood``, AIC and BIC remain
        likelihood-based. Standard errors include the prior: its negative
        second derivative is added to the observed information (to the score
        cross-product for ``"crossprod"`` and to each coordinate's curvature
        for ``"complete_data"``), so they describe the curvature of the
        log-posterior at the mode. Parameters shared by all items take no
        priors. Priors bypass the fused native 3PL iteration, the batched
        Newton logistic M-step and the native polytomous M-step.
    constraints : sequence, optional
        Equality constraints across items, as in the ``CONSTRAIN`` statement
        of R's ``mirt.model``. Each group ties one stored parameter of several
        items: a mapping ``{"parameter": "discrimination", "items": [0, 1,
        2]}``, a tuple ``("discrimination", [0, 1, 2])``, or an
        :class:`~mirt.estimation.constraints.EqualityConstraint`. Items are
        zero-based positions or item names, and omitted items mean every item.
        For array parameters, ``"column"`` (the third tuple element) ties one
        coordinate of each item's row, such as one threshold or the slope on
        one factor; without it whole rows are tied column by column. Tied
        coordinates must be free, start from their mean and stay exactly
        equal. The items they link are optimized jointly in each M-step with
        every group as one coordinate, so EM stays monotone. Each group counts
        as one parameter in ``n_parameters``, AIC and BIC. Standard errors
        come from the information of the constrained parameters,
        ``J' I J`` for the 0/1 matrix ``J`` mapping groups to coordinates;
        ``FitResult.vcov`` repeats a group's row for each of its
        coordinates, so tied coordinates have equal errors and are perfectly
        correlated. Constraints bypass the fused native 3PL iteration; the
        batched Newton logistic M-step and the native polytomous M-step
        update only the other items. Constraints are not available for
        mixed-format models.

    References
    ----------
    Louis, T. A. (1982). Finding the observed information matrix when using
    the EM algorithm. *Journal of the Royal Statistical Society B*, 44(2),
    226-233.

    Varadhan, R., & Roland, C. (2008). Simple and globally convergent methods
    for accelerating the convergence of any EM algorithm. *Scandinavian
    Journal of Statistics*, 35(2), 335-353.
    """

    _estimates_shared_parameters = True

    def __init__(
        self,
        n_quadpts: int = 21,
        max_iter: int = 500,
        tol: float = 1e-4,
        verbose: bool = False,
        latent_density: LatentDensity
        | Literal["gaussian", "empirical", "davidian", "mixture", "ehw"]
        | None = None,
        prob_epsilon: float = 1e-10,
        item_optim_maxiter: int = 50,
        item_optim_ftol: float = 1e-6,
        se_step_size: float = 1e-5,
        n_jobs: int = 1,
        use_gpu: bool | Literal["auto"] = "auto",
        use_rust: bool = True,
        compute_standard_errors: bool = True,
        accelerate: Literal["none", "squarem"] = "none",
        item_priors: ItemPriors | None = None,
        se_method: StandardErrorMethod = "auto",
        constraints: EqualityConstraints | None = None,
    ) -> None:
        super().__init__(max_iter, tol, verbose)

        if n_quadpts < 5:
            raise MirtValidationError(
                "n_quadpts should be at least 5",
                parameter="n_quadpts",
                value=n_quadpts,
                expected=">= 5",
            )

        self.n_quadpts = n_quadpts
        self.prob_epsilon = prob_epsilon
        self.item_optim_maxiter = item_optim_maxiter
        self.item_optim_ftol = item_optim_ftol
        self.se_step_size = se_step_size
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.use_rust = use_rust
        if not isinstance(compute_standard_errors, (bool, np.bool_)):
            raise MirtValidationError(
                "compute_standard_errors must be a boolean",
                parameter="compute_standard_errors",
                value=compute_standard_errors,
                expected="bool",
            )
        self.compute_standard_errors = bool(compute_standard_errors)
        if not isinstance(accelerate, str) or accelerate not in _ACCELERATIONS:
            raise MirtValidationError(
                "accelerate must be 'none' or 'squarem'",
                parameter="accelerate",
                value=accelerate,
                expected="'none' or 'squarem'",
            )
        self.accelerate = accelerate
        self.item_priors = validate_item_priors(item_priors)
        self._prior_penalty: ItemPriorPenalty | None = None
        self.se_method = validate_se_method(se_method)
        self.constraints = validate_equality_constraints(constraints)
        # Coordinates tied by ``constraints`` in the model being fitted.
        self._tied: TiedCoordinates | None = None
        # Method and free-parameter covariance behind the latest standard errors.
        self._se_details: tuple[str, NDArray[np.float64] | None] | None = None
        self._quadrature: GaussHermiteQuadrature | None = None
        if isinstance(latent_density, str):
            name = latent_density.strip().lower()
            if name not in _DENSITY_NAMES:
                raise MirtValidationError(
                    f"Unknown latent density: {latent_density!r}",
                    parameter="latent_density",
                    value=latent_density,
                    expected=", ".join(repr(known) for known in _DENSITY_NAMES),
                )
            latent_density = name
        self._latent_density_spec = latent_density
        self._latent_density: LatentDensity | None = None
        self._pattern_frequencies: NDArray[np.float64] | None = None
        self._fit_context: EMFitContext | None = None
        self._precise_m_steps = False

    @property
    def _should_use_gpu(self) -> bool:
        """Determine if GPU should be used based on settings and availability."""
        if self.use_gpu is False:
            return False
        return is_gpu_available()

    def fit(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        prior_mean: NDArray[np.float64] | None = None,
        prior_cov: NDArray[np.float64] | None = None,
        *,
        start: StartValues = "default",
    ) -> FitResult:
        """Fit a model by marginal maximum likelihood or Bayes modal EM.

        Parameters
        ----------
        model : BaseItemModel
            Model to fit in place. Coordinates fixed with
            ``set_free_parameter_masks`` keep their values.
        responses : ndarray of shape (n_persons, n_items)
            Response matrix with negative values marking missing responses.
        prior_mean, prior_cov : ndarray, optional
            Mean and covariance of the default Gaussian latent density.
        start : {"default", "model"} or mapping, default="default"
            Starting values. ``"default"`` resets the free coordinates of an
            unfitted model to the family defaults and starts a fitted model
            from its current estimates. ``"model"`` starts from the model's
            current values. A mapping of parameter names to arrays overrides
            the ``"default"`` values; it also sets the values at which fixed
            coordinates are held. Coordinates tied by ``constraints`` start
            from the mean of their starting values.

        Returns
        -------
        FitResult
            Fitted model, log-likelihood, standard errors and fit statistics.
            The final mean and covariance of a Gaussian latent density, for
            example the correlations estimated by a
            :class:`~mirt.estimation.latent_density.FactorCovarianceDensity`,
            are its ``latent_mean`` and ``latent_covariance`` (``None`` for a
            zero mean and an identity covariance), and ``refit_recipe``
            records the estimator settings and the final density.

        Raises
        ------
        MirtValidationError
            If ``constraints`` do not match the model's free coordinates.
        """
        from mirt.estimation._patterns import supports_pattern_compression
        from mirt.estimation._shared_step import resolve_equality_constraints
        from mirt.estimation.latent_density import GaussianDensity, create_density

        start = _validate_start(start)
        responses = self._validate_responses(responses, model.n_items)
        self._tied = resolve_equality_constraints(self.constraints, model)
        _warn_large_grid(model, self.n_quadpts)
        self._quadrature = GaussHermiteQuadrature(
            n_points=self.n_quadpts,
            n_dimensions=model.n_factors,
        )

        if prior_mean is None:
            prior_mean = np.zeros(model.n_factors)
        if prior_cov is None:
            prior_cov = np.eye(model.n_factors)

        spec = self._latent_density_spec
        if spec is None or spec in ("gaussian", "normal"):
            self._latent_density = GaussianDensity(
                mean=prior_mean,
                cov=prior_cov,
                n_dimensions=model.n_factors,
            )
        elif isinstance(spec, str):
            if model.n_factors != 1 and spec not in _MULTIVARIATE_DENSITY_NAMES:
                raise MirtValidationError(
                    f"latent_density={spec!r} is univariate, but the model has "
                    f"{model.n_factors} factors; use 'gaussian' or 'empirical'",
                    parameter="latent_density",
                    value=spec,
                    expected="'gaussian' or 'empirical'",
                )
            self._latent_density = create_density(spec)
        else:
            self._latent_density = spec

        priors = resolve_item_priors(self.item_priors, model)
        _apply_starting_values(model, start)
        if self._tied is not None:
            self._tied.equalize(model)

        builtin = supports_pattern_compression(model)
        with EMFitContext(
            responses,
            compress=type(self) is EMEstimator and builtin,
            native=builtin and should_use_rust(self.use_rust),
        ) as context:
            self._fit_context = context
            self._pattern_frequencies = context.frequencies
            self._prior_penalty = ItemPriorPenalty(priors) if priors else None
            try:
                return self._fit_prepared(model, context)
            finally:
                self._fit_context = None
                self._prior_penalty = None

    def _fit_prepared(self, model: BaseItemModel, context: EMFitContext) -> FitResult:
        from mirt.estimation._refit import gaussian_population, recipe_for
        from mirt.results.fit_result import FitResult

        responses = context.responses
        n_persons = context.n_observations
        frequencies = context.frequencies

        self._convergence_history = []
        squarem = self._uses_squarem()
        # Extrapolation needs precise M-steps, so SQUAREM tightens item
        # optimizer tolerances while it runs.
        self._precise_m_steps = squarem
        try:
            run = self._run_squarem if squarem else self._run_em
            posterior_weights, objective, converged, n_iterations = run(
                model, responses, frequencies
            )
        finally:
            self._precise_m_steps = False

        model._is_fitted = True
        current_ll = objective - self._log_item_prior(model)

        weighted_posterior = _weight_posterior(posterior_weights, frequencies)
        self._se_details = None
        standard_errors = (
            self._compute_standard_errors(model, responses, weighted_posterior)
            if self.compute_standard_errors
            else {}
        )
        se_method, covariance = self._se_details or (None, None)

        n_params = model.n_parameters + self._latent_density.n_parameters
        if self._tied is not None:
            n_params -= self._tied.n_redundant
        aic = self._compute_aic(current_ll, n_params)
        bic = self._compute_bic(current_ll, n_params, n_persons)
        latent_mean, latent_covariance = gaussian_population(self._latent_density)

        return FitResult(
            model=model,
            log_likelihood=current_ll,
            n_iterations=n_iterations,
            converged=converged,
            standard_errors=standard_errors,
            aic=aic,
            bic=bic,
            n_observations=n_persons,
            n_parameters=n_params,
            log_posterior=None if self._prior_penalty is None else objective,
            se_method=se_method,
            vcov=covariance,
            latent_covariance=latent_covariance,
            latent_mean=latent_mean,
            refit_recipe=recipe_for(self),
        )

    def _evaluate(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        frequencies: NDArray[np.float64] | None,
    ) -> tuple[NDArray[np.float64], float]:
        """Run an E-step and return the posterior with the EM objective.

        The objective is the marginal log-likelihood, plus the item log-prior
        when ``item_priors`` are set.
        """
        posterior_weights, log_marginal = self._e_step(model, responses)
        log_likelihood = float(
            np.sum(log_marginal * (1.0 if frequencies is None else frequencies))
        )
        return posterior_weights, log_likelihood + self._log_item_prior(model)

    def _log_item_prior(self, model: BaseItemModel) -> float:
        """Return the log-prior of the free item coordinates, or zero."""
        penalty = self._prior_penalty
        return 0.0 if penalty is None else penalty.log_prior(model)

    def _run_em(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        frequencies: NDArray[np.float64] | None,
    ) -> tuple[NDArray[np.float64], float, bool, int]:
        """Iterate plain EM and return the final posterior, LL and status."""
        use_rust_3pl = self._can_use_rust_3pl(model, responses)
        prev_ll = -np.inf
        converged = False

        for iteration in range(self.max_iter):
            rust_result = None
            if use_rust_3pl:
                rust_result = self._run_rust_3pl_iteration(model, responses)
                if rust_result is None:
                    use_rust_3pl = False

            if rust_result is None:
                posterior_weights, current_ll = self._evaluate(
                    model, responses, frequencies
                )
            else:
                (
                    new_discrimination,
                    new_difficulty,
                    new_guessing,
                    posterior_weights,
                    current_ll,
                ) = rust_result

            self._convergence_history.append(current_ll)

            self._log_iteration(iteration, current_ll)

            if self._check_convergence(prev_ll, current_ll):
                converged = True
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                break

            prev_ll = current_ll

            weighted_posterior = _weight_posterior(posterior_weights, frequencies)
            if rust_result is None:
                self._m_step(model, responses, weighted_posterior)
            else:
                model.set_parameters(
                    discrimination=new_discrimination,
                    difficulty=new_difficulty,
                    guessing=new_guessing,
                )

            n_k = weighted_posterior.sum(axis=0)
            self._latent_density.update(self._quadrature.nodes, n_k)
        else:
            posterior_weights, current_ll = self._evaluate(
                model, responses, frequencies
            )
            self._convergence_history.append(current_ll)
            converged = self._check_convergence(prev_ll, current_ll)

        return posterior_weights, current_ll, converged, iteration + 1

    def _uses_squarem(self) -> bool:
        """Return whether SQUAREM applies, warning when a request falls back."""
        from mirt.estimation.latent_density import GaussianDensity

        if self.accelerate != "squarem":
            return False
        density = self._latent_density
        # The plain loop updates the density after each M-step; SQUAREM
        # extrapolates item parameters only, so the density must stay fixed.
        if (
            type(density) is GaussianDensity
            and not density.estimate_mean
            and not density.estimate_cov
        ):
            return True
        warnings.warn(
            "accelerate='squarem' requires a fixed Gaussian latent density; "
            "using plain EM",
            UserWarning,
            stacklevel=4,
        )
        return False

    def _run_squarem(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        frequencies: NDArray[np.float64] | None,
    ) -> tuple[NDArray[np.float64], float, bool, int]:
        """Iterate SQUAREM cycles of the E- and M-steps.

        Each cycle takes two EM steps from the current iterate, extrapolates
        along the SqS3 step length and stabilizes the extrapolated point with
        one more EM step. The result is accepted only if its log-likelihood is
        at least that of the first EM step, otherwise the cycle keeps the
        second EM step. Convergence uses the plain EM rule on the first step
        of each cycle, so results stay comparable with ``accelerate="none"``.
        """
        from mirt.estimation._acceleration import (
            FreeItemParameters,
            squarem_point,
            squarem_step_length,
        )

        parameters = FreeItemParameters(model, self._tied)
        history = self._convergence_history
        n_e_steps = n_m_steps = 0

        def evaluate() -> tuple[NDArray[np.float64], float]:
            nonlocal n_e_steps
            n_e_steps += 1
            return self._evaluate(model, responses, frequencies)

        def update(posterior: NDArray[np.float64]) -> NDArray[np.float64]:
            nonlocal n_m_steps
            n_m_steps += 1
            self._m_step(model, responses, _weight_posterior(posterior, frequencies))
            return parameters.get(model)

        def record(log_likelihood: float) -> None:
            self._log_iteration(len(history), log_likelihood)
            history.append(log_likelihood)

        step_max = 1.0
        posterior, current_ll = evaluate()
        record(current_ll)
        converged = False
        while n_m_steps < self.max_iter:
            start = parameters.get(model)
            first = update(posterior)
            posterior, first_ll = evaluate()
            record(first_ll)
            converged = self._check_convergence(current_ll, first_ll)
            current_ll = first_ll
            if converged or n_m_steps >= self.max_iter:
                break

            second = update(posterior)
            # A unit step length reproduces the second EM step itself.
            alpha = squarem_step_length(start, first, second, step_max)
            accepted = alpha == 1.0
            if not accepted:
                point = squarem_point(
                    start, first, second, alpha, parameters.lower, parameters.upper
                )
                if parameters.set(model, point, check_order=True):
                    trial, trial_ll = evaluate()
                    if np.isfinite(trial_ll) and n_m_steps < self.max_iter:
                        update(trial)
                        trial, trial_ll = evaluate()
                    accepted = bool(np.isfinite(trial_ll) and trial_ll >= first_ll)
                    if accepted:
                        posterior, current_ll = trial, trial_ll
                    else:
                        parameters.set(model, second)
            if alpha == step_max:
                step_max = (
                    step_max * _SQUAREM_STEP_FACTOR
                    if accepted
                    else max(1.0, step_max / _SQUAREM_STEP_FACTOR)
                )
            if alpha == 1.0 or not accepted:
                posterior, current_ll = evaluate()
            record(current_ll)

        if self.verbose and converged:
            print(f"Converged after {n_e_steps} E-steps")
        return posterior, current_ll, converged, n_e_steps

    def _can_use_rust_3pl(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> bool:
        """Return whether the batched native 3PL iteration preserves semantics."""
        from mirt.estimation.latent_density import GaussianDensity
        from mirt.models.dichotomous import ThreeParameterLogistic

        if (
            not RUST_AVAILABLE
            or not should_use_rust(self.use_rust)
            or self._should_use_gpu
            or self._prior_penalty is not None
            or self._tied is not None
            or type(model) is not ThreeParameterLogistic
            or not uses_builtin_model_hooks(model, likelihood=True)
            or model._free_parameter_restrictions
            or model.n_factors != 1
            or self.n_jobs != 1
            or self.prob_epsilon != PROB_EPSILON
        ):
            return False

        if (
            isinstance(self.item_optim_maxiter, (bool, np.bool_))
            or not isinstance(self.item_optim_maxiter, Integral)
            or self.item_optim_maxiter < 1
            or isinstance(self.item_optim_ftol, (bool, np.bool_))
            or not isinstance(self.item_optim_ftol, Real)
            or not np.isfinite(self.item_optim_ftol)
            or self.item_optim_ftol <= 0.0
        ):
            return False

        density = self._latent_density
        if (
            not isinstance(density, GaussianDensity)
            or density.n_dimensions != 1
            or density.estimate_mean
            or density.estimate_cov
            or not np.array_equal(density.mean, np.zeros(1))
            or not np.array_equal(density.cov, np.eye(1))
        ):
            return False

        params = model.parameters
        expected_shape = (model.n_items,)
        if any(
            name not in params or params[name].shape != expected_shape
            for name in ("discrimination", "difficulty", "guessing")
        ):
            return False

        observed = responses[responses >= 0]
        return bool(np.all((observed == 0) | (observed == 1)))

    def _run_rust_3pl_iteration(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> (
        tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            float,
        ]
        | None
    ):
        """Run and validate one batched native 3PL E/M iteration."""
        if self._quadrature is None:
            return None

        params = model.parameters
        disc_bounds = _parameter_bounds(model, "discrimination")
        diff_bounds = _parameter_bounds(model, "difficulty")
        guess_bounds = _parameter_bounds(model, "guessing")
        try:
            result = em_iteration_3pl(
                responses,
                self._quadrature.nodes.ravel(),
                self._quadrature.weights,
                params["discrimination"],
                params["difficulty"],
                params["guessing"],
                prior_mean=0.0,
                prior_var=1.0,
                max_m_iter=int(self.item_optim_maxiter),
                m_tol=float(self.item_optim_ftol),
                disc_bounds=disc_bounds,
                diff_bounds=diff_bounds,
                guess_bounds=guess_bounds,
                regularization=0.01,
                regularization_c=0.1,
                frequencies=self._pattern_frequencies,
            )
        except Exception:
            return None

        try:
            if result is None or len(result) != 5:
                return None

            discrimination, difficulty, guessing, posterior, log_likelihood = result
            discrimination = np.asarray(discrimination, dtype=np.float64)
            difficulty = np.asarray(difficulty, dtype=np.float64)
            guessing = np.asarray(guessing, dtype=np.float64)
            posterior = np.asarray(posterior, dtype=np.float64)
            log_likelihood = float(log_likelihood)
        except (TypeError, ValueError, OverflowError):
            return None

        parameter_shape = (model.n_items,)
        posterior_shape = (responses.shape[0], self.n_quadpts)
        if (
            discrimination.shape != parameter_shape
            or difficulty.shape != parameter_shape
            or guessing.shape != parameter_shape
            or posterior.shape != posterior_shape
            or not np.all(np.isfinite(discrimination))
            or not np.all(np.isfinite(difficulty))
            or not np.all(np.isfinite(guessing))
            or not np.all(np.isfinite(posterior))
            or not np.isfinite(log_likelihood)
            or np.any(
                (discrimination < disc_bounds[0]) | (discrimination > disc_bounds[1])
            )
            or np.any((difficulty < diff_bounds[0]) | (difficulty > diff_bounds[1]))
            or np.any((guessing < guess_bounds[0]) | (guessing > guess_bounds[1]))
            or np.any(posterior < 0.0)
        ):
            return None

        row_sums = posterior.sum(axis=1)
        if not np.allclose(row_sums, 1.0, rtol=1e-10, atol=1e-12):
            return None
        posterior = posterior / row_sums[:, None]

        return (
            discrimination.copy(),
            difficulty.copy(),
            guessing.copy(),
            posterior,
            log_likelihood,
        )

    def _e_step(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return normalized posterior weights and per-person log marginals."""
        quad_points = self._quadrature.nodes
        quad_weights = self._quadrature.weights

        log_likelihoods = self._compute_log_likelihoods(model, responses, quad_points)

        log_prior_mass = self._latent_density.log_quadrature_mass(
            quad_points, quad_weights
        )

        # Built-in likelihoods allocate their output. Custom likelihoods may
        # return cached arrays or views, which must not be overwritten.
        from mirt.estimation._patterns import supports_pattern_compression

        if not (
            type(self) is EMEstimator
            and "_compute_log_likelihoods" not in vars(self)
            and supports_pattern_compression(model)
            and log_likelihoods.flags.writeable
        ):
            log_likelihoods = log_likelihoods.copy()
        return normalize_log_posterior(log_likelihoods, log_prior_mass)

    def _compute_log_likelihoods(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        quad_points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihoods, using GPU if available and appropriate."""
        if (
            self._should_use_gpu
            and model.n_factors == 1
            and uses_builtin_model_hooks(model, likelihood=True)
        ):
            return self._compute_log_likelihoods_gpu(model, responses, quad_points)

        if hasattr(model, "log_likelihood_batch"):
            return model.log_likelihood_batch(responses, quad_points)

        return self._compute_log_likelihoods_python(model, responses, quad_points)

    def _compute_log_likelihoods_python(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        quad_points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Pure-Python log-likelihood fallback used by CPU/GPU paths."""
        n_persons = responses.shape[0]
        n_quad = quad_points.shape[0]
        log_likelihoods = np.zeros((n_persons, n_quad))
        for q in range(n_quad):
            theta_q = quad_points[q : q + 1]
            log_likelihoods[:, q] = model.log_likelihood(responses, theta_q)
        return log_likelihoods

    def _compute_log_likelihoods_gpu(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        quad_points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute log-likelihoods using GPU acceleration."""
        from mirt.models.dichotomous import (
            OneParameterLogistic,
            ThreeParameterLogistic,
            TwoParameterLogistic,
        )
        from mirt.models.polytomous import (
            GeneralizedPartialCredit,
            GradedResponseModel,
            PartialCreditModel,
        )

        params = model.parameters

        if type(model) in (OneParameterLogistic, TwoParameterLogistic):
            discrimination = params["discrimination"]
            difficulty = params["difficulty"]
            return compute_log_likelihoods_2pl_gpu(
                responses,
                quad_points.ravel(),
                discrimination,
                difficulty,
            )

        if type(model) is ThreeParameterLogistic:
            discrimination = params["discrimination"]
            difficulty = params["difficulty"]
            guessing = params["guessing"]
            return compute_log_likelihoods_3pl_gpu(
                responses,
                quad_points.ravel(),
                discrimination,
                difficulty,
                guessing,
            )

        if type(model) is GradedResponseModel:
            discrimination = params["discrimination"]
            thresholds = params["thresholds"]
            return compute_log_likelihoods_grm_gpu(
                responses,
                quad_points.ravel(),
                discrimination,
                thresholds,
                n_categories=model.n_categories,
            )

        if type(model) in (GeneralizedPartialCredit, PartialCreditModel):
            discrimination = params["discrimination"]
            steps = params["steps"]
            return compute_log_likelihoods_gpcm_gpu(
                responses,
                quad_points.ravel(),
                discrimination,
                steps,
                n_categories=model.n_categories,
            )

        if hasattr(model, "log_likelihood_batch"):
            return model.log_likelihood_batch(responses, quad_points)

        return self._compute_log_likelihoods_python(model, responses, quad_points)

    def _m_step(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
    ) -> None:
        import os
        from concurrent.futures import ThreadPoolExecutor

        quad_points = self._quadrature.nodes
        fit_context = self._fit_context
        tied = self._tied
        context = (
            fit_context
            if fit_context is not None and fit_context.responses is responses
            else EMFitContext(responses)
        )

        if (
            model.is_polytomous
            and self._prior_penalty is None
            and should_use_rust(self.use_rust)
        ):
            from mirt.backends.rust.polytomous_mstep import try_polytomous_m_step
            from mirt.models.polytomous import GradedResponseModel

            # The native step moves every item, so items linked by equality
            # constraints get their values back and are optimized jointly.
            held = {
                item: self._get_item_params_and_bounds(model, item)[0]
                for item in (() if tied is None else tied.items)
            }
            # Loose relative tolerances stop the native graded optimizer early,
            # and the noisy M-steps triple the EM iterations. Partial-credit
            # items converge cleanly and would only slow down.
            if try_polytomous_m_step(
                model,
                responses,
                posterior_weights,
                quad_points,
                max_iter=self.item_optim_maxiter,
                ftol=self._item_optim_ftol(type(model) is GradedResponseModel),
                epsilon=self.prob_epsilon,
                n_jobs=self.n_jobs,
                context=fit_context,
            ):
                for item, values in held.items():
                    self._set_item_params(model, item, values)
                if held:
                    self._m_step_tied(
                        model,
                        responses,
                        posterior_weights,
                        quad_points,
                        {
                            item: {
                                "r_kc": context.expected_category_counts(
                                    item, model.n_categories[item], posterior_weights
                                )
                            }
                            for item in held
                        },
                    )
                return

        item_counts: dict[int, dict[str, NDArray[np.float64]]]
        if model.is_polytomous:
            category_counts = context.category_counts(
                model.n_categories, posterior_weights
            )
            item_counts = {
                item_idx: {"r_kc": counts}
                for item_idx, counts in enumerate(category_counts)
            }
        else:
            correct, observed = context.expected_counts(posterior_weights)
            items: Iterable[int] = range(model.n_items)
            if self._uses_newton_logistic_m_step(model):
                items = self._newton_logistic_m_step(
                    model, correct, observed, skip=() if tied is None else tied.items
                )
            item_counts = {
                item_idx: {"r_k": correct[item_idx], "n_k_valid": observed[item_idx]}
                for item_idx in items
            }
        # Items linked by equality constraints are optimized jointly below.
        tied_counts: dict[int, dict[str, NDArray[np.float64]]] = {}
        for item in () if tied is None else tied.items:
            item_counts.pop(item, None)
            tied_counts[item] = (
                {"r_kc": category_counts[item]}
                if model.is_polytomous
                else {"r_k": correct[item], "n_k_valid": observed[item]}
            )

        n_jobs = self.n_jobs
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        if n_jobs == 1:
            for item_idx, counts in item_counts.items():
                self._optimize_item(
                    model,
                    item_idx,
                    responses,
                    posterior_weights,
                    quad_points,
                    **counts,
                )
        elif item_counts:

            def optimize_single_item(item_idx):
                return item_idx, self._optimize_item_return(
                    model,
                    item_idx,
                    responses,
                    posterior_weights,
                    quad_points,
                    **item_counts[item_idx],
                )

            if context is fit_context:
                results = list(
                    context.executor(n_jobs).map(optimize_single_item, item_counts)
                )
            else:
                with ThreadPoolExecutor(
                    max_workers=min(n_jobs, len(item_counts))
                ) as executor:
                    results = list(executor.map(optimize_single_item, item_counts))

            for item_idx, optimal_params in results:
                self._set_item_params(model, item_idx, optimal_params)

        if tied_counts:
            self._m_step_tied(
                model, responses, posterior_weights, quad_points, tied_counts
            )

        if _free_shared_parameters(model):
            from mirt.estimation._shared_step import binary_category_counts

            self._m_step_shared(
                model,
                category_counts
                if model.is_polytomous
                else binary_category_counts(correct, observed),
            )

    def _m_step_shared(
        self, model: BaseItemModel, counts: list[NDArray[np.float64]]
    ) -> None:
        """Update parameters shared by all items with item parameters fixed.

        ``counts`` holds each item's expected ``(n_points, n_categories)``
        category counts; a dichotomous item has incorrect and correct columns.
        """
        from mirt.estimation._shared_step import optimize_shared_parameters

        optimize_shared_parameters(
            model,
            self._quadrature.nodes,
            counts,
            epsilon=self.prob_epsilon,
            max_iter=self.item_optim_maxiter,
            ftol=self._item_optim_ftol(precise=True),
        )

    def _m_step_tied(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        quad_points: NDArray[np.float64],
        item_counts: dict[int, dict[str, NDArray[np.float64]]],
    ) -> None:
        """Optimize the items linked by equality constraints jointly.

        ``item_counts`` holds each tied item's expected counts, as passed to
        :meth:`_optimize_item`. Each component of linked items is a separate
        problem.
        """
        from mirt.estimation._shared_step import (
            TiedItemObjective,
            optimize_tied_items,
            with_differenced_gradient,
        )

        assert self._tied is not None
        for component in self._tied.components:
            parts: list[TiedItemObjective] = []
            numerical: list[TiedItemObjective] = []
            try:
                for item in component:
                    start, bounds, objective, analytic = self._item_objective(
                        model,
                        item,
                        responses,
                        posterior_weights,
                        quad_points,
                        **item_counts[item],
                    )
                    if objective is not None and not analytic:
                        objective = with_differenced_gradient(objective, bounds)
                    part = TiedItemObjective(
                        item,
                        start,
                        bounds,
                        objective,
                        graded_threshold_constraint(model, item, start.size),
                    )
                    parts.append(part)
                    if not analytic:
                        numerical.append(part)
                optimized = optimize_tied_items(
                    model,
                    self._tied,
                    parts,
                    max_iter=self.item_optim_maxiter,
                    ftol=self._item_optim_ftol(precise=True),
                )
            finally:
                # Numerical objectives write trial values to the model.
                for part in numerical:
                    self._set_item_params(model, part.item, part.start)
            if optimized is not None:
                for part, values in zip(parts, optimized, strict=True):
                    self._set_item_params(model, part.item, values)

    def _item_optim_ftol(self, precise: bool = False) -> float:
        """Return the relative item-optimizer tolerance for this M-step."""
        if precise or self._precise_m_steps:
            return min(self.item_optim_ftol, _PRECISE_ITEM_FTOL)
        return self.item_optim_ftol

    def _uses_newton_logistic_m_step(self, model: BaseItemModel) -> bool:
        """Return whether every item is a free built-in 1PL/2PL logistic item.

        The batched Newton solver maximizes the unclipped item objective, which
        matches the clipped one only for the default probability clipping.
        Estimator subclasses that customize itemwise optimization keep it.
        """
        from mirt.models.dichotomous import OneParameterLogistic, TwoParameterLogistic

        if (
            type(model) not in (OneParameterLogistic, TwoParameterLogistic)
            or self._prior_penalty is not None
            or not uses_builtin_model_hooks(model)
            or model._free_parameter_restrictions
            or tuple(model._parameters) != ("discrimination", "difficulty")
            or not 0.0 < self.prob_epsilon <= PROB_EPSILON
            or any(
                name in vars(self)
                or getattr(type(self), name) is not getattr(EMEstimator, name)
                for name in _ITEM_OPTIMIZATION_HOOKS
            )
        ):
            return False
        masks = model.free_parameter_masks
        return bool(
            np.all(masks["difficulty"])
            and (type(model) is OneParameterLogistic or np.all(masks["discrimination"]))
        )

    def _newton_logistic_m_step(
        self,
        model: BaseItemModel,
        correct: NDArray[np.float64],
        observed: NDArray[np.float64],
        skip: Sequence[int] = (),
    ) -> NDArray[np.intp]:
        """Solve all 1PL/2PL items jointly and return items needing the fallback.

        Items with no observed responses, and the items in ``skip`` (which
        another step updates), keep their values. Estimates that do not
        converge or leave the optimizer box are left to the bounded itemwise
        optimizer, which then returns the constrained optimum.
        """
        from mirt.estimation._logistic_newton import newton_logistic_items
        from mirt.models.dichotomous import OneParameterLogistic

        params = model.parameters
        estimate_slopes = type(model) is not OneParameterLogistic
        slopes = params["discrimination"].reshape(model.n_items, -1)
        difficulty = params["difficulty"]
        new_slopes, intercepts, converged = newton_logistic_items(
            self._quadrature.nodes,
            correct,
            observed,
            slopes,
            -slopes.sum(axis=1) * difficulty,
            estimate_slopes=estimate_slopes,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            new_difficulty = -intercepts / new_slopes.sum(axis=1)
        low, high = _parameter_bounds(model, "difficulty")
        accepted = converged & (new_difficulty >= low) & (new_difficulty <= high)
        if estimate_slopes:
            low, high = _parameter_bounds(model, "discrimination")
            accepted &= np.all((new_slopes >= low) & (new_slopes <= high), axis=1)
        observed_items = np.any(observed, axis=1)
        observed_items[np.asarray(skip, dtype=np.intp)] = False
        accepted &= observed_items

        updates = {"difficulty": np.where(accepted, new_difficulty, difficulty)}
        if estimate_slopes:
            updates["discrimination"] = np.where(
                accepted[:, None], new_slopes, slopes
            ).reshape(params["discrimination"].shape)
        model.set_parameters(**updates)
        return np.flatnonzero(observed_items & ~accepted)

    def _neg_expected_loglik_with_grad_dichotomous(
        self,
        model: BaseItemModel,
        item_idx: int,
        quad_points: NDArray[np.float64],
        n_k_valid: NDArray[np.float64],
        r_k: NDArray[np.float64],
        params: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        """Evaluate the clipped item objective without assuming optimizer bounds."""
        from mirt.estimation._dichotomous_objective import prepare_dichotomous_objective

        objective = prepare_dichotomous_objective(
            model, item_idx, quad_points, n_k_valid, r_k, self.prob_epsilon
        )
        if objective is None:
            raise ValueError(
                f"Analytic gradient not implemented for {type(model).__name__}"
            )
        return objective(params)

    def _optimize_item_params(
        self,
        model: BaseItemModel,
        item_idx: int,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        quad_points: NDArray[np.float64],
        valid_mask: NDArray[np.bool_] | None = None,
        r_k: NDArray[np.float64] | None = None,
        n_k_valid: NDArray[np.float64] | None = None,
        r_kc: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Optimize item parameters and return optimal parameter vector."""
        current_params, bounds, objective, analytic = self._item_objective(
            model,
            item_idx,
            responses,
            posterior_weights,
            quad_points,
            valid_mask,
            r_k,
            n_k_valid,
            r_kc,
        )
        if objective is None:
            return current_params

        order = graded_order(model, item_idx, current_params.size)
        constrained = {} if order is None else {"constraints": (order.constraint,)}
        try:
            result = minimize(
                objective,
                x0=current_params,
                method="L-BFGS-B" if order is None else "SLSQP",
                jac=analytic,
                bounds=bounds,
                options={
                    "maxiter": self.item_optim_maxiter,
                    # Loosely solved 3PL/4PL items jitter between M-steps,
                    # which stalls EM short of the optimum.
                    "ftol": self._item_optim_ftol(
                        precise=analytic and not model.is_polytomous
                    ),
                },
                **constrained,
            )
            if order is None:
                return result.x
            return _ordered_step(order, objective, bounds, current_params, result.x)
        finally:
            if not analytic:
                self._set_item_params(model, item_idx, current_params)

    def _item_objective(
        self,
        model: BaseItemModel,
        item_idx: int,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        quad_points: NDArray[np.float64],
        valid_mask: NDArray[np.bool_] | None = None,
        r_k: NDArray[np.float64] | None = None,
        n_k_valid: NDArray[np.float64] | None = None,
        r_kc: NDArray[np.float64] | None = None,
    ) -> tuple[
        NDArray[np.float64],
        list[tuple[float, float]],
        Callable[[NDArray[np.float64]], Any] | None,
        bool,
    ]:
        """Return one item's M-step problem in its free-coordinate layout.

        Returns the current free coordinates, the optimizer box, the negative
        expected log-likelihood (minus any item log-prior), and whether that
        objective also returns its gradient. The objective is None when the
        item has no free coordinates or no expected responses. A numerical
        objective writes trial values to the model, which the caller restores.
        """
        item_responses = responses[:, item_idx]
        if valid_mask is None:
            valid_mask = item_responses >= 0

        current_params, bounds = self._get_item_params_and_bounds(model, item_idx)
        if not current_params.size:
            return current_params, bounds, None, True
        # Prepared objectives use the model family's free layout, so items
        # with coordinates fixed by user masks fill those in.
        target, layout_bounds, layout = model, bounds, None
        if model._free_parameter_restrictions:
            target, values, free, layout_bounds = _family_item_layout(model, item_idx)
            layout = None if free.all() else (values, free)

        objective: Callable[
            [NDArray[np.float64]], float | tuple[float, NDArray[np.float64]]
        ]
        if model.is_polytomous:
            from mirt.estimation._polytomous_objective import (
                prepare_polytomous_objective,
            )

            if r_kc is None:
                context = self._fit_context
                if context is None or context.responses is not responses:
                    context = EMFitContext(responses)
                r_kc = context.expected_category_counts(
                    item_idx,
                    model.n_categories[item_idx],
                    posterior_weights,
                    valid_mask,
                )
            if not np.any(r_kc):
                return current_params, bounds, None, True
            prepared = prepare_polytomous_objective(
                target, item_idx, quad_points, r_kc, self.prob_epsilon
            )
            if prepared is None:

                def polytomous_objective(params: NDArray[np.float64]) -> float:
                    self._set_item_params(model, item_idx, params)
                    probs = np.clip(
                        model.probability(quad_points, item_idx),
                        self.prob_epsilon,
                        1 - self.prob_epsilon,
                    )
                    return -float(np.sum(xlogy(r_kc, probs)))

                objective = polytomous_objective
        else:
            if n_k_valid is None:
                n_k_valid = np.sum(posterior_weights[valid_mask], axis=0)
            if not np.any(n_k_valid):
                return current_params, bounds, None, True
            if r_k is None:
                r_k = np.sum(
                    item_responses[valid_mask, None] * posterior_weights[valid_mask, :],
                    axis=0,
                )

            from mirt.estimation._affine_objective import prepare_affine_objective
            from mirt.estimation._dichotomous_objective import (
                prepare_dichotomous_objective,
            )

            prepared = prepare_dichotomous_objective(
                target,
                item_idx,
                quad_points,
                n_k_valid,
                r_k,
                self.prob_epsilon,
                layout_bounds,
            )
            if prepared is None:
                prepared = prepare_affine_objective(
                    target,
                    item_idx,
                    quad_points,
                    n_k_valid,
                    r_k,
                    self.prob_epsilon,
                    layout_bounds,
                )
            if prepared is None:

                def dichotomous_objective(params: NDArray[np.float64]) -> float:
                    self._set_item_params(model, item_idx, params)
                    probs = np.clip(
                        model.probability(quad_points, item_idx),
                        self.prob_epsilon,
                        1 - self.prob_epsilon,
                    )
                    return -float(
                        np.sum(xlogy(r_k, probs) + xlog1py(n_k_valid - r_k, -probs))
                    )

                objective = dichotomous_objective

        analytic = prepared is not None
        if prepared is not None:
            objective = (
                prepared if layout is None else _fix_coordinates(prepared, *layout)
            )
        if self._prior_penalty is not None:
            objective, bounds = self._prior_penalty.penalize(
                model, item_idx, objective, bounds, analytic=analytic
            )
        return current_params, bounds, objective, analytic

    def _optimize_item(
        self,
        model: BaseItemModel,
        item_idx: int,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        quad_points: NDArray[np.float64],
        valid_mask: NDArray[np.bool_] | None = None,
        r_k: NDArray[np.float64] | None = None,
        n_k_valid: NDArray[np.float64] | None = None,
        r_kc: NDArray[np.float64] | None = None,
    ) -> None:
        optimal = self._optimize_item_params(
            model,
            item_idx,
            responses,
            posterior_weights,
            quad_points,
            valid_mask,
            r_k,
            n_k_valid,
            r_kc,
        )
        self._set_item_params(model, item_idx, optimal)

    def _optimize_item_return(
        self,
        model: BaseItemModel,
        item_idx: int,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        quad_points: NDArray[np.float64],
        valid_mask: NDArray[np.bool_] | None = None,
        r_k: NDArray[np.float64] | None = None,
        n_k_valid: NDArray[np.float64] | None = None,
        r_kc: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Optimize item parameters and return the result (for parallel execution)."""
        from copy import deepcopy

        # Numerical objectives update whole parameter arrays before evaluating
        # one item. Keep custom state and instance method overrides as well as
        # parameters when isolating workers from lost trial updates.
        local_model = deepcopy(model)
        return self._optimize_item_params(
            local_model,
            item_idx,
            responses,
            posterior_weights,
            quad_points,
            valid_mask,
            r_k,
            n_k_valid,
            r_kc,
        )

    def _resolve_se_method(
        self,
        model: BaseItemModel,
        person_weights: NDArray[np.float64] | None,
    ) -> str:
        """Choose the estimator behind ``se_method`` for this fit."""
        from mirt.estimation._louis_information import has_analytic_item_derivatives

        # Survey-weighted subclasses keep their documented complete-data errors.
        if person_weights is not None or self.se_method == "complete_data":
            return "complete_data"
        if self.se_method != "auto":
            return self.se_method
        # The exact information recomputes posteriors from the model, so it
        # must not replace an estimator-specific likelihood.
        own_likelihood = any(
            name in vars(self)
            or getattr(type(self), name) is not getattr(EMEstimator, name)
            for name in ("_e_step", "_compute_log_likelihoods")
        )
        if own_likelihood or not has_analytic_item_derivatives(model):
            return "complete_data"
        return "oakes"

    def _prior_mass(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the quadrature prior mass behind the final posterior."""
        from mirt.estimation.standard_errors import _infer_prior_mass

        quadrature = self._quadrature
        if self._latent_density is None:
            return _infer_prior_mass(model, responses, posterior_weights, quadrature)
        log_mass = self._latent_density.log_quadrature_mass(
            quadrature.nodes, quadrature.weights
        )
        mass = np.exp(log_mass - np.max(log_mass))
        return mass / mass.sum()

    def _compute_standard_errors(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        *,
        person_weights: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Return standard errors by the resolved ``se_method``.

        ``posterior_weights`` are final E-step posteriors already scaled by
        any pattern frequencies. The method and free-parameter covariance are
        kept in ``_se_details`` for the fit result. Coordinates tied by
        equality constraints share the error of their group.
        """
        from mirt.estimation.standard_errors import estimate_covariance

        method = self._resolve_se_method(model, person_weights)
        if method == "complete_data":
            errors = self._complete_data_standard_errors(
                model, responses, posterior_weights, person_weights=person_weights
            )
            if self._tied is not None:
                errors = self._tied.combine_standard_errors(errors)
            self._se_details = ("complete_data", None)
            return errors

        context = self._fit_context
        frequencies = (
            self._pattern_frequencies
            if context is not None and context.responses is responses
            else None
        )
        estimate = estimate_covariance(
            model,
            responses,
            self._quadrature,
            self._prior_mass(model, responses, posterior_weights),
            method,
            frequencies=frequencies,
            h=self.se_step_size,
            bounds=lambda name: _parameter_bounds(model, name),
            prior_information=self._prior_information(model),
            tied=self._tied,
        )
        self._se_details = (method, estimate.covariance)
        return estimate.standard_errors

    def _prior_information(
        self, model: BaseItemModel
    ) -> dict[str, NDArray[np.float64]] | None:
        """Return the item log-prior's negative curvature, or None without priors."""
        penalty = self._prior_penalty
        return None if penalty is None else penalty.information(model)

    def _complete_data_standard_errors(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        *,
        person_weights: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Itemwise diagonal curvature of the expected complete-data likelihood.

        Parameters shared by all items are differenced in the expected
        log-likelihood summed over items. Item priors add their curvature to
        each free coordinate's.
        """
        from mirt.estimation._item_information import item_standard_errors
        from mirt.estimation._shared_step import binary_category_counts
        from mirt.estimation.se_methods import shared_parameter_standard_errors

        prior = self._prior_information(model) or {}
        context = self._fit_context
        if context is None or context.responses is not responses:
            context = EMFitContext(responses)
        if self.se_step_size == 1e-5:
            analytic = item_standard_errors(
                model,
                responses,
                posterior_weights,
                self._quadrature.nodes,
                self.prob_epsilon,
                person_weights=person_weights,
                context=context,
            )
            if analytic is not None:
                return {
                    name: _add_prior_curvature(errors, prior.get(name))
                    for name, errors in analytic.items()
                }
        standard_errors: dict[str, NDArray[np.float64]] = {}
        params = model.parameters
        free_masks = model.free_parameter_masks
        correct = observed = None
        category_counts: dict[int, NDArray[np.float64]] = {}
        if not model.is_polytomous:
            correct, observed = context.expected_counts(
                posterior_weights, person_weights
            )

        def item_category_counts(item_idx: int) -> NDArray[np.float64]:
            if item_idx not in category_counts:
                category_counts[item_idx] = context.expected_category_counts(
                    item_idx,
                    model.n_categories[item_idx],
                    posterior_weights,
                    person_weights,
                )
            return category_counts[item_idx]

        for name, values in params.items():
            free_mask = free_masks[name]
            if not np.any(free_mask) or name in model._shared_parameters:
                standard_errors[name] = model._expand_parameter_standard_errors(
                    name, np.zeros_like(values)
                )
                continue

            se = np.zeros_like(values)

            for item_idx in range(model.n_items):
                if not np.any(free_mask[item_idx]):
                    continue
                if model.is_polytomous:
                    counts = item_category_counts(item_idx)
                    item_observed = counts.sum(axis=1)
                    item_correct = None
                else:
                    counts = None
                    item_observed = observed[item_idx]
                    item_correct = correct[item_idx]
                item_se = self._compute_item_se(
                    model,
                    item_idx,
                    name,
                    responses,
                    posterior_weights,
                    r_k=item_correct,
                    n_k_valid=item_observed,
                    r_kc=counts,
                )
                se[item_idx] = item_se

            se[~free_mask] = 0.0
            standard_errors[name] = model._expand_parameter_standard_errors(
                name, _add_prior_curvature(se, prior.get(name))
            )

        if _free_shared_parameters(model):
            shared_counts = (
                [item_category_counts(item) for item in range(model.n_items)]
                if model.is_polytomous
                else binary_category_counts(correct, observed)
            )
            shared = shared_parameter_standard_errors(
                model,
                self._quadrature.nodes,
                shared_counts,
                self.se_step_size,
                epsilon=self.prob_epsilon,
                exact=self.se_step_size == 1e-5,
            )
            for name, errors in shared.items():
                standard_errors[name] = model._expand_parameter_standard_errors(
                    name, errors
                )
        return standard_errors

    def _compute_item_se(
        self,
        model: BaseItemModel,
        item_idx: int,
        param_name: str,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        *,
        r_k: NDArray[np.float64] | None = None,
        n_k_valid: NDArray[np.float64] | None = None,
        r_kc: NDArray[np.float64] | None = None,
    ) -> float | NDArray[np.float64]:
        return _compute_item_se_curvature(
            model,
            item_idx,
            param_name,
            responses,
            self._quadrature,
            posterior_weights,
            self.se_step_size,
            scheme="central",
            r_k=r_k,
            n_k_valid=n_k_valid,
            r_kc=r_kc,
            epsilon=self.prob_epsilon,
        )
