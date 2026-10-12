from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Iterable, Sequence
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import xlog1py, xlogy

from mirt._backend_config import (
    gpu_dispatch_allowed,
    should_use_rust,
    validate_use_gpu,
)
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
    _family_redundancy,
    _free_shared_parameters,
    _parameter_bounds,
    _validate_start,
    _with_family_ties,
)
from mirt.estimation.quadrature import (
    GaussHermiteQuadrature,
    QuadratureRule,
    RectangularQuadrature,
    build_quadrature,
    resolve_quadpts,
    validate_quadpts,
    validate_quadrature_rule,
)
from mirt.estimation.se_methods import _compute_item_se_curvature
from mirt.estimation.standard_errors import StandardErrorMethod, validate_se_method
from mirt.exceptions import MirtValidationError

if TYPE_CHECKING:
    from mirt.estimation.latent_density import LatentDensity
    from mirt.estimation.priors import Prior
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult

# Relative function-change tolerance for precise M-steps: analytic
# dichotomous, graded, generalized partial credit and nominal items (native
# or NumPy), shared and tied-item steps, and every item optimizer under
# SQUAREM. Item objectives are large sums, so looser values can stop with
# parameter errors near 1e-3, and the resulting noisy M-steps slow EM and
# defeat extrapolation.
_PRECISE_ITEM_FTOL = 1e-10
# Quadrature grids with more nodes than this, or whose E-step arrays would
# need more memory than this, trigger a warning.
_LARGE_GRID_NODES = 1_000_000
_LARGE_GRID_BYTES = 1 << 30
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
# Estimator methods whose EMEstimator versions parameter-expansion steps
# assume: the item likelihood, the E-step and the M-step.
_PX_HOOKS = (
    "_evaluate",
    "_e_step",
    "_compute_log_likelihoods",
    "_m_step",
    "_m_step_common_slope",
    *_ITEM_OPTIMIZATION_HOOKS,
)
# Parameter-expansion steps stop once the objective changes by less than this
# multiple of ``tol``, so the last iterations are plain EM and the fit ends at
# the EM fixed point of the quadrature grid.
_PX_FINISH = 100.0
# Parameter expansion needs rectangular grids with nodes at most this far
# apart. On coarser grids, and on Gauss-Hermite grids, the posterior moments
# at the EM fixed point of the grid stray from those of the standard normal,
# so expansion steps pull away from that fixed point and can slow EM down.
_PX_MAX_SPACING = 0.4
# Marks item priors that the native 3PL iteration cannot apply.
_UNSUPPORTED_PRIORS = object()


def _weight_posterior(
    posterior: NDArray[np.float64], frequencies: NDArray[np.float64] | None
) -> NDArray[np.float64]:
    """Scale compressed-pattern posteriors by their pattern frequencies."""
    return posterior if frequencies is None else posterior * frequencies[:, None]


def _needs_precise_m_step(model: BaseItemModel) -> bool:
    """Whether polytomous items of ``model`` need the precise M-step tolerance.

    Loose relative tolerances stop graded, generalized partial credit and
    nominal item optimizers early, and the noisy M-steps multiply the EM
    iterations (up to tenfold for a GPCM) and stall EM short of the optimum.
    Partial-credit and rating-scale items converge cleanly and would only
    slow down.
    """
    from mirt.models.polytomous import (
        GeneralizedPartialCredit,
        GradedResponseModel,
        NominalResponseModel,
    )

    return type(model) in (
        GradedResponseModel,
        GeneralizedPartialCredit,
        NominalResponseModel,
    )


def _warn_large_grid(
    model: BaseItemModel, n_quadpts: int, n_rows: int, *, stacklevel: int = 3
) -> None:
    """Warn before an EM quadrature grid of over a million nodes or a gigabyte.

    The memory estimate counts the posterior and log-likelihood arrays over
    response rows and nodes, and the item probabilities over nodes, at eight
    bytes each.
    """
    # Python integers keep NumPy integer point counts from overflowing.
    n_nodes = int(n_quadpts) ** int(model.n_factors)
    categories = max(np.atleast_1d(model.n_categories)) if model.is_polytomous else 2
    n_bytes = 8 * n_nodes * (2 * int(n_rows) + model.n_items * int(categories))
    if n_nodes <= _LARGE_GRID_NODES and n_bytes <= _LARGE_GRID_BYTES:
        return
    from mirt.models.bifactor import BifactorModel, GradedBifactorModel
    from mirt.models.testlet import TestletModel

    advice = (
        "mirt.bfactor or BifactorEMEstimator integrates bifactor and testlet "
        "models exactly on two-dimensional grids"
        if isinstance(model, (BifactorModel, GradedBifactorModel, TestletModel))
        else "reduce n_quadpts (omit it for the dimension-dependent default), or "
        "integrate by sampling with fit_mirt(estimation='MHRM') or QMCEMEstimator"
    )
    warnings.warn(
        f"EM quadrature over {model.n_factors} factors with {n_quadpts} points "
        f"each needs {n_nodes:,} nodes and about {n_bytes / 2**30:.1f} GiB for "
        f"{n_rows:,} response rows, which is slow and may exhaust memory; "
        f"{advice}",
        RuntimeWarning,
        stacklevel=stacklevel,
    )


class _PartialCreditCounts(NamedTuple):
    """Response statistics of a PCM slope step that are constant over a fit.

    Attributes
    ----------
    scores : ndarray of shape (n_rows,)
        Total observed score of each response row.
    category_counts : ndarray of shape (n_items, n_categories)
        Weighted number of rows in each category of each item.
    observed : ndarray of shape (n_rows, n_items) or None
        Observed-response indicators, or None when no response is missing.
    """

    scores: NDArray[np.float64]
    category_counts: NDArray[np.float64]
    observed: NDArray[np.float64] | None


def _partial_credit_counts(
    responses: NDArray[np.int_], weights: NDArray[np.float64], width: int
) -> _PartialCreditCounts:
    """Tabulate the scores and category counts of ``responses`` for a PCM.

    ``weights`` weights each response row, such as by its pattern frequency;
    missing (negative) responses count in no category.
    """
    valid = responses >= 0
    codes = np.where(valid, responses, 0)
    cells = np.arange(responses.shape[1]) * width + codes
    category_counts = np.bincount(
        cells[valid],
        weights=np.broadcast_to(weights[:, None], codes.shape)[valid],
        minlength=responses.shape[1] * width,
    ).reshape(responses.shape[1], width)
    return _PartialCreditCounts(
        codes.sum(axis=1, dtype=np.float64),
        category_counts,
        None if valid.all() else valid.astype(np.float64),
    )


def _common_slope_newton(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    counts: Any,
    bounds: tuple[float, float],
    max_iter: int = 50,
    *,
    prior: tuple[Prior, int] | None = None,
) -> float | None:
    """Maximize the expected log-likelihood over a 1PL or PCM common slope.

    The item locations are held. Unidimensional logistic and partial-credit
    log-likelihoods are concave in the slope, so damped Newton steps on its
    exact first and second derivatives converge. ``counts`` holds the 1PL's
    expected correct and observed counts, of shape ``(n_items, n_points)``,
    or, for a PCM, the :class:`_PartialCreditCounts` of the responses, with
    rows weighted like the posterior's, and the weighted posterior. With
    ``prior``, a slope prior and the number of free coordinates that carry
    it, the objective is the log-posterior. Returns the new slope, or None
    if it does not raise the objective.
    """

    if model.is_polytomous:
        statistics, posterior = counts
        n_categories = np.asarray(model.n_categories)
        width = statistics.category_counts.shape[1]
        steps = model.parameters["steps"][:, : width - 1]
        # The logit of category c is a * F_c with F_c = c theta - S_c and
        # S_c the sum of the first c steps; unused categories never occur.
        cumulative = np.column_stack(
            (np.zeros(model.n_items), np.cumsum(steps, axis=1))
        )
        features = (
            np.arange(width)[None, None, :] * theta[None, :, None]
            - cumulative[:, None, :]
        )
        squared_features = features**2
        unused = np.where(
            np.arange(width)[None, None, :] < n_categories[:, None, None],
            0.0,
            -np.inf,
        )
        # Observed features sum over items: F = theta * total score - sum S,
        # where the step sums of the observed categories are fit constants.
        observed_feature = float(
            theta @ (posterior.T @ statistics.scores)
            - np.sum(cumulative * statistics.category_counts)
        )
        if statistics.observed is None:
            totals = np.broadcast_to(posterior.sum(axis=0), (model.n_items, theta.size))
        else:
            totals = (posterior.T @ statistics.observed).T

        def evaluate(slope: float) -> tuple[float, float, float]:
            logits = slope * features + unused
            top = logits.max(axis=2, keepdims=True)
            probability = np.exp(logits - top)
            normalizer = probability.sum(axis=2)
            probability /= normalizer[..., None]
            log_norm = top[..., 0] + np.log(normalizer)
            mean = np.sum(probability * features, axis=2)
            variance = np.sum(probability * squared_features, axis=2) - mean**2
            value = slope * observed_feature - float(np.sum(totals * log_norm))
            return (
                value,
                observed_feature - float(np.sum(totals * mean)),
                float(-np.sum(totals * variance)),
            )

    else:
        correct, observed = counts
        centered = theta[None, :] - model.parameters["difficulty"][:, None]

        def evaluate(slope: float) -> tuple[float, float, float]:
            logits = slope * centered
            log_p = -np.logaddexp(0.0, -logits)
            log_q = -np.logaddexp(0.0, logits)
            probability = np.exp(log_p)
            value = float(np.sum(correct * log_p + (observed - correct) * log_q))
            residual = correct - observed * probability
            gradient = float(np.sum(residual * centered))
            curvature = float(
                -np.sum(observed * probability * (1.0 - probability) * centered**2)
            )
            return value, gradient, curvature

    objective = evaluate if prior is None else _with_slope_prior(evaluate, *prior)
    low, high = bounds
    current = float(model.parameters["discrimination"][0])
    start_value, gradient, curvature = objective(current)
    slope, value = current, start_value
    for _ in range(max_iter):
        if not (np.isfinite(gradient) and curvature < 0.0):
            break
        step = -gradient / curvature
        trial = float(np.clip(slope + step, low, high))
        trial_value, trial_gradient, trial_curvature = objective(trial)
        # Halve steps that overshoot; the objective is concave.
        while trial_value < value and abs(trial - slope) > 1e-12:
            trial = 0.5 * (trial + slope)
            trial_value, trial_gradient, trial_curvature = objective(trial)
        if trial_value < value:
            break
        converged = abs(trial - slope) <= 1e-10 * max(1.0, abs(slope))
        slope, value = trial, trial_value
        gradient, curvature = trial_gradient, trial_curvature
        if converged:
            break
    return slope if value > start_value else None


def _with_slope_prior(
    evaluate: Callable[[float], tuple[float, float, float]],
    prior: Prior,
    count: int,
) -> Callable[[float], tuple[float, float, float]]:
    """Add ``count`` copies of a common slope's log-prior to its objective.

    The value and slope of the log-prior enter exactly, its curvature only
    where it is negative, so the Newton steps of :func:`_common_slope_newton`
    keep ascending.
    """

    def penalized(slope: float) -> tuple[float, float, float]:
        value, gradient, curvature = evaluate(slope)
        point = np.array([slope])
        return (
            value + count * float(prior.log_pdf(point)[0]),
            gradient + count * float(prior.grad_log_pdf(point)[0]),
            curvature + count * min(float(prior.hess_log_pdf(point)[0]), 0.0),
        )

    return penalized


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


def _latent_scale_layout(model: BaseItemModel) -> tuple[str, str, bool] | None:
    """Return the slopes and locations that parameter expansion rescales.

    The families are the built-in 1PL, 2PL, graded response, (generalized)
    partial credit and nominal models and the slope-intercept
    ``MultidimensionalModel``. The flag is whether the locations are
    intercepts, ``a' theta + d``, rather than locations scaled by the slope
    sum, ``a' theta - sum(a) b``. Other families, such as the 3PL, 4PL and
    rating scale models, return None.
    """
    from mirt.models.dichotomous import OneParameterLogistic, TwoParameterLogistic
    from mirt.models.multidimensional import MultidimensionalModel
    from mirt.models.polytomous import (
        GeneralizedPartialCredit,
        GradedResponseModel,
        NominalResponseModel,
        PartialCreditModel,
    )

    layout = {
        OneParameterLogistic: ("discrimination", "difficulty", False),
        TwoParameterLogistic: ("discrimination", "difficulty", False),
        GradedResponseModel: ("discrimination", "thresholds", False),
        GeneralizedPartialCredit: ("discrimination", "steps", False),
        PartialCreditModel: ("discrimination", "steps", False),
        MultidimensionalModel: ("slopes", "intercepts", True),
        NominalResponseModel: ("slopes", "intercepts", True),
    }.get(type(model))
    if layout is None or set(model._parameters) != set(layout[:2]):
        return None
    return layout


def _latent_scale_updates(
    model: BaseItemModel,
    layout: tuple[str, str, bool],
    nodes: NDArray[np.float64],
    node_counts: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]] | None:
    """Return item parameters that absorb the posterior latent mean and scale.

    The expected respondents at each node give the latent mean ``m`` and
    covariance ``C`` that maximize the expected complete-data
    log-likelihood of an expanded model with a free Gaussian population.
    Writing ``theta = m + L theta'`` with ``C = L L'`` returns that model to
    the standard normal population: every item keeps its family, with
    slopes ``a L`` and intercepts ``d + a'm`` or locations
    ``(sum(a) b - a'm) / sum(a L)``, in one dimension ``(b - m) / s``. ``L``
    is the Cholesky factor when the slopes of each item (or category) are
    all free or all fixed, so that the factors can rotate, and otherwise the
    diagonal of standard deviations, which keeps fixed zero loadings at
    zero. The diagonal is also tried when the Cholesky factor would leave a
    parameter box.

    Returns
    -------
    dict of str to ndarray or None
        The new slopes and locations, or None when the step would change
        nothing, move a free coordinate out of its box or disorder graded
        thresholds.
    """
    from mirt.estimation._acceleration import _keeps_threshold_order

    total = float(node_counts.sum())
    if not (np.isfinite(total) and total > 0.0):
        return None
    mean = node_counts @ nodes / total
    centered = nodes - mean
    covariance = (centered * node_counts[:, None]).T @ centered / total
    slope_name, location_name, intercepts = layout
    n_factors = model.n_factors
    masks = model.free_parameter_masks
    params = model.parameters
    slopes, locations = params[slope_name], params[location_name]
    rows = masks[slope_name].reshape(-1, n_factors)
    factors = []
    if np.all(rows.any(axis=1) == rows.all(axis=1)):
        try:
            factors.append(np.linalg.cholesky(covariance))
        except np.linalg.LinAlgError:
            pass
    if n_factors > 1 or not factors:
        factors.append(np.diag(np.sqrt(np.diag(covariance))))
    # The slopes of each item (or category) on the factors, as rows.
    per_factor = slopes.ndim > (locations.ndim if intercepts else 1)
    a = slopes if per_factor else slopes[..., None]
    shift = a @ mean

    def within_box(name: str, values: NDArray[np.float64]) -> bool:
        low, high = _parameter_bounds(model, name)
        free = values[masks[name]]
        return bool(
            np.all(np.isfinite(values)) and np.all((free >= low) & (free <= high))
        )

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for factor in factors:
            if not (np.all(np.isfinite(factor)) and np.all(np.diag(factor) > 0.0)):
                continue
            if (
                np.max(np.abs(mean)) <= 1e-12
                and np.max(np.abs(factor - np.eye(n_factors))) <= 1e-12
            ):
                return None
            new_a = a @ factor
            if intercepts:
                new_locations = locations + shift
            elif n_factors == 1:
                new_locations = (locations - mean[0]) / factor[0, 0]
            else:
                scale = a.sum(axis=-1)[:, None]
                new_scale = new_a.sum(axis=-1)[:, None]
                item_locations = locations.reshape(model.n_items, -1)
                new_locations = (
                    (scale * item_locations - shift[:, None]) / new_scale
                ).reshape(locations.shape)
            updates = {
                slope_name: np.where(
                    masks[slope_name], new_a.reshape(slopes.shape), slopes
                ),
                location_name: np.where(masks[location_name], new_locations, locations),
            }
            if all(
                within_box(name, values) for name, values in updates.items()
            ) and _keeps_threshold_order(model, updates):
                return updates
    return None


class EMEstimator(BaseEstimator):
    """Marginal maximum likelihood estimation by the EM algorithm.

    Each M-step optimizes the item parameters item by item. Parameters shared
    by all items, such as the common thresholds of the rating scale models,
    are then optimized jointly with the item parameters held fixed, which
    keeps every iteration a monotone conditional maximization step.

    Built-in testlet models are fitted by
    :class:`~mirt.estimation.bifactor_em.BifactorEMEstimator` on one
    two-dimensional grid per testlet, with ``n_quadpts`` (21 by default) per
    dimension, unless a latent density, prior, constraint, acceleration or
    rectangular quadrature needs the product grid. Models whose likelihood
    the item M-steps cannot represent, such as
    :class:`~mirt.models.mixture.MixtureIRT`, or whose model-level
    parameters they would leave unchanged, raise ``MirtModelError``.

    Parameters
    ----------
    n_quadpts : int, optional
        Quadrature points per latent dimension, at least 5. The default
        depends on the number of factors, as in R mirt: 61 points for one
        factor, 31 for two, 15 for three, 9 for four, 7 for five and 5 for
        six (see :func:`~mirt.estimation.quadrature.default_quadpts`). More
        factors need an explicit value, or sampling-based integration such
        as MHRM or :class:`~mirt.estimation.mcem.QMCEMEstimator`. A grid
        whose E-step arrays would exceed about a gigabyte warns.
    max_iter : int, default=500
        Maximum number of M-steps.
    tol : float, default=1e-4
        Convergence tolerance for the absolute change in marginal
        log-likelihood (log-posterior with ``item_priors``) between
        consecutive EM iterates.
    verbose : bool, default=False
        Print the log-likelihood at each recorded iterate.
    quadrature : {"auto", "rectangular", "gauss-hermite"}, default="auto"
        Quadrature rule of the product grid. ``"rectangular"`` places
        ``n_quadpts`` equally spaced nodes on ``[-6, 6]`` in every dimension
        and weights them by the normal density, as R mirt does; long or
        highly discriminating tests have sharply peaked person likelihoods,
        which this even grid integrates far more accurately than
        Gauss-Hermite nodes. ``"gauss-hermite"`` uses Gauss-Hermite nodes,
        which are more accurate for coarse grids. ``"auto"`` is rectangular
        for 15 or more points per dimension and Gauss-Hermite otherwise.
    latent_density : LatentDensity or str, optional
        Latent density specification. Defaults to a fixed Gaussian density
        with the prior mean and covariance passed to :meth:`fit`, which
        ``"gaussian"`` (or ``"normal"``) also selects. ``"empirical"`` (or
        ``"histogram"``/``"eh"``) estimates a histogram over the quadrature
        nodes, for any number of factors. The univariate ``"davidian"``
        (Davidian curve of degree 4), ``"mixture"`` (two Gaussian components)
        and ``"ehw"`` (``"empiricalhist_woods"``, the histogram standardized
        by Woods' interpolation-extrapolation) require one factor; these
        three are held at mean 0 and variance 1 in every EM cycle, so the
        item parameters stay on the scale of a standard-normal fit.
        Each fit builds a new density from a name, while a
        :class:`~mirt.estimation.latent_density.LatentDensity` instance is
        used, and updated, as given.
    prob_epsilon : float, default=1e-10
        Probability clipping bound used by item objectives. The analytic
        objectives of built-in 1PL, 2PL and slope-intercept items, and of 3PL
        items near their upper asymptote, keep the exact likelihood beyond
        it, so that items whose expected responses saturate keep a gradient.
    item_optim_maxiter : int, default=50
        Iteration limit for itemwise numerical optimizers.
    item_optim_ftol : float, default=1e-6
        Relative function-change tolerance for itemwise numerical optimizers.
        Built-in dichotomous items with analytic gradients (1PL-4PL and the
        logistic items of multidimensional and bifactor models) and graded,
        generalized partial credit and nominal items, in the native and the
        NumPy M-steps, use at most ``1e-10``, because looser relative
        criteria make their EM iterations jitter and stall short of the
        optimum. Built-in 1PL and 2PL items and the slope-intercept items of
        :class:`~mirt.models.multidimensional.MultidimensionalModel` (the
        multidimensional 2PL of model-syntax fits) are solved jointly by
        Newton's method to a tight tolerance, with coordinates fixed by
        masks or a loading pattern held at their values. They use the
        numerical optimizer only as a fallback, for example when an
        estimate reaches a parameter bound or a 2PL item has a fixed
        difficulty but free slopes.
    se_step_size : float, default=1e-5
        Finite-difference step for itemwise complete-data standard errors of
        custom item models, and for marginal-likelihood differences of models
        without exact item derivatives.
    n_jobs : int, default=1
        Worker threads for itemwise optimization. ``-1`` uses every CPU. The
        native GRM, GPCM, PCM and 3PL M-steps share one native pool of that
        size per fit; other items run in Python threads.
    use_gpu : bool or "auto", default="auto"
        Whether to use a GPU likelihood backend when one is available.
        ``"auto"`` follows :func:`~mirt.set_backend`: the ``"auto"`` and
        ``"gpu"`` preferences use an available GPU, ``"rust"`` and
        ``"numpy"`` do not. ``True`` and ``False`` override the preference.
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
        unobserved latent trait and understates uncertainty, often two- to
        fivefold for nominal and multidimensional models. ``"auto"`` uses
        ``"oakes"`` for every model whose likelihood is a product of item
        curves with exact item-local derivatives: the built-in 1PL-4PL, GRM,
        GPCM, PCM, NRM, RSM and GRSM families with any number of factors
        (including model-syntax fits), ``MultidimensionalModel``,
        ``BifactorModel``, the other curve-product families listed in
        :func:`~mirt.estimation._louis_information.supports_louis_information`,
        :class:`~mirt.models.custom.CustomItemModel` and mixed-format
        combinations of them. Other models, such as testlet or mixture
        models, and subclasses that replace the E-step likelihood get
        ``"complete_data"``. ``FitResult.se_method`` records the estimator
        used. The latent density is treated as fixed, and coordinates at an
        optimizer bound (for example a guessing parameter of 0) are held
        fixed with ``NaN`` standard errors. An exploratory model with ``F``
        factors is rotationally unidentified, so the matrix methods hold the
        upper triangle of the first ``F - 1`` items' slopes at their
        estimates, as R mirt fixes it at zero; those slopes get ``NaN``
        standard errors. The matrix methods cost O(N Q P^2) for N response
        patterns, Q nodes and P parameters (O(N Q J^2) for J dichotomous
        items); models outside the families above fall back to O(P^2)
        marginal likelihood evaluations.
    accelerate : {"none", "squarem"}, default="none"
        EM acceleration. ``"squarem"`` applies SQUAREM extrapolation (Varadhan
        and Roland, 2008) with step-length adaptation, projection onto the
        item parameter bounds and a monotone fallback to the plain EM step.
        Extrapolation needs precise M-steps, so item optimizers then use a
        relative tolerance of at most ``1e-10``.
        SQUAREM extrapolates the fused native 3PL iteration wherever plain EM
        would use it, and the generic E- and M-steps otherwise. It does not
        take the parameter-expansion steps of ``px_em``. A latent density
        other than a fixed Gaussian falls back to plain EM with a warning.
        With acceleration, ``max_iter`` still bounds the number of M-steps,
        ``n_iterations`` counts E-steps, which includes the evaluation of
        each extrapolated point, and ``convergence_history`` records the
        log-likelihood at accepted iterates only.
    px_em : bool, default=True
        Whether plain EM (``accelerate="none"``) takes parameter-expanded
        steps (PX-EM; Liu, Rubin and Wu, 1998). After each M-step, the mean
        and standard deviation of the latent trait that the E-step posterior
        implies (in several dimensions, the Cholesky factor of its
        covariance) are absorbed into the item slopes and locations, which
        keeps the standard normal latent density and every item family. EM
        then no longer creeps along the scale of the latent trait, so fits
        typically take five to ten times fewer iterations to the same
        optimum. A step whose next E-step lowers the log-likelihood is
        undone, at the cost of one more E-step, and the fit continues with
        plain EM, as it also does once the log-likelihood changes by less
        than ``100 * tol``; the recorded log-likelihood therefore never
        decreases, and the fit converges to the fixed point of plain EM on
        its quadrature grid. Steps apply to the built-in 2PL, graded
        response, generalized partial credit and nominal models with any
        number of factors, the 1PL and PCM with an estimated common slope,
        and :class:`~mirt.models.multidimensional.MultidimensionalModel`,
        with the default fixed standard normal latent density, every slope
        free or fixed at zero by a loading pattern, no other masks, item
        priors or user equality constraints, and the estimator's own E- and
        M-steps, on rectangular grids with nodes at most 0.4 apart (at least
        31 points per dimension, as the defaults for one and two factors).
        On coarser and Gauss-Hermite grids the EM fixed point of the grid
        drifts from the standard normal scale that the steps restore, and
        they could slow EM down. Item priors are not invariant under the
        rescaling, which would change the log-posterior, and mixed-format,
        testlet and multiple-group fits have their own E- and M-steps. A
        step that would move a free coordinate out of its optimizer box, or
        disorder graded thresholds, is skipped. Other fits use plain EM, as
        ``px_em=False`` always does.
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
        priors. The fused native 3PL iteration applies normal, log-normal and
        beta priors on 3PL items. Other priors bypass it, and every prior
        bypasses the native polytomous M-step; batched NumPy M-steps that
        include the log-prior stand in for them (Newton steps for 1PL, 2PL
        and MIRT items, Fisher scoring for unidimensional 3PL, graded and
        partial credit items), and the 1PL and PCM common slope keeps its
        conditional step.
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
    Liu, C., Rubin, D. B., & Wu, Y. N. (1998). Parameter expansion to
    accelerate EM: The PX-EM algorithm. *Biometrika*, 85(4), 755-770.

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
        n_quadpts: int | None = None,
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
        quadrature: QuadratureRule = "auto",
        px_em: bool = True,
    ) -> None:
        super().__init__(max_iter, tol, verbose)

        self.n_quadpts = None if n_quadpts is None else validate_quadpts(n_quadpts)
        self.quadrature = validate_quadrature_rule(quadrature)
        self.prob_epsilon = prob_epsilon
        self.item_optim_maxiter = item_optim_maxiter
        self.item_optim_ftol = item_optim_ftol
        self.se_step_size = se_step_size
        self.n_jobs = n_jobs
        self.use_gpu = validate_use_gpu(use_gpu)
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
        if not isinstance(px_em, (bool, np.bool_)):
            raise MirtValidationError(
                "px_em must be a boolean",
                parameter="px_em",
                value=px_em,
                expected="bool",
            )
        self.px_em = bool(px_em)
        self.item_priors = validate_item_priors(item_priors)
        self._prior_penalty: ItemPriorPenalty | None = None
        self.se_method = validate_se_method(se_method)
        self.constraints = validate_equality_constraints(constraints)
        # Coordinates tied by ``constraints`` in the model being fitted.
        self._tied: TiedCoordinates | None = None
        # Method and free-parameter covariance behind the latest standard errors.
        self._se_details: tuple[str, NDArray[np.float64] | None] | None = None
        self._quadrature: GaussHermiteQuadrature | RectangularQuadrature | None = None
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
        # The fit's responses and their PCM slope-step statistics.
        self._slope_counts: tuple[NDArray[np.int_], _PartialCreditCounts] | None = None

    @property
    def _should_use_gpu(self) -> bool:
        """Whether ``use_gpu``, the backend preference and a GPU allow GPU use."""
        return gpu_dispatch_allowed(self.use_gpu) and is_gpu_available()

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
        MirtDataError
            If an observed response is not a category of its item, for
            example a code 2 for a dichotomous model.
        MirtValidationError
            If ``constraints`` do not match the model's free coordinates.
        """
        from mirt.estimation._patterns import supports_pattern_compression
        from mirt.estimation._shared_step import resolve_equality_constraints
        from mirt.estimation.bifactor_em import reduced_testlet_estimator
        from mirt.estimation.latent_density import GaussianDensity, create_density

        start = _validate_start(start)
        # A testlet model is a bifactor model, whose dimension reduction
        # replaces the product grid over every testlet when no option needs it.
        reduced = reduced_testlet_estimator(self, model, prior_mean, prior_cov)
        if reduced is not None:
            result = reduced.fit(model, responses, start=start)
            self._convergence_history = reduced.convergence_history
            return result
        responses = self._validate_responses(responses, model.n_items, model)
        # Family ties, such as the common slope of a 1PL, join the user's.
        self._tied = resolve_equality_constraints(
            _with_family_ties(self.constraints, model), model
        )
        self._reject_unsupported_models(model)
        n_points = resolve_quadpts(self.n_quadpts, model.n_factors)
        _warn_large_grid(model, n_points, responses.shape[0])
        self._quadrature = build_quadrature(n_points, model.n_factors, self.quadrature)

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
                self._slope_counts = None

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
            # The model counts its family-tied coordinates once already.
            n_params -= self._tied.n_redundant - _family_redundancy(model)
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
        """Iterate plain EM and return the final posterior, LL and status.

        With ``px_em``, each M-step is followed by a parameter-expansion step
        (:meth:`_expand_latent_scale`) until the objective changes by less
        than ``_PX_FINISH * tol``. If the next E-step finds a lower objective
        than the previous iterate, the expansion is undone, that E-step is
        repeated and the fit continues with plain EM, so the recorded
        objective never decreases.
        """
        use_rust_3pl = self._can_use_rust_3pl(model, responses)
        expand = not use_rust_3pl and self._uses_parameter_expansion(model)
        # Parameter values before the latest expansion step, until the next
        # E-step accepts it.
        expanded: dict[str, NDArray[np.float64]] | None = None
        prev_ll = -np.inf
        converged = False

        def evaluate() -> tuple[NDArray[np.float64], float]:
            nonlocal expand, expanded
            posterior, objective = self._evaluate(model, responses, frequencies)
            if expanded is not None:
                if not objective >= prev_ll:
                    model.set_parameters(**expanded)
                    posterior, objective = self._evaluate(model, responses, frequencies)
                    expand = False
                expanded = None
            return posterior, objective

        for iteration in range(self.max_iter):
            rust_result = None
            if use_rust_3pl:
                rust_result = self._run_rust_3pl_iteration(model, responses)
                if rust_result is None:
                    use_rust_3pl = False

            if rust_result is None:
                posterior_weights, current_ll = evaluate()
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

            change = abs(current_ll - prev_ll)
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
            if expand:
                if change < _PX_FINISH * self.tol:
                    expand = False
                else:
                    expanded = self._expand_latent_scale(model, n_k)
        else:
            posterior_weights, current_ll = evaluate()
            self._convergence_history.append(current_ll)
            converged = self._check_convergence(prev_ll, current_ll)

        return posterior_weights, current_ll, converged, iteration + 1

    def _uses_parameter_expansion(self, model: BaseItemModel) -> bool:
        """Return whether parameter-expansion steps keep this fit's model.

        The rescaled items must stay in their family and keep its
        constraints, and the latent density must remain the fixed standard
        normal. A common 1PL or PCM slope stays common, because every slope
        is scaled alike. Fixed slopes are allowed only at zero, where the
        diagonal rescaling keeps them. The grid must be rectangular with
        nodes at most ``_PX_MAX_SPACING`` apart.
        """
        from mirt.estimation.latent_density import GaussianDensity

        density = self._latent_density
        quadrature = self._quadrature
        layout = _latent_scale_layout(model)
        if (
            not self.px_em
            or layout is None
            or not isinstance(quadrature, RectangularQuadrature)
            or quadrature.limits[1] - quadrature.limits[0]
            > _PX_MAX_SPACING * (quadrature.n_points - 1)
            or self._prior_penalty is not None
            or self.constraints
            or model._free_parameter_restrictions
            or not uses_builtin_model_hooks(model, likelihood=True)
            or type(density) is not GaussianDensity
            or density.estimate_mean
            or density.estimate_cov
            or np.any(density.mean)
            or not np.array_equal(density.cov, np.eye(model.n_factors))
            or any(
                name in vars(self)
                or getattr(type(self), name) is not getattr(EMEstimator, name)
                for name in _PX_HOOKS
            )
        ):
            return False
        free = model.free_parameter_masks[layout[0]]
        return bool(np.any(free) and not np.any(model._parameters[layout[0]][~free]))

    def _expand_latent_scale(
        self, model: BaseItemModel, node_counts: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]] | None:
        """Take a PX-EM step: absorb the posterior latent scale into the items.

        ``node_counts`` holds the expected respondents at each quadrature
        node after the E-step. In the expanded model the latent population
        has a free mean and covariance, whose M-step values these counts
        give; rescaling the items to return the population to the standard
        normal (see :func:`_latent_scale_updates`) removes the slow
        latent-scale direction of plain EM (Liu, Rubin and Wu, 1998). On a
        discrete grid this is exact only approximately, so the caller checks
        the next objective. Returns the previous values of the changed
        parameters, or None when no step was taken.
        """
        layout = _latent_scale_layout(model)
        if layout is None:
            return None
        updates = _latent_scale_updates(
            model, layout, self._quadrature.nodes, node_counts
        )
        if updates is None:
            return None
        params = model.parameters
        previous = {name: params[name] for name in updates}
        try:
            model.set_parameters(**updates)
        except ValueError:
            model.set_parameters(**previous)
            return None
        return previous

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

        Where plain EM would use the fused native 3PL iteration, each E-step
        runs it: it returns the posterior and log-likelihood at the current
        parameters together with their M-step, which the next M-step then
        installs if the parameters have not moved in between.
        """
        from mirt.estimation._acceleration import (
            FreeItemParameters,
            squarem_point,
            squarem_step_length,
        )

        parameters = FreeItemParameters(model, self._tied)
        history = self._convergence_history
        n_e_steps = n_m_steps = 0
        native = self._can_use_rust_3pl(model, responses)
        # The parameters of the latest native E-step and their M-step update.
        pending: tuple[NDArray[np.float64], dict[str, NDArray[np.float64]]] | None
        pending = None

        def evaluate() -> tuple[NDArray[np.float64], float]:
            nonlocal n_e_steps, native, pending
            n_e_steps += 1
            pending = None
            if native:
                result = self._run_rust_3pl_iteration(model, responses)
                if result is not None:
                    discrimination, difficulty, guessing, posterior, objective = result
                    pending = (
                        parameters.get(model),
                        {
                            "discrimination": discrimination,
                            "difficulty": difficulty,
                            "guessing": guessing,
                        },
                    )
                    return posterior, objective
                native = False
            return self._evaluate(model, responses, frequencies)

        def update(posterior: NDArray[np.float64]) -> NDArray[np.float64]:
            nonlocal n_m_steps, pending
            n_m_steps += 1
            if pending is not None and np.array_equal(
                pending[0], parameters.get(model)
            ):
                model.set_parameters(**pending[1])
            else:
                self._m_step(
                    model, responses, _weight_posterior(posterior, frequencies)
                )
            pending = None
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

        # The GPU probe may import PyTorch, so it is checked last.
        if (
            not RUST_AVAILABLE
            or not should_use_rust(self.use_rust)
            or self._native_3pl_priors() is _UNSUPPORTED_PRIORS
            or self._tied is not None
            or type(model) is not ThreeParameterLogistic
            or not uses_builtin_model_hooks(model, likelihood=True)
            or model._free_parameter_restrictions
            or model.n_factors != 1
            or self.prob_epsilon != PROB_EPSILON
            or self._should_use_gpu
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

    def _native_3pl_priors(self) -> Any:
        """Return the native prior table of the item priors.

        ``None`` means no priors and ``_UNSUPPORTED_PRIORS`` priors that the
        native 3PL iteration cannot apply.
        """
        from mirt.backends.rust.estimation import native_3pl_prior_table

        penalty = self._prior_penalty
        if penalty is None:
            return None
        table = native_3pl_prior_table(penalty.priors)
        return _UNSUPPORTED_PRIORS if table is None else table

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
        """Run and validate one batched native 3PL E/M iteration.

        Returns the updated parameters, the posterior, and the EM objective
        at the current parameters: the marginal log-likelihood plus the item
        log-prior, whose modes the native M-steps then reach inside the boxes
        of :meth:`_coordinate_box`.
        """
        if self._quadrature is None:
            return None
        priors = self._native_3pl_priors()
        if priors is _UNSUPPORTED_PRIORS:
            return None

        params = model.parameters
        disc_bounds = self._coordinate_box(model, "discrimination")
        diff_bounds = self._coordinate_box(model, "difficulty")
        guess_bounds = self._coordinate_box(model, "guessing")
        from mirt.utils._parallel import resolve_n_jobs

        workers = resolve_n_jobs(self.n_jobs)
        context = self._fit_context
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
                n_jobs=workers,
                pool=None if context is None else context.native_pool(workers),
                priors=None if priors is None else self._prior_penalty.priors,
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
        posterior_shape = (responses.shape[0], self._quadrature.n_total_points)
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
            # The kernel's E-step ran at the parameters the model still holds.
            log_likelihood + self._log_item_prior(model),
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
            model.n_factors == 1
            and uses_builtin_model_hooks(model, likelihood=True)
            and self._should_use_gpu
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
        if self._m_step_common_slope(model, responses, posterior_weights, context):
            return

        if (
            model.is_polytomous
            and self._prior_penalty is None
            and should_use_rust(self.use_rust)
            # Nothing is left to the native step when every item is tied.
            and (tied is None or len(tied.items) < model.n_items)
        ):
            from mirt.backends.rust.polytomous_mstep import try_polytomous_m_step

            # The native step moves every item, so items linked by equality
            # constraints get their values back and are optimized jointly.
            held = {
                item: self._get_item_params_and_bounds(model, item)[0]
                for item in (() if tied is None else tied.items)
            }
            if try_polytomous_m_step(
                model,
                responses,
                posterior_weights,
                quad_points,
                max_iter=self.item_optim_maxiter,
                ftol=self._item_optim_ftol(_needs_precise_m_step(model)),
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
        items: Iterable[int] = range(model.n_items)
        if model.is_polytomous:
            category_counts = context.category_counts(
                model.n_categories, posterior_weights
            )
            if self._uses_newton_polytomous_m_step(model):
                items = self._newton_polytomous_m_step(
                    model, category_counts, skip=() if tied is None else tied.items
                )
            item_counts = {
                item_idx: {"r_kc": category_counts[item_idx]} for item_idx in items
            }
        else:
            correct, observed = context.expected_counts(posterior_weights)
            if self._uses_newton_logistic_m_step(model):
                items = self._newton_logistic_m_step(
                    model, correct, observed, skip=() if tied is None else tied.items
                )
            elif self._uses_newton_guessing_m_step(model):
                items = self._newton_guessing_m_step(
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

    def _m_step_common_slope(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
        context: EMFitContext,
    ) -> bool:
        """Update a family's common slope by conditional maximization.

        When the only tie is the common discrimination of a 1PL or PCM, the
        items are first updated with the slope held at its common value, by
        the batched Newton or native polytomous kernels, and the slope then
        maximizes the summed item objectives with the item parameters held.
        Each step raises the expected complete-data log-likelihood, so EM
        stays monotone (an ECM step; Meng and Rubin, 1993), at a fraction of
        the cost of one joint optimization over every item. Under item
        priors each step raises the log-posterior instead: the location
        priors enter the batched item steps (Fisher scoring for a PCM, whose
        native kernels cannot apply priors) and the slope prior, once per
        item that carries it, the slope step. Returns False, leaving the
        joint tied step to the caller, when the fast kernels do not apply.
        """
        from mirt.models.dichotomous import OneParameterLogistic
        from mirt.models.polytomous import PartialCreditModel

        tied = self._tied
        penalty = self._prior_penalty
        if (
            tied is None
            or len(tied.groups) != 1
            or tied.groups[0][0] != "discrimination"
            or tied.groups[0][1].size != model.n_items
            or type(model) not in (OneParameterLogistic, PartialCreditModel)
            or not model.estimate_discrimination
            or model._free_parameter_restrictions
        ):
            return False
        # A copy that holds the slope: the fast kernels then fit the item
        # locations at the current common value.
        held = copy.copy(model)
        held._parameters = {
            name: values.copy() for name, values in model.parameters.items()
        }
        held._estimate_discrimination = False
        quad_points = self._quadrature.nodes
        if model.is_polytomous and penalty is not None:
            if not self._uses_newton_polytomous_m_step(held):
                return False
            category_counts = context.category_counts(
                held.n_categories, posterior_weights
            )
            self._tied = None
            try:
                for item in self._newton_polytomous_m_step(held, category_counts):
                    self._optimize_item(
                        held,
                        int(item),
                        responses,
                        posterior_weights,
                        quad_points,
                        r_kc=category_counts[item],
                    )
            finally:
                self._tied = tied
            counts = (
                self._partial_credit_counts(model, responses, posterior_weights),
                posterior_weights,
            )
        elif model.is_polytomous:
            from mirt.backends.rust.polytomous_mstep import try_polytomous_m_step

            if not should_use_rust(self.use_rust) or not try_polytomous_m_step(
                held,
                responses,
                posterior_weights,
                quad_points,
                max_iter=self.item_optim_maxiter,
                ftol=self._item_optim_ftol(),
                epsilon=self.prob_epsilon,
                n_jobs=self.n_jobs,
                context=self._fit_context,
            ):
                return False
            counts = (
                self._partial_credit_counts(model, responses, posterior_weights),
                posterior_weights,
            )
        else:
            if not self._uses_newton_logistic_m_step(held):
                return False
            counts = correct, observed = context.expected_counts(posterior_weights)
            self._tied = None
            try:
                # The batched Newton step, with the bounded itemwise
                # optimizer for items it leaves.
                for item in self._newton_logistic_m_step(held, correct, observed):
                    self._optimize_item(
                        held,
                        int(item),
                        responses,
                        posterior_weights,
                        quad_points,
                        r_k=correct[item],
                        n_k_valid=observed[item],
                    )
            finally:
                self._tied = tied
        model.set_parameters(
            **{
                name: values
                for name, values in held._parameters.items()
                if name != "discrimination"
            }
        )
        slope_prior = None
        if penalty is not None and "discrimination" in penalty.priors:
            # The log-prior counts the common slope once per item.
            slope_prior = (
                penalty.priors["discrimination"],
                int(np.count_nonzero(model.free_parameter_masks["discrimination"])),
            )
        slope = _common_slope_newton(
            model,
            quad_points[:, 0],
            counts,
            self._coordinate_box(model, "discrimination"),
            prior=slope_prior,
        )
        if slope is not None:
            model.set_parameters(discrimination=np.full(model.n_items, slope))
        return True

    def _partial_credit_counts(
        self,
        model: BaseItemModel,
        responses: NDArray[np.int_],
        posterior_weights: NDArray[np.float64],
    ) -> _PartialCreditCounts:
        """Return the PCM slope step's response statistics, once per fit.

        The fit's posteriors weight each response row by its pattern
        frequency. Other responses get the row sums of
        ``posterior_weights`` as weights and are not cached.
        """
        width = int(np.max(model.n_categories))
        context = self._fit_context
        if context is None or context.responses is not responses:
            return _partial_credit_counts(
                responses, posterior_weights.sum(axis=1), width
            )
        cached = self._slope_counts
        if cached is None or cached[0] is not responses:
            weights = context.frequencies
            if weights is None:
                weights = np.ones(responses.shape[0])
            cached = (responses, _partial_credit_counts(responses, weights, width))
            self._slope_counts = cached
        return cached[1]

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

    def _batched_m_step_applies(
        self, model: BaseItemModel, names: tuple[str, ...]
    ) -> bool:
        """Return whether a batched M-step can stand in for the itemwise optimizer.

        ``model`` must use the built-in model hooks and store exactly the
        parameters ``names``. The probability clipping must be the default,
        item priors (if any) the plain coordinate priors of
        :class:`~mirt.estimation._item_priors.ItemPriorPenalty`, and
        estimator subclasses must not customize itemwise optimization.
        """
        penalty = self._prior_penalty
        return (
            uses_builtin_model_hooks(model)
            and tuple(model._parameters) == names
            and 0.0 < self.prob_epsilon <= PROB_EPSILON
            and (penalty is None or type(penalty) is ItemPriorPenalty)
            and not any(
                name in vars(self)
                or getattr(type(self), name) is not getattr(EMEstimator, name)
                for name in _ITEM_OPTIMIZATION_HOOKS
            )
        )

    def _uses_newton_logistic_m_step(self, model: BaseItemModel) -> bool:
        """Return whether every item is a built-in 1PL, 2PL or MIRT logistic item.

        Coordinates fixed by masks or a loading pattern stay in the batched
        step (see :meth:`_newton_logistic_m_step`), and item priors enter its
        Newton steps. The batched Newton solver maximizes the unclipped item
        objective, which the itemwise objective also evaluates beyond its
        probability bounds. Estimator subclasses that customize itemwise
        optimization keep it.
        """
        from mirt.models.dichotomous import OneParameterLogistic, TwoParameterLogistic
        from mirt.models.multidimensional import MultidimensionalModel

        names = (
            ("slopes", "intercepts")
            if type(model) is MultidimensionalModel
            else ("discrimination", "difficulty")
        )
        return type(model) in (
            OneParameterLogistic,
            TwoParameterLogistic,
            MultidimensionalModel,
        ) and self._batched_m_step_applies(model, names)

    def _newton_logistic_m_step(
        self,
        model: BaseItemModel,
        correct: NDArray[np.float64],
        observed: NDArray[np.float64],
        skip: Sequence[int] = (),
    ) -> NDArray[np.intp]:
        """Solve all logistic items jointly and return items needing the fallback.

        Each item is a logistic regression on its slopes and intercept, the
        slope-intercept form of a 1PL or 2PL and the stored form of a
        ``MultidimensionalModel``. Fixed slopes and intercepts, such as the
        zero loadings of a confirmatory model, keep their values. A fixed
        2PL difficulty with free slopes ties the intercept to the slopes, so
        such items, and estimates that do not converge or leave the
        optimizer box on a free coordinate, are left to the bounded itemwise
        optimizer, which then returns the constrained optimum. Items with no
        observed responses or no free coordinates, and the items in ``skip``
        (which another step updates), keep their values. Item priors on the
        slopes and on the difficulties or intercepts enter the Newton steps
        through the chain rule, so the batched step reaches the posterior
        mode of the itemwise optimizer.
        """
        from mirt.estimation._logistic_newton import newton_logistic_items
        from mirt.models.dichotomous import OneParameterLogistic
        from mirt.models.multidimensional import MultidimensionalModel

        penalty = self._prior_penalty
        priors = {} if penalty is None else penalty.priors
        params = model.parameters
        masks = model.free_parameter_masks
        n_items = model.n_items
        affine = type(model) is MultidimensionalModel
        slope_name, location_name = (
            ("slopes", "intercepts") if affine else ("discrimination", "difficulty")
        )
        estimate_slopes = type(model) is not OneParameterLogistic
        slopes = params[slope_name].reshape(n_items, -1)
        locations = params[location_name]
        intercepts = locations if affine else -slopes.sum(axis=1) * locations
        slope_free = np.asarray(masks[slope_name], dtype=np.bool_).reshape(n_items, -1)
        if not estimate_slopes:
            slope_free = np.zeros_like(slope_free)
        location_free = np.asarray(masks[location_name], dtype=np.bool_)
        coupled = np.zeros(n_items, dtype=np.bool_)
        if not affine:
            coupled = ~location_free & slope_free.any(axis=1)
        free = np.column_stack((slope_free, location_free & ~coupled))
        if estimate_slopes:
            free[coupled, :-1] = False
        else:
            free = free[:, -1:]
        new_slopes, new_intercepts, converged = newton_logistic_items(
            self._quadrature.nodes,
            correct,
            observed,
            slopes,
            intercepts,
            estimate_slopes=estimate_slopes,
            free=free,
            slope_prior=priors.get(slope_name),
            location_prior=priors.get(location_name),
            difficulty=not affine,
        )
        if affine:
            new_locations = new_intercepts
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                new_locations = -new_intercepts / new_slopes.sum(axis=1)

        def inside(values, name, free_mask):
            low, high = self._coordinate_box(model, name)
            return ~free_mask | ((values >= low) & (values <= high))

        accepted = converged & inside(new_locations, location_name, location_free)
        if estimate_slopes:
            accepted &= np.all(inside(new_slopes, slope_name, slope_free), axis=1)
        observed_items = np.any(observed, axis=1) & (free.any(axis=1) | coupled)
        observed_items[np.asarray(skip, dtype=np.intp)] = False
        accepted &= observed_items & ~coupled

        updates = {
            location_name: np.where(accepted & location_free, new_locations, locations)
        }
        if estimate_slopes:
            updates[slope_name] = np.where(
                accepted[:, None] & slope_free, new_slopes, slopes
            ).reshape(params[slope_name].shape)
        model.set_parameters(**updates)
        return np.flatnonzero(observed_items & ~accepted)

    def _coordinate_box(self, model: BaseItemModel, name: str) -> tuple[float, float]:
        """Return the itemwise optimizer box of ``name``, within its prior's support."""
        bound = _parameter_bounds(model, name)
        penalty = self._prior_penalty
        return bound if penalty is None else penalty.bounds(name, bound)

    def _uses_newton_guessing_m_step(self, model: BaseItemModel) -> bool:
        """Return whether 3PL items take the batched Fisher-scoring M-step.

        It serves built-in unidimensional 3PL items without fixed coordinates
        under item priors when the native 3PL iteration does not run, for
        example for priors other than normal, log-normal and beta ones or
        with the NumPy backend (see :meth:`_newton_guessing_m_step`).
        """
        from mirt.models.dichotomous import ThreeParameterLogistic

        return (
            self._prior_penalty is not None
            and type(model) is ThreeParameterLogistic
            and model.n_factors == 1
            and not model._free_parameter_restrictions
            and self._batched_m_step_applies(
                model, ("discrimination", "difficulty", "guessing")
            )
        )

    def _newton_guessing_m_step(
        self,
        model: BaseItemModel,
        correct: NDArray[np.float64],
        observed: NDArray[np.float64],
        skip: Sequence[int] = (),
    ) -> NDArray[np.intp]:
        """Fisher-score all 3PL items jointly and return items needing the fallback.

        The batched steps maximize each item's expected log-likelihood,
        clipped near the guessing asymptote, plus its log-prior, the itemwise
        objective, inside the itemwise box. Items whose steps do not converge
        inside that box, such as guessing parameters that a flat prior lets
        reach zero, are left to the bounded itemwise optimizer. Items with no
        observed responses, and the items in ``skip``, keep their values.
        """
        from mirt.estimation._dichotomous_objective import guessing_items_objective
        from mirt.estimation._logistic_newton import newton_items

        names = ("discrimination", "difficulty", "guessing")
        penalty = self._prior_penalty
        priors = {} if penalty is None else penalty.priors
        params = model.parameters
        candidates = np.any(observed, axis=1)
        candidates[np.asarray(skip, dtype=np.intp)] = False
        items = np.flatnonzero(candidates)
        if not items.size:
            return items
        lower, upper = np.asarray(
            [self._coordinate_box(model, name) for name in names]
        ).T
        objective = guessing_items_objective(
            self._quadrature.nodes[:, 0],
            correct[items],
            observed[items],
            self.prob_epsilon,
            [
                (slice(column, column + 1), priors[name])
                for column, name in enumerate(names)
                if name in priors
            ],
        )
        estimates, converged = newton_items(
            np.column_stack([params[name][items] for name in names]),
            objective,
            lower,
            upper,
        )
        updates = {name: params[name].copy() for name in names}
        for column, name in enumerate(names):
            updates[name][items[converged]] = estimates[converged, column]
        model.set_parameters(**updates)
        return items[~converged]

    def _uses_newton_polytomous_m_step(self, model: BaseItemModel) -> bool:
        """Return whether polytomous items take the batched Fisher-scoring M-step.

        It covers built-in nominal items of any dimension, which have no
        native M-step, and built-in unidimensional graded, generalized
        partial credit and partial credit items under item priors, which the
        native M-steps cannot apply. Items with fixed coordinates keep the
        itemwise optimizer (see :meth:`_newton_polytomous_m_step`).
        """
        from mirt.models.polytomous import (
            GeneralizedPartialCredit,
            GradedResponseModel,
            NominalResponseModel,
            PartialCreditModel,
        )

        family = type(model)
        if family is NominalResponseModel:
            names = ("slopes", "intercepts")
        elif (
            family
            in (GradedResponseModel, GeneralizedPartialCredit, PartialCreditModel)
            and self._prior_penalty is not None
            and model.n_factors == 1
        ):
            names = (
                "discrimination",
                "thresholds" if family is GradedResponseModel else "steps",
            )
        else:
            return False
        return not model._free_parameter_restrictions and self._batched_m_step_applies(
            model, names
        )

    def _newton_polytomous_m_step(
        self,
        model: BaseItemModel,
        counts: Sequence[NDArray[np.float64]],
        skip: Sequence[int] = (),
    ) -> NDArray[np.intp]:
        """Fisher-score polytomous items jointly and return items needing the fallback.

        Items with the same number of categories are solved together on
        their expected ``(n_points, n_categories)`` category ``counts``, in
        the itemwise free layout, objective and box, so the batched steps
        reach the itemwise optimum. For nominal items, whose logits are
        linear in the coordinates, the expected information is the Hessian,
        so their steps are Newton steps. Items whose steps do not converge
        inside the box, such as graded items whose thresholds an empty
        category pulls together, are left to the itemwise optimizer. Items
        with no expected responses, and the items in ``skip``, keep their
        values.
        """
        from mirt.estimation._logistic_newton import newton_items
        from mirt.estimation._polytomous_objective import polytomous_items_objective
        from mirt.models.polytomous import (
            GradedResponseModel,
            NominalResponseModel,
            PartialCreditModel,
        )

        penalty = self._prior_penalty
        priors = {} if penalty is None else penalty.priors
        params = model.parameters
        nominal = type(model) is NominalResponseModel
        if nominal:
            family, names = "nominal", ("slopes", "intercepts")
        elif type(model) is GradedResponseModel:
            family, names = "graded", ("discrimination", "thresholds")
        else:
            family, names = "partial_credit", ("discrimination", "steps")
        # A partial credit slope that the family fixes is no coordinate.
        fixed = type(model) is PartialCreditModel and not model.estimate_discrimination
        layout = names[1:] if fixed else names
        canonical = {
            name: model._canonical_parameter_values(name, params[name])
            for name in names
        }
        updates = {name: params[name].copy() for name in layout}
        n_categories = np.asarray(model.n_categories)
        candidates = np.array([np.any(item_counts) for item_counts in counts])
        candidates[np.asarray(skip, dtype=np.intp)] = False
        fallback = []
        for size in np.unique(n_categories[candidates]):
            items = np.flatnonzero(candidates & (n_categories == size))
            if nominal:
                columns = [
                    canonical["slopes"][items, 1:size].reshape(items.size, -1),
                    canonical["intercepts"][items, 1:size],
                ]
            else:
                columns = [canonical[names[1]][items, : size - 1]]
                if not fixed:
                    columns.insert(0, canonical["discrimination"][items, None])
            widths = [column.shape[1] for column in columns]
            ends = np.cumsum(widths)
            boxes = [self._coordinate_box(model, name) for name in layout]
            objective = polytomous_items_objective(
                family,
                self._quadrature.nodes,
                np.stack([counts[int(item)] for item in items]),
                self.prob_epsilon,
                fixed_slopes=params["discrimination"][items] if fixed else None,
                priors=[
                    (slice(end - width, end), priors[name])
                    for name, width, end in zip(layout, widths, ends, strict=True)
                    if name in priors
                ],
            )
            estimates, converged = newton_items(
                np.hstack(columns),
                objective,
                np.repeat([box[0] for box in boxes], widths),
                np.repeat([box[1] for box in boxes], widths),
            )
            fallback.append(items[~converged])
            solved, estimates = items[converged], estimates[converged]
            for name, width, end in zip(layout, widths, ends, strict=True):
                values = estimates[:, end - width : end]
                if name == "discrimination":
                    updates[name][solved] = values[:, 0]
                elif nominal:
                    # Canonical rows, whose reference category is zero.
                    rows = canonical[name][solved]
                    rows[:, 1:size] = values.reshape(rows[:, 1:size].shape)
                    updates[name][solved] = rows
                else:
                    updates[name][solved, : size - 1] = values
        model.set_parameters(**updates)
        return np.concatenate([np.zeros(0, dtype=np.intp), *fallback])

    def _neg_expected_loglik_with_grad_dichotomous(
        self,
        model: BaseItemModel,
        item_idx: int,
        quad_points: NDArray[np.float64],
        n_k_valid: NDArray[np.float64],
        r_k: NDArray[np.float64],
        params: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        """Evaluate the prepared item objective without assuming optimizer bounds."""
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
                    # Loosely solved 3PL/4PL, graded, generalized partial
                    # credit and nominal items jitter between M-steps, which
                    # stalls EM short of the optimum.
                    "ftol": self._item_optim_ftol(
                        precise=analytic
                        and (not model.is_polytomous or _needs_precise_m_step(model))
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
        """Choose the estimator behind ``se_method`` for this fit.

        ``"auto"`` selects the exact observed information (``"oakes"``), or
        with survey ``person_weights`` the design-based ``"sandwich"``, for
        every model whose information item-local derivatives give exactly
        (``supports_louis_information``). Other models, whose information
        would take O(P^2) marginal likelihood evaluations, and estimators
        that replace the likelihood, which the exact information recomputes
        from the model, get itemwise complete-data curvature.
        """
        from mirt.estimation._louis_information import supports_louis_information

        if self.se_method != "auto":
            return self.se_method
        own_likelihood = any(
            name in vars(self)
            or getattr(type(self), name) is not getattr(EMEstimator, name)
            for name in ("_e_step", "_compute_log_likelihoods")
        )
        if own_likelihood or not supports_louis_information(model):
            return "complete_data"
        return "oakes" if person_weights is None else "sandwich"

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
        equality constraints share the error of their group. Survey
        ``person_weights`` weight each person's information and score
        cross-product, except that the ``"sandwich"`` meat weights the score
        cross-products by their squares (Binder, 1983).
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
        meat_weights = None
        if person_weights is not None:
            weights = np.asarray(person_weights, dtype=np.float64)
            frequencies = weights if frequencies is None else frequencies * weights
            if method == "sandwich":
                meat_weights = weights * frequencies
        estimate = estimate_covariance(
            model,
            responses,
            self._quadrature,
            self._prior_mass(model, responses, posterior_weights),
            method,
            frequencies=frequencies,
            meat_weights=meat_weights,
            h=self.se_step_size,
            bounds=lambda name: _parameter_bounds(model, name),
            prior_information=self._prior_information(model),
            tied=self._tied,
            held=self._rotation_anchors(model),
        )
        self._se_details = (method, estimate.covariance)
        return estimate.standard_errors

    def _rotation_anchors(
        self, model: BaseItemModel
    ) -> dict[str, NDArray[np.bool_]] | None:
        """Return the slopes held to identify an exploratory rotation, or None.

        An exploratory model with ``F >= 2`` factors and a fixed Gaussian
        latent density keeps its likelihood under an orthogonal rotation of
        the factors, so its observed information is singular. As R mirt
        identifies the rotation by fixing the upper triangle of the first
        ``F - 1`` items' slopes at zero, those slopes are held at their
        estimates here: they get ``NaN`` standard errors, and the other
        errors are those of the solution with that rotation.
        """
        from mirt.estimation.latent_density import GaussianDensity

        density = self._latent_density
        name = model._rotatable_slopes
        if (
            name is None
            or not isinstance(density, GaussianDensity)
            or density.estimate_cov
            or model._rotational_indeterminacy(model.free_parameter_masks) == 0
            or (
                self._tied is not None
                and any(group[0] == name for group in self._tied.groups)
            )
        ):
            return None
        n_factors = model.n_factors
        free = np.asarray(model.free_parameter_masks[name], dtype=np.bool_)
        rows = free.reshape(-1, n_factors)
        held = np.zeros_like(rows)
        anchored = np.flatnonzero(rows.all(axis=1))[: n_factors - 1]
        for position, row in enumerate(anchored):
            held[row, position + 1 :] = True
        return {name: held.reshape(free.shape)}

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
            # Shared parameters get their errors below; other model-level
            # parameters are held fixed by the item M-steps.
            if (
                not np.any(free_mask)
                or name in model._shared_parameters
                or not model._item_indexed(name)
            ):
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
