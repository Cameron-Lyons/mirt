"""True and observed score equating methods.

This module provides IRT-based score equating including true score
equating and observed score equating via Lord-Wingersky recursion.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from mirt._model_defaults import uses_builtin_model_hooks
from mirt.backends.rust.equating import (
    observed_score_distribution_2pl as _rust_observed_score_distribution_2pl,
)
from mirt.equating.linking import LinkingResult

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


_PROBABILITY_TOLERANCE = 1e-10
_SMOOTHING_METHODS = ("none", "loglinear", "kernel")
_RECURSION_CHUNK_ELEMENTS = 262_144
# Offsets of max(1, half-width of theta_range) times 2**k searched beyond each
# end; the first few, below one half-width, keep near-tail brackets narrow.
_TCC_SEARCH_POWERS = np.arange(-3.0, 61.0)
_ROOT_MAX_ITERATIONS = 200
# Residual of an equated score's old-form root before first-order correction.
_ROOT_RELATIVE_TOLERANCE = 1e-10
# Interpolation nodes per root on the evenly spaced reporting grid.
_UNIFORM_NODES = 8
# Relative agreement of slope-ratio estimates that permits relaxed roots.
_RATIO_AGREEMENT = 1e-6


@dataclass
class ScoreEquatingResult:
    """Result of score equating procedure.

    Attributes
    ----------
    old_scores : NDArray[np.float64]
        Raw scores on old form.
    new_scores : NDArray[np.float64]
        Equivalent scores on new form.
    theta : NDArray[np.float64]
        Reporting grid for true-score equating, or the population grid for
        observed-score equating. Empty when no ability grid is used.
    standard_errors : NDArray[np.float64] | None
        Standard errors of equated scores.
    method : str
        Equating method used.
    """

    old_scores: NDArray[np.float64]
    new_scores: NDArray[np.float64]
    theta: NDArray[np.float64]
    standard_errors: NDArray[np.float64] | None
    method: str


def true_score_equating(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    linking_result: LinkingResult | None = None,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 201,
    items_old: list[int] | None = None,
    items_new: list[int] | None = None,
) -> ScoreEquatingResult:
    """Perform IRT true score equating.

    Maps raw scores between forms using expected score functions.
    A score on Form X is equivalent to a score on Form Y if they
    correspond to the same theta value.

    Parameters
    ----------
    model_old : BaseItemModel
        Reference form model.
    model_new : BaseItemModel
        New form model (on same scale or after linking).
    linking_result : LinkingResult | None
        Constants mapping new abilities onto the old/reference scale as
        ``theta_old = A * theta_new + B``. The new form is evaluated at
        ``(theta_old - B) / A``.
    theta_range : tuple[float, float]
        Range of the reporting theta grid returned as ``result.theta``. It
        also seeds the root search; abilities outside it are still found.
    n_theta : int
        Number of reporting theta points.
    items_old : list[int] | None
        Subset of items for old form. None = all items.
    items_new : list[int] | None
        Subset of items for new form. None = all items.

    Returns
    -------
    ScoreEquatingResult
        Score conversion table and diagnostics.

    Notes
    -----
    Each old-form score strictly between the limits of the old test
    characteristic curve (TCC) is solved for its ability by bracketed root
    finding, and the new TCC is evaluated there. Both curves are validated
    on a search grid that extends the reporting grid geometrically, and
    local inverse interpolation of that grid seeds the roots, so equated
    scores usually reach rounding accuracy with one further old-form curve
    evaluation per score. The limits are the sums of
    the lower and upper item asymptotes (for example the 3PL guessing
    parameters), evaluated numerically far outside ``theta_range``. Both
    models must therefore return valid probabilities at extreme abilities;
    overflow warnings from their curves there are suppressed.

    Scores outside those limits have no true-score ability. Following
    Kolen and Brennan (2014, sec. 6.5), scores at or below the old lower
    limit ``L_X`` map linearly from ``(0, 0)`` to ``(L_X, L_Y)``. Scores at
    or above the old upper limit ``U_X`` map linearly from ``(U_X, U_Y)`` to
    the maximum scores ``(K_X, K_Y)``. A zero score therefore maps to zero
    and a perfect score to a perfect score.
    """
    lower, upper = _validate_theta_range(theta_range)
    n_theta = _validate_count(n_theta, "n_theta", minimum=2)
    _validate_model(model_old, "model_old")
    _validate_model(model_new, "model_new")
    old_item_indices = _resolve_items(model_old, items_old, "items_old")
    new_item_indices = _resolve_items(model_new, items_new, "items_new")

    theta_grid = np.linspace(lower, upper, n_theta)
    # Reject invalid constants before any model curve is evaluated.
    _new_scale_theta(theta_grid, linking_result)
    search_old, search_new = _true_score_search_abilities(theta_grid, linking_result)
    expected_old = _extreme_expected_scores(model_old, search_old, old_item_indices)
    expected_new = _extreme_expected_scores(model_new, search_new, new_item_indices)

    _validate_expected_score_curve(expected_old, "model_old")
    _validate_expected_score_curve(expected_new, "model_new")
    expected_old = np.maximum.accumulate(expected_old)
    expected_new = np.maximum.accumulate(expected_new)

    max_score_old = _maximum_score(model_old, old_item_indices)
    max_score_new = _maximum_score(model_new, new_item_indices)
    old_scores = np.arange(max_score_old + 1, dtype=np.float64)
    new_scores = np.empty_like(old_scores)
    lower_old, upper_old = float(expected_old[0]), float(expected_old[-1])
    lower_new, upper_new = float(expected_new[0]), float(expected_new[-1])

    below = old_scores <= lower_old
    above = (old_scores >= upper_old) & ~below
    interior = ~(below | above)
    if lower_old > 0.0:
        new_scores[below] = old_scores[below] / lower_old * lower_new
    else:
        new_scores[below] = 0.0
    if max_score_old > upper_old:
        fraction = (max_score_old - old_scores[above]) / (max_score_old - upper_old)
        new_scores[above] = max_score_new - fraction * (max_score_new - upper_new)
    else:
        new_scores[above] = max_score_new

    if np.any(interior):
        new_scores[interior] = _interior_true_scores(
            model_old,
            model_new,
            old_item_indices,
            new_item_indices,
            old_scores[interior],
            theta_grid,
            search_old,
            expected_old,
            expected_new,
            linking_result,
        )

    return ScoreEquatingResult(
        old_scores=old_scores,
        new_scores=new_scores,
        theta=theta_grid,
        standard_errors=None,
        method="true_score",
    )


def observed_score_equating(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    theta_distribution: NDArray[np.float64] | None = None,
    theta_grid: NDArray[np.float64] | None = None,
    n_theta: int = 61,
    items_old: list[int] | None = None,
    items_new: list[int] | None = None,
    smoothing: Literal["none", "loglinear", "kernel"] = "none",
    linking_result: LinkingResult | None = None,
    *,
    batch_size: int | None = None,
) -> ScoreEquatingResult:
    """Perform IRT observed score equating.

    Uses Lord-Wingersky recursion to compute score distributions,
    then applies equipercentile equating.

    Parameters
    ----------
    model_old : BaseItemModel
        Reference form model.
    model_new : BaseItemModel
        New form model.
    theta_distribution : NDArray | None
        Probability masses at each point on the old/reference theta scale.
        Default: weights proportional to standard normal density.
    theta_grid : NDArray | None
        Grid of theta values for integration.
    n_theta : int
        Number of theta points if grid not provided.
    items_old : list[int] | None
        Subset of items for old form.
    items_new : list[int] | None
        Subset of items for new form.
    smoothing : {"none", "loglinear", "kernel"}
        Score-distribution smoothing applied before equipercentile inversion.
        ``"loglinear"`` is the moment-preserving maximum-likelihood
        log-linear presmoother of :func:`kernel_equating` (degree 4, lower
        for short distributions); ``"kernel"`` is Gaussian smoothing of the
        discrete distribution.
    linking_result : LinkingResult | None
        Constants mapping new abilities onto the old/reference scale as
        ``theta_old = A * theta_new + B``. The same reference population weights
        are used for both forms, evaluating the new form at ``(theta_old - B) / A``.
    batch_size : int | None
        Maximum theta points evaluated together. None chooses a bounded size
        from the form lengths and category counts.

    Returns
    -------
    ScoreEquatingResult
        Score conversion table.
    """
    if smoothing not in _SMOOTHING_METHODS:
        raise ValueError("smoothing must be one of 'none', 'loglinear', or 'kernel'")

    theta_grid, score_dist_old, score_dist_new = _population_score_distributions(
        model_old,
        model_new,
        theta_distribution,
        theta_grid,
        n_theta,
        items_old,
        items_new,
        linking_result,
        batch_size,
    )

    new_scores = equipercentile_equating(
        score_dist_old, score_dist_new, smoothing=smoothing
    )

    old_scores = np.arange(len(score_dist_old), dtype=np.float64)

    return ScoreEquatingResult(
        old_scores=old_scores,
        new_scores=new_scores,
        theta=theta_grid,
        standard_errors=None,
        method="observed_score",
    )


def _population_score_distributions(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    theta_distribution: NDArray[np.float64] | None,
    theta_grid: NDArray[np.float64] | None,
    n_theta: int,
    items_old: list[int] | None,
    items_new: list[int] | None,
    linking_result: LinkingResult | None,
    batch_size: int | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return both forms' score distributions in one reference population."""
    _validate_model(model_old, "model_old")
    _validate_model(model_new, "model_new")
    _resolve_items(model_old, items_old, "items_old")
    _resolve_items(model_new, items_new, "items_new")
    if theta_grid is None:
        n_theta = _validate_count(n_theta, "n_theta", minimum=1)
        theta_grid = np.linspace(-4.0, 4.0, n_theta)
    else:
        theta_grid = _validate_vector(theta_grid, "theta_grid")

    if theta_distribution is None:
        theta_distribution = np.exp(-0.5 * theta_grid**2)
    theta_distribution = _validate_weights(
        theta_distribution, len(theta_grid), "theta_distribution"
    )
    theta_new = _new_scale_theta(theta_grid, linking_result)

    score_dist_old = lord_wingersky_recursion(
        model_old, theta_grid, theta_distribution, items_old, batch_size=batch_size
    )
    score_dist_new = lord_wingersky_recursion(
        model_new, theta_new, theta_distribution, items_new, batch_size=batch_size
    )
    return theta_grid, score_dist_old, score_dist_new


def lord_wingersky_recursion(
    model: "BaseItemModel",
    theta_grid: NDArray[np.float64],
    theta_weights: NDArray[np.float64],
    items: list[int] | None = None,
    *,
    batch_size: int | None = None,
) -> NDArray[np.float64]:
    """Compute observed score distribution using Lord-Wingersky recursion.

    Recursively computes P(X=x) for each possible sum score x. Both
    dichotomous and ordered polytomous items are supported.

    Parameters
    ----------
    model : BaseItemModel
        IRT model with item parameters.
    theta_grid : NDArray
        Grid of theta values for integration.
    theta_weights : NDArray
        Weights for theta integration (e.g., prior distribution).
    items : list[int] | None
        Subset of items. None = all items.
    batch_size : int | None
        Maximum theta points evaluated together. None chooses a bounded size
        from the form length and category counts. Only the marginal score
        distribution is retained across batches.

    Returns
    -------
    NDArray
        Marginal score distribution over every attainable sum score.
    """
    _validate_model(model, "model")
    theta_grid = _validate_vector(theta_grid, "theta_grid")
    weights = _validate_weights(theta_weights, len(theta_grid), "theta_weights")
    item_indices = _resolve_items(model, items, "items")
    width = _maximum_score(model, item_indices) + 1
    if batch_size is None:
        categories = int(np.max(model.n_categories)) if model.is_polytomous else 2
        elements_per_point = 3 * width + model.n_items * categories
        batch_size = max(1, _RECURSION_CHUNK_ELEMENTS // elements_per_point)
    else:
        batch_size = _validate_count(batch_size, "batch_size", minimum=1)

    marginal = np.zeros(width, dtype=np.float64)
    for start in range(0, len(theta_grid), batch_size):
        stop = min(start + batch_size, len(theta_grid))
        theta_batch = theta_grid[start:stop]
        weights_batch = weights[start:stop]
        distribution = _native_score_distribution(
            model, theta_batch, weights_batch, item_indices
        )
        if distribution is None:
            distribution = _numpy_score_distribution(
                model, theta_batch, weights_batch, item_indices, width
            )
        marginal += distribution
    return _normalize_score_distribution(marginal)


def _numpy_score_distribution(
    model: "BaseItemModel",
    theta: NDArray[np.float64],
    weights: NDArray[np.float64],
    item_indices: NDArray[np.intp],
    width: int,
) -> NDArray[np.float64]:
    """Convolve item probabilities using reusable bounded work buffers."""
    item_probabilities = _item_score_probabilities(model, theta, item_indices)
    # Score-major storage keeps the growing active region contiguous.
    current = np.empty((width, len(theta)), dtype=np.float64)
    updated = np.empty_like(current)
    workspace = np.empty_like(current)
    current[0] = 1.0
    current_width = 1
    for probabilities in item_probabilities:
        n_categories = probabilities.shape[1]
        next_width = current_width + n_categories - 1
        updated[:next_width] = 0.0
        conditional = current[:current_width]
        work = workspace[:current_width]
        for score in range(n_categories):
            np.multiply(conditional, probabilities[:, score], out=work)
            target = updated[score : score + current_width]
            np.add(target, work, out=target)
        current, updated = updated, current
        current_width = next_width
    return current @ weights


def _interior_true_scores(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    old_item_indices: NDArray[np.intp],
    new_item_indices: NDArray[np.intp],
    targets: NDArray[np.float64],
    theta_grid: NDArray[np.float64],
    search_old: NDArray[np.float64],
    expected_old: NDArray[np.float64],
    expected_new: NDArray[np.float64],
    linking_result: LinkingResult | None,
) -> NDArray[np.float64]:
    """Equate old scores strictly inside the old curve's limits.

    Local inverse interpolation of the validated search grid seeds each
    ability: degree 7 on the evenly spaced reporting grid, where one curve
    evaluation usually confirms the root, and cubic in the log distance to
    the curve's limit on its geometric extension. Where the reporting grid
    resolves the local new-to-old slope ratio, a root may keep a residual of
    ``1e-10`` relative to its score, divided by that ratio when it exceeds
    one, and the new score is corrected to first order by the residual times
    the ratio, leaving an error at rounding level. Other roots are solved to
    rounding level directly.
    """
    bracket = np.searchsorted(expected_old, targets, side="left")
    first = int(np.searchsorted(search_old, theta_grid[0]))
    last = first + len(theta_grid) - 1
    uniform = (bracket - 1 >= first) & (bracket <= last)
    uniform &= len(theta_grid) >= _UNIFORM_NODES
    # Below and above the reporting grid the curve approaches its limits
    # roughly exponentially, so ability is close to linear in the log gap.
    lower_limit, upper_limit = expected_old[0], expected_old[-1]
    lower_tail = ~uniform & (bracket <= first)
    upper_tail = ~uniform & (bracket - 1 >= last)
    middle = ~(uniform | lower_tail | upper_tail)
    inner = (1, len(search_old) - 2)
    groups = (
        (uniform, _local_nodes(bracket, first, last, _UNIFORM_NODES), None),
        (lower_tail, _local_nodes(bracket, *inner, 4), "lower"),
        (upper_tail, _local_nodes(bracket, *inner, 4), "upper"),
        (middle, _local_nodes(bracket, 0, len(search_old) - 1, 4), None),
    )

    def gap(values: NDArray[np.float64], tail: str | None) -> NDArray[np.float64]:
        if tail == "lower":
            return np.log(values - lower_limit)
        if tail == "upper":
            return -np.log(upper_limit - values)
        return values

    def inverse(
        values: NDArray[np.float64], rows: NDArray[np.intp]
    ) -> NDArray[np.float64]:
        result = np.empty(len(rows), dtype=np.float64)
        with np.errstate(all="ignore"):
            for members, nodes, tail in groups:
                selected = members[rows]
                if not np.any(selected):
                    continue
                row_nodes = nodes[rows[selected]]
                weights = _lagrange_weights(
                    gap(expected_old[row_nodes], tail), gap(values[selected], tail)
                )
                result[selected] = np.sum(weights * search_old[row_nodes], axis=1)
        return result

    # Relaxed roots are corrected by the local new-to-old slope ratio, so
    # they are allowed only where that ratio is well resolved: its degree-7
    # and degree-5 estimates at the bracket agree to 1e-6.
    fine_nodes = groups[0][1]
    candidates = np.flatnonzero(uniform)
    midpoints = 0.5 * (
        search_old[bracket[candidates] - 1] + search_old[bracket[candidates]]
    )
    coarse_nodes = _local_nodes(bracket[candidates], first, last, _UNIFORM_NODES - 2)
    with np.errstate(all="ignore"):
        fine = _slope_ratio(
            search_old, expected_old, expected_new, fine_nodes[candidates], midpoints
        )
        coarse = _slope_ratio(
            search_old, expected_old, expected_new, coarse_nodes, midpoints
        )
        resolved = np.abs(fine - coarse) <= _RATIO_AGREEMENT * np.abs(fine)
    relaxed = np.zeros(len(targets), dtype=bool)
    relaxed[candidates[resolved]] = True
    scale = np.maximum(np.abs(targets), 1.0)
    tolerance = 4.0 * np.finfo(np.float64).eps * scale
    tolerance[relaxed] = np.maximum(
        tolerance[relaxed],
        _ROOT_RELATIVE_TOLERANCE
        * scale[relaxed]
        / np.maximum(np.abs(fine[resolved]), 1.0),
    )
    roots, residuals = _solve_bracketed(
        _expected_score_evaluator(model_old, old_item_indices),
        targets,
        search_old[bracket - 1],
        search_old[bracket],
        expected_old[bracket - 1],
        expected_old[bracket],
        inverse=inverse,
        value_tolerance=tolerance,
    )
    scores = _expected_score_evaluator(model_new, new_item_indices)(
        _new_scale_theta(roots, linking_result)
    )
    with np.errstate(all="ignore"):
        ratio = _slope_ratio(
            search_old,
            expected_old,
            expected_new,
            fine_nodes[relaxed],
            roots[relaxed],
        )
    finite = np.isfinite(ratio)
    corrected = np.flatnonzero(relaxed)[finite]
    scores[corrected] -= residuals[corrected] * ratio[finite]
    return scores


def _slope_ratio(
    grid: NDArray[np.float64],
    old_curve: NDArray[np.float64],
    new_curve: NDArray[np.float64],
    nodes: NDArray[np.intp],
    points: NDArray[np.float64],
) -> NDArray[np.float64]:
    """New-to-old curve slope ratio from local interpolants at ``points``."""
    weights = _lagrange_weights(grid[nodes], points, derivative=True)
    return np.asarray(
        np.sum(weights * new_curve[nodes], axis=1)
        / np.sum(weights * old_curve[nodes], axis=1),
        dtype=np.float64,
    )


def _new_scale_theta(
    theta: NDArray[np.float64], linking_result: LinkingResult | None
) -> NDArray[np.float64]:
    """Evaluate the same abilities on the new calibration's scale."""
    transformed = _unchecked_new_scale_theta(theta, linking_result)
    if not np.all(np.isfinite(transformed)):
        raise ValueError(
            "linking_result produces non-finite theta values on the new scale"
        )
    return transformed


def _unchecked_new_scale_theta(
    theta: NDArray[np.float64], linking_result: LinkingResult | None
) -> NDArray[np.float64]:
    """Transform abilities, leaving overflowed values non-finite."""
    if linking_result is None:
        return theta
    A = float(linking_result.constants.A)
    B = float(linking_result.constants.B)
    if not np.isfinite(A) or A <= 0.0:
        raise ValueError("linking_result.constants.A must be finite and positive")
    if not np.isfinite(B):
        raise ValueError("linking_result.constants.B must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = (theta - B) / A
        invalid = ~np.isfinite(transformed)
        if A >= 1.0 and np.any(invalid):
            # Scale before subtracting only when the ordinary subtraction
            # overflowed. This retains its cancellation precision elsewhere.
            transformed[invalid] = theta[invalid] / A - B / A
    return np.asarray(transformed, dtype=np.float64)


def _true_score_search_abilities(
    theta_grid: NDArray[np.float64], linking_result: LinkingResult | None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extend the reporting grid geometrically to bracket every true score.

    The outermost points approximate the limits of the expected-score curve.
    Points that overflow on either calibration's scale are dropped.
    """
    lower, upper = float(theta_grid[0]), float(theta_grid[-1])
    half_width = max(0.5 * upper - 0.5 * lower, 1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        offsets = half_width * np.exp2(_TCC_SEARCH_POWERS)
        theta_old = np.concatenate((lower - offsets[::-1], theta_grid, upper + offsets))
    theta_new = _unchecked_new_scale_theta(theta_old, linking_result)
    keep = np.isfinite(theta_old) & np.isfinite(theta_new)
    return theta_old[keep], theta_new[keep]


def _extreme_expected_scores(
    model: "BaseItemModel",
    theta: NDArray[np.float64],
    item_indices: NDArray[np.intp],
) -> NDArray[np.float64]:
    """Compute expected scores at abilities that may lie far outside the grid.

    Saturating terms such as ``exp(-a * theta)`` in model curves overflow
    harmlessly there, so those warnings are silenced. Invalid probabilities
    are still rejected.
    """
    with np.errstate(over="ignore"):
        return _compute_expected_scores(model, theta, item_indices)


def _expected_score_evaluator(
    model: "BaseItemModel", item_indices: NDArray[np.intp]
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Return a lean expected-score function for repeated root-finding calls.

    Like :func:`_compute_expected_scores`, it reads only the selected items
    and the observed categories, so other items' curves cannot affect the
    scores, and it rejects probabilities outside [0, 1]. The selection and
    category scores are reused across calls, and the shape, clipping, and
    category-sum checks are left to the caller's validation of the curves.
    """
    # A full selection in model order needs no gather.
    selection = (
        None if np.array_equal(item_indices, np.arange(model.n_items)) else item_indices
    )
    if model.is_polytomous:
        counts = np.asarray(getattr(model, "n_categories"), dtype=np.intp)
        counts = counts[item_indices]
        width = int(np.max(counts))
        observed = np.arange(width) < counts[:, None]
        padded = not np.all(observed)
        scores = np.where(observed, np.arange(width, dtype=np.float64), 0.0).ravel()
    else:
        scores = np.ones(len(item_indices), dtype=np.float64)

    def evaluate(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        with np.errstate(over="ignore"):
            probabilities = np.asarray(
                model.probability(theta[:, None]), dtype=np.float64
            )
        if selection is not None:
            probabilities = probabilities[:, selection]
        if model.is_polytomous:
            if probabilities.shape[2] != width:
                probabilities = probabilities[:, :, :width]
            if padded:
                probabilities = np.where(observed, probabilities, 0.0)
            probabilities = probabilities.reshape(len(theta), -1)
        _check_probability_range(probabilities)
        return np.asarray(probabilities @ scores, dtype=np.float64)

    return evaluate


def _lagrange_weights(
    nodes: NDArray[np.float64], points: NDArray[np.float64], derivative: bool = False
) -> NDArray[np.float64]:
    """Row-wise Lagrange basis values, or their derivatives, at ``points``.

    The basis polynomial of node ``j`` is
    ``L_j(x) = prod_{k != j} (x - x_k) / (x_j - x_k)``, with derivative
    ``L_j(x) * sum_{k != j} 1 / (x - x_k)`` away from the nodes. At a node
    the derivative sums the products that also leave out one ``m != j``.
    Coincident nodes give non-finite weights, which callers reject.
    """
    diagonal = np.eye(nodes.shape[1], dtype=bool)
    offsets = points[:, None] - nodes
    with np.errstate(all="ignore"):
        gaps = nodes[:, :, None] - nodes[:, None, :]
        denominators = np.prod(np.where(diagonal, 1.0, gaps), axis=2)
        numerators = np.prod(np.where(diagonal, 1.0, offsets[:, None, :]), axis=2)
        values = numerators / denominators
        if not derivative:
            return np.asarray(values, dtype=np.float64)
        reciprocal = 1.0 / offsets
        slopes = values * (np.sum(reciprocal, axis=1, keepdims=True) - reciprocal)
        at_node = np.flatnonzero(np.any(offsets == 0.0, axis=1))
        if at_node.size:
            # Axes (j, m, k): leave out k == j and k == m, then sum over m != j.
            left_out = diagonal[:, None, :] | diagonal[None, :, :]
            products = np.prod(
                np.where(left_out, 1.0, offsets[at_node, None, None, :]), axis=3
            )
            slopes[at_node] = (
                np.sum(np.where(diagonal, 0.0, products), axis=2)
                / denominators[at_node]
            )
        return np.asarray(slopes, dtype=np.float64)


def _local_nodes(
    bracket: NDArray[np.intp], first: int, last: int, count: int
) -> NDArray[np.intp]:
    """``count`` indices in ``[first, last]`` centered on each bracket."""
    width = min(last - first + 1, count)
    start = np.clip(bracket - width // 2, first, last - width + 1)
    return start[:, None] + np.arange(width)


def _bracketed_root(
    function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    targets: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    value_lower: NDArray[np.float64],
    value_upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve ``function(x) = target`` for a vectorized increasing function.

    Each bracket must satisfy ``value_lower < target <= value_upper``. Illinois
    false-position steps converge superlinearly. A bracket that has not halved
    within three steps is bisected, which bounds the worst case.
    """
    roots, _ = _solve_bracketed(
        function, targets, lower, upper, value_lower, value_upper
    )
    return roots


def _solve_bracketed(
    function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    targets: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    value_lower: NDArray[np.float64],
    value_upper: NDArray[np.float64],
    *,
    inverse: Callable[[NDArray[np.float64], NDArray[np.intp]], NDArray[np.float64]]
    | None = None,
    value_tolerance: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return bracketed roots and ``function(root) - target`` at them.

    Without ``inverse`` this is :func:`_bracketed_root`. With an approximate
    inverse ``inverse(values, rows)`` for the targets ``rows``, the first
    step evaluates its estimate of each root, the second removes the
    inverse's error at that point, and later steps are secants through the
    last two points. Steps that leave the bracket fall back to false
    position; every third step bisects when the residual (with an inverse)
    or the bracket (without) has not halved. ``value_tolerance`` defaults to
    four rounding units of each target.
    """
    lower = np.array(lower, dtype=np.float64)
    upper = np.array(upper, dtype=np.float64)
    excess_lower = np.asarray(value_lower, dtype=np.float64) - targets
    excess_upper = np.asarray(value_upper, dtype=np.float64) - targets
    roots = upper.copy()
    excess_at_roots = excess_upper.copy()
    earlier_roots = np.full(len(targets), np.nan)
    earlier_excess = np.full(len(targets), np.nan)
    replaced = np.zeros(len(targets), dtype=np.int8)
    # One-sided secant steps shrink the residual while the bracket may not.
    reference = np.full(len(targets), np.inf) if inverse is not None else upper - lower
    if value_tolerance is None:
        value_tolerance = (
            4.0 * np.finfo(np.float64).eps * np.maximum(np.abs(targets), 1.0)
        )
    active = np.flatnonzero(excess_upper > value_tolerance)
    for iteration in range(_ROOT_MAX_ITERATIONS):
        if active.size == 0:
            break
        a, b = lower[active], upper[active]
        fa, fb = excess_lower[active], excess_upper[active]
        with np.errstate(all="ignore"):
            false_position = a + fa / (fa - fb) * (b - a)
            if inverse is None:
                candidate = false_position
            elif iteration == 0:
                candidate = inverse(targets[active], active)
            elif iteration == 1:
                # Newton-like step with the inverse's slope near the root.
                candidate = roots[active] - (
                    inverse(targets[active] + excess_at_roots[active], active)
                    - inverse(targets[active], active)
                )
            else:
                candidate = roots[active] - excess_at_roots[active] * (
                    roots[active] - earlier_roots[active]
                ) / (excess_at_roots[active] - earlier_excess[active])
        candidate = np.where(
            (candidate > a) & (candidate < b), candidate, false_position
        )
        unusable = ~((candidate > a) & (candidate < b))
        if iteration % 3 == 2:
            progress = np.abs(excess_at_roots[active]) if inverse is not None else b - a
            unusable |= progress > 0.5 * reference[active]
            reference[active] = progress
        candidate[unusable] = 0.5 * a[unusable] + 0.5 * b[unusable]
        excess = function(candidate) - targets[active]

        rises = excess >= 0.0
        upper[active[rises]] = candidate[rises]
        excess_upper[active[rises]] = excess[rises]
        lower[active[~rises]] = candidate[~rises]
        excess_lower[active[~rises]] = excess[~rises]
        # Illinois: halve the stale endpoint's value when it is retained twice.
        side = np.where(rises, np.int8(1), np.int8(-1))
        stale_lower = rises & (replaced[active] == 1)
        stale_upper = ~rises & (replaced[active] == -1)
        excess_lower[active[stale_lower]] *= 0.5
        excess_upper[active[stale_upper]] *= 0.5
        replaced[active] = side
        earlier_roots[active] = roots[active]
        earlier_excess[active] = excess_at_roots[active]
        roots[active] = candidate
        excess_at_roots[active] = excess

        width = upper[active] - lower[active]
        scale = np.maximum(
            np.maximum(np.abs(lower[active]), np.abs(upper[active])), 1.0
        )
        converged = (np.abs(excess) <= value_tolerance[active]) | (
            width <= 4.0 * np.finfo(np.float64).eps * scale
        )
        converged |= candidate == a
        converged |= candidate == b
        active = active[~converged]
    return roots, excess_at_roots


def _normalize_score_distribution(
    marginal: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Clip numerical noise and normalize a marginal score distribution."""
    marginal = np.clip(marginal, 0.0, None)
    total = float(np.sum(marginal))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("score distribution has zero or non-finite probability mass")
    return np.asarray(marginal / total, dtype=np.float64)


def _native_score_distribution(
    model: "BaseItemModel",
    theta_grid: NDArray[np.float64],
    weights: NDArray[np.float64],
    item_indices: NDArray[np.intp],
) -> NDArray[np.float64] | None:
    """Use the compiled 1PL/2PL recursion when the model is compatible."""
    from mirt.models.dichotomous import OneParameterLogistic, TwoParameterLogistic

    if type(model) not in (
        OneParameterLogistic,
        TwoParameterLogistic,
    ) or not uses_builtin_model_hooks(model):
        return None

    parameters = model.parameters
    discrimination = np.asarray(parameters.get("discrimination"))
    difficulty = np.asarray(parameters.get("difficulty"))
    if discrimination.ndim != 1 or difficulty.shape != discrimination.shape:
        return None

    conditional = _rust_observed_score_distribution_2pl(
        theta_grid,
        discrimination[item_indices],
        difficulty[item_indices],
    )
    if conditional is None:
        return None
    expected_shape = (len(theta_grid), len(item_indices) + 1)
    if conditional.shape != expected_shape:
        raise RuntimeError(
            f"native score distribution has shape {conditional.shape}, "
            f"expected {expected_shape}"
        )
    return weights @ conditional


def equipercentile_equating(
    score_dist_old: NDArray[np.float64],
    score_dist_new: NDArray[np.float64],
    smoothing: Literal["none", "loglinear", "kernel"] = "none",
) -> NDArray[np.float64]:
    """Perform equipercentile equating between score distributions.

    Finds the score on the new form with the same percentile rank as each
    score on the old form, using the definitions of Kolen and Brennan (2014,
    eqs. 2.14-2.18).

    Parameters
    ----------
    score_dist_old : NDArray
        Score distribution for old form P(X=x).
    score_dist_new : NDArray
        Score distribution for new form P(Y=y).
    smoothing : str
        Smoothing method: "none", "loglinear", or "kernel". ``"loglinear"``
        fits the maximum-likelihood polynomial log-linear model of degree 4,
        which preserves the first four moments. Distributions with fewer than
        six scores, or fewer than five with positive probability, use the
        highest degree they support.

    Returns
    -------
    NDArray
        Equivalent scores on new form for each old score, in
        ``[-0.5, K_Y + 0.5]`` where ``K_Y`` is the new maximum score.

    Notes
    -----
    Each integer score ``x`` is treated as uniformly spread over
    ``[x - 0.5, x + 0.5]``. The old percentile rank is
    ``P(x) = F(x - 1) + f(x) / 2``. Its equivalent is
    ``y_u - 0.5 + (P(x) - G(y_u - 1)) / g(y_u)``, where ``y_u`` is the
    smallest new score with ``G(y_u) > P(x)``. New scores with zero
    probability are therefore skipped. When ``P(x) = 1``, which happens only
    for old scores above the highest old score with positive probability,
    the equivalent is ``K_Y + 0.5``.
    """
    score_dist_old = _validate_distribution(score_dist_old, "score_dist_old")
    score_dist_new = _validate_distribution(score_dist_new, "score_dist_new")

    if smoothing == "loglinear":
        score_dist_old = _loglinear_smooth(score_dist_old)
        score_dist_new = _loglinear_smooth(score_dist_new)
    elif smoothing == "kernel":
        score_dist_old = _kernel_smooth(score_dist_old)
        score_dist_new = _kernel_smooth(score_dist_new)
    elif smoothing != "none":
        raise ValueError("smoothing must be one of 'none', 'loglinear', or 'kernel'")

    return _percentile_rank_equivalents(score_dist_old, score_dist_new)


def _percentile_rank_equivalents(
    score_dist_old: NDArray[np.float64], score_dist_new: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Invert uniformly continuized distributions (Kolen and Brennan, 2014).

    Ranks above the median are matched through upper-tail sums so that
    small tail probabilities keep their relative precision.
    """
    n_new = len(score_dist_new)
    half = 0.5 * score_dist_old
    below_old = _mass_below(score_dist_old)
    above_old = _mass_below(score_dist_old[::-1])[::-1]
    below_new = _mass_below(score_dist_new)
    above_new = _mass_below(score_dist_new[::-1])[::-1]
    lower_rank = below_old + half
    upper_rank = above_old + half
    in_lower_tail = lower_rank <= upper_rank

    # Smallest y with G(y) > P(x), or with S(y + 1) < 1 - P(x) from above.
    cell_lower = np.searchsorted(below_new + score_dist_new, lower_rank, side="right")
    cell_upper = n_new - np.searchsorted(above_new[::-1], upper_rank, side="left")
    cell = np.where(in_lower_tail, cell_lower, cell_upper)
    # Only zero-mass top scores of the old form have no new cell above them.
    equated = np.full(len(score_dist_old), n_new - 0.5)
    found = cell < n_new
    y = cell[found]
    density = score_dist_new[y]
    # Cumulative sums are differenced before the half cell is added, so
    # identical distributions reproduce every score exactly.
    offset_lower = (below_old[found] - below_new[y]) + half[found]
    offset_upper = (above_old[found] - above_new[y]) + half[found]
    equated[found] = np.where(
        in_lower_tail[found],
        y - 0.5 + offset_lower / density,
        y + 0.5 - offset_upper / density,
    )
    return equated


def _mass_below(distribution: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the probability strictly below each score."""
    return np.concatenate(([0.0], np.cumsum(distribution)[:-1]))


def _validate_count(value: int, name: str, minimum: int) -> int:
    """Validate an integer configuration value."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validate_vector(values: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Return a non-empty, finite one-dimensional float array."""
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_theta_range(theta_range: tuple[float, float]) -> tuple[float, float]:
    """Validate and normalize a theta range."""
    values = np.asarray(theta_range, dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("theta_range must contain two finite values")
    lower, upper = float(values[0]), float(values[1])
    if lower >= upper:
        raise ValueError("theta_range lower bound must be less than upper bound")
    return lower, upper


def _validate_model(model: "BaseItemModel", name: str) -> None:
    """Reject models whose score scale cannot be inverted unambiguously."""
    if model.n_factors != 1:
        raise ValueError(f"{name} must be unidimensional")
    if model.n_items < 1:
        raise ValueError(f"{name} must contain at least one item")


def _resolve_items(
    model: "BaseItemModel",
    items: list[int] | NDArray[np.intp] | None,
    name: str,
) -> NDArray[np.intp]:
    """Validate item indices and return them in caller-specified order."""
    if items is None:
        return np.arange(model.n_items, dtype=np.intp)

    raw = np.asarray(items)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer item indices")

    indices = np.asarray(raw, dtype=np.intp)
    if np.any(indices < 0) or np.any(indices >= model.n_items):
        raise ValueError(f"{name} contains an item index outside the model")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{name} must not contain duplicate item indices")
    return indices


def _validate_weights(
    weights: NDArray[np.float64], expected_size: int, name: str
) -> NDArray[np.float64]:
    """Validate integration weights and normalize their probability mass."""
    result = _validate_vector(weights, name)
    if len(result) != expected_size:
        raise ValueError(f"{name} must have the same length as theta_grid")
    if np.any(result < 0.0):
        raise ValueError(f"{name} must be non-negative")
    total = float(np.sum(result))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have positive finite mass")
    return np.asarray(result / total, dtype=np.float64)


def _validate_distribution(
    distribution: NDArray[np.float64], name: str
) -> NDArray[np.float64]:
    """Validate and normalize a discrete score distribution."""
    result = _validate_vector(distribution, name)
    if np.any(result < 0.0):
        raise ValueError(f"{name} must be non-negative")
    total = float(np.sum(result))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have positive finite mass")
    return np.asarray(result / total, dtype=np.float64)


def _item_score_probabilities(
    model: "BaseItemModel",
    theta: NDArray[np.float64],
    item_indices: NDArray[np.intp],
) -> list[NDArray[np.float64]]:
    """Return category probabilities for each selected item and theta."""
    if not model.is_polytomous:
        selected = _correct_probabilities(model, theta, item_indices)
        return [np.column_stack((1.0 - correct, correct)) for correct in selected.T]
    probabilities, counts = _polytomous_probabilities(model, theta, item_indices)
    return [probabilities[:, item, :count] for item, count in enumerate(counts)]


def _correct_probabilities(
    model: "BaseItemModel",
    theta: NDArray[np.float64],
    item_indices: NDArray[np.intp],
) -> NDArray[np.float64]:
    """Return validated theta-by-item correct-response probabilities."""
    raw_probabilities = np.asarray(model.probability(theta[:, None]), dtype=np.float64)
    if raw_probabilities.shape != (len(theta), model.n_items):
        raise ValueError("model returned invalid dichotomous probabilities")
    selected = raw_probabilities[:, item_indices]
    _check_probability_range(selected)
    return np.clip(selected, 0.0, 1.0)


def _polytomous_probabilities(
    model: "BaseItemModel",
    theta: NDArray[np.float64],
    item_indices: NDArray[np.intp],
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Return validated theta-by-item-by-category probabilities.

    Categories beyond an item's count are zero. The category counts of the
    selected items are returned alongside.
    """
    raw_probabilities = np.asarray(model.probability(theta[:, None]), dtype=np.float64)
    category_counts = np.asarray(getattr(model, "n_categories"), dtype=np.intp)
    if (
        raw_probabilities.ndim != 3
        or raw_probabilities.shape[:2] != (len(theta), model.n_items)
        or category_counts.shape != (model.n_items,)
    ):
        raise ValueError("model returned invalid polytomous probabilities")
    counts = category_counts[item_indices]
    if np.any(counts < 2) or np.any(counts > raw_probabilities.shape[2]):
        raise ValueError("model returned an invalid category probability matrix")
    width = int(np.max(counts))
    observed = np.arange(width) < counts[:, None]
    probabilities = np.where(observed, raw_probabilities[:, item_indices, :width], 0.0)
    _check_probability_range(probabilities)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    totals = np.sum(probabilities, axis=2, keepdims=True)
    if np.any(np.abs(totals - 1.0) > 1e-7):
        raise ValueError("model category probabilities must sum to one")
    return np.asarray(probabilities / totals, dtype=np.float64), counts


def _check_probability_range(probabilities: NDArray[np.float64]) -> None:
    """Reject non-finite values or values outside [0, 1] beyond rounding."""
    if probabilities.size == 0:
        return
    # NaN propagates through min and max, and infinities fall outside.
    if not (
        np.min(probabilities) >= -_PROBABILITY_TOLERANCE
        and np.max(probabilities) <= 1.0 + _PROBABILITY_TOLERANCE
    ):
        raise ValueError("model returned probabilities outside [0, 1]")


def _maximum_score(model: "BaseItemModel", item_indices: NDArray[np.intp]) -> int:
    """Return the largest attainable sum score for selected items."""
    if not model.is_polytomous:
        return len(item_indices)
    category_counts = np.asarray(getattr(model, "n_categories"), dtype=np.intp)
    return int(np.sum(category_counts[item_indices] - 1))


def _conditional_score_variance(
    model: "BaseItemModel",
    theta: NDArray[np.float64],
    item_indices: NDArray[np.intp],
) -> NDArray[np.float64]:
    """Compute conditional raw-score variance under local independence."""
    variance = np.zeros(len(theta), dtype=np.float64)
    for probabilities in _item_score_probabilities(model, theta, item_indices):
        scores = np.arange(probabilities.shape[1], dtype=np.float64)
        first_moment = probabilities @ scores
        second_moment = probabilities @ (scores**2)
        variance += np.maximum(second_moment - first_moment**2, 0.0)
    return variance


def _validate_expected_score_curve(
    expected: NDArray[np.float64], model_name: str
) -> None:
    """Require a finite, non-decreasing, identifiable score curve."""
    if not np.all(np.isfinite(expected)):
        raise ValueError(f"{model_name} expected score curve must be finite")
    if np.any(np.diff(expected) < -_PROBABILITY_TOLERANCE):
        raise ValueError(f"{model_name} expected score curve must be non-decreasing")
    if float(expected[-1] - expected[0]) <= _PROBABILITY_TOLERANCE:
        raise ValueError(f"{model_name} expected score curve must vary across theta")


def _invert_expected_scores(
    expected: NDArray[np.float64],
    theta: NDArray[np.float64],
    scores: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Invert a validated expected-score curve while preserving input shape."""
    unique_expected, unique_indices = np.unique(expected, return_index=True)
    score_array = np.asarray(scores, dtype=np.float64)
    inverted = np.interp(
        score_array.reshape(-1),
        unique_expected,
        theta[unique_indices],
        left=float(theta[0]),
        right=float(theta[-1]),
    )
    return np.asarray(inverted.reshape(score_array.shape), dtype=np.float64)


def _loglinear_smooth(
    dist: NDArray[np.float64], degree: int = 4
) -> NDArray[np.float64]:
    """Fit the maximum-likelihood polynomial log-linear model to a distribution.

    This is the presmoother of kernel equating (Holland and Thayer, 2000):
    the fit preserves the first ``degree`` moments. The degree is lowered
    below the number of scores with positive probability and below
    ``len(dist) - 1``; distributions too short to smooth are returned as
    they are.
    """
    from mirt.equating.kernel import _fit_loglinear, _polynomial_design

    degree = min(degree, len(dist) - 2, int(np.count_nonzero(dist)) - 1)
    if degree < 1:
        return np.array(dist, dtype=np.float64, copy=True)
    return _fit_loglinear(dist, _polynomial_design(dist, degree))


def _kernel_smooth(
    dist: NDArray[np.float64],
    bandwidth: float | None = None,
) -> NDArray[np.float64]:
    """Apply kernel smoothing to score distribution."""
    n = len(dist)
    if n == 1:
        return dist.copy()
    scores = np.arange(n, dtype=np.float64)

    if bandwidth is None:
        bandwidth = 0.5
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be finite and positive")

    scaled_differences = (scores[:, None] - scores[None, :]) / bandwidth
    kernel = np.exp(-0.5 * scaled_differences**2)
    kernel /= np.sum(kernel, axis=1, keepdims=True)
    smoothed = kernel @ dist
    return np.asarray(smoothed / np.sum(smoothed), dtype=np.float64)


def _compute_expected_scores(
    model: "BaseItemModel",
    theta: NDArray[np.float64],
    items: list[int] | NDArray[np.intp] | None = None,
) -> NDArray[np.float64]:
    """Compute expected scores at each theta."""
    _validate_model(model, "model")
    theta = _validate_vector(theta, "theta")
    item_indices = _resolve_items(model, items, "items")
    if not model.is_polytomous:
        correct = _correct_probabilities(model, theta, item_indices)
        return np.asarray(np.sum(correct, axis=1), dtype=np.float64)
    probabilities, _ = _polytomous_probabilities(model, theta, item_indices)
    categories = np.arange(probabilities.shape[2], dtype=np.float64)
    return np.asarray(np.sum(probabilities, axis=1) @ categories, dtype=np.float64)


def score_to_theta(
    model: "BaseItemModel",
    scores: NDArray[np.float64],
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 201,
    items: list[int] | None = None,
) -> NDArray[np.float64]:
    """Convert raw scores to theta estimates.

    Uses inverse of expected score function.

    Parameters
    ----------
    model : BaseItemModel
        IRT model.
    scores : NDArray
        Raw scores to convert.
    theta_range : tuple[float, float]
        Range for theta lookup.
    n_theta : int
        Number of theta points.
    items : list[int] | None
        Subset of items.

    Returns
    -------
    NDArray
        Theta estimates corresponding to scores.
    """
    lower, upper = _validate_theta_range(theta_range)
    n_theta = _validate_count(n_theta, "n_theta", minimum=2)
    _validate_model(model, "model")
    item_indices = _resolve_items(model, items, "items")
    scores_array = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(scores_array)):
        raise ValueError("scores must contain only finite values")

    theta_grid = np.linspace(lower, upper, n_theta)
    expected = _compute_expected_scores(model, theta_grid, item_indices)
    _validate_expected_score_curve(expected, "model")
    return _invert_expected_scores(expected, theta_grid, scores_array)


def theta_to_score(
    model: "BaseItemModel",
    theta: NDArray[np.float64],
    items: list[int] | None = None,
) -> NDArray[np.float64]:
    """Convert theta estimates to expected scores.

    Parameters
    ----------
    model : BaseItemModel
        IRT model.
    theta : NDArray
        Theta values.
    items : list[int] | None
        Subset of items.

    Returns
    -------
    NDArray
        Expected scores at each theta.
    """
    _validate_model(model, "model")
    item_indices = _resolve_items(model, items, "items")
    theta_array = np.asarray(theta, dtype=np.float64)
    if not np.all(np.isfinite(theta_array)):
        raise ValueError("theta must contain only finite values")
    original_shape = theta_array.shape
    expected = _compute_expected_scores(model, theta_array.reshape(-1), item_indices)
    return expected.reshape(original_shape)


def score_equating_summary(result: ScoreEquatingResult) -> str:
    """Generate summary table of score equating.

    Parameters
    ----------
    result : ScoreEquatingResult
        Score equating result.

    Returns
    -------
    str
        Formatted score conversion table.
    """
    lines = []
    lines.append("=" * 50)
    lines.append(f"Score Equating Table ({result.method})")
    lines.append("=" * 50)
    lines.append(f"{'Old Score':>12} {'New Score':>12} {'Rounded':>12}")
    lines.append("-" * 50)

    for old, new in zip(result.old_scores, result.new_scores, strict=True):
        rounded = round(new)
        lines.append(f"{old:>12.1f} {new:>12.2f} {rounded:>12d}")

    lines.append("-" * 50)

    corr = np.corrcoef(result.old_scores, result.new_scores)[0, 1]
    lines.append(f"Correlation: {corr:.4f}")
    lines.append(
        f"Mean difference: {np.mean(result.new_scores - result.old_scores):.2f}"
    )

    lines.append("=" * 50)

    return "\n".join(lines)


def compute_see(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    theta_grid: NDArray[np.float64],
    items_old: list[int] | None = None,
    items_new: list[int] | None = None,
) -> NDArray[np.float64]:
    """Compute standard error of equating (SEE).

    Based on delta method approximation.

    Parameters
    ----------
    model_old : BaseItemModel
        Old form model.
    model_new : BaseItemModel
        New form model.
    theta_grid : NDArray
        Grid of theta values.
    items_old : list[int] | None
        Old form items.
    items_new : list[int] | None
        New form items.

    Returns
    -------
    NDArray
        Standard error of equating at each theta.
    """
    _validate_model(model_old, "model_old")
    _validate_model(model_new, "model_new")
    theta_grid = _validate_vector(theta_grid, "theta_grid")
    old_item_indices = _resolve_items(model_old, items_old, "items_old")
    new_item_indices = _resolve_items(model_new, items_new, "items_new")

    variance_old = _conditional_score_variance(model_old, theta_grid, old_item_indices)
    variance_new = _conditional_score_variance(model_new, theta_grid, new_item_indices)
    return np.sqrt(np.maximum(variance_old + variance_new, 0.0))
