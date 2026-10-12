"""Batched Newton solves for built-in item M-steps."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

if TYPE_CHECKING:
    from mirt.estimation.priors import Prior

_MAX_HALVINGS = 30
_RIDGE = 1e-12

# ``objective(rows, x, derivatives)`` of :func:`newton_items`: the objectives
# of items ``rows`` at coordinates ``x``, and with ``derivatives`` their
# gradients and curvature matrices (None otherwise).
ItemObjective: TypeAlias = Callable[
    [NDArray[np.intp], NDArray[np.float64], bool],
    tuple[NDArray[np.float64], NDArray[np.float64] | None, NDArray[np.float64] | None],
]


def _negative_log_likelihood(
    logits: NDArray[np.float64],
    correct: NDArray[np.float64],
    observed: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate each row's binomial objective stably in both logistic tails."""
    return np.sum(
        correct * np.logaddexp(0.0, -logits)
        + (observed - correct) * np.logaddexp(0.0, logits),
        axis=-1,
    )


def _at_free(
    function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    values: NDArray[np.float64],
    free: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Evaluate a prior function at the free entries only, with zeros elsewhere."""
    result = np.zeros_like(values)
    if free.any():
        result[free] = function(values[free])
    return result


def _logistic_log_prior(
    coefficients: NDArray[np.float64],
    free: NDArray[np.bool_],
    slope_sums: NDArray[np.float64],
    slope_prior: Prior | None,
    location_prior: Prior | None,
    *,
    estimate_slopes: bool,
    difficulty: bool,
    derivatives: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return item log-priors with their gradients and Hessians.

    ``coefficients`` holds each item's slopes and intercept ``d``, or only
    ``d`` when the slopes are fixed, and ``slope_sums`` the items' summed
    slopes. The slope prior applies to every free slope, the location prior
    to a free intercept or, with ``difficulty``, to the difficulty
    ``b = -d / sum(a)``, whose derivatives in the Newton coordinates follow
    by the chain rule. Without ``derivatives`` the gradients and Hessians
    are zero.
    """
    n_rows, width = coefficients.shape
    value = np.zeros(n_rows)
    gradient = np.zeros((n_rows, width))
    hessian = np.zeros((n_rows, width, width))
    if slope_prior is not None and estimate_slopes:
        slopes, mask = coefficients[:, :-1], free[:, :-1]
        value += _at_free(slope_prior.log_pdf, slopes, mask).sum(axis=1)
        if derivatives:
            gradient[:, :-1] = _at_free(slope_prior.grad_log_pdf, slopes, mask)
            diagonal = np.arange(width - 1)
            hessian[:, diagonal, diagonal] = _at_free(
                slope_prior.hess_log_pdf, slopes, mask
            )
    if location_prior is None:
        return value, gradient, hessian
    mask = free[:, -1]
    intercepts = coefficients[:, -1]
    if not difficulty:
        value += _at_free(location_prior.log_pdf, intercepts, mask)
        if derivatives:
            gradient[:, -1] += _at_free(location_prior.grad_log_pdf, intercepts, mask)
            hessian[:, -1, -1] += _at_free(
                location_prior.hess_log_pdf, intercepts, mask
            )
        return value, gradient, hessian

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inverse = np.where(mask, 1.0 / slope_sums, 0.0)
        locations = -intercepts * inverse
    value += _at_free(location_prior.log_pdf, locations, mask)
    if not derivatives:
        return value, gradient, hessian
    first = _at_free(location_prior.grad_log_pdf, locations, mask)
    second = _at_free(location_prior.hess_log_pdf, locations, mask)
    # b = -d / s with s the slope sum: db/da_f = -b / s and db/dd = -1 / s,
    # d2b/da_f da_g = 2 b / s^2, d2b/da_f dd = 1 / s^2 and d2b/dd2 = 0.
    jacobian = np.empty((n_rows, width))
    jacobian[:, -1] = -inverse
    curvature = np.zeros((n_rows, width, width))
    if estimate_slopes:
        jacobian[:, :-1] = (-locations * inverse)[:, None]
        curvature[:, :-1, :-1] = (2.0 * locations * inverse**2)[:, None, None]
        curvature[:, :-1, -1] = curvature[:, -1, :-1] = (inverse**2)[:, None]
    gradient += first[:, None] * jacobian
    hessian += (
        second[:, None, None] * jacobian[:, :, None] * jacobian[:, None, :]
        + first[:, None, None] * curvature
    )
    return value, gradient, hessian


def _definite_steps(
    gradient: NDArray[np.float64], curvature: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return Newton steps and whether each item's curvature is positive definite.

    Items whose curvature is not positive definite take a gradient step
    scaled by the absolute curvature diagonal (at least one), which still
    descends.
    """
    diagonal = np.arange(curvature.shape[-1])
    positive = np.linalg.eigvalsh(curvature)[:, 0] > 0.0
    if not positive.all():
        rows = np.flatnonzero(~positive)
        scale = np.maximum(np.abs(curvature[rows][:, diagonal, diagonal]), 1.0)
        curvature = curvature.copy()
        curvature[rows] = 0.0
        curvature[rows[:, None], diagonal, diagonal] = scale
    return np.linalg.solve(curvature, gradient[..., None])[..., 0], positive


def newton_logistic_items(
    points: NDArray[np.float64],
    correct: NDArray[np.float64],
    observed: NDArray[np.float64],
    slopes: NDArray[np.float64],
    intercepts: NDArray[np.float64],
    *,
    estimate_slopes: bool = True,
    free: NDArray[np.bool_] | None = None,
    slope_prior: Prior | None = None,
    location_prior: Prior | None = None,
    difficulty: bool = False,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Maximize every item's expected binomial log-likelihood at once.

    Item ``j`` has logits ``points @ slopes[j] + intercepts[j]`` at the
    quadrature nodes. Each item is an independent logistic regression on the
    expected counts, solved by Newton's method with per-item step halving. The
    objective is unbounded and unclipped, so callers check the returned
    estimates against their own parameter boxes. With item priors the
    objective is the log-posterior: steps to points without prior density
    are halved, and items whose Hessian is not positive definite take
    gradient steps scaled by its diagonal until it is.

    Parameters
    ----------
    points : ndarray of shape (n_points, n_factors)
        Quadrature nodes.
    correct : ndarray of shape (n_items, n_points)
        Expected numbers of correct responses at each node.
    observed : ndarray of shape (n_items, n_points)
        Expected numbers of observed responses at each node.
    slopes : ndarray of shape (n_items, n_factors)
        Starting slopes. They stay fixed when ``estimate_slopes`` is false.
    intercepts : ndarray of shape (n_items,)
        Starting intercepts.
    estimate_slopes : bool, default=True
        Whether slopes are free coordinates.
    free : ndarray of bool, optional
        Free coefficients of each item, of shape ``(n_items, n_factors + 1)``
        for the slopes and the intercept, or ``(n_items, 1)`` for the
        intercept when ``estimate_slopes`` is false. Fixed coefficients,
        such as the zero loadings of a confirmatory model, keep their
        starting values: Newton steps solve for the free ones only, with the
        fixed slopes acting as offsets of the logits. By default every
        coefficient is free.
    slope_prior : Prior, optional
        Prior of every free slope. It applies only when ``estimate_slopes``
        is true.
    location_prior : Prior, optional
        Prior of each free intercept, or with ``difficulty`` of the
        difficulty ``b = -d / sum(a)`` of the 1PL and 2PL, whose derivatives
        in the slope-intercept coordinates follow by the chain rule.
    difficulty : bool, default=False
        Whether ``location_prior`` is on the difficulty rather than the
        intercept.
    max_iter : int, default=50
        Maximum Newton iterations per item.
    tol : float, default=1e-10
        An item converges once its full Newton step is at most ``tol`` in
        every coordinate.

    Returns
    -------
    slopes : ndarray of shape (n_items, n_factors)
        Updated slopes.
    intercepts : ndarray of shape (n_items,)
        Updated intercepts.
    converged : ndarray of shape (n_items,)
        Whether each item reached the Newton-step tolerance. Unconverged items
        include those whose maximum lies at infinity, such as items every
        respondent answered correctly, and with priors those whose
        log-posterior or derivatives are not finite.
    """
    points = np.asarray(points, dtype=np.float64)
    n_items = correct.shape[0]
    slopes = np.array(slopes, dtype=np.float64).reshape(n_items, -1)
    intercepts = np.asarray(intercepts, dtype=np.float64).reshape(n_items)
    ones = np.ones((points.shape[0], 1))
    if estimate_slopes:
        design = np.hstack((points, ones))
        coefficients = np.hstack((slopes, intercepts[:, None]))
        offsets = np.zeros_like(correct)
    else:
        design = ones
        coefficients = intercepts[:, None].copy()
        offsets = slopes @ points.T
    diagonal = np.arange(design.shape[1])
    free_mask = None
    if free is not None:
        free_mask = np.asarray(free, dtype=np.bool_).reshape(n_items, design.shape[1])
        if free_mask.all():
            free_mask = None
    penalized = slope_prior is not None or location_prior is not None
    prior_free = (
        np.ones(coefficients.shape, dtype=np.bool_) if free_mask is None else free_mask
    )
    fixed_sums = slopes.sum(axis=1)

    def negative_log_prior(
        rows: NDArray[np.intp], values: NDArray[np.float64], derivatives: bool
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        sums = values[:, :-1].sum(axis=1) if estimate_slopes else fixed_sums[rows]
        value, gradient, hessian = _logistic_log_prior(
            values,
            prior_free[rows],
            sums,
            slope_prior,
            location_prior,
            estimate_slopes=estimate_slopes,
            difficulty=difficulty,
            derivatives=derivatives,
        )
        return -value, -gradient, -hessian

    logits = offsets + coefficients @ design.T
    loss = _negative_log_likelihood(logits, correct, observed)
    if penalized:
        with np.errstate(invalid="ignore", over="ignore"):
            loss += negative_log_prior(np.arange(n_items), coefficients, False)[0]
    # Starting values without prior density leave the item to the caller.
    active = np.isfinite(loss)
    converged = np.zeros(n_items, dtype=np.bool_)
    for _ in range(max_iter):
        items = np.flatnonzero(active)
        if not items.size:
            break
        counts = observed[items]
        probability = expit(logits[items])
        gradient = (counts * probability - correct[items]) @ design
        weights = counts * probability * (1.0 - probability)
        hessian = (design.T * weights[:, None, :]) @ design
        if penalized:
            _, prior_gradient, prior_hessian = negative_log_prior(
                items, coefficients[items], True
            )
            gradient += prior_gradient
            hessian += prior_hessian
        hessian[:, diagonal, diagonal] += _RIDGE * (
            1.0 + np.abs(hessian[:, diagonal, diagonal])
        )
        if free_mask is not None:
            # Fixed coefficients get identity rows and columns and no
            # gradient, so their steps are zero and the free coefficients
            # solve the Newton system restricted to them.
            item_free = free_mask[items]
            gradient = np.where(item_free, gradient, 0.0)
            pairs = item_free[:, :, None] & item_free[:, None, :]
            hessian = np.where(pairs, hessian, 0.0)
            hessian[:, diagonal, diagonal] += ~item_free
        definite = None
        if penalized:
            # Points outside a prior's support have no finite derivatives.
            usable = np.isfinite(gradient).all(axis=1) & np.isfinite(hessian).all(
                axis=(1, 2)
            )
            if not usable.all():
                active[items[~usable]] = False
                items = items[usable]
                gradient, hessian = gradient[usable], hessian[usable]
                if not items.size:
                    continue
        try:
            if penalized:
                step, definite = _definite_steps(gradient, hessian)
            else:
                step = np.linalg.solve(hessian, gradient[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        with np.errstate(invalid="ignore"):
            small = np.max(np.abs(step), axis=1) <= tol
        if definite is not None:
            # A small gradient step is not a stationary point of a curvature
            # that is not positive definite.
            small &= definite

        start = coefficients[items]
        base_loss = loss[items]
        slack = 1e-12 * np.maximum(1.0, np.abs(base_loss))
        scale = np.ones(items.size)
        pending = np.ones(items.size, dtype=np.bool_)
        for _ in range(_MAX_HALVINGS):
            rows = np.flatnonzero(pending)
            if not rows.size:
                break
            selected = items[rows]
            trial = start[rows] - scale[rows, None] * step[rows]
            trial_logits = offsets[selected] + trial @ design.T
            with np.errstate(invalid="ignore", over="ignore"):
                trial_loss = _negative_log_likelihood(
                    trial_logits, correct[selected], observed[selected]
                )
                if penalized:
                    trial_loss += negative_log_prior(selected, trial, False)[0]
                accept = trial_loss <= base_loss[rows] + slack[rows]
            accepted = selected[accept]
            coefficients[accepted] = trial[accept]
            logits[accepted] = trial_logits[accept]
            loss[accepted] = trial_loss[accept]
            pending[rows[accept]] = False
            scale[rows[~accept]] *= 0.5

        # A tiny full step marks a stationary point even when roundoff
        # prevents any further decrease in the objective.
        converged[items[small]] = True
        active[items[small | pending]] = False

    if estimate_slopes:
        return coefficients[:, :-1], coefficients[:, -1], converged
    return slopes, coefficients[:, 0], converged


def newton_items(
    start: NDArray[np.float64],
    objective: ItemObjective,
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    *,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Minimize independent item objectives by damped Newton steps.

    ``objective(rows, x, derivatives)`` returns the objectives of items
    ``rows`` at the coordinate rows ``x``, and with ``derivatives`` also their
    gradients and symmetric curvature matrices of shape
    ``(len(rows), n_coordinates, n_coordinates)``: Hessians, or the expected
    information for Fisher scoring. Items whose curvature is not positive
    definite take gradient steps scaled by its diagonal. Steps are halved
    until they stay in the box ``[lower, upper]`` and do not raise the
    objective.

    Parameters
    ----------
    start : ndarray of shape (n_items, n_coordinates)
        Starting coordinates, inside the box.
    objective : callable
        Item objectives, as above; infeasible points have infinite values.
    lower, upper : ndarray
        Box bounds, broadcastable to ``start``.
    max_iter : int, default=50
        Maximum Newton iterations per item.
    tol : float, default=1e-10
        An item converges once its full step, taken with a positive definite
        curvature, is at most ``tol`` in every coordinate.

    Returns
    -------
    x : ndarray of shape (n_items, n_coordinates)
        Updated coordinates.
    converged : ndarray of shape (n_items,)
        Whether each item reached the step tolerance. Unconverged items
        include those whose objective or derivatives are not finite and
        those whose minimum lies on the box.
    """
    x = np.array(start, dtype=np.float64)
    n_items, width = x.shape
    lower = np.broadcast_to(np.asarray(lower, dtype=np.float64), x.shape)
    upper = np.broadcast_to(np.asarray(upper, dtype=np.float64), x.shape)
    diagonal = np.arange(width)
    converged = np.zeros(n_items, dtype=np.bool_)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        loss, gradient, curvature = objective(np.arange(n_items), x, True)
    assert gradient is not None and curvature is not None
    active = np.isfinite(loss) & np.all((x >= lower) & (x <= upper), axis=1)
    for _ in range(max_iter):
        items = np.flatnonzero(active)
        if not items.size:
            break
        item_gradient, item_curvature = gradient[items], curvature[items].copy()
        usable = np.isfinite(item_gradient).all(axis=1) & np.isfinite(
            item_curvature
        ).all(axis=(1, 2))
        if not usable.all():
            active[items[~usable]] = False
            items = items[usable]
            item_gradient, item_curvature = (
                item_gradient[usable],
                item_curvature[usable],
            )
            if not items.size:
                break
        item_curvature[:, diagonal, diagonal] += _RIDGE * (
            1.0 + np.abs(item_curvature[:, diagonal, diagonal])
        )
        try:
            step, definite = _definite_steps(item_gradient, item_curvature)
        except np.linalg.LinAlgError:
            break
        small = definite & (np.max(np.abs(step), axis=1) <= tol)

        origin = x[items]
        base_loss = loss[items]
        slack = 1e-12 * np.maximum(1.0, np.abs(base_loss))
        scale = np.ones(items.size)
        pending = np.ones(items.size, dtype=np.bool_)
        for _ in range(_MAX_HALVINGS):
            rows = np.flatnonzero(pending)
            if not rows.size:
                break
            selected = items[rows]
            trial = origin[rows] - scale[rows, None] * step[rows]
            inside = np.all(
                (trial >= lower[selected]) & (trial <= upper[selected]), axis=1
            )
            trial_loss = np.full(rows.size, np.inf)
            if inside.any():
                with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                    trial_loss[inside] = objective(
                        selected[inside], trial[inside], False
                    )[0]
            with np.errstate(invalid="ignore"):
                accept = trial_loss <= base_loss[rows] + slack[rows]
            accepted = selected[accept]
            x[accepted] = trial[accept]
            loss[accepted] = trial_loss[accept]
            pending[rows[accept]] = False
            scale[rows[~accept]] *= 0.5

        # A tiny full step marks a stationary point even when roundoff
        # prevents any further decrease in the objective.
        converged[items[small]] = True
        active[items[small | pending]] = False
        moved = items[~pending & ~small]
        if moved.size:
            with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
                _, moved_gradient, moved_curvature = objective(moved, x[moved], True)
            assert moved_gradient is not None and moved_curvature is not None
            gradient[moved], curvature[moved] = moved_gradient, moved_curvature
    return x, converged
