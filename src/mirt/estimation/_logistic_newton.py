"""Batched Newton solves for built-in logistic item M-steps."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

_MAX_HALVINGS = 30
_RIDGE = 1e-12


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


def newton_logistic_items(
    points: NDArray[np.float64],
    correct: NDArray[np.float64],
    observed: NDArray[np.float64],
    slopes: NDArray[np.float64],
    intercepts: NDArray[np.float64],
    *,
    estimate_slopes: bool = True,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Maximize every item's expected binomial log-likelihood at once.

    Item ``j`` has logits ``points @ slopes[j] + intercepts[j]`` at the
    quadrature nodes. Each item is an independent logistic regression on the
    expected counts, solved by Newton's method with per-item step halving. The
    objective is unbounded and unclipped, so callers check the returned
    estimates against their own parameter boxes.

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
        respondent answered correctly.
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

    logits = offsets + coefficients @ design.T
    loss = _negative_log_likelihood(logits, correct, observed)
    active = np.ones(n_items, dtype=np.bool_)
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
        hessian[:, diagonal, diagonal] += _RIDGE * (
            1.0 + hessian[:, diagonal, diagonal]
        )
        try:
            step = np.linalg.solve(hessian, gradient[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        with np.errstate(invalid="ignore"):
            small = np.max(np.abs(step), axis=1) <= tol

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
