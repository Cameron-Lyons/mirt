"""True and observed score equating methods.

This module provides IRT-based score equating including true score
equating and observed score equating via Lord-Wingersky recursion.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate, stats

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

from mirt.equating.linking import LinkingResult


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
        Theta values used for mapping.
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
        Linking constants if new model is on different scale.
    theta_range : tuple[float, float]
        Range of theta for score mapping.
    n_theta : int
        Number of theta points.
    items_old : list[int] | None
        Subset of items for old form. None = all items.
    items_new : list[int] | None
        Subset of items for new form. None = all items.

    Returns
    -------
    ScoreEquatingResult
        Score conversion table and diagnostics.
    """
    theta_grid = np.linspace(theta_range[0], theta_range[1], n_theta)

    expected_old = _compute_expected_scores(model_old, theta_grid, items_old)
    expected_new = _compute_expected_scores(model_new, theta_grid, items_new)

    if linking_result is not None:
        A = linking_result.constants.A
        B = linking_result.constants.B
        theta_transformed = A * theta_grid + B
        expected_new = _compute_expected_scores(model_new, theta_transformed, items_new)

    min_score_old = 0
    max_score_old = int(np.ceil(np.max(expected_old)))
    old_scores = np.arange(min_score_old, max_score_old + 1, dtype=np.float64)

    new_scores = np.zeros_like(old_scores)

    for i, score in enumerate(old_scores):
        idx = np.searchsorted(expected_old, score)

        if idx == 0:
            theta_at_score = theta_grid[0]
        elif idx >= len(theta_grid):
            theta_at_score = theta_grid[-1]
        else:
            t0 = theta_grid[idx - 1]
            t1 = theta_grid[idx]
            e0 = expected_old[idx - 1]
            e1 = expected_old[idx]
            if abs(e1 - e0) < 1e-10:
                theta_at_score = (t0 + t1) / 2
            else:
                theta_at_score = t0 + (score - e0) * (t1 - t0) / (e1 - e0)

        theta_at_score = np.clip(theta_at_score, theta_range[0], theta_range[1])

        idx_new = np.searchsorted(theta_grid, theta_at_score)
        if idx_new == 0:
            new_scores[i] = expected_new[0]
        elif idx_new >= len(theta_grid):
            new_scores[i] = expected_new[-1]
        else:
            t0 = theta_grid[idx_new - 1]
            t1 = theta_grid[idx_new]
            e0 = expected_new[idx_new - 1]
            e1 = expected_new[idx_new]
            frac = (theta_at_score - t0) / max(t1 - t0, 1e-10)
            new_scores[i] = e0 + frac * (e1 - e0)

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
        Prior distribution of theta. Default: standard normal.
    theta_grid : NDArray | None
        Grid of theta values for integration.
    n_theta : int
        Number of theta points if grid not provided.
    items_old : list[int] | None
        Subset of items for old form.
    items_new : list[int] | None
        Subset of items for new form.

    Returns
    -------
    ScoreEquatingResult
        Score conversion table.
    """
    if theta_grid is None:
        theta_grid = np.linspace(-4, 4, n_theta)

    if theta_distribution is None:
        theta_distribution = stats.norm.pdf(theta_grid)
        theta_distribution = theta_distribution / np.sum(theta_distribution)

    score_dist_old = lord_wingersky_recursion(
        model_old, theta_grid, theta_distribution, items_old
    )
    score_dist_new = lord_wingersky_recursion(
        model_new, theta_grid, theta_distribution, items_new
    )

    new_scores = equipercentile_equating(score_dist_old, score_dist_new)

    old_scores = np.arange(len(score_dist_old), dtype=np.float64)

    return ScoreEquatingResult(
        old_scores=old_scores,
        new_scores=new_scores,
        theta=theta_grid,
        standard_errors=None,
        method="observed_score",
    )


def lord_wingersky_recursion(
    model: "BaseItemModel",
    theta_grid: NDArray[np.float64],
    theta_weights: NDArray[np.float64],
    items: list[int] | None = None,
) -> NDArray[np.float64]:
    """Compute observed score distribution using Lord-Wingersky recursion.

    For dichotomous items, recursively computes P(X=x) for each
    possible sum score x.

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

    Returns
    -------
    NDArray
        Marginal score distribution P(X=x) for x = 0, 1, ..., n_items.
    """
    disc = np.asarray(model.discrimination)
    diff = np.asarray(model.difficulty)

    if disc.ndim > 1:
        disc = disc[:, 0]

    if items is not None:
        disc = disc[items]
        diff = diff[items]

    n_items = len(disc)
    n_theta = len(theta_grid)
    max_score = n_items

    probs = np.zeros((n_theta, n_items))
    for j in range(n_items):
        z = disc[j] * (theta_grid - diff[j])
        probs[:, j] = 1.0 / (1.0 + np.exp(-z))

    f_prev = np.zeros((n_theta, max_score + 1))
    f_prev[:, 0] = 1.0

    for j in range(n_items):
        f_curr = np.zeros((n_theta, max_score + 1))

        p_j = probs[:, j]
        q_j = 1.0 - p_j

        for x in range(j + 2):
            if x == 0:
                f_curr[:, x] = f_prev[:, x] * q_j
            elif x == j + 1:
                f_curr[:, x] = f_prev[:, x - 1] * p_j
            else:
                f_curr[:, x] = f_prev[:, x] * q_j + f_prev[:, x - 1] * p_j

        f_prev = f_curr

    marginal = np.zeros(max_score + 1)
    for x in range(max_score + 1):
        marginal[x] = np.sum(theta_weights * f_prev[:, x])

    marginal = marginal / np.sum(marginal)

    return marginal


def equipercentile_equating(
    score_dist_old: NDArray[np.float64],
    score_dist_new: NDArray[np.float64],
    smoothing: Literal["none", "loglinear", "kernel"] = "none",
) -> NDArray[np.float64]:
    """Perform equipercentile equating between score distributions.

    Finds score on new form with same percentile rank as each
    score on old form.

    Parameters
    ----------
    score_dist_old : NDArray
        Score distribution for old form P(X=x).
    score_dist_new : NDArray
        Score distribution for new form P(Y=y).
    smoothing : str
        Smoothing method: "none", "loglinear", or "kernel".

    Returns
    -------
    NDArray
        Equivalent scores on new form for each old score.
    """
    if smoothing == "loglinear":
        score_dist_old = _loglinear_smooth(score_dist_old)
        score_dist_new = _loglinear_smooth(score_dist_new)
    elif smoothing == "kernel":
        score_dist_old = _kernel_smooth(score_dist_old)
        score_dist_new = _kernel_smooth(score_dist_new)

    cum_old = np.cumsum(score_dist_old)
    cum_new = np.cumsum(score_dist_new)

    n_old = len(score_dist_old)
    n_new = len(score_dist_new)

    percentiles_old = np.zeros(n_old)
    for x in range(n_old):
        if x == 0:
            percentiles_old[x] = score_dist_old[x] / 2
        else:
            percentiles_old[x] = cum_old[x - 1] + score_dist_old[x] / 2

    new_scores = np.arange(n_new, dtype=np.float64)
    percentiles_new = np.zeros(n_new)
    for y in range(n_new):
        if y == 0:
            percentiles_new[y] = score_dist_new[y] / 2
        else:
            percentiles_new[y] = cum_new[y - 1] + score_dist_new[y] / 2

    equated = np.zeros(n_old)
    for x in range(n_old):
        p = percentiles_old[x]

        idx = np.searchsorted(percentiles_new, p)

        if idx == 0:
            equated[x] = new_scores[0]
        elif idx >= n_new:
            equated[x] = new_scores[-1]
        else:
            y0 = new_scores[idx - 1]
            y1 = new_scores[idx]
            p0 = percentiles_new[idx - 1]
            p1 = percentiles_new[idx]
            if abs(p1 - p0) < 1e-10:
                equated[x] = (y0 + y1) / 2
            else:
                equated[x] = y0 + (p - p0) * (y1 - y0) / (p1 - p0)

    return equated


def _loglinear_smooth(
    dist: NDArray[np.float64], degree: int = 4
) -> NDArray[np.float64]:
    """Apply log-linear smoothing to score distribution."""
    n = len(dist)
    scores = np.arange(n)

    dist = np.maximum(dist, 1e-10)
    log_dist = np.log(dist)

    X = np.column_stack([scores**k for k in range(degree + 1)])

    try:
        coeffs = np.linalg.lstsq(X, log_dist, rcond=None)[0]
        smoothed = np.exp(X @ coeffs)
    except np.linalg.LinAlgError:
        return dist

    smoothed = smoothed / np.sum(smoothed)

    return smoothed


def _kernel_smooth(
    dist: NDArray[np.float64],
    bandwidth: float | None = None,
) -> NDArray[np.float64]:
    """Apply kernel smoothing to score distribution."""
    n = len(dist)
    scores = np.arange(n)

    if bandwidth is None:
        bandwidth = 0.5

    smoothed = np.zeros(n)
    for i in range(n):
        weights = stats.norm.pdf((scores - i) / bandwidth)
        weights = weights / np.sum(weights)
        smoothed[i] = np.sum(weights * dist)

    smoothed = smoothed / np.sum(smoothed)

    return smoothed


def _compute_expected_scores(
    model: "BaseItemModel",
    theta: NDArray[np.float64],
    items: list[int] | None = None,
) -> NDArray[np.float64]:
    """Compute expected scores at each theta."""
    disc = np.asarray(model.discrimination)
    diff = np.asarray(model.difficulty)

    if disc.ndim > 1:
        disc = disc[:, 0]

    if items is not None:
        disc = disc[items]
        diff = diff[items]

    n_items = len(disc)
    n_theta = len(theta)

    expected = np.zeros(n_theta)
    for j in range(n_items):
        z = disc[j] * (theta - diff[j])
        p = 1.0 / (1.0 + np.exp(-z))
        expected += p

    return expected


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
    theta_grid = np.linspace(theta_range[0], theta_range[1], n_theta)
    expected = _compute_expected_scores(model, theta_grid, items)

    interp_func = interpolate.interp1d(
        expected,
        theta_grid,
        kind="linear",
        bounds_error=False,
        fill_value=(theta_range[0], theta_range[1]),
    )

    return interp_func(scores)


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
    return _compute_expected_scores(model, theta, items)


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

    for old, new in zip(result.old_scores, result.new_scores):
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
    disc_old = np.asarray(model_old.discrimination)
    diff_old = np.asarray(model_old.difficulty)
    if disc_old.ndim > 1:
        disc_old = disc_old[:, 0]
    if items_old is not None:
        disc_old = disc_old[items_old]
        diff_old = diff_old[items_old]

    disc_new = np.asarray(model_new.discrimination)
    diff_new = np.asarray(model_new.difficulty)
    if disc_new.ndim > 1:
        disc_new = disc_new[:, 0]
    if items_new is not None:
        disc_new = disc_new[items_new]
        diff_new = diff_new[items_new]

    see = np.zeros(len(theta_grid))
    for i, theta in enumerate(theta_grid):
        var_old = 0.0
        for j in range(len(disc_old)):
            z = disc_old[j] * (theta - diff_old[j])
            p = 1.0 / (1.0 + np.exp(-z))
            var_old += p * (1 - p)

        var_new = 0.0
        for j in range(len(disc_new)):
            z = disc_new[j] * (theta - diff_new[j])
            p = 1.0 / (1.0 + np.exp(-z))
            var_new += p * (1 - p)

        see[i] = np.sqrt(var_old + var_new)

    return see
