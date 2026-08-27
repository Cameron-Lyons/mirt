"""True and observed score equating methods.

This module provides IRT-based score equating including true score
equating and observed score equating via Lord-Wingersky recursion.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from mirt.backends.rust.equating import (
    observed_score_distribution_2pl as _rust_observed_score_distribution_2pl,
)
from mirt.equating.linking import LinkingResult

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


_PROBABILITY_TOLERANCE = 1e-10
_SMOOTHING_METHODS = ("none", "loglinear", "kernel")


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
    lower, upper = _validate_theta_range(theta_range)
    n_theta = _validate_count(n_theta, "n_theta", minimum=2)
    _validate_model(model_old, "model_old")
    _validate_model(model_new, "model_new")
    old_item_indices = _resolve_items(model_old, items_old, "items_old")
    new_item_indices = _resolve_items(model_new, items_new, "items_new")

    theta_grid = np.linspace(lower, upper, n_theta)
    expected_old = _compute_expected_scores(model_old, theta_grid, old_item_indices)
    expected_new = _compute_expected_scores(model_new, theta_grid, new_item_indices)

    if linking_result is not None:
        A = float(linking_result.constants.A)
        B = float(linking_result.constants.B)
        if not np.isfinite(A) or A <= 0.0:
            raise ValueError("linking_result.constants.A must be finite and positive")
        if not np.isfinite(B):
            raise ValueError("linking_result.constants.B must be finite")
        theta_transformed = A * theta_grid + B
        expected_new = _compute_expected_scores(
            model_new, theta_transformed, new_item_indices
        )

    _validate_expected_score_curve(expected_old, "model_old")
    _validate_expected_score_curve(expected_new, "model_new")

    max_score_old = _maximum_score(model_old, old_item_indices)
    old_scores = np.arange(max_score_old + 1, dtype=np.float64)
    theta_at_score = _invert_expected_scores(expected_old, theta_grid, old_scores)
    new_scores = np.interp(theta_at_score, theta_grid, expected_new)

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
    smoothing : {"none", "loglinear", "kernel"}
        Score-distribution smoothing applied before equipercentile inversion.

    Returns
    -------
    ScoreEquatingResult
        Score conversion table.
    """
    _validate_model(model_old, "model_old")
    _validate_model(model_new, "model_new")
    _resolve_items(model_old, items_old, "items_old")
    _resolve_items(model_new, items_new, "items_new")
    if smoothing not in _SMOOTHING_METHODS:
        raise ValueError("smoothing must be one of 'none', 'loglinear', or 'kernel'")

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

    score_dist_old = lord_wingersky_recursion(
        model_old, theta_grid, theta_distribution, items_old
    )
    score_dist_new = lord_wingersky_recursion(
        model_new, theta_grid, theta_distribution, items_new
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


def lord_wingersky_recursion(
    model: "BaseItemModel",
    theta_grid: NDArray[np.float64],
    theta_weights: NDArray[np.float64],
    items: list[int] | None = None,
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

    Returns
    -------
    NDArray
        Marginal score distribution over every attainable sum score.
    """
    _validate_model(model, "model")
    theta_grid = _validate_vector(theta_grid, "theta_grid")
    weights = _validate_weights(theta_weights, len(theta_grid), "theta_weights")
    item_indices = _resolve_items(model, items, "items")
    native_distribution = _native_score_distribution(
        model, theta_grid, weights, item_indices
    )
    if native_distribution is not None:
        return native_distribution

    item_probabilities = _item_score_probabilities(model, theta_grid, item_indices)

    conditional = np.ones((len(theta_grid), 1), dtype=np.float64)
    for probabilities in item_probabilities:
        current_width = conditional.shape[1]
        n_categories = probabilities.shape[1]
        updated = np.zeros(
            (len(theta_grid), current_width + n_categories - 1),
            dtype=np.float64,
        )
        for score in range(n_categories):
            updated[:, score : score + current_width] += (
                conditional * probabilities[:, score, None]
            )
        conditional = updated

    return _normalize_score_distribution(weights @ conditional)


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
    if model.is_polytomous or model.model_name not in {"1PL", "2PL"}:
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
    return _normalize_score_distribution(weights @ conditional)


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

    percentiles_old = np.cumsum(score_dist_old) - 0.5 * score_dist_old
    percentiles_new = np.cumsum(score_dist_new) - 0.5 * score_dist_new
    new_scores = np.arange(len(score_dist_new), dtype=np.float64)

    unique_percentiles, inverse = np.unique(percentiles_new, return_inverse=True)
    if len(unique_percentiles) != len(percentiles_new):
        score_sums = np.bincount(inverse, weights=new_scores)
        counts = np.bincount(inverse)
        new_scores = score_sums / counts

    return np.interp(percentiles_old, unique_percentiles, new_scores)


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
    raw_probabilities = np.asarray(model.probability(theta[:, None]), dtype=np.float64)
    result: list[NDArray[np.float64]] = []

    if model.is_polytomous:
        category_counts = np.asarray(getattr(model, "n_categories"), dtype=np.intp)
        if (
            raw_probabilities.ndim != 3
            or raw_probabilities.shape[:2] != (len(theta), model.n_items)
            or category_counts.shape != (model.n_items,)
        ):
            raise ValueError("model returned invalid polytomous probabilities")
        for item_idx in item_indices:
            n_categories = int(category_counts[item_idx])
            probabilities = raw_probabilities[:, item_idx, :n_categories]
            result.append(_validate_item_probabilities(probabilities))
        return result

    if raw_probabilities.shape != (len(theta), model.n_items):
        raise ValueError("model returned invalid dichotomous probabilities")
    selected = raw_probabilities[:, item_indices]
    if not np.all(np.isfinite(selected)) or np.any(
        (selected < -_PROBABILITY_TOLERANCE) | (selected > 1.0 + _PROBABILITY_TOLERANCE)
    ):
        raise ValueError("model returned probabilities outside [0, 1]")
    selected = np.clip(selected, 0.0, 1.0)
    for column in range(selected.shape[1]):
        correct = selected[:, column]
        result.append(np.column_stack((1.0 - correct, correct)))
    return result


def _validate_item_probabilities(
    probabilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Validate a theta-by-category probability matrix."""
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("model returned an invalid category probability matrix")
    if not np.all(np.isfinite(probabilities)) or np.any(
        (probabilities < -_PROBABILITY_TOLERANCE)
        | (probabilities > 1.0 + _PROBABILITY_TOLERANCE)
    ):
        raise ValueError("model returned probabilities outside [0, 1]")

    result = np.clip(probabilities, 0.0, 1.0)
    row_sums = np.sum(result, axis=1, keepdims=True)
    if np.any(np.abs(row_sums - 1.0) > 1e-7):
        raise ValueError("model category probabilities must sum to one")
    return np.asarray(result / row_sums, dtype=np.float64)


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
    """Apply log-linear smoothing to score distribution."""
    n = len(dist)
    if n == 1:
        return dist.copy()
    scores = np.linspace(-1.0, 1.0, n)
    degree = min(degree, n - 1)

    dist = np.maximum(dist, 1e-10)
    log_dist = np.log(dist)

    design = np.vander(scores, N=degree + 1, increasing=True)

    try:
        coeffs = np.linalg.lstsq(design, log_dist, rcond=None)[0]
        fitted = design @ coeffs
        smoothed = np.exp(fitted - np.max(fitted))
    except np.linalg.LinAlgError:
        return dist

    return np.asarray(smoothed / np.sum(smoothed), dtype=np.float64)


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
    item_probabilities = _item_score_probabilities(model, theta, item_indices)
    expected = np.zeros(len(theta), dtype=np.float64)
    for probabilities in item_probabilities:
        scores = np.arange(probabilities.shape[1], dtype=np.float64)
        expected += probabilities @ scores
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
    _validate_model(model_old, "model_old")
    _validate_model(model_new, "model_new")
    theta_grid = _validate_vector(theta_grid, "theta_grid")
    old_item_indices = _resolve_items(model_old, items_old, "items_old")
    new_item_indices = _resolve_items(model_new, items_new, "items_new")

    variance_old = _conditional_score_variance(model_old, theta_grid, old_item_indices)
    variance_new = _conditional_score_variance(model_new, theta_grid, new_item_indices)
    return np.sqrt(np.maximum(variance_old + variance_new, 0.0))
