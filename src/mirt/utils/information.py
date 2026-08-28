"""Information functions for IRT models.

Provides functions for computing test and item information,
area under information curves, and probability traces.
"""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


_SCORE_INVERSION_TARGET_ELEMENTS = 2_000_000
_SCORE_INVERSION_THETA_TOLERANCE = 1e-6
_SCORE_MONOTONICITY_POINTS = 65


def _theta_matrix(
    theta: NDArray[np.float64] | float | list[float],
) -> NDArray[np.float64]:
    """Normalize scalar and one-dimensional ability inputs."""
    theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    if theta_arr.ndim == 1:
        theta_arr = theta_arr.reshape(-1, 1)
    return theta_arr


def _item_information_matrix(
    model: "BaseItemModel",
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return item-level information regardless of model-family convention."""
    if getattr(model, "is_polytomous", False):
        return np.column_stack(
            [model.information(theta, item_idx=idx) for idx in range(model.n_items)]
        )

    information = np.asarray(model.information(theta), dtype=np.float64)
    expected_shape = (theta.shape[0], model.n_items)
    if information.shape == expected_shape:
        return information

    if information.shape == (theta.shape[0],):
        return np.column_stack(
            [model.information(theta, item_idx=idx) for idx in range(model.n_items)]
        )

    raise ValueError(
        f"model information has shape {information.shape}, expected "
        f"{expected_shape} or {(theta.shape[0],)}"
    )


def testinfo(
    model: "BaseItemModel",
    theta: NDArray[np.float64] | float | list[float],
) -> NDArray[np.float64]:
    """Compute test information function at given theta values.

    The test information is the sum of item information across all items.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    theta : array-like
        Ability values at which to compute information.
        Can be a scalar, list, or numpy array.

    Returns
    -------
    NDArray[np.float64]
        Test information values at each theta point.
        Shape: (n_theta,) for unidimensional models.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL")
    >>> theta = np.linspace(-3, 3, 61)
    >>> info = testinfo(result.model, theta)
    >>> print(f"Max information at theta = {theta[np.argmax(info)]:.2f}")
    """
    theta_arr = _theta_matrix(theta)
    information = np.asarray(model.information(theta_arr), dtype=np.float64)

    if information.shape == (theta_arr.shape[0],):
        return information
    if information.shape == (theta_arr.shape[0], model.n_items):
        return np.sum(information, axis=1)

    raise ValueError(
        f"model information has shape {information.shape}, expected "
        f"{(theta_arr.shape[0], model.n_items)} or {(theta_arr.shape[0],)}"
    )


def iteminfo(
    model: "BaseItemModel",
    theta: NDArray[np.float64] | float | list[float],
    item_idx: int | list[int] | None = None,
) -> NDArray[np.float64]:
    """Compute item information function at given theta values.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    theta : array-like
        Ability values at which to compute information.
    item_idx : int, list of int, or None
        Index or indices of items. If None, returns information for all items.

    Returns
    -------
    NDArray[np.float64]
        Item information values.
        Shape: (n_theta,) if single item, (n_theta, n_items) otherwise.

    Examples
    --------
    >>> info = iteminfo(result.model, theta=0.0, item_idx=0)
    >>> print(f"Item 0 information at theta=0: {info[0]:.3f}")
    """
    theta_arr = _theta_matrix(theta)

    if item_idx is None:
        return _item_information_matrix(model, theta_arr)

    if isinstance(item_idx, int):
        return np.asarray(
            model.information(theta_arr, item_idx=item_idx), dtype=np.float64
        )

    all_info = _item_information_matrix(model, theta_arr)
    return all_info[:, item_idx]


def areainfo(
    model: "BaseItemModel",
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_points: int = 100,
    item_idx: int | None = None,
) -> float:
    """Compute area under the information curve.

    Integrates the information function over a range of theta values
    using the trapezoidal rule.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    theta_range : tuple of float
        Range of theta values for integration. Default (-4, 4).
    n_points : int
        Number of quadrature points. Default 100.
    item_idx : int or None
        If provided, compute area for specific item.
        If None, compute area for test information.

    Returns
    -------
    float
        Area under the information curve.

    Examples
    --------
    >>> area = areainfo(result.model)
    >>> print(f"Total test information area: {area:.2f}")
    >>> item_area = areainfo(result.model, item_idx=0)
    >>> print(f"Item 0 information area: {item_area:.2f}")
    """
    theta = np.linspace(theta_range[0], theta_range[1], n_points)

    if item_idx is not None:
        info = iteminfo(model, theta, item_idx)
    else:
        info = testinfo(model, theta)

    return float(np.trapezoid(info, theta))


def probtrace(
    model: "BaseItemModel",
    theta: NDArray[np.float64] | float | list[float],
    item_idx: int | None = None,
) -> NDArray[np.float64]:
    """Compute probability traces (category response functions).

    For dichotomous items, returns P(X=1|theta).
    For polytomous items, returns P(X=k|theta) for each category k.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    theta : array-like
        Ability values at which to compute probabilities.
    item_idx : int or None
        Index of item. If None, returns traces for all items.

    Returns
    -------
    NDArray[np.float64]
        Probability traces.
        For dichotomous: shape (n_theta, n_items) or (n_theta,)
        For polytomous: shape (n_theta, n_items, n_categories) or (n_theta, n_categories)

    Examples
    --------
    >>> theta = np.linspace(-3, 3, 61)
    >>> traces = probtrace(result.model, theta, item_idx=0)
    >>> # For 2PL: traces has shape (61,) - probability of correct response
    """
    theta_arr = _theta_matrix(theta)

    probs = model.probability(theta_arr, item_idx=item_idx)
    return probs


def expected_score(
    model: "BaseItemModel",
    theta: NDArray[np.float64] | float | list[float],
    item_idx: int | list[int] | None = None,
) -> NDArray[np.float64]:
    """Compute expected score at given theta values.

    For dichotomous items, this equals the probability of correct response.
    For polytomous items, this is the expected category score.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    theta : array-like
        Ability values.
    item_idx : int, list of int, or None
        Item index or indices. If None, returns expected test score.

    Returns
    -------
    NDArray[np.float64]
        Expected scores at each theta point.

    Examples
    --------
    >>> theta = np.array([[-2], [0], [2]])
    >>> expected = expected_score(result.model, theta)
    >>> print(f"Expected test score at theta=0: {expected[1]:.2f}")
    """
    theta_arr = _theta_matrix(theta)

    def score_one(index: int | None) -> NDArray[np.float64]:
        score_method = getattr(model, "expected_score", None)
        if callable(score_method):
            return np.asarray(score_method(theta_arr, item_idx=index), dtype=np.float64)

        probabilities = np.asarray(
            model.probability(theta_arr, item_idx=index), dtype=np.float64
        )
        if getattr(model, "is_polytomous", False):
            category_scores = np.arange(probabilities.shape[-1])
            scores = (
                probabilities @ category_scores
                if index is not None
                else np.sum(probabilities * category_scores, axis=-1)
            )
        else:
            scores = probabilities

        return np.sum(scores, axis=1) if index is None and scores.ndim > 1 else scores

    if item_idx is None or isinstance(item_idx, int):
        return score_one(item_idx)

    if not item_idx:
        return np.empty((theta_arr.shape[0], 0), dtype=np.float64)
    return np.column_stack([score_one(index) for index in item_idx])


def gen_difficulty(
    model: "BaseItemModel",
    item_idx: int | None = None,
    target_prob: float = 0.5,
) -> NDArray[np.float64] | float:
    """Compute generalized difficulty (theta where P(X=1) = target_prob).

    For dichotomous items, this finds the theta value where the
    probability of a correct response equals target_prob.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    item_idx : int or None
        Item index. If None, returns for all items.
    target_prob : float
        Target probability. Default 0.5 (traditional difficulty).

    Returns
    -------
    NDArray[np.float64] or float
        Generalized difficulty value(s).

    Examples
    --------
    >>> # Find theta where P(correct) = 0.5
    >>> b = gen_difficulty(result.model, item_idx=0)
    >>> print(f"Item 0 difficulty: {b:.3f}")
    >>> # Find theta where P(correct) = 0.8
    >>> b80 = gen_difficulty(result.model, item_idx=0, target_prob=0.8)
    >>> print(f"Theta for 80% correct: {b80:.3f}")
    """
    from scipy.optimize import brentq

    def find_theta_for_item(j: int) -> float:
        def objective(theta):
            theta_2d = np.array([[theta]])
            prob = model.probability(theta_2d, item_idx=j)
            if prob.ndim > 1:
                prob = prob[0, 0] if prob.shape[1] > 0 else prob[0]
            else:
                prob = prob[0]
            return prob - target_prob

        try:
            return brentq(objective, -10, 10, xtol=1e-6)
        except ValueError:
            if objective(-10) > 0:
                return -10.0
            elif objective(10) < 0:
                return 10.0
            else:
                return 0.0

    if item_idx is not None:
        return find_theta_for_item(item_idx)

    n_items = model.n_items
    difficulties = np.zeros(n_items)
    for j in range(n_items):
        difficulties[j] = find_theta_for_item(j)

    return difficulties


def expected_test_score(
    model: "BaseItemModel",
    theta: NDArray[np.float64] | float | list[float],
) -> NDArray[np.float64]:
    """Compute expected total test score at given theta values.

    This is the sum of expected item scores across all items,
    representing the test characteristic curve (TCC).

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    theta : array-like
        Ability values.

    Returns
    -------
    NDArray[np.float64]
        Expected total scores at each theta point.

    Examples
    --------
    >>> theta = np.linspace(-3, 3, 61)
    >>> tcc = expected_test_score(result.model, theta)
    >>> plt.plot(theta, tcc)
    >>> plt.xlabel("Theta")
    >>> plt.ylabel("Expected Score")
    """
    return expected_score(model, theta, item_idx=None)


def _validated_expected_scores(
    model: "BaseItemModel", theta: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Evaluate a test curve and enforce one finite score per theta value."""
    scores = np.asarray(expected_score(model, theta), dtype=np.float64)
    if scores.shape != theta.shape or not np.all(np.isfinite(scores)):
        raise ValueError(
            "model expected scores must be finite with one value per theta"
        )
    return scores


def _score_curve_direction(
    model: "BaseItemModel", lower: float, upper: float
) -> tuple[bool, float, float]:
    """Validate the inversion interval and return its score direction."""
    theta_grid = np.linspace(lower, upper, _SCORE_MONOTONICITY_POINTS)
    score_grid = _validated_expected_scores(model, theta_grid)
    differences = np.diff(score_grid)
    tolerance = np.sqrt(np.finfo(np.float64).eps) * max(
        1.0, float(np.max(np.abs(score_grid)))
    )
    increasing = bool(score_grid[-1] >= score_grid[0])
    monotonic = (
        np.all(differences >= -tolerance)
        if increasing
        else np.all(differences <= tolerance)
    )
    if not monotonic:
        raise ValueError("test characteristic curve must be monotonic over theta_range")
    if abs(float(score_grid[-1] - score_grid[0])) <= tolerance:
        raise ValueError("test characteristic curve must vary over theta_range")
    return increasing, float(score_grid[0]), float(score_grid[-1])


def _score_inversion_chunk_size(model: "BaseItemModel") -> int:
    """Choose a chunk that bounds the largest temporary probability tensor."""
    elements_per_target = max(1, int(model.n_items))
    if getattr(model, "is_polytomous", False):
        category_counts = np.asarray(model.n_categories)
        if category_counts.size:
            elements_per_target *= max(1, int(np.max(category_counts)))
    return max(1, _SCORE_INVERSION_TARGET_ELEMENTS // elements_per_target)


def _invert_score_chunk(
    model: "BaseItemModel",
    targets: NDArray[np.float64],
    lower: float,
    upper: float,
    *,
    increasing: bool,
    n_iterations: int,
) -> NDArray[np.float64]:
    """Invert one target chunk with simultaneous bisection."""
    low = np.full(targets.size, lower)
    high = np.full(targets.size, upper)
    for _ in range(n_iterations):
        midpoint = (low + high) * 0.5
        midpoint_scores = _validated_expected_scores(model, midpoint)
        move_lower = (
            midpoint_scores < targets if increasing else midpoint_scores > targets
        )
        low[move_lower] = midpoint[move_lower]
        high[~move_lower] = midpoint[~move_lower]
    return (low + high) * 0.5


def theta_for_score(
    model: "BaseItemModel",
    target_score: ArrayLike,
    theta_range: tuple[float, float] = (-6.0, 6.0),
) -> NDArray[np.float64] | float:
    """Find theta values corresponding to target expected scores.

    Inverts a monotonic test characteristic curve using bounded, vectorized
    bisection. Scalar inputs preserve the historical scalar return type, while
    arrays are inverted in memory-aware chunks.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    target_score : array-like
        Target expected score or scores. Array inputs retain their shape.
    theta_range : tuple
        Range to search. Default (-6, 6).

    Returns
    -------
    float or NDArray[np.float64]
        Theta value or values where the expected score equals each target.
        Targets outside the score range are clamped to the corresponding
        theta bound.

    Examples
    --------
    >>> # Find theta where expected score is 10
    >>> theta = theta_for_score(result.model, target_score=10)
    >>> print(f"Theta for score=10: {theta:.3f}")
    >>> targets = np.array([5.0, 10.0, 15.0])
    >>> theta = theta_for_score(result.model, targets)
    """
    if model.n_factors != 1:
        raise ValueError("theta_for_score supports unidimensional models only")

    try:
        lower, upper = (float(value) for value in theta_range)
    except (TypeError, ValueError) as exc:
        raise ValueError("theta_range must contain exactly two numeric bounds") from exc
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("theta_range must contain finite bounds with lower < upper")

    try:
        targets = np.asarray(target_score, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_score must contain numeric values") from exc
    if not np.all(np.isfinite(targets)):
        raise ValueError("target_score must contain only finite values")

    scalar_input = targets.ndim == 0
    target_shape = targets.shape
    flat_targets = targets.reshape(-1)
    if flat_targets.size == 0:
        return np.empty(target_shape, dtype=np.float64)

    increasing, lower_score, upper_score = _score_curve_direction(model, lower, upper)
    minimum_score = min(lower_score, upper_score)
    maximum_score = max(lower_score, upper_score)
    minimum_theta = lower if increasing else upper
    maximum_theta = upper if increasing else lower

    result = np.empty_like(flat_targets)
    below_range = flat_targets <= minimum_score
    above_range = flat_targets >= maximum_score
    result[below_range] = minimum_theta
    result[above_range] = maximum_theta

    interior_indices = np.flatnonzero(~(below_range | above_range))
    if interior_indices.size:
        chunk_size = _score_inversion_chunk_size(model)
        n_iterations = max(
            1,
            int(np.ceil(np.log2((upper - lower) / _SCORE_INVERSION_THETA_TOLERANCE))),
        )

        for start in range(0, interior_indices.size, chunk_size):
            indices = interior_indices[start : start + chunk_size]
            result[indices] = _invert_score_chunk(
                model,
                flat_targets[indices],
                lower,
                upper,
                increasing=increasing,
                n_iterations=n_iterations,
            )

    shaped_result = result.reshape(target_shape)
    return float(shaped_result) if scalar_input else shaped_result
