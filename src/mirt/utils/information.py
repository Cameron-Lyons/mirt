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
_GENERALIZED_DIFFICULTY_MODELS = frozenset({"1PL", "2PL", "3PL", "4PL", "5PL"})
_GENERALIZED_DIFFICULTY_GRID_POINTS = 65
_GENERALIZED_DIFFICULTY_GRID_ELEMENTS = 2_000_000
_GENERALIZED_DIFFICULTY_THETA_TOLERANCE = 1e-6


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


def _difficulty_item_indices(
    model: "BaseItemModel",
    item_idx: int | ArrayLike | None,
) -> tuple[NDArray[np.intp], bool, tuple[int, ...]]:
    """Validate requested generalized-difficulty item indices."""
    if item_idx is None:
        return np.arange(model.n_items, dtype=np.intp), False, (model.n_items,)
    if isinstance(item_idx, (bool, np.bool_)):
        raise ValueError("item_idx must contain integer item indices")
    if isinstance(item_idx, (int, np.integer)):
        index = int(item_idx)
        if index < 0 or index >= model.n_items:
            raise IndexError(f"item_idx {index} out of range [0, {model.n_items})")
        return np.array([index], dtype=np.intp), True, ()

    raw_indices = np.asarray(item_idx)
    if raw_indices.ndim == 0:
        raise ValueError("item_idx must contain integer item indices")
    if raw_indices.ndim != 1:
        raise ValueError("item_idx must be an integer or one-dimensional array")
    if raw_indices.size == 0:
        return np.empty(0, dtype=np.intp), False, raw_indices.shape
    if raw_indices.dtype.kind not in "iu":
        raise ValueError("item_idx must contain integer item indices")
    indices = raw_indices.astype(np.intp, copy=False)
    invalid = indices[(indices < 0) | (indices >= model.n_items)]
    if invalid.size:
        index = int(invalid[0])
        raise IndexError(f"item_idx {index} out of range [0, {model.n_items})")
    return indices, False, raw_indices.shape


def _difficulty_targets(
    target_prob: ArrayLike,
    output_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    """Broadcast and validate generalized-difficulty target probabilities."""
    try:
        targets = np.asarray(target_prob, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_prob must contain numeric values") from exc
    try:
        broadcast = np.broadcast_to(targets, output_shape).astype(np.float64, copy=True)
    except ValueError as exc:
        raise ValueError(
            f"target_prob must be scalar or broadcast to shape {output_shape}"
        ) from exc
    if not np.all(np.isfinite(broadcast)) or np.any(
        (broadcast <= 0.0) | (broadcast >= 1.0)
    ):
        raise ValueError(
            "target_prob must contain finite values strictly between 0 and 1"
        )
    return broadcast.reshape(-1)


def _difficulty_theta_bounds(theta_range: tuple[float, float]) -> tuple[float, float]:
    """Validate generalized-difficulty theta bounds."""
    try:
        bounds = np.asarray(theta_range, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("theta_range must contain exactly two numeric bounds") from exc
    if bounds.shape != (2,):
        raise ValueError("theta_range must contain exactly two numeric bounds")
    lower, upper = float(bounds[0]), float(bounds[1])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("theta_range must contain finite bounds with lower < upper")
    return lower, upper


def _logistic_difficulties(
    model: "BaseItemModel",
    indices: NDArray[np.intp],
    targets: NDArray[np.float64],
    lower_bound: float,
    upper_bound: float,
) -> NDArray[np.float64] | None:
    """Invert standard logistic-family item curves analytically."""
    if getattr(model, "model_name", None) not in _GENERALIZED_DIFFICULTY_MODELS:
        return None

    parameters = model.parameters
    discrimination = np.asarray(parameters.get("discrimination"), dtype=np.float64)
    difficulty = np.asarray(parameters.get("difficulty"), dtype=np.float64)
    expected_shape = (model.n_items,)
    if discrimination.shape != expected_shape or difficulty.shape != expected_shape:
        raise ValueError(
            "generalized difficulty requires one discrimination and difficulty per item"
        )

    lower = np.asarray(
        parameters.get("guessing", np.zeros(model.n_items)), dtype=np.float64
    )
    upper = np.asarray(
        parameters.get("upper", np.ones(model.n_items)), dtype=np.float64
    )
    asymmetry = np.asarray(
        parameters.get("asymmetry", np.ones(model.n_items)), dtype=np.float64
    )
    if lower.shape != expected_shape or upper.shape != expected_shape:
        raise ValueError("item asymptotes must contain one value per item")
    if asymmetry.shape != expected_shape:
        raise ValueError("item asymmetry must contain one value per item")
    if (
        not np.all(np.isfinite(discrimination))
        or not np.all(np.isfinite(difficulty))
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or not np.all(np.isfinite(asymmetry))
        or np.any(discrimination <= 0.0)
        or np.any(asymmetry <= 0.0)
        or np.any(lower < 0.0)
        or np.any(upper > 1.0)
        or np.any(lower >= upper)
    ):
        raise ValueError("model parameters do not define valid monotonic item curves")

    selected_discrimination = discrimination[indices]
    selected_difficulty = difficulty[indices]
    selected_lower = lower[indices]
    selected_upper = upper[indices]
    selected_asymmetry = asymmetry[indices]
    result = np.empty(indices.size, dtype=np.float64)

    below = targets <= selected_lower
    above = targets >= selected_upper
    result[below] = lower_bound
    result[above] = upper_bound
    interior = ~(below | above)
    if np.any(interior):
        scaled = (targets[interior] - selected_lower[interior]) / (
            selected_upper[interior] - selected_lower[interior]
        )
        logistic = scaled ** (1.0 / selected_asymmetry[interior])
        logit = np.log(logistic) - np.log1p(-logistic)
        result[interior] = selected_difficulty[interior] + (
            logit / selected_discrimination[interior]
        )
    return np.clip(result, lower_bound, upper_bound)


def _difficulty_probability_curve(
    model: "BaseItemModel",
    theta: NDArray[np.float64],
    item_index: int,
) -> NDArray[np.float64]:
    """Evaluate and validate one dichotomous probability curve."""
    probabilities = np.asarray(
        model.probability(theta[:, None], item_idx=item_index), dtype=np.float64
    )
    if probabilities.size != theta.size:
        raise ValueError(
            "model must return one dichotomous probability per theta and item"
        )
    curve = probabilities.reshape(-1)
    tolerance = 1e-10
    if not np.all(np.isfinite(curve)) or np.any(
        (curve < -tolerance) | (curve > 1.0 + tolerance)
    ):
        raise ValueError("model probabilities must be finite and lie in [0, 1]")
    return np.clip(curve, 0.0, 1.0)


def _difficulty_probability_curves(
    model: "BaseItemModel",
    theta: NDArray[np.float64],
    indices: NDArray[np.intp],
) -> NDArray[np.float64]:
    """Evaluate selected custom-model curves with bounded temporary memory."""
    if theta.size * model.n_items <= _GENERALIZED_DIFFICULTY_GRID_ELEMENTS:
        probabilities = np.asarray(model.probability(theta[:, None]), dtype=np.float64)
        expected_shape = (theta.size, model.n_items)
        if probabilities.shape == expected_shape:
            curves = probabilities[:, indices]
            tolerance = 1e-10
            if not np.all(np.isfinite(curves)) or np.any(
                (curves < -tolerance) | (curves > 1.0 + tolerance)
            ):
                raise ValueError("model probabilities must be finite and lie in [0, 1]")
            return np.clip(curves, 0.0, 1.0)

    return np.column_stack(
        [
            _difficulty_probability_curve(model, theta, int(item_index))
            for item_index in indices
        ]
    )


def _numerical_difficulties(
    model: "BaseItemModel",
    indices: NDArray[np.intp],
    targets: NDArray[np.float64],
    lower_bound: float,
    upper_bound: float,
) -> NDArray[np.float64]:
    """Invert custom monotonic dichotomous curves numerically."""
    from scipy.optimize import brentq

    theta_grid = np.linspace(
        lower_bound, upper_bound, _GENERALIZED_DIFFICULTY_GRID_POINTS
    )
    curves = _difficulty_probability_curves(model, theta_grid, indices)
    differences = np.diff(curves, axis=0)
    scale = np.maximum(1.0, np.max(np.abs(curves), axis=0))
    tolerance = np.sqrt(np.finfo(np.float64).eps) * scale
    increasing = curves[-1] >= curves[0]
    monotonic = np.where(
        increasing,
        np.all(differences >= -tolerance[None, :], axis=0),
        np.all(differences <= tolerance[None, :], axis=0),
    )
    monotonic &= np.abs(curves[-1] - curves[0]) > tolerance
    if not np.all(monotonic):
        invalid = indices[~monotonic].tolist()
        raise ValueError(
            f"item probability curves must be monotonic and vary over theta_range; "
            f"invalid item indices: {invalid}"
        )

    first = curves[0]
    last = curves[-1]
    minimum = np.minimum(first, last)
    maximum = np.maximum(first, last)
    minimum_theta = np.where(increasing, lower_bound, upper_bound)
    maximum_theta = np.where(increasing, upper_bound, lower_bound)
    result = np.empty(indices.size, dtype=np.float64)
    below = targets <= minimum
    above = targets >= maximum
    result[below] = minimum_theta[below]
    result[above] = maximum_theta[above]

    for position in np.flatnonzero(~(below | above)):
        item_index = int(indices[position])
        target = float(targets[position])

        def objective(theta_value: float) -> float:
            probability = _difficulty_probability_curve(
                model, np.array([theta_value]), item_index
            )[0]
            return float(probability - target)

        result[position] = brentq(
            objective,
            lower_bound,
            upper_bound,
            xtol=_GENERALIZED_DIFFICULTY_THETA_TOLERANCE,
        )
    return result


def gen_difficulty(
    model: "BaseItemModel",
    item_idx: int | ArrayLike | None = None,
    target_prob: ArrayLike = 0.5,
    theta_range: tuple[float, float] = (-10.0, 10.0),
) -> NDArray[np.float64] | float:
    """Compute generalized difficulty (theta where P(X=1) = target_prob).

    For dichotomous items, this finds the theta value where the
    probability of a correct response equals target_prob.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    item_idx : int, array-like, or None
        Item index or one-dimensional item selection. If None, returns all items.
    target_prob : array-like
        Target probability in ``(0, 1)``. A scalar applies to every selected
        item; an array may provide one target per selected item.
    theta_range : tuple of float
        Finite search bounds for custom models and asymptote clamping.

    Returns
    -------
    NDArray[np.float64] or float
        Generalized difficulty value(s). A scalar item selection returns a
        float; other selections retain their one-dimensional shape.

    Examples
    --------
    >>> # Find theta where P(correct) = 0.5
    >>> b = gen_difficulty(result.model, item_idx=0)
    >>> print(f"Item 0 difficulty: {b:.3f}")
    >>> # Find theta where P(correct) = 0.8
    >>> b80 = gen_difficulty(result.model, item_idx=0, target_prob=0.8)
    >>> print(f"Theta for 80% correct: {b80:.3f}")

    Notes
    -----
    Standard 1PL through 5PL curves are inverted analytically. Other
    unidimensional dichotomous models use a validated numerical inversion and
    must be monotonic and vary over ``theta_range``. Polytomous category
    probabilities are not a single monotonic item-response curve; use
    ``expected_score`` or ``theta_for_score`` for those models.
    """
    if getattr(model, "is_polytomous", False):
        raise ValueError(
            "gen_difficulty supports dichotomous models only; use "
            "theta_for_score for polytomous expected scores"
        )
    if model.n_factors != 1:
        raise ValueError("gen_difficulty supports unidimensional models only")

    indices, scalar_item, output_shape = _difficulty_item_indices(model, item_idx)
    targets = _difficulty_targets(target_prob, output_shape)
    lower_bound, upper_bound = _difficulty_theta_bounds(theta_range)
    if indices.size == 0:
        return np.empty(output_shape, dtype=np.float64)

    difficulties = _logistic_difficulties(
        model,
        indices,
        targets,
        lower_bound,
        upper_bound,
    )
    if difficulties is None:
        difficulties = _numerical_difficulties(
            model,
            indices,
            targets,
            lower_bound,
            upper_bound,
        )

    shaped = difficulties.reshape(output_shape)
    return float(shaped) if scalar_item else shaped


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
