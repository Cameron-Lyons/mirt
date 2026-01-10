"""Information functions for IRT models.

Provides functions for computing test and item information,
area under information curves, and probability traces.
"""

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


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
    theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    if theta_arr.ndim == 1:
        theta_arr = theta_arr.reshape(-1, 1)

    item_info = model.information(theta_arr)
    return np.sum(item_info, axis=1)


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
    theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    if theta_arr.ndim == 1:
        theta_arr = theta_arr.reshape(-1, 1)

    all_info = model.information(theta_arr)

    if item_idx is None:
        return all_info

    if isinstance(item_idx, int):
        return all_info[:, item_idx]

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
    theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    if theta_arr.ndim == 1:
        theta_arr = theta_arr.reshape(-1, 1)

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
    theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    if theta_arr.ndim == 1:
        theta_arr = theta_arr.reshape(-1, 1)

    if hasattr(model, "expected_score"):
        scores = model.expected_score(theta_arr)
    else:
        scores = model.probability(theta_arr)

    if item_idx is None:
        return np.sum(scores, axis=1) if scores.ndim > 1 else scores

    if isinstance(item_idx, int):
        return scores[:, item_idx] if scores.ndim > 1 else scores

    return scores[:, item_idx] if scores.ndim > 1 else scores


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


def theta_for_score(
    model: "BaseItemModel",
    target_score: float,
    theta_range: tuple[float, float] = (-6.0, 6.0),
) -> float:
    """Find theta value corresponding to a target expected score.

    Inverts the test characteristic curve.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    target_score : float
        Target expected score.
    theta_range : tuple
        Range to search. Default (-6, 6).

    Returns
    -------
    float
        Theta value where expected score equals target.

    Examples
    --------
    >>> # Find theta where expected score is 10
    >>> theta = theta_for_score(result.model, target_score=10)
    >>> print(f"Theta for score=10: {theta:.3f}")
    """
    from scipy.optimize import brentq

    def objective(theta):
        scores = expected_score(model, np.array([theta]))
        return scores[0] - target_score

    try:
        return brentq(objective, theta_range[0], theta_range[1], xtol=1e-6)
    except ValueError:
        theta_grid = np.linspace(theta_range[0], theta_range[1], 100)
        scores = expected_score(model, theta_grid)
        idx = np.argmin(np.abs(scores - target_score))
        return float(theta_grid[idx])
