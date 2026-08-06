"""Reliability functions for IRT models."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


Density = Literal["norm", "uniform"] | Callable[[NDArray[np.float64]], ArrayLike]
Reliability = float | NDArray[np.float64]


def _theta_array(
    model: "BaseItemModel",
    theta: ArrayLike,
) -> NDArray[np.float64]:
    """Normalize theta without confusing factors with respondents."""
    values = np.asarray(theta, dtype=np.float64)
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        if model.n_factors == 1:
            values = values.reshape(-1, 1)
        elif values.size == model.n_factors:
            values = values.reshape(1, -1)
        else:
            raise ValueError(
                f"theta must have {model.n_factors} columns for this model"
            )

    if values.ndim != 2:
        raise ValueError("theta must be a scalar, a one-dimensional array, or a matrix")
    if values.shape[0] == 0:
        raise ValueError("theta must contain at least one estimate")
    if values.shape[1] != model.n_factors:
        raise ValueError(
            f"theta has {values.shape[1]} factors, expected {model.n_factors}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("theta must contain only finite values")
    return values


def _test_information(
    model: "BaseItemModel",
    theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return total information for models with item or test-level output."""
    information = np.asarray(model.information(theta), dtype=np.float64)
    if information.ndim == 1:
        test_information = information
    elif information.ndim == 2:
        test_information = np.sum(information, axis=1)
    else:
        raise ValueError(
            "model.information() must return test information or item information"
        )

    if test_information.shape != (theta.shape[0],):
        raise ValueError(
            "model.information() returned an incompatible number of theta points"
        )
    if not np.all(np.isfinite(test_information)):
        raise ValueError("model information must contain only finite values")
    if np.any(test_information < 0.0):
        raise ValueError("model information must be non-negative")
    return np.maximum(test_information, PROB_EPSILON)


def _quadrature_weights(
    theta: NDArray[np.float64],
    density: Density,
) -> NDArray[np.float64]:
    if isinstance(density, str) and density == "norm":
        density_values = np.exp(-0.5 * theta**2) / np.sqrt(2.0 * np.pi)
    elif isinstance(density, str) and density == "uniform":
        density_values = np.ones_like(theta)
    elif callable(density):
        density_values = np.asarray(density(theta), dtype=np.float64)
        try:
            density_values = np.broadcast_to(density_values, theta.shape).copy()
        except ValueError as exc:
            raise ValueError("density must return one weight per theta point") from exc
    else:
        raise ValueError("density must be 'norm', 'uniform', or a callable")

    if not np.all(np.isfinite(density_values)):
        raise ValueError("density must return only finite weights")
    if np.any(density_values < 0.0):
        raise ValueError("density weights must be non-negative")

    # The points are evenly spaced, so the common interval width cancels
    # during normalization. Half-weighting the endpoints gives trapezoidal
    # integration without allocating another grid-sized array.
    density_values[[0, -1]] *= 0.5
    total = float(np.sum(density_values))
    if total <= 0.0:
        raise ValueError("density weights must have a positive sum")
    return density_values / total


def marginal_rxx(
    model: "BaseItemModel",
    theta_range: tuple[float, float] = (-6.0, 6.0),
    n_points: int = 61,
    density: Density = "norm",
) -> float:
    """Compute marginal reliability over an ability distribution.

    The coefficient is the density-weighted mean of
    ``I(theta) / (I(theta) + 1 / Var(theta))``, where ``I(theta)`` is test
    information. This function supports both dichotomous and polytomous
    unidimensional models.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    theta_range : tuple of float, default=(-6, 6)
        Finite lower and upper integration bounds.
    n_points : int, default=61
        Number of evenly spaced quadrature points. Must be at least 2.
    density : {"norm", "uniform"} or callable, default="norm"
        Ability density. A callable receives the theta grid and returns
        non-negative weights.

    Returns
    -------
    float
        Marginal reliability, clipped to the interval [0, 1].

    Raises
    ------
    ValueError
        If the model is multidimensional or an argument is invalid.
    """
    if model.n_factors != 1:
        raise ValueError("marginal_rxx supports unidimensional models only")
    if isinstance(n_points, bool) or not isinstance(n_points, int) or n_points < 2:
        raise ValueError("n_points must be an integer greater than or equal to 2")

    try:
        lower, upper = (float(value) for value in theta_range)
    except (TypeError, ValueError) as exc:
        raise ValueError("theta_range must contain exactly two numeric bounds") from exc
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("theta_range must contain finite bounds with lower < upper")

    theta = np.linspace(lower, upper, n_points)
    weights = _quadrature_weights(theta, density)
    test_information = _test_information(model, theta.reshape(-1, 1))

    mean_theta = float(np.sum(weights * theta))
    theta_variance = float(np.sum(weights * (theta - mean_theta) ** 2))
    if theta_variance <= PROB_EPSILON:
        raise ValueError("the weighted theta distribution must have positive variance")

    local_reliability = test_information / (test_information + 1.0 / theta_variance)
    reliability = float(np.sum(weights * local_reliability))
    return float(np.clip(reliability, 0.0, 1.0))


def empirical_rxx(
    model: "BaseItemModel",
    theta_estimates: ArrayLike,
    method: Literal["posterior_variance", "information"] = "information",
    standard_errors: ArrayLike | None = None,
) -> Reliability:
    """Compute empirical reliability from ability estimates.

    Reliability is ``Var(theta) / (Var(theta) + E[SE**2])``. Information-
    based estimates derive the standard errors from test information and
    currently require a unidimensional model. Posterior-variance estimates
    use standard errors returned by a scoring method and support any number
    of factors.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    theta_estimates : array-like
        Ability estimates with shape ``(n_persons,)`` or
        ``(n_persons, n_factors)``.
    method : {"posterior_variance", "information"}, default="information"
        Source of each respondent's error variance.
    standard_errors : array-like, optional
        Score standard errors matching ``theta_estimates``. Required when
        ``method="posterior_variance"``.

    Returns
    -------
    float or ndarray
        One coefficient for a unidimensional model, or one coefficient per
        factor for a multidimensional posterior-variance estimate.
    """
    if method not in {"posterior_variance", "information"}:
        raise ValueError("method must be 'posterior_variance' or 'information'")

    theta = _theta_array(model, theta_estimates)
    if theta.shape[0] < 2:
        raise ValueError("theta_estimates must contain at least two respondents")

    observed_variance = np.var(theta, axis=0, ddof=1)

    if method == "posterior_variance":
        if standard_errors is None:
            raise ValueError(
                "standard_errors are required when method='posterior_variance'"
            )
        errors = np.asarray(standard_errors, dtype=np.float64)
        if errors.ndim == 1:
            errors = errors.reshape(-1, 1)
        if errors.shape != theta.shape:
            raise ValueError(
                f"standard_errors has shape {errors.shape}, expected {theta.shape}"
            )
        if not np.all(np.isfinite(errors)) or np.any(errors < 0.0):
            raise ValueError("standard_errors must contain finite, non-negative values")
        average_error_variance = np.mean(errors**2, axis=0)
    else:
        if standard_errors is not None:
            raise ValueError("standard_errors can only be used with posterior_variance")
        if model.n_factors != 1:
            raise ValueError(
                "information reliability supports unidimensional models only; "
                "use posterior_variance with factor-specific standard_errors"
            )
        test_information = _test_information(model, theta)
        average_error_variance = np.array([np.mean(1.0 / test_information)])

    denominator = observed_variance + average_error_variance
    reliability = np.divide(
        observed_variance,
        denominator,
        out=np.zeros_like(observed_variance),
        where=denominator > PROB_EPSILON,
    )
    reliability = np.clip(reliability, 0.0, 1.0)
    if model.n_factors == 1:
        return float(reliability[0])
    return reliability


def sem(
    model: "BaseItemModel",
    theta: ArrayLike,
) -> NDArray[np.float64]:
    """Compute the standard error of measurement at given theta values.

    The standard error is ``1 / sqrt(I(theta))``. Models may return either
    total test information or an item-information matrix; both conventions
    are supported.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    theta : array-like
        Ability values at which to compute the standard error.

    Returns
    -------
    ndarray
        One standard error for each theta point.
    """
    theta_array = _theta_array(model, theta)
    return 1.0 / np.sqrt(_test_information(model, theta_array))
