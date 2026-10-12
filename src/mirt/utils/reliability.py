"""Reliability functions for IRT models."""

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

from mirt._information import _resolve_model, _test_information, _theta_array
from mirt.exceptions import MirtValidationError

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult
    from mirt.results.score_result import ScoreResult


Density = Literal["norm", "uniform"] | Callable[[NDArray[np.float64]], ArrayLike]
Reliability = float | NDArray[np.float64]


def _centered_theta(theta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Remove a finite midpoint before scaling, preserving small score differences."""
    midpoint = 0.5 * np.min(theta, axis=0) + 0.5 * np.max(theta, axis=0)
    return theta - midpoint


def _conditional_reliability(
    test_information: NDArray[np.float64],
    latent_variance: float,
) -> NDArray[np.float64]:
    """Convert test information to pointwise reliability."""
    with np.errstate(over="ignore"):
        scaled_information = test_information * latent_variance
    reliability = np.ones_like(scaled_information)
    finite = np.isfinite(scaled_information)
    np.divide(
        scaled_information,
        1.0 + scaled_information,
        out=reliability,
        where=finite,
    )
    return np.clip(reliability, 0.0, 1.0)


def conditional_rxx(
    model: "BaseItemModel | FitResult",
    theta: ArrayLike,
    latent_variance: float | None = None,
) -> NDArray[np.float64]:
    """Compute reliability at each supplied ability value.

    Conditional reliability is ``I(theta) * Var(theta) / (1 + I(theta) *
    Var(theta))``, where ``I(theta)`` is total test information. The result
    is vectorized over ability values and supports both dichotomous and
    polytomous unidimensional models.

    Parameters
    ----------
    model : BaseItemModel or FitResult
        A fitted unidimensional IRT model, or the ``FitResult`` of a fit.
    theta : array-like
        Ability values at which to compute reliability.
    latent_variance : float, optional
        Positive finite variance of the reference ability distribution.
        Defaults to the estimated ``latent_covariance`` of a ``FitResult``
        when it has one, and to one otherwise.

    Returns
    -------
    ndarray
        One reliability coefficient in the interval [0, 1] for each theta
        point. Points with zero test information have zero reliability.

    Raises
    ------
    ValueError
        If the model is multidimensional or an argument is invalid.
    """
    from mirt.results._common import resolve_latent_prior

    model, _, latent_cov = resolve_latent_prior(model)
    model = _resolve_model(model)
    if model.n_factors != 1:
        raise ValueError("conditional_rxx supports unidimensional models only")
    if latent_variance is None:
        latent_variance = 1.0 if latent_cov is None else float(latent_cov[0, 0])
    if isinstance(latent_variance, (bool, np.bool_)) or np.ndim(latent_variance) != 0:
        raise ValueError("latent_variance must be a positive finite scalar")
    try:
        variance = float(latent_variance)
    except (TypeError, ValueError) as exc:
        raise ValueError("latent_variance must be a positive finite scalar") from exc
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("latent_variance must be a positive finite scalar")

    theta_array = _theta_array(model, theta)
    test_information = _test_information(model, theta_array)
    return _conditional_reliability(test_information, variance)


def _quadrature_weights(
    theta: NDArray[np.float64],
    density: Density,
) -> NDArray[np.float64]:
    if isinstance(density, str) and density == "norm":
        absolute_theta = np.abs(theta)
        closest = np.min(absolute_theta)
        # Subtract the largest log-density before exponentiating. Factoring
        # the difference of squares also works when theta**2 would overflow.
        with np.errstate(over="ignore", under="ignore"):
            density_values = np.exp(
                -(absolute_theta - closest) * (0.5 * absolute_theta + 0.5 * closest)
            )
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

    largest = float(np.max(density_values))
    if largest <= 0.0:
        raise ValueError("density weights must have a positive sum")
    density_values /= largest

    # The points are evenly spaced, so the common interval width cancels
    # during normalization. Half-weighting the endpoints gives trapezoidal
    # integration without allocating another grid-sized array.
    density_values[[0, -1]] *= 0.5
    density_values /= np.sum(density_values)
    return density_values


def marginal_rxx(
    model: "BaseItemModel | FitResult",
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
    model : BaseItemModel or FitResult
        A fitted IRT model, or the ``FitResult`` of a fit.
    theta_range : tuple of float, default=(-6, 6)
        Finite lower and upper integration bounds.
    n_points : int, default=61
        Number of evenly spaced quadrature points. Must be at least 2.
    density : {"norm", "uniform"} or callable, default="norm"
        Ability density. ``"norm"`` is the normal with the estimated
        ``latent_mean`` and ``latent_covariance`` of a ``FitResult`` as its
        mean and variance when it has them, and the standard normal
        otherwise. A callable receives the theta grid and returns
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
    from mirt.results._common import resolve_latent_prior

    model, latent_mean, latent_cov = resolve_latent_prior(model)
    model = _resolve_model(model)
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

    if lower < 0.0 < upper and upper > np.finfo(np.float64).max + lower:
        theta = np.linspace(0.5 * lower, 0.5 * upper, n_points) * 2.0
    else:
        theta = np.linspace(lower, upper, n_points)
    if (latent_mean is not None or latent_cov is not None) and (
        isinstance(density, str) and density == "norm"
    ):
        # The standard-normal weights of (theta - mean) / sd are
        # N(mean, sd^2) weights.
        mean = 0.0 if latent_mean is None else float(latent_mean[0])
        sd = 1.0 if latent_cov is None else float(np.sqrt(latent_cov[0, 0]))
        weights = _quadrature_weights((theta - mean) / sd, density)
    else:
        weights = _quadrature_weights(theta, density)
    test_information = _test_information(model, theta.reshape(-1, 1))

    centered = _centered_theta(theta[:, None])[:, 0]
    scale = float(np.max(np.abs(centered)))
    if scale == 0.0:
        raise ValueError("the weighted theta distribution must have positive variance")
    centered /= scale
    centered -= np.dot(weights, centered)
    theta_variance = float(np.dot(weights, centered**2))
    if theta_variance <= 0.0:
        raise ValueError("the weighted theta distribution must have positive variance")

    # Keep the scale separate so tiny and very large positive variances do
    # not disappear or overflow before combining with test information.
    with np.errstate(divide="ignore"):
        log_scaled_information = (
            np.log(test_information) + np.log(theta_variance) + 2.0 * np.log(scale)
        )
    local_reliability = expit(log_scaled_information)
    reliability = float(np.sum(weights * local_reliability))
    return float(np.clip(reliability, 0.0, 1.0))


_ML_SCORE_METHODS = frozenset({"ML", "WLE"})


def _score_result_arrays(
    scores: "ScoreResult",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a ScoreResult's finite score rows, as R's ``empirical_rxx`` does.

    Rows whose estimate or standard error is not finite, such as maximum
    likelihood scores of all-correct patterns, are dropped.
    """
    theta = np.asarray(scores.theta, dtype=np.float64)
    errors = np.asarray(scores.standard_error, dtype=np.float64)
    if theta.ndim == 1:
        theta = theta.reshape(-1, 1)
        errors = errors.reshape(-1, 1)
    keep = np.all(np.isfinite(theta), axis=1) & np.all(np.isfinite(errors), axis=1)
    return theta[keep], errors[keep]


def empirical_rxx(
    model: "BaseItemModel | FitResult",
    theta_estimates: "ArrayLike | ScoreResult",
    method: Literal["posterior_variance", "information"] | None = None,
    standard_errors: ArrayLike | None = None,
    scores_include_error: bool | None = None,
) -> Reliability:
    """Compute empirical reliability from ability estimates.

    As in R's ``empirical_rxx``, reliability is either
    ``Var(theta_hat) / (Var(theta_hat) + E[SE**2])``, for posterior scores
    (EAP, MAP, EAPsum) whose shrunken variance estimates the true-score
    variance, or ``1 - E[SE**2] / Var(theta_hat)`` (R's ``T_as_X = TRUE``),
    for maximum-likelihood scores (ML, WLE) whose variance already contains
    the measurement error.

    Pass the ``ScoreResult`` of :func:`mirt.fscores` to use its standard
    errors and pick the formula from its scoring method; this reproduces R's
    ``empirical_rxx(fscores(mod, full.scores.SE = TRUE))`` for EAP scores.
    Without standard errors, they are derived from test information as
    ``1 / sqrt(I(theta_hat))``, the asymptotic error of maximum-likelihood
    estimates, which suits ML and WLE scores only and needs a
    unidimensional model.

    Parameters
    ----------
    model : BaseItemModel or FitResult
        A fitted IRT model, or the ``FitResult`` of a fit.
    theta_estimates : array-like or ScoreResult
        Ability estimates with shape ``(n_persons,)`` or
        ``(n_persons, n_factors)``, or a ``ScoreResult``. Rows of a
        ``ScoreResult`` with a non-finite estimate or standard error are
        dropped, as R does.
    method : {"posterior_variance", "information"}, optional
        Source of each respondent's error variance. ``"posterior_variance"``
        uses the score standard errors; ``"information"`` uses
        ``1 / I(theta_hat)``. Defaults to ``"posterior_variance"`` when
        standard errors are available (``standard_errors`` or a
        ``ScoreResult``) and to ``"information"`` otherwise, with a warning
        because the information-based errors assume ML or WLE scores.
    standard_errors : array-like, optional
        Score standard errors matching ``theta_estimates``. Required when
        ``method="posterior_variance"`` and ``theta_estimates`` is an array.
    scores_include_error : bool, optional
        Whether ``Var(theta_hat)`` already contains the error variance (R's
        ``T_as_X``), selecting ``1 - E[SE**2] / Var(theta_hat)``. Defaults to
        True for ``method="information"`` and for ML or WLE ``ScoreResult``
        input, and to False (R's default) otherwise.

    Returns
    -------
    float or ndarray
        One coefficient in [0, 1] for a unidimensional model, or one
        coefficient per factor for a multidimensional posterior-variance
        estimate.

    Examples
    --------
    >>> scores = fscores(result, responses, method="EAP")
    >>> empirical_rxx(result, scores)  # R: empirical_rxx(fscores(mod, full.scores.SE=TRUE))
    >>> ml_scores = fscores(result, responses, method="ML")
    >>> empirical_rxx(result, ml_scores)  # R: T_as_X = TRUE
    """
    from mirt.results.score_result import ScoreResult

    model = _resolve_model(model)
    score_method: str | None = None
    if isinstance(theta_estimates, ScoreResult):
        if standard_errors is not None:
            raise MirtValidationError(
                "standard_errors cannot be combined with a ScoreResult, which "
                "carries its own standard errors",
                parameter="standard_errors",
            )
        score_method = theta_estimates.method
        theta_values, score_errors = _score_result_arrays(theta_estimates)
        if method is None or method == "posterior_variance":
            standard_errors = score_errors
    else:
        theta_values = theta_estimates
    if method is None:
        method = "information" if standard_errors is None else "posterior_variance"
        if method == "information":
            warnings.warn(
                "empirical_rxx received no standard errors, so they are derived "
                "from test information, which assumes ML or WLE ability "
                "estimates; pass the ScoreResult from fscores() or "
                "standard_errors for EAP or MAP scores",
                UserWarning,
                stacklevel=2,
            )
    if method not in {"posterior_variance", "information"}:
        raise MirtValidationError(
            "method must be 'posterior_variance' or 'information'",
            parameter="method",
            value=method,
            expected="'posterior_variance' or 'information'",
        )
    if scores_include_error is None:
        scores_include_error = method == "information" or (
            score_method is not None and score_method.upper() in _ML_SCORE_METHODS
        )
    elif not isinstance(scores_include_error, (bool, np.bool_)):
        raise MirtValidationError(
            "scores_include_error must be boolean",
            parameter="scores_include_error",
            value=scores_include_error,
            expected="True, False, or None",
        )

    theta = _theta_array(model, theta_values)
    if theta.shape[0] < 2:
        raise ValueError("theta_estimates must contain at least two respondents")

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
    else:
        if standard_errors is not None:
            raise ValueError("standard_errors can only be used with posterior_variance")
        if model.n_factors != 1:
            raise ValueError(
                "information reliability supports unidimensional models only; "
                "use posterior_variance with factor-specific standard_errors"
            )
        test_information = _test_information(model, theta)
        if np.any(test_information == 0.0):
            return 0.0
        # Taking the square root first keeps errors finite even when the
        # inverse of small positive information would overflow.
        errors = (1.0 / np.sqrt(test_information))[:, None]

    centered = _centered_theta(theta)
    scale = np.maximum(np.max(np.abs(centered), axis=0), np.max(errors, axis=0))
    scale[scale == 0.0] = 1.0
    centered /= scale
    centered -= np.mean(centered, axis=0)
    observed_variance = np.einsum("ij,ij->j", centered, centered) / (theta.shape[0] - 1)
    scaled_errors = errors / scale
    average_error_variance = (
        np.einsum("ij,ij->j", scaled_errors, scaled_errors) / theta.shape[0]
    )

    if scores_include_error:
        # Var(theta_hat) = Var(theta) + E[SE^2], so the true-score share is
        # what remains after removing the error variance.
        error_share = np.divide(
            average_error_variance,
            observed_variance,
            out=np.ones_like(observed_variance),
            where=observed_variance > 0.0,
        )
        reliability = 1.0 - error_share
    else:
        denominator = observed_variance + average_error_variance
        reliability = np.divide(
            observed_variance,
            denominator,
            out=np.zeros_like(observed_variance),
            where=denominator > 0.0,
        )
    reliability = np.clip(reliability, 0.0, 1.0)
    if model.n_factors == 1:
        return float(reliability[0])
    return reliability


def sem(
    model: "BaseItemModel | FitResult",
    theta: ArrayLike,
) -> NDArray[np.float64]:
    """Compute the standard error of measurement at given theta values.

    The standard error is ``1 / sqrt(I(theta))``. Models may return either
    total test information or an item-information matrix; both conventions
    are supported. A point with zero test information has infinite standard
    error rather than an arbitrary finite cap.

    Parameters
    ----------
    model : BaseItemModel or FitResult
        A fitted IRT model, or the ``FitResult`` of a fit.
    theta : array-like
        Ability values at which to compute the standard error.

    Returns
    -------
    ndarray
        One standard error for each theta point. Uninformative points are
        represented by positive infinity.
    """
    model = _resolve_model(model)
    theta_array = _theta_array(model, theta)
    test_information = _test_information(model, theta_array)
    return np.divide(
        1.0,
        np.sqrt(test_information),
        out=np.full_like(test_information, np.inf),
        where=test_information > 0.0,
    )
