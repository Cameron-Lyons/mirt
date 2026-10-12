"""Residual analysis functions for IRT models.

Provides functions for computing various types of residuals
for model diagnostics with vectorized model-agnostic kernels. :func:`Q3` and
:func:`LD_X2` share the definitions of :mod:`mirt.diagnostics.ld`.
"""

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt._correlation import pairwise_correlations
from mirt.constants import PROB_EPSILON
from mirt.utils.data import _response_array

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


@dataclass
class ResidualResult:
    """Container for residual analysis results.

    Attributes
    ----------
    raw : NDArray[np.float64]
        Raw residuals (observed - expected).
    standardized : NDArray[np.float64]
        Residuals of the requested ``type``: raw, standardized (also returned
        for ``"pearson"``) or deviance residuals.
    ld_matrix : NDArray[np.float64] | None
        Pairwise correlations of the ``standardized`` residuals, with a unit
        diagonal. With ``type="raw"`` this is Yen's Q3, as returned by
        :func:`Q3`; other types correlate standardized or deviance residuals.
    summary : dict
        Summary statistics for residuals.
    """

    raw: NDArray[np.float64]
    standardized: NDArray[np.float64]
    ld_matrix: NDArray[np.float64] | None
    summary: dict


@dataclass(frozen=True)
class _ResidualData:
    """Validated observations and their model-implied score moments."""

    responses: NDArray[np.float64]
    theta: NDArray[np.float64]
    observed: NDArray[np.bool_]
    probabilities: NDArray[np.float64]
    expected: NDArray[np.float64]
    variance: NDArray[np.float64]
    n_categories: NDArray[np.int_]


def _coerce_responses_theta(
    model: "BaseItemModel",
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Normalize and validate response/theta inputs."""
    responses_f = np.asarray(_response_array(responses), dtype=np.float64)
    theta_2d = np.asarray(np.atleast_1d(theta), dtype=np.float64)
    if theta_2d.ndim == 1:
        theta_2d = theta_2d.reshape(-1, 1)

    if responses_f.ndim != 2 or 0 in responses_f.shape:
        raise ValueError("responses must be a nonempty two-dimensional matrix")
    if theta_2d.ndim != 2:
        raise ValueError("theta must be one- or two-dimensional")
    if theta_2d.shape[0] != responses_f.shape[0]:
        raise ValueError("theta must have one row per response record")
    if not np.all(np.isfinite(theta_2d)):
        raise ValueError("theta must contain only finite values")

    responses_f, observed = _coerce_responses(model, responses_f)
    n_factors = getattr(model, "n_factors", None)
    if n_factors is not None and int(n_factors) != theta_2d.shape[1]:
        raise ValueError("theta columns must match model.n_factors")
    return responses_f, theta_2d, observed


def _coerce_responses(
    model: "BaseItemModel",
    responses: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Normalize and validate a response matrix and its observed mask."""
    responses_f = np.asarray(_response_array(responses), dtype=np.float64)
    if responses_f.ndim != 2 or 0 in responses_f.shape:
        raise ValueError("responses must be a nonempty two-dimensional matrix")
    n_items = getattr(model, "n_items", None)
    if n_items is None or int(n_items) != responses_f.shape[1]:
        raise ValueError("response columns must match model.n_items")

    finite_responses = np.isfinite(responses_f)
    if np.any(np.isinf(responses_f)):
        raise ValueError("responses may contain finite category codes, -1, or NaN")
    negative = finite_responses & (responses_f < 0)
    if np.any(negative & (responses_f != -1)):
        raise ValueError("-1 is the only supported negative missing-response code")
    observed = finite_responses & ~negative
    if np.any(observed & (responses_f != np.floor(responses_f))):
        raise ValueError("observed responses must use integer category codes")
    return responses_f, observed


def _probability_moments(
    model: "BaseItemModel",
    theta: NDArray[np.float64],
    n_items: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int_],
]:
    """Return response probabilities plus expected scores and variances."""
    probabilities = np.asarray(model.probability(theta), dtype=np.float64)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("model probabilities must contain only finite values")
    if np.any(probabilities < -PROB_EPSILON) or np.any(
        probabilities > 1.0 + PROB_EPSILON
    ):
        raise ValueError("model probabilities must lie in [0, 1]")
    probabilities = np.clip(probabilities, 0.0, 1.0)

    expected_shape = (theta.shape[0], n_items)
    if probabilities.ndim == 2:
        if probabilities.shape != expected_shape:
            raise ValueError(
                f"dichotomous probabilities must have shape {expected_shape}"
            )
        expected = probabilities
        variance = probabilities * (1.0 - probabilities)
        n_categories = np.full(n_items, 2, dtype=np.int64)
        return probabilities, expected, variance, n_categories

    if probabilities.ndim != 3 or probabilities.shape[:2] != expected_shape:
        raise ValueError(
            "model probabilities must have shape (n_persons, n_items) or "
            "(n_persons, n_items, n_categories)"
        )
    if probabilities.shape[2] < 2:
        raise ValueError("polytomous probabilities require at least two categories")

    configured_categories = getattr(model, "n_categories", probabilities.shape[2])
    if isinstance(configured_categories, (int, np.integer)):
        n_categories = np.full(n_items, int(configured_categories), dtype=np.int64)
    else:
        n_categories = np.asarray(configured_categories, dtype=np.int64)
    if n_categories.shape != (n_items,) or np.any(n_categories < 2):
        raise ValueError("model.n_categories must provide at least two per item")
    if np.any(n_categories > probabilities.shape[2]):
        raise ValueError("model.n_categories exceeds the probability output width")

    probabilities = probabilities.copy()
    for item_index, count in enumerate(n_categories):
        if np.any(probabilities[:, item_index, count:] > PROB_EPSILON):
            raise ValueError("probabilities beyond an item's categories must be zero")
        mass = np.sum(probabilities[:, item_index, :count], axis=1)
        if not np.allclose(mass, 1.0, rtol=1e-8, atol=1e-10):
            raise ValueError("category probabilities must sum to one for every item")
        probabilities[:, item_index, :count] /= mass[:, None]
        probabilities[:, item_index, count:] = 0.0

    category_scores = np.arange(probabilities.shape[2], dtype=np.float64)
    expected = probabilities @ category_scores
    second_moment = probabilities @ (category_scores**2)
    variance = np.maximum(second_moment - expected**2, 0.0)
    return probabilities, expected, variance, n_categories


def _prepare_residual_data(
    model: "BaseItemModel",
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> _ResidualData:
    """Validate inputs and evaluate the model once for every item."""
    responses_f, theta_2d, observed = _coerce_responses_theta(model, responses, theta)
    probabilities, expected, variance, n_categories = _probability_moments(
        model, theta_2d, responses_f.shape[1]
    )
    invalid_codes = observed & (responses_f >= n_categories[None, :])
    if np.any(invalid_codes):
        person_index, item_index = np.argwhere(invalid_codes)[0]
        raise ValueError(
            f"response at row {person_index}, item {item_index} exceeds the "
            f"{n_categories[item_index] - 1} maximum category"
        )
    return _ResidualData(
        responses=responses_f,
        theta=theta_2d,
        observed=observed,
        probabilities=probabilities,
        expected=expected,
        variance=variance,
        n_categories=n_categories,
    )


def _standardized_score_residuals(data: _ResidualData) -> NDArray[np.float64]:
    """Compute score residuals using each item's modeled score variance."""
    raw = data.responses - data.expected
    denominator = np.sqrt(np.maximum(data.variance, PROB_EPSILON))
    standardized = raw / denominator
    return np.where(data.observed, standardized, np.nan)


def _deviance_residuals(data: _ResidualData) -> NDArray[np.float64]:
    """Compute signed likelihood deviance residuals for categorical responses."""
    response_indices = np.where(data.observed, data.responses, 0).astype(np.int64)
    if data.probabilities.ndim == 2:
        response_probability = np.where(
            response_indices == 1,
            data.probabilities,
            1.0 - data.probabilities,
        )
    else:
        response_probability = np.take_along_axis(
            data.probabilities,
            response_indices[:, :, None],
            axis=2,
        )[:, :, 0]
    response_probability = np.clip(response_probability, PROB_EPSILON, 1.0)
    raw = data.responses - data.expected
    deviance = np.sign(raw) * np.sqrt(-2.0 * np.log(response_probability))
    return np.where(data.observed, deviance, np.nan)


def _residual_summary(
    raw: NDArray[np.float64],
    standardized: NDArray[np.float64],
    observed: NDArray[np.bool_],
) -> dict[str, NDArray[np.float64] | NDArray[np.int_]]:
    """Reduce residual columns without warnings for entirely missing items."""
    counts = np.count_nonzero(observed, axis=0)
    mean_raw = np.divide(
        np.nansum(raw, axis=0),
        counts,
        out=np.full(raw.shape[1], np.nan),
        where=counts > 0,
    )
    mean_standardized = np.divide(
        np.nansum(standardized, axis=0),
        counts,
        out=np.full(raw.shape[1], np.nan),
        where=counts > 0,
    )
    centered = np.where(
        observed,
        standardized - mean_standardized[None, :],
        0.0,
    )
    std_standardized = np.sqrt(
        np.divide(
            np.sum(centered**2, axis=0),
            counts,
            out=np.full(raw.shape[1], np.nan),
            where=counts > 0,
        )
    )
    max_abs = np.max(
        np.where(observed, np.abs(standardized), -np.inf),
        axis=0,
    )
    max_abs[counts == 0] = np.nan
    return {
        "mean_raw": mean_raw,
        "std_standardized": std_standardized,
        "max_abs_standardized": max_abs,
        "n_large": np.count_nonzero(observed & (np.abs(standardized) > 2), axis=0),
    }


def residuals(
    model: "BaseItemModel",
    responses: ArrayLike,
    theta: NDArray[np.float64],
    type: Literal["raw", "standardized", "pearson", "deviance"] = "standardized",
    suppress_abs: float | None = None,
    use_rust: bool = True,
) -> ResidualResult:
    """Compute model residuals.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    responses : array-like of shape (n_persons, n_items)
        Integer-coded response matrix or DataFrame. Missing responses may be
        coded as ``-1``, ``NaN`` or the nulls of nullable DataFrame columns.
    theta : NDArray[np.float64]
        Ability estimates. Shape: (n_persons,) or (n_persons, n_dims).
    type : str
        Type of residuals:

        - "raw": ``x - E[x | theta]``, with the expected item score ``E``.
        - "standardized": ``(x - E) / sqrt(Var[x | theta])``.
        - "pearson": alias for "standardized", the Pearson residual of the
          item score. For binary items it is the Bernoulli Pearson residual
          ``(x - P) / sqrt(P(1 - P))``. It differs from
          ``residual_type="pearson"`` of
          :func:`mirt.diagnostics.compute_residuals`, which divides by
          ``sqrt(E)``.
        - "deviance": Signed deviance residuals
    suppress_abs : float, optional
        Suppress residuals with absolute value below this threshold
        when computing LD matrix.
    use_rust : bool
        Accepted for backward compatibility and validated. Residuals are
        always computed with vectorized NumPy kernels. Default True.

    Returns
    -------
    ResidualResult
        Container with residuals and local dependence matrix.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL")
    >>> resid = residuals(result.model, responses, result.theta)
    >>> print(f"Mean absolute residual: {np.mean(np.abs(resid.raw)):.3f}")
    """
    valid_types = {"raw", "standardized", "pearson", "deviance"}
    if type not in valid_types:
        raise ValueError(f"Unknown residual type: {type}")
    if not isinstance(use_rust, (bool, np.bool_)):
        raise ValueError("use_rust must be a boolean")
    if suppress_abs is not None and (
        not np.isfinite(suppress_abs) or suppress_abs < 0.0
    ):
        raise ValueError("suppress_abs must be finite and nonnegative")

    data = _prepare_residual_data(model, responses, theta)
    raw = np.where(data.observed, data.responses - data.expected, np.nan)
    if type == "raw":
        standardized = raw
    elif type in ("standardized", "pearson"):
        standardized = _standardized_score_residuals(data)
    else:
        standardized = _deviance_residuals(data)
    ld_matrix = _compute_ld_matrix(standardized, suppress_abs)

    return ResidualResult(
        raw=raw,
        standardized=standardized,
        ld_matrix=ld_matrix,
        summary=_residual_summary(raw, standardized, data.observed),
    )


def _compute_ld_matrix(
    standardized_residuals: NDArray[np.float64],
    suppress_abs: float | None = None,
) -> NDArray[np.float64]:
    """Compute the pairwise correlation matrix of residuals.

    Correlations of raw residuals are Yen's Q3 statistics. Pairs with fewer
    than three joint residuals are ``NaN`` and the diagonal is one.
    """
    values = np.asarray(standardized_residuals, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("standardized_residuals must be a nonempty matrix")
    if np.any(np.isinf(values)):
        raise ValueError("standardized_residuals may contain finite values or NaN")
    if suppress_abs is not None and (
        not np.isfinite(suppress_abs) or suppress_abs < 0.0
    ):
        raise ValueError("suppress_abs must be finite and nonnegative")

    ld_matrix, _ = pairwise_correlations(
        values, min_count=3, min_denominator=PROB_EPSILON
    )
    if suppress_abs is not None:
        ld_matrix[np.abs(ld_matrix) < suppress_abs] = 0.0
    np.fill_diagonal(ld_matrix, 1.0)
    return ld_matrix


def Q3(
    model: "BaseItemModel",
    responses: ArrayLike,
    theta: NDArray[np.float64],
    use_rust: bool = True,
) -> NDArray[np.float64]:
    """Compute Yen's Q3 statistic for local dependence.

    Q3 is the correlation between the raw residuals ``x - E[x | theta]`` of
    pairs of items, where ``E[x | theta]`` is the expected item score (the
    expected category for polytomous items) at the ability estimates.
    Large positive values suggest local dependence.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    responses : array-like of shape (n_persons, n_items)
        Integer-coded response matrix or DataFrame. Missing values may be
        ``-1``, ``NaN`` or the nulls of nullable DataFrame columns.
    theta : NDArray[np.float64]
        Ability estimates.
    use_rust : bool
        Accepted for backward compatibility and validated. Q3 is always
        computed with vectorized NumPy kernels. Default True.

    Returns
    -------
    NDArray[np.float64]
        Q3 correlation matrix. Shape: (n_items, n_items). The diagonal is
        one and pairs with fewer than three joint responses are ``NaN``.

    Notes
    -----
    Values above 0.2 may indicate local dependence between items. Under
    local independence the average Q3 is about ``-1 / (n_items - 1)``. The
    values match ``residuals(type = "Q3")`` in the R package mirt for the
    same abilities.

    References
    ----------
    Yen, W. M. (1984). Effects of local item dependence on the fit and
    equating performance of the three-parameter logistic model.
    Applied Psychological Measurement, 8(2), 125-145.
    """
    if not isinstance(use_rust, (bool, np.bool_)):
        raise ValueError("use_rust must be a boolean")
    data = _prepare_residual_data(model, responses, theta)
    raw = np.where(data.observed, data.responses - data.expected, np.nan)
    return _compute_ld_matrix(raw)


def LD_X2(
    model: "BaseItemModel",
    responses: ArrayLike,
    theta: NDArray[np.float64] | None = None,
    use_rust: bool = True,
    *,
    n_quadpts: int | None = None,
    prior_mean: ArrayLike | None = None,
    prior_cov: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute Chen & Thissen LD chi-square statistic.

    For items ``i`` and ``j`` the statistic compares the observed
    ``K_i x K_j`` table of the ``N_ij`` persons who answered both items with
    the model-implied table ``N_ij * sum_q w_q P_ik(theta_q) P_jl(theta_q)``.
    The expected table integrates over the latent population on a
    Gauss-Hermite grid, so it does not depend on ability estimates. This is
    the statistic of :func:`mirt.diagnostics.compute_ld_chi2`.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    responses : array-like of shape (n_persons, n_items)
        Integer-coded response matrix or DataFrame. Missing values may be
        ``-1``, ``NaN`` or the nulls of nullable DataFrame columns.
    theta : NDArray[np.float64], optional
        Deprecated and ignored. Passing it emits a ``DeprecationWarning``.
    use_rust : bool
        Accepted for backward compatibility and validated. The native kernel
        conditions on abilities, so the statistic is always computed with
        vectorized NumPy reductions. Default True.
    n_quadpts : int, optional
        Quadrature points per latent dimension. The default depends on the
        number of factors, as in :func:`mirt.diagnostics.compute_ld_statistics`.
    prior_mean, prior_cov : array-like, optional
        Mean and covariance of the normal latent population. Defaults to the
        standard normal.

    Returns
    -------
    ld_x2 : NDArray[np.float64]
        LD chi-square matrix. Shape: (n_items, n_items).
    p_values : NDArray[np.float64]
        P-value matrix, on ``(K_i - 1)(K_j - 1)`` degrees of freedom. The
        diagonal and pairs with fewer than 10 joint responses are ``NaN`` in
        both matrices.

    References
    ----------
    Chen, W. H., & Thissen, D. (1997). Local dependence indexes for item
    pairs using item response theory. Journal of Educational and
    Behavioral Statistics, 22(3), 265-289.
    """
    from scipy import stats

    from mirt._local_dependence import model_ld_statistics
    from mirt.diagnostics._quadrature import resolve_diagnostic_quadpts

    if not isinstance(use_rust, (bool, np.bool_)):
        raise ValueError("use_rust must be a boolean")
    if theta is None:
        responses_f, observed = _coerce_responses(model, responses)
    else:
        warnings.warn(
            "theta is deprecated for LD_X2 and ignored: the LD chi-square "
            "integrates over the latent population",
            DeprecationWarning,
            stacklevel=2,
        )
        responses_f, _, observed = _coerce_responses_theta(model, responses, theta)

    statistics = model_ld_statistics(
        model,
        np.where(observed, responses_f, -1.0),
        n_quadpts=resolve_diagnostic_quadpts(
            n_quadpts, int(getattr(model, "n_factors", 1))
        ),
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        compute_g2=False,
    )
    ld_x2 = statistics.chi2
    p_values = np.full_like(ld_x2, np.nan)
    eligible = ~np.isnan(ld_x2)
    p_values[eligible] = stats.chi2.sf(ld_x2[eligible], df=statistics.df[eligible])
    return ld_x2, p_values
