"""Residual analysis functions for IRT models.

Provides functions for computing various types of residuals
for model diagnostics. Uses the native backend when its formulas match the
model and vectorized model-agnostic kernels otherwise.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON

_MIN_LD_PAIR_RESPONSES = 5
_MIN_EXPECTED_CELL = 0.5
_LD_CATEGORY_CHUNK_ELEMENTS = 1_000_000

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

try:
    from mirt._backend_config import should_use_rust
    from mirt._rust_backend import (
        compute_ld_chi2_matrix,
        compute_q3_matrix,
        compute_standardized_residuals,
    )
except ImportError:

    def should_use_rust(use_rust: bool = True) -> bool:
        return False

    compute_ld_chi2_matrix = None
    compute_q3_matrix = None
    compute_standardized_residuals = None


@dataclass
class ResidualResult:
    """Container for residual analysis results.

    Attributes
    ----------
    raw : NDArray[np.float64]
        Raw residuals (observed - expected).
    standardized : NDArray[np.float64]
        Standardized residuals.
    ld_matrix : NDArray[np.float64] | None
        Local dependence matrix (Q3 or other).
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
    responses_f = np.asarray(responses, dtype=np.float64)
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

    n_items = getattr(model, "n_items", None)
    if n_items is None or int(n_items) != responses_f.shape[1]:
        raise ValueError("response columns must match model.n_items")
    n_factors = getattr(model, "n_factors", None)
    if n_factors is not None and int(n_factors) != theta_2d.shape[1]:
        raise ValueError("theta columns must match model.n_factors")

    finite_responses = np.isfinite(responses_f)
    if np.any(np.isinf(responses_f)):
        raise ValueError("responses may contain finite category codes, -1, or NaN")
    negative = finite_responses & (responses_f < 0)
    if np.any(negative & (responses_f != -1)):
        raise ValueError("-1 is the only supported negative missing-response code")
    observed = finite_responses & ~negative
    if np.any(observed & (responses_f != np.floor(responses_f))):
        raise ValueError("observed responses must use integer category codes")
    return responses_f, theta_2d, observed


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


def _supports_binary_2pl_kernel(model: "BaseItemModel") -> bool:
    """Return whether the native formulas exactly represent this model."""
    from mirt.models.dichotomous import OneParameterLogistic, TwoParameterLogistic

    return type(model) in (OneParameterLogistic, TwoParameterLogistic) and (
        model.n_factors == 1
    )


def _prepare_rust_payload(
    model: "BaseItemModel",
    data: _ResidualData,
    use_rust: bool,
) -> tuple[
    bool,
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    NDArray[np.float64] | None,
    NDArray[np.int32] | None,
]:
    """Build a payload only when the native formulas exactly match the model."""
    can_use_rust = should_use_rust(use_rust) and _supports_binary_2pl_kernel(model)

    if not can_use_rust:
        return False, None, None, None, None

    disc = np.asarray(model.discrimination, dtype=np.float64)
    diff = np.asarray(model.difficulty, dtype=np.float64)
    theta_1d = data.theta[:, 0].astype(np.float64, copy=False)
    responses_int = np.where(data.observed, data.responses, -1).astype(np.int32)

    return True, disc, diff, theta_1d, responses_int


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
    responses: NDArray[np.float64],
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
    responses : NDArray[np.float64]
        Integer-coded response matrix. Shape: (n_persons, n_items). Missing
        responses may be coded as ``-1`` or ``NaN``.
    theta : NDArray[np.float64]
        Ability estimates. Shape: (n_persons,) or (n_persons, n_dims).
    type : str
        Type of residuals:
        - "raw": observed - expected
        - "standardized": raw / sqrt(modeled score variance)
        - "pearson": alias for standardized score residuals
        - "deviance": Signed deviance residuals
    suppress_abs : float, optional
        Suppress residuals with absolute value below this threshold
        when computing LD matrix.
    use_rust : bool
        Use the native backend when its binary 1PL/2PL formulas exactly match
        the model. Richer dichotomous and polytomous models use the vectorized
        model-agnostic path. Default True.

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
    rust_ready, disc, diff, theta_1d, responses_int = _prepare_rust_payload(
        model, data, bool(use_rust)
    )
    can_use_rust = rust_ready and type in ("standardized", "pearson")
    raw = np.where(data.observed, data.responses - data.expected, np.nan)

    if can_use_rust:
        assert disc is not None
        assert diff is not None
        assert theta_1d is not None
        assert responses_int is not None
        assert compute_standardized_residuals is not None
        assert compute_q3_matrix is not None

        standardized = np.asarray(
            compute_standardized_residuals(responses_int, theta_1d, disc, diff),
            dtype=np.float64,
        )
        ld_matrix = np.asarray(
            compute_q3_matrix(responses_int, theta_1d, disc, diff),
            dtype=np.float64,
        )
        np.fill_diagonal(ld_matrix, 1.0)
        if suppress_abs is not None:
            ld_matrix = np.where(np.abs(ld_matrix) < suppress_abs, 0.0, ld_matrix)
            np.fill_diagonal(ld_matrix, 1.0)
    else:
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
    """Compute local dependence matrix (Q3 statistic).

    The Q3 statistic is the correlation between standardized residuals
    for pairs of items.
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

    valid = np.isfinite(values)
    valid_float = valid.astype(np.float64)
    filled = np.where(valid, values, 0.0)
    counts = valid_float.T @ valid_float
    sums = filled.T @ valid_float
    squared_sums = (filled**2).T @ valid_float
    cross_products = filled.T @ filled

    n_items = values.shape[1]
    ld_matrix = np.full((n_items, n_items), np.nan, dtype=np.float64)
    row_indices, column_indices = np.triu_indices(n_items, k=1)
    pair_counts = counts[row_indices, column_indices]
    sum_rows = sums[row_indices, column_indices]
    sum_columns = sums[column_indices, row_indices]
    covariance = cross_products[row_indices, column_indices] - np.divide(
        sum_rows * sum_columns,
        pair_counts,
        out=np.zeros_like(pair_counts),
        where=pair_counts > 0,
    )
    variance_rows = squared_sums[row_indices, column_indices] - np.divide(
        sum_rows**2,
        pair_counts,
        out=np.zeros_like(pair_counts),
        where=pair_counts > 0,
    )
    variance_columns = squared_sums[column_indices, row_indices] - np.divide(
        sum_columns**2,
        pair_counts,
        out=np.zeros_like(pair_counts),
        where=pair_counts > 0,
    )
    denominator = np.sqrt(
        np.maximum(variance_rows, 0.0) * np.maximum(variance_columns, 0.0)
    )
    pair_correlations = np.divide(
        covariance,
        denominator,
        out=np.full_like(covariance, np.nan),
        where=(pair_counts > 2) & (denominator > PROB_EPSILON),
    )
    if suppress_abs is not None:
        pair_correlations = np.where(
            np.abs(pair_correlations) < suppress_abs,
            0.0,
            pair_correlations,
        )
    ld_matrix[row_indices, column_indices] = pair_correlations
    ld_matrix[column_indices, row_indices] = pair_correlations
    np.fill_diagonal(ld_matrix, 1.0)
    return ld_matrix


def Q3(
    model: "BaseItemModel",
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
    use_rust: bool = True,
) -> NDArray[np.float64]:
    """Compute Yen's Q3 statistic for local dependence.

    Q3 is the correlation between standardized residuals for pairs
    of items. Large positive values suggest local dependence.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    responses : NDArray[np.float64]
        Integer-coded response matrix. Missing values may be ``-1`` or ``NaN``.
    theta : NDArray[np.float64]
        Ability estimates.
    use_rust : bool
        Use the native backend for unidimensional 1PL/2PL models. Other
        supported models use the vectorized model-agnostic path. Default True.

    Returns
    -------
    NDArray[np.float64]
        Q3 correlation matrix. Shape: (n_items, n_items).

    Notes
    -----
    Values above 0.2 may indicate local dependence between items.
    The average Q3 should be close to zero if model fits well.

    References
    ----------
    Yen, W. M. (1984). Effects of local item dependence on the fit and
    equating performance of the three-parameter logistic model.
    Applied Psychological Measurement, 8(2), 125-145.
    """
    if not isinstance(use_rust, (bool, np.bool_)):
        raise ValueError("use_rust must be a boolean")
    data = _prepare_residual_data(model, responses, theta)
    can_use_rust, disc, diff, theta_1d, responses_int = _prepare_rust_payload(
        model, data, bool(use_rust)
    )

    if can_use_rust:
        assert disc is not None
        assert diff is not None
        assert theta_1d is not None
        assert responses_int is not None
        assert compute_q3_matrix is not None

        q3_matrix = np.asarray(
            compute_q3_matrix(responses_int, theta_1d, disc, diff),
            dtype=np.float64,
        )
        np.fill_diagonal(q3_matrix, 1.0)
        return q3_matrix

    return _compute_ld_matrix(_standardized_score_residuals(data))


def _binary_ld_x2(
    data: _ResidualData,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Compute every binary LD statistic with matrix reductions."""
    observed = data.observed.astype(np.float64)
    response_one = np.where(data.observed, data.responses, 0.0)
    response_zero = observed - response_one
    probability_one = np.where(data.observed, data.probabilities, 0.0)
    probability_zero = observed - probability_one

    observed_cells = (
        response_zero.T @ response_zero,
        response_zero.T @ response_one,
        response_one.T @ response_zero,
        response_one.T @ response_one,
    )
    expected_cells = (
        probability_zero.T @ probability_zero,
        probability_zero.T @ probability_one,
        probability_one.T @ probability_zero,
        probability_one.T @ probability_one,
    )
    ld_x2 = np.zeros_like(observed_cells[0], dtype=np.float64)
    for observed_cell, expected_cell in zip(
        observed_cells, expected_cells, strict=True
    ):
        ld_x2 += (observed_cell - expected_cell) ** 2 / np.maximum(
            expected_cell, _MIN_EXPECTED_CELL
        )

    pair_counts = observed.T @ observed
    eligible = pair_counts >= _MIN_LD_PAIR_RESPONSES
    np.fill_diagonal(eligible, False)
    ld_x2 = np.where(eligible, (ld_x2 + ld_x2.T) / 2.0, 0.0)
    return ld_x2, eligible


def _polytomous_ld_x2(
    data: _ResidualData,
) -> tuple[NDArray[np.float64], NDArray[np.int_], NDArray[np.bool_]]:
    """Compute generalized categorical LD statistics in respondent chunks."""
    n_persons, n_items = data.responses.shape
    ld_x2 = np.zeros((n_items, n_items), dtype=np.float64)
    category_degrees = data.n_categories.astype(np.int64, copy=False) - 1
    degrees = np.multiply.outer(category_degrees, category_degrees)
    np.fill_diagonal(degrees, 1)

    max_categories = int(np.max(data.n_categories))
    # Compact item-pair tables perform fewer reductions for small instruments
    # whose response scale is wider than the item set.
    if n_items <= max_categories:
        return _pairwise_polytomous_ld_x2(data, degrees)

    observed_float = data.observed.astype(np.float64)
    pair_counts = observed_float.T @ observed_float
    eligible = pair_counts >= _MIN_LD_PAIR_RESPONSES
    np.fill_diagonal(eligible, False)
    if not np.any(eligible):
        return ld_x2, degrees, eligible

    category_exists = np.arange(max_categories)[:, None] < data.n_categories[None, :]
    chunk_size = min(
        n_persons,
        max(1, _LD_CATEGORY_CHUNK_ELEMENTS // n_items),
    )

    for first_category in range(max_categories):
        first_exists = category_exists[first_category]
        for second_category in range(first_category, max_categories):
            valid_cells = (
                first_exists[:, None] & category_exists[second_category][None, :]
            )
            if not np.any(valid_cells & eligible):
                continue

            observed_cells = np.zeros((n_items, n_items), dtype=np.float64)
            expected_cells = np.zeros((n_items, n_items), dtype=np.float64)
            for start in range(0, n_persons, chunk_size):
                stop = min(start + chunk_size, n_persons)
                observed = data.observed[start:stop]
                responses = data.responses[start:stop]
                first_observed = (observed & (responses == first_category)).astype(
                    np.float64
                )
                first_probability = np.where(
                    observed,
                    data.probabilities[start:stop, :, first_category],
                    0.0,
                )
                if first_category == second_category:
                    second_observed = first_observed
                    second_probability = first_probability
                else:
                    second_observed = (
                        observed & (responses == second_category)
                    ).astype(np.float64)
                    second_probability = np.where(
                        observed,
                        data.probabilities[start:stop, :, second_category],
                        0.0,
                    )

                observed_cells += first_observed.T @ second_observed
                expected_cells += first_probability.T @ second_probability

            contribution = (observed_cells - expected_cells) ** 2
            contribution /= np.maximum(expected_cells, _MIN_EXPECTED_CELL)
            ld_x2 += np.where(valid_cells, contribution, 0.0)
            if first_category != second_category:
                ld_x2 += np.where(valid_cells.T, contribution.T, 0.0)

    ld_x2 = np.where(eligible, (ld_x2 + ld_x2.T) / 2.0, 0.0)

    return ld_x2, degrees, eligible


def _pairwise_polytomous_ld_x2(
    data: _ResidualData,
    degrees: NDArray[np.int_],
) -> tuple[NDArray[np.float64], NDArray[np.int_], NDArray[np.bool_]]:
    """Use compact per-pair tables when categories outnumber items."""
    n_items = data.responses.shape[1]
    ld_x2 = np.zeros((n_items, n_items), dtype=np.float64)
    eligible = np.zeros((n_items, n_items), dtype=np.bool_)
    for first in range(n_items):
        first_categories = int(data.n_categories[first])
        for second in range(first + 1, n_items):
            valid = data.observed[:, first] & data.observed[:, second]
            if np.count_nonzero(valid) < _MIN_LD_PAIR_RESPONSES:
                continue
            second_categories = int(data.n_categories[second])
            first_responses = data.responses[valid, first].astype(np.int64)
            second_responses = data.responses[valid, second].astype(np.int64)
            flat_cells = first_responses * second_categories + second_responses
            observed_cells = np.bincount(
                flat_cells,
                minlength=first_categories * second_categories,
            ).reshape(first_categories, second_categories)
            expected_cells = (
                data.probabilities[valid, first, :first_categories].T
                @ data.probabilities[valid, second, :second_categories]
            )
            statistic = np.sum(
                (observed_cells - expected_cells) ** 2
                / np.maximum(expected_cells, _MIN_EXPECTED_CELL)
            )
            ld_x2[first, second] = ld_x2[second, first] = statistic
            eligible[first, second] = eligible[second, first] = True

    return ld_x2, degrees, eligible


def LD_X2(
    model: "BaseItemModel",
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
    use_rust: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute Chen & Thissen LD chi-square statistic.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    responses : NDArray[np.float64]
        Integer-coded response matrix. Missing values may be ``-1`` or ``NaN``.
    theta : NDArray[np.float64]
        Ability estimates.
    use_rust : bool
        Use the native backend for unidimensional binary 1PL/2PL models. Other
        dichotomous models use batched matrix reductions, while polytomous
        models use generalized category-pair tables. Default True.

    Returns
    -------
    ld_x2 : NDArray[np.float64]
        LD chi-square matrix. Shape: (n_items, n_items).
    p_values : NDArray[np.float64]
        P-value matrix.

    References
    ----------
    Chen, W. H., & Thissen, D. (1997). Local dependence indexes for item
    pairs using item response theory. Journal of Educational and
    Behavioral Statistics, 22(3), 265-289.
    """
    from scipy import stats

    if not isinstance(use_rust, (bool, np.bool_)):
        raise ValueError("use_rust must be a boolean")
    data = _prepare_residual_data(model, responses, theta)
    can_use_rust, disc, diff, theta_1d, responses_int = _prepare_rust_payload(
        model, data, bool(use_rust)
    )

    if can_use_rust:
        assert disc is not None
        assert diff is not None
        assert theta_1d is not None
        assert responses_int is not None
        assert compute_ld_chi2_matrix is not None

        ld_x2 = np.asarray(
            compute_ld_chi2_matrix(responses_int, theta_1d, disc, diff),
            dtype=np.float64,
        )
        observed = data.observed.astype(np.int64)
        eligible = observed.T @ observed >= _MIN_LD_PAIR_RESPONSES
        np.fill_diagonal(eligible, False)
        ld_x2 = np.where(eligible, ld_x2, 0.0)
        degrees = np.ones_like(ld_x2, dtype=np.int64)
    elif data.probabilities.ndim == 2:
        ld_x2, eligible = _binary_ld_x2(data)
        degrees = np.ones_like(ld_x2, dtype=np.int64)
    else:
        ld_x2, degrees, eligible = _polytomous_ld_x2(data)

    p_values = np.ones_like(ld_x2, dtype=np.float64)
    p_values[eligible] = stats.chi2.sf(ld_x2[eligible], df=degrees[eligible])
    return ld_x2, p_values
