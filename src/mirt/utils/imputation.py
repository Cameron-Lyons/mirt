"""Missing data imputation methods for IRT.

This module provides methods for handling missing responses:
- Simple imputation (mean, mode, random)
- Model-based imputation (EM)
- Multiple imputation
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.utils.data import validate_responses

LARGE_DF = 1e10
_PAIRWISE_BLOCK_SIZE = np.iinfo(np.uint8).max
_MAX_VECTORIZED_CATEGORIES = 32
_MAX_CATEGORY_FREQUENCY_ENTRIES = 5_000_000
_IMPUTATION_METHODS = ("mean", "median", "mode", "random", "EM", "multiple")
ImputationMethod = Literal["mean", "median", "mode", "random", "EM", "multiple"]


def _prepare_response_matrix(
    responses: NDArray[np.int_], missing_code: int
) -> NDArray[np.int_]:
    if not isinstance(missing_code, (int, np.integer)) or isinstance(
        missing_code, (bool, np.bool_)
    ):
        raise MirtValidationError(
            "missing_code must be an integer",
            parameter="missing_code",
            value=missing_code,
        )

    int_bounds = np.iinfo(np.int_)
    if missing_code < int_bounds.min or missing_code > int_bounds.max:
        raise MirtValidationError(
            "missing_code exceeds the supported integer range",
            parameter="missing_code",
            value=missing_code,
        )

    return validate_responses(
        responses,
        allow_missing=True,
        missing_code=int(missing_code),
    )


def _draw_categorical(
    probabilities: NDArray[np.float64], rng: np.random.Generator
) -> NDArray[np.int_]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        raise MirtDataError(
            "categorical probabilities must have shape (n_persons, n_categories)"
        )

    safe_probabilities = np.nan_to_num(probabilities, nan=0.0, posinf=0.0, neginf=0.0)
    safe_probabilities = np.maximum(safe_probabilities, 0.0)
    totals = safe_probabilities.sum(axis=1, keepdims=True)
    normalized = np.divide(
        safe_probabilities,
        totals,
        out=np.full_like(safe_probabilities, 1.0 / probabilities.shape[1]),
        where=totals > 0,
    )
    cumulative = np.cumsum(normalized, axis=1)
    cumulative[:, -1] = 1.0
    draws = (rng.random(probabilities.shape[0])[:, None] > cumulative).sum(axis=1)
    return draws.astype(np.int_, copy=False)


def impute_responses(
    responses: NDArray[np.int_],
    method: ImputationMethod = "EM",
    model: Literal["1PL", "2PL", "3PL", "GRM", "GPCM"] | None = None,
    n_imputations: int = 5,
    missing_code: int = -1,
    seed: int | None = None,
) -> NDArray[np.int_] | list[NDArray[np.int_]]:
    """Impute missing responses in a response matrix.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items) with missing values coded as missing_code
    method : str
        Imputation method:
        - 'mean': Replace with item mean (rounded)
        - 'median': Replace with item median (rounded)
        - 'mode': Replace with item mode
        - 'random': Random draw from item distribution
        - 'EM': Model-based imputation using IRT
        - 'multiple': Multiple imputation (returns list)
    model : str, optional
        IRT model to use for EM imputation (default: '2PL')
    n_imputations : int
        Number of imputations for multiple imputation
    missing_code : int
        Code used to indicate missing values (default: -1)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    NDArray or list of NDArray
        Imputed response matrix (or list for multiple imputation)
    """
    if method not in _IMPUTATION_METHODS:
        raise MirtValidationError(
            "Unknown imputation method",
            parameter="method",
            value=method,
            expected=", ".join(_IMPUTATION_METHODS),
        )
    if method == "multiple" and (
        not isinstance(n_imputations, (int, np.integer))
        or isinstance(n_imputations, (bool, np.bool_))
        or n_imputations < 1
    ):
        raise MirtValidationError(
            "n_imputations must be a positive integer",
            parameter="n_imputations",
            value=n_imputations,
        )

    rng = np.random.default_rng(seed)
    responses = _prepare_response_matrix(responses, missing_code).copy()

    missing_mask = responses == missing_code

    if not np.any(missing_mask):
        if method == "multiple":
            return [responses.copy() for _ in range(n_imputations)]
        return responses

    all_missing_items = np.flatnonzero(missing_mask.all(axis=0))
    if all_missing_items.size:
        raise MirtDataError(
            "Cannot impute items with no observed responses",
            item_indices=all_missing_items.tolist(),
        )

    if method == "mean":
        return _impute_mean(responses, missing_mask)

    if method == "median":
        return _impute_median(responses, missing_mask)

    if method == "mode":
        return _impute_mode(responses, missing_mask)

    if method == "random":
        return _impute_random(responses, missing_mask, rng)

    if method == "EM":
        if model is None:
            model = "2PL"
        return _impute_em(responses, missing_mask, model, rng)

    if model is None:
        model = "2PL"
    return _impute_multiple(responses, missing_mask, model, n_imputations, rng)


def _impute_mean(
    responses: NDArray[np.int_],
    missing_mask: NDArray[np.bool_],
) -> NDArray[np.int_]:
    """Impute with item means (rounded)."""
    imputed = responses.copy()
    valid = ~missing_mask
    counts = np.sum(valid, axis=0, dtype=np.intp)
    sums = np.sum(
        responses,
        axis=0,
        dtype=np.float64,
        where=valid,
        initial=0.0,
    )
    means = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums),
        where=counts > 0,
    )
    _fill_missing_items(imputed, missing_mask, np.rint(means), counts > 0)

    return imputed


def _item_category_frequencies(
    responses: NDArray[np.int_],
    missing_mask: NDArray[np.bool_],
) -> NDArray[np.intp] | None:
    """Count item categories together when the bounded table stays compact."""
    valid = ~missing_mask
    maximum = int(np.max(responses, where=valid, initial=-1))
    n_categories = maximum + 1
    n_items = responses.shape[1]
    if (
        n_categories < 1
        or n_categories > _MAX_VECTORIZED_CATEGORIES
        or n_categories * n_items > _MAX_CATEGORY_FREQUENCY_ENTRIES
    ):
        return None

    frequencies = np.empty((n_categories, n_items), dtype=np.intp)
    for category in range(n_categories):
        frequencies[category] = np.count_nonzero(
            (responses == category) & valid,
            axis=0,
        )
    return frequencies


def _fill_missing_items(
    imputed: NDArray[np.int_],
    missing_mask: NDArray[np.bool_],
    fill_values: NDArray[np.integer] | NDArray[np.floating],
    has_observations: NDArray[np.bool_],
) -> None:
    """Broadcast item fill values only into eligible missing cells."""
    values = np.asarray(fill_values, dtype=np.int_)
    if np.all(has_observations):
        np.copyto(imputed, values[None, :], where=missing_mask)
    else:
        np.copyto(
            imputed,
            values[None, :],
            where=missing_mask & has_observations[None, :],
        )


def _impute_median(
    responses: NDArray[np.int_],
    missing_mask: NDArray[np.bool_],
) -> NDArray[np.int_]:
    """Impute with rounded item medians."""
    imputed = responses.copy()
    frequencies = _item_category_frequencies(responses, missing_mask)
    if frequencies is None:
        for item in range(responses.shape[1]):
            observed = responses[~missing_mask[:, item], item]
            if observed.size:
                imputed[missing_mask[:, item], item] = np.rint(
                    np.median(observed)
                ).astype(np.int_)
        return imputed

    counts = np.sum(frequencies, axis=0, dtype=np.intp)
    cumulative = np.cumsum(frequencies, axis=0, dtype=np.intp)
    lower_ranks = (counts - 1) // 2
    upper_ranks = counts // 2
    lower = np.argmax(cumulative > lower_ranks[None, :], axis=0)
    upper = np.argmax(cumulative > upper_ranks[None, :], axis=0)
    medians = np.rint((lower.astype(np.float64) + upper) * 0.5)
    _fill_missing_items(imputed, missing_mask, medians, counts > 0)
    return imputed


def _impute_mode(
    responses: NDArray[np.int_],
    missing_mask: NDArray[np.bool_],
) -> NDArray[np.int_]:
    """Impute with item modes."""
    imputed = responses.copy()
    frequencies = _item_category_frequencies(responses, missing_mask)
    if frequencies is None:
        for item in range(responses.shape[1]):
            observed = responses[~missing_mask[:, item], item]
            if observed.size:
                values, counts = np.unique(observed, return_counts=True)
                imputed[missing_mask[:, item], item] = values[np.argmax(counts)]
        return imputed

    counts = np.sum(frequencies, axis=0, dtype=np.intp)
    modes = np.argmax(frequencies, axis=0)
    _fill_missing_items(imputed, missing_mask, modes, counts > 0)

    return imputed


def _impute_random(
    responses: NDArray[np.int_],
    missing_mask: NDArray[np.bool_],
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Impute by random draw from item distribution."""
    imputed = responses.copy()
    n_items = responses.shape[1]

    for j in range(n_items):
        valid = ~missing_mask[:, j]
        n_missing = missing_mask[:, j].sum()

        if valid.any() and n_missing > 0:
            observed = responses[valid, j]
            imputed[missing_mask[:, j], j] = rng.choice(
                observed, size=n_missing, replace=True
            )

    return imputed


def _impute_em(
    responses: NDArray[np.int_],
    missing_mask: NDArray[np.bool_],
    model: str,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Model-based imputation using EM algorithm."""
    from mirt import fit_mirt
    from mirt.scoring import fscores

    imputed = responses.copy()
    n_items = responses.shape[1]

    imputed = _impute_mode(responses, missing_mask)

    for _ in range(10):
        try:
            result = fit_mirt(imputed, model=model, verbose=False)
            scores = fscores(result.model, imputed, method="EAP")
            theta = scores.theta
        except (
            ValueError,
            RuntimeError,
            ArithmeticError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ):
            return _impute_random(responses, missing_mask, rng)

        old_imputed = imputed.copy()

        for j in range(n_items):
            item_missing = missing_mask[:, j]
            if not item_missing.any():
                continue

            theta_missing = theta[item_missing]
            if theta_missing.ndim == 1:
                theta_missing = theta_missing.reshape(-1, 1)

            probs = result.model.probability(theta_missing, j)

            if probs.ndim > 1:
                missing_rows = np.flatnonzero(item_missing)
                imputed[missing_rows, j] = _draw_categorical(probs, rng)
            else:
                imputed[item_missing, j] = (
                    rng.random(item_missing.sum()) < probs
                ).astype(np.int_)

        if np.array_equal(old_imputed[missing_mask], imputed[missing_mask]):
            break

    return imputed


def _impute_multiple(
    responses: NDArray[np.int_],
    missing_mask: NDArray[np.bool_],
    model: str,
    n_imputations: int,
    rng: np.random.Generator,
) -> list[NDArray[np.int_]]:
    """Multiple imputation using proper imputation."""
    from mirt import fit_mirt
    from mirt.scoring import fscores

    n_persons, n_items = responses.shape
    imputations: list[NDArray[np.int_]] = []

    initial = _impute_mode(responses, missing_mask)

    try:
        result = fit_mirt(initial, model=model, verbose=False)
        scores = fscores(result.model, initial, method="EAP")
    except (
        ValueError,
        RuntimeError,
        ArithmeticError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ):
        return [
            _impute_random(responses, missing_mask, rng) for _ in range(n_imputations)
        ]

    theta_mean = scores.theta
    theta_se = scores.standard_error

    for _ in range(n_imputations):
        imputed = responses.copy()

        if theta_mean.ndim == 1:
            theta_draw = theta_mean + rng.standard_normal(n_persons) * theta_se
            theta_draw = theta_draw.reshape(-1, 1)
        else:
            theta_draw = theta_mean + rng.standard_normal(theta_mean.shape) * theta_se

        for j in range(n_items):
            item_missing = missing_mask[:, j]
            if not item_missing.any():
                continue

            theta_missing = theta_draw[item_missing]
            probs = result.model.probability(theta_missing, j)

            if probs.ndim > 1:
                missing_rows = np.flatnonzero(item_missing)
                imputed[missing_rows, j] = _draw_categorical(probs, rng)
            else:
                imputed[item_missing, j] = (
                    rng.random(item_missing.sum()) < probs
                ).astype(np.int_)

        imputations.append(imputed)

    return imputations


def analyze_missing(
    responses: NDArray[np.int_],
    missing_code: int = -1,
) -> dict[str, NDArray[np.float64] | float | int]:
    """Analyze missing data patterns.

    Parameters
    ----------
    responses : NDArray
        Response matrix
    missing_code : int
        Code used for missing values

    Returns
    -------
    dict
        Dictionary with:
        - 'total_missing_rate': Overall missing rate
        - 'item_missing_rate': Missing rate per item
        - 'person_missing_rate': Missing rate per person
        - 'n_complete_cases': Number of persons with no missing
        - 'n_complete_items': Number of items with no missing
    """
    responses = _prepare_response_matrix(responses, missing_code)
    missing_mask = responses == missing_code

    return {
        "total_missing_rate": float(missing_mask.mean()),
        "item_missing_rate": missing_mask.mean(axis=0).astype(np.float64),
        "person_missing_rate": missing_mask.mean(axis=1).astype(np.float64),
        "n_complete_cases": int((~missing_mask.any(axis=1)).sum()),
        "n_complete_items": int((~missing_mask.any(axis=0)).sum()),
    }


@dataclass(frozen=True, slots=True)
class MissingPatternResult:
    """Frequency summary of distinct missing-response patterns.

    ``patterns`` contains one Boolean row per distinct pattern, where ``True``
    denotes a missing response. Rows are ordered by decreasing frequency, with
    first appearance breaking ties. ``indices`` maps every input row back to
    its pattern row.
    """

    patterns: NDArray[np.bool_]
    frequencies: NDArray[np.int_]
    indices: NDArray[np.int_]
    n_persons: int
    n_items: int
    n_patterns: int

    @property
    def proportions(self) -> NDArray[np.float64]:
        """Proportion of respondents represented by each pattern."""
        return self.frequencies.astype(np.float64) / self.n_persons

    @property
    def missing_counts(self) -> NDArray[np.int_]:
        """Number of missing items in each pattern."""
        return np.sum(self.patterns, axis=1, dtype=np.int_)

    @property
    def complete_case_count(self) -> int:
        """Number of respondents with no missing responses."""
        complete = ~np.any(self.patterns, axis=1)
        return int(np.sum(self.frequencies, where=complete, initial=0))

    @property
    def complete_case_rate(self) -> float:
        """Proportion of respondents with no missing responses."""
        return self.complete_case_count / self.n_persons

    @property
    def compression_ratio(self) -> float:
        """Ratio of distinct patterns to respondents."""
        return self.n_patterns / self.n_persons

    def expand(self, pattern_values: NDArray) -> NDArray:
        """Expand pattern-level values back to respondent order."""
        values = np.asarray(pattern_values)
        if values.ndim == 0 or values.shape[0] != self.n_patterns:
            raise MirtValidationError(
                "pattern_values must have one leading entry per missing pattern",
                parameter="pattern_values",
                value=values.shape,
                expected=f"first dimension of {self.n_patterns}",
            )
        return values[self.indices]

    def to_dataframe(self, item_names: Sequence[str] | None = None) -> Any:
        """Return a pandas or Polars table when an optional backend is installed."""
        from mirt.utils.dataframe import create_dataframe

        if item_names is None:
            names = [f"Item_{index + 1}" for index in range(self.n_items)]
        else:
            if isinstance(item_names, (str, bytes)):
                raise MirtValidationError(
                    "item_names must be a sequence of unique strings",
                    parameter="item_names",
                )
            names = list(item_names)
            if len(names) != self.n_items:
                raise MirtValidationError(
                    "item_names length must match the number of items",
                    parameter="item_names",
                    value=len(names),
                    expected=str(self.n_items),
                )
            if not all(isinstance(name, str) and name for name in names) or len(
                set(names)
            ) != len(names):
                raise MirtValidationError(
                    "item_names must contain unique non-empty strings",
                    parameter="item_names",
                )

        data: dict[str, NDArray] = {
            f"{name}_missing": self.patterns[:, index]
            for index, name in enumerate(names)
        }
        data["n_missing"] = self.missing_counts
        data["frequency"] = self.frequencies
        data["proportion"] = self.proportions
        return create_dataframe(data, index_name="pattern")


def missing_patterns(
    responses: NDArray[np.int_],
    missing_code: int = -1,
) -> MissingPatternResult:
    """Summarize distinct missing-response patterns efficiently.

    Parameters
    ----------
    responses : ndarray of shape (n_persons, n_items)
        Response matrix. Negative values and ``missing_code`` are treated as
        missing using the same normalization as :func:`impute_responses`.
    missing_code : int, default=-1
        Code used to identify missing responses.

    Returns
    -------
    MissingPatternResult
        Packed-bit pattern summary with frequencies, proportions, and a
        respondent-to-pattern mapping.

    Notes
    -----
    Missingness rows are packed to one bit per item before grouping. This
    bounds temporary memory for wide item banks without adding a dependency.
    """
    normalized = _prepare_response_matrix(responses, missing_code)
    missing_mask = normalized == missing_code
    n_persons, n_items = missing_mask.shape

    packed = np.packbits(missing_mask, axis=1, bitorder="little")
    del missing_mask
    row_dtype = np.dtype((np.void, packed.shape[1]))
    packed_rows = np.ascontiguousarray(packed).view(row_dtype).ravel()
    _, first_indices, inverse, counts = np.unique(
        packed_rows,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )

    order = np.lexsort((first_indices, -counts))
    ordered_first = first_indices[order]
    patterns = np.unpackbits(
        packed[ordered_first],
        axis=1,
        count=n_items,
        bitorder="little",
    ).astype(np.bool_, copy=False)
    frequencies = counts[order].astype(np.int_, copy=False)

    sorted_to_frequency = np.empty(len(order), dtype=np.int_)
    sorted_to_frequency[order] = np.arange(len(order), dtype=np.int_)
    indices = sorted_to_frequency[inverse]

    return MissingPatternResult(
        patterns=patterns,
        frequencies=frequencies,
        indices=indices,
        n_persons=n_persons,
        n_items=n_items,
        n_patterns=len(order),
    )


def listwise_deletion(
    responses: NDArray[np.int_],
    missing_code: int = -1,
) -> NDArray[np.int_]:
    """Remove all rows (persons) with any missing values.

    Parameters
    ----------
    responses : NDArray
        Response matrix
    missing_code : int
        Code used for missing values

    Returns
    -------
    NDArray
        Response matrix with complete cases only
    """
    responses = _prepare_response_matrix(responses, missing_code)
    missing_mask = responses == missing_code
    complete_mask = ~missing_mask.any(axis=1)
    return responses[complete_mask]


def pairwise_available(
    responses: NDArray[np.int_],
    missing_code: int = -1,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Get counts for pairwise available analysis.

    Returns counts of valid pairs for each item pair.

    Parameters
    ----------
    responses : NDArray
        Response matrix
    missing_code : int
        Code used for missing values

    Returns
    -------
    tuple
        (n_available, joint_available) where:
        - n_available: (n_items,) count of valid responses per item
        - joint_available: (n_items, n_items) count of valid pairs
    """
    responses = _prepare_response_matrix(responses, missing_code)
    n_items = responses.shape[1]
    valid = responses != missing_code

    n_available = valid.sum(axis=0, dtype=np.int_)
    joint_available = np.zeros((n_items, n_items), dtype=np.int_)

    # A uint8 product is substantially faster than pairwise Python loops. Keep
    # blocks at 255 rows so each partial count remains within uint8 range, then
    # accumulate into platform-sized integers without overflow.
    for start in range(0, len(valid), _PAIRWISE_BLOCK_SIZE):
        block = valid[start : start + _PAIRWISE_BLOCK_SIZE].astype(np.uint8)
        joint_available += block.T @ block

    return n_available, joint_available


@dataclass
class MIResult:
    """Container for combined multiple imputation results.

    Attributes
    ----------
    estimate : float or NDArray[np.float64]
        Combined point estimate.
    within_variance : float or NDArray[np.float64]
        Average within-imputation variance.
    between_variance : float or NDArray[np.float64]
        Between-imputation variance.
    total_variance : float or NDArray[np.float64]
        Total variance (within + between + correction).
    standard_error : float or NDArray[np.float64]
        Standard error of combined estimate.
    df : float or NDArray[np.float64]
        Degrees of freedom for inference.
    fmi : float or NDArray[np.float64]
        Fraction of missing information.
    lambda_hat : float or NDArray[np.float64]
        Proportion of variation due to missingness.
    """

    estimate: float | NDArray[np.float64]
    within_variance: float | NDArray[np.float64]
    between_variance: float | NDArray[np.float64]
    total_variance: float | NDArray[np.float64]
    standard_error: float | NDArray[np.float64]
    df: float | NDArray[np.float64]
    fmi: float | NDArray[np.float64]
    lambda_hat: float | NDArray[np.float64]


def averageMI(
    estimates: Sequence[float | NDArray[np.float64]],
    variances: Sequence[float | NDArray[np.float64]] | None = None,
    standard_errors: Sequence[float | NDArray[np.float64]] | None = None,
) -> MIResult:
    """Combine results from multiple imputation using Rubin's rules.

    Combines point estimates and standard errors/variances from multiple
    imputations into a single inference, properly accounting for both
    within-imputation and between-imputation variability.

    Parameters
    ----------
    estimates : list of float or ndarray
        Point estimates from each imputation. Each element should be
        the same shape (scalar, 1D array, or 2D array).
    variances : list of float or ndarray, optional
        Variance estimates from each imputation. Provide exactly one of
        variances or standard_errors.
    standard_errors : list of float or ndarray, optional
        Standard errors from each imputation. Squared to get variances. Provide
        exactly one of standard_errors or variances.

    Returns
    -------
    MIResult
        Combined results with:
        - estimate: Combined point estimate
        - within_variance: Average within-imputation variance
        - between_variance: Between-imputation variance
        - total_variance: Total variance (Rubin's formula)
        - standard_error: Square root of total variance
        - df: Degrees of freedom for t-distribution
        - fmi: Fraction of missing information
        - lambda_hat: Proportion of variation due to missingness

    Examples
    --------
    >>> # Fit model on multiple imputations
    >>> imputations = impute_responses(data, method='multiple', n_imputations=5)
    >>> estimates = []
    >>> se_list = []
    >>> for imp in imputations:
    ...     result = fit_mirt(imp, model='2PL')
    ...     estimates.append(result.model.parameters['difficulty'])
    ...     se_list.append(result.standard_errors['difficulty'])
    >>> combined = averageMI(estimates, standard_errors=se_list)
    >>> print(f"Combined estimate: {combined.estimate}")
    >>> print(f"SE: {combined.standard_error}")

    Notes
    -----
    Uses Rubin's (1987) combining rules:
    - Combined estimate: Q_bar = mean(Q_m)
    - Within variance: U_bar = mean(U_m)
    - Between variance: B = var(Q_m)
    - Total variance: T = U_bar + (1 + 1/m) * B
    - Degrees of freedom: Rubin's large-sample approximation
    """
    if (variances is None) == (standard_errors is None):
        raise MirtValidationError("Provide exactly one of variances or standard_errors")

    m = len(estimates)
    if m < 2:
        raise MirtValidationError("Need at least 2 imputations to combine")

    try:
        estimates_arr = [np.asarray(e, dtype=np.float64) for e in estimates]
    except (TypeError, ValueError) as exc:
        raise MirtValidationError("estimates must contain numeric values") from exc

    estimate_shape = estimates_arr[0].shape
    if any(estimate.shape != estimate_shape for estimate in estimates_arr):
        raise MirtValidationError("All estimates must have the same shape")
    estimates_stacked = np.stack(estimates_arr, axis=0)
    if not np.all(np.isfinite(estimates_stacked)):
        raise MirtValidationError("All estimates must be finite")

    uncertainty_values = variances if variances is not None else standard_errors
    assert uncertainty_values is not None
    if len(uncertainty_values) != m:
        raise MirtValidationError(
            "The number of uncertainty estimates must match the imputations"
        )

    try:
        uncertainty_arr = [
            np.asarray(value, dtype=np.float64) for value in uncertainty_values
        ]
    except (TypeError, ValueError) as exc:
        raise MirtValidationError(
            "Uncertainty estimates must contain numeric values"
        ) from exc

    if any(value.shape != estimate_shape for value in uncertainty_arr):
        raise MirtValidationError("Uncertainty estimates must match the estimate shape")
    uncertainty_stacked = np.stack(uncertainty_arr, axis=0)
    if not np.all(np.isfinite(uncertainty_stacked)):
        raise MirtValidationError("All uncertainty estimates must be finite")
    if np.any(uncertainty_stacked < 0):
        kind = "Variances" if variances is not None else "Standard errors"
        raise MirtValidationError(f"{kind} must be nonnegative")

    if variances is not None:
        variances_stacked = uncertainty_stacked
    else:
        variances_stacked = uncertainty_stacked**2

    q_bar = estimates_stacked.mean(axis=0)
    u_bar = variances_stacked.mean(axis=0)
    b = estimates_stacked.var(axis=0, ddof=1)
    extra_variance = (1 + 1 / m) * b
    total_var = u_bar + extra_variance

    se = np.sqrt(total_var)

    lambda_hat = np.divide(
        extra_variance,
        total_var,
        out=np.zeros_like(total_var),
        where=total_var > 0,
    )
    df_old = np.full_like(total_var, LARGE_DF, dtype=np.float64)
    has_between_variance = extra_variance > 0
    relative_within = np.divide(
        u_bar,
        extra_variance,
        out=np.zeros_like(u_bar),
        where=has_between_variance,
    )
    df_old = np.where(
        has_between_variance,
        (m - 1) * (1 + relative_within) ** 2,
        df_old,
    )

    fmi = lambda_hat + 2 / (df_old + 3) * (1 - lambda_hat)
    fmi = np.clip(fmi, 0, 1)

    if q_bar.ndim == 0:
        return MIResult(
            estimate=float(q_bar),
            within_variance=float(u_bar),
            between_variance=float(b),
            total_variance=float(total_var),
            standard_error=float(se),
            df=float(df_old),
            fmi=float(fmi),
            lambda_hat=float(lambda_hat),
        )

    return MIResult(
        estimate=q_bar,
        within_variance=u_bar,
        between_variance=b,
        total_variance=total_var,
        standard_error=se,
        df=df_old,
        fmi=fmi,
        lambda_hat=lambda_hat,
    )
