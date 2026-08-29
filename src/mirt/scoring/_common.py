"""Shared helper utilities for scoring implementations."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.estimation.quadrature import GaussHermiteQuadrature

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


def observed_test_information(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    observed_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Return test information contributed by observed response items."""
    observed_mask = np.asarray(observed_mask, dtype=np.bool_)
    if observed_mask.shape != (model.n_items,):
        raise ValueError(
            f"observed_mask must have shape ({model.n_items},), "
            f"got {observed_mask.shape}"
        )

    n_theta = theta.shape[0]
    observed_items = np.flatnonzero(observed_mask)
    if observed_items.size == 0:
        return np.zeros(n_theta, dtype=np.float64)

    # Dichotomous models expose an itemwise information matrix, so use their
    # vectorized path. Polytomous models expose only total information in bulk.
    if not model.is_polytomous:
        item_information = np.asarray(model.information(theta), dtype=np.float64)
        if item_information.shape == (n_theta, model.n_items):
            return item_information[:, observed_mask].sum(axis=1)

    information = np.zeros(n_theta, dtype=np.float64)
    for item_idx in observed_items:
        item_information = np.asarray(
            model.information(theta, int(item_idx)), dtype=np.float64
        )
        information += item_information.reshape(n_theta, -1).sum(axis=1)

    return information


def resolve_prior_distribution(
    *,
    n_factors: int,
    prior_mean: NDArray[np.float64] | None,
    prior_cov: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return prior mean/covariance defaults for scoring."""
    if prior_mean is None:
        mean = np.zeros(n_factors, dtype=np.float64)
    else:
        mean = np.asarray(prior_mean, dtype=np.float64)

    if prior_cov is None:
        cov = np.eye(n_factors, dtype=np.float64)
    else:
        cov = np.asarray(prior_cov, dtype=np.float64)

    return mean, cov


def build_quadrature(
    *,
    n_quadpts: int,
    n_factors: int,
    prior_mean: NDArray[np.float64] | None,
    prior_cov: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build Gauss-Hermite quadrature nodes and weights from prior settings."""
    mean, cov = resolve_prior_distribution(
        n_factors=n_factors,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
    )
    quadrature = GaussHermiteQuadrature(
        n_points=n_quadpts,
        n_dimensions=n_factors,
        mean=mean,
        cov=cov,
    )
    return quadrature.nodes, quadrature.weights


def resolve_n_jobs(n_jobs: int) -> int:
    """Resolve n_jobs configuration, including -1 for all cores."""
    if (
        isinstance(n_jobs, (bool, np.bool_))
        or not isinstance(n_jobs, (int, np.integer))
        or n_jobs == 0
        or n_jobs < -1
    ):
        raise ValueError("n_jobs must be -1 or a positive integer")
    if n_jobs == -1:
        return os.cpu_count() or 1
    return int(n_jobs)


def validate_scoring_responses(
    model: BaseItemModel,
    responses: NDArray[np.int_],
) -> NDArray[np.int_]:
    """Validate a scoring matrix and normalize every missing code to -1."""
    raw = np.asarray(responses)
    if raw.ndim != 2:
        raise ValueError(f"responses must be 2D, got {raw.ndim}D")
    if raw.shape[1] != model.n_items:
        raise ValueError(
            f"responses has {raw.shape[1]} items, expected {model.n_items}"
        )
    if raw.dtype.kind not in "biuf":
        raise ValueError("responses must contain numeric values")

    if raw.dtype.kind in "biu":
        values = raw
    else:
        values = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("responses must contain only finite values")
    observed = values >= 0.0
    if values.dtype.kind == "f" and np.any(
        values[observed] != np.floor(values[observed])
    ):
        raise ValueError("observed responses must be integer-valued")

    if model.is_polytomous:
        category_counts = np.asarray(model.n_categories, dtype=np.float64)
        invalid = observed & (values >= category_counts[None, :])
        if np.any(invalid):
            raise ValueError("observed responses exceed an item's category range")
    elif np.any(observed & (values > 1.0)):
        raise ValueError(
            "dichotomous responses must contain only 0, 1, or missing values"
        )

    if values.dtype == np.dtype(np.int_) and np.all(observed | (values == -1)):
        return values
    return np.where(observed, values, -1.0).astype(np.int_, copy=False)


def unique_response_patterns(
    responses: NDArray[np.int_],
) -> tuple[NDArray[np.int_], NDArray[np.intp]]:
    """Collapse duplicate response rows without sorting every item column."""
    if responses.shape[0] == 0:
        return responses.copy(), np.empty(0, dtype=np.intp)

    contiguous = np.ascontiguousarray(responses)
    minimum = int(np.min(contiguous))
    maximum = int(np.max(contiguous))
    if np.iinfo(np.int8).min <= minimum and maximum <= np.iinfo(np.int8).max:
        key_values = contiguous.astype(np.int8)
    elif np.iinfo(np.int16).min <= minimum and maximum <= np.iinfo(np.int16).max:
        key_values = contiguous.astype(np.int16)
    elif np.iinfo(np.int32).min <= minimum and maximum <= np.iinfo(np.int32).max:
        key_values = contiguous.astype(np.int32)
    else:
        key_values = contiguous
    row_dtype = np.dtype((np.void, key_values.dtype.itemsize * key_values.shape[1]))
    row_keys = key_values.view(row_dtype).reshape(-1)
    _, first_indices, inverse = np.unique(
        row_keys,
        return_index=True,
        return_inverse=True,
    )
    patterns = contiguous[first_indices]
    return patterns, inverse.astype(np.intp, copy=False)


def finite_difference_se(
    objective: Callable[[float], float],
    estimate: float,
    *,
    step: float = 1e-5,
) -> float:
    """Estimate SE from second finite difference of a scalar objective."""
    f_plus = objective(estimate + step)
    f_minus = objective(estimate - step)
    f_center = objective(estimate)
    hessian = (f_plus - 2 * f_center + f_minus) / (step**2)

    if hessian > 0:
        return float(np.sqrt(1.0 / hessian))
    return float(np.nan)


def score_responses_parallel(
    *,
    model: BaseItemModel,
    responses: NDArray[np.int_],
    n_jobs: int,
    score_person: Callable[
        [int],
        tuple[float | NDArray[np.float64], float | NDArray[np.float64]],
    ],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Execute per-person scoring either serially or with a thread pool."""
    n_persons = responses.shape[0]
    n_factors = model.n_factors

    theta_values = np.zeros((n_persons, n_factors), dtype=np.float64)
    se_values = np.zeros((n_persons, n_factors), dtype=np.float64)

    if n_persons == 0:
        if n_factors == 1:
            return theta_values.ravel(), se_values.ravel()
        return theta_values, se_values

    worker_count = resolve_n_jobs(n_jobs)

    if worker_count == 1:
        results = map(score_person, range(n_persons))
    else:
        with ThreadPoolExecutor(max_workers=min(worker_count, n_persons)) as executor:
            results = executor.map(score_person, range(n_persons))

    for i, (theta_est, se_est) in enumerate(results):
        if n_factors == 1:
            theta_values[i, 0] = float(
                np.asarray(theta_est, dtype=np.float64).ravel()[0]
            )
            se_values[i, 0] = float(np.asarray(se_est, dtype=np.float64).ravel()[0])
        else:
            theta_values[i] = np.asarray(theta_est, dtype=np.float64).reshape(n_factors)
            se_values[i] = np.asarray(se_est, dtype=np.float64).reshape(n_factors)

    if n_factors == 1:
        return theta_values.ravel(), se_values.ravel()
    return theta_values, se_values
