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
    if n_jobs == -1:
        return os.cpu_count() or 1
    return n_jobs


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
