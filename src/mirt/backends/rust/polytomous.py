"""Rust backend: polytomous."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.backends.rust._helpers import (
    RUST_AVAILABLE,
    mirt_rs,
)
from mirt.constants import PROB_EPSILON


def compute_log_likelihoods_grm(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    thresholds: NDArray[np.float64],
    n_categories: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute log-likelihoods for GRM at all quadrature points.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    quad_points : NDArray
        Quadrature points (n_quad,)
    discrimination : NDArray
        Item discrimination parameters (n_items,)
    thresholds : NDArray
        Threshold parameters (n_items, max_categories-1)
    n_categories : NDArray
        Number of categories per item (n_items,)

    Returns
    -------
    NDArray
        Log-likelihoods (n_persons, n_quad)
    """
    if RUST_AVAILABLE:
        return mirt_rs.compute_log_likelihoods_grm(
            responses.astype(np.int32),
            quad_points.astype(np.float64),
            discrimination.astype(np.float64),
            thresholds.astype(np.float64),
            n_categories.astype(np.int32),
        )

    n_persons, n_items = responses.shape
    n_quad = len(quad_points)
    log_likes = np.zeros((n_persons, n_quad))
    eps = PROB_EPSILON

    for q in range(n_quad):
        theta = quad_points[q]
        for i in range(n_persons):
            ll = 0.0
            for j in range(n_items):
                resp = responses[i, j]
                if resp < 0:
                    continue
                n_cat = n_categories[j]
                if resp == 0:
                    z = discrimination[j] * (theta - thresholds[j, 0])
                    p_above = sigmoid(z)
                    prob = max(1.0 - p_above, eps)
                elif resp == n_cat - 1:
                    z = discrimination[j] * (theta - thresholds[j, resp - 1])
                    prob = max(sigmoid(z), eps)
                else:
                    z_upper = discrimination[j] * (theta - thresholds[j, resp - 1])
                    z_lower = discrimination[j] * (theta - thresholds[j, resp])
                    p_upper = sigmoid(z_upper)
                    p_lower = sigmoid(z_lower)
                    prob = max(p_upper - p_lower, eps)
                ll += np.log(prob)
            log_likes[i, q] = ll

    return log_likes


def compute_log_likelihoods_gpcm(
    responses: NDArray[np.int_],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    steps: NDArray[np.float64],
    n_categories: NDArray[np.int_],
) -> NDArray[np.float64]:
    """Compute log-likelihoods for GPCM at all quadrature points.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    quad_points : NDArray
        Quadrature points (n_quad,)
    discrimination : NDArray
        Item discrimination parameters (n_items,)
    steps : NDArray
        Step parameters (n_items, max_categories)
    n_categories : NDArray
        Number of categories per item (n_items,)

    Returns
    -------
    NDArray
        Log-likelihoods (n_persons, n_quad)
    """
    if RUST_AVAILABLE:
        return mirt_rs.compute_log_likelihoods_gpcm(
            responses.astype(np.int32),
            quad_points.astype(np.float64),
            discrimination.astype(np.float64),
            steps.astype(np.float64),
            n_categories.astype(np.int32),
        )

    n_persons, n_items = responses.shape
    n_quad = len(quad_points)
    log_likes = np.zeros((n_persons, n_quad))

    for q in range(n_quad):
        theta = quad_points[q]
        for i in range(n_persons):
            ll = 0.0
            for j in range(n_items):
                resp = responses[i, j]
                if resp < 0:
                    continue
                n_cat = n_categories[j]
                a = discrimination[j]
                numerators = np.zeros(n_cat)
                for k in range(1, n_cat):
                    numerators[k] = numerators[k - 1] + a * (theta - steps[j, k])
                max_num = np.max(numerators)
                sum_exp = np.sum(np.exp(numerators - max_num))
                log_denom = max_num + np.log(sum_exp)
                prob = np.exp(numerators[resp] - log_denom)
                ll += np.log(max(prob, PROB_EPSILON))
            log_likes[i, q] = ll

    return log_likes


def compute_alpha_if_deleted(
    responses: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Cronbach's alpha if each item is deleted.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), NaN for missing

    Returns
    -------
    NDArray
        Alpha-if-deleted for each item (n_items,)
    """
    if RUST_AVAILABLE:
        return mirt_rs.compute_alpha_if_deleted(
            responses.astype(np.float64),
        )

    n_persons, n_items = responses.shape

    total_scores = np.nansum(responses, axis=1)

    item_variances = np.zeros(n_items)
    for j in range(n_items):
        col = responses[:, j]
        valid = ~np.isnan(col)
        if valid.sum() > 1:
            mean = np.nanmean(col)
            item_variances[j] = np.nansum((col[valid] - mean) ** 2) / (valid.sum() - 1)

    alpha_if_deleted = np.zeros(n_items)
    for j in range(n_items):
        remaining_scores = total_scores - np.where(
            np.isnan(responses[:, j]), 0, responses[:, j]
        )
        remaining_var_sum = np.sum(item_variances[np.arange(n_items) != j])
        remaining_mean = np.mean(remaining_scores)
        remaining_total_var = np.sum((remaining_scores - remaining_mean) ** 2) / max(
            n_persons - 1, 1
        )

        k = n_items - 1
        if remaining_total_var > 0 and k > 1:
            alpha_if_deleted[j] = (k / (k - 1)) * (
                1 - remaining_var_sum / remaining_total_var
            )
        else:
            alpha_if_deleted[j] = 0.0

    return alpha_if_deleted
