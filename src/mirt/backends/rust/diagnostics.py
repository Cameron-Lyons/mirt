"""Rust backend: diagnostics.

Fallback mode: numpy. All functions provide NumPy fallbacks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.backends.rust._helpers import (
    _ensure_f64,
    _ensure_i32,
    mirt_rs,
    rust_enabled,
)
from mirt.constants import PROB_EPSILON

FALLBACK_MODE = "numpy"


def _prepare_item_information_inputs(
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    h: float,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Validate and normalize inputs for 2PL item information blocks."""
    responses_array = np.asarray(responses)
    if responses_array.dtype.kind not in "biuf":
        raise ValueError("responses must contain numeric values")
    responses_float = np.asarray(responses_array, dtype=np.float64)
    posterior_array = np.asarray(posterior_weights, dtype=np.float64)
    quad_array = np.asarray(quad_points, dtype=np.float64)
    discrimination_array = np.asarray(discrimination, dtype=np.float64)
    difficulty_array = np.asarray(difficulty, dtype=np.float64)

    if responses_array.ndim != 2 or not all(responses_array.shape):
        raise ValueError("responses must be a non-empty 2D matrix")
    n_persons, n_items = responses_array.shape
    if (
        posterior_array.ndim != 2
        or posterior_array.shape[0] != n_persons
        or posterior_array.shape[1] == 0
    ):
        raise ValueError(
            "posterior_weights must be a 2D matrix with one row per respondent"
        )
    if quad_array.ndim != 1 or quad_array.shape[0] != posterior_array.shape[1]:
        raise ValueError(
            "quad_points must be one-dimensional with one value per posterior column"
        )
    if discrimination_array.shape != (n_items,):
        raise ValueError(f"discrimination must have shape ({n_items},)")
    if difficulty_array.shape != (n_items,):
        raise ValueError(f"difficulty must have shape ({n_items},)")

    if np.any(np.isinf(responses_float)):
        raise ValueError("responses must contain finite values or NaN for missing")
    observed = np.isfinite(responses_float) & (responses_float >= 0)
    if np.any(observed & (responses_float != 0) & (responses_float != 1)):
        raise ValueError("observed responses must contain only 0 or 1")
    if not np.all(np.isfinite(posterior_array)) or np.any(posterior_array < 0):
        raise ValueError("posterior_weights must contain finite non-negative values")
    if not np.all(np.isfinite(quad_array)):
        raise ValueError("quad_points must contain only finite values")
    if not np.all(np.isfinite(discrimination_array)):
        raise ValueError("discrimination must contain only finite values")
    if not np.all(np.isfinite(difficulty_array)):
        raise ValueError("difficulty must contain only finite values")
    if (
        isinstance(h, (bool, np.bool_))
        or not isinstance(h, (int, float, np.integer, np.floating))
        or not np.isfinite(h)
        or h <= 0
    ):
        raise ValueError("h must be a finite positive number")

    responses_i32 = _ensure_i32(
        np.where(np.isnan(responses_float), -1, responses_float)
    )
    posterior_f64 = _ensure_f64(posterior_array)
    quad_f64 = _ensure_f64(quad_array)
    discrimination_f64 = _ensure_f64(discrimination_array)
    difficulty_f64 = _ensure_f64(difficulty_array)
    assert responses_i32 is not None
    assert posterior_f64 is not None
    assert quad_f64 is not None
    assert discrimination_f64 is not None
    assert difficulty_f64 is not None
    return (
        responses_i32,
        posterior_f64,
        quad_f64,
        discrimination_f64,
        difficulty_f64,
    )


def _item_hessian_blocks(
    responses: NDArray[np.int32],
    posterior_weights: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute exact 2PL complete-data Hessian blocks for every item."""
    valid = responses >= 0
    correct = responses == 1
    expected_totals = valid.T @ posterior_weights
    expected_correct = correct.T @ posterior_weights

    centered_theta = quad_points[None, :] - difficulty[:, None]
    slopes = discrimination[:, None]
    probabilities = sigmoid(slopes * centered_theta)
    curvature = expected_totals * probabilities * (1.0 - probabilities)
    score_residual = expected_correct - expected_totals * probabilities

    hessian_aa = -np.sum(curvature * centered_theta**2, axis=1)
    hessian_bb = -np.sum(curvature * slopes**2, axis=1)
    hessian_ab = np.sum(
        curvature * slopes * centered_theta - score_residual,
        axis=1,
    )

    blocks = np.empty((responses.shape[1], 2, 2), dtype=np.float64)
    blocks[:, 0, 0] = hessian_aa
    blocks[:, 0, 1] = hessian_ab
    blocks[:, 1, 0] = hessian_ab
    blocks[:, 1, 1] = hessian_bb
    return blocks


def _standard_errors_from_blocks(
    blocks: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Invert negative 2x2 Hessian blocks and return marginal errors."""
    information_aa = -blocks[:, 0, 0]
    information_ab = -blocks[:, 0, 1]
    information_bb = -blocks[:, 1, 1]
    determinant = information_aa * information_bb - information_ab**2
    valid = (
        (information_aa > PROB_EPSILON)
        & (information_bb > PROB_EPSILON)
        & (determinant > PROB_EPSILON)
    )

    se_discrimination = np.full(blocks.shape[0], np.nan)
    se_difficulty = np.full(blocks.shape[0], np.nan)
    se_discrimination[valid] = np.sqrt(information_bb[valid] / determinant[valid])
    se_difficulty[valid] = np.sqrt(information_aa[valid] / determinant[valid])
    return se_discrimination, se_difficulty


def sibtest_compute_beta(
    ref_data: NDArray[np.int_],
    focal_data: NDArray[np.int_],
    ref_scores: NDArray[np.int_],
    focal_scores: NDArray[np.int_],
    suspect_items: NDArray[np.int_],
) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
    """Compute SIBTEST beta statistic."""
    if rust_enabled():
        return mirt_rs.sibtest_compute_beta(
            ref_data.astype(np.int32),
            focal_data.astype(np.int32),
            ref_scores.astype(np.int32),
            focal_scores.astype(np.int32),
            suspect_items.astype(np.int32),
        )

    all_scores = np.concatenate([ref_scores, focal_scores])
    unique_scores = np.unique(all_scores)

    beta_k = []
    n_k = []

    for k in unique_scores:
        ref_at_k = ref_data[ref_scores == k]
        focal_at_k = focal_data[focal_scores == k]

        n_ref_k = len(ref_at_k)
        n_focal_k = len(focal_at_k)

        if n_ref_k > 0 and n_focal_k > 0:
            mean_ref_k = ref_at_k[:, suspect_items].sum(axis=1).mean()
            mean_focal_k = focal_at_k[:, suspect_items].sum(axis=1).mean()
            beta_k.append(mean_ref_k - mean_focal_k)
            n_k.append(2 * n_ref_k * n_focal_k / (n_ref_k + n_focal_k))

    if not beta_k:
        return np.nan, np.nan, np.array([]), np.array([])

    beta_k = np.array(beta_k)
    n_k = np.array(n_k)
    beta = np.sum(n_k * beta_k) / np.sum(n_k)

    weighted_var = np.sum(n_k * (beta_k - beta) ** 2) / np.sum(n_k)
    n_total = len(ref_scores) + len(focal_scores)
    se = np.sqrt(weighted_var / n_total)

    return beta, se, beta_k, n_k


def sibtest_all_items(
    data: NDArray[np.int_],
    groups: NDArray[np.int_],
    anchor_items: NDArray[np.int_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Run SIBTEST for all items in parallel."""
    if rust_enabled():
        return mirt_rs.sibtest_all_items(
            data.astype(np.int32),
            groups.astype(np.int32),
            anchor_items.astype(np.int32) if anchor_items is not None else None,
        )

    from scipy import stats

    n_items = data.shape[1]
    unique_groups = np.unique(groups)
    ref_group, focal_group = unique_groups[0], unique_groups[1]

    ref_mask = groups == ref_group
    focal_mask = groups == focal_group

    betas = np.zeros(n_items)
    zs = np.zeros(n_items)
    p_values = np.zeros(n_items)

    for i in range(n_items):
        if anchor_items is None:
            matching = [j for j in range(n_items) if j != i]
        else:
            matching = [j for j in anchor_items if j != i]

        if not matching:
            betas[i] = np.nan
            zs[i] = np.nan
            p_values[i] = np.nan
            continue

        ref_scores = data[ref_mask][:, matching].sum(axis=1)
        focal_scores = data[focal_mask][:, matching].sum(axis=1)

        beta, se, _, _ = sibtest_compute_beta(
            data[ref_mask],
            data[focal_mask],
            ref_scores,
            focal_scores,
            np.array([i]),
        )

        betas[i] = beta
        if se > PROB_EPSILON:
            zs[i] = beta / se
            p_values[i] = 2 * (1 - stats.norm.cdf(abs(zs[i])))
        else:
            zs[i] = np.nan
            p_values[i] = np.nan

    return betas, zs, p_values


def compute_standardized_residuals(
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute standardized residuals for each person-item combination.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    theta : NDArray
        Ability estimates (n_persons,)
    discrimination : NDArray
        Item discrimination parameters (n_items,)
    difficulty : NDArray
        Item difficulty parameters (n_items,)

    Returns
    -------
    NDArray
        Standardized residuals (n_persons, n_items)
    """
    if rust_enabled():
        theta_f64 = _ensure_f64(theta)
        return mirt_rs.compute_standardized_residuals(
            _ensure_i32(responses),
            theta_f64.ravel()
            if theta_f64 is not None
            else theta.astype(np.float64).ravel(),
            _ensure_f64(discrimination),
            _ensure_f64(difficulty),
        )

    n_persons, n_items = responses.shape
    residuals = np.full((n_persons, n_items), np.nan)

    for j in range(n_items):
        z = discrimination[j] * (theta - difficulty[j])
        p = sigmoid(z)
        p = np.clip(p, PROB_EPSILON, 1 - PROB_EPSILON)
        variance = p * (1 - p)

        valid = responses[:, j] >= 0
        residuals[valid, j] = (responses[valid, j] - p[valid]) / np.sqrt(
            variance[valid] + PROB_EPSILON
        )

    return residuals


def compute_q3_matrix(
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute Yen's Q3 (residual correlation) matrix.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    theta : NDArray
        Ability estimates (n_persons,)
    discrimination : NDArray
        Item discrimination parameters (n_items,)
    difficulty : NDArray
        Item difficulty parameters (n_items,)

    Returns
    -------
    NDArray
        Q3 correlation matrix (n_items, n_items)
    """
    if rust_enabled():
        theta_f64 = _ensure_f64(theta)
        return mirt_rs.compute_q3_matrix(
            _ensure_i32(responses),
            theta_f64.ravel()
            if theta_f64 is not None
            else theta.astype(np.float64).ravel(),
            _ensure_f64(discrimination),
            _ensure_f64(difficulty),
        )

    residuals = compute_standardized_residuals(
        responses, theta, discrimination, difficulty
    )

    n_items = responses.shape[1]
    q3_matrix = np.zeros((n_items, n_items))

    for i in range(n_items):
        for j in range(i + 1, n_items):
            valid = (responses[:, i] >= 0) & (responses[:, j] >= 0)
            valid &= ~np.isnan(residuals[:, i]) & ~np.isnan(residuals[:, j])

            if valid.sum() > 2:
                r_i = residuals[valid, i]
                r_j = residuals[valid, j]
                q3 = np.corrcoef(r_i, r_j)[0, 1]
                q3_matrix[i, j] = q3
                q3_matrix[j, i] = q3

    return q3_matrix


def compute_ld_chi2_matrix(
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute LD chi-square statistics for all item pairs.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    theta : NDArray
        Ability estimates (n_persons,)
    discrimination : NDArray
        Item discrimination parameters (n_items,)
    difficulty : NDArray
        Item difficulty parameters (n_items,)

    Returns
    -------
    NDArray
        LD chi-square matrix (n_items, n_items)
    """
    if rust_enabled():
        theta_f64 = _ensure_f64(theta)
        return mirt_rs.compute_ld_chi2_matrix(
            _ensure_i32(responses),
            theta_f64.ravel()
            if theta_f64 is not None
            else theta.astype(np.float64).ravel(),
            _ensure_f64(discrimination),
            _ensure_f64(difficulty),
        )

    n_persons, n_items = responses.shape
    chi2_matrix = np.full((n_items, n_items), np.nan)

    for i in range(n_items):
        for j in range(i + 1, n_items):
            valid = (responses[:, i] >= 0) & (responses[:, j] >= 0)
            n_valid = valid.sum()

            if n_valid < 10:
                continue

            resp_i = responses[valid, i]
            resp_j = responses[valid, j]
            theta_valid = theta[valid]

            z_i = discrimination[i] * (theta_valid - difficulty[i])
            z_j = discrimination[j] * (theta_valid - difficulty[j])
            prob_i = sigmoid(z_i)
            prob_j = sigmoid(z_j)

            resp_i_bin = (resp_i > 0).astype(int)
            resp_j_bin = (resp_j > 0).astype(int)

            obs_00 = np.sum((resp_i_bin == 0) & (resp_j_bin == 0))
            obs_01 = np.sum((resp_i_bin == 0) & (resp_j_bin == 1))
            obs_10 = np.sum((resp_i_bin == 1) & (resp_j_bin == 0))
            obs_11 = np.sum((resp_i_bin == 1) & (resp_j_bin == 1))

            exp_00 = np.sum((1 - prob_i) * (1 - prob_j))
            exp_01 = np.sum((1 - prob_i) * prob_j)
            exp_10 = np.sum(prob_i * (1 - prob_j))
            exp_11 = np.sum(prob_i * prob_j)

            observed = np.array([obs_00, obs_01, obs_10, obs_11])
            expected = np.array([exp_00, exp_01, exp_10, exp_11])
            expected = np.maximum(expected, 0.5)

            chi2 = np.sum((observed - expected) ** 2 / expected)
            chi2_matrix[i, j] = chi2
            chi2_matrix[j, i] = chi2

    return chi2_matrix


def compute_item_se_parallel(
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    h: float = 1e-5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute per-item standard errors in parallel.

    Exploits block diagonal structure of the Hessian.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    posterior_weights : NDArray
        Posterior weights from E-step (n_persons, n_quad)
    quad_points : NDArray
        Quadrature points (n_quad,)
    discrimination : NDArray
        Discrimination parameters (n_items,)
    difficulty : NDArray
        Difficulty parameters (n_items,)
    h : float
        Retained for API compatibility. Exact analytic derivatives are used.

    Returns
    -------
    tuple
        (se_discrimination, se_difficulty)
    """
    responses, posterior_weights, quad_points, discrimination, difficulty = (
        _prepare_item_information_inputs(
            responses,
            posterior_weights,
            quad_points,
            discrimination,
            difficulty,
            h,
        )
    )

    if rust_enabled():
        return mirt_rs.compute_item_se_parallel(
            responses,
            posterior_weights,
            quad_points,
            discrimination,
            difficulty,
            h,
        )

    blocks = _item_hessian_blocks(
        responses,
        posterior_weights,
        quad_points,
        discrimination,
        difficulty,
    )
    return _standard_errors_from_blocks(blocks)


def compute_hessian_block_diagonal(
    responses: NDArray[np.int_],
    posterior_weights: NDArray[np.float64],
    quad_points: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    h: float = 1e-5,
) -> NDArray[np.float64]:
    """Compute full Hessian matrix exploiting block diagonal structure.

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items)
    posterior_weights : NDArray
        Posterior weights (n_persons, n_quad)
    quad_points : NDArray
        Quadrature points (n_quad,)
    discrimination : NDArray
        Discrimination parameters (n_items,)
    difficulty : NDArray
        Difficulty parameters (n_items,)
    h : float
        Retained for API compatibility. Exact analytic derivatives are used.

    Returns
    -------
    NDArray
        Hessian matrix (n_params, n_params) where n_params = n_items * 2
    """
    responses, posterior_weights, quad_points, discrimination, difficulty = (
        _prepare_item_information_inputs(
            responses,
            posterior_weights,
            quad_points,
            discrimination,
            difficulty,
            h,
        )
    )

    if rust_enabled():
        return mirt_rs.compute_hessian_block_diagonal(
            responses,
            posterior_weights,
            quad_points,
            discrimination,
            difficulty,
            h,
        )

    blocks = _item_hessian_blocks(
        responses,
        posterior_weights,
        quad_points,
        discrimination,
        difficulty,
    )
    n_items = discrimination.shape[0]
    n_params = n_items * 2
    hessian = np.zeros((n_params, n_params))
    indices = np.arange(n_items) * 2
    hessian[indices, indices] = blocks[:, 0, 0]
    hessian[indices, indices + 1] = blocks[:, 0, 1]
    hessian[indices + 1, indices] = blocks[:, 1, 0]
    hessian[indices + 1, indices + 1] = blocks[:, 1, 1]

    return hessian


def compute_fit_statistics(
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Compute item and person fit statistics (infit/outfit).

    Parameters
    ----------
    responses : NDArray
        Response matrix (n_persons, n_items), missing coded as negative
    theta : NDArray
        Ability estimates (n_persons,)
    discrimination : NDArray
        Item discrimination parameters (n_items,)
    difficulty : NDArray
        Item difficulty parameters (n_items,)

    Returns
    -------
    tuple
        (item_outfit, item_infit, person_outfit, person_infit)
    """
    if rust_enabled():
        return mirt_rs.compute_fit_statistics(
            responses.astype(np.int32),
            theta.astype(np.float64).ravel(),
            discrimination.astype(np.float64),
            difficulty.astype(np.float64),
        )

    n_persons, n_items = responses.shape

    z_sq = np.full((n_persons, n_items), np.nan)
    variance = np.full((n_persons, n_items), np.nan)

    for j in range(n_items):
        z = discrimination[j] * (theta - difficulty[j])
        p = sigmoid(z)
        p = np.clip(p, PROB_EPSILON, 1 - PROB_EPSILON)
        var = p * (1 - p)

        valid = responses[:, j] >= 0
        raw_resid = responses[valid, j] - p[valid]
        z_sq[valid, j] = (raw_resid**2) / (var[valid] + PROB_EPSILON)
        variance[valid, j] = var[valid]

    item_outfit = np.nanmean(z_sq, axis=0)
    item_infit = np.nansum(z_sq * variance, axis=0) / np.nansum(variance, axis=0)

    person_outfit = np.nanmean(z_sq, axis=1)
    person_infit = np.nansum(z_sq * variance, axis=1) / np.nansum(variance, axis=1)

    return item_outfit, item_infit, person_outfit, person_infit


def compute_probabilities_batch(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute probabilities for all items in batch (2PL model).

    Parameters
    ----------
    theta : NDArray
        Ability estimates (n_persons,)
    discrimination : NDArray
        Item discrimination parameters (n_items,)
    difficulty : NDArray
        Item difficulty parameters (n_items,)

    Returns
    -------
    NDArray
        Probabilities (n_persons, n_items)
    """
    if rust_enabled():
        return mirt_rs.compute_probabilities_batch(
            theta.astype(np.float64).ravel(),
            discrimination.astype(np.float64).ravel(),
            difficulty.astype(np.float64).ravel(),
        )

    z = discrimination[None, :] * (theta[:, None] - difficulty[None, :])
    return sigmoid(z)


def compute_probabilities_batch_3pl(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
    guessing: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute probabilities for all items in batch (3PL model).

    Parameters
    ----------
    theta : NDArray
        Ability estimates (n_persons,)
    discrimination : NDArray
        Item discrimination parameters (n_items,)
    difficulty : NDArray
        Item difficulty parameters (n_items,)
    guessing : NDArray
        Item guessing parameters (n_items,)

    Returns
    -------
    NDArray
        Probabilities (n_persons, n_items)
    """
    if rust_enabled():
        return mirt_rs.compute_probabilities_batch_3pl(
            theta.astype(np.float64).ravel(),
            discrimination.astype(np.float64).ravel(),
            difficulty.astype(np.float64).ravel(),
            guessing.astype(np.float64).ravel(),
        )

    z = discrimination[None, :] * (theta[:, None] - difficulty[None, :])
    p_star = sigmoid(z)
    return guessing[None, :] + (1 - guessing[None, :]) * p_star


def compute_expected_variance_batch(
    theta: NDArray[np.float64],
    discrimination: NDArray[np.float64],
    difficulty: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute expected values and variances for all items in batch.

    Parameters
    ----------
    theta : NDArray
        Ability estimates (n_persons,)
    discrimination : NDArray
        Item discrimination parameters (n_items,)
    difficulty : NDArray
        Item difficulty parameters (n_items,)

    Returns
    -------
    tuple
        (expected, variance) both shape (n_persons, n_items)
    """
    if rust_enabled():
        return mirt_rs.compute_expected_variance_batch(
            theta.astype(np.float64).ravel(),
            discrimination.astype(np.float64).ravel(),
            difficulty.astype(np.float64).ravel(),
        )

    probs = compute_probabilities_batch(theta, discrimination, difficulty)
    expected = probs
    variance = probs * (1 - probs)
    return expected, variance
