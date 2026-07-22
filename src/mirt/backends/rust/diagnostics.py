"""Rust backend: diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mirt._core import sigmoid
from mirt.backends.rust._helpers import (
    RUST_AVAILABLE,
    _ensure_f64,
    _ensure_i32,
    mirt_rs,
)
from mirt.constants import PROB_EPSILON


def sibtest_compute_beta(
    ref_data: NDArray[np.int_],
    focal_data: NDArray[np.int_],
    ref_scores: NDArray[np.int_],
    focal_scores: NDArray[np.int_],
    suspect_items: NDArray[np.int_],
) -> tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
    """Compute SIBTEST beta statistic."""
    if RUST_AVAILABLE:
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
    if RUST_AVAILABLE:
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
    if RUST_AVAILABLE:
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
    if RUST_AVAILABLE:
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
    if RUST_AVAILABLE:
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
        Step size for finite difference

    Returns
    -------
    tuple
        (se_discrimination, se_difficulty)
    """
    if RUST_AVAILABLE:
        qp = _ensure_f64(quad_points)
        disc = _ensure_f64(discrimination)
        diff = _ensure_f64(difficulty)
        return mirt_rs.compute_item_se_parallel(
            _ensure_i32(responses),
            _ensure_f64(posterior_weights),
            qp.ravel() if qp is not None else quad_points.astype(np.float64).ravel(),
            disc.ravel()
            if disc is not None
            else discrimination.astype(np.float64).ravel(),
            diff.ravel() if diff is not None else difficulty.astype(np.float64).ravel(),
            h,
        )

    n_items = responses.shape[1]
    se_disc = np.zeros(n_items)
    se_diff = np.zeros(n_items)

    for j in range(n_items):
        item_responses = responses[:, j]
        valid_mask = item_responses >= 0

        r_k = np.sum(
            item_responses[valid_mask, None] * posterior_weights[valid_mask, :],
            axis=0,
        )
        n_k = np.sum(posterior_weights[valid_mask], axis=0)

        def item_ll(a, b):
            z = a * (quad_points - b)
            p = sigmoid(z)
            p = np.clip(p, PROB_EPSILON, 1 - PROB_EPSILON)
            return np.sum(r_k * np.log(p) + (n_k - r_k) * np.log(1 - p))

        a, b = discrimination[j], difficulty[j]
        ll_center = item_ll(a, b)

        ll_a_plus = item_ll(a + h, b)
        ll_a_minus = item_ll(a - h, b)
        hess_aa = (ll_a_plus - 2 * ll_center + ll_a_minus) / (h**2)
        se_disc[j] = np.sqrt(-1.0 / hess_aa) if hess_aa < -PROB_EPSILON else np.nan

        ll_b_plus = item_ll(a, b + h)
        ll_b_minus = item_ll(a, b - h)
        hess_bb = (ll_b_plus - 2 * ll_center + ll_b_minus) / (h**2)
        se_diff[j] = np.sqrt(-1.0 / hess_bb) if hess_bb < -PROB_EPSILON else np.nan

    return se_disc, se_diff


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
        Step size for finite difference

    Returns
    -------
    NDArray
        Hessian matrix (n_params, n_params) where n_params = n_items * 2
    """
    if RUST_AVAILABLE:
        qp = _ensure_f64(quad_points)
        disc = _ensure_f64(discrimination)
        diff = _ensure_f64(difficulty)
        return mirt_rs.compute_hessian_block_diagonal(
            _ensure_i32(responses),
            _ensure_f64(posterior_weights),
            qp.ravel() if qp is not None else quad_points.astype(np.float64).ravel(),
            disc.ravel()
            if disc is not None
            else discrimination.astype(np.float64).ravel(),
            diff.ravel() if diff is not None else difficulty.astype(np.float64).ravel(),
            h,
        )

    n_items = len(discrimination)
    n_params = n_items * 2
    hessian = np.zeros((n_params, n_params))

    se_disc, se_diff = compute_item_se_parallel(
        responses, posterior_weights, quad_points, discrimination, difficulty, h
    )

    for j in range(n_items):
        idx_a = j * 2
        idx_b = j * 2 + 1

        if not np.isnan(se_disc[j]):
            hessian[idx_a, idx_a] = -1.0 / (se_disc[j] ** 2)
        if not np.isnan(se_diff[j]):
            hessian[idx_b, idx_b] = -1.0 / (se_diff[j] ** 2)

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
    if RUST_AVAILABLE:
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
    if RUST_AVAILABLE:
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
    if RUST_AVAILABLE:
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
    if RUST_AVAILABLE:
        return mirt_rs.compute_expected_variance_batch(
            theta.astype(np.float64).ravel(),
            discrimination.astype(np.float64).ravel(),
            difficulty.astype(np.float64).ravel(),
        )

    probs = compute_probabilities_batch(theta, discrimination, difficulty)
    expected = probs
    variance = probs * (1 - probs)
    return expected, variance
