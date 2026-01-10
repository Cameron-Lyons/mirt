"""Fixed-item calibration and test equating functions.

Provides functions for calibrating new items to an existing scale
and equating test forms. Uses Rust backend for performance when available.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

try:
    from mirt._rust_backend import (
        RUST_AVAILABLE,
        fixed_calib_em,
    )
except ImportError:
    RUST_AVAILABLE = False


@dataclass
class CalibrationResult:
    """Result of fixed-item calibration.

    Attributes
    ----------
    new_discrimination : NDArray[np.float64]
        Calibrated discrimination parameters for new items.
    new_difficulty : NDArray[np.float64]
        Calibrated difficulty parameters for new items.
    theta : NDArray[np.float64]
        Estimated abilities on the anchor scale.
    log_likelihood : float
        Final log-likelihood.
    n_iterations : int
        Number of iterations.
    converged : bool
        Whether estimation converged.
    """

    new_discrimination: NDArray[np.float64]
    new_difficulty: NDArray[np.float64]
    theta: NDArray[np.float64]
    log_likelihood: float
    n_iterations: int
    converged: bool


@dataclass
class EquatingResult:
    """Result of test equating.

    Attributes
    ----------
    A : float
        Slope of linear transformation.
    B : float
        Intercept of linear transformation.
    method : str
        Equating method used.
    anchor_items : list[int]
        Indices of anchor items.
    rmse : float
        Root mean square error of equating.
    """

    A: float
    B: float
    method: str
    anchor_items: list[int]
    rmse: float


def fixed_calib(
    responses: NDArray[np.float64],
    anchor_model: "BaseItemModel",
    anchor_items: list[int],
    new_items: list[int] | None = None,
    model_type: str = "2PL",
    n_quadpts: int = 21,
    max_iter: int = 500,
    tol: float = 1e-4,
    use_rust: bool = True,
    disc_bounds: tuple[float, float] = (0.2, 5.0),
    diff_bounds: tuple[float, float] = (-5.0, 5.0),
    prob_clamp: tuple[float, float] = (0.01, 0.99),
    init_disc: float = 1.0,
    init_diff: float = 0.0,
    min_count: float = 1.0,
    min_valid_points: int = 3,
) -> CalibrationResult:
    """Calibrate new items while holding anchor items fixed.

    This function estimates parameters for new items on a scale
    defined by anchor items with known (fixed) parameters.

    Parameters
    ----------
    responses : NDArray[np.float64]
        Response matrix containing both anchor and new items.
        Shape: (n_persons, n_total_items).
    anchor_model : BaseItemModel
        Model with fixed parameters for anchor items.
    anchor_items : list of int
        Column indices of anchor items in responses.
    new_items : list of int, optional
        Column indices of new items. If None, all non-anchor items.
    model_type : str
        Model type for new items. Default "2PL".
    n_quadpts : int
        Number of quadrature points. Default 21.
    max_iter : int
        Maximum iterations. Default 500.
    tol : float
        Convergence tolerance. Default 1e-4.
    use_rust : bool
        Use Rust backend for performance. Default True.
    disc_bounds : tuple[float, float]
        Bounds for discrimination parameters (min, max). Default (0.2, 5.0).
    diff_bounds : tuple[float, float]
        Bounds for difficulty parameters (min, max). Default (-5.0, 5.0).
    prob_clamp : tuple[float, float]
        Bounds for probability clipping (min, max). Default (0.01, 0.99).
    init_disc : float
        Initial discrimination value. Default 1.0.
    init_diff : float
        Initial difficulty value. Default 0.0.
    min_count : float
        Minimum count threshold for valid quadrature points. Default 1.0.
    min_valid_points : int
        Minimum number of valid points for regression. Default 3.

    Returns
    -------
    CalibrationResult
        Calibrated parameters for new items.

    Examples
    --------
    >>> result = fixed_calib(
    ...     responses=all_responses,
    ...     anchor_model=existing_model,
    ...     anchor_items=[0, 1, 2, 3, 4],
    ... )
    >>> print(f"New item difficulties: {result.new_difficulty}")
    """
    from scipy import stats

    responses = np.asarray(responses, dtype=np.float64)
    n_persons, n_total_items = responses.shape

    if new_items is None:
        new_items = [i for i in range(n_total_items) if i not in anchor_items]

    anchor_disc = np.asarray(anchor_model.discrimination)
    anchor_diff = np.asarray(anchor_model.difficulty)

    if anchor_disc.ndim > 1:
        anchor_disc = anchor_disc[:, 0]

    can_use_rust = use_rust and RUST_AVAILABLE and model_type in ("2PL", "2pl")

    theta_grid = np.linspace(-4, 4, n_quadpts)
    weights = stats.norm.pdf(theta_grid)
    weights = weights / np.sum(weights)

    if can_use_rust:
        responses_int = np.where(np.isnan(responses), -1, responses).astype(np.int32)

        new_disc, new_diff, theta_est, log_likelihood, n_iterations, converged = (
            fixed_calib_em(
                responses_int,
                anchor_items,
                new_items,
                anchor_disc.astype(np.float64),
                anchor_diff.astype(np.float64),
                theta_grid.astype(np.float64),
                weights.astype(np.float64),
                max_iter,
                tol,
                disc_bounds,
                diff_bounds,
                prob_clamp,
                init_disc,
                init_diff,
                min_count,
                min_valid_points,
            )
        )

        return CalibrationResult(
            new_discrimination=np.asarray(new_disc),
            new_difficulty=np.asarray(new_diff),
            theta=np.asarray(theta_est),
            log_likelihood=float(log_likelihood),
            n_iterations=int(n_iterations),
            converged=bool(converged),
        )

    n_anchor = len(anchor_items)
    n_new = len(new_items)

    if anchor_disc.ndim == 1:
        anchor_disc = anchor_disc.reshape(-1, 1)

    new_disc = np.full((n_new, 1), init_disc)
    new_diff = np.full(n_new, init_diff)

    theta_grid = theta_grid.reshape(-1, 1)

    anchor_responses = responses[:, anchor_items]
    new_responses = responses[:, new_items]

    def compute_anchor_likelihood(theta_grid):
        n_quad = len(theta_grid)
        ll = np.zeros((n_persons, n_quad))

        for q in range(n_quad):
            theta_q = theta_grid[q]
            logit = anchor_disc[:, 0] * (theta_q - anchor_diff)
            probs = 1 / (1 + np.exp(-logit))
            probs = np.clip(probs, 1e-10, 1 - 1e-10)

            for j in range(n_anchor):
                mask = ~np.isnan(anchor_responses[:, j])
                ll[mask, q] += anchor_responses[mask, j] * np.log(probs[j]) + (
                    1 - anchor_responses[mask, j]
                ) * np.log(1 - probs[j])

        return ll

    anchor_ll = compute_anchor_likelihood(theta_grid)

    converged = False
    log_likelihood = -np.inf

    for iteration in range(max_iter):
        posterior = np.zeros((n_persons, n_quadpts))

        for q in range(n_quadpts):
            theta_q = theta_grid[q, 0]

            logit_new = new_disc[:, 0] * (theta_q - new_diff)
            probs_new = 1 / (1 + np.exp(-logit_new))
            probs_new = np.clip(probs_new, 1e-10, 1 - 1e-10)

            ll_new = np.zeros(n_persons)
            for j in range(n_new):
                mask = ~np.isnan(new_responses[:, j])
                ll_new[mask] += new_responses[mask, j] * np.log(probs_new[j]) + (
                    1 - new_responses[mask, j]
                ) * np.log(1 - probs_new[j])

            posterior[:, q] = np.exp(anchor_ll[:, q] + ll_new) * weights[q]

        posterior_sum = np.sum(posterior, axis=1, keepdims=True)
        posterior = posterior / np.maximum(posterior_sum, 1e-10)

        new_ll = np.sum(np.log(np.maximum(posterior_sum.ravel(), 1e-10)))

        if abs(new_ll - log_likelihood) < tol:
            converged = True
            log_likelihood = new_ll
            break

        log_likelihood = new_ll

        old_disc = new_disc.copy()
        old_diff = new_diff.copy()

        for j in range(n_new):
            mask = ~np.isnan(new_responses[:, j])
            if np.sum(mask) < 10:
                continue

            r_j = np.zeros(n_quadpts)
            n_j = np.zeros(n_quadpts)

            for q in range(n_quadpts):
                r_j[q] = np.sum(posterior[mask, q] * new_responses[mask, j])
                n_j[q] = np.sum(posterior[mask, q])

            p_j = r_j / np.maximum(n_j, 1e-10)
            p_j = np.clip(p_j, prob_clamp[0], prob_clamp[1])

            logit_j = np.log(p_j / (1 - p_j))
            valid = n_j > min_count

            if np.sum(valid) >= min_valid_points:
                theta_valid = theta_grid[valid, 0]
                logit_valid = logit_j[valid]
                weights_valid = n_j[valid]

                mean_theta = np.average(theta_valid, weights=weights_valid)
                mean_logit = np.average(logit_valid, weights=weights_valid)

                var_theta = np.average(
                    (theta_valid - mean_theta) ** 2, weights=weights_valid
                )
                cov_theta_logit = np.average(
                    (theta_valid - mean_theta) * (logit_valid - mean_logit),
                    weights=weights_valid,
                )

                if var_theta > 1e-10:
                    new_disc[j, 0] = np.clip(
                        cov_theta_logit / var_theta, disc_bounds[0], disc_bounds[1]
                    )
                    new_diff[j] = mean_theta - mean_logit / new_disc[j, 0]
                    new_diff[j] = np.clip(new_diff[j], diff_bounds[0], diff_bounds[1])

        param_change = np.max(np.abs(new_disc - old_disc)) + np.max(
            np.abs(new_diff - old_diff)
        )
        if param_change < tol:
            converged = True
            break

    theta_est = np.sum(posterior * theta_grid.T, axis=1)

    return CalibrationResult(
        new_discrimination=new_disc.ravel(),
        new_difficulty=new_diff,
        theta=theta_est,
        log_likelihood=log_likelihood,
        n_iterations=iteration + 1,
        converged=converged,
    )


def equate(
    model_old: "BaseItemModel",
    model_new: "BaseItemModel",
    anchor_items_old: list[int],
    anchor_items_new: list[int],
    method: Literal[
        "mean_sigma", "mean_mean", "stocking_lord", "haebara"
    ] = "stocking_lord",
) -> EquatingResult:
    """Equate two test forms using anchor items.

    Finds transformation constants A and B such that:
        theta_new = A * theta_old + B

    Parameters
    ----------
    model_old : BaseItemModel
        Model for old/reference form.
    model_new : BaseItemModel
        Model for new form.
    anchor_items_old : list of int
        Indices of anchor items in old model.
    anchor_items_new : list of int
        Indices of anchor items in new model.
    method : str
        Equating method:
        - "mean_sigma": Mean/sigma method
        - "mean_mean": Mean/mean method
        - "stocking_lord": Stocking-Lord method (characteristic curve)
        - "haebara": Haebara method

    Returns
    -------
    EquatingResult
        Transformation constants and diagnostics.

    Examples
    --------
    >>> eq = equate(old_model, new_model, [0,1,2], [0,1,2])
    >>> theta_equated = eq.A * theta_new + eq.B
    """
    disc_old = np.asarray(model_old.discrimination)[anchor_items_old]
    diff_old = np.asarray(model_old.difficulty)[anchor_items_old]
    disc_new = np.asarray(model_new.discrimination)[anchor_items_new]
    diff_new = np.asarray(model_new.difficulty)[anchor_items_new]

    if disc_old.ndim > 1:
        disc_old = disc_old[:, 0]
    if disc_new.ndim > 1:
        disc_new = disc_new[:, 0]

    if method == "mean_sigma":
        A = np.std(disc_old) / np.std(disc_new)
        B = np.mean(diff_old) - A * np.mean(diff_new)

    elif method == "mean_mean":
        A = np.mean(disc_old) / np.mean(disc_new)
        B = np.mean(diff_old) - A * np.mean(diff_new)

    elif method == "stocking_lord":
        from scipy.optimize import minimize

        def criterion(params):
            A, B = params
            theta = np.linspace(-4, 4, 41)

            total_diff = 0.0
            for j in range(len(anchor_items_old)):
                p_old = 1 / (1 + np.exp(-disc_old[j] * (theta - diff_old[j])))
                theta_trans = A * theta + B
                p_new = 1 / (1 + np.exp(-disc_new[j] * (theta_trans - diff_new[j])))
                total_diff += np.sum((p_old - p_new) ** 2)

            return total_diff

        result = minimize(criterion, [1.0, 0.0], method="Nelder-Mead")
        A, B = result.x

    elif method == "haebara":
        from scipy.optimize import minimize

        def criterion(params):
            A, B = params
            theta = np.linspace(-4, 4, 41)

            total_diff = 0.0
            for j in range(len(anchor_items_old)):
                p_old = 1 / (1 + np.exp(-disc_old[j] * (theta - diff_old[j])))
                theta_trans = A * theta + B
                p_new = 1 / (1 + np.exp(-disc_new[j] * (theta_trans - diff_new[j])))

                diff_sq = (p_old - p_new) ** 2
                total_diff += np.sum(diff_sq)

            return total_diff

        result = minimize(criterion, [1.0, 0.0], method="Nelder-Mead")
        A, B = result.x

    else:
        raise ValueError(f"Unknown equating method: {method}")

    disc_new_trans = disc_new / A
    diff_new_trans = A * diff_new + B
    rmse = np.sqrt(
        np.mean((disc_old - disc_new_trans) ** 2)
        + np.mean((diff_old - diff_new_trans) ** 2)
    )

    return EquatingResult(
        A=float(A),
        B=float(B),
        method=method,
        anchor_items=anchor_items_old,
        rmse=float(rmse),
    )


def transform_theta(
    theta: NDArray[np.float64],
    equating_result: EquatingResult,
) -> NDArray[np.float64]:
    """Transform theta estimates using equating constants.

    Parameters
    ----------
    theta : NDArray[np.float64]
        Theta estimates from new form.
    equating_result : EquatingResult
        Result from equate() function.

    Returns
    -------
    NDArray[np.float64]
        Transformed theta on old/reference scale.
    """
    return equating_result.A * np.asarray(theta) + equating_result.B
