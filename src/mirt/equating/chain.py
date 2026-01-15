"""Chain linking across multiple time points.

This module provides functions for linking IRT models across
multiple time points or administrations using pairwise linking.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

from mirt.equating.linking import LinkingResult, link


@dataclass
class ChainLinkingResult:
    """Result of chain linking across multiple time points.

    Attributes
    ----------
    cumulative_A : list[float]
        Cumulative slope transformations to reference scale.
    cumulative_B : list[float]
        Cumulative intercept transformations to reference scale.
    pairwise_results : list[LinkingResult]
        Results from each pairwise linking.
    drift_accumulation : NDArray[np.float64] | None
        Accumulated drift statistics.
    reference_index : int
        Index of reference time point.
    """

    cumulative_A: list[float]
    cumulative_B: list[float]
    pairwise_results: list[LinkingResult]
    drift_accumulation: NDArray[np.float64] | None
    reference_index: int


@dataclass
class TimePointModel:
    """Model and anchor information for a time point.

    Attributes
    ----------
    model : BaseItemModel
        Fitted model for this time point.
    anchor_items : list[int]
        Indices of anchor items in this model.
    time_label : str
        Label for this time point.
    """

    model: "BaseItemModel"
    anchor_items: list[int]
    time_label: str = ""


def chain_link(
    models: list["BaseItemModel"],
    anchor_item_pairs: list[tuple[list[int], list[int]]],
    method: str = "stocking_lord",
    reference_index: int = 0,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
    compute_drift: bool = True,
) -> ChainLinkingResult:
    """Perform chain linking across multiple time points.

    Links a sequence of models to a common reference scale by
    accumulating pairwise transformations.

    Parameters
    ----------
    models : list[BaseItemModel]
        List of models in temporal order.
    anchor_item_pairs : list[tuple[list[int], list[int]]]
        List of (anchors_t, anchors_t+1) pairs for consecutive models.
        Length should be len(models) - 1.
    method : str
        Linking method for pairwise linking.
    reference_index : int
        Index of reference model (default 0 = first).
    theta_range : tuple[float, float]
        Range for curve matching methods.
    n_theta : int
        Number of theta points.
    compute_drift : bool
        Whether to track drift accumulation.

    Returns
    -------
    ChainLinkingResult
        Cumulative transformations and pairwise results.

    Examples
    --------
    >>> result = chain_link(
    ...     models=[model_t1, model_t2, model_t3],
    ...     anchor_item_pairs=[
    ...         ([0, 1, 2], [0, 1, 2]),  # t1 -> t2
    ...         ([0, 1, 2], [0, 1, 2]),  # t2 -> t3
    ...     ],
    ...     reference_index=0,
    ... )
    >>> # Transform t3 parameters to t1 scale
    >>> A_t3_to_t1 = result.cumulative_A[2]
    >>> B_t3_to_t1 = result.cumulative_B[2]
    """
    n_models = len(models)

    if len(anchor_item_pairs) != n_models - 1:
        raise ValueError(
            f"Expected {n_models - 1} anchor pairs, got {len(anchor_item_pairs)}"
        )

    if reference_index < 0 or reference_index >= n_models:
        raise ValueError(f"Invalid reference_index: {reference_index}")

    pairwise_results: list[LinkingResult] = []

    for t in range(n_models - 1):
        anchors_t, anchors_t1 = anchor_item_pairs[t]

        result = link(
            models[t],
            models[t + 1],
            anchors_t,
            anchors_t1,
            method=method,
            theta_range=theta_range,
            n_theta=n_theta,
            compute_diagnostics=True,
        )
        pairwise_results.append(result)

    pairwise_A = [r.constants.A for r in pairwise_results]
    pairwise_B = [r.constants.B for r in pairwise_results]

    cumulative_A, cumulative_B = accumulate_constants(
        pairwise_A, pairwise_B, reference_index
    )

    drift_accumulation = None
    if compute_drift:
        drift_accumulation = _compute_drift_accumulation(pairwise_results)

    return ChainLinkingResult(
        cumulative_A=cumulative_A,
        cumulative_B=cumulative_B,
        pairwise_results=pairwise_results,
        drift_accumulation=drift_accumulation,
        reference_index=reference_index,
    )


def accumulate_constants(
    pairwise_A: list[float],
    pairwise_B: list[float],
    reference_index: int = 0,
) -> tuple[list[float], list[float]]:
    """Accumulate pairwise constants to reference scale.

    For transformations A_t, B_t from time t to t+1:
        theta_ref = A_cum * theta_t + B_cum

    Parameters
    ----------
    pairwise_A : list[float]
        Pairwise slope constants.
    pairwise_B : list[float]
        Pairwise intercept constants.
    reference_index : int
        Index of reference time point.

    Returns
    -------
    tuple[list[float], list[float]]
        Cumulative A and B for each time point.
    """
    n_models = len(pairwise_A) + 1

    cumulative_A = [1.0] * n_models
    cumulative_B = [0.0] * n_models

    for t in range(reference_index - 1, -1, -1):
        A_t = pairwise_A[t]
        B_t = pairwise_B[t]
        cumulative_A[t] = A_t * cumulative_A[t + 1]
        cumulative_B[t] = A_t * cumulative_B[t + 1] + B_t

    for t in range(reference_index + 1, n_models):
        A_prev = pairwise_A[t - 1]
        B_prev = pairwise_B[t - 1]
        cumulative_A[t] = cumulative_A[t - 1] / A_prev
        cumulative_B[t] = (cumulative_B[t - 1] - B_prev) / A_prev

    return cumulative_A, cumulative_B


def _compute_drift_accumulation(
    pairwise_results: list[LinkingResult],
) -> NDArray[np.float64]:
    """Compute accumulated drift across time points."""
    n_pairs = len(pairwise_results)

    max_anchors = max(
        len(r.anchor_items) for r in pairwise_results if r.anchor_diagnostics
    )

    drift_matrix = np.zeros((n_pairs, max_anchors))
    drift_matrix[:] = np.nan

    for t, result in enumerate(pairwise_results):
        if result.anchor_diagnostics is not None:
            n_anchors = len(result.anchor_diagnostics.robust_z)
            drift_matrix[t, :n_anchors] = result.anchor_diagnostics.robust_z

    return drift_matrix


def transform_to_reference(
    model: "BaseItemModel",
    chain_result: ChainLinkingResult,
    time_index: int,
    in_place: bool = False,
) -> "BaseItemModel":
    """Transform model parameters to reference scale.

    Parameters
    ----------
    model : BaseItemModel
        Model to transform.
    chain_result : ChainLinkingResult
        Chain linking result.
    time_index : int
        Index of model's time point.
    in_place : bool
        Modify in place or return copy.

    Returns
    -------
    BaseItemModel
        Model on reference scale.
    """
    from mirt.equating.linking import transform_parameters

    A = chain_result.cumulative_A[time_index]
    B = chain_result.cumulative_B[time_index]

    return transform_parameters(model, A, B, in_place=in_place)


def transform_theta_to_reference(
    theta: NDArray[np.float64],
    chain_result: ChainLinkingResult,
    time_index: int,
) -> NDArray[np.float64]:
    """Transform theta estimates to reference scale.

    Parameters
    ----------
    theta : NDArray
        Theta estimates from time point.
    chain_result : ChainLinkingResult
        Chain linking result.
    time_index : int
        Index of time point.

    Returns
    -------
    NDArray
        Theta on reference scale.
    """
    A = chain_result.cumulative_A[time_index]
    B = chain_result.cumulative_B[time_index]

    return A * theta + B


def concurrent_link(
    models: list["BaseItemModel"],
    anchor_matrices: list[list[list[tuple[int, int]]]],
    method: str = "stocking_lord",
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_theta: int = 61,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> list[tuple[float, float]]:
    """Perform concurrent (simultaneous) linking of multiple forms.

    Unlike chain linking, this optimizes all transformations
    simultaneously to minimize total discrepancy.

    Parameters
    ----------
    models : list[BaseItemModel]
        List of models to link.
    anchor_matrices : list[list[list[tuple[int, int]]]]
        For each model i, for each subsequent model j, list of (item_i, item_j) anchor pairs.
    method : str
        Linking criterion.
    theta_range : tuple[float, float]
        Range for curve matching.
    n_theta : int
        Number of theta points.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    list[tuple[float, float]]
        (A, B) transformation for each model to common scale.
    """
    from scipy import optimize, stats

    n_models = len(models)

    theta_grid = np.linspace(theta_range[0], theta_range[1], n_theta)
    weights = stats.norm.pdf(theta_grid)
    weights = weights / np.sum(weights)

    def criterion(params):
        A_list = [1.0] + list(params[: n_models - 1])
        B_list = [0.0] + list(params[n_models - 1 :])

        total_loss = 0.0

        for i in range(n_models):
            for j in range(i + 1, n_models):
                if i >= len(anchor_matrices) or j - i - 1 >= len(anchor_matrices[i]):
                    continue

                pairs = (
                    anchor_matrices[i][j - i - 1]
                    if len(anchor_matrices[i]) > j - i - 1
                    else []
                )
                if not pairs:
                    continue

                disc_i = np.asarray(models[i].discrimination)
                diff_i = np.asarray(models[i].difficulty)
                disc_j = np.asarray(models[j].discrimination)
                diff_j = np.asarray(models[j].difficulty)

                if disc_i.ndim > 1:
                    disc_i = disc_i[:, 0]
                if disc_j.ndim > 1:
                    disc_j = disc_j[:, 0]

                for idx_i, idx_j in pairs:
                    a_i_trans = disc_i[idx_i] / A_list[i]
                    b_i_trans = A_list[i] * diff_i[idx_i] + B_list[i]

                    a_j_trans = disc_j[idx_j] / A_list[j]
                    b_j_trans = A_list[j] * diff_j[idx_j] + B_list[j]

                    p_i = 1.0 / (1.0 + np.exp(-a_i_trans * (theta_grid - b_i_trans)))
                    p_j = 1.0 / (1.0 + np.exp(-a_j_trans * (theta_grid - b_j_trans)))

                    total_loss += np.sum(weights * (p_i - p_j) ** 2)

        return total_loss

    x0 = np.ones(2 * (n_models - 1))
    x0[n_models - 1 :] = 0.0

    result = optimize.minimize(
        criterion,
        x0,
        method="Nelder-Mead",
        options={"maxiter": max_iter * 100, "xatol": tol, "fatol": tol},
    )

    A_list = [1.0] + list(result.x[: n_models - 1])
    B_list = [0.0] + list(result.x[n_models - 1 :])

    return list(zip(A_list, B_list))


def chain_linking_summary(result: ChainLinkingResult) -> str:
    """Generate summary of chain linking results.

    Parameters
    ----------
    result : ChainLinkingResult
        Chain linking result.

    Returns
    -------
    str
        Formatted summary string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Chain Linking Summary")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"Number of time points: {len(result.cumulative_A)}")
    lines.append(f"Reference index: {result.reference_index}")
    lines.append("")

    lines.append("Cumulative Transformations to Reference Scale")
    lines.append("-" * 50)
    lines.append(f"{'Time':>6} {'A':>12} {'B':>12}")
    lines.append("-" * 50)

    for t, (A, B) in enumerate(zip(result.cumulative_A, result.cumulative_B)):
        marker = " *" if t == result.reference_index else ""
        lines.append(f"{t:>6} {A:>12.4f} {B:>12.4f}{marker}")

    lines.append("-" * 50)
    lines.append("* = reference point")
    lines.append("")

    lines.append("Pairwise Linking Results")
    lines.append("-" * 50)

    for t, pr in enumerate(result.pairwise_results):
        lines.append(f"Time {t} -> {t + 1}:")
        lines.append(f"  A = {pr.constants.A:.4f}, B = {pr.constants.B:.4f}")
        lines.append(f"  Method: {pr.constants.method}")

        if pr.fit_statistics is not None:
            lines.append(f"  TCC RMSE: {pr.fit_statistics.tcc_rmse:.4f}")

        if pr.anchor_diagnostics is not None:
            n_flagged = int(np.sum(pr.anchor_diagnostics.flagged))
            lines.append(f"  Flagged items: {n_flagged}")

        lines.append("")

    if result.drift_accumulation is not None:
        lines.append("Drift Accumulation")
        lines.append("-" * 50)

        valid_mask = ~np.isnan(result.drift_accumulation)
        if np.any(valid_mask):
            mean_drift = np.nanmean(np.abs(result.drift_accumulation))
            max_drift = np.nanmax(np.abs(result.drift_accumulation))
            lines.append(f"Mean |z| across all pairs: {mean_drift:.3f}")
            lines.append(f"Max |z| across all pairs: {max_drift:.3f}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def detect_longitudinal_drift(
    chain_result: ChainLinkingResult,
    threshold: float = 2.5,
) -> dict[str, list]:
    """Detect items with consistent drift across time points.

    Parameters
    ----------
    chain_result : ChainLinkingResult
        Chain linking result with drift accumulation.
    threshold : float
        Z-score threshold for flagging.

    Returns
    -------
    dict[str, list]
        Dictionary with consistently drifting items and patterns.
    """
    if chain_result.drift_accumulation is None:
        return {"consistently_flagged": [], "drift_direction": []}

    drift = chain_result.drift_accumulation
    n_pairs, n_items = drift.shape

    consistently_flagged = []
    drift_direction = []

    for j in range(n_items):
        item_drift = drift[:, j]
        valid = ~np.isnan(item_drift)

        if np.sum(valid) < 2:
            continue

        valid_drift = item_drift[valid]
        n_flagged = np.sum(np.abs(valid_drift) > threshold)

        if n_flagged >= np.sum(valid) / 2:
            consistently_flagged.append(j)

            mean_dir = np.mean(valid_drift)
            if mean_dir > 0.5:
                drift_direction.append("increasing")
            elif mean_dir < -0.5:
                drift_direction.append("decreasing")
            else:
                drift_direction.append("variable")

    return {
        "consistently_flagged": consistently_flagged,
        "drift_direction": drift_direction,
    }
