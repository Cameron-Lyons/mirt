"""Shared utilities for diagnostic functions."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mirt.results.fit_result import FitResult


def create_paired_resample_chunks(
    *,
    rng: np.random.Generator,
    n_replicates: int,
    n_jobs: int,
    first_size: int,
    second_size: int,
) -> list[tuple[dict[str, Any], int]]:
    """Capture compact RNG chunks for paired stratified resampling.

    Advancing the calling generator before work starts preserves its public
    state and makes seeded samples independent of worker scheduling without
    materializing every replicate's row indices.
    """
    if n_replicates < 0:
        raise ValueError("n_replicates must be nonnegative")
    if n_replicates == 0:
        return []
    chunk_count = min(n_jobs, n_replicates)
    quotient, remainder = divmod(n_replicates, chunk_count)
    chunks: list[tuple[dict[str, Any], int]] = []
    for chunk_index in range(chunk_count):
        chunk_size = quotient + (chunk_index < remainder)
        chunks.append((deepcopy(rng.bit_generator.state), chunk_size))
        for _ in range(chunk_size):
            rng.integers(0, first_size, size=first_size)
            rng.integers(0, second_size, size=second_size)
    return chunks


def split_groups(
    data: NDArray[np.int_],
    groups: NDArray[np.int_] | NDArray[np.str_],
    focal_group: str | int | None = None,
) -> tuple[
    NDArray[np.int_],
    NDArray[np.int_],
    NDArray[np.bool_],
    NDArray[np.bool_],
    Any,
    Any,
]:
    """Split data into reference and focal groups.

    Parameters
    ----------
    data : ndarray
        Response matrix (n_persons, n_items)
    groups : ndarray
        Group membership array
    focal_group : str or int, optional
        Which group to use as focal. If None, uses second unique group.

    Returns
    -------
    ref_data : ndarray
        Reference group responses
    focal_data : ndarray
        Focal group responses
    ref_mask : ndarray
        Boolean mask for reference group
    focal_mask : ndarray
        Boolean mask for focal group
    ref_group : any
        Reference group identifier
    focal_group : any
        Focal group identifier
    """
    data = np.asarray(data)
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    if len(unique_groups) != 2:
        raise ValueError(f"Expected 2 groups, found {len(unique_groups)}")

    ref_group_id = unique_groups[0]
    if focal_group is None:
        focal_group_id = unique_groups[1]
    elif focal_group not in unique_groups:
        raise ValueError(f"focal_group {focal_group} not found in groups")
    else:
        focal_group_id = focal_group
        if focal_group_id == ref_group_id:
            ref_group_id = (
                unique_groups[1]
                if unique_groups[0] == focal_group_id
                else unique_groups[0]
            )

    ref_mask = groups == ref_group_id
    focal_mask = groups == focal_group_id

    ref_data = data[ref_mask]
    focal_data = data[focal_mask]

    return ref_data, focal_data, ref_mask, focal_mask, ref_group_id, focal_group_id


def fit_group_models(
    ref_data: NDArray[np.int_],
    focal_data: NDArray[np.int_],
    model: str = "2PL",
    **fit_kwargs: Any,
) -> tuple[FitResult, FitResult]:
    """Fit IRT models to reference and focal groups.

    Parameters
    ----------
    ref_data : ndarray
        Reference group responses
    focal_data : ndarray
        Focal group responses
    model : str
        IRT model type
    **fit_kwargs
        Additional arguments for fit_mirt

    Returns
    -------
    ref_result : FitResult
        Fitted model for reference group
    focal_result : FitResult
        Fitted model for focal group
    """
    from mirt import fit_mirt

    fit_kwargs.setdefault("compute_standard_errors", False)
    ref_result = fit_mirt(ref_data, model=model, verbose=False, **fit_kwargs)
    focal_result = fit_mirt(focal_data, model=model, verbose=False, **fit_kwargs)

    return ref_result, focal_result


def create_theta_grid(
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_points: int = 100,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create theta grid for evaluation.

    Parameters
    ----------
    theta_range : tuple
        Range of theta values (min, max)
    n_points : int
        Number of grid points

    Returns
    -------
    theta_grid : ndarray
        1D theta values
    theta_2d : ndarray
        2D theta values for model evaluation
    """
    theta_grid = np.linspace(theta_range[0], theta_range[1], n_points)
    theta_2d = theta_grid.reshape(-1, 1)
    return theta_grid, theta_2d


def extract_item_se(
    se_array: NDArray[np.float64],
    item_idx: int,
) -> NDArray[np.float64]:
    """Extract standard error for a specific item.

    Parameters
    ----------
    se_array : ndarray
        Standard error array (may be 1D or 2D)
    item_idx : int
        Item index

    Returns
    -------
    ndarray
        Standard error for the item
    """
    if se_array.ndim > 1:
        return se_array[item_idx]
    elif len(se_array) > 1:
        return np.atleast_1d(se_array[item_idx])
    return se_array
