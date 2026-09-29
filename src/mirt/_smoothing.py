"""Shared Gaussian-kernel smoothing for calibration and empirical diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def smooth_response_curves(
    samples: NDArray[np.float64],
    grid: NDArray[np.float64],
    responses: NDArray[np.float64],
    observed: NDArray[np.bool_] | None,
    bandwidth: float,
    *,
    max_elements: int,
    sample_weight: NDArray[np.float64] | None = None,
    calculate_se: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Smooth validated, nonnegative integer item scores using shared grid blocks.

    Missing responses must be zero-filled, with their locations marked in
    ``observed``. Passing ``None`` denotes complete responses. All-missing
    items return NaN curves. Kernel buffers hold at most
    ``max(max_elements, len(samples))`` values; output is item by grid point.
    """
    n_items = responses.shape[1]
    curves = np.empty((n_items, grid.size), dtype=np.float64)
    standard_errors = np.empty_like(curves) if calculate_se else None
    if observed is not None and not np.any(observed):
        curves.fill(np.nan)
        if standard_errors is not None:
            standard_errors.fill(np.nan)
        return curves, standard_errors
    if n_items == 1 and observed is not None:
        contributing = observed[:, 0].copy()
        if sample_weight is not None:
            contributing &= sample_weight > 0.0
        if not np.any(contributing):
            curves.fill(np.nan)
            if standard_errors is not None:
                standard_errors.fill(np.nan)
            return curves, standard_errors
        return smooth_response_curves(
            samples[contributing],
            grid,
            responses[contributing],
            None,
            bandwidth,
            max_elements=max_elements,
            sample_weight=(
                sample_weight[contributing] if sample_weight is not None else None
            ),
            calculate_se=calculate_se,
        )
    valid_values = observed.astype(np.float64) if observed is not None else None
    squared_responses = None
    if calculate_se:
        squared_responses = (
            responses if np.max(responses) <= 1.0 else np.square(responses)
        )

    # Squared weights require earlier recentering than means alone. This bound
    # ensures at least one contributing squared weight remains representable.
    tiny = np.finfo(np.float64).tiny
    minimum_mass = np.sqrt(tiny) * samples.size if calculate_se else tiny
    grid_batch_size = max(1, max_elements // samples.size)
    for start in range(0, grid.size, grid_batch_size):
        stop = start + grid_batch_size
        grid_block = grid[start:stop]
        weights = _stable_gaussian_weights(
            samples, grid_block, bandwidth, sample_weight
        )
        block_curves = responses.T @ weights
        mass = weights.sum(axis=0) if valid_values is None else valid_values.T @ weights
        if calculate_se:
            assert squared_responses is not None
            second_moment = (
                block_curves.copy()
                if squared_responses is responses
                else squared_responses.T @ weights
            )
            np.square(weights, out=weights)
            squared_mass = (
                weights.sum(axis=0)
                if valid_values is None
                else valid_values.T @ weights
            )
        del weights

        np.divide(block_curves, mass, out=block_curves, where=mass > 0.0)
        if observed is not None:
            block_curves[mass <= 0.0] = np.nan
        curves[:, start:stop] = block_curves

        if standard_errors is not None:
            np.divide(second_moment, mass, out=second_moment, where=mass > 0.0)
            second_moment -= np.square(block_curves)
            np.maximum(second_moment, 0.0, out=second_moment)
            # Divide successively to avoid squaring very small masses.
            np.divide(squared_mass, mass, out=squared_mass, where=mass > minimum_mass)
            np.divide(squared_mass, mass, out=squared_mass, where=mass > minimum_mass)
            second_moment *= squared_mass
            np.sqrt(second_moment, out=second_moment)
            standard_errors[:, start:stop] = second_moment

        if observed is None:
            continue
        needs_fallback = mass <= minimum_mass
        for item_idx in np.flatnonzero(np.any(needs_fallback, axis=1)):
            contributing = observed[:, item_idx].copy()
            if sample_weight is not None:
                contributing &= sample_weight > 0.0
            if not np.any(contributing):
                continue
            columns = needs_fallback[item_idx]
            item_curves, item_se = smooth_response_curves(
                samples[contributing],
                grid_block[columns],
                responses[contributing, item_idx, None],
                None,
                bandwidth,
                max_elements=max_elements,
                sample_weight=(
                    sample_weight[contributing] if sample_weight is not None else None
                ),
                calculate_se=calculate_se,
            )
            output_columns = start + np.flatnonzero(columns)
            curves[item_idx, output_columns] = item_curves[0]
            if standard_errors is not None:
                assert item_se is not None
                standard_errors[item_idx, output_columns] = item_se[0]

    return curves, standard_errors


def _stable_gaussian_weights(
    samples: NDArray[np.float64],
    grid: NDArray[np.float64],
    bandwidth: float,
    sample_weight: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Return stable Gaussian and person weights relative to each grid maximum."""
    sample_matrix = samples[:, None]
    grid_matrix = grid[None, :]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        log_weights = sample_matrix - grid_matrix
        log_weights /= bandwidth
        np.square(log_weights, out=log_weights)
        log_weights *= -0.5
    if sample_weight is None:
        column_maximum = np.max(log_weights, axis=0)
    else:
        log_sample_weight = np.full(samples.size, -np.inf, dtype=np.float64)
        np.log(sample_weight, out=log_sample_weight, where=sample_weight > 0.0)
        column_maximum = np.max(
            log_weights,
            axis=0,
            where=np.isfinite(log_sample_weight[:, None]),
            initial=-np.inf,
        )
    ordinary = np.isfinite(column_maximum)
    # Remove the common distance before adding person weights; a large common
    # distance can otherwise round away even substantial weight differences.
    np.subtract(log_weights, column_maximum, out=log_weights, where=ordinary)
    if sample_weight is not None:
        log_weights += log_sample_weight[:, None]
    if not np.all(ordinary):
        if sample_weight is None:
            log_sample_weight = np.zeros(samples.size, dtype=np.float64)
        log_weights[:, ~ordinary] = _extreme_gaussian_log_weights(
            samples, grid[~ordinary], bandwidth, log_sample_weight
        )
    if sample_weight is not None:
        log_weights -= np.max(log_weights, axis=0)
    np.exp(log_weights, out=log_weights)
    return log_weights


def _extreme_gaussian_log_weights(
    samples: NDArray[np.float64],
    grid: NDArray[np.float64],
    bandwidth: float,
    log_sample_weight: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Resolve overflowing squared distances relative to a contributing sample."""
    sample_matrix = samples[:, None]
    grid_matrix = grid[None, :]

    scale = np.maximum(np.abs(sample_matrix), np.abs(grid_matrix))
    nonzero_scale = scale > 0
    scaled_samples = np.divide(
        sample_matrix, scale, out=np.zeros_like(scale), where=nonzero_scale
    )
    scaled_grid = np.divide(
        grid_matrix, scale, out=np.zeros_like(scale), where=nonzero_scale
    )
    normalized_distance = np.abs(scaled_samples - scaled_grid)
    with np.errstate(divide="ignore"):
        log_distance = np.log(scale) + np.log(normalized_distance)

    nearest_log_distance = np.min(
        log_distance,
        axis=0,
        where=np.isfinite(log_sample_weight[:, None]),
        initial=np.inf,
    )
    non_nearest = log_distance > nearest_log_distance[None, :]
    log_squared_gap = np.full_like(log_distance, -np.inf)
    relative_log_square = np.zeros_like(log_distance)
    np.subtract(
        nearest_log_distance[None, :],
        log_distance,
        out=relative_log_square,
        where=non_nearest,
    )
    relative_log_square *= 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_squared_gap[non_nearest] = 2.0 * log_distance[non_nearest] + np.log1p(
            -np.exp(relative_log_square[non_nearest])
        )

    log_penalty = log_squared_gap - np.log(2.0) - 2.0 * np.log(bandwidth)
    # Infinite penalties correctly give zero mass. Capping them would make
    # observations at arbitrarily different distances contribute equally.
    with np.errstate(over="ignore"):
        penalty = np.exp(log_penalty)
    return -penalty + log_sample_weight[:, None]
