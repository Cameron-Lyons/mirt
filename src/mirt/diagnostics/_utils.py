"""Shared utilities for diagnostic functions."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from mirt.constants import PROB_EPSILON

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


_POLYTOMOUS_LINK_TYPES = {"GRM": "grm", "GPCM": "gpcm", "PCM": "gpcm", "NRM": "nrm"}
_POLYTOMOUS_MODELS = frozenset(_POLYTOMOUS_LINK_TYPES)


@dataclass(frozen=True)
class BootstrapSummary:
    """Bootstrap standard error, normal p-value and percentile interval."""

    standard_error: float
    p_value: float
    confidence_interval: tuple[float, float]
    n_successful: int
    n_failed: int


class LinkedGroupModels(NamedTuple):
    """Separately calibrated groups with the focal model on the reference scale."""

    reference: FitResult
    focal: FitResult
    focal_on_reference: BaseItemModel
    A: float
    B: float
    anchor_items: list[int]


def summarize_bootstrap(
    replicates: Sequence[float],
    *,
    observed: float,
    n_requested: int,
    confidence_level: float,
) -> BootstrapSummary:
    """Summarize finite bootstrap replicates of a two-group statistic.

    The p-value is the two-sided normal approximation ``|observed| / SE``.
    Fewer than two finite replicates give ``NaN`` summaries.
    """
    estimates = np.asarray(
        [value for value in replicates if np.isfinite(value)], dtype=np.float64
    )
    n_successful = int(estimates.size)
    n_failed = n_requested - n_successful
    if n_successful < 2:
        return BootstrapSummary(
            np.nan, np.nan, (np.nan, np.nan), n_successful, n_failed
        )

    standard_error = float(np.std(estimates, ddof=1))
    if standard_error <= PROB_EPSILON:
        p_value = 1.0 if abs(observed) <= PROB_EPSILON else 0.0
    else:
        p_value = float(2.0 * stats.norm.sf(abs(observed) / standard_error))

    tail_probability = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(estimates, [tail_probability, 1.0 - tail_probability])
    return BootstrapSummary(
        standard_error,
        p_value,
        (float(lower), float(upper)),
        n_successful,
        n_failed,
    )


def validate_two_group_inputs(
    data: object,
    groups: object,
    theta_range: object,
) -> tuple[NDArray[np.int_], NDArray[Any], tuple[float, float]]:
    """Validate a response matrix, group labels and a theta interval."""
    limits = np.asarray(theta_range, dtype=np.float64)
    if limits.shape != (2,) or not np.all(np.isfinite(limits)):
        raise ValueError("theta_range must contain two finite values")
    if limits[0] >= limits[1]:
        raise ValueError("theta_range must be strictly increasing")

    values = np.asarray(data)
    labels = np.asarray(groups)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("data must be a nonempty two-dimensional response matrix")
    if labels.ndim != 1:
        raise ValueError("groups must be one-dimensional")
    if labels.shape[0] != values.shape[0]:
        raise ValueError("groups length must match the number of response-matrix rows")
    if labels.dtype.kind in "fc" and not np.all(np.isfinite(labels)):
        raise ValueError("groups must not contain missing or non-finite labels")
    if labels.dtype.kind == "O" and any(
        label is None
        or (isinstance(label, (float, np.floating)) and not np.isfinite(label))
        for label in labels
    ):
        raise ValueError("groups must not contain missing labels")
    return values, labels, (float(limits[0]), float(limits[1]))


def resolve_anchor_items(
    anchor_items: Sequence[int] | NDArray[np.integer] | None,
    n_items: int,
    *,
    name: str = "anchor_items",
    minimum: int = 2,
) -> list[int] | None:
    """Validate unique in-range anchor indices; ``None`` passes through."""
    if anchor_items is None:
        return None
    if isinstance(anchor_items, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of item indices")
    values = list(np.atleast_1d(np.asarray(anchor_items, dtype=object)))
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in values
    ):
        raise ValueError(f"{name} must contain integer item indices")
    resolved = [int(value) for value in values]
    if any(value < 0 or value >= n_items for value in resolved):
        raise ValueError(f"{name} must contain indices in [0, {n_items})")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must not contain duplicate items")
    if len(resolved) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} items")
    return sorted(resolved)


def link_focal_to_reference(
    reference_model: BaseItemModel,
    focal_model: BaseItemModel,
    anchor_items: Sequence[int],
    *,
    method: str = "stocking_lord",
) -> tuple[BaseItemModel, float, float]:
    """Place a separately calibrated focal model on the reference scale.

    Returns a transformed copy of ``focal_model`` and the constants of
    ``theta_reference = A * theta_focal + B``, estimated over the anchors
    with :func:`mirt.equating.link` or the matching polytomous linker. Unit
    1PL slopes cannot absorb a change of scale, so 1PL models use mean/mean
    linking, which fixes ``A = 1`` and shifts difficulties only.
    """
    from mirt.equating.linking import link, transform_parameters
    from mirt.equating.polytomous import (
        link_gpcm,
        link_grm,
        link_nrm,
        transform_polytomous_parameters,
    )

    anchors = [int(item) for item in anchor_items]
    model_name = getattr(focal_model, "model_name", "")
    polytomous_type = _POLYTOMOUS_LINK_TYPES.get(model_name)
    if polytomous_type is not None:
        linker = {"grm": link_grm, "gpcm": link_gpcm, "nrm": link_nrm}[polytomous_type]
        constants = linker(
            reference_model,
            focal_model,
            anchors,
            anchors,
            method=method,
            compute_diagnostics=False,
        ).constants
        linked = transform_polytomous_parameters(
            focal_model, constants.A, constants.B, polytomous_type
        )
        return linked, float(constants.A), float(constants.B)

    if model_name == "1PL":
        shift = link(
            reference_model,
            focal_model,
            anchors,
            anchors,
            method="mean_mean",
            compute_diagnostics=False,
        ).constants.B
        linked = focal_model.copy()
        linked.set_parameters(
            difficulty=np.asarray(focal_model.difficulty, dtype=np.float64) + shift
        )
        return linked, 1.0, float(shift)

    constants = link(
        reference_model,
        focal_model,
        anchors,
        anchors,
        method=method,
        compute_diagnostics=False,
    ).constants
    linked = transform_parameters(focal_model, constants.A, constants.B)
    return linked, float(constants.A), float(constants.B)


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

    Notes
    -----
    Polytomous category counts default to those of the pooled responses, so
    both groups share one item structure even when a group never uses an
    item's top category.
    """
    from mirt import fit_mirt
    from mirt.models._factory import resolve_category_counts
    from mirt.utils.data import validate_responses

    fit_kwargs.setdefault("compute_standard_errors", False)
    if model in _POLYTOMOUS_MODELS and fit_kwargs.get("n_categories") is None:
        pooled = validate_responses(
            np.vstack([np.asarray(ref_data), np.asarray(focal_data)])
        )
        fit_kwargs["n_categories"] = resolve_category_counts(
            pooled.shape[1], None, pooled
        )
    ref_result = fit_mirt(ref_data, model=model, verbose=False, **fit_kwargs)
    focal_result = fit_mirt(focal_data, model=model, verbose=False, **fit_kwargs)

    return ref_result, focal_result


def fit_linked_group_models(
    ref_data: NDArray[np.int_],
    focal_data: NDArray[np.int_],
    model: str = "2PL",
    *,
    anchor_items: Sequence[int] | None = None,
    link_method: str = "stocking_lord",
    **fit_kwargs: Any,
) -> LinkedGroupModels:
    """Calibrate both groups separately and link the focal group.

    Parameters
    ----------
    ref_data, focal_data : ndarray
        Reference and focal group responses.
    model : str
        IRT model type.
    anchor_items : sequence of int, optional
        Items assumed free of DIF that define the link. ``None`` uses every
        item, which assumes no DIF or DIF that balances across items.
    link_method : str
        Linking criterion (Stocking-Lord by default).
    **fit_kwargs
        Additional arguments for ``fit_mirt``.

    Returns
    -------
    LinkedGroupModels
        Both fits, the focal model on the reference scale, and ``A`` and
        ``B`` such that ``theta_reference = A * theta_focal + B``.
    """
    ref_result, focal_result = fit_group_models(
        ref_data, focal_data, model=model, **fit_kwargs
    )
    n_items = int(ref_result.model.n_items)
    anchors = resolve_anchor_items(anchor_items, n_items) or list(range(n_items))
    linked, A, B = link_focal_to_reference(
        ref_result.model, focal_result.model, anchors, method=link_method
    )
    return LinkedGroupModels(ref_result, focal_result, linked, A, B, anchors)


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
