"""Empirical analysis functions for IRT models.

Provides functions for computing DIF effect sizes and generating
data for observed vs expected plots.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt._smoothing import smooth_response_curves

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

SILVERMAN_CONSTANT = 1.06
SILVERMAN_EXPONENT = -1 / 5
KERNEL_BLOCK_ELEMENTS = 2_000_000
_EMPIRICAL_MAX_PROBABILITY_VALUES = 262_144


@dataclass
class DIFEffectSize:
    """Container for DIF effect size statistics.

    Attributes
    ----------
    item_idx : int
        Item index.
    signed_es : float
        Signed effect size (positive = favors focal group).
    unsigned_es : float
        Unsigned (absolute) effect size.
    sids : float
        Signed Item Difference in the Sample.
    uids : float
        Unsigned Item Difference in the Sample.
    classification : str
        ETS classification ("A", "B", or "C").
    """

    item_idx: int
    signed_es: float
    unsigned_es: float
    sids: float
    uids: float
    classification: str


@dataclass
class EmpiricalPlotData:
    """Container for empirical plot data.

    Attributes
    ----------
    item_idx : int
        Item index.
    theta_bins : NDArray[np.float64]
        Theta bin midpoints.
    observed_prop : NDArray[np.float64]
        Observed mean item scores in each bin.
    expected_prop : NDArray[np.float64]
        Model-predicted mean item scores in each bin.
    n_per_bin : NDArray[np.intp]
        Number of observations in each bin.
    residuals : NDArray[np.float64]
        Observed - expected differences.
    """

    item_idx: int
    theta_bins: NDArray[np.float64]
    observed_prop: NDArray[np.float64]
    expected_prop: NDArray[np.float64]
    n_per_bin: NDArray[np.intp]
    residuals: NDArray[np.float64]


def _validate_positive_integer(value: int, name: str, minimum: int = 1) -> int:
    """Return a validated integer control parameter."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validate_empirical_inputs(
    model: BaseItemModel,
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate the shared unidimensional empirical-diagnostic inputs."""
    if model.n_factors != 1:
        raise ValueError("empirical diagnostics require a unidimensional model")

    response_values = np.asarray(responses, dtype=np.float64)
    if response_values.ndim != 2:
        raise ValueError("responses must be a 2D matrix")
    if response_values.shape[0] == 0:
        raise ValueError("responses must contain at least one person")
    if response_values.shape[1] != model.n_items:
        raise ValueError(
            f"responses must contain {model.n_items} items, "
            f"got {response_values.shape[1]}"
        )
    if np.any(np.isinf(response_values)):
        raise ValueError("responses must not contain infinite values")

    theta_values = np.asarray(theta, dtype=np.float64)
    if theta_values.ndim == 1:
        theta_values = theta_values.reshape(-1, 1)
    if theta_values.ndim != 2 or theta_values.shape[1] != 1:
        raise ValueError("theta must have shape (n_persons,) or (n_persons, 1)")
    if theta_values.shape[0] != response_values.shape[0]:
        raise ValueError("theta and responses must contain the same number of persons")
    if not np.all(np.isfinite(theta_values)):
        raise ValueError("theta must contain only finite values")

    return response_values, theta_values


def _validate_item_index(model: BaseItemModel, item_idx: int) -> int:
    """Return a validated zero-based item index."""
    if isinstance(item_idx, (bool, np.bool_)) or not isinstance(
        item_idx, (int, np.integer)
    ):
        raise ValueError("item_idx must be an integer")
    result = int(item_idx)
    if result < 0 or result >= model.n_items:
        raise ValueError(f"item_idx must be between 0 and {model.n_items - 1}")
    return result


def _validate_probabilities(probabilities: NDArray[np.float64]) -> None:
    """Reject malformed model probability output early."""
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("model probabilities must contain only finite values")
    tolerance = 1e-8
    if np.any(probabilities < -tolerance) or np.any(probabilities > 1 + tolerance):
        raise ValueError("model probabilities must be between 0 and 1")


def _expected_item_score(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    item_idx: int,
) -> tuple[NDArray[np.float64], int]:
    """Return the item expected score and its maximum category score."""
    probabilities = np.asarray(
        model.probability(theta, item_idx=item_idx), dtype=np.float64
    )
    n_persons = theta.shape[0]

    if probabilities.ndim == 1 and probabilities.shape[0] == n_persons:
        _validate_probabilities(probabilities)
        return probabilities, 1

    if probabilities.ndim != 2 or probabilities.shape[0] != n_persons:
        raise ValueError(
            "item probability output must have shape (n_persons,) or "
            "(n_persons, n_categories)"
        )
    if probabilities.shape[1] == 0:
        raise ValueError("item probability output must contain a category")

    _validate_probabilities(probabilities)
    if probabilities.shape[1] == 1:
        return probabilities[:, 0], 1
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("polytomous category probabilities must sum to 1")

    categories = np.arange(probabilities.shape[1], dtype=np.float64)
    return probabilities @ categories, probabilities.shape[1] - 1


def _model_category_counts(model: BaseItemModel) -> NDArray[np.intp] | None:
    """Resolve shared or item-specific category counts without truncation."""
    category_counts = getattr(model, "n_categories", None)
    if category_counts is None:
        return None
    counts = np.asarray(category_counts)
    if counts.ndim == 0:
        counts = np.broadcast_to(counts, (model.n_items,))
    if (
        counts.shape != (model.n_items,)
        or counts.dtype.kind not in "iu"
        or np.any(counts < 2)
        or np.any(counts > np.iinfo(np.intp).max)
    ):
        raise ValueError("model category counts are malformed")
    return counts.astype(np.intp, copy=False)


def _expected_all_item_scores(
    model: BaseItemModel,
    theta: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Evaluate expected scores for every item with one public model call."""
    probabilities = np.asarray(model.probability(theta), dtype=np.float64)
    n_persons = theta.shape[0]

    if probabilities.ndim == 1 and model.n_items == 1:
        probabilities = probabilities.reshape(-1, 1)

    if probabilities.ndim == 2 and probabilities.shape == (
        n_persons,
        model.n_items,
    ):
        _validate_probabilities(probabilities)
        return probabilities, np.ones(model.n_items, dtype=np.intp)

    if (
        probabilities.ndim != 3
        or probabilities.shape[0] != n_persons
        or probabilities.shape[1] != model.n_items
        or probabilities.shape[2] < 2
    ):
        raise ValueError(
            "model probability output must have shape (n_persons, n_items) "
            "or (n_persons, n_items, n_categories)"
        )

    _validate_probabilities(probabilities)
    if not np.allclose(probabilities.sum(axis=2), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("polytomous category probabilities must sum to 1")

    categories = np.arange(probabilities.shape[2], dtype=np.float64)
    expected_scores = probabilities @ categories

    counts = _model_category_counts(model)
    if counts is not None:
        if np.any(counts > probabilities.shape[2]):
            raise ValueError("model category counts are malformed")
        max_scores = counts - 1
    else:
        max_scores = np.full(model.n_items, probabilities.shape[2] - 1, dtype=np.intp)
    return expected_scores, max_scores


def _validate_observed_scores(
    values: NDArray[np.float64],
    max_score: int,
    item_idx: int,
) -> None:
    """Validate observed, non-missing category scores for one item."""
    observed = values[np.isfinite(values) & (values >= 0)]
    if observed.size == 0:
        return
    if np.any(observed != np.floor(observed)):
        raise ValueError(
            f"responses for item {item_idx} must be integer category scores"
        )
    if np.any(observed > max_score):
        raise ValueError(
            f"responses for item {item_idx} must be between 0 and {max_score}"
        )


def _theta_bin_indices(
    theta: NDArray[np.float64],
    n_bins: int,
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Assign shared quantile bins without overflowing finite edge values."""
    percentiles = np.linspace(0.0, 100.0, n_bins + 1)
    largest = max(abs(float(theta.min())), abs(float(theta.max())))
    if largest > np.finfo(np.float64).max / 2:
        # Power-of-two scaling keeps interpolation differences finite without
        # changing the precision of the represented input values.
        bin_edges = np.percentile(theta * 0.5, percentiles) * 2.0
    else:
        bin_edges = np.percentile(theta, percentiles)
    # Interior edges suffice: the first and last bins include the endpoints.
    # This also avoids nextafter(max_float, inf) and retains right-sided ties.
    bin_indices = np.searchsorted(bin_edges[1:-1], theta, side="right")
    return bin_indices, bin_edges


def _build_theta_bins(
    theta: NDArray[np.float64],
    n_bins: int,
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Return shared theta bins with finite midpoints and observed bin means."""
    bin_indices, bin_edges = _theta_bin_indices(theta, n_bins)
    n_per_bin = np.bincount(bin_indices, minlength=n_bins).astype(np.intp)
    nonempty = n_per_bin > 0

    scale = max(1.0, abs(float(bin_edges[0])), abs(float(bin_edges[-1])))
    scaled_edges = bin_edges / scale
    theta_bins = (scaled_edges[:-1] + scaled_edges[1:]) * 0.5
    theta_sums = np.bincount(bin_indices, weights=theta / scale, minlength=n_bins)
    theta_bins[nonempty] = theta_sums[nonempty] / n_per_bin[nonempty]
    np.clip(theta_bins, -1.0, 1.0, out=theta_bins)
    theta_bins *= scale

    return bin_indices, theta_bins


def _aggregate_item_bins(
    bin_indices: NDArray[np.intp],
    observed: NDArray[np.float64],
    expected: NDArray[np.float64],
    n_bins: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """Aggregate observed and expected item scores into assigned bins."""
    n_per_bin = np.bincount(bin_indices, minlength=n_bins).astype(np.intp)
    nonempty = n_per_bin > 0
    observed_scores = np.zeros(n_bins, dtype=np.float64)
    expected_scores = np.zeros(n_bins, dtype=np.float64)

    observed_sums = np.bincount(bin_indices, weights=observed, minlength=n_bins)
    expected_sums = np.bincount(bin_indices, weights=expected, minlength=n_bins)
    observed_scores[nonempty] = observed_sums[nonempty] / n_per_bin[nonempty]
    expected_scores[nonempty] = expected_sums[nonempty] / n_per_bin[nonempty]

    return observed_scores, expected_scores, n_per_bin


def _validate_integration_grid(
    model_ref: BaseItemModel,
    model_focal: BaseItemModel,
    item_idx: int,
    theta_range: tuple[float, float],
    n_points: int,
) -> tuple[int, NDArray[np.float64]]:
    """Validate DIF model inputs and construct a unidimensional grid."""
    if model_ref.n_factors != 1 or model_focal.n_factors != 1:
        raise ValueError("empirical DIF diagnostics require unidimensional models")
    ref_idx = _validate_item_index(model_ref, item_idx)
    _validate_item_index(model_focal, item_idx)
    point_count = _validate_positive_integer(n_points, "n_points", minimum=2)

    limits = np.asarray(theta_range, dtype=np.float64)
    if limits.shape != (2,) or not np.all(np.isfinite(limits)):
        raise ValueError("theta_range must contain two finite values")
    if limits[0] >= limits[1]:
        raise ValueError("theta_range lower bound must be less than its upper bound")

    theta = np.linspace(limits[0], limits[1], point_count).reshape(-1, 1)
    return ref_idx, theta


def empirical_ES(
    model_ref: BaseItemModel,
    model_focal: BaseItemModel,
    item_idx: int,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_points: int = 101,
    focal_weight: float = 0.5,
) -> DIFEffectSize:
    """Compute empirical effect size for DIF.

    Computes effect sizes comparing item response functions between
    reference and focal groups.

    Parameters
    ----------
    model_ref : BaseItemModel
        Model fitted on reference group.
    model_focal : BaseItemModel
        Model fitted on focal group.
    item_idx : int
        Index of item to evaluate.
    theta_range : tuple
        Range for integration. Default (-4, 4).
    n_points : int
        Number of integration points. Default 101.
    focal_weight : float
        Relative focal-group weight, retained for API compatibility. Because
        this function integrates over one shared standard-normal density, the
        result is invariant to this mixture weight. Must be between 0 and 1.

    Returns
    -------
    DIFEffectSize
        Container with effect size statistics.

    Examples
    --------
    >>> model_ref = fit_mirt(responses_ref, model="2PL").model
    >>> model_focal = fit_mirt(responses_focal, model="2PL").model
    >>> es = empirical_ES(model_ref, model_focal, item_idx=0)
    >>> print(f"Signed ES: {es.signed_es:.3f}")
    >>> print(f"ETS Classification: {es.classification}")
    """
    item_idx, theta_2d = _validate_integration_grid(
        model_ref, model_focal, item_idx, theta_range, n_points
    )
    if not np.isscalar(focal_weight) or not np.isfinite(focal_weight):
        raise ValueError("focal_weight must be a finite value between 0 and 1")
    if focal_weight < 0 or focal_weight > 1:
        raise ValueError("focal_weight must be between 0 and 1")

    from scipy import stats

    theta = theta_2d[:, 0]
    weights = stats.norm.pdf(theta)
    weights = weights / np.sum(weights)

    score_ref, ref_max_score = _expected_item_score(model_ref, theta_2d, item_idx)
    score_focal, focal_max_score = _expected_item_score(model_focal, theta_2d, item_idx)
    if ref_max_score != focal_max_score:
        raise ValueError("reference and focal items must use the same score range")

    diff = score_focal - score_ref

    sids = np.sum(weights * diff)
    uids = np.sum(weights * np.abs(diff))

    signed_es = sids
    unsigned_es = uids

    if unsigned_es < 0.05:
        classification = "A"
    elif unsigned_es < 0.10:
        classification = "B"
    else:
        classification = "C"

    return DIFEffectSize(
        item_idx=item_idx,
        signed_es=float(signed_es),
        unsigned_es=float(unsigned_es),
        sids=float(sids),
        uids=float(uids),
        classification=classification,
    )


def empirical_plot(
    model: BaseItemModel,
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
    item_idx: int,
    n_bins: int = 10,
) -> EmpiricalPlotData:
    """Compute data for observed vs expected empirical plot.

    Groups examinees by theta estimate and computes observed vs
    model-predicted mean item scores for model-data fit assessment. For
    dichotomous items these scores are proportions correct; for polytomous
    items they are expected category scores. Missing item responses are
    excluded without changing the theta-bin boundaries.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    responses : NDArray[np.float64]
        Response matrix. Shape: (n_persons, n_items).
    theta : NDArray[np.float64]
        Ability estimates. Shape: (n_persons,) or (n_persons, 1).
    item_idx : int
        Index of item to plot.
    n_bins : int
        Number of theta bins. Default 10.

    Returns
    -------
    EmpiricalPlotData
        Container with plot data.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL")
    >>> plot_data = empirical_plot(result.model, responses, result.theta, item_idx=0)
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(plot_data.theta_bins, plot_data.observed_prop)
    >>> plt.plot(plot_data.theta_bins, plot_data.expected_prop)
    """
    responses, theta_2d = _validate_empirical_inputs(model, responses, theta)
    item_idx = _validate_item_index(model, item_idx)
    n_bins = _validate_positive_integer(n_bins, "n_bins")

    item_responses = responses[:, item_idx]
    valid_mask = np.isfinite(item_responses) & (item_responses >= 0)
    if not np.any(valid_mask):
        return EmpiricalPlotData(
            item_idx=item_idx,
            theta_bins=np.array([]),
            observed_prop=np.array([]),
            expected_prop=np.array([]),
            n_per_bin=np.array([], dtype=np.intp),
            residuals=np.array([]),
        )

    bin_indices, theta_bins = _build_theta_bins(theta_2d[:, 0], n_bins)
    item_responses = item_responses[valid_mask]
    expected_scores, max_score = _expected_item_score(
        model, theta_2d[valid_mask], item_idx
    )
    _validate_observed_scores(item_responses, max_score, item_idx)
    observed_prop, expected_prop, n_per_bin = _aggregate_item_bins(
        bin_indices[valid_mask], item_responses, expected_scores, n_bins
    )

    residuals = observed_prop - expected_prop

    return EmpiricalPlotData(
        item_idx=item_idx,
        theta_bins=theta_bins,
        observed_prop=observed_prop,
        expected_prop=expected_prop,
        n_per_bin=n_per_bin,
        residuals=residuals,
    )


def _expected_score_blocks(
    model: BaseItemModel,
    theta: NDArray[np.float64],
) -> Iterator[tuple[slice, NDArray[np.float64], NDArray[np.intp]]]:
    """Budget probability output by known or probed category width."""
    counts = _model_category_counts(model)
    width = 1 if counts is None else int(np.max(counts))
    start = 0
    if counts is None and getattr(model, "is_polytomous", None) is not False:
        expected, max_scores = _expected_all_item_scores(model, theta[:1])
        yield slice(0, 1), expected, max_scores
        width = int(np.max(max_scores)) + 1
        start = 1
    rows_per_block = max(
        1, _EMPIRICAL_MAX_PROBABILITY_VALUES // (model.n_items * width)
    )
    for row in range(start, theta.shape[0], rows_per_block):
        rows = slice(row, row + rows_per_block)
        expected, max_scores = _expected_all_item_scores(model, theta[rows])
        yield rows, expected, max_scores


def _accumulate_bin_residuals(
    counts: NDArray[np.intp],
    residual_sums: NDArray[np.float64],
    bin_indices: NDArray[np.intp],
    responses: NDArray[np.float64],
    expected: NDArray[np.float64],
) -> None:
    """Reduce one response block without a dense person-by-bin matrix."""
    n_bins, n_items = counts.shape
    # Compact sparse bin labels when a full histogram would exceed the block
    # budget. Each update then allocates only the bins present in this block.
    if counts.size > _EMPIRICAL_MAX_PROBABILITY_VALUES:
        occupied_bins, local_bins = np.unique(bin_indices, return_inverse=True)
        output_size = occupied_bins.size * n_items
    else:
        occupied_bins = slice(None)
        local_bins = bin_indices
        output_size = n_bins * n_items
    valid = np.isfinite(responses) & (responses >= 0.0)
    codes = local_bins[:, None] * n_items + np.arange(n_items, dtype=np.intp)
    residuals = responses - expected
    if np.all(valid):
        counts[occupied_bins] += np.bincount(
            local_bins, minlength=output_size // n_items
        )[:, None]
        codes = codes.ravel()
        weights = residuals.ravel()
    else:
        codes = codes[valid]
        weights = residuals[valid]
        counts[occupied_bins] += np.bincount(codes, minlength=output_size).reshape(
            -1, n_items
        )
    residual_sums[occupied_bins] += np.bincount(
        codes,
        weights=weights,
        minlength=output_size,
    ).reshape(-1, n_items)


def empirical_rmsea(
    model: BaseItemModel,
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
    n_bins: int = 10,
) -> NDArray[np.float64]:
    """Compute RMSEA-like fit statistic per item.

    Measures root mean square error of approximation between observed and
    expected item scores across ability bins. Model expectations for all
    items are evaluated together in bounded response blocks. Every item uses
    the same theta-bin boundaries so item-level missingness cannot shift the
    conditioning groups. Empty bins do not contribute to the mean square.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    responses : NDArray[np.float64]
        Response matrix.
    theta : NDArray[np.float64]
        Ability estimates.
    n_bins : int
        Number of theta bins. Default 10.

    Returns
    -------
    NDArray[np.float64]
        RMSEA values for each item.
    """
    responses, theta_2d = _validate_empirical_inputs(model, responses, theta)
    n_bins = _validate_positive_integer(n_bins, "n_bins")
    bin_indices, _ = _theta_bin_indices(theta_2d[:, 0], n_bins)
    if n_bins > responses.shape[0]:
        occupied, bin_indices = np.unique(bin_indices, return_inverse=True)
        n_bins = occupied.size
    counts = np.zeros((n_bins, model.n_items), dtype=np.intp)
    residual_sums = np.zeros((n_bins, model.n_items), dtype=np.float64)
    for rows, expected_scores, max_scores in _expected_score_blocks(model, theta_2d):
        response_block = responses[rows]
        for item_idx, max_score in enumerate(max_scores):
            _validate_observed_scores(
                response_block[:, item_idx], int(max_score), item_idx
            )
        _accumulate_bin_residuals(
            counts,
            residual_sums,
            bin_indices[rows],
            response_block,
            expected_scores,
        )

    estimable = counts > 0
    np.divide(residual_sums, counts, out=residual_sums, where=estimable)
    np.square(residual_sums, out=residual_sums)
    estimable_bins = estimable.sum(axis=0)
    mean_squared = np.divide(
        residual_sums.sum(axis=0),
        estimable_bins,
        out=np.full(model.n_items, np.nan, dtype=np.float64),
        where=estimable_bins > 0,
    )
    return np.sqrt(mean_squared)


def _validate_mantel_haenszel_inputs(
    responses: NDArray[np.float64],
    group: NDArray[np.intp],
    theta: NDArray[np.float64],
    item_idx: int,
    n_strata: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.intp],
    NDArray[np.float64],
    int,
]:
    """Validate and filter inputs for the dichotomous MH statistic."""
    response_values = np.asarray(responses, dtype=np.float64)
    if response_values.ndim != 2:
        raise ValueError("responses must be a 2D matrix")
    if response_values.shape[0] == 0:
        raise ValueError("responses must contain at least one person")
    if response_values.shape[1] == 0:
        raise ValueError("responses must contain at least one item")

    if isinstance(item_idx, (bool, np.bool_)) or not isinstance(
        item_idx, (int, np.integer)
    ):
        raise ValueError("item_idx must be an integer")
    item_idx = int(item_idx)
    if item_idx < 0 or item_idx >= response_values.shape[1]:
        raise ValueError(
            f"item_idx must be between 0 and {response_values.shape[1] - 1}"
        )

    group_values = np.asarray(group, dtype=np.float64)
    if group_values.ndim == 2 and group_values.shape[1] == 1:
        group_values = group_values[:, 0]
    if group_values.ndim != 1:
        raise ValueError("group must have shape (n_persons,) or (n_persons, 1)")

    theta_values = np.asarray(theta, dtype=np.float64)
    if theta_values.ndim == 2 and theta_values.shape[1] == 1:
        theta_values = theta_values[:, 0]
    if theta_values.ndim != 1:
        raise ValueError("theta must have shape (n_persons,) or (n_persons, 1)")

    n_persons = response_values.shape[0]
    if group_values.shape[0] != n_persons:
        raise ValueError("group and responses must contain the same number of persons")
    if theta_values.shape[0] != n_persons:
        raise ValueError("theta and responses must contain the same number of persons")
    if not np.all(np.isfinite(group_values)) or np.any(
        (group_values != 0) & (group_values != 1)
    ):
        raise ValueError("group must contain only 0 (reference) and 1 (focal)")
    if not np.all(np.isfinite(theta_values)):
        raise ValueError("theta must contain only finite values")

    item_responses = response_values[:, item_idx]
    if np.any(np.isinf(item_responses)):
        raise ValueError("item responses must not contain infinite values")
    observed = np.isfinite(item_responses) & (item_responses >= 0)
    if not np.any(observed):
        raise ValueError("item must contain at least one observed response")
    item_responses = item_responses[observed]
    if np.any((item_responses != 0) & (item_responses != 1)):
        raise ValueError("Mantel-Haenszel DIF requires binary item responses")

    group_indices = group_values[observed].astype(np.intp)
    if group_indices.min() == group_indices.max():
        raise ValueError("both reference and focal groups must have observed responses")

    return (
        item_responses,
        group_indices,
        theta_values[observed],
        _validate_positive_integer(n_strata, "n_strata"),
    )


def mantel_haenszel(
    responses: NDArray[np.float64],
    group: NDArray[np.intp],
    theta: NDArray[np.float64],
    item_idx: int,
    n_strata: int = 5,
    correct: bool = True,
) -> tuple[float, float, float]:
    """Compute Mantel-Haenszel DIF statistic.

    Parameters
    ----------
    responses : NDArray[np.float64]
        Response matrix.
    group : NDArray[np.intp]
        Group membership (0 = reference, 1 = focal).
    theta : NDArray[np.float64]
        Matching variable (e.g., total score or theta estimate).
    item_idx : int
        Index of item to test.
    n_strata : int
        Number of matching strata. Default 5.
    correct : bool
        Apply the 0.5 continuity correction when possible. Default True.

    Returns
    -------
    mh_chisq : float
        Mantel-Haenszel chi-square statistic.
    p_value : float
        P-value.
    mh_odds : float
        Common odds ratio for the reference group relative to the focal group.
        Returns infinity or NaN when the ratio is infinite or unestimable.

    Notes
    -----
    Responses must be dichotomous. Negative and NaN item responses are treated
    as missing. Matching strata are quantile bins of the observed responses'
    theta values.
    """
    from scipy import stats

    item_resp, group_valid, theta_valid, n_strata = _validate_mantel_haenszel_inputs(
        responses, group, theta, item_idx, n_strata
    )
    if not isinstance(correct, (bool, np.bool_)):
        raise ValueError("correct must be a boolean")

    percentiles = np.linspace(0.0, 100.0, n_strata + 1)
    bins = np.percentile(theta_valid, percentiles)
    bins[-1] = np.nextafter(bins[-1], np.inf)
    stratum = np.clip(np.digitize(theta_valid, bins) - 1, 0, n_strata - 1).astype(
        np.intp
    )

    cells = stratum * 2 + group_valid
    counts = np.bincount(cells, minlength=2 * n_strata).reshape(n_strata, 2)
    correct_counts = np.bincount(
        cells, weights=item_resp, minlength=2 * n_strata
    ).reshape(n_strata, 2)

    eligible = (counts[:, 0] > 0) & (counts[:, 1] > 0)
    counts = counts[eligible].astype(np.float64)
    correct_counts = correct_counts[eligible]
    n_ref = counts[:, 0]
    n_focal = counts[:, 1]
    a = correct_counts[:, 0]
    c = correct_counts[:, 1]
    b = n_ref - a
    d = n_focal - c
    n_total = n_ref + n_focal

    total_correct = a + c
    total_incorrect = b + d
    delta = np.sum(a - n_ref * total_correct / n_total)
    variance = np.sum(
        n_ref * n_focal * total_correct * total_incorrect / (n_total**2 * (n_total - 1))
    )

    if variance > 0:
        continuity = 0.5 if correct and abs(delta) >= 0.5 else 0.0
        mh_chisq = (abs(delta) - continuity) ** 2 / variance
        p_value = stats.chi2.sf(mh_chisq, 1)
    else:
        mh_chisq = 0.0
        p_value = 1.0

    odds_numerator = np.sum(a * d / n_total)
    odds_denominator = np.sum(b * c / n_total)
    if odds_denominator > 0:
        mh_odds = odds_numerator / odds_denominator
    elif odds_numerator > 0:
        mh_odds = np.inf
    else:
        mh_odds = np.nan

    return float(mh_chisq), float(p_value), float(mh_odds)


def RMSD_DIF(
    model_ref: BaseItemModel,
    model_focal: BaseItemModel,
    item_idx: int,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_points: int = 101,
) -> float:
    """Compute RMSD-based DIF statistic.

    The Root Mean Square Difference compares item response functions
    between reference and focal groups.

    Parameters
    ----------
    model_ref : BaseItemModel
        Model fitted on reference group.
    model_focal : BaseItemModel
        Model fitted on focal group.
    item_idx : int
        Index of item to evaluate.
    theta_range : tuple
        Range for integration. Default (-4, 4).
    n_points : int
        Number of integration points. Default 101.

    Returns
    -------
    float
        RMSD value. Larger values indicate more DIF.

    Examples
    --------
    >>> model_ref = fit_mirt(responses_ref, model="2PL").model
    >>> model_focal = fit_mirt(responses_focal, model="2PL").model
    >>> rmsd = RMSD_DIF(model_ref, model_focal, item_idx=0)
    >>> print(f"RMSD DIF: {rmsd:.4f}")

    Notes
    -----
    Guidelines for interpretation (Meade, 2010):
    - RMSD < 0.05: Negligible DIF
    - 0.05 <= RMSD < 0.10: Slight DIF
    - RMSD >= 0.10: Notable DIF
    """
    item_idx, theta = _validate_integration_grid(
        model_ref, model_focal, item_idx, theta_range, n_points
    )
    score_ref, ref_max_score = _expected_item_score(model_ref, theta, item_idx)
    score_focal, focal_max_score = _expected_item_score(model_focal, theta, item_idx)
    if ref_max_score != focal_max_score:
        raise ValueError("reference and focal items must use the same score range")

    squared_diff = (score_ref - score_focal) ** 2
    rmsd = np.sqrt(np.mean(squared_diff))

    return float(rmsd)


def weighted_RMSD_DIF(
    model_ref: BaseItemModel,
    model_focal: BaseItemModel,
    item_idx: int,
    theta_range: tuple[float, float] = (-4.0, 4.0),
    n_points: int = 101,
) -> float:
    """Compute weighted RMSD-based DIF statistic.

    Weights the squared differences by the standard normal density,
    giving more weight to typical ability levels.

    Parameters
    ----------
    model_ref : BaseItemModel
        Model fitted on reference group.
    model_focal : BaseItemModel
        Model fitted on focal group.
    item_idx : int
        Index of item to evaluate.
    theta_range : tuple
        Range for integration. Default (-4, 4).
    n_points : int
        Number of integration points. Default 101.

    Returns
    -------
    float
        Weighted RMSD value.
    """
    from scipy import stats

    item_idx, theta_2d = _validate_integration_grid(
        model_ref, model_focal, item_idx, theta_range, n_points
    )

    theta = theta_2d[:, 0]
    weights = stats.norm.pdf(theta)
    weights = weights / np.sum(weights)

    score_ref, ref_max_score = _expected_item_score(model_ref, theta_2d, item_idx)
    score_focal, focal_max_score = _expected_item_score(model_focal, theta_2d, item_idx)
    if ref_max_score != focal_max_score:
        raise ValueError("reference and focal items must use the same score range")

    squared_diff = (score_ref - score_focal) ** 2
    weighted_rmsd = np.sqrt(np.sum(weights * squared_diff))

    return float(weighted_rmsd)


@dataclass
class ItemGAMResult:
    """Container for itemGAM results.

    Attributes
    ----------
    item_idx : int
        Item index.
    theta_grid : NDArray[np.float64]
        Grid of theta values for smooth curve.
    smoothed_probs : NDArray[np.float64]
        Smoothed empirical mean scores.
    model_probs : NDArray[np.float64]
        Model-predicted expected scores.
    se_bands : NDArray[np.float64]
        Standard error bands (lower, upper) for smoothed curve.
    raw_theta : NDArray[np.float64]
        Raw theta values from data.
    raw_probs : NDArray[np.float64]
        Raw observed item scores at each theta.
    """

    item_idx: int
    theta_grid: NDArray[np.float64]
    smoothed_probs: NDArray[np.float64]
    model_probs: NDArray[np.float64]
    se_bands: NDArray[np.float64]
    raw_theta: NDArray[np.float64]
    raw_probs: NDArray[np.float64]


def itemGAM(
    model: BaseItemModel,
    responses: NDArray[np.float64],
    theta: NDArray[np.float64],
    item_idx: int | list[int] | None = None,
    n_grid: int = 100,
    bandwidth: float | None = None,
    se: bool = True,
    alpha: float = 0.05,
    theta_margin: float = 0.1,
) -> ItemGAMResult | list[ItemGAMResult]:
    """Compute parametric smoothed regression lines for item response functions.

    Fits a kernel-smoothed regression to compare observed item performance
    with model predictions. Dichotomous results are proportions correct;
    polytomous results are collapsed to expected category scores.

    Parameters
    ----------
    model : BaseItemModel
        A fitted IRT model.
    responses : NDArray[np.float64]
        Response matrix. Shape: (n_persons, n_items).
    theta : NDArray[np.float64]
        Ability estimates. Shape: (n_persons,) or (n_persons, 1).
    item_idx : int, list of int, or None
        Item index or indices to analyze. If None, all items.
    n_grid : int
        Number of points in theta grid. Default 100.
    bandwidth : float, optional
        Kernel bandwidth. If None, uses Silverman's rule of thumb.
    se : bool
        Whether to compute standard error bands. Default True.
    alpha : float
        Significance level for confidence bands. Default 0.05 (95% CI).
    theta_margin : float
        Fraction of theta range to extend grid beyond observed values.
        Default 0.1 (10% on each side).

    Returns
    -------
    ItemGAMResult or list of ItemGAMResult
        Smoothed regression results for each item.

    Examples
    --------
    >>> result = fit_mirt(responses, model="2PL")
    >>> scores = fscores(result, responses)
    >>> gam = itemGAM(result.model, responses, scores.theta, item_idx=0)
    >>> # Plot smoothed vs model curve
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(gam.theta_grid, gam.smoothed_probs, label='Observed (smoothed)')
    >>> plt.plot(gam.theta_grid, gam.model_probs, label='Model')
    >>> plt.fill_between(gam.theta_grid, gam.se_bands[0], gam.se_bands[1], alpha=0.2)
    >>> plt.legend()
    """
    from scipy import stats

    responses, theta_2d = _validate_empirical_inputs(model, responses, theta)
    theta_values = theta_2d[:, 0]
    n_grid = _validate_positive_integer(n_grid, "n_grid", minimum=2)

    if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a finite value between 0 and 1")
    if (
        not np.isscalar(theta_margin)
        or not np.isfinite(theta_margin)
        or theta_margin < 0
    ):
        raise ValueError("theta_margin must be a finite non-negative value")

    z_crit = stats.norm.ppf(1 - alpha / 2)

    if item_idx is None:
        item_indices = list(range(model.n_items))
        single_item = False
    elif isinstance(item_idx, (int, np.integer)) and not isinstance(
        item_idx, (bool, np.bool_)
    ):
        item_indices = [_validate_item_index(model, int(item_idx))]
        single_item = True
    else:
        try:
            item_indices = [_validate_item_index(model, idx) for idx in list(item_idx)]
        except TypeError as exc:
            raise ValueError(
                "item_idx must be an integer, a list of integers, or None"
            ) from exc
        single_item = False

    if bandwidth is None:
        resolved_bandwidth = (
            SILVERMAN_CONSTANT
            * np.std(theta_values)
            * len(theta_values) ** SILVERMAN_EXPONENT
        )
        if not np.isfinite(resolved_bandwidth) or resolved_bandwidth <= 0:
            resolved_bandwidth = max(float(np.ptp(theta_values)), 1.0)
    else:
        if not np.isscalar(bandwidth) or not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("bandwidth must be a finite positive value")
        resolved_bandwidth = float(bandwidth)

    theta_min, theta_max = np.min(theta_values), np.max(theta_values)
    margin = theta_margin * (theta_max - theta_min)
    theta_grid = np.linspace(theta_min - margin, theta_max + margin, n_grid)

    if not item_indices:
        return []
    selected = (
        responses
        if item_indices == list(range(model.n_items))
        else responses[:, item_indices]
    )
    valid = np.isfinite(selected) & (selected >= 0)
    all_observed = bool(np.all(valid))
    expected = []
    max_scores = []
    for column, idx in enumerate(item_indices):
        model_probs, max_score = _expected_item_score(
            model, theta_grid.reshape(-1, 1), idx
        )
        _validate_observed_scores(selected[:, column], max_score, idx)
        expected.append(model_probs)
        max_scores.append(max_score)

    smoothed_curves, standard_errors = smooth_response_curves(
        theta_values,
        theta_grid,
        selected if all_observed else np.where(valid, selected, 0.0),
        None if all_observed else valid,
        resolved_bandwidth,
        max_elements=KERNEL_BLOCK_ELEMENTS,
        calculate_se=bool(se),
    )

    results = []
    for column, idx in enumerate(item_indices):
        smoothed_probs = smoothed_curves[column]
        if standard_errors is None:
            se_lower = np.zeros(n_grid, dtype=np.float64)
            se_upper = np.zeros(n_grid, dtype=np.float64)
        else:
            margin = z_crit * standard_errors[column]
            se_lower = np.clip(smoothed_probs - margin, 0, max_scores[column])
            se_upper = np.clip(smoothed_probs + margin, 0, max_scores[column])
        results.append(
            ItemGAMResult(
                item_idx=idx,
                theta_grid=theta_grid,
                smoothed_probs=smoothed_probs,
                model_probs=expected[column],
                se_bands=np.array([se_lower, se_upper]),
                raw_theta=theta_values[valid[:, column]],
                raw_probs=selected[valid[:, column], column],
            )
        )

    if single_item:
        return results[0]
    return results
