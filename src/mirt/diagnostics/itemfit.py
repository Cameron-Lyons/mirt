"""Mean-square item fit and summed-score conditional S-X2 diagnostics."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import chdtrc

from mirt.diagnostics.multiple_testing import (
    PValueAdjustment,
    _validate_p_value_adjustment,
    adjust_p_values,
)
from mirt.utils.numeric import _FitStatsAccumulator, compute_expected_variance

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel


_SX2_TARGET_CHUNK_ELEMENTS = 1_000_000
_ITEMFIT_TARGET_CHUNK_ELEMENTS = 262_144


def compute_itemfit(
    model: BaseItemModel,
    responses: NDArray[np.int_] | None = None,
    statistics: list[str] | None = None,
    theta: NDArray[np.float64] | None = None,
    n_groups: int | None = None,
    p_adjust: PValueAdjustment = "none",
    *,
    min_expected: float = 1.0,
    n_quadpts: int = 41,
    quadrature_points: ArrayLike | None = None,
    quadrature_weights: ArrayLike | None = None,
    item_parameter_counts: ArrayLike | None = None,
    na_rm: bool = False,
) -> dict[str, NDArray[np.float64]]:
    """Compute mean-square statistics and Orlando-Thissen S-X2 item fit.

    S-X2 compares observed category counts with model-implied counts
    conditional on the *exact total score*, integrating over the latent
    distribution. It supports complete binary or consecutively scored ordinal
    responses with conditionally independent items. MixtureIRT and
    HigherOrderCDM require joint integration over shared classes or mastery
    patterns and are not supported by S-X2. Negative codes and NaN denote
    missing S-X2 responses.
    ``na_rm=True`` excludes incomplete persons from S-X2;
    otherwise missing responses raise an error. Infit and outfit use all
    available responses regardless of ``na_rm``.

    ``theta`` supplies person abilities for mean-square statistics, but does
    not define S-X2 expected counts. S-X2 uses standard-normal quadrature by
    default; explicit points and probability-mass weights can specify a
    different fitted latent distribution. ``n_groups`` is deprecated because
    score quantiles are not the conditioning groups in S-X2.

    ``min_expected`` controls sparse-cell pooling, with zero disabling it.
    Binary items pool adjacent score rows; ordinal items pool extreme score
    rows and then adjacent response categories within each row. Degrees of
    freedom equal retained category contrasts minus estimated item parameters.
    The default parameter counts come from ``model.free_parameter_masks``;
    ``item_parameter_counts`` can supply counts for constrained/shared models
    or zeros when testing externally known item parameters. Nonpositive
    degrees of freedom give ``df=0`` and ``p_value=NaN``.
    An ordinal item whose maximum score exceeds that of the remaining test
    has no full-category score group; its S-X2 statistic is also ``NaN``.

    The S-X2 result keys are ``S_X2``, ``df``, and ``p_value``; requesting a
    multiplicity correction adds ``p_value_adjusted``. Probability evaluation,
    response counting, and score recursion use bounded row blocks.
    """
    p_adjust = _validate_p_value_adjustment(p_adjust, name="p_adjust")
    if statistics is None:
        statistics = ["infit", "outfit"]
    if responses is None:
        raise ValueError("responses required for item fit statistics")
    responses = np.asarray(responses)
    if responses.ndim != 2 or not all(responses.shape):
        raise ValueError("responses must be a nonempty two-dimensional matrix")
    n_persons, n_items = responses.shape
    if n_items != model.n_items:
        raise ValueError(f"responses must contain {model.n_items} model items")
    compute_mean_squares = "infit" in statistics or "outfit" in statistics
    compute_sx2 = "S_X2" in statistics
    if not compute_mean_squares and not compute_sx2:
        return {}

    if theta is not None:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta.reshape(-1, 1)
        if theta.ndim != 2 or theta.shape != (
            n_persons,
            getattr(model, "n_factors", 1),
        ):
            raise ValueError(
                "theta must be a matrix with one row per person and model factor"
            )
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta must contain only finite values")

    result: dict[str, NDArray[np.float64]] = {}
    if compute_sx2:
        if n_groups is not None:
            _validate_n_groups(n_groups)
            warnings.warn(
                "n_groups is deprecated for S-X2; exact total scores and sparse-cell pooling define its groups",
                DeprecationWarning,
                stacklevel=2,
            )
        result.update(
            _compute_s_x2(
                model,
                responses,
                min_expected=min_expected,
                n_quadpts=n_quadpts,
                quadrature_points=quadrature_points,
                quadrature_weights=quadrature_weights,
                item_parameter_counts=item_parameter_counts,
                na_rm=na_rm,
            )
        )
        if p_adjust != "none":
            result["p_value_adjusted"] = adjust_p_values(result["p_value"], p_adjust)

    if compute_mean_squares:
        if theta is None:
            from mirt.scoring import fscores

            theta = fscores(model, responses, method="EAP").theta
            theta = np.asarray(theta, dtype=np.float64).reshape(
                n_persons, getattr(model, "n_factors", 1)
            )
        category_width = max(model.n_categories) if model.is_polytomous else 1
        rows_per_chunk = max(
            1, _ITEMFIT_TARGET_CHUNK_ELEMENTS // (n_items * category_width)
        )
        accumulator = _FitStatsAccumulator(n_items)
        for start in range(0, n_persons, rows_per_chunk):
            stop = min(start + rows_per_chunk, n_persons)
            expected, variance = compute_expected_variance(
                model, theta[start:stop], n_items
            )
            accumulator.add(responses[start:stop], expected, variance)
        infit, outfit = accumulator.finish()
        if "outfit" in statistics:
            result["outfit"] = outfit
        if "infit" in statistics:
            result["infit"] = infit
    return result


def _validate_n_groups(n_groups: int) -> int:
    if isinstance(n_groups, (bool, np.bool_)) or not isinstance(
        n_groups, (int, np.integer)
    ):
        raise ValueError("n_groups must be an integer")
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2")
    return int(n_groups)


def _sx2_categories(model: BaseItemModel) -> NDArray[np.int64]:
    categories = np.asarray(
        model.n_categories if model.is_polytomous else [2] * model.n_items
    )
    if (
        categories.shape != (model.n_items,)
        or categories.dtype.kind not in "iu"
        or np.any(categories < 2)
    ):
        raise ValueError(
            "S-X2 requires at least two consecutive score categories per item"
        )
    return categories.astype(np.int64, copy=False)


def _sx2_response_counts(
    responses: NDArray[np.int_], categories: NDArray[np.int64], *, na_rm: bool
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
    """Count exact total-score/category cells without full matrix copies."""
    n_persons, n_items = responses.shape
    if responses.dtype.kind not in "biuf":
        raise ValueError("S-X2 responses must contain numeric category codes")
    n_scores = int(np.sum(categories - 1)) + 1
    tables = [np.zeros((n_scores, int(count))) for count in categories]
    score_counts = np.zeros(n_scores)
    chunk_rows = max(1, _SX2_TARGET_CHUNK_ELEMENTS // n_items)
    for start in range(0, n_persons, chunk_rows):
        block = responses[start : start + chunk_rows]
        if np.any(np.isinf(block)):
            raise ValueError("S-X2 responses must not contain infinite values")
        missing = np.isnan(block) | (block < 0)
        if np.any(missing) and not na_rm:
            raise ValueError(
                "S-X2 requires complete responses; set na_rm=True to exclude incomplete persons"
            )
        if np.any(missing):
            block = block[~np.any(missing, axis=1)]
        if len(block) == 0:
            continue
        if np.any(block != np.floor(block)) or np.any(block >= categories):
            raise ValueError(
                "S-X2 responses must be integer category codes within each item's range"
            )
        integer_block = block.astype(np.int64, copy=False)
        totals = integer_block.sum(axis=1)
        score_counts += np.bincount(totals, minlength=n_scores)
        for item, count in enumerate(categories):
            codes = totals * count + integer_block[:, item]
            tables[item] += np.bincount(codes, minlength=n_scores * int(count)).reshape(
                n_scores, int(count)
            )
    if score_counts.sum() == 0:
        raise ValueError("S-X2 responses contain no complete persons")
    return tables, score_counts


def _sx2_parameter_counts(
    model: BaseItemModel, counts: ArrayLike | None
) -> NDArray[np.int64]:
    if counts is not None:
        values = np.asarray(counts)
        if (
            values.shape != (model.n_items,)
            or values.dtype.kind not in "iu"
            or np.any(values < 0)
        ):
            raise ValueError(
                "item_parameter_counts must contain one nonnegative integer per item"
            )
        return values.astype(np.int64, copy=False)
    result = np.zeros(model.n_items, dtype=np.int64)
    shared_design = (
        getattr(model, "model_name", "") in {"RSM", "GRSM"}
        or hasattr(model, "item_features")
        or hasattr(model, "testlet_membership")
        or "class_proportions" in getattr(model, "parameters", {})
    )
    if shared_design:
        raise ValueError(
            "shared parameters require explicit item_parameter_counts for S-X2"
        )
    masks = getattr(model, "free_parameter_masks", {})
    for mask in masks.values():
        values = np.asarray(mask, dtype=bool)
        if values.ndim == 0 or values.shape[0] != model.n_items:
            raise ValueError(
                "shared parameters require explicit item_parameter_counts for S-X2"
            )
        result += np.count_nonzero(values.reshape(model.n_items, -1), axis=1)
    return result


def _sx2_quadrature(
    model: BaseItemModel,
    n_quadpts: int,
    points: ArrayLike | None,
    weights: ArrayLike | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_factors = getattr(model, "n_factors", 1)
    if (points is None) != (weights is None):
        raise ValueError(
            "quadrature_points and quadrature_weights must be supplied together"
        )
    if points is None:
        if (
            isinstance(n_quadpts, (bool, np.bool_))
            or not isinstance(n_quadpts, (int, np.integer))
            or n_quadpts < 2
        ):
            raise ValueError("n_quadpts must be an integer of at least 2")
        if n_quadpts**n_factors > 100_000:
            raise ValueError(
                "default quadrature exceeds 100000 points; supply a bounded explicit quadrature grid"
            )
        from mirt.estimation.quadrature import GaussHermiteQuadrature

        grid = GaussHermiteQuadrature(n_points=int(n_quadpts), n_dimensions=n_factors)
        return grid.nodes, grid.weights
    nodes = np.asarray(points, dtype=np.float64)
    masses = np.asarray(weights, dtype=np.float64)
    if nodes.ndim == 1 and n_factors == 1:
        nodes = nodes.reshape(-1, 1)
    if (
        nodes.ndim != 2
        or nodes.shape[1] != n_factors
        or nodes.shape[0] == 0
        or not np.all(np.isfinite(nodes))
    ):
        raise ValueError(
            "quadrature_points must contain finite rows with one column per model factor"
        )
    if (
        masses.shape != (nodes.shape[0],)
        or not np.all(np.isfinite(masses))
        or np.any(masses < 0)
        or not np.any(masses > 0)
    ):
        raise ValueError(
            "quadrature_weights must contain one finite nonnegative mass per point and positive total mass"
        )
    masses = masses / np.max(masses)
    return nodes, masses / masses.sum()


def _score_distribution(
    probabilities: NDArray[np.float64], categories: NDArray[np.int64], skip: int = -1
) -> NDArray[np.float64]:
    """Lord-Wingersky score recursion, avoiding unstable polynomial division."""
    distribution = np.ones((probabilities.shape[0], 1))
    for item, count in enumerate(categories):
        if item == skip:
            continue
        updated = np.zeros(
            (probabilities.shape[0], distribution.shape[1] + int(count) - 1)
        )
        for category in range(int(count)):
            updated[:, category : category + distribution.shape[1]] += (
                distribution * probabilities[:, item, category, None]
            )
        distribution = updated
    return distribution


def _conditional_category_probabilities(
    model: BaseItemModel,
    categories: NDArray[np.int64],
    nodes: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
    """Integrate joint item-category/total-score probabilities over latent mass."""
    n_scores = int(np.sum(categories - 1)) + 1
    n_categories = int(np.max(categories))
    joint = [np.zeros((n_scores, int(count))) for count in categories]
    marginal = np.zeros(n_scores)
    block_rows = max(
        1,
        _ITEMFIT_TARGET_CHUNK_ELEMENTS
        // max(model.n_items * n_categories, n_scores * 3),
    )
    for start in range(0, len(nodes), block_rows):
        stop = min(start + block_rows, len(nodes))
        raw = np.asarray(model.probability(nodes[start:stop]), dtype=np.float64)
        if not model.is_polytomous:
            if raw.shape != (stop - start, model.n_items):
                raise ValueError(
                    "model probabilities must have one row per quadrature point and one column per item"
                )
            raw = np.stack((1 - raw, raw), axis=2)
        if raw.shape != (stop - start, model.n_items, n_categories):
            raise ValueError(
                "model probabilities must match quadrature points, items, and maximum category count"
            )
        if not np.all(np.isfinite(raw)) or np.any((raw < 0) | (raw > 1)):
            raise ValueError(
                "model probabilities must be finite and between zero and one"
            )
        for item, count in enumerate(categories):
            if np.any(raw[:, item, count:] != 0) or not np.allclose(
                raw[:, item, :count].sum(axis=1), 1, rtol=1e-8, atol=1e-10
            ):
                raise ValueError(
                    "model category probabilities must sum to one with zero padding"
                )
        block_weights = weights[start:stop]
        marginal += block_weights @ _score_distribution(raw, categories)
        for item, count in enumerate(categories):
            rest = _score_distribution(raw, categories, skip=item)
            for category in range(int(count)):
                joint[item][category : category + rest.shape[1], category] += (
                    block_weights * raw[:, item, category]
                ) @ rest
    conditional = [
        np.divide(
            table,
            marginal[:, None],
            out=np.zeros_like(table),
            where=marginal[:, None] > 0,
        )
        for table in joint
    ]
    return conditional, marginal


def _pool_score_rows(
    observed: NDArray[np.float64], expected: NDArray[np.float64], minimum: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pool binary score rows into their less populated adjacent neighbor."""
    observed = observed.copy()
    expected = expected.copy()
    while len(expected) > 1:
        sparse = np.flatnonzero(np.min(expected, axis=1) < minimum)
        if sparse.size == 0:
            break
        row = int(sparse[0])
        if row == 0:
            neighbor = 1
        elif row == len(expected) - 1:
            neighbor = row - 1
        else:
            neighbor = (
                row - 1
                if expected[row - 1].sum() <= expected[row + 1].sum()
                else row + 1
            )
        observed[neighbor] += observed[row]
        expected[neighbor] += expected[row]
        observed = np.delete(observed, row, axis=0)
        expected = np.delete(expected, row, axis=0)
    return observed, expected


def _pool_categories(
    observed: NDArray[np.float64], expected: NDArray[np.float64], minimum: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pool sparse adjacent ordinal response categories within one score row."""
    observed = observed.copy()
    expected = expected.copy()
    while len(expected) > 1 and np.min(expected) < minimum:
        category = int(np.argmin(expected))
        if category == 0:
            neighbor = 1
        elif category == len(expected) - 1:
            neighbor = category - 1
        else:
            neighbor = (
                category - 1
                if expected[category - 1] <= expected[category + 1]
                else category + 1
            )
        observed[neighbor] += observed[category]
        expected[neighbor] += expected[category]
        observed = np.delete(observed, category)
        expected = np.delete(expected, category)
    return observed, expected


def _sx2_from_tables(
    observed: NDArray[np.float64],
    expected: NDArray[np.float64],
    n_parameters: int,
    minimum: float,
) -> tuple[float, int, float]:
    """Apply ordered pooling and count remaining independent category contrasts."""
    n_categories = observed.shape[1]
    n_scores = observed.shape[0]
    # Perfect and zero total scores are deterministic and supply no item-fit
    # information. For ordinal items, pool remaining structurally incomplete
    # tail rows into the nearest full-category score row (Kang and Chen, 2008).
    observed = observed[1:-1].copy()
    expected = expected[1:-1].copy()
    if n_categories > 2 and len(observed):
        high = n_scores - n_categories - 1
        low = n_categories - 2
        if high < low:
            return np.nan, 0, np.nan
        if high == low:
            observed = observed.sum(axis=0, keepdims=True)
            expected = expected.sum(axis=0, keepdims=True)
        else:
            observed[low] += observed[:low].sum(axis=0)
            expected[low] += expected[:low].sum(axis=0)
            observed[high] += observed[high + 1 :].sum(axis=0)
            expected[high] += expected[high + 1 :].sum(axis=0)
            observed = observed[low : high + 1]
            expected = expected[low : high + 1]
    populated = observed.sum(axis=1) > 0
    observed, expected = observed[populated], expected[populated]
    if n_categories == 2 and minimum > 0:
        observed, expected = _pool_score_rows(observed, expected, minimum)
    statistic = 0.0
    contrasts = 0
    sparse_remaining = False
    for observed_row, expected_row in zip(observed, expected, strict=True):
        if n_categories > 2 and minimum > 0:
            observed_row, expected_row = _pool_categories(
                observed_row, expected_row, minimum
            )
        positive = expected_row > 0
        contrasts += max(int(np.count_nonzero(positive)) - 1, 0)
        if np.any((~positive) & (observed_row > 0)):
            statistic = np.inf
        statistic += float(
            np.sum(
                (observed_row[positive] - expected_row[positive]) ** 2
                / expected_row[positive]
            )
        )
        sparse_remaining |= bool(np.any(expected_row[positive] < minimum))
    degrees = max(contrasts - n_parameters, 0)
    p_value = (
        float(chdtrc(degrees, statistic))
        if degrees > 0 and not sparse_remaining
        else np.nan
    )
    return statistic, degrees, p_value


def _compute_s_x2(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    *,
    min_expected: float,
    n_quadpts: int,
    quadrature_points: ArrayLike | None,
    quadrature_weights: ArrayLike | None,
    item_parameter_counts: ArrayLike | None,
    na_rm: bool,
) -> dict[str, NDArray[np.float64]]:
    from mirt.models.cdm_advanced import HigherOrderCDM
    from mirt.models.mixture import MixtureIRT

    if isinstance(model, MixtureIRT):
        raise ValueError(
            "S-X2 does not support MixtureIRT: latent class integration is "
            "required for the joint score distribution"
        )
    if isinstance(model, HigherOrderCDM):
        raise ValueError(
            "S-X2 does not support HigherOrderCDM: shared mastery-pattern "
            "integration is required for the joint score distribution"
        )
    if isinstance(min_expected, (bool, np.bool_)):
        raise ValueError("min_expected must be finite and nonnegative")
    try:
        minimum = float(min_expected)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_expected must be finite and nonnegative") from exc
    if not np.isfinite(minimum) or minimum < 0:
        raise ValueError("min_expected must be finite and nonnegative")
    if not isinstance(na_rm, (bool, np.bool_)):
        raise ValueError("na_rm must be boolean")
    categories = _sx2_categories(model)
    observed, score_counts = _sx2_response_counts(responses, categories, na_rm=na_rm)
    parameter_counts = _sx2_parameter_counts(model, item_parameter_counts)
    nodes, weights = _sx2_quadrature(
        model, n_quadpts, quadrature_points, quadrature_weights
    )
    conditional, marginal = _conditional_category_probabilities(
        model, categories, nodes, weights
    )
    if np.any((score_counts > 0) & (marginal == 0)):
        raise ValueError(
            "an observed total score has zero probability under the model and latent distribution"
        )
    summaries = [
        _sx2_from_tables(
            table,
            conditional[item] * score_counts[:, None],
            int(parameter_counts[item]),
            minimum,
        )
        for item, table in enumerate(observed)
    ]
    values = np.asarray(summaries)
    return {"S_X2": values[:, 0], "df": values[:, 1], "p_value": values[:, 2]}


def compute_s_x2(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64] | None = None,
    n_groups: int | None = None,
    p_adjust: PValueAdjustment = "none",
    *,
    min_expected: float = 1.0,
    n_quadpts: int = 41,
    quadrature_points: ArrayLike | None = None,
    quadrature_weights: ArrayLike | None = None,
    item_parameter_counts: ArrayLike | None = None,
    na_rm: bool = False,
) -> dict[str, NDArray[np.float64]]:
    """Compute exact-total-score conditional Orlando-Thissen S-X2 item fit.

    Binary and ordinal expected counts are integrated over the latent ability
    distribution by score recursion. See :func:`compute_itemfit` for pooling,
    quadrature, parameter-count, missing-response, and multiplicity controls.
    ``theta`` is accepted and validated for compatibility; S-X2 does not use
    plug-in respondent ability estimates. ``n_groups`` is deprecated.
    """
    return compute_itemfit(
        model,
        responses,
        statistics=["S_X2"],
        theta=theta,
        n_groups=n_groups,
        p_adjust=p_adjust,
        min_expected=min_expected,
        n_quadpts=n_quadpts,
        quadrature_points=quadrature_points,
        quadrature_weights=quadrature_weights,
        item_parameter_counts=item_parameter_counts,
        na_rm=na_rm,
    )
