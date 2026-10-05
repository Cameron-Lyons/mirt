"""Mean-square, theta-binned and summed-score conditional S-X2 item fit."""

from __future__ import annotations

import warnings
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import chdtrc

from mirt.diagnostics.multiple_testing import (
    PValueAdjustment,
    _validate_p_value_adjustment,
    adjust_p_values,
)
from mirt.exceptions import MirtValidationError
from mirt.typing import ItemFitStatistic
from mirt.utils.numeric import (
    _FitStatsAccumulator,
    _fourth_central_moment,
    compute_probability_moments,
)

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult


_SX2_TARGET_CHUNK_ELEMENTS = 1_000_000
_SPARSE_RELATIVE_TOLERANCE = 1e-10
_ITEMFIT_TARGET_CHUNK_ELEMENTS = 262_144
_ITEMFIT_STATISTICS: tuple[str, ...] = get_args(ItemFitStatistic)
_MEAN_SQUARE_STATISTICS = frozenset({"infit", "outfit", "z_infit", "z_outfit"})
_BINNED_STATISTICS = ("X2", "G2", "PV_Q1")
_DEFAULT_THETA_GROUPS = 10


def compute_itemfit(
    model: BaseItemModel | FitResult,
    responses: NDArray[np.int_] | None = None,
    statistics: Sequence[str] | str | None = None,
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
    n_plausible: int = 100,
    seed: int | None = None,
    prior_mean: ArrayLike | None = None,
    prior_cov: ArrayLike | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute mean-square, theta-binned and Orlando-Thissen S-X2 item fit.

    Parameters
    ----------
    model : BaseItemModel or FitResult
        Fitted item response model, or the ``FitResult`` of a fit, whose
        estimated ``latent_covariance`` then defines the latent population.
    responses : ndarray of shape (n_persons, n_items)
        Integer category codes. Negative codes and NaN denote missing
        responses.
    statistics : list of str, optional
        Any of ``"infit"``, ``"outfit"``, ``"z_infit"``, ``"z_outfit"``,
        ``"S_X2"``, ``"X2"``, ``"G2"`` and ``"PV_Q1"``. Defaults to
        ``["infit", "outfit"]``. Unknown names raise
        :class:`~mirt.exceptions.MirtValidationError`.
    theta : ndarray of shape (n_persons,) or (n_persons, n_factors), optional
        Person abilities for the mean-square and X2/G2 statistics. EAP scores
        are computed when omitted.
    n_groups : int, optional
        Number of ability groups for X2, G2 and PV_Q1 (default 10). It is
        deprecated for S-X2, which conditions on exact total scores.
    p_adjust : {"none", "bonferroni", "holm", "fdr_bh"}, default="none"
        Multiple-testing adjustment across items for every chi-square test.
    min_expected : float, default=1.0
        Minimum expected S-X2 cell count for sparse-cell pooling.
    n_quadpts : int, default=41
        Gauss-Hermite quadrature points per model factor for S-X2.
    quadrature_points, quadrature_weights : array-like, optional
        Explicit latent grid and probability masses for S-X2. They replace
        the normal population of ``prior_mean`` and ``prior_cov`` for S-X2.
    item_parameter_counts : array-like, optional
        Estimated parameters per item for chi-square degrees of freedom.
    na_rm : bool, default=False
        Exclude incomplete persons from S-X2.
    n_plausible : int, default=100
        Plausible-value draws for PV_Q1.
    seed : int, optional
        Seed for the PV_Q1 plausible-value draws.
    prior_mean : array-like of shape (n_factors,), optional
        Mean of the normal latent population. Default zero.
    prior_cov : array-like of shape (n_factors, n_factors), optional
        Covariance of the normal latent population. Defaults to the
        ``latent_covariance`` of a ``FitResult`` when it has one, and to the
        identity otherwise.

    Returns
    -------
    dict of str to ndarray
        One array of length ``n_items`` per requested statistic and its
        companion degrees of freedom and p-values.

    Notes
    -----
    Infit and outfit are information-weighted and unweighted mean squares.
    Outfit excludes near-deterministic entries whose modeled variance is at
    most ``PROB_EPSILON``. ``z_infit`` and ``z_outfit`` are their
    Wilson-Hilferty standardizations, ``(MS^(1/3) - 1)(3/q) + q/3``, with the
    mean-square variance ``q^2`` taken from the second and fourth central
    moments of each modeled score (Wright & Masters, 1982). They are
    approximately standard normal under the model.

    S-X2 compares observed category counts with model-implied counts
    conditional on the *exact total score*, integrating over the latent
    distribution. It supports complete binary or consecutively scored ordinal
    responses with conditionally independent items. MixtureIRT and
    HigherOrderCDM require joint integration over shared classes or mastery
    patterns and are not supported by S-X2.
    ``na_rm=True`` excludes incomplete persons from S-X2;
    otherwise missing responses raise an error. Infit and outfit use all
    available responses regardless of ``na_rm``.

    ``theta`` supplies person abilities for mean-square statistics, but does
    not define S-X2 expected counts. S-X2 integrates over the normal latent
    population, by default the standard normal or a ``FitResult``'s estimated
    factor covariance, on a Gauss-Hermite grid; explicit points and
    probability-mass weights can specify a different fitted latent
    distribution. The EAP abilities that replace an omitted ``theta``, and
    the PV_Q1 posteriors, use the same population as their prior.

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

    ``X2`` (Bock, 1972; Yen, 1981), its likelihood-ratio analogue ``G2`` and
    ``PV_Q1`` (Chalmers & Ng, 2017) group persons into ``n_groups`` ability
    quantiles and compare observed category counts with the model's
    probabilities at each group's mean ability. They support unidimensional
    models only. Each adds ``<name>_df`` and ``<name>_p`` (and
    ``<name>_p_adjusted`` under ``p_adjust``). Because X2 and G2 group on
    estimated abilities, their p-values are approximate and liberal, severely
    so on short tests where an ability estimate depends heavily on the item's
    own response. Sparse cells are not pooled, so rare categories in small
    groups inflate the statistics further. PV_Q1 instead recomputes X2 on
    ``n_plausible`` plausible-value draws from the normal approximation
    ``N(EAP, PSD^2)`` of each person's posterior (independently of ``theta``)
    and reports the median statistic, which keeps closer to the nominal error
    rate.
    """
    from mirt.results._common import resolve_latent_prior

    model, prior_mean, prior_cov = resolve_latent_prior(model, prior_mean, prior_cov)
    p_adjust = _validate_p_value_adjustment(p_adjust, name="p_adjust")
    requested = _validate_statistics(
        statistics, _ITEMFIT_STATISTICS, default=("infit", "outfit")
    )
    if responses is None:
        raise ValueError("responses required for item fit statistics")
    responses = np.asarray(responses)
    if responses.ndim != 2 or not all(responses.shape):
        raise ValueError("responses must be a nonempty two-dimensional matrix")
    n_persons, n_items = responses.shape
    if n_items != model.n_items:
        raise ValueError(f"responses must contain {model.n_items} model items")
    mean_squares = _MEAN_SQUARE_STATISTICS.intersection(requested)
    binned = [name for name in _BINNED_STATISTICS if name in requested]
    n_factors = getattr(model, "n_factors", 1)

    if theta is not None:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta.reshape(-1, 1)
        if theta.ndim != 2 or theta.shape != (n_persons, n_factors):
            raise ValueError(
                "theta must be a matrix with one row per person and model factor"
            )
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta must contain only finite values")

    if binned:
        from mirt.diagnostics.itemfit_binned import _validate_binned_options

        n_groups = _validate_binned_options(
            model,
            _DEFAULT_THETA_GROUPS if n_groups is None else n_groups,
            n_plausible,
            plausible="PV_Q1" in binned,
        )

    result: dict[str, NDArray[np.float64]] = {}
    if "S_X2" in requested:
        if n_groups is not None and not binned:
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
                prior_mean=prior_mean,
                prior_cov=prior_cov,
            )
        )
        if p_adjust != "none":
            result["p_value_adjusted"] = adjust_p_values(result["p_value"], p_adjust)

    if not mean_squares and not binned:
        return result

    if responses.dtype.kind == "f":
        responses = np.where(np.isnan(responses), -1.0, responses)
    posterior: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
    if theta is None or "PV_Q1" in binned:
        from mirt.scoring import fscores

        scores = fscores(
            model,
            responses,
            method="EAP",
            prior_mean=prior_mean,
            prior_cov=prior_cov,
        )
        eap = np.asarray(scores.theta, dtype=np.float64).reshape(n_persons, n_factors)
        if "PV_Q1" in binned:
            spread = np.asarray(scores.standard_error, dtype=np.float64)
            posterior = (eap[:, 0], spread.reshape(n_persons, n_factors)[:, 0])
        if theta is None:
            theta = eap

    if mean_squares:
        result.update(_mean_square_fit(model, responses, theta, mean_squares))

    if binned:
        from mirt.diagnostics.itemfit_binned import _compute_binned_itemfit

        result.update(
            _compute_binned_itemfit(
                model,
                responses,
                binned,
                theta=theta[:, 0],
                posterior=posterior,
                n_groups=int(n_groups),
                n_plausible=n_plausible,
                seed=seed,
                item_parameter_counts=item_parameter_counts,
                p_adjust=p_adjust,
            )
        )
    return result


def _validate_statistics(
    statistics: Sequence[str] | str | None,
    allowed: Collection[str],
    *,
    default: Sequence[str],
) -> list[str]:
    """Return requested fit statistic names, rejecting unknown names."""
    if statistics is None:
        return list(default)
    names = [statistics] if isinstance(statistics, str) else list(statistics)
    if not names:
        raise MirtValidationError(
            "statistics must name at least one statistic",
            parameter="statistics",
            expected=", ".join(allowed),
        )
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise MirtValidationError(
            f"Unknown fit statistic(s) {unknown}; choose from {list(allowed)}",
            parameter="statistics",
            value=unknown,
            expected=", ".join(allowed),
        )
    return names


def _mean_square_fit(
    model: BaseItemModel,
    responses: NDArray[np.int_],
    theta: NDArray[np.float64],
    statistics: Collection[str],
) -> dict[str, NDArray[np.float64]]:
    """Accumulate item infit/outfit (and their z statistics) in row blocks."""
    n_persons, n_items = responses.shape
    standardized = "z_infit" in statistics or "z_outfit" in statistics
    category_width = max(model.n_categories) if model.is_polytomous else 1
    rows_per_chunk = max(
        1, _ITEMFIT_TARGET_CHUNK_ELEMENTS // (n_items * category_width)
    )
    accumulator = _FitStatsAccumulator(n_items, standardized=standardized)
    for start in range(0, n_persons, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_persons)
        probabilities, expected, variance = compute_probability_moments(
            model, theta[start:stop], n_items
        )
        accumulator.add(
            responses[start:stop],
            expected,
            variance,
            fourth_moment=(
                _fourth_central_moment(probabilities, expected)
                if standardized
                else None
            ),
        )
    return {
        name: values
        for name, values in accumulator.statistics().items()
        if name in statistics
    }


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
    model: BaseItemModel, counts: ArrayLike | None, *, statistic: str = "S-X2"
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
    from mirt.models.mixed_format import MixedItemModel

    if isinstance(model, MixedItemModel):
        # Component arrays are indexed by each component's own items.
        for component, items in model.components:
            result[items] = _sx2_parameter_counts(component, None, statistic=statistic)
        return result
    shared_design = (
        bool(getattr(model, "_shared_parameters", ()))
        or hasattr(model, "item_features")
        or hasattr(model, "testlet_membership")
        or "class_proportions" in getattr(model, "parameters", {})
    )
    if shared_design:
        raise ValueError(
            f"shared parameters require explicit item_parameter_counts for {statistic}"
        )
    masks = getattr(model, "free_parameter_masks", {})
    for mask in masks.values():
        values = np.asarray(mask, dtype=bool)
        if values.ndim == 0 or values.shape[0] != model.n_items:
            raise ValueError(
                f"shared parameters require explicit item_parameter_counts for {statistic}"
            )
        result += np.count_nonzero(values.reshape(model.n_items, -1), axis=1)
    return result


def _sx2_quadrature(
    model: BaseItemModel,
    n_quadpts: int,
    points: ArrayLike | None,
    weights: ArrayLike | None,
    prior_mean: ArrayLike | None = None,
    prior_cov: ArrayLike | None = None,
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
        from mirt.scoring._common import build_quadrature

        return build_quadrature(
            n_quadpts=int(n_quadpts),
            n_factors=n_factors,
            prior_mean=None if prior_mean is None else np.asarray(prior_mean),
            prior_cov=None if prior_cov is None else np.asarray(prior_cov),
        )
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


def _add_item_to_distribution(
    distribution: NDArray[np.float64], probabilities: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Add one item to per-node score distributions (Lord-Wingersky step)."""
    width = distribution.shape[1]
    updated = np.zeros((distribution.shape[0], width + probabilities.shape[1] - 1))
    for category in range(probabilities.shape[1]):
        updated[:, category : category + width] += (
            distribution * probabilities[:, category, None]
        )
    return updated


def _antidiagonal_sums(matrices: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``out[..., s] = sum(matrices[..., a, b] for a + b == s)``."""
    if matrices.shape[-2] > matrices.shape[-1]:
        matrices = matrices.swapaxes(-1, -2)
    rows, columns = matrices.shape[-2:]
    padded = np.zeros((*matrices.shape[:-2], rows, rows + columns - 1))
    row_stride, column_stride = padded.strides[-2:]
    # Shift row a right by a places; the shifted cells never overlap.
    sheared = np.lib.stride_tricks.as_strided(
        padded,
        shape=matrices.shape,
        strides=(*padded.strides[:-2], row_stride + column_stride, column_stride),
    )
    sheared[...] = matrices
    return padded.sum(axis=-2)


def _node_category_probabilities(
    model: BaseItemModel,
    nodes: NDArray[np.float64],
    categories: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Validate zero-padded category probabilities at quadrature nodes.

    Probabilities are evaluated in node batches of bounded size.
    """
    n_categories = int(np.max(categories))
    batch_rows = max(
        1, _ITEMFIT_TARGET_CHUNK_ELEMENTS // (model.n_items * n_categories)
    )
    batches = []
    for start in range(0, len(nodes), batch_rows):
        stop = min(start + batch_rows, len(nodes))
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
        batches.append(raw)
    return batches[0] if len(batches) == 1 else np.concatenate(batches)


def _conditional_category_probabilities(
    model: BaseItemModel,
    categories: NDArray[np.int64],
    nodes: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
    """Integrate joint item-category/total-score probabilities over latent mass.

    The rest-score distribution of every item is the convolution of the
    score distributions of the items before it (prefix) and after it
    (suffix), so one forward and one backward recursion replace a full
    leave-one-out recursion per item. Each item's joint table then takes one
    matrix product over the nodes followed by antidiagonal sums.
    """
    n_scores = int(np.sum(categories - 1)) + 1
    n_items = len(categories)
    joint = [np.zeros((n_scores, int(count))) for count in categories]
    marginal = np.zeros(n_scores)
    # Suffix distributions hold about n_items * n_scores / 2 values per node.
    # Larger node blocks keep the matrix products efficient on long forms.
    block_rows = max(1, _SX2_TARGET_CHUNK_ELEMENTS // (n_items * n_scores))
    for start in range(0, len(nodes), block_rows):
        stop = min(start + block_rows, len(nodes))
        raw = _node_category_probabilities(model, nodes[start:stop], categories)
        block_weights = weights[start:stop]

        suffixes: list[NDArray[np.float64]] = []
        distribution = np.ones((stop - start, 1))
        for item in range(n_items - 1, -1, -1):
            suffixes.append(distribution)
            distribution = _add_item_to_distribution(
                distribution, raw[:, item, : categories[item]]
            )
        marginal += block_weights @ distribution

        prefix = np.ones((stop - start, 1))
        for item, count in enumerate(categories):
            item_probabilities = raw[:, item, :count]
            suffix = suffixes.pop()
            weighted = (block_weights[:, None] * item_probabilities)[:, :, None]
            weighted = (weighted * prefix[:, None, :]).reshape(stop - start, -1)
            products = (weighted.T @ suffix).reshape(int(count), prefix.shape[1], -1)
            sums = _antidiagonal_sums(products)
            for category in range(int(count)):
                joint[item][category : category + sums.shape[1], category] += sums[
                    category
                ]
            prefix = _add_item_to_distribution(prefix, item_probabilities)
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
    """Pool binary score rows into their less populated adjacent neighbor.

    Row populations are the observed person counts. Expected row totals equal
    them only up to rounding, which must not break ties between neighbors.
    """
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
                if observed[row - 1].sum() <= observed[row + 1].sum()
                else row + 1
            )
        observed[neighbor] += observed[row]
        expected[neighbor] += expected[row]
        observed = np.delete(observed, row, axis=0)
        expected = np.delete(expected, row, axis=0)
    return observed, expected


def _pool_categories(
    observed: list[float], expected: list[float], minimum: float
) -> tuple[list[float], list[float]]:
    """Pool sparse adjacent ordinal response categories within one score row."""
    observed = list(observed)
    expected = list(expected)
    while len(expected) > 1 and min(expected) < minimum:
        category = expected.index(min(expected))
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
        del observed[category], expected[category]
    return observed, expected


def _chi_square_terms(
    observed: NDArray[np.float64], expected: NDArray[np.float64], minimum: float
) -> tuple[float, int, bool, bool]:
    """Sum Pearson terms and contrasts over score rows without pooling."""
    positive = expected > 0
    safe_expected = np.where(positive, expected, 1.0)
    statistic = float(
        np.sum(np.where(positive, (observed - expected) ** 2 / safe_expected, 0.0))
    )
    contrasts = int(np.sum(np.maximum(np.count_nonzero(positive, axis=-1) - 1, 0)))
    impossible = bool(np.any(~positive & (observed > 0)))
    sparse = bool(np.any(positive & (expected < minimum)))
    return statistic, contrasts, impossible, sparse


def _sx2_from_tables(
    observed: NDArray[np.float64],
    expected: NDArray[np.float64],
    n_parameters: int,
    minimum: float,
) -> tuple[float, int, float]:
    """Apply ordered pooling and count remaining independent category contrasts."""
    # A pooled row's expected total equals its integer person count only up to
    # rounding, so relax the threshold slightly; otherwise a one-person row at
    # min_expected=1 is "sparse" or not depending on summation order.
    minimum *= 1.0 - _SPARSE_RELATIVE_TOLERANCE
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
    sparse_rows = np.zeros(len(expected), dtype=bool)
    if n_categories > 2 and minimum > 0 and len(expected):
        sparse_rows = np.min(expected, axis=1) < minimum
    # Rows without sparse categories need no pooling and reduce together.
    statistic, contrasts, impossible, sparse_remaining = _chi_square_terms(
        observed[~sparse_rows], expected[~sparse_rows], minimum
    )
    # Sparse rows pool a few cells each, which plain Python handles fastest.
    for observed_row, expected_row in zip(
        observed[sparse_rows].tolist(), expected[sparse_rows].tolist(), strict=True
    ):
        positive = 0
        for observed_cell, expected_cell in zip(
            *_pool_categories(observed_row, expected_row, minimum), strict=True
        ):
            if expected_cell > 0:
                positive += 1
                statistic += (observed_cell - expected_cell) ** 2 / expected_cell
                sparse_remaining |= expected_cell < minimum
            elif observed_cell > 0:
                impossible = True
        contrasts += max(positive - 1, 0)
    if impossible:
        statistic = np.inf
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
    prior_mean: ArrayLike | None = None,
    prior_cov: ArrayLike | None = None,
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
        model, n_quadpts, quadrature_points, quadrature_weights, prior_mean, prior_cov
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
    model: BaseItemModel | FitResult,
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
    prior_mean: ArrayLike | None = None,
    prior_cov: ArrayLike | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Compute exact-total-score conditional Orlando-Thissen S-X2 item fit.

    Binary and ordinal expected counts are integrated over the latent ability
    distribution by score recursion. See :func:`compute_itemfit` for pooling,
    quadrature, latent-population, parameter-count, missing-response, and
    multiplicity controls.
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
        prior_mean=prior_mean,
        prior_cov=prior_cov,
    )
