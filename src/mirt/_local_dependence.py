"""Shared, bounded-memory local-dependence statistics.

:func:`marginal_ld_statistics` and :func:`model_ld_statistics` implement the
Chen & Thissen (1997) LD statistics used by :mod:`mirt.diagnostics.ld` and
:func:`mirt.utils.residuals.LD_X2`.

References:
    Chen, W. H., & Thissen, D. (1997). Local dependence indexes for item pairs
        using item response theory. Journal of Educational and Behavioral
        Statistics, 22(3), 265-289.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.exceptions import MirtValidationError

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

_LD_CHUNK_ELEMENTS = 1_000_000
# Item pairs with fewer jointly observed responses have no LD statistic.
MIN_LD_PAIR_RESPONSES = 10


class MarginalLDStatistics(NamedTuple):
    """Chen-Thissen statistics for every item pair.

    Every matrix is symmetric. Statistics of the diagonal and of pairs with
    fewer than the minimum number of joint responses are ``NaN``.
    """

    chi2: NDArray[np.float64]
    g2: NDArray[np.float64] | None
    df: NDArray[np.float64]
    cramers_v: NDArray[np.float64]
    pair_counts: NDArray[np.intp]


def marginal_ld_statistics(
    responses: NDArray[np.int_],
    n_categories: NDArray[np.int_],
    node_probabilities: Callable[[slice], NDArray[np.float64]],
    weights: NDArray[np.float64],
    *,
    compute_g2: bool = True,
    min_pair_count: int = MIN_LD_PAIR_RESPONSES,
    chunk_size: int | None = None,
) -> MarginalLDStatistics:
    """Compare observed pair tables with the model-implied marginal tables.

    For items ``i`` and ``j`` the expected count of the categories ``(k, l)``
    is ``N_ij * sum_q w_q P_ik(theta_q) P_jl(theta_q)``. Here ``N_ij`` is the
    number of persons who answered both items and ``(theta_q, w_q)`` is a
    quadrature rule for the latent population. The expected table does not
    depend on ability estimates (Chen & Thissen, 1997). The statistics are
    Pearson's ``X2`` and the likelihood-ratio ``G2`` on all
    ``K_i x K_j`` cells, with ``(K_i - 1)(K_j - 1)`` degrees of freedom. The
    signed Cramer's V is ``sqrt(X2 / (N_ij (min(K_i, K_j) - 1)))``. Its sign
    is that of the observed minus the expected Goodman-Kruskal gamma, and a
    zero difference counts as positive. These are the conventions of
    ``residuals(type = "LD")`` in the R package mirt.

    Parameters
    ----------
    responses : ndarray of shape (n_persons, n_items)
        Integer category codes ``0, ..., K_i - 1``; negative codes are missing.
        Callers validate the codes.
    n_categories : ndarray of shape (n_items,)
        Category count ``K_i`` of every item.
    node_probabilities : callable
        Receives a slice of quadrature nodes and returns category
        probabilities of shape ``(n_nodes_in_slice, n_items, max(K))``. It is
        called on bounded node blocks.
    weights : ndarray of shape (n_nodes,)
        Quadrature weights summing to one.
    compute_g2 : bool, default=True
        Also return the likelihood-ratio statistics.
    min_pair_count : int
        Smallest number of joint responses for which a pair has statistics.
    chunk_size : int, optional
        Rows per person and node block.

    Returns
    -------
    MarginalLDStatistics
        Pair statistics, degrees of freedom and joint response counts.
    """
    n_persons, n_items = responses.shape
    counts = np.asarray(n_categories, dtype=np.intp).reshape(n_items)
    width = int(np.max(counts, initial=2))
    flat = n_items * width
    if chunk_size is None:
        chunk_size = max(1, _LD_CHUNK_ELEMENTS // max(1, flat))

    category_values = np.arange(width)
    category_mask = (category_values[None, :] < counts[:, None]).reshape(flat)

    observed_tables = np.zeros((flat, flat), dtype=np.float64)
    pair_counts = np.zeros((n_items, n_items), dtype=np.float64)
    for start in range(0, n_persons, chunk_size):
        block = responses[start : start + chunk_size]
        indicators = (block[:, :, None] == category_values).reshape(len(block), flat)
        indicators = indicators.astype(np.float64)
        observed_tables += indicators.T @ indicators
        valid = (block >= 0).astype(np.float64)
        pair_counts += valid.T @ valid

    # One weighted product of the category probabilities at the nodes gives
    # every pair's integrated table.
    integrated_tables = np.zeros((flat, flat), dtype=np.float64)
    node_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    for start in range(0, node_weights.size, chunk_size):
        rows = slice(start, min(start + chunk_size, node_weights.size))
        probabilities = np.asarray(node_probabilities(rows), dtype=np.float64)
        probabilities = probabilities.reshape(-1, flat) * category_mask
        integrated_tables += probabilities.T @ (
            node_weights[rows, None] * probabilities
        )

    chi2 = np.full((n_items, n_items), np.nan)
    g2 = np.full_like(chi2, np.nan) if compute_g2 else None
    cramers_v = np.full_like(chi2, np.nan)
    degrees = np.multiply.outer(counts - 1, counts - 1).astype(np.float64)
    np.fill_diagonal(degrees, np.nan)

    first, second = np.triu_indices(n_items, k=1)
    pair_n = pair_counts[first, second]
    eligible = pair_n >= min_pair_count
    first, second, pair_n = first[eligible], second[eligible], pair_n[eligible]
    if first.size:
        observed = observed_tables.reshape(n_items, width, n_items, width)[
            first, :, second, :
        ]
        expected = (
            pair_n[:, None, None]
            * integrated_tables.reshape(n_items, width, n_items, width)[
                first, :, second, :
            ]
        )
        cells = (category_values < counts[first, None])[:, :, None] & (
            category_values < counts[second, None]
        )[:, None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            pearson = np.where(
                expected > 0.0,
                (observed - expected) ** 2 / expected,
                np.where(observed > 0.0, np.inf, 0.0),
            )
            chi2_values = np.sum(np.where(cells, pearson, 0.0), axis=(1, 2))
            if g2 is not None:
                likelihood = np.where(
                    observed > 0.0, observed * np.log(observed / expected), 0.0
                )
                g2_values = 2.0 * np.sum(np.where(cells, likelihood, 0.0), axis=(1, 2))
                g2[first, second] = g2[second, first] = g2_values

        gamma_difference = _goodman_kruskal_gamma(observed) - _goodman_kruskal_gamma(
            expected
        )
        smaller_degrees = np.minimum(counts[first], counts[second]) - 1
        magnitude = np.sqrt(chi2_values / (pair_n * smaller_degrees))
        signed = np.where(gamma_difference < 0.0, -magnitude, magnitude)
        signed[np.isnan(gamma_difference)] = np.nan
        chi2[first, second] = chi2[second, first] = chi2_values
        cramers_v[first, second] = cramers_v[second, first] = signed

    return MarginalLDStatistics(
        chi2=chi2,
        g2=g2,
        df=degrees,
        cramers_v=cramers_v,
        pair_counts=pair_counts.astype(np.intp),
    )


def _goodman_kruskal_gamma(tables: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return Goodman-Kruskal gamma of each ``(..., K, L)`` table.

    Tables without concordant or discordant pairs have no gamma (``NaN``).
    """
    # Counts in cells strictly below and to the right, or to the left.
    at_or_after = np.flip(
        np.cumsum(np.cumsum(np.flip(tables, (-2, -1)), axis=-2), axis=-1), (-2, -1)
    )
    after = np.zeros_like(tables)
    after[..., :-1, :-1] = at_or_after[..., 1:, 1:]
    below_at_or_before = np.cumsum(
        np.flip(np.cumsum(np.flip(tables, -2), axis=-2), -2), axis=-1
    )
    before = np.zeros_like(tables)
    before[..., :-1, 1:] = below_at_or_before[..., 1:, :-1]

    concordant = np.sum(tables * after, axis=(-2, -1))
    discordant = np.sum(tables * before, axis=(-2, -1))
    total = concordant + discordant
    return np.divide(
        concordant - discordant,
        total,
        out=np.full(total.shape, np.nan),
        where=total > 0.0,
    )


def item_category_counts(model: BaseItemModel, n_items: int) -> NDArray[np.intp]:
    """Return every item's category count; models without counts are binary."""
    configured = getattr(model, "n_categories", None)
    if configured is None:
        return np.full(n_items, 2, dtype=np.intp)
    if np.isscalar(configured):
        return np.full(n_items, int(configured), dtype=np.intp)
    counts = np.asarray(configured, dtype=np.intp)
    if counts.shape != (n_items,) or np.any(counts < 2):
        raise MirtValidationError(
            "model.n_categories must give at least two categories for every item"
        )
    return counts


def category_probabilities(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    n_categories: NDArray[np.intp],
) -> NDArray[np.float64]:
    """Evaluate category probabilities as a ``(n, n_items, max(K))`` array.

    One all-item call is used when the model returns the batch shape; other
    models are evaluated item by item. Columns beyond an item's categories
    are zero.
    """
    n_rows = theta.shape[0]
    n_items = n_categories.size
    width = int(np.max(n_categories, initial=2))
    result = np.zeros((n_rows, n_items, width), dtype=np.float64)

    probabilities = np.asarray(model.probability(theta), dtype=np.float64)
    if probabilities.ndim == 3 and probabilities.shape[:2] == (n_rows, n_items):
        columns = min(width, probabilities.shape[2])
        result[:, :, :columns] = probabilities[:, :, :columns]
    elif probabilities.shape == (n_rows, n_items) and np.all(n_categories == 2):
        result[:, :, 0] = 1.0 - probabilities
        result[:, :, 1] = probabilities
    else:
        for item in range(n_items):
            values = np.asarray(model.probability(theta, item), dtype=np.float64)
            if values.ndim == 1:
                result[:, item, 0] = 1.0 - values
                result[:, item, 1] = values
            else:
                columns = min(width, values.shape[1])
                result[:, item, :columns] = values[:, :columns]

    if not np.all(np.isfinite(result)):
        raise MirtValidationError("model probabilities must contain only finite values")
    np.clip(result, 0.0, 1.0, out=result)
    result *= np.arange(width)[None, None, :] < n_categories[None, :, None]
    return result


def model_ld_statistics(
    model: BaseItemModel,
    responses: ArrayLike,
    *,
    n_quadpts: int,
    prior_mean: ArrayLike | None = None,
    prior_cov: ArrayLike | None = None,
    compute_g2: bool = True,
) -> MarginalLDStatistics:
    """Chen-Thissen statistics of a model over its normal latent population.

    Parameters
    ----------
    model : BaseItemModel
        Item model whose responses are conditionally independent given theta.
    responses : array-like of shape (n_persons, n_items)
        Category codes. Negative codes and ``NaN`` are missing.
    n_quadpts : int
        Gauss-Hermite points per latent dimension.
    prior_mean, prior_cov : array-like, optional
        Mean and covariance of the normal latent population. Defaults to the
        standard normal.
    compute_g2 : bool, default=True
        Also return the likelihood-ratio statistics.

    Returns
    -------
    MarginalLDStatistics
        Pair statistics, degrees of freedom and joint response counts.

    Raises
    ------
    MirtValidationError
        If the model's pair probabilities are not products of item
        probabilities given theta, or a response is not a category code.
    """
    from mirt.models.cdm_advanced import HigherOrderCDM
    from mirt.models.mixture import MixtureIRT
    from mirt.scoring._common import build_quadrature

    if isinstance(model, (MixtureIRT, HigherOrderCDM)):
        raise MirtValidationError(
            f"LD chi-square does not support {type(model).__name__}: its item "
            "probabilities given theta share a latent class or mastery pattern, "
            "so they do not identify the joint pair tables"
        )
    values = np.asarray(responses)
    if values.ndim != 2:
        raise MirtValidationError("responses must be a two-dimensional matrix")
    n_items = values.shape[1]
    n_categories = item_category_counts(model, n_items)
    codes = _category_codes(values, n_categories)

    nodes, weights = build_quadrature(
        n_quadpts=n_quadpts,
        n_factors=int(getattr(model, "n_factors", 1)),
        prior_mean=None if prior_mean is None else np.asarray(prior_mean),
        prior_cov=None if prior_cov is None else np.asarray(prior_cov),
    )
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)

    def node_probabilities(rows: slice) -> NDArray[np.float64]:
        return category_probabilities(model, nodes[rows], n_categories)

    return marginal_ld_statistics(
        codes, n_categories, node_probabilities, weights, compute_g2=compute_g2
    )


def _category_codes(
    values: NDArray[np.generic],
    n_categories: NDArray[np.intp],
) -> NDArray[np.intp]:
    """Validate category codes and code every missing response as ``-1``."""
    if values.dtype.kind not in "biuf":
        raise MirtValidationError("responses must contain numeric category codes")
    numeric = values.astype(np.float64, copy=False)
    if np.any(np.isinf(numeric)):
        raise MirtValidationError("responses must not contain infinite values")
    observed = np.isfinite(numeric) & (numeric >= 0)
    if np.any(observed & (numeric != np.floor(numeric))):
        raise MirtValidationError("observed responses must be integer category codes")
    if np.any(observed & (numeric >= n_categories[None, :])):
        raise MirtValidationError(
            "observed responses exceed an item's category range "
            f"(at most {int(np.max(n_categories)) - 1})"
        )
    return np.where(observed, numeric, -1).astype(np.intp)
