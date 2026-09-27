"""Shared, bounded-memory binary local-dependence statistics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON

_LD_CHUNK_ELEMENTS = 1_000_000


def ld_pair_statistics(
    responses: NDArray[np.int_],
    positive_probabilities: NDArray[np.float64]
    | Callable[[slice], NDArray[np.float64]],
    *,
    compute_g2: bool = True,
    chunk_size: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Compare observed and expected binary cross-classification frequencies.

    Nonnegative responses are observed; positive categories are pooled. A
    probability callback receives row slices so backends can evaluate bounded
    blocks. Inputs are already validated by callers and are never modified.

    Only the 00, 01, and 11 cells require products: 10 is the transpose of 01.
    Complete blocks derive observed cells from integer margins and one product.
    Expected cells use direct products of probabilities and their complements
    to avoid cancellation near zero and one. The 0.5 expected-count floor is
    applied after all blocks have been accumulated.
    """
    n_persons, n_items = responses.shape
    chi2 = np.full((n_items, n_items), np.nan)
    g2 = np.full_like(chi2, np.nan) if compute_g2 else None
    if n_persons < 10 or n_items < 2:
        return chi2, g2
    if chunk_size is None:
        chunk_size = max(1, _LD_CHUNK_ELEMENTS // n_items)

    observed = np.zeros((3, n_items, n_items), dtype=np.float64)
    expected = np.zeros_like(observed)
    for start in range(0, n_persons, chunk_size):
        rows = slice(start, min(start + chunk_size, n_persons))
        response_block = responses[rows]
        valid = response_block >= 0
        positive = (response_block > 0).astype(np.float64)
        joint = positive.T @ positive
        observed[2] += joint
        if np.all(valid):
            margins = positive.sum(axis=0)
            observed[0] += (
                len(response_block) - margins[:, None] - margins[None, :] + joint
            )
            observed[1] += margins[None, :] - joint
        else:
            zero = valid - positive
            observed[0] += zero.T @ zero
            observed[1] += zero.T @ positive

        probabilities = (
            positive_probabilities(rows)
            if callable(positive_probabilities)
            else positive_probabilities[rows]
        )
        positive = np.where(valid, probabilities, 0.0)
        zero = valid - positive
        expected[0] += zero.T @ zero
        expected[1] += zero.T @ positive
        expected[2] += positive.T @ positive

    first, second = np.triu_indices(n_items, k=1)
    pair_counts = (
        observed[0, first, second]
        + observed[1, first, second]
        + observed[1, second, first]
        + observed[2, first, second]
    )
    eligible = pair_counts >= 10
    first, second = first[eligible], second[eligible]
    chi2_values = np.zeros(first.size)
    g2_values = np.zeros(first.size) if compute_g2 else None
    for cell, left, right in (
        (0, first, second),
        (1, first, second),
        (1, second, first),
        (2, first, second),
    ):
        observed_counts = observed[cell, left, right]
        expected_counts = np.maximum(expected[cell, left, right], 0.5)
        chi2_values += (observed_counts - expected_counts) ** 2 / expected_counts
        if g2_values is not None:
            g2_values += (
                2.0
                * observed_counts
                * np.log(observed_counts / expected_counts + PROB_EPSILON)
            )

    chi2[first, second] = chi2[second, first] = chi2_values
    if g2 is not None:
        g2[first, second] = g2[second, first] = g2_values
    return chi2, g2
