"""Scalar-reference coverage for the shared Chen-Thissen LD tables."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mirt._local_dependence import _goodman_kruskal_gamma, marginal_ld_statistics


def _marginal_reference(responses, n_categories, node_probabilities, weights):
    """Loop over pairs, persons and nodes to build each pair's two tables."""
    n_items = responses.shape[1]
    chi2 = np.full((n_items, n_items), np.nan)
    g2 = np.full_like(chi2, np.nan)
    for first in range(n_items):
        for second in range(first + 1, n_items):
            valid = (responses[:, first] >= 0) & (responses[:, second] >= 0)
            if valid.sum() < 10:
                continue
            observed = np.zeros((n_categories[first], n_categories[second]))
            for row in np.flatnonzero(valid):
                observed[responses[row, first], responses[row, second]] += 1
            expected = np.zeros_like(observed)
            for node, weight in enumerate(weights):
                expected += weight * np.outer(
                    node_probabilities[node, first, : n_categories[first]],
                    node_probabilities[node, second, : n_categories[second]],
                )
            expected *= valid.sum()
            chi2[first, second] = chi2[second, first] = np.sum(
                (observed - expected) ** 2 / expected
            )
            positive = observed > 0
            g2[first, second] = g2[second, first] = 2.0 * np.sum(
                observed[positive] * np.log(observed[positive] / expected[positive])
            )
    return chi2, g2


def _random_category_probabilities(rng, n_nodes, n_categories):
    width = max(n_categories)
    values = rng.uniform(0.1, 1.0, size=(n_nodes, len(n_categories), width))
    values[:, np.arange(width)[None, :] >= np.array(n_categories)[:, None]] = 0.0
    return values / values.sum(axis=2, keepdims=True)


@pytest.mark.parametrize("chunk_size", [1, 6, None])
def test_marginal_tables_match_scalar_reference(chunk_size):
    rng = np.random.default_rng(904)
    n_categories = [2, 4, 3, 5, 2]
    n_nodes = 13
    node_probabilities = _random_category_probabilities(rng, n_nodes, n_categories)
    weights = rng.uniform(size=n_nodes)
    weights /= weights.sum()
    responses = np.column_stack(
        [rng.integers(0, count, size=60) for count in n_categories]
    )
    responses[rng.random(responses.shape) < 0.2] = -1
    responses[12:, 4] = -2  # Item 4 shares at most 12 responses with any item.
    seen = []

    def probabilities(rows):
        seen.append(rows)
        return node_probabilities[rows]

    expected_chi2, expected_g2 = _marginal_reference(
        responses, n_categories, node_probabilities, weights
    )
    result = marginal_ld_statistics(
        responses,
        np.array(n_categories),
        probabilities,
        weights,
        chunk_size=chunk_size,
    )

    assert_allclose(result.chi2, expected_chi2, rtol=1e-12, equal_nan=True)
    assert_allclose(result.g2, expected_g2, rtol=1e-11, atol=1e-12, equal_nan=True)
    assert_array_equal(np.isnan(result.chi2), np.isnan(expected_chi2))
    assert np.all(np.isnan(np.diag(result.chi2)))
    assert_array_equal(
        result.pair_counts,
        (responses >= 0).astype(int).T @ (responses >= 0).astype(int),
    )
    counts = np.array(n_categories)
    degrees = np.multiply.outer(counts - 1, counts - 1).astype(float)
    np.fill_diagonal(degrees, np.nan)
    assert_allclose(result.df, degrees, equal_nan=True)
    if chunk_size is not None:
        assert all(rows.stop - rows.start <= chunk_size for rows in seen)


def test_pairs_below_the_minimum_count_are_undefined():
    responses = np.zeros((12, 3), dtype=int)
    responses[::2, :] = 1
    responses[9:, 2] = -1
    probabilities = np.full((5, 3, 2), 0.5)

    result = marginal_ld_statistics(
        responses, np.full(3, 2), lambda rows: probabilities[rows], np.full(5, 0.2)
    )

    assert np.isfinite(result.chi2[0, 1])
    assert np.isnan(result.chi2[0, 2]) and np.isnan(result.cramers_v[1, 2])
    assert result.pair_counts[0, 2] == 9


def test_gamma_matches_pair_counting_and_signs_cramers_v():
    rng = np.random.default_rng(55)
    tables = rng.integers(0, 9, size=(20, 3, 4)).astype(float)
    expected = []
    for table in tables:
        concordant = discordant = 0.0
        for k in range(3):
            for m in range(4):
                for other_k in range(k + 1, 3):
                    for other_m in range(4):
                        product = table[k, m] * table[other_k, other_m]
                        if other_m > m:
                            concordant += product
                        elif other_m < m:
                            discordant += product
        expected.append((concordant - discordant) / (concordant + discordant))
    assert_allclose(_goodman_kruskal_gamma(tables), expected, rtol=1e-13)
    assert np.isnan(_goodman_kruskal_gamma(np.array([[3.0, 4.0], [0.0, 0.0]])))

    # Positive dependence: agreeing responses are more frequent than expected.
    responses = np.repeat([[0, 0], [1, 1], [0, 1], [1, 0]], [40, 40, 10, 10], axis=0)
    probabilities = np.full((1, 2, 2), 0.5)
    positive = marginal_ld_statistics(
        responses, np.full(2, 2), lambda rows: probabilities[rows], np.ones(1)
    )
    negative = marginal_ld_statistics(
        responses[:, ::-1] * [1, -1] + [0, 1],
        np.full(2, 2),
        lambda rows: probabilities[rows],
        np.ones(1),
    )
    assert positive.cramers_v[0, 1] == pytest.approx(0.6)
    assert negative.cramers_v[0, 1] == pytest.approx(-0.6)
