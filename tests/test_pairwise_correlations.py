"""Numerical and missing-data contracts shared by residual diagnostics."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mirt._correlation as correlation
from mirt._correlation import pairwise_correlations, q3_correlations
from mirt.backends.rust.diagnostics import _q3_from_residuals_numpy
from mirt.diagnostics.ld import _compute_q3
from mirt.models.testlet import _pairwise_complete_correlations
from mirt.utils.residuals import _compute_ld_matrix


def _reference(values, observed):
    n_items = values.shape[1]
    correlations = np.full((n_items, n_items), np.nan)
    counts = np.zeros((n_items, n_items), dtype=np.intp)
    for first in range(n_items):
        for second in range(n_items):
            valid = observed[:, first] & observed[:, second]
            valid &= np.isfinite(values[:, first]) & np.isfinite(values[:, second])
            counts[first, second] = valid.sum()
            if valid.sum() < 2:
                continue
            left = values[valid, first]
            right = values[valid, second]
            if np.ptp(left) > 0 and np.ptp(right) > 0:
                correlations[first, second] = np.corrcoef(left, right)[0, 1]
    return correlations, counts


@pytest.mark.parametrize("chunk_size", [1, 7, 1000])
@pytest.mark.parametrize("missing", [False, True])
def test_matches_pairwise_reference_without_mutating_inputs(chunk_size, missing):
    rng = np.random.default_rng(65)
    values = rng.normal(size=(58, 14))[::2, ::2]
    observed = np.ones(values.shape, dtype=bool)
    if missing:
        observed[rng.random(values.shape) < 0.2] = False
        observed[:9] = False  # Offsets must wait for the first observed block.
        observed[:20, 2] = False
        values[12, 0] = np.inf
        values[13, 1] = np.nan
        values[:, 4] = np.nan
    values[:, 5] = 0.3
    before_values, before_observed = values.copy(), observed.copy()
    expected, expected_counts = _reference(values, observed)

    actual, counts = pairwise_correlations(values, observed, chunk_size=chunk_size)

    assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
    assert_array_equal(counts, expected_counts)
    assert_array_equal(values, before_values)
    assert_array_equal(observed, before_observed)


@pytest.mark.parametrize("chunk_size", [1, 13, 1000])
@pytest.mark.parametrize("missing", [False, True])
def test_large_column_offsets_preserve_correlations(chunk_size, missing):
    rng = np.random.default_rng(66)
    values = rng.integers(-20, 20, size=(71, 6)).astype(float) / 8
    observed = rng.random(values.shape) > (0.2 if missing else 0.0)
    expected, expected_counts = _reference(values, observed)
    shifted = values + 2.0 ** np.arange(35, 41)

    actual, counts = pairwise_correlations(shifted, observed, chunk_size=chunk_size)

    assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
    assert_array_equal(counts, expected_counts)


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (4, 0), (1, 3)])
def test_empty_and_single_person_inputs(shape):
    correlations, counts = pairwise_correlations(np.zeros(shape))
    assert correlations.shape == (shape[1], shape[1])
    assert np.all(np.isnan(correlations))
    assert_array_equal(counts, np.full((shape[1], shape[1]), shape[0]))


def test_working_blocks_obey_element_budget(monkeypatch):
    rng = np.random.default_rng(67)
    values = rng.normal(size=(31, 4))
    finite = np.isfinite
    block_shapes = []

    def record_block(block):
        block_shapes.append(block.shape)
        return finite(block)

    monkeypatch.setattr(correlation, "_CORRELATION_CHUNK_ELEMENTS", 20)
    monkeypatch.setattr(correlation.np, "isfinite", record_block)
    pairwise_correlations(values)

    assert len(block_shapes) == 7
    assert all(rows * columns <= 20 for rows, columns in block_shapes)


def test_callers_preserve_diagonals_sparse_pairs_and_suppression():
    values = np.column_stack(
        (
            [0.0, 1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, np.nan, np.nan],
            [0.3] * 4,
            [np.nan] * 4,
        )
    )
    observed = np.isfinite(values)
    responses = np.where(observed, 1, -1)
    expected, counts = _reference(values, observed)
    testlet, testlet_counts = _pairwise_complete_correlations(values, observed)
    assert_allclose(testlet, expected, atol=1e-14)
    assert_array_equal(testlet_counts, counts)

    q3_expected = expected.copy()
    q3_expected[counts < 3] = 0.0
    np.fill_diagonal(q3_expected, 0.0)
    assert_allclose(q3_correlations(values, observed), q3_expected, atol=1e-14)
    assert_allclose(_compute_q3(values, responses), q3_expected, atol=1e-14)
    assert_allclose(
        _q3_from_residuals_numpy(responses, values), q3_expected, atol=1e-14
    )

    expected[counts < 3] = np.nan
    np.fill_diagonal(expected, 1.0)
    assert_allclose(_compute_ld_matrix(values), expected, atol=1e-14)
    expected[np.abs(expected) < 1.1] = 0.0
    np.fill_diagonal(expected, 1.0)
    assert_allclose(_compute_ld_matrix(values, suppress_abs=1.1), expected)


def test_denominator_threshold_preserves_undefined_tiny_variances():
    values = np.arange(12.0).reshape(6, 2) * 1e-10
    result = _compute_ld_matrix(values)
    assert np.isnan(result[0, 1])
    assert_array_equal(np.diag(result), np.ones(2))
