"""Scalar-reference coverage for shared LD tables and bounded probability calls."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mirt._local_dependence import ld_pair_statistics
from mirt.constants import PROB_EPSILON


def _reference(responses, probabilities):
    n_items = responses.shape[1]
    chi2 = np.full((n_items, n_items), np.nan)
    g2 = np.full_like(chi2, np.nan)
    for first in range(n_items):
        for second in range(first + 1, n_items):
            valid = (responses[:, first] >= 0) & (responses[:, second] >= 0)
            if valid.sum() < 10:
                continue
            observed = np.zeros((2, 2))
            expected = np.zeros((2, 2))
            for row in np.flatnonzero(valid):
                left, right = probabilities[row, [first, second]]
                observed[
                    int(responses[row, first] > 0), int(responses[row, second] > 0)
                ] += 1
                expected += np.outer([1.0 - left, left], [1.0 - right, right])
            expected = np.maximum(expected, 0.5)
            chi2[first, second] = chi2[second, first] = np.sum(
                (observed - expected) ** 2 / expected
            )
            g2[first, second] = g2[second, first] = np.sum(
                2.0 * observed * np.log(observed / expected + PROB_EPSILON)
            )
    return chi2, g2


@pytest.mark.parametrize("chunk_size", [1, 7, 100])
@pytest.mark.parametrize("missing", ["none", "random", "mixed_blocks"])
@pytest.mark.parametrize("compute_g2", [False, True])
def test_matches_scalar_tables(chunk_size, missing, compute_g2):
    rng = np.random.default_rng(702)
    # Strided inputs, pooled polytomous responses, and probabilities at boundaries.
    responses = rng.integers(0, 4, size=(86, 12))[::2, ::2]
    probabilities = rng.uniform(size=(86, 12))[::2, ::2]
    probabilities[:, 0] = 0.0
    probabilities[:, 1] = 1.0
    if missing == "random":
        responses[rng.random(responses.shape) < 0.2] = -3
        responses[9:, -1] = -1
    elif missing == "mixed_blocks":
        responses[14:28, ::2] = -1
    probabilities[responses < 0] = np.nan
    original_responses = responses.copy()
    original_probabilities = probabilities.copy()
    responses.flags.writeable = False
    probabilities.flags.writeable = False

    expected_chi2, expected_g2 = _reference(responses, probabilities)
    chi2, g2 = ld_pair_statistics(
        responses, probabilities, compute_g2=compute_g2, chunk_size=chunk_size
    )

    assert_allclose(chi2, expected_chi2, rtol=1e-12, atol=1e-12)
    assert_allclose(chi2, chi2.T, rtol=0, atol=0)
    if compute_g2:
        assert_allclose(g2, expected_g2, rtol=1e-12, atol=1e-12)
    else:
        assert g2 is None
    assert_array_equal(responses, original_responses)
    assert_array_equal(probabilities, original_probabilities)


def test_probability_callback_uses_bounded_slices_and_accumulates_before_flooring():
    responses = np.zeros((23, 4), dtype=np.int32)
    responses[:, 0] = 1
    responses[9:, 2] = -1
    responses[10:, 3] = -1
    probabilities = np.full(responses.shape, 0.2)
    seen = []

    def probability_block(rows):
        seen.append(rows)
        return probabilities[rows]

    expected_chi2, expected_g2 = _reference(responses, probabilities)
    chi2, g2 = ld_pair_statistics(responses, probability_block, chunk_size=4)

    assert seen == [slice(start, min(start + 4, 23)) for start in range(0, 23, 4)]
    assert_allclose(chi2, expected_chi2, rtol=1e-12, atol=1e-12)
    assert_allclose(g2, expected_g2, rtol=1e-12, atol=1e-12)
    assert np.isnan(chi2[0, 2])  # Nine shared responses are insufficient.
    assert np.isfinite(chi2[0, 3])  # Ten shared responses are eligible.


def test_small_expected_cells_avoid_subtractive_cancellation():
    n_persons = 20_000
    responses = np.ones((n_persons, 2), dtype=np.int32)
    responses[::1000, 0] = 0
    p = 1.0 - 1e-4
    probabilities = np.full(responses.shape, p)
    observed = np.array([0, 20, 0, n_persons - 20])
    expected = np.maximum(
        n_persons * np.array([(1 - p) ** 2, (1 - p) * p, p * (1 - p), p**2]),
        0.5,
    )
    chi2, g2 = ld_pair_statistics(responses, probabilities, chunk_size=137)

    assert_allclose(
        chi2[0, 1], np.sum((observed - expected) ** 2 / expected), rtol=1e-12
    )
    assert_allclose(
        g2[0, 1],
        np.sum(2.0 * observed * np.log(observed / expected + PROB_EPSILON)),
        # The near-unit ratio in the large 11 cell amplifies summation rounding.
        atol=2e-10,
        rtol=1e-12,
    )


@pytest.mark.parametrize("shape", [(0, 3), (9, 3), (20, 0), (20, 1), (20, 3)])
def test_empty_or_ineligible_pairs_remain_nan(shape):
    responses = np.full(shape, -1)
    chi2, g2 = ld_pair_statistics(responses, np.full(shape, np.nan), chunk_size=3)
    assert chi2.shape == g2.shape == (shape[1], shape[1])
    assert np.all(np.isnan(chi2))
    assert np.all(np.isnan(g2))
