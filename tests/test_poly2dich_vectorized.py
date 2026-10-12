"""Regression coverage for vectorized polytomous transformations."""

import numpy as np
import pytest

from mirt import poly2dich


@pytest.mark.parametrize("cutoff", [1.5, np.float64(1.5), np.int64(2), np.array(1.5)])
def test_poly2dich_accepts_scalar_cutoffs(cutoff):
    responses = np.array([[0.0, 1.0], [2.0, 3.0]])

    result = poly2dich(responses, cutoff=cutoff)

    np.testing.assert_array_equal(result, [[0.0, 0.0], [1.0, 1.0]])


def test_poly2dich_applies_item_cutoffs_and_preserves_missing_values():
    responses = np.array([[0.0, np.nan, 2.0], [2.0, 3.0, 1.0]])

    result = poly2dich(responses, cutoff=[1.0, 2.0, 2.0])

    np.testing.assert_array_equal(
        result,
        [[0.0, np.nan, 1.0], [1.0, 1.0, 0.0]],
    )


def test_poly2dich_median_split_is_vectorized_by_item():
    responses = np.array([[0.0, 4.0], [1.0, np.nan], [2.0, 2.0]])

    result = poly2dich(responses, method="median")

    np.testing.assert_array_equal(
        result,
        [[0.0, 1.0], [1.0, np.nan], [1.0, 0.0]],
    )


def test_poly2dich_cumulative_expansion_preserves_item_major_order():
    responses = np.array([[0.0, 2.0], [2.0, np.nan]])

    result = poly2dich(responses, method="cumulative")

    np.testing.assert_array_equal(
        result,
        [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, np.nan, np.nan]],
    )


def test_poly2dich_adjacent_contrasts_only_neighbouring_categories():
    responses = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, -1.0]])

    result = poly2dich(responses, method="adjacent")

    # Split k is 1 for category k, 0 for category k - 1, missing otherwise.
    np.testing.assert_array_equal(
        result,
        [
            [0.0, np.nan, np.nan, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [np.nan, 1.0, np.nan, np.nan],
        ],
    )


def test_poly2dich_dummy_matches_r_mirt():
    responses = np.array([[0, 2, np.nan], [1, 1, 3], [2, np.nan, 0], [0, 2, 1]])

    result = poly2dich(responses, method="dummy")

    # R mirt 1.48 poly2dich(): A_cat.0-2, B_cat.1-2 and C_cat.0, 1, 3.
    np.testing.assert_array_equal(
        result,
        [
            [1, 0, 0, 0, 1, np.nan, np.nan, np.nan],
            [0, 1, 0, 1, 0, 0, 0, 1],
            [0, 0, 1, np.nan, np.nan, 1, 0, 0],
            [1, 0, 0, 0, 1, 0, 1, 0],
        ],
    )


@pytest.mark.parametrize("method", ["threshold", "median"])
def test_poly2dich_keeps_negative_missing_codes_missing(method):
    responses = np.array([[-1.0, 0.0], [-1.0, 2.0], [1.0, -9.0], [2.0, 1.0]])

    result = poly2dich(
        responses, cutoff=0 if method == "threshold" else None, method=method
    )

    np.testing.assert_array_equal(np.isnan(result), responses < 0)
    if method == "median":
        # The medians of the observed responses are 1.5 and 1.
        np.testing.assert_array_equal(result[2:, 0], [0.0, 1.0])
        np.testing.assert_array_equal(result[[0, 1, 3], 1], [0.0, 1.0, 1.0])


@pytest.mark.parametrize(
    ("cutoff", "message"),
    [
        ([1.0], "one value per item; expected 2"),
        ([[1.0, 2.0]], "one value per item; expected 2"),
        ([1.0, np.nan], "finite values"),
        ("high", "numeric values"),
    ],
)
def test_poly2dich_rejects_invalid_cutoffs(cutoff, message):
    responses = np.array([[0.0, 1.0], [2.0, 3.0]])

    with pytest.raises(ValueError, match=message):
        poly2dich(responses, cutoff=cutoff)


@pytest.mark.parametrize(
    "responses", [np.array([0.0, 1.0]), np.empty((0, 2)), np.empty((2, 0))]
)
def test_poly2dich_rejects_invalid_response_shapes(responses):
    with pytest.raises(ValueError, match="2D|at least one"):
        poly2dich(responses)
