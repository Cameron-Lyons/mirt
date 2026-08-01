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


def test_poly2dich_adjacent_expansion_preserves_item_major_order():
    responses = np.array([[0.0, 2.0], [2.0, np.nan]])

    result = poly2dich(responses, method="adjacent")

    np.testing.assert_array_equal(
        result,
        [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, np.nan, np.nan]],
    )


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
