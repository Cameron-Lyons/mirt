"""Tests for reusable multiple-testing adjustments."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mirt.diagnostics.multiple_testing import adjust_p_values


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("none", [0.01, 0.04, 0.03, 0.002, np.nan]),
        ("bonferroni", [0.04, 0.16, 0.12, 0.008, np.nan]),
        ("holm", [0.03, 0.06, 0.06, 0.008, np.nan]),
        ("fdr_bh", [0.02, 0.04, 0.04, 0.008, np.nan]),
    ],
)
def test_adjustments_match_known_values(method, expected) -> None:
    raw = np.array([0.01, 0.04, 0.03, 0.002, np.nan])

    actual = adjust_p_values(raw, method)

    assert_allclose(actual, expected, equal_nan=True)
    assert_array_equal(raw, [0.01, 0.04, 0.03, 0.002, np.nan])


def test_axis_adjusts_each_family_independently() -> None:
    raw = np.array(
        [
            [0.01, 0.20, np.nan],
            [0.04, 0.10, 0.50],
            [0.03, np.nan, 0.25],
        ]
    )

    actual = adjust_p_values(raw, "holm", axis=0)

    assert_allclose(
        actual,
        [
            [0.03, 0.20, np.nan],
            [0.06, 0.20, 0.50],
            [0.06, np.nan, 0.50],
        ],
        equal_nan=True,
    )


def test_none_axis_flattens_the_full_array() -> None:
    raw = np.array([[0.01, 0.20], [0.03, np.nan]])

    flattened = adjust_p_values(raw, "bonferroni")
    rowwise = adjust_p_values(raw, "bonferroni", axis=1)

    assert_allclose(flattened, [[0.03, 0.60], [0.09, np.nan]], equal_nan=True)
    assert_allclose(rowwise, [[0.02, 0.40], [0.03, np.nan]], equal_nan=True)


def test_none_preserves_scalar_and_empty_shapes() -> None:
    scalar = adjust_p_values(0.25, "none")
    empty = adjust_p_values(np.empty((2, 0)), "fdr_bh", axis=1)

    assert scalar.shape == ()
    assert scalar == pytest.approx(0.25)
    assert empty.shape == (2, 0)


def test_all_missing_family_remains_missing() -> None:
    raw = np.full((3, 4), np.nan)

    actual = adjust_p_values(raw, "holm", axis=-1)

    assert np.all(np.isnan(actual))


@pytest.mark.parametrize("method", ["", "BH", "unknown", None])
def test_rejects_unknown_method(method) -> None:
    with pytest.raises(ValueError, match="method"):
        adjust_p_values([0.1, 0.2], method)


@pytest.mark.parametrize(
    "p_values",
    [
        [-0.1, 0.2],
        [0.1, 1.1],
        [0.1, np.inf],
        [0.1 + 0.2j],
        ["invalid"],
    ],
)
def test_rejects_invalid_probabilities(p_values) -> None:
    with pytest.raises(ValueError, match="p_values"):
        adjust_p_values(p_values)


@pytest.mark.parametrize("axis", [True, 1.5, "0"])
def test_rejects_non_integer_axis(axis) -> None:
    with pytest.raises(TypeError, match="axis"):
        adjust_p_values([[0.1, 0.2]], axis=axis)


@pytest.mark.parametrize("axis", [-3, 2])
def test_rejects_out_of_bounds_axis(axis) -> None:
    with pytest.raises(ValueError, match="axis"):
        adjust_p_values([[0.1, 0.2]], axis=axis)
