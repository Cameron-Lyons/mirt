"""Longitudinal drift oracles from known physical items and affine metrics."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.equating.chain import (
    ChainLinkingResult,
    chain_link,
    detect_longitudinal_drift,
)
from mirt.models.dichotomous import TwoParameterLogistic


def _model(discrimination, difficulty):
    model = TwoParameterLogistic(len(difficulty))
    model.set_parameters(
        discrimination=np.asarray(discrimination, dtype=np.float64),
        difficulty=np.asarray(difficulty, dtype=np.float64),
    )
    return model


@pytest.mark.parametrize("reference_index", [0, 1, 2])
def test_permuted_physical_items_have_known_signed_changes(reference_index):
    """Zero-mean changes leave the mean/mean metric exactly identifiable.

    Item 0 gets harder, item 1 gets easier, and items 2 and 3 reverse direction.
    Eight unchanged items make the discrepancy median and MAD zero, so each
    changed item has an infinite robust z regardless of its change's sign.
    These expectations follow directly from the supplied physical parameters,
    without using a linking or diagnostic helper to construct the oracle.
    """
    n_items = 12
    physical_a = np.linspace(0.7, 1.8, n_items)
    original_b = np.linspace(-1.8, 1.8, n_items)
    increments = np.zeros((2, n_items))
    increments[0, :4] = [0.8, -0.8, 0.6, -0.6]
    increments[1, :4] = [0.8, -0.8, -0.6, 0.6]
    physical_b = [
        original_b,
        original_b + increments[0],
        original_b + increments.sum(axis=0),
    ]
    metric_a = np.array([1.0, 1.6, 0.75])
    metric_b = np.array([0.0, -0.45, 0.7])
    physical_order = [
        np.arange(n_items),
        np.array([3, 7, 0, 10, 2, 8, 5, 1, 11, 4, 9, 6]),
        np.array([8, 0, 7, 4, 3, 10, 6, 9, 1, 11, 5, 2]),
    ]
    positions = [np.argsort(order) for order in physical_order]
    models = [
        _model(
            (physical_a * metric_a[t])[order],
            ((physical_b[t] - metric_b[t]) / metric_a[t])[order],
        )
        for t, order in enumerate(physical_order)
    ]
    anchor_order = [
        np.array([4, 0, 8, 1, 5, 11, 2, 9, 6, 3, 10, 7]),
        np.array([3, 8, 11, 5, 1, 9, 2, 4, 6, 0, 7, 10]),
    ]
    pairs = [
        (positions[t][order].tolist(), positions[t + 1][order].tolist())
        for t, order in enumerate(anchor_order)
    ]

    result = chain_link(
        models, pairs, method="mean_mean", reference_index=reference_index
    )

    assert_allclose(result.cumulative_A, metric_a / metric_a[reference_index])
    assert_allclose(
        result.cumulative_B,
        (metric_b - metric_b[reference_index]) / metric_a[reference_index],
        atol=1e-14,
    )
    assert result.drift_item_ids == [(0, j) for j in range(n_items)]
    expected_z = np.zeros((2, n_items))
    expected_z[:, :4] = np.inf
    assert_allclose(result.drift_accumulation, expected_z)
    assert_allclose(
        result.drift_difficulty_changes,
        increments / metric_a[reference_index],
        atol=1e-14,
    )
    detection = detect_longitudinal_drift(result)
    assert detection == {
        "consistently_flagged": [0, 1, 2, 3],
        "flagged_item_ids": [(0, 0), (0, 1), (0, 2), (0, 3)],
        "drift_direction": ["increasing", "decreasing", "variable", "variable"],
    }

    # Pair-list order affects neither physical columns nor classifications.
    reversed_pairs = [(left[::-1], right[::-1]) for left, right in pairs]
    reordered = chain_link(
        models, reversed_pairs, method="mean_mean", reference_index=reference_index
    )
    assert reordered.drift_item_ids == result.drift_item_ids
    assert_allclose(reordered.drift_accumulation, result.drift_accumulation)
    assert_allclose(
        reordered.drift_difficulty_changes, result.drift_difficulty_changes, atol=1e-14
    )
    assert detect_longitudinal_drift(reordered) == detection


def test_disjoint_bridges_do_not_create_repeated_physical_drift():
    """Two different item families have changed items in the same list slots."""
    a = np.linspace(0.8, 1.4, 8)
    b = np.linspace(-1.5, 1.5, 8)
    delta = np.array([0.8, -0.8, 0, 0, 0, 0, 0, 0])
    first_positions = np.array([2, 7, 5, 0, 6, 3, 1, 4])
    second_positions = np.array([11, 9, 15, 8, 14, 12, 10, 13])
    middle_a = np.empty(16)
    middle_b = np.empty(16)
    middle_a[first_positions] = a
    middle_b[first_positions] = b + delta
    middle_a[second_positions] = a
    middle_b[second_positions] = b
    models = [
        _model(a, b),
        _model(middle_a, middle_b),
        _model(a, b + delta),
    ]
    pairs = [
        (list(range(8)), first_positions.tolist()),
        (second_positions.tolist(), list(range(8))),
    ]

    result = chain_link(models, pairs, method="mean_mean")

    # Identities are the actual earliest occurrences, not positions in pairs.
    identities = [(0, j) for j in range(8)] + [(1, j) for j in range(8, 16)]
    assert result.drift_item_ids == identities
    expected = np.full((2, 16), np.nan)
    expected_changes = expected.copy()
    expected[0, :8] = [np.inf, np.inf, 0, 0, 0, 0, 0, 0]
    expected_changes[0, :8] = delta
    for physical_item, middle_position in enumerate(second_positions):
        column = identities.index((1, int(middle_position)))
        expected[1, column] = np.inf if physical_item < 2 else 0
        expected_changes[1, column] = delta[physical_item]
    assert_allclose(result.drift_accumulation, expected)
    assert_allclose(result.drift_difficulty_changes, expected_changes, atol=1e-14)
    assert detect_longitudinal_drift(result) == {
        "consistently_flagged": [],
        "drift_direction": [],
        "flagged_item_ids": [],
    }


def test_legacy_results_do_not_infer_parameter_direction_from_unsigned_discrepancy():
    result = ChainLinkingResult(
        [1.0] * 3,
        [0.0] * 3,
        [],
        np.array([[4.0, -4.0], [3.0, -3.0]]),
        0,
    )
    detection = detect_longitudinal_drift(result)
    assert detection["consistently_flagged"] == [0, 1]
    assert detection["drift_direction"] == ["unknown", "unknown"]
    assert detection["flagged_item_ids"] == [None, None]


@pytest.mark.parametrize(
    "threshold", [0, -1, np.nan, np.inf, -np.inf, True, "2.5", 2 + 1j]
)
def test_invalid_threshold_is_rejected_even_without_drift(threshold):
    result = ChainLinkingResult([1.0], [0.0], [], None, 0)
    with pytest.raises(ValueError, match="finite and positive"):
        detect_longitudinal_drift(result, threshold=threshold)


@pytest.mark.parametrize(
    ("matrix", "identities", "changes", "message"),
    [
        (np.array([3.0, 4.0]), None, None, "two-dimensional"),
        (np.ones((2, 1)), [], None, "columns"),
        (np.ones((2, 1)), None, np.ones((2, 2)), "must match"),
        (np.ones((2, 1)), None, np.array([[1.0], [np.inf]]), "finite values"),
    ],
)
def test_inconsistent_legacy_metadata_is_rejected(matrix, identities, changes, message):
    result = ChainLinkingResult(
        [1.0] * 3, [0.0] * 3, [], matrix, 0, identities, changes
    )
    with pytest.raises(ValueError, match=message):
        detect_longitudinal_drift(result)
