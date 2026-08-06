"""Regression coverage for response transformation utilities."""

import numpy as np
import pytest

from mirt.utils.transform import (
    collapse_table,
    expand_table,
    key2binary,
    likert2int,
    recode_responses,
)


def test_key2binary_scores_matrix_with_custom_values():
    responses = np.array([["A", "B", "C"], ["B", "B", "A"]])

    result = key2binary(responses, ["A", "B", "C"], scored_values=(-1, 2))

    np.testing.assert_array_equal(result, [[2.0, 2.0, 2.0], [-1.0, 2.0, -1.0]])


@pytest.mark.parametrize(
    ("responses", "key", "message"),
    [
        (np.array(["A", "B"]), ["A", "B"], "responses must be a 2D"),
        (np.array([["A", "B"]]), [["A", "B"]], "key must be a 1D"),
        (np.array([["A", "B"]]), ["A"], "does not match key length"),
    ],
)
def test_key2binary_rejects_invalid_shapes(responses, key, message):
    with pytest.raises(ValueError, match=message):
        key2binary(responses, key)


def test_expand_table_repeats_patterns_from_any_frequency_column():
    table = np.array([[2, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=float)

    result = expand_table(table, freq_col=0)

    np.testing.assert_array_equal(result, [[0, 1], [0, 1], [1, 0]])


@pytest.mark.parametrize(
    ("frequencies", "message"),
    [
        ([1.5, 2.0], "non-negative integers"),
        ([-1.0, 2.0], "non-negative integers"),
        ([np.nan, 2.0], "finite values"),
        ([np.inf, 2.0], "finite values"),
    ],
)
def test_expand_table_rejects_invalid_frequencies(frequencies, message):
    table = np.column_stack(([0.0, 1.0], frequencies))

    with pytest.raises(ValueError, match=message):
        expand_table(table)


def test_expand_and_collapse_table_round_trip():
    table = np.array([[0, 0, 2], [1, 0, 1], [1, 1, 3]], dtype=float)

    expanded = expand_table(table)
    patterns, frequencies = collapse_table(expanded)

    np.testing.assert_array_equal(patterns, table[:, :2])
    np.testing.assert_array_equal(frequencies, table[:, 2])


def test_collapse_table_groups_matching_missing_patterns():
    responses = np.array([[1.0, np.nan], [1.0, np.nan], [0.0, 1.0]])

    patterns, frequencies = collapse_table(responses)

    np.testing.assert_array_equal(patterns, [[0.0, 1.0], [1.0, np.nan]])
    np.testing.assert_array_equal(frequencies, [1, 2])


def test_recode_responses_applies_mapping_simultaneously():
    responses = np.array([[1, 2, 3], [2, 1, 1]], dtype=float)

    result = recode_responses(responses, {1: 2, 2: 3})

    np.testing.assert_array_equal(result, [[2, 3, 3], [3, 2, 2]])


def test_recode_responses_limits_changes_to_selected_items():
    responses = np.array([[1, 1, 1], [2, 2, 2]], dtype=float)

    result = recode_responses(responses, {1: 9, 2: 8}, items=[-1])

    np.testing.assert_array_equal(result, [[1, 1, 9], [2, 2, 8]])


def test_recode_responses_accepts_an_empty_item_selection():
    responses = np.array([[1, 2], [2, 1]], dtype=float)

    result = recode_responses(responses, {1: 9}, items=[])

    np.testing.assert_array_equal(result, responses)


def test_likert2int_preserves_order_and_marks_unknown_labels_missing():
    responses = np.array([["Agree", "Unknown"], ["Disagree", "Agree"]])

    result = likert2int(responses, labels=["Disagree", "Agree"])

    np.testing.assert_array_equal(result, [[1.0, np.nan], [0.0, 1.0]])


def test_likert2int_infers_sorted_label_order():
    responses = np.array([["Neutral", "Agree"], ["Disagree", "Neutral"]])

    result = likert2int(responses)

    np.testing.assert_array_equal(result, [[2.0, 0.0], [1.0, 2.0]])
