"""Tests for scalable missing-response pattern analysis."""

from __future__ import annotations

import numpy as np
import pytest

import mirt
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.utils import MissingPatternResult, missing_patterns


def test_patterns_are_frequency_ranked_and_map_to_input_rows() -> None:
    responses = np.array(
        [
            [1, -1, 0, 1],
            [-1, 1, 1, 0],
            [0, -1, 1, 1],
            [-1, 0, 0, 1],
            [1, 1, -1, 0],
            [1, -1, 1, 0],
        ]
    )

    result = missing_patterns(responses)

    assert isinstance(result, MissingPatternResult)
    np.testing.assert_array_equal(result.frequencies, [3, 2, 1])
    np.testing.assert_allclose(result.proportions, [0.5, 1 / 3, 1 / 6])
    np.testing.assert_array_equal(result.indices, [0, 1, 0, 1, 2, 0])
    np.testing.assert_array_equal(
        result.patterns[result.indices],
        responses < 0,
    )
    np.testing.assert_array_equal(result.missing_counts, [1, 1, 1])
    assert result.n_persons == 6
    assert result.n_items == 4
    assert result.n_patterns == 3
    assert result.compression_ratio == 0.5


def test_equal_frequency_patterns_retain_first_appearance_order() -> None:
    responses = np.array(
        [
            [1, -1, 0],
            [-1, 1, 0],
            [1, 0, -1],
            [-1, 0, 1],
            [0, -1, 1],
            [0, 1, -1],
        ]
    )

    result = missing_patterns(responses)

    np.testing.assert_array_equal(
        result.patterns,
        [
            [False, True, False],
            [True, False, False],
            [False, False, True],
        ],
    )
    np.testing.assert_array_equal(result.frequencies, [2, 2, 2])


def test_packed_patterns_preserve_items_beyond_byte_boundaries() -> None:
    responses = np.zeros((5, 73), dtype=int)
    responses[0:2, 8] = -1
    responses[2:4, 72] = -1

    result = missing_patterns(responses)

    np.testing.assert_array_equal(result.patterns[result.indices], responses < 0)
    assert result.patterns.shape == (3, 73)
    assert result.patterns[:, 72].sum() == 1


def test_custom_code_and_negative_values_share_a_missing_pattern() -> None:
    responses = np.array([[1, 99, 0], [0, -7, 1], [1, 1, 0]])
    original = responses.copy()

    result = missing_patterns(responses, missing_code=99)

    np.testing.assert_array_equal(result.frequencies, [2, 1])
    np.testing.assert_array_equal(
        result.patterns,
        [[False, True, False], [False, False, False]],
    )
    assert result.complete_case_count == 1
    assert result.complete_case_rate == pytest.approx(1 / 3)
    np.testing.assert_array_equal(responses, original)


def test_expand_restores_pattern_values_to_respondent_order() -> None:
    result = missing_patterns([[1, -1], [0, 1], [1, -1]])
    values = np.array([[10.0, 11.0], [20.0, 21.0]])

    np.testing.assert_array_equal(
        result.expand(values),
        [[10.0, 11.0], [20.0, 21.0], [10.0, 11.0]],
    )

    with pytest.raises(MirtValidationError, match="leading entry"):
        result.expand(np.ones(3))


@pytest.mark.parametrize(
    "item_names",
    [["only_one"], ["same", "same"], ["", "second"], "items"],
)
def test_dataframe_rejects_invalid_item_names(item_names) -> None:
    result = missing_patterns([[1, -1], [0, 1]])

    with pytest.raises(MirtValidationError, match="item_names"):
        result.to_dataframe(item_names)


def test_dataframe_contains_pattern_metadata() -> None:
    result = missing_patterns([[1, -1], [0, 1], [1, -1]])

    table = result.to_dataframe(["score", "time"])
    columns = table.columns if hasattr(table, "columns") else table.schema.names()

    assert set(columns) == {
        "pattern",
        "score_missing",
        "time_missing",
        "n_missing",
        "frequency",
        "proportion",
    }


@pytest.mark.parametrize("responses", [np.ones(3), np.empty((0, 2)), np.empty((2, 0))])
def test_invalid_response_shapes_are_rejected(responses) -> None:
    with pytest.raises(MirtDataError):
        missing_patterns(responses)


def test_public_namespaces_share_the_same_symbols() -> None:
    assert mirt.missing_patterns is missing_patterns
    assert mirt.MissingPatternResult is MissingPatternResult
    assert mirt.utils.missing_patterns is missing_patterns
    assert mirt.utils.MissingPatternResult is MissingPatternResult
