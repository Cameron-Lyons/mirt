import numpy as np
import pytest

import mirt
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.utils import (
    collapse_patterns,
    collapse_with_groups,
    compute_pattern_likelihood,
    weighted_sum_from_collapsed,
)


def test_patterns_retain_first_appearance_order():
    responses = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1], [1, 0, 1]], dtype=np.int8)

    collapsed = collapse_patterns(responses)

    np.testing.assert_array_equal(collapsed.patterns, np.array([[1, 0, 1], [0, 1, 0]]))
    np.testing.assert_array_equal(collapsed.frequencies, [3, 1])
    np.testing.assert_array_equal(collapsed.indices, [0, 1, 0, 0])
    assert collapsed.n_persons == 4
    assert collapsed.n_patterns == 2
    assert collapsed.compression_ratio == 0.5
    assert collapsed.observations_saved == 2


def test_indices_reconstruct_normalized_responses():
    responses = np.array([[2, -1, 0], [1, 2, 0], [2, -8, 0], [1, 2, 0]], dtype=np.int16)

    collapsed = collapse_patterns(responses, missing_code=-7)

    expected = responses.astype(np.int_)
    expected[expected < 0] = -7
    np.testing.assert_array_equal(collapsed.patterns[collapsed.indices], expected)
    assert collapsed.patterns.dtype == np.dtype(np.int_)


def test_nan_and_negative_missing_codes_collapse_together():
    responses = np.array([[1.0, np.nan, 2.0], [1.0, -9.0, 2.0], [0.0, -1.0, 1.0]])
    original = responses.copy()

    collapsed = collapse_patterns(responses, missing_code=-7)

    np.testing.assert_array_equal(
        collapsed.patterns, np.array([[1, -7, 2], [0, -7, 1]])
    )
    np.testing.assert_array_equal(collapsed.frequencies, [2, 1])
    np.testing.assert_array_equal(responses, original)


def test_noncontiguous_response_view_is_supported():
    base = np.array([[1, 9, 0, 9, 2], [0, 9, 1, 9, 2], [1, 9, 0, 9, 2]])
    responses = base[:, ::2]
    assert not responses.flags.c_contiguous

    collapsed = collapse_patterns(responses)

    np.testing.assert_array_equal(collapsed.patterns[collapsed.indices], responses)
    np.testing.assert_array_equal(collapsed.frequencies, [2, 1])


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        (np.ones(3), "2D"),
        (np.empty((0, 3)), "at least one person"),
        (np.empty((2, 0)), "at least one item"),
        (np.array([["yes", "no"]]), "numeric"),
        (np.array([[0.0, 1.5]]), "integer-valued"),
        (np.array([[0.0, np.inf]]), "finite"),
    ],
)
def test_invalid_response_data_raises(responses, message):
    with pytest.raises(MirtDataError, match=message):
        collapse_patterns(responses)


@pytest.mark.parametrize("missing_code", [0, 1, -1.5, True, -(2**100)])
def test_missing_code_must_be_a_negative_integer(missing_code):
    with pytest.raises(MirtValidationError, match="negative integer"):
        collapse_patterns([[0, 1]], missing_code=missing_code)


def test_expand_weights_and_scores_restore_person_order():
    collapsed = collapse_patterns([[1, 0], [0, 1], [1, 0]])
    weights = np.array([[0.2, 0.8], [0.7, 0.3]])
    scores = np.array([-0.4, 0.6])

    np.testing.assert_array_equal(collapsed.expand_weights(weights), weights[[0, 1, 0]])
    np.testing.assert_array_equal(collapsed.expand_scores(scores), scores[[0, 1, 0]])


@pytest.mark.parametrize("values", [1.0, np.ones(3), np.ones((3, 2))])
def test_expand_rejects_wrong_pattern_dimension(values):
    collapsed = collapse_patterns([[1, 0], [0, 1], [1, 0]])

    with pytest.raises(MirtValidationError, match="one leading entry"):
        collapsed.expand_scores(values)


def test_group_collapsing_retains_group_and_pattern_appearance_order():
    responses = np.array([[1, 0], [0, 1], [1, 0], [1, 1], [0, 1], [1, 1]])
    groups = np.array(["later", "first", "later", "first", "first", "later"])

    collapsed_groups, masks = collapse_with_groups(responses, groups)

    np.testing.assert_array_equal(masks[0], groups == "later")
    np.testing.assert_array_equal(masks[1], groups == "first")
    np.testing.assert_array_equal(
        collapsed_groups[0].patterns, np.array([[1, 0], [1, 1]])
    )
    np.testing.assert_array_equal(collapsed_groups[0].frequencies, [2, 1])
    np.testing.assert_array_equal(
        collapsed_groups[1].patterns, np.array([[0, 1], [1, 1]])
    )
    np.testing.assert_array_equal(collapsed_groups[1].frequencies, [2, 1])


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        (np.array([[0], [1]]), "1D"),
        (np.array([0]), "one value per person"),
        (np.array([0.0, np.nan]), "missing values"),
        (np.array([0, np.nan], dtype=object), "missing values"),
        (np.array([0, "a"], dtype=object), "mutually comparable"),
    ],
)
def test_invalid_groups_raise(groups, message):
    with pytest.raises(MirtDataError, match=message):
        collapse_with_groups([[1, 0], [0, 1]], groups)


def test_compute_pattern_likelihood_validates_callback_output():
    collapsed = collapse_patterns([[1, 0], [0, 1], [1, 0]])
    theta = np.array([-1.0, 0.0, 1.0])

    values = compute_pattern_likelihood(
        collapsed,
        lambda patterns, points: patterns.sum(axis=1, keepdims=True) + points,
        theta,
    )
    assert values.shape == (2, 3)

    with pytest.raises(MirtValidationError, match="one leading entry"):
        compute_pattern_likelihood(
            collapsed, lambda _patterns, _theta: np.ones(3), theta
        )


def test_weighted_sum_uses_pattern_frequencies():
    collapsed = collapse_patterns([[1, 0], [0, 1], [1, 0]])

    assert weighted_sum_from_collapsed(collapsed, [2.5, -1.0]) == 4.0

    with pytest.raises(MirtValidationError, match="one value per pattern"):
        weighted_sum_from_collapsed(collapsed, [[2.5], [-1.0]])


def test_collapse_helpers_are_available_from_public_namespaces():
    assert mirt.collapse_patterns is collapse_patterns
    assert mirt.collapse_with_groups is collapse_with_groups
    assert mirt.compute_pattern_likelihood is compute_pattern_likelihood
    assert mirt.weighted_sum_from_collapsed is weighted_sum_from_collapsed
