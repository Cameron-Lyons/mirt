"""Tests for response-pattern summaries."""

import numpy as np
import pytest

from mirt.utils.data import check_response_pattern


def test_mixed_category_extremes_use_item_specific_maxima():
    responses = np.array(
        [
            [0, 0],
            [1, 3],
            [1, 2],
            [-1, -1],
        ]
    )

    summary = check_response_pattern(responses, n_categories=[2, 4])

    assert summary["extreme_patterns"] == {
        "all_minimum": 1,
        "all_maximum": 1,
    }


def test_all_missing_data_has_no_extreme_patterns():
    responses = np.full((2, 3), -1)

    summary = check_response_pattern(responses)

    assert summary["missing_rate"] == 1.0
    assert summary["missing_by_item"] == [1.0, 1.0, 1.0]
    assert summary["missing_by_person"] == [3, 3]
    assert summary["extreme_patterns"] == {
        "all_minimum": 0,
        "all_maximum": 0,
    }


def test_inferred_maxima_are_item_specific():
    responses = np.array([[0, 0], [1, 3], [1, 2]])

    summary = check_response_pattern(responses)

    assert summary["extreme_patterns"]["all_maximum"] == 1


@pytest.mark.parametrize(
    "n_categories,match",
    [
        ([2], "must contain 2 values"),
        ([2, 0], "must be positive"),
        ([2, 3.5], "must contain integers"),
        (0, "must be positive"),
    ],
)
def test_invalid_category_definitions_raise(n_categories, match):
    with pytest.raises(ValueError, match=match):
        check_response_pattern(np.array([[0, 1]]), n_categories=n_categories)


def test_response_above_declared_category_range_raises():
    with pytest.raises(ValueError, match="outside n_categories"):
        check_response_pattern(np.array([[0, 2]]), n_categories=2)


def test_align_response_columns_selects_named_items_in_model_order():
    pl = pytest.importorskip("polars")
    from mirt.utils.data import align_response_columns

    frame = pl.DataFrame({"id": [1, 2], "b": [0, 1], "a": [1, None]})

    aligned = align_response_columns(frame, ["a", "b"])

    assert isinstance(aligned, pl.DataFrame)
    assert aligned.columns == ["a", "b"]
    assert align_response_columns(frame, ["id", "b", "a"]) is frame
    # Duplicate model labels cannot select columns, so order is kept.
    assert align_response_columns(frame, ["a", "a", "b"]) is frame


def test_align_response_columns_reports_missing_named_items():
    pd = pytest.importorskip("pandas")
    from mirt.exceptions import MirtDataError
    from mirt.utils.data import align_response_columns

    frame = pd.DataFrame(np.zeros((2, 3), dtype=int), columns=[10, 20, 30])
    assert align_response_columns(frame, ["30", "10"]).columns.tolist() == [30, 10]

    names = [f"item{index}" for index in range(12)]
    with pytest.raises(MirtDataError, match=r"item0, .*item9, \.\.\. \(12 in total\)"):
        align_response_columns(frame, names)


def test_dead_data_expand_table_is_removed():
    import mirt.utils.data as data_module
    from mirt.utils import expand_table

    assert not hasattr(data_module, "expand_table")
    assert expand_table.__module__ == "mirt.utils.transform"
