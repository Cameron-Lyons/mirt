"""Exact category counts shared by item summaries and imputation."""

import numpy as np
import pytest

import mirt._categorical as categorical


@pytest.mark.parametrize("chunk_elements", [1, 35, 1_000_000])
@pytest.mark.parametrize("floating", [False, True])
def test_frequencies_match_independent_item_counts(
    monkeypatch, chunk_elements, floating
):
    rng = np.random.default_rng(54)
    responses = rng.integers(-1, 6, size=(37, 10))[:, ::2]
    valid = responses >= 0
    valid[:, 0] = False
    valid[::3, 1] = False
    if floating:
        responses = responses.astype(float)
        responses[~valid] = np.nan
    monkeypatch.setattr(categorical, "_FREQUENCY_CHUNK_ELEMENTS", chunk_elements)
    expected = np.zeros((6, 5), dtype=np.intp)
    for item in range(5):
        values, counts = np.unique(responses[valid[:, item], item], return_counts=True)
        expected[values.astype(int), item] = counts

    actual = categorical.item_category_frequencies(responses, valid)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.dtype(np.intp)


def test_dense_table_limits_keep_sparse_and_missing_fallbacks(monkeypatch):
    responses = np.array([[0, 2], [2, 1]])
    valid = np.ones_like(responses, dtype=bool)
    assert categorical.item_category_frequencies(responses, ~valid) is None
    assert (
        categorical.item_category_frequencies(responses, valid, max_categories=2)
        is None
    )
    monkeypatch.setattr(categorical, "_MAX_FREQUENCY_ENTRIES", 5)
    assert categorical.item_category_frequencies(responses, valid) is None


@pytest.mark.parametrize("method", ["median", "mode"])
def test_imputation_ignores_positive_missing_category(method):
    from mirt.utils.imputation import impute_responses

    responses = np.array([[0, 1], [1, 1], [2, 0], [2, 2]])
    actual = impute_responses(responses, method=method, missing_code=2)
    np.testing.assert_array_equal(actual, [[0, 1], [1, 1], [0, 0], [0, 1]])
