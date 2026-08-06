"""Tests for classical test theory utilities."""

import numpy as np
import pytest

import mirt.utils.classical as classical
from mirt.utils import ItemStats
from mirt.utils import itemstats as exported_itemstats
from mirt.utils.classical import (
    item_fit_chisq,
    itemstats,
    itemstats_to_dataframe,
    traditional,
)


def _alpha(responses: np.ndarray) -> float:
    item_variances = np.var(responses, axis=0, ddof=1)
    total_variance = np.var(np.sum(responses, axis=1), ddof=1)
    n_items = responses.shape[1]
    return float(
        n_items / (n_items - 1) * (1.0 - item_variances.sum() / total_variance)
    )


def test_itemstats_is_exported_from_utils() -> None:
    result = exported_itemstats([[0, 1], [1, 0]])

    assert isinstance(result, ItemStats)


def test_traditional_matches_reference_statistics() -> None:
    responses = np.array(
        [
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
        ]
    )

    result = traditional(responses)

    totals = responses.sum(axis=1)
    expected_discrimination = np.array(
        [
            np.corrcoef(responses[:, j], totals - responses[:, j])[0, 1]
            for j in range(responses.shape[1])
        ]
    )
    expected_deleted = np.array(
        [_alpha(np.delete(responses, j, axis=1)) for j in range(responses.shape[1])]
    )

    np.testing.assert_allclose(result.difficulty, responses.mean(axis=0))
    np.testing.assert_allclose(result.discrimination, expected_discrimination)
    assert result.alpha == pytest.approx(_alpha(responses))
    np.testing.assert_allclose(result.alpha_if_deleted, expected_deleted)
    assert result.mean_score == pytest.approx(totals.mean())
    assert result.sd_score == pytest.approx(totals.std(ddof=1))


def test_traditional_normalizes_supported_missing_representations() -> None:
    missing_code = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 99.0],
            [1.0, -7.0, 1.0],
            [0.0, 0.0, 0.0],
            [99.0, 99.0, 99.0],
        ]
    )
    normalized = missing_code.copy()
    normalized[(normalized < 0.0) | (normalized == 99.0)] = np.nan

    actual = traditional(missing_code, missing_code=99.0)
    expected = traditional(normalized)

    np.testing.assert_allclose(actual.difficulty, expected.difficulty, equal_nan=True)
    np.testing.assert_allclose(actual.discrimination, expected.discrimination)
    assert actual.alpha == pytest.approx(expected.alpha)
    np.testing.assert_allclose(actual.alpha_if_deleted, expected.alpha_if_deleted)
    assert actual.mean_score == pytest.approx(expected.mean_score)
    assert actual.sd_score == pytest.approx(expected.sd_score)


def test_traditional_python_fallback_matches_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = np.array(
        [
            [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    native = traditional(responses).alpha_if_deleted
    monkeypatch.setattr(classical, "RUST_AVAILABLE", False)

    fallback = traditional(responses).alpha_if_deleted

    np.testing.assert_allclose(fallback, native)


def test_traditional_supports_uncorrected_correlations() -> None:
    responses = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    totals = responses.sum(axis=1)

    result = traditional(responses, use_corrected_correlation=False)

    expected = np.array([np.corrcoef(responses[:, j], totals)[0, 1] for j in range(3)])
    np.testing.assert_allclose(result.discrimination, expected)


def test_traditional_validates_correlation_option() -> None:
    with pytest.raises(ValueError, match="must be boolean"):
        traditional([[0, 1], [1, 0]], use_corrected_correlation="yes")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        ([0, 1], "2D matrix"),
        (np.empty((0, 2)), "at least one person"),
        ([[1, 0]], "at least two respondents"),
        ([[0], [1]], "at least two items"),
        ([[0, 2], [1, 0]], "only 0, 1"),
        ([[0, 0.5], [1, 0]], "integer-valued"),
        ([[0, np.inf], [1, 0]], "finite"),
        ([[0, -np.inf], [1, 0]], "finite"),
        ([[np.nan, np.nan], [1, 0]], "at least two respondents with data"),
    ],
)
def test_traditional_validates_responses(
    responses: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        traditional(responses)  # type: ignore[arg-type]


def test_item_fit_uses_expected_count_variance() -> None:
    responses = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    expected = np.array(
        [
            [0.2, 0.8],
            [0.4, 0.7],
            [0.6, 0.3],
            [0.8, 0.6],
            [0.3, 0.2],
            [0.4, 0.6],
            [0.7, 0.4],
            [0.9, 0.5],
        ]
    )
    grouping = np.arange(8, dtype=float)

    chisq, p_values = item_fit_chisq(
        responses,
        expected,
        n_groups=2,
        grouping=grouping,
        min_group_size=2,
    )

    expected_chisq = np.zeros(2)
    for item in range(2):
        for group in (slice(0, 4), slice(4, 8)):
            observed_count = responses[group, item].sum()
            probabilities = expected[group, item]
            expected_count = probabilities.sum()
            variance = np.sum(probabilities * (1.0 - probabilities))
            expected_chisq[item] += (observed_count - expected_count) ** 2 / variance

    np.testing.assert_allclose(chisq, expected_chisq)
    assert np.all((p_values >= 0.0) & (p_values <= 1.0))


def test_item_fit_excludes_missing_responses() -> None:
    responses = np.array(
        [
            [0.0, 1.0],
            [1.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    expected = np.full_like(responses, 0.5)
    expected[1, 1] = np.nan
    normalized = responses.copy()
    normalized[normalized < 0.0] = np.nan

    actual = item_fit_chisq(
        responses,
        expected,
        n_groups=2,
        grouping=np.arange(8),
        min_group_size=2,
    )
    reference = item_fit_chisq(
        normalized,
        expected,
        n_groups=2,
        grouping=np.arange(8),
        min_group_size=2,
    )

    np.testing.assert_allclose(actual[0], reference[0])
    np.testing.assert_allclose(actual[1], reference[1])


def test_item_fit_ignores_fully_missing_respondents_when_grouping() -> None:
    responses = np.array(
        [
            [-1.0, -1.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    expected = np.full_like(responses, 0.5)

    actual = item_fit_chisq(
        responses,
        expected,
        n_groups=2,
        grouping=[np.nan, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        min_group_size=2,
    )
    reference = item_fit_chisq(
        responses[1:],
        expected[1:],
        n_groups=2,
        grouping=np.arange(6),
        min_group_size=2,
    )

    np.testing.assert_allclose(actual[0], reference[0])
    np.testing.assert_allclose(actual[1], reference[1])


@pytest.mark.parametrize(
    ("responses", "expected", "kwargs", "message"),
    [
        ([[0, 1], [1, 0]], [[0.5], [0.5]], {}, "expected has shape"),
        ([[0, 2], [1, 0]], [[0.5, 0.5], [0.5, 0.5]], {}, "only 0, 1"),
        (
            [[0, 1], [1, 0]],
            [[0.5, 1.1], [0.5, 0.5]],
            {},
            "between 0 and 1",
        ),
        (
            [[0, 1], [1, 0]],
            [[0.5, np.nan], [0.5, 0.5]],
            {},
            "must be finite",
        ),
        ([[0, 1], [1, 0]], [[0.5, 0.5], [0.5, 0.5]], {"n_groups": 1}, "n_groups"),
        (
            [[0, 1], [1, 0]],
            [[0.5, 0.5], [0.5, 0.5]],
            {"grouping": [0.0]},
            "one value per respondent",
        ),
        (
            [[0, 1], [1, 0]],
            [[0.5, 0.5], [0.5, 0.5]],
            {"min_group_size": 0},
            "positive integer",
        ),
        (
            [[-1, -1], [-1, -1]],
            [[np.nan, np.nan], [np.nan, np.nan]],
            {},
            "at least one observed",
        ),
    ],
)
def test_item_fit_validates_arguments(
    responses: object,
    expected: object,
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        item_fit_chisq(responses, expected, **kwargs)  # type: ignore[arg-type]


def test_itemstats_reports_missing_proportions_and_valid_counts() -> None:
    responses = np.array(
        [
            [0.0, 1.0, -1.0],
            [1.0, np.nan, -1.0],
            [2.0, 1.0, -1.0],
            [1.0, 0.0, -1.0],
        ]
    )

    result = itemstats(responses)

    np.testing.assert_array_equal(result.n, np.array([4, 3, 0]))
    np.testing.assert_array_equal(result.n_missing, np.array([0, 1, 4]))
    np.testing.assert_allclose(result.pct_missing, np.array([0.0, 0.25, 1.0]))
    np.testing.assert_allclose(result.mean[:2], np.array([1.0, 2.0 / 3.0]))
    np.testing.assert_allclose(
        result.sd[:2], np.array([np.sqrt(2 / 3), 1 / np.sqrt(3)])
    )
    np.testing.assert_allclose(result.min[:2], np.array([0.0, 0.0]))
    np.testing.assert_allclose(result.max[:2], np.array([2.0, 1.0]))
    assert np.isnan(result.mean[2])
    assert np.isnan(result.sd[2])
    assert np.isnan(result.min[2])
    assert np.isnan(result.max[2])
    assert result.frequencies == [{0: 1, 1: 2, 2: 1}, {0: 1, 1: 2}, {}]


def test_itemstats_matches_population_shape_statistics() -> None:
    responses = np.array([[0.0], [1.0], [1.0], [2.0]])

    result = itemstats(responses)

    centered = responses[:, 0] - responses[:, 0].mean()
    second = np.mean(centered**2)
    assert result.skewness[0] == pytest.approx(np.mean(centered**3) / second**1.5)
    assert result.kurtosis[0] == pytest.approx(np.mean(centered**4) / second**2 - 3.0)


def test_itemstats_can_propagate_missing_statistics() -> None:
    responses = np.array([[0.0, 1.0], [1.0, -1.0], [2.0, 0.0]])

    result = itemstats(responses, na_rm=False)

    assert np.all(np.isfinite(result.mean[:1]))
    assert np.isnan(result.mean[1])
    assert result.n[1] == 2
    assert result.frequencies[1] == {0: 1, 1: 1}


def test_itemstats_validates_missing_policy() -> None:
    with pytest.raises(ValueError, match="na_rm must be boolean"):
        itemstats([[0, 1], [1, 0]], na_rm="yes")  # type: ignore[arg-type]


def test_itemstats_honors_custom_and_negative_missing_codes() -> None:
    responses = np.array([[0.0, 1.0], [99.0, -7.0], [2.0, 0.0]])

    result = itemstats(responses, missing_code=99.0)

    np.testing.assert_array_equal(result.n, np.array([2, 2]))
    np.testing.assert_allclose(result.pct_missing, np.array([1 / 3, 1 / 3]))
    assert result.frequencies == [{0: 1, 2: 1}, {0: 1, 1: 1}]


def test_itemstats_dataframe_validates_item_names() -> None:
    stats = itemstats([[0, 1], [1, 0]])

    with pytest.raises(ValueError, match="item_names has length"):
        itemstats_to_dataframe(stats, item_names=["one"])


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        ([0, 1], "2D matrix"),
        (np.empty((2, 0)), "at least one person"),
        ([[0, 1.5], [1, 0]], "integer-valued"),
        ([[0, np.inf], [1, 0]], "finite"),
        ([[0, -np.inf], [1, 0]], "finite"),
        ([["yes", "no"]], "numeric"),
    ],
)
def test_itemstats_validates_responses(responses: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        itemstats(responses)  # type: ignore[arg-type]
