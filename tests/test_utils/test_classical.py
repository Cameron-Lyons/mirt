"""Tests for classical test theory utilities."""

import numpy as np
import pytest

import mirt._classical as classical_core
import mirt.backends.rust.polytomous as polytomous_backend
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


def _scalar_item_fit_groups(
    responses: np.ndarray,
    expected: np.ndarray,
    missing: np.ndarray,
    group_idx: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference the former one-item-at-a-time grouped reductions."""
    n_items = responses.shape[1]
    group_counts = np.zeros((n_items, n_groups), dtype=np.intp)
    observed_counts = np.zeros((n_items, n_groups), dtype=np.float64)
    expected_counts = np.zeros((n_items, n_groups), dtype=np.float64)
    expected_variances = np.zeros((n_items, n_groups), dtype=np.float64)

    for item_idx in range(n_items):
        valid = (group_idx >= 0) & ~missing[:, item_idx]
        item_groups = group_idx[valid]
        probabilities = expected[valid, item_idx]
        group_counts[item_idx] = np.bincount(item_groups, minlength=n_groups)
        observed_counts[item_idx] = np.bincount(
            item_groups,
            weights=responses[valid, item_idx],
            minlength=n_groups,
        )
        expected_counts[item_idx] = np.bincount(
            item_groups,
            weights=probabilities,
            minlength=n_groups,
        )
        expected_variances[item_idx] = np.bincount(
            item_groups,
            weights=probabilities * (1.0 - probabilities),
            minlength=n_groups,
        )

    return group_counts, observed_counts, expected_counts, expected_variances


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


def test_alpha_if_deleted_batches_match_scalar_sparse_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20260828)
    responses = rng.integers(0, 2, size=(37, 13)).astype(np.float64)
    responses[rng.random(responses.shape) < 0.35] = np.nan
    responses[0] = np.nan
    responses[0, 4] = 1.0
    responses[1] = np.nan
    responses[1, 9] = 0.0
    expected = np.array(
        [
            classical._cronbach_alpha(np.delete(responses, item, axis=1))
            for item in range(responses.shape[1])
        ]
    )
    monkeypatch.setattr(
        classical_core,
        "_ALPHA_DELETED_CHUNK_ELEMENTS",
        responses.shape[0] * 3,
    )

    actual = classical._alpha_if_deleted_numpy(responses)

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=4 * np.finfo(np.float64).eps,
        atol=4 * np.finfo(np.float64).eps,
    )


@pytest.mark.skipif(not classical.RUST_AVAILABLE, reason="native extension unavailable")
def test_alpha_if_deleted_native_matches_sparse_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = np.array(
        [
            [1.0, np.nan, np.nan, np.nan],
            [np.nan, 1.0, np.nan, np.nan],
            [0.0, 1.0, 1.0, np.nan],
            [1.0, 0.0, np.nan, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    native = traditional(responses).alpha_if_deleted
    monkeypatch.setattr(classical, "RUST_AVAILABLE", False)

    fallback = traditional(responses).alpha_if_deleted

    np.testing.assert_allclose(
        native,
        fallback,
        rtol=4 * np.finfo(np.float64).eps,
        atol=4 * np.finfo(np.float64).eps,
    )


def test_alpha_if_deleted_with_two_items_is_zero() -> None:
    result = traditional([[0, 1], [1, 0], [1, 1]])

    np.testing.assert_array_equal(result.alpha_if_deleted, np.zeros(2))


def test_direct_backend_fallback_uses_shared_alpha_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = np.array(
        [
            [1.0, np.nan, 0.0, 1.0],
            [0.0, 1.0, np.nan, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    monkeypatch.setattr(polytomous_backend, "rust_enabled", lambda: False)

    actual = polytomous_backend.compute_alpha_if_deleted(responses)
    expected = classical_core._alpha_if_deleted_numpy(responses)

    np.testing.assert_array_equal(actual, expected)


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


def test_item_fit_group_aggregation_matches_scalar_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20260828)
    n_persons, n_items, n_groups = 19, 11, 4
    responses = rng.integers(0, 2, size=(n_persons, n_items)).astype(float)
    expected = rng.uniform(0.05, 0.95, size=responses.shape)
    missing = rng.random(responses.shape) < 0.2
    responses[missing] = np.nan
    expected[missing] = np.nan
    group_idx = rng.integers(0, n_groups, size=n_persons, dtype=np.intp)
    group_idx[[0, 7]] = -1
    monkeypatch.setattr(
        classical,
        "_ITEM_FIT_CHUNK_ELEMENTS",
        n_persons * 2,
    )

    actual = classical._aggregate_item_fit_groups(
        responses,
        expected,
        missing,
        group_idx,
        n_groups,
    )
    reference = _scalar_item_fit_groups(
        responses,
        expected,
        missing,
        group_idx,
        n_groups,
    )

    np.testing.assert_array_equal(actual[0], reference[0])
    for actual_values, reference_values in zip(actual[1:], reference[1:], strict=True):
        np.testing.assert_allclose(actual_values, reference_values, rtol=0.0, atol=0.0)


def test_item_fit_batches_group_reductions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(808)
    n_persons, n_items = 12, 7
    expected = rng.uniform(0.1, 0.9, size=(n_persons, n_items))
    responses = (rng.random(expected.shape) < expected).astype(int)
    grouping = np.arange(n_persons, dtype=float)
    monkeypatch.setattr(
        classical,
        "_ITEM_FIT_CHUNK_ELEMENTS",
        n_persons * 2,
    )
    original_bincount = np.bincount
    call_count = 0

    def counted_bincount(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_bincount(*args, **kwargs)

    monkeypatch.setattr(classical.np, "bincount", counted_bincount)

    item_fit_chisq(
        responses,
        expected,
        n_groups=3,
        grouping=grouping,
        min_group_size=1,
    )

    n_chunks = (n_items + 1) // 2
    assert call_count == 4 * n_chunks


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


def test_itemstats_exposes_distribution_descriptors() -> None:
    responses = np.array(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
            [1.0, 2.0, -1.0],
            [1.0, 2.0, -1.0],
        ]
    )

    result = itemstats(responses)

    np.testing.assert_array_equal(result.n_categories, [2, 3, 0])
    np.testing.assert_allclose(result.mode[:2], [0.0, 2.0])
    assert np.isnan(result.mode[2])
    expected_entropy = np.array(
        [
            np.log(2.0),
            -(0.25 * np.log(0.25) * 2 + 0.5 * np.log(0.5)),
        ]
    )
    np.testing.assert_allclose(result.entropy[:2], expected_entropy)
    np.testing.assert_allclose(
        result.effective_categories[:2],
        np.exp(expected_entropy),
    )
    assert np.isnan(result.entropy[2])
    assert np.isnan(result.effective_categories[2])


def test_itemstats_chunked_moments_match_direct_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20260829)
    responses = rng.integers(0, 6, size=(31, 13)).astype(np.float64)
    responses[rng.random(responses.shape) < 0.2] = np.nan
    monkeypatch.setattr(classical, "_ITEM_STATS_MOMENT_CHUNK_ELEMENTS", 62)

    result = itemstats(responses)

    for item_index in range(responses.shape[1]):
        observed = responses[np.isfinite(responses[:, item_index]), item_index]
        centered = observed - np.mean(observed)
        second = np.mean(centered**2)
        expected_skewness = np.mean(centered**3) / second**1.5
        expected_kurtosis = np.mean(centered**4) / second**2 - 3.0
        assert result.skewness[item_index] == pytest.approx(expected_skewness)
        assert result.kurtosis[item_index] == pytest.approx(expected_kurtosis)


def test_itemstats_sparse_large_codes_use_exact_frequency_fallback() -> None:
    responses = np.array([[0, 1000], [1000, 7], [1000, 1000], [-1, 7]])

    result = itemstats(responses)

    assert result.frequencies == [{0: 1, 1000: 2}, {7: 2, 1000: 2}]


def test_itemstats_dataframe_includes_distribution_descriptors() -> None:
    table = itemstats_to_dataframe(itemstats([[0, 1], [1, 1], [1, 2]]))
    columns = table.columns if hasattr(table, "columns") else table.schema.names()

    assert {
        "mode",
        "n_categories",
        "entropy",
        "effective_categories",
    }.issubset(columns)


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


def test_response_validation_checks_later_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(classical, "_RESPONSE_VALIDATION_CHUNK_ELEMENTS", 2)

    with pytest.raises(ValueError, match="integer-valued"):
        itemstats([[0.0, 1.0], [2.0, 1.5]])
