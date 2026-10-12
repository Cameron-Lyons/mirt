"""Tests for classical test theory utilities."""

import numpy as np
import pytest

import mirt
import mirt._classical as classical_core
import mirt.backends.rust.polytomous as polytomous_backend
import mirt.utils.classical as classical
from mirt.exceptions import MirtValidationError
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


def _variance_with_missing(values: np.ndarray) -> float:
    observed = values[np.isfinite(values)]
    return float(np.var(observed, ddof=1)) if observed.size > 1 else 0.0


def _alpha_with_missing(responses: np.ndarray) -> float:
    if responses.shape[1] < 2:
        return 0.0
    scored = np.any(np.isfinite(responses), axis=1)
    variance = _variance_with_missing(np.nansum(responses[scored], axis=1))
    if variance == 0:
        return 0.0
    item_variances = [
        _variance_with_missing(responses[:, item]) for item in range(responses.shape[1])
    ]
    n_items = responses.shape[1]
    return n_items / (n_items - 1) * (1 - sum(item_variances) / variance)


def _complete_case_reference(
    responses: np.ndarray,
) -> tuple[float, np.ndarray, float, float]:
    """R itemstats: listwise alpha, mean and SD; alpha_if_rm on other items."""

    def listwise_alpha(values: np.ndarray) -> float:
        rows = values[np.all(np.isfinite(values), axis=1)]
        totals = rows.sum(axis=1)
        if np.var(totals) == 0.0:
            return 0.0
        return _alpha(rows)

    answered = np.any(np.isfinite(responses), axis=0)
    kept = responses[:, answered]
    deleted = np.full(responses.shape[1], np.nan)
    deleted[answered] = [
        listwise_alpha(np.delete(kept, item, axis=1)) for item in range(kept.shape[1])
    ]
    totals = kept[np.all(np.isfinite(kept), axis=1)].sum(axis=1)
    return listwise_alpha(kept), deleted, totals.mean(), totals.std(ddof=1)


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
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 1.0],
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
            _alpha_with_missing(np.delete(responses, item, axis=1))
            for item in range(responses.shape[1])
        ]
    )
    monkeypatch.setattr(
        classical_core,
        "_ALPHA_DELETED_CHUNK_ELEMENTS",
        responses.shape[0] * 3,
    )

    actual = classical._alpha_if_deleted_numpy(responses)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.skipif(not classical.RUST_AVAILABLE, reason="native extension unavailable")
def test_alpha_if_deleted_native_matches_sparse_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The zero-filled kernels agree; traditional() itself handles missing
    # responses by complete cases or pairs.
    responses = np.array(
        [
            [1.0, np.nan, np.nan, np.nan],
            [np.nan, 1.0, np.nan, np.nan],
            [0.0, 1.0, 1.0, np.nan],
            [1.0, 0.0, np.nan, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    monkeypatch.setattr(polytomous_backend, "rust_enabled", lambda: True)
    native = polytomous_backend.compute_alpha_if_deleted(responses)

    fallback = classical_core._alpha_if_deleted_numpy(responses)

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


@pytest.mark.parametrize("persons, items", [(97, 2000), (97, 10000), (101, 5000)])
def test_traditional_centers_long_nearly_constant_tests(persons, items) -> None:
    responses = np.ones((persons, items))
    responses[0, 0] = 0.0

    uncorrected = traditional(responses, use_corrected_correlation=False)
    corrected = traditional(responses)

    assert uncorrected.discrimination[0] == pytest.approx(1.0, abs=1e-14, rel=0)
    assert np.all(np.abs(uncorrected.discrimination) <= 1.0)
    np.testing.assert_array_equal(uncorrected.discrimination[1:], 0.0)
    np.testing.assert_array_equal(corrected.discrimination, 0.0)


@pytest.mark.parametrize("backend", ["auto", "numpy"])
@pytest.mark.parametrize("budget", [1, 71, 262_144])
@pytest.mark.parametrize("corrected", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_traditional_blocks_match_scalar_statistics(
    monkeypatch, backend, budget, corrected, missing
) -> None:
    rng = np.random.default_rng(961)
    responses = rng.integers(0, 2, (53, 9)).astype(float)
    if missing:
        responses[rng.random(responses.shape) < 0.2] = np.nan
        responses[:, -1] = np.nan
        responses[3:, -2] = 1.0
        responses[:3] = np.nan
        responses[1, 0] = 1.0
        responses[2, 1] = 0.0
    responses = np.asfortranarray(responses)
    responses.setflags(write=False)
    before = responses.copy()
    totals = np.nansum(responses, axis=1)
    scored = np.any(np.isfinite(responses), axis=1)
    expected_difficulty = []
    expected_discrimination = []
    for item in range(responses.shape[1]):
        valid = np.isfinite(responses[:, item])
        values = responses[valid, item]
        item_totals = totals[valid] - values if corrected else totals[valid]
        expected_difficulty.append(values.mean() if values.size else np.nan)
        correlation = 0.0
        if values.size > 1 and np.ptp(values) > 0 and np.ptp(item_totals) > 0:
            correlation = np.corrcoef(values, item_totals)[0, 1]
        expected_discrimination.append(correlation)
    expected_alpha, expected_deleted, expected_mean, expected_sd = (
        _complete_case_reference(responses)
    )
    monkeypatch.setattr(classical, "_TRADITIONAL_CHUNK_ELEMENTS", budget)
    original_backend = mirt.get_backend()
    try:
        mirt.set_backend(backend)
        result = traditional(responses, use_corrected_correlation=corrected)
    finally:
        mirt.set_backend(original_backend)

    np.testing.assert_allclose(result.difficulty, expected_difficulty, atol=1e-14)
    np.testing.assert_allclose(
        result.discrimination, expected_discrimination, atol=1e-14
    )
    np.testing.assert_allclose(
        result.alpha_if_deleted, expected_deleted, atol=1e-13, equal_nan=True
    )
    assert result.alpha == pytest.approx(expected_alpha, abs=1e-13)
    assert result.mean_score == pytest.approx(expected_mean)
    assert result.sd_score == pytest.approx(expected_sd)
    if not missing:
        assert result.mean_score == pytest.approx(np.mean(totals[scored]))
    assert result.n_persons == 53
    assert result.n_items == 9
    np.testing.assert_array_equal(responses, before)


def test_traditional_numpy_reuses_moments_with_bounded_variance_buffers(
    monkeypatch,
) -> None:
    responses = np.random.default_rng(96).integers(0, 2, (53, 9)).astype(float)
    original_variance = classical_core._sample_variance
    matrix_sizes = []

    def bounded_variance(values, **kwargs):
        if values.ndim == 2:
            matrix_sizes.append(values.size)
        return original_variance(values, **kwargs)

    def unexpected_native(*args, **kwargs):
        raise AssertionError("the NumPy backend must reuse its prepared moments")

    monkeypatch.setattr(classical, "_sample_variance", bounded_variance)
    monkeypatch.setattr(classical_core, "_sample_variance", bounded_variance)
    monkeypatch.setattr(classical_core, "_ALPHA_DELETED_CHUNK_ELEMENTS", 53 * 2)
    monkeypatch.setattr(classical, "_rust_alpha_if_deleted", unexpected_native)
    original_backend = mirt.get_backend()
    try:
        mirt.set_backend("numpy")
        actual = traditional(responses)
    finally:
        mirt.set_backend(original_backend)

    assert matrix_sizes
    assert max(matrix_sizes) <= 53 * 2
    expected = [
        _alpha(np.delete(responses, item, axis=1)) for item in range(responses.shape[1])
    ]
    np.testing.assert_allclose(actual.alpha_if_deleted, expected, atol=1e-14)


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
def test_sample_variance_reuses_owned_buffer_without_mutating_input(axis) -> None:
    values = np.asfortranarray(np.arange(20, dtype=float).reshape(4, 5))
    values[:3, 0] = np.nan
    values[:, 2] = np.nan
    before = values.copy()
    values.setflags(write=False)
    expected = (
        _variance_with_missing(values)
        if axis is None
        else np.apply_along_axis(_variance_with_missing, axis, values)
    )

    actual = classical_core._sample_variance(values, axis=axis)

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(values, before)


@pytest.mark.skipif(not classical.RUST_AVAILABLE, reason="native extension unavailable")
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_native_alpha_accepts_readonly_response_layouts(monkeypatch, layout) -> None:
    responses = np.random.default_rng(13).integers(0, 4, (17, 12)).astype(float)
    responses[::3, ::2] = np.nan
    if layout == "F":
        responses = np.asfortranarray(responses)
    elif layout == "strided":
        responses = responses[::-1, ::2]
    before = responses.copy()
    responses.setflags(write=False)
    expected = [
        _alpha_with_missing(np.delete(responses, item, axis=1))
        for item in range(responses.shape[1])
    ]
    monkeypatch.setattr(polytomous_backend, "rust_enabled", lambda: True)

    actual = polytomous_backend.compute_alpha_if_deleted(responses)

    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_array_equal(responses, before)


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


# Polytomous scores and R mirt 1.48 itemstats(small) / itemstats(small_missing).
_SMALL_POLYTOMOUS = np.array(
    [
        [1, 2, 0, 1],
        [2, 1, 3, 3],
        [1, 2, 0, 0],
        [2, 2, 1, 3],
        [1, 1, 2, 3],
        [0, 1, 0, 3],
        [0, 0, 1, 1],
        [0, 0, 1, 0],
        [1, 2, 3, 2],
        [2, 1, 3, 2],
        [2, 3, 3, 1],
        [0, 2, 2, 2],
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [2, 3, 3, 3],
        [2, 1, 2, 2],
    ],
    dtype=float,
)


def _small_polytomous_with_missing() -> np.ndarray:
    responses = _SMALL_POLYTOMOUS.copy()
    responses[[1, 4, 9, 13], [0, 2, 3, 1]] = np.nan
    responses[6, [0, 3]] = np.nan
    return responses


@pytest.mark.parametrize("backend", ["auto", "numpy"])
@pytest.mark.parametrize(
    ("with_missing", "overall", "items"),
    [
        (
            False,
            (16, 5.5, 3.386247, 0.7935249, 0.7819767, 1.581139),
            (
                [1.0, 1.3125, 1.5625, 1.625],
                [0.8944272, 1.0144785, 1.2093387, 1.2041595],
                [0.7222222, 0.5295545, 0.6039572, 0.5420949],
                [0.6788194, 0.7570873, 0.7228440, 0.7575188],
            ),
        ),
        (
            True,
            (11, 5.545455, 3.445682, 0.7886875, 0.7738642, 1.638551),
            (
                [1.0, 1.4, 1.533333, 1.642857],
                [0.8770580, 0.9856108, 1.2459458, 1.2774459],
                [0.6320804, 0.6296587, 0.4815220, 0.4048728],
                [0.6706231, 0.6896186, 0.6692607, 0.75],
            ),
        ),
    ],
)
def test_traditional_matches_r_itemstats_for_polytomous_items(
    backend: str, with_missing: bool, overall: tuple, items: tuple
) -> None:
    responses = _small_polytomous_with_missing() if with_missing else _SMALL_POLYTOMOUS
    original_backend = mirt.get_backend()
    try:
        mirt.set_backend(backend)
        result = traditional(responses)
    finally:
        mirt.set_backend(original_backend)

    n_complete, mean_total, sd_total, std_alpha, alpha, sem_alpha = overall
    assert result.n_complete == n_complete
    assert result.mean_score == pytest.approx(mean_total, abs=5e-7)
    assert result.sd_score == pytest.approx(sd_total, abs=5e-7)
    assert result.std_alpha == pytest.approx(std_alpha, abs=5e-7)
    assert result.alpha == pytest.approx(alpha, abs=5e-7)
    assert result.sem_alpha == pytest.approx(sem_alpha, abs=5e-7)
    item_mean, item_sd, cor_if_rm, alpha_if_rm = items
    np.testing.assert_allclose(result.difficulty, item_mean, atol=5e-7)
    np.testing.assert_allclose(result.item_sd, item_sd, atol=5e-7)
    np.testing.assert_allclose(result.discrimination, cor_if_rm, atol=5e-7)
    np.testing.assert_allclose(result.alpha_if_deleted, alpha_if_rm, atol=5e-7)


def test_traditional_pairwise_uses_the_pairwise_covariance_matrix() -> None:
    responses = _small_polytomous_with_missing()
    observed = np.isfinite(responses)
    n_items = responses.shape[1]
    covariance = np.empty((n_items, n_items))
    for row in range(n_items):
        for column in range(n_items):
            both = observed[:, row] & observed[:, column]
            covariance[row, column] = np.cov(
                responses[both, row], responses[both, column]
            )[0, 1]

    def alpha(matrix: np.ndarray) -> float:
        k = matrix.shape[0]
        return k / (k - 1) * (1.0 - np.trace(matrix) / matrix.sum())

    result = traditional(responses, missing="pairwise")

    assert result.missing == "pairwise"
    assert result.alpha == pytest.approx(alpha(covariance))
    np.testing.assert_allclose(
        result.alpha_if_deleted,
        [
            alpha(np.delete(np.delete(covariance, item, 0), item, 1))
            for item in range(n_items)
        ],
    )
    assert result.sd_score == pytest.approx(np.sqrt(covariance.sum()))
    assert result.mean_score == pytest.approx(np.nanmean(responses, axis=0).sum())
    rest_covariance = covariance.sum(axis=1) - np.diag(covariance)
    rest_variance = covariance.sum() - 2 * covariance.sum(axis=1) + np.diag(covariance)
    np.testing.assert_allclose(
        result.discrimination,
        rest_covariance / np.sqrt(np.diag(covariance) * rest_variance),
    )


def test_traditional_missing_modes_agree_on_complete_data() -> None:
    complete = traditional(_SMALL_POLYTOMOUS)
    pairwise = traditional(_SMALL_POLYTOMOUS, missing="pairwise")

    for name in ("alpha", "std_alpha", "mean_score", "sd_score", "sem_alpha"):
        assert getattr(pairwise, name) == pytest.approx(getattr(complete, name))
    np.testing.assert_allclose(pairwise.alpha_if_deleted, complete.alpha_if_deleted)
    np.testing.assert_allclose(pairwise.discrimination, complete.discrimination)


def test_traditional_alpha_is_unbiased_under_random_missingness() -> None:
    responses = mirt.simdata(model="2PL", n_persons=6000, n_items=12, seed=7)
    full = traditional(responses).alpha
    masked = np.random.default_rng(2).random(responses.shape) < 0.1
    incomplete = np.where(masked, -1, responses)

    # Counting missing responses as zero in the totals gave 0.63 here.
    assert traditional(incomplete).alpha == pytest.approx(full, abs=0.03)
    assert traditional(incomplete, missing="pairwise").alpha == pytest.approx(
        full, abs=0.01
    )


def test_traditional_warns_when_no_respondent_answers_every_item() -> None:
    # A planned-missing design: each respondent sees two of three items.
    responses = np.array(
        [
            [1, 1, -1],
            [0, 1, -1],
            [1, -1, 1],
            [0, -1, 0],
            [-1, 1, 1],
            [-1, 0, 0],
        ]
    )

    with pytest.warns(RuntimeWarning, match="missing='pairwise'"):
        result = traditional(responses)

    assert result.n_complete == 0
    assert np.isnan(result.alpha)
    assert np.isfinite(traditional(responses, missing="pairwise").alpha)


def test_traditional_validates_missing_policy() -> None:
    with pytest.raises(MirtValidationError, match="'complete' or 'pairwise'"):
        traditional([[0, 1], [1, 0]], missing="listwise")  # type: ignore[arg-type]


def test_item_fit_chisq_is_deprecated() -> None:
    with pytest.warns(DeprecationWarning, match='statistics=\\["X2"\\]'):
        item_fit_chisq([[0, 1], [1, 0]], [[0.5, 0.5], [0.5, 0.5]])


@pytest.mark.filterwarnings("ignore:item_fit_chisq is deprecated:DeprecationWarning")
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


@pytest.mark.filterwarnings("ignore:item_fit_chisq is deprecated:DeprecationWarning")
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


@pytest.mark.filterwarnings("ignore:item_fit_chisq is deprecated:DeprecationWarning")
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


@pytest.mark.filterwarnings("ignore:item_fit_chisq is deprecated:DeprecationWarning")
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
@pytest.mark.filterwarnings("ignore:item_fit_chisq is deprecated:DeprecationWarning")
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
