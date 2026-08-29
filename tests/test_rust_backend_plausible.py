"""Contracts for sampling and diagnostic NumPy fallback kernels."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mirt
import mirt.backends.rust.plausible as plausible
from mirt._core import sigmoid
from mirt.backends.rust._helpers import RUST_AVAILABLE


@pytest.fixture
def numpy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plausible, "rust_enabled", lambda: False)


def _item_parameters(n_items: int) -> tuple[np.ndarray, np.ndarray]:
    return np.linspace(0.7, 1.6, n_items), np.linspace(-0.8, 0.9, n_items)


def _reference_observed_margins(
    responses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_items = responses.shape[1]
    univariate = np.zeros(n_items)
    bivariate = np.zeros((n_items, n_items))
    for first in range(n_items):
        valid = responses[:, first] >= 0
        if valid.any():
            univariate[first] = responses[valid, first].mean()
        for second in range(first + 1, n_items):
            pair_valid = valid & (responses[:, second] >= 0)
            if pair_valid.any():
                value = np.mean(
                    responses[pair_valid, first] * responses[pair_valid, second]
                )
                bivariate[first, second] = value
                bivariate[second, first] = value
    return univariate, bivariate


def _reference_expected_margins(
    points: np.ndarray,
    weights: np.ndarray,
    discrimination: np.ndarray,
    difficulty: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = sigmoid(
        discrimination[:, None] * (points[None, :] - difficulty[:, None])
    )
    univariate = probabilities @ weights
    bivariate = np.zeros((discrimination.size, discrimination.size))
    for first in range(discrimination.size):
        for second in range(first + 1, discrimination.size):
            value = np.sum(probabilities[first] * probabilities[second] * weights)
            bivariate[first, second] = value
            bivariate[second, first] = value
    return univariate, bivariate


def test_posterior_sampler_matches_discrete_posterior(numpy_fallback: None) -> None:
    responses = np.array([[1, 1, 0, -1]], dtype=np.int32)
    points = np.array([-1.5, -0.25, 0.75, 2.0])
    weights = np.array([0.15, 0.35, 0.4, 0.1])
    discrimination, difficulty = _item_parameters(responses.shape[1])

    draws = plausible.generate_plausible_values_posterior(
        responses,
        points,
        weights,
        discrimination,
        difficulty,
        n_plausible=10_000,
        jitter_sd=0.0,
        seed=20260828,
    )[0]

    probabilities = sigmoid(
        discrimination[None, :] * (points[:, None] - difficulty[None, :])
    )
    likelihood = probabilities[:, 0] * probabilities[:, 1] * (1.0 - probabilities[:, 2])
    expected = likelihood * weights
    expected /= expected.sum()
    actual = np.array([(draws == point).mean() for point in points])
    assert_allclose(actual, expected, atol=0.015)


def test_posterior_sampler_is_reproducible_and_handles_missingness(
    numpy_fallback: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plausible, "_entry_chunk_size", lambda *_: 1)
    responses = np.array([[1, -1, 0], [-9, -1, -4]], dtype=np.int32)
    points = np.array([-1.0, 0.0, 1.0])
    weights = np.array([0.2, 0.6, 0.2])
    discrimination, difficulty = _item_parameters(3)
    arguments = (
        responses,
        points,
        weights,
        discrimination,
        difficulty,
    )

    first = plausible.generate_plausible_values_posterior(
        *arguments, n_plausible=20, seed=17
    )
    second = plausible.generate_plausible_values_posterior(
        *arguments, n_plausible=20, seed=17
    )

    assert_array_equal(first, second)
    assert first.shape == (2, 20)
    assert np.isfinite(first).all()


def test_mcmc_sampler_tracks_response_direction_and_prior(
    numpy_fallback: None,
) -> None:
    n_items = 20
    responses = np.vstack(
        [
            np.ones((20, n_items), dtype=np.int32),
            np.zeros((20, n_items), dtype=np.int32),
            -np.ones((20, n_items), dtype=np.int32),
        ]
    )

    draws = plausible.generate_plausible_values_mcmc(
        responses,
        np.ones(n_items),
        np.zeros(n_items),
        n_plausible=100,
        n_iter=20,
        seed=5,
    )

    assert draws[:20].mean() > 1.5
    assert draws[20:40].mean() < -1.5
    assert abs(draws[40:].mean()) < 0.15


def test_mcmc_sampler_is_seed_reproducible(numpy_fallback: None) -> None:
    responses = np.array([[1, 0, -1], [0, 1, 1]], dtype=np.int32)
    discrimination, difficulty = _item_parameters(3)

    first = plausible.generate_plausible_values_mcmc(
        responses,
        discrimination,
        difficulty,
        n_plausible=8,
        n_iter=12,
        seed=91,
    )
    second = plausible.generate_plausible_values_mcmc(
        responses,
        discrimination,
        difficulty,
        n_plausible=8,
        n_iter=12,
        seed=91,
    )

    assert_array_equal(first, second)


def test_observed_margins_match_pairwise_complete_reference(
    numpy_fallback: None,
) -> None:
    responses = np.array(
        [[0, 1, 2, -1], [1, -1, 0, 2], [2, 1, 1, 0], [-1, -1, -1, -1]],
        dtype=np.int32,
    )

    actual = plausible.compute_observed_margins(responses)

    expected = _reference_observed_margins(responses)
    assert_allclose(actual[0], expected[0])
    assert_allclose(actual[1], expected[1])
    assert_allclose(np.diag(actual[1]), 0.0)


def test_expected_margins_match_quadrature_reference(
    numpy_fallback: None,
) -> None:
    points = np.linspace(-3.0, 3.0, 17)
    weights = np.exp(-0.5 * points**2)
    weights /= weights.sum()
    discrimination, difficulty = _item_parameters(7)

    actual = plausible.compute_expected_margins(
        points, weights, discrimination, difficulty
    )

    expected = _reference_expected_margins(points, weights, discrimination, difficulty)
    assert_allclose(actual[0], expected[0], rtol=1e-13, atol=1e-13)
    assert_allclose(actual[1], expected[1], rtol=1e-13, atol=1e-13)


def test_bootstrap_indices_are_reproducible_bounded_and_shaped(
    numpy_fallback: None,
) -> None:
    first = plausible.generate_bootstrap_indices(11, 5, seed=44)
    second = plausible.generate_bootstrap_indices(11, 5, seed=44)

    assert_array_equal(first, second)
    assert first.shape == (5, 11)
    assert first.dtype == np.int64
    assert np.all((first >= 0) & (first < 11))
    assert plausible.generate_bootstrap_indices(0, 3, seed=1).shape == (3, 0)
    assert plausible.generate_bootstrap_indices(4, 0, seed=1).shape == (0, 4)


def test_resampling_preserves_requested_order_and_duplicates(
    numpy_fallback: None,
) -> None:
    responses = np.arange(15, dtype=np.int32).reshape(5, 3)
    indices = np.array([4, 1, 1, 0], dtype=np.int64)

    assert_array_equal(
        plausible.resample_responses(responses, indices), responses[indices]
    )


@pytest.mark.parametrize("indices", [np.array([-1]), np.array([3])])
def test_resampling_rejects_out_of_range_indices(
    numpy_fallback: None, indices: np.ndarray
) -> None:
    with pytest.raises(IndexError, match="existing response rows"):
        plausible.resample_responses(np.zeros((3, 2), dtype=np.int32), indices)


def test_imputation_matches_row_major_reference(numpy_fallback: None) -> None:
    responses = np.array([[-1, 1, -1], [0, -1, 1]], dtype=np.int32)
    theta = np.array([-0.5, 1.2])
    discrimination, difficulty = _item_parameters(3)
    seed = 93
    expected = responses.copy()
    rng = np.random.default_rng(seed)
    for person, item in zip(*np.nonzero(responses == -1), strict=True):
        probability = sigmoid(discrimination[item] * (theta[person] - difficulty[item]))
        expected[person, item] = int(rng.random() < probability)

    actual = plausible.impute_from_probabilities(
        responses, theta, discrimination, difficulty, seed=seed
    )

    assert_array_equal(actual, expected)
    assert_array_equal(actual[responses >= 0], responses[responses >= 0])


def test_imputation_supports_explicit_positive_missing_code(
    numpy_fallback: None,
) -> None:
    responses = np.array([[7, 1], [0, 7]], dtype=np.int32)

    actual = plausible.impute_from_probabilities(
        responses,
        np.array([-100.0, 100.0]),
        np.ones(2),
        np.zeros(2),
        missing_code=7,
        seed=3,
    )

    assert_array_equal(actual, [[0, 1], [0, 1]])


def test_multiple_imputation_is_reproducible_and_preserves_observed_values(
    numpy_fallback: None,
) -> None:
    responses = np.array([[-1, 1, 0], [1, -1, -1]], dtype=np.int32)
    discrimination, difficulty = _item_parameters(3)
    arguments = (
        responses,
        np.array([-0.4, 0.7]),
        np.array([0.2, 0.3]),
        discrimination,
        difficulty,
    )

    first = plausible.multiple_imputation(*arguments, n_imputations=6, seed=31)
    second = plausible.multiple_imputation(*arguments, n_imputations=6, seed=31)

    assert_array_equal(first, second)
    assert first.shape == (6, 2, 3)
    assert np.isin(first, [0, 1]).all()
    for imputation in first:
        assert_array_equal(imputation[responses >= 0], responses[responses >= 0])


@pytest.mark.parametrize(
    ("responses", "error", "match"),
    [
        (np.array([0, 1]), ValueError, "two-dimensional"),
        (np.array([[0.0, 1.0]]), TypeError, "integers"),
        (np.array([[0, 2]], dtype=np.int32), ValueError, "coded as 0 or 1"),
        (
            np.array([[np.iinfo(np.int32).max + 1]], dtype=np.int64),
            ValueError,
            "32-bit",
        ),
    ],
)
def test_samplers_validate_response_storage(
    numpy_fallback: None,
    responses: np.ndarray,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        plausible.generate_plausible_values_mcmc(
            responses,
            np.ones(responses.shape[-1]),
            np.zeros(responses.shape[-1]),
        )


@pytest.mark.parametrize(
    ("points", "weights", "match"),
    [
        (np.array([]), np.array([]), "must not be empty"),
        (np.array([0.0]), np.array([0.5, 0.5]), "length 1"),
        (np.array([0.0, 1.0]), np.array([1.0, -0.1]), "nonnegative"),
        (np.array([0.0, 1.0]), np.zeros(2), "positive total mass"),
        (np.array([0.0, np.nan]), np.ones(2), "finite"),
    ],
)
def test_quadrature_contract_is_validated(
    numpy_fallback: None,
    points: np.ndarray,
    weights: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        plausible.compute_expected_margins(points, weights, np.ones(2), np.zeros(2))


@pytest.mark.parametrize(
    ("keyword", "value", "error", "match"),
    [
        ("n_plausible", 0, ValueError, "positive"),
        ("n_plausible", True, TypeError, "integer"),
        ("n_iter", 0, ValueError, "positive"),
        ("proposal_sd", 0.0, ValueError, "positive"),
        ("proposal_sd", np.inf, ValueError, "finite"),
        ("seed", -1, ValueError, "between"),
        ("seed", 1.5, TypeError, "integer"),
    ],
)
def test_mcmc_controls_are_validated(
    numpy_fallback: None,
    keyword: str,
    value: object,
    error: type[Exception],
    match: str,
) -> None:
    responses = np.array([[0, 1]], dtype=np.int32)
    with pytest.raises(error, match=match):
        plausible.generate_plausible_values_mcmc(
            responses,
            np.ones(2),
            np.zeros(2),
            **{keyword: value},
        )


@pytest.mark.parametrize(
    ("keyword", "value", "match"),
    [
        ("theta", np.array([0.0]), "length 2"),
        ("discrimination", np.ones(3), "length 2"),
        ("difficulty", np.array([0.0, np.nan]), "finite"),
        ("missing_code", True, "integer"),
    ],
)
def test_imputation_contract_is_validated(
    numpy_fallback: None,
    keyword: str,
    value: object,
    match: str,
) -> None:
    arguments: dict[str, object] = {
        "responses": np.array([[-1, 1], [0, -1]], dtype=np.int32),
        "theta": np.zeros(2),
        "discrimination": np.ones(2),
        "difficulty": np.zeros(2),
    }
    arguments[keyword] = value
    with pytest.raises((TypeError, ValueError), match=match):
        plausible.impute_from_probabilities(**arguments)


def test_multiple_imputation_rejects_negative_uncertainty(
    numpy_fallback: None,
) -> None:
    with pytest.raises(ValueError, match="theta_se must be nonnegative"):
        plausible.multiple_imputation(
            np.array([[-1, 0]], dtype=np.int32),
            np.zeros(1),
            np.array([-0.1]),
            np.ones(2),
            np.zeros(2),
        )


@pytest.mark.skipif(not RUST_AVAILABLE, reason="native backend is unavailable")
def test_native_and_numpy_margin_paths_agree() -> None:
    previous = mirt.get_backend()
    responses = np.array([[0, 1, -1], [1, 0, 1], [1, 1, 0], [-1, 0, 1]], dtype=np.int32)
    points = np.linspace(-2.0, 2.0, 9)
    weights = np.exp(-0.5 * points**2)
    weights /= weights.sum()
    discrimination, difficulty = _item_parameters(3)
    try:
        mirt.set_backend("numpy")
        observed_numpy = plausible.compute_observed_margins(responses)
        expected_numpy = plausible.compute_expected_margins(
            points, weights, discrimination, difficulty
        )
        mirt.set_backend("rust")
        observed_native = plausible.compute_observed_margins(responses)
        expected_native = plausible.compute_expected_margins(
            points, weights, discrimination, difficulty
        )
    finally:
        mirt.set_backend(previous)

    assert_allclose(observed_native[0], observed_numpy[0])
    assert_allclose(observed_native[1], observed_numpy[1])
    assert_allclose(expected_native[0], expected_numpy[0], rtol=1e-14, atol=1e-14)
    assert_allclose(expected_native[1], expected_numpy[1], rtol=1e-14, atol=1e-14)
