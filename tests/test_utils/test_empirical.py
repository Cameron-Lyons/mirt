"""Regression tests for empirical item diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

import mirt.utils.empirical as empirical_module
from mirt.models.custom import CustomItemModel, create_item_type
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel
from mirt.utils import itemGAM as exported_item_gam
from mirt.utils.empirical import (
    RMSD_DIF,
    empirical_ES,
    empirical_plot,
    empirical_rmsea,
    itemGAM,
    mantel_haenszel,
    weighted_RMSD_DIF,
)


def _binary_model(n_items: int = 3) -> TwoParameterLogistic:
    model = TwoParameterLogistic(n_items)
    model.set_parameters(
        discrimination=np.linspace(0.8, 1.4, n_items),
        difficulty=np.linspace(-0.7, 0.7, n_items),
    )
    return model


def _polytomous_model() -> GradedResponseModel:
    model = GradedResponseModel(2, [4, 3])
    model.set_parameters(
        discrimination=np.array([1.2, 0.9]),
        thresholds=np.array([[-1.0, 0.0, 1.0], [-0.6, 0.8, 0.0]]),
    )
    return model


class Counting2PL(TwoParameterLogistic):
    """Track public probability evaluations without changing their output."""

    def __init__(self, n_items: int) -> None:
        super().__init__(n_items)
        self.probability_calls = 0

    def probability(self, theta: np.ndarray, item_idx: int | None = None) -> np.ndarray:
        self.probability_calls += 1
        return super().probability(theta, item_idx)


def test_empirical_plot_uses_one_probability_call_and_manual_bin_means() -> None:
    model = Counting2PL(1)
    model.set_parameters(discrimination=np.array([1.3]), difficulty=np.array([-0.2]))
    theta = np.linspace(-2.0, 2.0, 20)
    responses = (theta > 0).astype(float).reshape(-1, 1)

    result = empirical_plot(model, responses, theta, item_idx=0, n_bins=4)

    expected = TwoParameterLogistic.probability(model, theta[:, None], 0)
    assert model.probability_calls == 1
    assert result.n_per_bin.tolist() == [5, 5, 5, 5]
    assert_allclose(
        result.expected_prop,
        [expected[start : start + 5].mean() for start in range(0, 20, 5)],
    )


def test_empirical_plot_treats_negative_and_nan_responses_as_missing() -> None:
    model = _binary_model(1)
    theta = np.array([-1.5, -0.5, 0.5, 1.5])
    responses_negative = np.array([[-1.0], [0.0], [1.0], [np.nan]])
    responses_nan = responses_negative.copy()
    responses_nan[0, 0] = np.nan

    negative = empirical_plot(model, responses_negative, theta, 0, n_bins=2)
    missing = empirical_plot(model, responses_nan, theta, 0, n_bins=2)

    assert negative.n_per_bin.sum() == 2
    assert_allclose(negative.theta_bins, missing.theta_bins)
    assert_allclose(negative.observed_prop, missing.observed_prop)
    assert_allclose(negative.expected_prop, missing.expected_prop)


def test_empirical_plot_uses_polytomous_expected_scores() -> None:
    model = _polytomous_model()
    theta = np.linspace(-2.0, 2.0, 7)
    responses = np.column_stack(
        [np.array([0, 0, 1, -1, 2, 3, 3], dtype=float), np.zeros(7)]
    )

    result = empirical_plot(model, responses, theta, item_idx=0, n_bins=1)

    valid = responses[:, 0] >= 0
    probabilities = model.probability(theta[valid, None], item_idx=0)
    expected_scores = probabilities @ np.arange(4)
    assert result.n_per_bin.tolist() == [6]
    assert_allclose(result.observed_prop, [responses[valid, 0].mean()])
    assert_allclose(result.expected_prop, [expected_scores.mean()])
    assert result.expected_prop[0] != pytest.approx(0.25)


def test_empirical_plot_returns_empty_arrays_when_item_is_all_missing() -> None:
    result = empirical_plot(
        _binary_model(1),
        np.array([[-1.0], [np.nan]]),
        np.array([-0.5, 0.5]),
        item_idx=0,
    )

    assert result.theta_bins.size == 0
    assert result.observed_prop.size == 0
    assert result.expected_prop.size == 0
    assert result.n_per_bin.dtype == np.intp


def test_empirical_rmsea_matches_itemwise_results_with_one_model_call() -> None:
    rng = np.random.default_rng(20260803)
    theta = np.linspace(-2.5, 2.5, 80)
    responses = rng.integers(0, 2, size=(80, 4)).astype(float)
    responses[::9, 1] = -1
    responses[::11, 2] = np.nan

    model = Counting2PL(4)
    model.set_parameters(
        discrimination=np.array([0.8, 1.0, 1.2, 1.4]),
        difficulty=np.array([-0.8, -0.2, 0.3, 0.9]),
    )
    result = empirical_rmsea(model, responses, theta, n_bins=8)

    baseline_model = _binary_model(4)
    baseline_model.set_parameters(**model.parameters)
    manual = []
    for item_idx in range(4):
        plot = empirical_plot(
            baseline_model, responses, theta, item_idx=item_idx, n_bins=8
        )
        nonempty = plot.n_per_bin > 0
        manual.append(np.sqrt(np.mean(plot.residuals[nonempty] ** 2)))

    assert model.probability_calls == 1
    assert_allclose(result, manual)


def test_empirical_rmsea_supports_polytomous_items() -> None:
    model = _polytomous_model()
    theta = np.linspace(-2.5, 2.5, 40)
    probabilities = model.probability(theta[:, None])
    responses = np.argmax(probabilities, axis=2).astype(float)
    responses[::7, 0] = -1

    result = empirical_rmsea(model, responses, theta, n_bins=5)

    assert result.shape == (2,)
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("budget", [1, 31, 262_144])
@pytest.mark.parametrize("n_bins", [1, 8, 101])
@pytest.mark.parametrize("polytomous", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_empirical_rmsea_streams_probabilities_and_bins(
    monkeypatch, budget, n_bins, polytomous, missing
) -> None:
    rng = np.random.default_rng(932)
    categories = [3, 4, 2] if polytomous else [2, 2, 2]
    model = GradedResponseModel(3, categories) if polytomous else _binary_model(3)
    theta = np.round(rng.normal(size=37), 1)
    responses = np.asfortranarray(
        np.column_stack(
            [rng.integers(0, count, size=theta.size) for count in categories]
        ),
        dtype=float,
    )
    if missing:
        responses[::3, 0] = -1
        responses[::5, 1] = np.nan
        responses[:, 2] = -1
    before = responses.copy()
    reference = []
    for item in range(3):
        plot = empirical_plot(model, responses, theta, item_idx=item, n_bins=n_bins)
        residuals = plot.residuals[plot.n_per_bin > 0]
        reference.append(np.sqrt(np.mean(residuals**2)) if residuals.size else np.nan)

    calls = []
    histogram_sizes = []
    original_probability = model.probability
    original_bincount = np.bincount

    def track_probability(points, item_idx=None):
        calls.append(len(points))
        return original_probability(points, item_idx)

    def track_bincount(values, weights=None, minlength=0):
        histogram_sizes.append(minlength)
        return original_bincount(values, weights=weights, minlength=minlength)

    monkeypatch.setattr(model, "probability", track_probability)
    monkeypatch.setattr(empirical_module, "_EMPIRICAL_MAX_PROBABILITY_VALUES", budget)
    monkeypatch.setattr(empirical_module.np, "bincount", track_bincount)
    actual = empirical_rmsea(model, responses, theta, n_bins=n_bins)

    assert_allclose(actual, reference, atol=1e-14)
    assert_allclose(responses, before)
    assert sum(calls) == len(theta)
    width = 3 * (max(categories) if polytomous else 1)
    assert max(calls) * width <= max(budget, width)
    assert max(histogram_sizes) <= max(budget, width)


def test_empirical_rmsea_supports_custom_shared_category_counts(monkeypatch) -> None:
    def ordinal(theta):
        probability = 1.0 / (1.0 + np.exp(-theta))
        return np.column_stack(
            [1.0 - probability, probability * (1.0 - probability), probability**2]
        )

    model = CustomItemModel(2, create_item_type("Ordinal", ordinal, n_categories=3))
    theta = np.linspace(-2, 2, 25)
    responses = np.random.default_rng(52).integers(0, 3, (25, 2))
    monkeypatch.setattr(empirical_module, "_EMPIRICAL_MAX_PROBABILITY_VALUES", 12)
    actual = empirical_rmsea(model, responses, theta, n_bins=5)
    reference = [
        np.sqrt(np.mean(empirical_plot(model, responses, theta, idx, 5).residuals ** 2))
        for idx in range(2)
    ]

    assert_allclose(actual, reference)


def test_empirical_rmsea_probes_unknown_probability_width(monkeypatch) -> None:
    class Model:
        n_items = 1
        n_factors = 1

        def __init__(self):
            self.rows = []

        def probability(self, theta):
            self.rows.append(len(theta))
            return np.broadcast_to([0.25, 0.25, 0.5], (len(theta), 1, 3))

    model = Model()
    monkeypatch.setattr(empirical_module, "_EMPIRICAL_MAX_PROBABILITY_VALUES", 6)
    actual = empirical_rmsea(model, np.ones((7, 1)), np.arange(7), n_bins=2)

    assert model.rows == [1, 2, 2, 2]
    assert_allclose(actual, 0.25)


@pytest.mark.parametrize("counts", [1, True, 3.5, [3], [3.0, 3.0], [3, 4]])
def test_empirical_rmsea_rejects_malformed_category_counts(counts) -> None:
    class Model:
        n_items = 2
        n_factors = 1
        n_categories = counts

        def probability(self, theta):
            return np.broadcast_to([0.25, 0.25, 0.5], (len(theta), 2, 3))

    with pytest.raises(ValueError, match="category counts"):
        empirical_rmsea(Model(), np.ones((5, 2)), np.arange(5))


@pytest.mark.parametrize("invalid", ["response", "probability"])
def test_empirical_rmsea_validates_later_blocks(monkeypatch, invalid) -> None:
    model = _binary_model(2)
    responses = np.ones((9, 2))
    theta = np.arange(9, dtype=float)
    if invalid == "response":
        responses[-1, 0] = 2
        message = "between 0 and 1"
    else:
        original = model.probability

        def bad_probability(points):
            probabilities = original(points)
            probabilities[points[:, 0] == 8, 0] = np.nan
            return probabilities

        monkeypatch.setattr(model, "probability", bad_probability)
        message = "finite"
    monkeypatch.setattr(empirical_module, "_EMPIRICAL_MAX_PROBABILITY_VALUES", 8)
    with pytest.raises(ValueError, match=message):
        empirical_rmsea(model, responses, theta)


@pytest.mark.parametrize(
    "theta, expected_indices, expected_means",
    [
        ([1e308] * 4, [1, 1, 1, 1], [1e308, 1e308]),
        ([-1e308, 1e308], [0, 1], [-1e308, 1e308]),
        ([np.finfo(float).max] * 4, [1, 1, 1, 1], [np.finfo(float).max] * 2),
        ([np.nextafter(0.0, 1.0)] * 4, [1, 1, 1, 1], [np.nextafter(0.0, 1.0)] * 2),
    ],
)
def test_theta_bins_handle_extreme_finite_values(
    theta, expected_indices, expected_means
):
    with np.errstate(over="raise", invalid="raise"):
        indices, means = empirical_module._build_theta_bins(np.asarray(theta), 2)

    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_array_equal(means, expected_means)


@pytest.mark.parametrize("n_bins", [1, 3, 12])
def test_theta_bins_preserve_quantile_ties_and_bin_means(n_bins) -> None:
    theta = np.array([1.0, -1.0, 0.0, 0.0, -1.0, 0.5, 1.0, 0.0])
    edges = np.percentile(theta, np.linspace(0, 100, n_bins + 1))
    expected_indices = np.clip(np.digitize(theta, edges) - 1, 0, n_bins - 1)
    expected_means = (edges[:-1] + edges[1:]) / 2
    for group in range(n_bins):
        if np.any(expected_indices == group):
            expected_means[group] = np.mean(theta[expected_indices == group])
    indices, means = empirical_module._build_theta_bins(theta, n_bins)

    np.testing.assert_array_equal(indices, expected_indices)
    assert_allclose(means, expected_means)


def test_polytomous_dif_metrics_use_expected_category_scores() -> None:
    reference = _polytomous_model()
    focal = _polytomous_model()
    shifted = focal.thresholds.copy()
    shifted[0, :3] += 0.45
    focal.set_parameters(thresholds=shifted)

    effect = empirical_ES(reference, focal, item_idx=0, n_points=51)
    rmsd = RMSD_DIF(reference, focal, item_idx=0, n_points=51)
    weighted_rmsd = weighted_RMSD_DIF(reference, focal, item_idx=0, n_points=51)

    theta = np.linspace(-4.0, 4.0, 51)[:, None]
    categories = np.arange(4)
    ref_scores = reference.probability(theta, 0) @ categories
    focal_scores = focal.probability(theta, 0) @ categories
    difference = focal_scores - ref_scores
    weights = stats.norm.pdf(theta[:, 0])
    weights /= weights.sum()

    assert_allclose(effect.signed_es, np.sum(weights * difference))
    assert_allclose(effect.unsigned_es, np.sum(weights * np.abs(difference)))
    assert_allclose(rmsd, np.sqrt(np.mean(difference**2)))
    assert_allclose(weighted_rmsd, np.sqrt(np.sum(weights * difference**2)))


def test_item_gam_is_exported_and_supports_polytomous_scores() -> None:
    assert exported_item_gam is itemGAM
    model = _polytomous_model()
    theta = np.linspace(-2.0, 2.0, 41)
    responses = np.column_stack(
        [
            np.clip(np.rint(theta + 1.5), 0, 3),
            np.clip(np.rint(theta + 1.0), 0, 2),
        ]
    )
    responses[::10, 0] = -1

    result = itemGAM(
        model,
        responses,
        theta,
        item_idx=0,
        n_grid=17,
        bandwidth=0.45,
    )

    expected_scores = model.probability(result.theta_grid[:, None], 0) @ np.arange(4)
    assert result.model_probs.shape == (17,)
    assert result.smoothed_probs.shape == (17,)
    assert result.se_bands.shape == (2, 17)
    assert_allclose(result.model_probs, expected_scores)
    assert np.all((result.smoothed_probs >= 0) & (result.smoothed_probs <= 3))
    assert np.all((result.se_bands >= 0) & (result.se_bands <= 3))
    assert result.raw_theta.size == 36


def test_item_gam_handles_constant_theta_without_zero_bandwidth() -> None:
    result = itemGAM(
        _binary_model(1),
        np.array([[0.0], [1.0], [1.0], [0.0]]),
        np.ones(4),
        item_idx=0,
        n_grid=5,
    )

    assert np.all(np.isfinite(result.smoothed_probs))
    assert np.all(np.isfinite(result.se_bands))


def test_item_gam_chunked_kernel_matches_single_block(monkeypatch) -> None:
    model = _binary_model(1)
    theta = np.linspace(-2.0, 2.0, 21)
    responses = (theta > 0).astype(float).reshape(-1, 1)
    baseline = itemGAM(model, responses, theta, item_idx=0, n_grid=11, bandwidth=0.4)

    monkeypatch.setattr(empirical_module, "KERNEL_BLOCK_ELEMENTS", 30)
    chunked = itemGAM(model, responses, theta, item_idx=0, n_grid=11, bandwidth=0.4)

    assert_allclose(chunked.smoothed_probs, baseline.smoothed_probs)
    assert_allclose(chunked.se_bands, baseline.se_bands)
    assert_allclose(chunked.model_probs, baseline.model_probs)


def test_item_gam_tiny_bandwidth_uses_nearest_observations_without_nan() -> None:
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        result = itemGAM(
            _binary_model(1),
            np.array([[0.0], [1.0], [0.0]]),
            np.array([0.0, 1.0, 2.0]),
            item_idx=0,
            n_grid=3,
            bandwidth=1e-300,
        )

    assert_allclose(result.smoothed_probs, [0.0, 1.0, 0.0])
    assert_allclose(result.se_bands, [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])


@pytest.mark.parametrize("budget", [1, 2_000_000])
def test_item_gam_missing_neighbors_preserve_tail_uncertainty(
    monkeypatch, budget
) -> None:
    monkeypatch.setattr(empirical_module, "KERNEL_BLOCK_ELEMENTS", budget)
    responses = np.array([[1.0, np.nan], [np.nan, 0.0], [np.nan, 1.0]])
    # At zero, the observed responses for the second item have kernel weights
    # near exp(-400): their means are representable, but squared weights vanish.
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        results = itemGAM(
            _binary_model(2),
            responses,
            np.array([0.0, np.sqrt(800), np.sqrt(800)]),
            n_grid=2,
            bandwidth=1.0,
            theta_margin=0.0,
            alpha=0.5,
        )

    assert_allclose(results[0].smoothed_probs, 1.0)
    assert_allclose(results[0].se_bands, 1.0)
    assert_allclose(results[1].smoothed_probs, 0.5)
    margin = stats.norm.ppf(0.75) * np.sqrt(0.25 / 2)
    assert_allclose(results[1].se_bands, [[0.5 - margin] * 2, [0.5 + margin] * 2])


@pytest.mark.parametrize("budget", [1, 100, 2_000_000])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("polytomous", [False, True])
@pytest.mark.parametrize("se", [False, True])
def test_item_gam_shared_smoothing_matches_scalar_reference(
    monkeypatch, budget, missing, polytomous, se
) -> None:
    model = _polytomous_model() if polytomous else _binary_model(2)
    rng = np.random.default_rng(961)
    theta = rng.normal(size=31)
    responses = np.column_stack(
        [
            rng.integers(0, 4 if polytomous else 2, theta.size),
            rng.integers(0, 3 if polytomous else 2, theta.size),
        ]
    ).astype(float)
    if missing:
        responses[::3, 0] = -1
        responses[::4, 1] = np.nan
    monkeypatch.setattr(empirical_module, "KERNEL_BLOCK_ELEMENTS", budget)
    results = itemGAM(model, responses, theta, n_grid=9, bandwidth=0.6, se=se)

    for idx, result in enumerate(results):
        valid = np.isfinite(responses[:, idx]) & (responses[:, idx] >= 0)
        observations = responses[valid, idx]
        weights = np.exp(-0.5 * ((theta[valid, None] - result.theta_grid) / 0.6) ** 2)
        weights /= weights.sum(axis=0)
        mean = observations @ weights
        assert_allclose(result.smoothed_probs, mean, atol=1e-14)
        assert_allclose(result.raw_theta, theta[valid])
        assert_allclose(result.raw_probs, observations)
        if se:
            variance = np.maximum(observations**2 @ weights - mean**2, 0.0)
            margin = stats.norm.ppf(0.975) * np.sqrt(
                variance * (weights**2).sum(axis=0)
            )
            maximum = [3, 2][idx] if polytomous else 1
            expected_bands = np.clip([mean - margin, mean + margin], 0.0, maximum)
            assert_allclose(result.se_bands, expected_bands, atol=1e-12)
        else:
            assert_allclose(result.se_bands, 0.0)


@pytest.mark.parametrize("missing", [False, True])
def test_item_gam_shares_bounded_kernels_across_items(monkeypatch, missing) -> None:
    import mirt._smoothing as smoothing

    theta = np.linspace(-1, 1, 31)
    responses = np.random.default_rng(67).integers(0, 2, (31, 5)).astype(float)
    if missing:
        responses[::3, ::2] = np.nan
    shapes = []
    original = smoothing._stable_gaussian_weights

    def track_kernel(samples, grid, *args):
        shapes.append((samples.size, grid.size))
        return original(samples, grid, *args)

    monkeypatch.setattr(smoothing, "_stable_gaussian_weights", track_kernel)
    monkeypatch.setattr(empirical_module, "KERNEL_BLOCK_ELEMENTS", 100)
    itemGAM(_binary_model(5), responses, theta, n_grid=7, bandwidth=0.5)

    assert shapes == [(31, 3), (31, 3), (31, 1)]


@pytest.mark.parametrize("se", [False, True])
def test_item_gam_all_missing_skips_kernel_work(monkeypatch, se) -> None:
    import mirt._smoothing as smoothing

    def unexpected_kernel(*args):
        raise AssertionError("all-missing items do not need kernel evaluation")

    monkeypatch.setattr(smoothing, "_stable_gaussian_weights", unexpected_kernel)
    results = itemGAM(
        _binary_model(2), np.full((5, 2), np.nan), np.linspace(-1, 1, 5), se=se
    )
    for result in results:
        assert np.all(np.isnan(result.smoothed_probs))
        if se:
            assert np.all(np.isnan(result.se_bands))
        else:
            assert_allclose(result.se_bands, 0.0)


@pytest.mark.parametrize("se", [False, True])
def test_item_gam_preserves_all_missing_items_order_and_duplicate_selections(
    se,
) -> None:
    model = _binary_model(3)
    responses = np.array([[0.0, np.nan, 1.0], [1.0, -1.0, 0.0]])
    original = responses.copy()
    results = itemGAM(
        model, responses, np.array([-1.0, 1.0]), item_idx=[2, 1, 0, 2], n_grid=5, se=se
    )

    assert [result.item_idx for result in results] == [2, 1, 0, 2]
    assert np.all(np.isnan(results[1].smoothed_probs))
    assert results[1].raw_theta.size == 0
    if se:
        assert np.all(np.isnan(results[1].se_bands))
    else:
        assert_allclose(results[1].se_bands, 0.0)
    assert_allclose(results[0].smoothed_probs, results[3].smoothed_probs)
    assert_allclose(results[0].se_bands, results[3].se_bands)
    assert_allclose(responses, original)

    empty = itemGAM(
        model, responses, np.array([-1.0, 1.0]), item_idx=1, n_grid=5, se=se
    )
    assert_allclose(empty.smoothed_probs, results[1].smoothed_probs)
    assert_allclose(empty.se_bands, results[1].se_bands)


@pytest.mark.parametrize(
    ("responses", "theta", "message"),
    [
        (np.array([0.0, 1.0]), np.array([-1.0, 1.0]), "2D matrix"),
        (np.zeros((2, 2)), np.array([-1.0, 1.0]), "contain 1 items"),
        (np.zeros((2, 1)), np.array([-1.0]), "same number of persons"),
        (np.zeros((2, 1)), np.array([-1.0, np.nan]), "finite values"),
    ],
)
def test_empirical_plot_validates_shared_shapes(
    responses: np.ndarray, theta: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        empirical_plot(_binary_model(1), responses, theta, item_idx=0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"item_idx": 1}, "between 0 and 0"),
        ({"item_idx": 0, "n_bins": 0}, "at least 1"),
    ],
)
def test_empirical_plot_validates_controls(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        empirical_plot(
            _binary_model(1),
            np.array([[0.0], [1.0]]),
            np.array([-1.0, 1.0]),
            **kwargs,
        )


def test_empirical_plot_rejects_out_of_range_categories() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        empirical_plot(
            _binary_model(1),
            np.array([[0.0], [2.0]]),
            np.array([-1.0, 1.0]),
            item_idx=0,
        )


def test_empirical_diagnostics_reject_multidimensional_theta() -> None:
    model = TwoParameterLogistic(1, n_factors=2)
    with pytest.raises(ValueError, match="unidimensional"):
        empirical_plot(
            model,
            np.array([[0.0], [1.0]]),
            np.array([[-1.0, 0.0], [1.0, 0.0]]),
            item_idx=0,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_grid": 1}, "at least 2"),
        ({"bandwidth": 0.0}, "positive value"),
        ({"alpha": 1.0}, "between 0 and 1"),
        ({"theta_margin": -0.1}, "non-negative"),
    ],
)
def test_item_gam_validates_controls(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        itemGAM(
            _binary_model(1),
            np.array([[0.0], [1.0]]),
            np.array([-1.0, 1.0]),
            item_idx=0,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("function", "kwargs", "message"),
    [
        (empirical_ES, {"n_points": 1}, "at least 2"),
        (RMSD_DIF, {"theta_range": (1.0, -1.0)}, "lower bound"),
        (weighted_RMSD_DIF, {"item_idx": 2}, "between 0 and 0"),
    ],
)
def test_dif_metrics_validate_integration_controls(
    function, kwargs: dict, message: str
) -> None:
    call_kwargs = {"item_idx": 0, **kwargs}
    with pytest.raises(ValueError, match=message):
        function(_binary_model(1), _binary_model(1), **call_kwargs)


def test_empirical_es_validates_focal_weight() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        empirical_ES(_binary_model(1), _binary_model(1), item_idx=0, focal_weight=1.1)


def _mantel_haenszel_data(
    tables: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand (ref correct, ref incorrect, focal correct, focal incorrect)."""
    responses: list[float] = []
    groups: list[int] = []
    theta: list[float] = []
    for stratum, (a, b, c, d) in enumerate(tables):
        responses.extend([1.0] * a + [0.0] * b + [1.0] * c + [0.0] * d)
        groups.extend([0] * (a + b) + [1] * (c + d))
        theta.extend([float(stratum)] * (a + b + c + d))
    return (
        np.asarray(responses)[:, None],
        np.asarray(groups, dtype=np.intp),
        np.asarray(theta),
    )


def _mantel_haenszel_reference(
    tables: list[tuple[int, int, int, int]], correct: bool
) -> tuple[float, float, float]:
    cells = np.asarray(tables, dtype=np.float64)
    a, b, c, d = cells.T
    n_ref = a + b
    n_focal = c + d
    n_total = n_ref + n_focal
    total_correct = a + c
    total_incorrect = b + d
    delta = np.sum(a - n_ref * total_correct / n_total)
    variance = np.sum(
        n_ref * n_focal * total_correct * total_incorrect / (n_total**2 * (n_total - 1))
    )
    continuity = 0.5 if correct and abs(delta) >= 0.5 else 0.0
    chi_square = (abs(delta) - continuity) ** 2 / variance
    odds = np.sum(a * d / n_total) / np.sum(b * c / n_total)
    return chi_square, stats.chi2.sf(chi_square, 1), odds


@pytest.mark.parametrize("correct", [True, False])
def test_mantel_haenszel_matches_stratified_reference(correct: bool) -> None:
    tables = [(6, 4, 3, 7), (5, 5, 4, 6), (7, 3, 5, 5), (4, 6, 2, 8)]
    responses, group, theta = _mantel_haenszel_data(tables)

    result = mantel_haenszel(
        responses, group, theta, item_idx=0, n_strata=4, correct=correct
    )

    assert_allclose(result, _mantel_haenszel_reference(tables, correct))


def test_mantel_haenszel_omits_correction_when_delta_is_small() -> None:
    tables = [(1, 2, 1, 3)]
    responses, group, theta = _mantel_haenszel_data(tables)

    corrected = mantel_haenszel(responses, group, theta, 0, n_strata=1)
    uncorrected = mantel_haenszel(responses, group, theta, 0, n_strata=1, correct=False)

    assert_allclose(corrected, uncorrected)
    assert corrected[0] > 0


def test_mantel_haenszel_treats_negative_and_nan_responses_as_missing() -> None:
    responses, group, theta = _mantel_haenszel_data([(6, 4, 3, 7), (5, 5, 4, 6)])
    baseline = mantel_haenszel(responses, group, theta, 0, n_strata=2)
    responses = np.vstack([responses, [[-1.0], [np.nan]]])
    group = np.concatenate([group, [0, 1]])
    theta = np.concatenate([theta, [-10.0, 10.0]])

    result = mantel_haenszel(responses, group, theta, 0, n_strata=2)

    assert_allclose(result, baseline)


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ((2, 0, 1, 1), np.inf),
        ((0, 2, 1, 1), 0.0),
        ((2, 0, 2, 0), np.nan),
    ],
)
def test_mantel_haenszel_reports_boundary_odds_ratios(
    table: tuple[int, int, int, int], expected: float
) -> None:
    responses, group, theta = _mantel_haenszel_data([table])

    odds = mantel_haenszel(responses, group, theta, 0, n_strata=1)[2]

    if np.isnan(expected):
        assert np.isnan(odds)
    else:
        assert odds == expected


def test_mantel_haenszel_uses_two_grouped_reductions(monkeypatch) -> None:
    rng = np.random.default_rng(20260827)
    n_persons = 20_000
    responses = rng.integers(0, 2, size=(n_persons, 1)).astype(float)
    group = rng.integers(0, 2, size=n_persons, dtype=np.intp)
    theta = rng.normal(size=n_persons)
    original_bincount = np.bincount
    calls = 0

    def counting_bincount(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_bincount(*args, **kwargs)

    monkeypatch.setattr(empirical_module.np, "bincount", counting_bincount)

    mantel_haenszel(responses, group, theta, 0, n_strata=100)

    assert calls == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"group": np.array([0, 2, 0, 1])}, "only 0 .* and 1"),
        ({"theta": np.array([0.0, 1.0, np.nan, 3.0])}, "finite values"),
        ({"responses": np.array([[0.0], [1.0], [2.0], [0.0]])}, "binary"),
        ({"group": np.array([0, 1, 0])}, "same number of persons"),
        ({"n_strata": 0}, "at least 1"),
        ({"correct": 1}, "boolean"),
    ],
)
def test_mantel_haenszel_validates_inputs(kwargs: dict, message: str) -> None:
    arguments = {
        "responses": np.array([[0.0], [1.0], [1.0], [0.0]]),
        "group": np.array([0, 1, 0, 1]),
        "theta": np.arange(4.0),
        "item_idx": 0,
        **kwargs,
    }

    with pytest.raises(ValueError, match=message):
        mantel_haenszel(**arguments)
