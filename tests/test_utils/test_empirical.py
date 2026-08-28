"""Regression tests for empirical item diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

import mirt.utils.empirical as empirical_module
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
