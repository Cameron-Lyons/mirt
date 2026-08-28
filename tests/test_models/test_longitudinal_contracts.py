"""Contracts for longitudinal binary and graded-response models."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mirt.models.dynamic as dynamic_module
from mirt.models.dynamic import LongitudinalIRTModel


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def test_grm_defaults_create_ordered_category_thresholds() -> None:
    model = LongitudinalIRTModel(
        n_items=4,
        n_timepoints=3,
        base_model="GRM",
    )

    assert model.n_categories == 4
    assert model.thresholds is not None
    assert model.thresholds.shape == (4, 3)
    assert np.all(np.diff(model.thresholds, axis=1) > 0.0)
    assert "Response Categories: 4" in model.summary()

    two_category = LongitudinalIRTModel(
        n_items=1,
        n_timepoints=2,
        base_model="GRM",
        n_categories=2,
    )
    assert_array_equal(two_category.thresholds, np.zeros((1, 1)))


def test_grm_probabilities_match_cumulative_curve_reference() -> None:
    discrimination = np.array([0.8, 1.4])
    thresholds = np.array(
        [
            [-1.5, -0.25, 1.0],
            [-1.0, 0.2, 1.6],
        ]
    )
    model = LongitudinalIRTModel(
        n_items=2,
        n_timepoints=3,
        base_model="GRM",
        discrimination=discrimination,
        thresholds=thresholds,
    )
    theta = np.array([-2.0, 0.0, 2.0])

    actual = model.probability(theta)
    cumulative = _sigmoid(
        discrimination[None, :, None] * (theta[:, None, None] - thresholds[None, :, :])
    )
    boundaries = np.concatenate(
        (
            np.ones((theta.size, model.n_items, 1)),
            cumulative,
            np.zeros((theta.size, model.n_items, 1)),
        ),
        axis=2,
    )
    expected = boundaries[:, :, :-1] - boundaries[:, :, 1:]

    assert actual.shape == (3, 2, 4)
    assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert_allclose(actual.sum(axis=2), 1.0, atol=1e-14)
    assert np.all(actual >= 0.0)
    assert_allclose(model.probability(theta, item_idx=1), actual[:, 1, :])

    category_scores = np.arange(model.n_categories, dtype=np.float64)
    expected_scores = actual @ category_scores
    assert np.all(np.diff(expected_scores, axis=0) > 0.0)


def test_grm_thresholds_infer_category_count() -> None:
    thresholds = np.array(
        [
            [-1.5, -0.5, 0.5, 1.5],
            [-1.25, -0.25, 0.75, 1.75],
        ]
    )

    model = LongitudinalIRTModel(
        n_items=2,
        n_timepoints=2,
        base_model="GRM",
        thresholds=thresholds,
    )

    assert model.n_categories == 5
    assert_array_equal(model.thresholds, thresholds)
    assert model.probability(np.array([0.0])).shape == (1, 2, 5)


def test_grm_simulation_is_polytomous_and_reproducible() -> None:
    model = LongitudinalIRTModel(
        n_items=6,
        n_timepoints=4,
        base_model="GRM",
        n_categories=5,
        discrimination=np.linspace(0.7, 1.6, 6),
    )

    first = model.simulate(250, seed=814)
    second = model.simulate(250, seed=814)
    responses, theta, factors = first

    assert responses.shape == (250, 4, 6)
    assert theta.shape == (250, 4)
    assert factors.shape == (250, 2)
    assert np.min(responses) == 0
    assert np.max(responses) == 4
    assert set(np.unique(responses)).issubset(set(range(5)))
    for left, right in zip(first, second):
        assert_array_equal(left, right)


def test_vectorized_linear_theta_matches_growth_equation() -> None:
    model = LongitudinalIRTModel(n_items=3, n_timepoints=4)
    growth_factors = np.array(
        [
            [0.2, 0.5],
            [-0.8, 0.1],
            [1.5, -0.25],
        ]
    )
    times = np.array([0.0, 0.5, 2.0, 4.0])
    residuals = np.array(
        [
            [0.1, 0.0, -0.1, 0.2],
            [0.0, 0.1, 0.2, 0.3],
            [-0.1, -0.2, 0.0, 0.1],
        ]
    )

    actual = model.compute_theta(growth_factors, times, residuals)
    expected = (
        growth_factors[:, :1] + growth_factors[:, 1:] * times[None, :] + residuals
    )

    assert_allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_vectorized_quadratic_theta_matches_growth_equation() -> None:
    model = LongitudinalIRTModel(
        n_items=3,
        n_timepoints=5,
        growth_model="quadratic",
    )
    growth_factors = np.array([[0.2, 0.5, -0.1], [-0.8, 0.1, 0.03]])
    times = np.array([-1.0, 0.0, 0.5, 2.0, 4.0])

    actual = model.compute_theta(growth_factors, times)
    expected = (
        growth_factors[:, :1]
        + growth_factors[:, 1:2] * times[None, :]
        + growth_factors[:, 2:] * times[None, :] ** 2
    )

    assert_allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_vectorized_2pl_simulation_preserves_seeded_draw_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = LongitudinalIRTModel(
        n_items=7,
        n_timepoints=4,
        discrimination=np.linspace(0.7, 1.6, 7),
        difficulty=np.linspace(-1.0, 1.0, 7),
        growth_mean=np.array([0.2, 0.1]),
        growth_cov=np.array([[1.0, 0.15], [0.15, 0.3]]),
        residual_variance=0.2,
    )
    times = np.array([0.0, 0.5, 1.5, 3.0])
    seed = 914
    n_persons = 25
    monkeypatch.setattr(dynamic_module, "_LONGITUDINAL_MAX_PROBABILITY_VALUES", 20)

    rng = np.random.default_rng(seed)
    expected_factors = rng.multivariate_normal(
        model.growth_mean,
        model.growth_cov,
        size=n_persons,
    )
    residuals = rng.normal(
        0.0,
        np.sqrt(model.residual_variance),
        size=(n_persons, model.n_timepoints),
    )
    expected_theta = (
        expected_factors[:, :1] + expected_factors[:, 1:] * times[None, :] + residuals
    )
    logits = model.discrimination[None, None, :] * (
        expected_theta[:, :, None] - model.difficulty[None, None, :]
    )
    expected_responses = (rng.random(logits.shape) < _sigmoid(logits)).astype(np.int32)

    responses, theta, factors = model.simulate(
        n_persons,
        time_values=times,
        seed=seed,
    )

    assert_array_equal(responses, expected_responses)
    assert_allclose(theta, expected_theta, rtol=0.0, atol=1e-14)
    assert_array_equal(factors, expected_factors)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_items": 0}, "n_items"),
        ({"n_timepoints": True}, "n_timepoints"),
        ({"base_model": "3PL"}, "base_model"),
        ({"growth_model": "cubic"}, "growth_model"),
        ({"item_names": ["duplicate", "duplicate"]}, "unique"),
        ({"discrimination": np.array([1.0, -0.5])}, "positive"),
        ({"difficulty": np.array([0.0, np.nan])}, "difficulty"),
        ({"growth_mean": np.zeros(3)}, "growth_mean"),
        ({"growth_cov": np.array([[1.0, 0.5], [0.0, 1.0]])}, "symmetric"),
        ({"growth_cov": np.array([[1.0, 2.0], [2.0, 1.0]])}, "semidefinite"),
        ({"residual_variance": -0.1}, "residual_variance"),
        ({"n_categories": 3}, "exactly 2"),
        ({"thresholds": np.zeros((2, 2))}, "only supported"),
        (
            {
                "base_model": "GRM",
                "thresholds": np.array([[0.0, -1.0], [-0.5, 0.5]]),
            },
            "strictly increasing",
        ),
        (
            {
                "base_model": "GRM",
                "n_categories": 4,
                "thresholds": np.array([[-0.5, 0.5], [-0.5, 0.5]]),
            },
            "does not match",
        ),
    ],
)
def test_invalid_longitudinal_configuration_is_rejected(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {"n_items": 2, "n_timepoints": 3}
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        LongitudinalIRTModel(**values)


@pytest.mark.parametrize(
    ("growth_factors", "times", "residuals", "message"),
    [
        (np.zeros((2, 3)), None, None, "growth_factors"),
        (np.zeros((2, 2)), np.zeros(2), None, "time_values"),
        (np.zeros((2, 2)), None, np.zeros((2, 2)), "residuals"),
        (np.array([[0.0, np.nan]]), None, None, "finite"),
    ],
)
def test_compute_theta_rejects_invalid_shapes_and_values(
    growth_factors: np.ndarray,
    times: np.ndarray | None,
    residuals: np.ndarray | None,
    message: str,
) -> None:
    model = LongitudinalIRTModel(n_items=2, n_timepoints=3)

    with pytest.raises(ValueError, match=message):
        model.compute_theta(growth_factors, times, residuals)


@pytest.mark.parametrize("item_idx", [-1, 2, True])
def test_probability_rejects_invalid_item_indices(item_idx: int) -> None:
    model = LongitudinalIRTModel(n_items=2, n_timepoints=3)

    with pytest.raises(ValueError, match="item_idx"):
        model.probability(np.array([0.0]), item_idx=item_idx)


def test_probability_and_simulation_validate_runtime_inputs() -> None:
    model = LongitudinalIRTModel(n_items=2, n_timepoints=3)

    with pytest.raises(ValueError, match="one-dimensional"):
        model.probability(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        model.probability(np.array([np.nan]))
    with pytest.raises(ValueError, match="n_persons"):
        model.simulate(0)
    with pytest.raises(ValueError, match="time_values"):
        model.simulate(2, time_values=np.zeros(2))
