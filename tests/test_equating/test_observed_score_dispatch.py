"""Regression tests for accelerated observed-score equating."""

from math import comb

import numpy as np
import pytest

import mirt
from mirt._rust_backend import RUST_AVAILABLE
from mirt.backends.rust import equating as rust_equating
from mirt.equating import LinkingConstants, LinkingResult, score_equating
from mirt.equating.score_equating import (
    equipercentile_equating,
    lord_wingersky_recursion,
    observed_score_equating,
)
from mirt.models.dichotomous import (
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)


@pytest.fixture
def two_pl() -> TwoParameterLogistic:
    model = TwoParameterLogistic(4)
    model.set_parameters(
        discrimination=np.array([0.7, 1.0, 1.3, 1.6]),
        difficulty=np.array([-1.0, -0.2, 0.5, 1.2]),
    )
    return model


def test_lord_wingersky_dispatches_compatible_item_subsets(monkeypatch, two_pl):
    theta = np.array([-1.0, 0.5])
    weights = np.array([0.25, 0.75])
    conditional = np.array(
        [
            [0.50, 0.35, 0.15],
            [0.20, 0.45, 0.35],
        ]
    )
    call = {}

    def fake_native(theta_values, discrimination, difficulty):
        call["theta"] = theta_values.copy()
        call["discrimination"] = discrimination.copy()
        call["difficulty"] = difficulty.copy()
        return conditional

    monkeypatch.setattr(
        score_equating, "_rust_observed_score_distribution_2pl", fake_native
    )

    actual = lord_wingersky_recursion(two_pl, theta, weights, items=[3, 1])

    np.testing.assert_array_equal(call["theta"], theta)
    np.testing.assert_array_equal(call["discrimination"], [1.6, 1.0])
    np.testing.assert_array_equal(call["difficulty"], [1.2, -0.2])
    np.testing.assert_allclose(actual, weights @ conditional, atol=1e-15)


def test_lord_wingersky_retains_numpy_fallback(monkeypatch, two_pl):
    theta = np.linspace(-2.0, 2.0, 7)
    weights = np.arange(1.0, 8.0)
    monkeypatch.setattr(
        score_equating,
        "_rust_observed_score_distribution_2pl",
        lambda *args: None,
    )

    actual = lord_wingersky_recursion(two_pl, theta, weights)

    probabilities = two_pl.probability(theta[:, None])
    conditional = np.ones((len(theta), 1))
    for probability in probabilities.T:
        updated = np.zeros((len(theta), conditional.shape[1] + 1))
        updated[:, :-1] += conditional * (1.0 - probability[:, None])
        updated[:, 1:] += conditional * probability[:, None]
        conditional = updated
    normalized_weights = weights / weights.sum()
    expected = normalized_weights @ conditional
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_lord_wingersky_does_not_dispatch_unsupported_models(monkeypatch):
    model = ThreeParameterLogistic(3)
    theta = np.linspace(-2.0, 2.0, 5)
    weights = np.ones(5)

    def unexpected_dispatch(*args):
        raise AssertionError("unsupported model reached compiled dispatch")

    monkeypatch.setattr(
        score_equating,
        "_rust_observed_score_distribution_2pl",
        unexpected_dispatch,
    )

    actual = lord_wingersky_recursion(model, theta, weights)

    assert actual.shape == (4,)
    assert actual.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("override", ["instance", "class", "parent", "subclass"])
def test_lord_wingersky_preserves_custom_probability_hooks(monkeypatch, override):
    correct = 0.22

    def custom_probability(self, theta, item_idx=None):
        shape = (len(theta), self.n_items) if item_idx is None else (len(theta),)
        return np.full(shape, correct)

    if override == "subclass":

        class CustomTwoPL(TwoParameterLogistic):
            probability = custom_probability

        model = CustomTwoPL(3)
    else:
        model = TwoParameterLogistic(3)
        if override == "instance":
            monkeypatch.setattr(
                model,
                "probability",
                lambda theta, item_idx=None: custom_probability(model, theta, item_idx),
            )
        elif override == "class":
            monkeypatch.setattr(TwoParameterLogistic, "probability", custom_probability)
        else:
            # Changing an inherited hook keeps the public probability method
            # intact, and must still rule out a compiled numerical shortcut.
            monkeypatch.setattr(
                TwoParameterLogistic.__bases__[0],
                "_evaluate_logistic",
                custom_probability,
            )

    def unexpected_dispatch(*args):
        raise AssertionError("custom model reached compiled dispatch")

    monkeypatch.setattr(
        score_equating,
        "_rust_observed_score_distribution_2pl",
        unexpected_dispatch,
    )
    actual = lord_wingersky_recursion(model, np.array([-1.0, 0.5]), np.ones(2))
    expected = np.array(
        [
            comb(3, score) * correct**score * (1.0 - correct) ** (3 - score)
            for score in range(4)
        ]
    )

    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_lord_wingersky_ignores_spoofed_model_display_name(monkeypatch):
    model = ThreeParameterLogistic(1)
    model.set_parameters(guessing=np.array([0.2]))
    model.model_name = "2PL"

    def unexpected_dispatch(*args):
        raise AssertionError("3PL model reached compiled 2PL dispatch")

    monkeypatch.setattr(
        score_equating,
        "_rust_observed_score_distribution_2pl",
        unexpected_dispatch,
    )
    actual = lord_wingersky_recursion(model, np.array([0.0]), np.ones(1))

    np.testing.assert_allclose(actual, [0.4, 0.6], atol=1e-15)


@pytest.mark.parametrize("model_type", [OneParameterLogistic, TwoParameterLogistic])
def test_lord_wingersky_dispatches_renamed_builtin_models(monkeypatch, model_type):
    model = model_type(1)
    model.model_name = "renamed form"
    calls = []

    def native(theta, discrimination, difficulty):
        calls.append((theta, discrimination, difficulty))
        return np.array([[0.25, 0.75]])

    monkeypatch.setattr(score_equating, "_rust_observed_score_distribution_2pl", native)
    actual = lord_wingersky_recursion(model, np.array([0.5]), np.ones(1))

    assert len(calls) == 1
    np.testing.assert_allclose(actual, [0.25, 0.75], atol=1e-15)


@pytest.mark.parametrize("smoothing", ["loglinear", "kernel"])
def test_observed_score_equating_exposes_smoothing(two_pl, smoothing):
    theta = np.linspace(-3.0, 3.0, 31)
    weights = np.exp(-0.5 * theta**2)
    distribution = lord_wingersky_recursion(two_pl, theta, weights)

    actual = observed_score_equating(
        two_pl,
        two_pl,
        theta_grid=theta,
        theta_distribution=weights,
        smoothing=smoothing,
    )
    expected = equipercentile_equating(
        distribution,
        distribution,
        smoothing=smoothing,
    )

    np.testing.assert_allclose(actual.new_scores, expected, atol=1e-15)


def test_observed_score_equating_rejects_unknown_smoothing(two_pl):
    with pytest.raises(ValueError, match="smoothing must be one of"):
        observed_score_equating(two_pl, two_pl, smoothing="unknown")  # type: ignore[arg-type]


def test_backend_wrapper_returns_none_when_dispatch_is_disabled(monkeypatch):
    monkeypatch.setattr(rust_equating, "rust_enabled", lambda: False)

    actual = rust_equating.observed_score_distribution_2pl(
        np.array([-1.0, 1.0]),
        np.array([1.0, 1.5]),
        np.array([-0.5, 0.5]),
    )

    assert actual is None


def test_backend_wrapper_matches_direct_probability_recursion(two_pl):
    theta = np.linspace(-3.0, 3.0, 11)
    conditional = rust_equating.observed_score_distribution_2pl(
        theta,
        two_pl.discrimination,
        two_pl.difficulty,
    )
    if conditional is None:
        pytest.skip("compiled backend unavailable")

    probabilities = two_pl.probability(theta[:, None])
    expected = np.ones((len(theta), 1))
    for probability in probabilities.T:
        updated = np.zeros((len(theta), expected.shape[1] + 1))
        updated[:, :-1] += expected * (1.0 - probability[:, None])
        updated[:, 1:] += expected * probability[:, None]
        expected = updated

    np.testing.assert_allclose(conditional, expected, atol=1e-15, rtol=1e-15)


@pytest.mark.parametrize("backend", ["numpy", "rust"])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("slope", [5e-309, 1e-308])
def test_score_distribution_recovers_finite_logits_after_difference_overflow(
    backend, direction, slope
):
    if backend == "rust" and not RUST_AVAILABLE:
        pytest.skip("compiled backend unavailable")
    model = TwoParameterLogistic(1)
    model.set_parameters(
        discrimination=np.array([slope]),
        difficulty=np.array([-direction * 1e308]),
    )
    previous_backend = mirt.get_backend()
    mirt.set_backend(backend)
    try:
        actual = lord_wingersky_recursion(
            model, np.array([direction * 1e308]), np.ones(1)
        )
    finally:
        mirt.set_backend(previous_backend)

    # The exact difference is +/-2e308, whose float representation overflows.
    # Its product with the small discrimination has a finite logit of +/-1 or +/-2.
    logit = direction * 2.0 * (slope * 1e308)
    correct = 1.0 / (1.0 + np.exp(-logit))
    np.testing.assert_allclose(actual, [1.0 - correct, correct], atol=1e-15, rtol=1e-14)


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_linked_observed_scores_preserve_population_at_extreme_finite_coordinates(
    backend,
):
    if backend == "rust" and not RUST_AVAILABLE:
        pytest.skip("compiled backend unavailable")
    old, new = TwoParameterLogistic(3), TwoParameterLogistic(3)
    old.set_parameters(difficulty=np.full(3, -1.0))
    new.set_parameters(discrimination=np.full(3, 1e-308), difficulty=np.full(3, -1e308))
    linking = LinkingResult(LinkingConstants(A=1e-308, B=0.0), anchor_items=[])
    previous_backend = mirt.get_backend()
    mirt.set_backend(backend)
    try:
        actual = observed_score_equating(
            old,
            new,
            theta_grid=np.array([-1.0, 0.0, 1.0]),
            theta_distribution=np.array([1.0, 2.0, 3.0]),
            linking_result=linking,
        )
    finally:
        mirt.set_backend(previous_backend)

    np.testing.assert_allclose(
        actual.new_scores, actual.old_scores, atol=1e-15, rtol=1e-14
    )


@pytest.mark.parametrize(
    ("theta", "discrimination", "difficulty", "message"),
    [
        (np.empty(0), np.ones(2), np.zeros(2), "theta must be"),
        (np.zeros((1, 1)), np.ones(2), np.zeros(2), "theta must be"),
        (np.zeros(2), np.empty(0), np.empty(0), "discrimination must be"),
        (np.zeros(2), np.ones(2), np.zeros(3), "difficulty must match"),
        (np.array([0.0, np.nan]), np.ones(2), np.zeros(2), "only finite"),
    ],
)
def test_backend_wrapper_validates_inputs(theta, discrimination, difficulty, message):
    with pytest.raises(ValueError, match=message):
        rust_equating.observed_score_distribution_2pl(theta, discrimination, difficulty)
