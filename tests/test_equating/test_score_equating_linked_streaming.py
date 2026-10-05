"""Scale invariance and exact probability oracles for score equating."""

import tracemalloc
from itertools import product

import numpy as np
import pytest
from scipy.special import expit

from mirt import get_backend, is_rust_available, set_backend
from mirt.equating import (
    LinkingConstants,
    LinkingResult,
    link,
    lord_wingersky_recursion,
    observed_score_equating,
    true_score_equating,
)
from mirt.equating import score_equating as implementation
from mirt.models import (
    FourParameterLogistic,
    GeneralizedPartialCredit,
    GradedResponseModel,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)


@pytest.fixture(params=["numpy", "rust"])
def backend(request):
    if request.param == "rust" and not is_rust_available():
        pytest.skip("compiled backend unavailable")
    previous = get_backend()
    set_backend(request.param)
    try:
        yield request.param
    finally:
        set_backend(previous)


def _enumerated_distribution(probabilities, weights):
    """Sum all response-pattern probabilities; no score recursion involved."""
    weights = weights / weights.sum()
    result = np.zeros(sum(p.shape[1] - 1 for p in probabilities) + 1)
    for responses in product(*(range(p.shape[1]) for p in probabilities)):
        pattern = np.ones(len(weights))
        for probability, response in zip(probabilities, responses, strict=True):
            pattern *= probability[:, response]
        result[sum(responses)] += weights @ pattern
    return result


def _kolen_brennan_equivalents(old, new):
    """Transcribe Kolen and Brennan (2014, eqs. 2.14-2.18) score by score."""
    cumulative_new = np.cumsum(new)
    equivalents = []
    for score in range(len(old)):
        rank = np.sum(old[:score]) + old[score] / 2
        (above,) = np.nonzero(cumulative_new > rank)
        cell = above[0]
        below = cumulative_new[cell - 1] if cell else 0.0
        equivalents.append(cell - 0.5 + (rank - below) / new[cell])
    return np.array(equivalents)


def _equivalent_forms(model_type):
    a = np.array([0.7, 1.1, 1.4, 1.8])
    b = np.array([-1.5, -0.3, 0.6, 1.4])
    A, B = 1.7, 0.9
    if model_type in (GradedResponseModel, GeneralizedPartialCredit):
        old = model_type(4, n_categories=[3, 4, 2, 3])
        new = model_type(4, n_categories=[3, 4, 2, 3])
        key = "thresholds" if model_type is GradedResponseModel else "steps"
        thresholds = np.array(
            [[-1.5, 0.2, 0.0], [-1.2, 0.4, 1.6], [0.3, 0.0, 0.0], [-0.1, 1.2, 0.0]]
        )
        old.set_parameters(discrimination=a, **{key: thresholds})
        new.set_parameters(discrimination=A * a, **{key: (thresholds - B) / A})
    else:
        old, new = model_type(4), model_type(4)
        asymptotes = {}
        if model_type in (ThreeParameterLogistic, FourParameterLogistic):
            asymptotes["guessing"] = np.array([0.1, 0.2, 0.15, 0.25])
        if model_type is FourParameterLogistic:
            asymptotes["upper"] = np.array([0.9, 0.8, 0.95, 0.85])
        old.set_parameters(discrimination=a, difficulty=b, **asymptotes)
        new.set_parameters(discrimination=A * a, difficulty=(b - B) / A, **asymptotes)
    linking = LinkingResult(LinkingConstants(A, B), [0, 1, 2, 3])
    return old, new, linking


@pytest.mark.parametrize(
    "model_type",
    [
        TwoParameterLogistic,
        ThreeParameterLogistic,
        FourParameterLogistic,
        GradedResponseModel,
        GeneralizedPartialCredit,
    ],
)
def test_equivalent_calibrations_have_identical_score_tables(model_type, backend):
    old, new, linking = _equivalent_forms(model_type)
    linked_true = true_score_equating(old, new, linking_result=linking)
    reference_true = true_score_equating(old, old)
    np.testing.assert_allclose(
        linked_true.new_scores, reference_true.new_scores, atol=1e-12
    )

    theta = np.array([-2.0, -0.3, 0.1, 0.9, 2.5])
    masses = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    linked_observed = observed_score_equating(
        old,
        new,
        theta_grid=theta,
        theta_distribution=masses,
        linking_result=linking,
        batch_size=2,
    )
    np.testing.assert_allclose(
        linked_observed.new_scores, linked_observed.old_scores, atol=1e-12
    )
    np.testing.assert_array_equal(linked_observed.theta, theta)


def test_estimated_linking_constants_flow_into_true_score_equating(backend):
    old, new, known = _equivalent_forms(TwoParameterLogistic)
    estimated = link(old, new, [0, 1, 2, 3], [0, 1, 2, 3], method="mean_mean")
    assert estimated.constants.A == pytest.approx(known.constants.A)
    assert estimated.constants.B == pytest.approx(known.constants.B)
    result = true_score_equating(old, new, linking_result=estimated)
    reference = true_score_equating(old, old)
    np.testing.assert_allclose(result.new_scores, reference.new_scores, atol=1e-12)


def test_representable_linked_abilities_survive_subtraction_overflow(backend):
    old = TwoParameterLogistic(1)
    old.set_parameters(discrimination=np.array([1e-308]), difficulty=np.array([0.0]))
    new = TwoParameterLogistic(1)
    new.set_parameters(discrimination=np.array([2e-308]), difficulty=np.array([5e307]))
    linking = LinkingResult(LinkingConstants(2.0, -1e308), [0])
    theta = np.array([9e307, 1e308])
    observed = observed_score_equating(
        old,
        new,
        theta_grid=theta,
        theta_distribution=np.ones(2),
        linking_result=linking,
    )
    np.testing.assert_allclose(observed.new_scores, [0.0, 1.0], atol=1e-14)
    actual = true_score_equating(
        old, new, linking_result=linking, theta_range=(9e307, 1e308), n_theta=3
    )
    reference = true_score_equating(old, old, theta_range=(9e307, 1e308), n_theta=3)
    np.testing.assert_allclose(actual.new_scores, reference.new_scores, atol=1e-14)


def test_observed_linking_preserves_reference_population_masses(backend):
    old, new, linking = _equivalent_forms(TwoParameterLogistic)
    # The new form is deliberately easier even after placing it on the old scale.
    new.set_parameters(difficulty=new.parameters["difficulty"] - 0.35)
    theta = np.array([-2.0, -1.0, 0.2, 1.0, 2.0])
    masses = np.array([1.0, 7.0, 2.0, 3.0, 1.0])
    items_old, items_new = [0, 3], [2, 0, 1]
    A, B = linking.constants.A, linking.constants.B
    probability_old = expit(
        old.discrimination[items_old] * (theta[:, None] - old.difficulty[items_old])
    )
    probability_new = expit(
        new.discrimination[items_new]
        * ((theta[:, None] - B) / A - new.difficulty[items_new])
    )
    distributions = [
        _enumerated_distribution(
            [np.column_stack((1 - p, p)) for p in probability.T], masses
        )
        for probability in (probability_old, probability_new)
    ]
    expected = _kolen_brennan_equivalents(*distributions)

    result = observed_score_equating(
        old,
        new,
        theta_grid=theta,
        theta_distribution=masses,
        items_old=items_old,
        items_new=items_new,
        linking_result=linking,
        batch_size=2,
    )
    np.testing.assert_allclose(result.new_scores, expected, atol=1e-13)


@pytest.mark.parametrize("batch_size", [1, 3, 100, None])
def test_batched_recursion_matches_exhaustive_heterogeneous_categories(
    batch_size, backend
):
    model, _, _ = _equivalent_forms(GradedResponseModel)
    theta = np.linspace(-2, 2, 8)
    masses = np.array([0.0, 0.0, 0.0, 2.0, 1.0, 5.0, 3.0, 0.0])
    probabilities = []
    for a, thresholds, count in zip(
        model.discrimination,
        model.parameters["thresholds"],
        model.n_categories,
        strict=True,
    ):
        cumulative = np.column_stack(
            (
                np.ones(len(theta)),
                expit(a * (theta[:, None] - thresholds[: count - 1])),
                np.zeros(len(theta)),
            )
        )
        probabilities.append(cumulative[:, :-1] - cumulative[:, 1:])
    expected = _enumerated_distribution(probabilities, masses)
    actual = lord_wingersky_recursion(model, theta, masses, batch_size=batch_size)
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_recursion_bounds_public_probability_evaluation(monkeypatch):
    model, _, _ = _equivalent_forms(ThreeParameterLogistic)
    original = model.probability
    evaluated = []

    def probability(theta):
        evaluated.append(theta[:, 0].copy())
        return original(theta)

    monkeypatch.setattr(model, "probability", probability)
    theta = np.linspace(-3, 3, 11)
    actual = lord_wingersky_recursion(model, theta, np.ones(11), batch_size=3)
    assert max(len(values) for values in evaluated) <= 3
    np.testing.assert_array_equal(np.concatenate(evaluated), theta)
    assert actual.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("override", ["instance", "subclass", "class", "renamed-3PL"])
def test_compiled_recursion_respects_public_model_curves(monkeypatch, override):
    class ConstantCurve(TwoParameterLogistic):
        def probability(self, theta, item_idx=None):
            return np.full((len(theta), self.n_items), 0.8)

    if override == "subclass":
        model = ConstantCurve(3)
        correct = 0.8
    elif override == "renamed-3PL":
        model = ThreeParameterLogistic(3)
        model.model_name = "2PL"
        model.set_parameters(discrimination=np.zeros(3), guessing=np.full(3, 0.2))
        correct = 0.6
    else:
        model = TwoParameterLogistic(3)
        if override == "instance":
            monkeypatch.setattr(
                model, "probability", lambda theta: np.full((len(theta), 3), 0.8)
            )
        else:
            monkeypatch.setattr(
                TwoParameterLogistic, "probability", ConstantCurve.probability
            )
        correct = 0.8

    def unexpected_native(*args):
        pytest.fail("custom curve reached built-in compiled recursion")

    monkeypatch.setattr(
        implementation, "_rust_observed_score_distribution_2pl", unexpected_native
    )
    theta = np.array([-1.0, 0.0, 1.0])
    probabilities = [np.tile([1 - correct, correct], (len(theta), 1))] * 3
    expected = _enumerated_distribution(probabilities, np.ones(3))
    actual = lord_wingersky_recursion(model, theta, np.ones(3), batch_size=2)
    np.testing.assert_allclose(actual, expected, atol=1e-14)


@pytest.mark.parametrize("batch_size", [0, -1, 2.5, True])
def test_invalid_batch_sizes_fail_before_evaluating_curves(monkeypatch, batch_size):
    model = TwoParameterLogistic(2)
    monkeypatch.setattr(
        model, "probability", lambda *args: pytest.fail("curve evaluated")
    )
    with pytest.raises(ValueError, match="batch_size"):
        lord_wingersky_recursion(
            model, np.array([0.0]), np.ones(1), batch_size=batch_size
        )


@pytest.mark.parametrize(
    "A, B", [(0.0, 0.0), (-1.0, 0.0), (np.nan, 0.0), (1.0, np.inf), (1e-310, 1.0)]
)
def test_observed_equating_rejects_invalid_scale_transforms(A, B):
    model = TwoParameterLogistic(2)
    linking = LinkingResult(LinkingConstants(A, B), [0, 1])
    with pytest.raises(ValueError, match="linking_result"):
        observed_score_equating(model, model, linking_result=linking)


@pytest.mark.performance
def test_default_recursion_memory_does_not_scale_with_the_full_grid(backend):
    model = ThreeParameterLogistic(80)
    model.set_parameters(
        discrimination=np.linspace(0.8, 1.5, 80),
        difficulty=np.linspace(-2, 2, 80),
        guessing=np.full(80, 0.15),
    )
    peaks = []
    for size in (1000, 10000):
        theta = np.linspace(-4, 4, size)
        weights = np.exp(-0.5 * theta**2)
        tracemalloc.start()
        try:
            distribution = lord_wingersky_recursion(model, theta, weights)
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
        assert distribution.sum() == pytest.approx(1.0)
        probability = 0.15 + 0.85 * expit(
            model.discrimination * (theta[:, None] - model.difficulty)
        )
        expected_mean = (weights / weights.sum()) @ probability.sum(axis=1)
        assert np.arange(81) @ distribution == pytest.approx(expected_mean, rel=1e-12)
    # A tenfold grid increase must not retain the full theta-by-score matrix.
    assert peaks[1] < 2 * peaks[0]
