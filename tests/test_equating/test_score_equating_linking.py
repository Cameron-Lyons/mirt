"""Score equating oracles for independently calibrated equivalent forms."""

import numpy as np
import pytest

import mirt
from mirt._rust_backend import RUST_AVAILABLE
from mirt.equating import (
    LinkingConstants,
    LinkingResult,
    link,
    observed_score_equating,
    true_score_equating,
)
from mirt.models.dichotomous import ThreeParameterLogistic, TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel


def _equivalent_forms(model_type, A: float, B: float):
    """Apply an independently specified change of latent ability coordinates."""
    discrimination = np.array([0.7, 1.0, 1.3, 1.6])
    difficulty = np.array([-1.2, -0.3, 0.4, 1.5])
    if model_type is GradedResponseModel:
        old = model_type(4, n_categories=[3, 4, 3, 4])
        new = model_type(4, n_categories=[3, 4, 3, 4])
        thresholds = np.array(
            [[-1.5, 0.5, 0.0], [-1.2, 0.0, 1.3], [-0.5, 1.5, 0.0], [-1.0, 0.4, 1.8]]
        )
        old.set_parameters(discrimination=discrimination, thresholds=thresholds)
        new.set_parameters(
            discrimination=A * discrimination,
            thresholds=(thresholds - B) / A,
        )
    else:
        old = model_type(4)
        new = model_type(4)
        asymptotes = (
            {"guessing": np.array([0.1, 0.15, 0.2, 0.25])}
            if model_type is ThreeParameterLogistic
            else {}
        )
        old.set_parameters(
            discrimination=discrimination, difficulty=difficulty, **asymptotes
        )
        new.set_parameters(
            discrimination=A * discrimination,
            difficulty=(difficulty - B) / A,
            **asymptotes,
        )
    linking = LinkingResult(LinkingConstants(A=A, B=B), anchor_items=list(range(4)))
    return old, new, linking


@pytest.mark.parametrize(
    "model_type", [TwoParameterLogistic, ThreeParameterLogistic, GradedResponseModel]
)
@pytest.mark.parametrize(("A", "B"), [(1.4, -0.7), (0.6, 0.8)])
@pytest.mark.parametrize("items", [None, [3, 1, 0]])
def test_true_scores_are_invariant_to_calibration_coordinates(model_type, A, B, items):
    old, new, linking = _equivalent_forms(model_type, A, B)
    baseline = true_score_equating(old, old, items_old=items, items_new=items)

    actual = true_score_equating(
        old, new, linking_result=linking, items_old=items, items_new=items
    )

    np.testing.assert_array_equal(actual.old_scores, baseline.old_scores)
    np.testing.assert_array_equal(actual.theta, baseline.theta)
    np.testing.assert_allclose(actual.new_scores, baseline.new_scores, atol=1e-12)


def test_true_scores_accept_constants_estimated_by_public_link_api():
    old, new, _ = _equivalent_forms(TwoParameterLogistic, A=1.4, B=-0.7)
    anchors = list(range(4))
    linking = link(old, new, anchors, anchors, method="mean_mean")

    actual = true_score_equating(old, new, linking_result=linking)

    assert linking.constants.A == pytest.approx(1.4)
    assert linking.constants.B == pytest.approx(-0.7)
    # Interior scores have unique expected-score inverses on this grid.
    np.testing.assert_allclose(actual.new_scores[1:-1], [1.0, 2.0, 3.0], atol=1e-12)


def test_linked_observed_scores_match_hand_computed_binomial_percentiles():
    A, B = 1.4, -0.7
    theta_old = 0.3
    old = TwoParameterLogistic(2)
    new = TwoParameterLogistic(3)
    # At theta_old, the old items have P(correct)=0.2. At the corresponding
    # new ability, the new items have P(correct)=0.7.
    old.set_parameters(difficulty=np.full(2, theta_old - np.log(0.2 / 0.8)))
    new.set_parameters(
        discrimination=np.full(3, A),
        difficulty=np.full(3, (theta_old - B - np.log(0.7 / 0.3)) / A),
    )
    linking = LinkingResult(LinkingConstants(A=A, B=B), anchor_items=[])

    actual = observed_score_equating(
        old,
        new,
        theta_grid=np.array([theta_old]),
        theta_distribution=np.array([1.0]),
        linking_result=linking,
    )

    # Old Binomial(2, .2) probabilities: [.64, .32, .04], giving percentile
    # ranks [.32, .80, .98]. New Binomial(3, .7) probabilities:
    # [.027, .189, .441, .343] have cumulative probabilities
    # [.027, .216, .657, 1]. Each rank is inverted within the new score
    # whose interval [y - .5, y + .5] contains it (Kolen and Brennan, 2014).
    expected = np.array(
        [
            1.5 + (0.32 - 0.216) / 0.441,
            2.5 + (0.8 - 0.657) / 0.343,
            2.5 + (0.98 - 0.657) / 0.343,
        ]
    )
    np.testing.assert_allclose(actual.new_scores, expected, atol=1e-12)


def test_linked_true_scores_match_closed_form_different_length_forms():
    A, B = 1.4, -0.7
    old = TwoParameterLogistic(4)
    new = TwoParameterLogistic(6)
    old.set_parameters(difficulty=np.full(4, 0.2))
    new.set_parameters(
        discrimination=np.full(6, A), difficulty=np.full(6, (-0.4 - B) / A)
    )
    linking = LinkingResult(LinkingConstants(A=A, B=B), anchor_items=[])

    actual = true_score_equating(old, new, linking_result=linking)

    # Inverting 4*logistic(theta-.2) gives theta=.2+log(score/(4-score)).
    # The unattainable endpoint scores map to the endpoints of the new form.
    theta = np.array([0.2 + np.log(1.0 / 3.0), 0.2, 0.2 + np.log(3.0)])
    interior = 6.0 / (1.0 + np.exp(-(theta + 0.4)))
    expected = np.concatenate(([0.0], interior, [6.0]))
    np.testing.assert_allclose(actual.new_scores, expected, atol=1e-12)


@pytest.mark.parametrize(
    "model_type", [TwoParameterLogistic, ThreeParameterLogistic, GradedResponseModel]
)
@pytest.mark.parametrize("smoothing", ["none", "loglinear", "kernel"])
@pytest.mark.parametrize("items", [None, [3, 1, 0]])
def test_observed_scores_use_same_population_after_linking(
    model_type, smoothing, items
):
    old, new, linking = _equivalent_forms(model_type, A=1.4, B=-0.7)
    # Uneven quadrature weights also detect an incorrect second population or
    # an unnecessary Jacobian when transforming discrete probability masses.
    theta = np.array([-2.3, -0.4, 0.1, 0.9, 2.7])
    weights = np.array([1.0, 3.0, 8.0, 4.0, 2.0])
    original_theta = theta.copy()
    original_weights = weights.copy()

    actual = observed_score_equating(
        old,
        new,
        theta_grid=theta,
        theta_distribution=weights,
        items_old=items,
        items_new=items,
        smoothing=smoothing,
        linking_result=linking,
    )

    np.testing.assert_allclose(actual.new_scores, actual.old_scores, atol=1e-12)
    np.testing.assert_array_equal(actual.theta, theta)
    np.testing.assert_array_equal(theta, original_theta)
    np.testing.assert_array_equal(weights, original_weights)


@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_observed_scores_recover_representable_linked_abilities_after_overflow(backend):
    if backend == "rust" and not RUST_AVAILABLE:
        pytest.skip("compiled backend unavailable")
    old, new = TwoParameterLogistic(3), TwoParameterLogistic(3)
    old.set_parameters(discrimination=np.full(3, 1e-308), difficulty=np.full(3, -1e308))
    # theta_old = 1e308 * theta_new - 1e308. These forms share logits 0, 1,
    # and 2 at the selected population points, despite overflowing centering.
    linking = LinkingResult(LinkingConstants(A=1e308, B=-1e308), anchor_items=[])
    theta = np.array([-1e308, 0.0, 1e308])
    original_theta = theta.copy()
    previous_backend = mirt.get_backend()
    mirt.set_backend(backend)
    try:
        actual = observed_score_equating(
            old,
            new,
            theta_grid=theta,
            theta_distribution=np.array([1.0, 2.0, 3.0]),
            linking_result=linking,
        )
    finally:
        mirt.set_backend(previous_backend)

    np.testing.assert_allclose(
        actual.new_scores, actual.old_scores, atol=1e-15, rtol=1e-14
    )
    np.testing.assert_array_equal(actual.theta, original_theta)
    np.testing.assert_array_equal(theta, original_theta)


@pytest.mark.parametrize("equate", [true_score_equating, observed_score_equating])
@pytest.mark.parametrize(
    ("A", "B", "message"),
    [
        (0.0, 0.0, "A must be finite and positive"),
        (-1.0, 0.0, "A must be finite and positive"),
        (np.inf, 0.0, "A must be finite and positive"),
        (np.nan, 0.0, "A must be finite and positive"),
        (1.0, np.inf, "B must be finite"),
        (1.0, np.nan, "B must be finite"),
        (np.finfo(float).tiny, np.finfo(float).max, "non-finite theta"),
    ],
)
def test_score_equating_validates_linking_constants(equate, A, B, message):
    model = TwoParameterLogistic(4)
    linking = LinkingResult(LinkingConstants(A=A, B=B), anchor_items=[0, 1])

    with pytest.raises(ValueError, match=message):
        equate(model, model, linking_result=linking)
