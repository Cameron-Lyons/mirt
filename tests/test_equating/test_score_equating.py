"""Tests for score equating functions."""

import warnings
from fractions import Fraction
from itertools import accumulate

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import expit, gammaln

from mirt import CustomItemModel, create_item_type
from mirt.equating import (
    ScoreEquatingResult,
    equipercentile_equating,
    lord_wingersky_recursion,
    observed_score_equating,
    score_to_theta,
    theta_to_score,
    true_score_equating,
)
from mirt.models.dichotomous import (
    FourParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.polytomous import GradedResponseModel
from mirt.models.zeroinflated import ZeroInflated3PL


def _asymptotic_form(model_type, n_items, seed, guessing=0.2):
    """Build a form whose expected-score curve has the family's asymptotes."""
    rng = np.random.default_rng(seed)
    discrimination = rng.uniform(0.6, 2.0, n_items)
    difficulty = rng.normal(0.0, 1.0, n_items)
    if model_type is GradedResponseModel:
        model = model_type(n_items, n_categories=4)
        thresholds = np.sort(rng.normal(0.0, 1.2, (n_items, 3)), axis=1)
        return model.set_parameters(
            discrimination=discrimination, thresholds=thresholds
        )
    model = model_type(n_items)
    parameters = {"discrimination": discrimination, "difficulty": difficulty}
    if model_type is not TwoParameterLogistic:
        parameters["guessing"] = np.full(n_items, guessing)
    if model_type is FourParameterLogistic:
        parameters["upper"] = rng.uniform(0.85, 0.95, n_items)
    if model_type is ZeroInflated3PL:
        parameters["zero_inflation"] = rng.uniform(0.02, 0.1, n_items)
    return model.set_parameters(**parameters)


def _three_pl_true_score_oracle(old, new, score):
    """Solve the old 3PL test characteristic curve with an independent solver."""

    def tcc(model, theta):
        logits = model.discrimination * (theta - model.difficulty)
        return np.sum(model.guessing + (1.0 - model.guessing) * expit(logits))

    theta = brentq(lambda t: tcc(old, t) - score, -60.0, 60.0, xtol=1e-14)
    return tcc(new, theta)


def _exact_kolen_brennan(old_counts, new_counts):
    """Evaluate Kolen and Brennan (2014, eqs. 2.14-2.18) in exact arithmetic."""
    old = [Fraction(int(count), int(sum(old_counts))) for count in old_counts]
    new = [Fraction(int(count), int(sum(new_counts))) for count in new_counts]
    cumulative = list(accumulate(new))
    equivalents = []
    for score, probability in enumerate(old):
        rank = sum(old[:score], Fraction(0)) + probability / 2
        cells = [y for y, total in enumerate(cumulative) if total > rank]
        if not cells:
            equivalents.append(len(new) - 0.5)
            continue
        cell = cells[0]
        below = cumulative[cell - 1] if cell else Fraction(0)
        equivalents.append(float(cell - Fraction(1, 2) + (rank - below) / new[cell]))
    return np.array(equivalents)


@pytest.fixture
def simple_model():
    """Create a simple 2PL model."""
    model = TwoParameterLogistic(n_items=10)
    disc = np.ones(10)
    diff = np.linspace(-2, 2, 10)
    model.set_parameters(discrimination=disc, difficulty=diff)
    model._is_fitted = True
    return model


@pytest.fixture
def model_pair():
    """Create a pair of models for equating."""
    model_old = TwoParameterLogistic(n_items=10)
    disc_old = np.array([1.0, 1.2, 0.8, 1.5, 1.1, 0.9, 1.3, 1.0, 1.4, 0.7])
    diff_old = np.linspace(-2, 2, 10)
    model_old.set_parameters(discrimination=disc_old, difficulty=diff_old)
    model_old._is_fitted = True

    model_new = TwoParameterLogistic(n_items=10)
    disc_new = np.array([1.1, 1.0, 0.9, 1.4, 1.2, 0.8, 1.2, 1.1, 1.3, 0.8])
    diff_new = np.linspace(-1.8, 2.2, 10)
    model_new.set_parameters(discrimination=disc_new, difficulty=diff_new)
    model_new._is_fitted = True

    return model_old, model_new


class TestTrueScoreEquating:
    """Tests for true score equating."""

    def test_true_score_returns_result(self, model_pair):
        """Test that true_score_equating returns ScoreEquatingResult."""
        model_old, model_new = model_pair

        result = true_score_equating(model_old, model_new)

        assert isinstance(result, ScoreEquatingResult)
        assert result.method == "true_score"

    def test_true_score_arrays_match(self, model_pair):
        """Test that old_scores and new_scores have same length."""
        model_old, model_new = model_pair

        result = true_score_equating(model_old, model_new)

        assert len(result.old_scores) == len(result.new_scores)

    def test_true_score_monotonic(self, model_pair):
        """Test that equated scores are monotonically increasing."""
        model_old, model_new = model_pair

        result = true_score_equating(model_old, model_new)

        diffs = np.diff(result.new_scores)
        assert np.all(diffs >= -0.01)

    @pytest.mark.parametrize(
        "model_type",
        [
            TwoParameterLogistic,
            ThreeParameterLogistic,
            FourParameterLogistic,
            GradedResponseModel,
            ZeroInflated3PL,
        ],
    )
    def test_true_score_same_form_identity(self, model_type):
        """Equating a form to itself is the identity, endpoints included."""
        model = _asymptotic_form(model_type, n_items=8, seed=3)

        result = true_score_equating(model, model)

        np.testing.assert_allclose(result.new_scores, result.old_scores, atol=1e-10)

    def test_true_score_with_item_subset(self, simple_model):
        """Test equating with item subsets."""
        result = true_score_equating(
            simple_model,
            simple_model,
            items_old=[0, 1, 2, 3, 4],
            items_new=[5, 6, 7, 8, 9],
        )

        assert len(result.old_scores) == 6


class TestTrueScoreEndpoints:
    """Kolen-Brennan conventions outside the attainable true-score range."""

    def test_scores_below_chance_follow_lord_line(self):
        old = _asymptotic_form(ThreeParameterLogistic, 20, seed=0, guessing=0.2)
        new = _asymptotic_form(ThreeParameterLogistic, 20, seed=1, guessing=0.25)

        result = true_score_equating(old, new)

        # Sum of guessing is 4 on the old form and 5 on the new form, so
        # scores 0..4 lie on the line from (0, 0) to (4, 5).
        np.testing.assert_allclose(
            result.new_scores[:5], np.arange(5) * 5.0 / 4.0, rtol=1e-12
        )
        assert np.all(np.diff(result.new_scores) > 0.0)

    def test_interior_scores_match_independent_root_oracle(self):
        old = _asymptotic_form(ThreeParameterLogistic, 20, seed=0, guessing=0.2)
        new = _asymptotic_form(ThreeParameterLogistic, 20, seed=1, guessing=0.25)

        result = true_score_equating(old, new)

        expected = [_three_pl_true_score_oracle(old, new, x) for x in range(5, 20)]
        np.testing.assert_allclose(result.new_scores[5:20], expected, atol=1e-10)

    def test_extreme_scores_map_to_extreme_scores(self):
        old = _asymptotic_form(TwoParameterLogistic, 12, seed=4)
        new = _asymptotic_form(TwoParameterLogistic, 17, seed=5)

        result = true_score_equating(old, new)

        assert result.new_scores[0] == 0.0
        assert result.new_scores[-1] == 17.0
        assert np.all(np.diff(result.new_scores) > 0.0)

    def test_scores_above_upper_asymptote_interpolate_to_maximum(self):
        old = FourParameterLogistic(10).set_parameters(
            discrimination=np.linspace(0.8, 1.6, 10),
            difficulty=np.linspace(-1.5, 1.5, 10),
            guessing=np.full(10, 0.1),
            upper=np.full(10, 0.85),
        )
        new = FourParameterLogistic(10).set_parameters(
            discrimination=np.linspace(0.9, 1.4, 10),
            difficulty=np.linspace(-1.0, 1.8, 10),
            guessing=np.full(10, 0.2),
            upper=np.full(10, 0.9),
        )

        result = true_score_equating(old, new)

        # Old upper limit 8.5 maps to 9; old maximum 10 maps to maximum 10.
        np.testing.assert_allclose(
            result.new_scores[[0, 1, 9, 10]],
            [0.0, 2.0, 9.0 + 1.0 / 3.0, 10.0],
            rtol=1e-12,
        )

    def test_reporting_range_does_not_limit_solved_abilities(self):
        old = _asymptotic_form(ThreeParameterLogistic, 15, seed=6)
        new = _asymptotic_form(ThreeParameterLogistic, 15, seed=7)

        narrow = true_score_equating(old, new, theta_range=(-0.5, 0.5), n_theta=5)
        default = true_score_equating(old, new)

        np.testing.assert_array_equal(narrow.theta, np.linspace(-0.5, 0.5, 5))
        np.testing.assert_allclose(narrow.new_scores, default.new_scores, atol=1e-11)

    def test_custom_curves_that_overflow_at_extreme_abilities_do_not_warn(self):
        def logistic(theta, slope, location):
            return 1 / (1 + np.exp(-slope * (theta - location)))

        spec = create_item_type(
            "Logistic",
            logistic,
            par_bounds={"slope": (0.05, 5), "location": (-5, 5)},
            par_defaults={"slope": 1, "location": 0},
        )
        model = CustomItemModel(n_items=3, item_type=spec)
        model.set_parameters(slope=[0.8, 1.2, 1.5], location=[-0.5, 0.0, 0.7])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = true_score_equating(model, model)

        np.testing.assert_allclose(result.new_scores, result.old_scores, atol=1e-10)


class TestObservedScoreEquating:
    """Tests for observed score equating."""

    def test_observed_score_returns_result(self, model_pair):
        """Test that observed_score_equating returns ScoreEquatingResult."""
        model_old, model_new = model_pair

        result = observed_score_equating(model_old, model_new)

        assert isinstance(result, ScoreEquatingResult)
        assert result.method == "observed_score"

    def test_observed_score_arrays(self, model_pair):
        """Test that arrays are properly formed."""
        model_old, model_new = model_pair

        result = observed_score_equating(model_old, model_new)

        assert len(result.old_scores) == len(result.new_scores)
        assert len(result.old_scores) == 11

    def test_observed_score_with_distribution(self, simple_model):
        """Test with custom theta distribution."""
        theta_grid = np.linspace(-3, 3, 31)
        theta_dist = np.exp(-(theta_grid**2) / 2)
        theta_dist = theta_dist / np.sum(theta_dist)

        result = observed_score_equating(
            simple_model,
            simple_model,
            theta_grid=theta_grid,
            theta_distribution=theta_dist,
        )

        assert isinstance(result, ScoreEquatingResult)


class TestLordWingerskyRecursion:
    """Tests for Lord-Wingersky recursion."""

    def test_lw_returns_distribution(self, simple_model):
        """Test that L-W returns valid probability distribution."""
        theta_grid = np.linspace(-3, 3, 21)
        weights = np.ones(21) / 21

        dist = lord_wingersky_recursion(simple_model, theta_grid, weights)

        assert len(dist) == 11
        assert abs(np.sum(dist) - 1.0) < 0.01
        assert np.all(dist >= 0)
        assert np.all(dist <= 1)

    def test_lw_distribution_shape(self, simple_model):
        """Test distribution has reasonable shape."""
        theta_grid = np.linspace(-3, 3, 51)
        from scipy import stats

        weights = stats.norm.pdf(theta_grid)
        weights = weights / np.sum(weights)

        dist = lord_wingersky_recursion(simple_model, theta_grid, weights)

        middle_idx = len(dist) // 2
        assert dist[middle_idx] > dist[0]
        assert dist[middle_idx] > dist[-1]


class TestEquipercentileEquating:
    """Tests for equipercentile equating."""

    def test_equipercentile_same_dist(self):
        """Test that same distribution gives identity."""
        dist = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

        equated = equipercentile_equating(dist, dist)

        np.testing.assert_allclose(equated, np.arange(5), atol=1e-12)

    def test_same_dist_identity_keeps_tail_precision(self):
        """Tail scores with tiny probabilities still equate to themselves."""
        scores = np.arange(61)
        dist = np.exp(
            gammaln(61) - gammaln(scores + 1) - gammaln(61 - scores) - 60 * np.log(2)
        )

        equated = equipercentile_equating(dist, dist)

        np.testing.assert_allclose(equated, scores, atol=1e-12)

    def test_equipercentile_shifted_dist(self):
        """Test with shifted distribution."""
        dist_old = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        dist_new = np.array([0.05, 0.1, 0.2, 0.4, 0.2, 0.05])

        equated = equipercentile_equating(dist_old, dist_new)

        assert len(equated) == len(dist_old)
        assert equated[0] >= -0.5
        assert equated[-1] <= len(dist_new) - 0.5
        assert np.all(np.diff(equated) > 0.0)

    def test_kolen_brennan_four_point_example(self):
        """Low ranks invert below zero instead of being clamped."""
        equated = equipercentile_equating(
            np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.4, 0.3, 0.2, 0.1])
        )

        np.testing.assert_allclose(
            equated, [-0.375, 0.0, 2.0 / 3.0, 2.0], rtol=1e-12, atol=1e-12
        )

    @pytest.mark.parametrize("seed", range(6))
    def test_matches_exact_kolen_brennan_oracle_with_zero_cells(self, seed):
        rng = np.random.default_rng(seed)
        old_counts = rng.integers(0, 20, 9) * (rng.random(9) > 0.3)
        new_counts = rng.integers(0, 20, 12) * (rng.random(12) > 0.3)
        old_counts[rng.integers(9)] += 1
        new_counts[rng.integers(12)] += 1

        equated = equipercentile_equating(old_counts, new_counts)

        expected = _exact_kolen_brennan(old_counts, new_counts)
        np.testing.assert_allclose(equated, expected, rtol=1e-12, atol=1e-12)
        assert np.all(equated >= -0.5) and np.all(equated <= len(new_counts) - 0.5)

    def test_equipercentile_smoothing(self):
        """Test smoothing options."""
        dist = np.array([0.1, 0.2, 0.4, 0.2, 0.1])

        for smoothing in ["none", "loglinear", "kernel"]:
            equated = equipercentile_equating(dist, dist, smoothing=smoothing)
            assert len(equated) == len(dist)


class TestScoreConversion:
    """Tests for score to theta conversion."""

    def test_score_to_theta_range(self, simple_model):
        """Test that theta estimates are in expected range."""
        scores = np.array([0, 3, 5, 7, 10])

        theta = score_to_theta(simple_model, scores)

        assert len(theta) == len(scores)
        assert np.all(theta >= -4)
        assert np.all(theta <= 4)
        assert np.all(np.diff(theta) > 0)

    def test_theta_to_score_range(self, simple_model):
        """Test that scores are in valid range."""
        theta = np.array([-2, -1, 0, 1, 2])

        scores = theta_to_score(simple_model, theta)

        assert len(scores) == len(theta)
        assert np.all(scores >= 0)
        assert np.all(scores <= 10)
        assert np.all(np.diff(scores) > 0)

    def test_round_trip_conversion(self, simple_model):
        """Test score -> theta -> score round trip."""
        original_scores = np.array([2.0, 4.0, 6.0, 8.0])

        theta = score_to_theta(simple_model, original_scores)
        recovered_scores = theta_to_score(simple_model, theta)

        np.testing.assert_allclose(recovered_scores, original_scores, atol=0.1)


class TestScoreEquatingValidation:
    """Validation tests for score equating."""

    def test_true_score_with_linking(self, model_pair):
        """Test true score equating with linking result."""
        from mirt.equating import link

        model_old, model_new = model_pair
        anchors = list(range(10))

        linking_result = link(model_old, model_new, anchors, anchors)

        result = true_score_equating(
            model_old, model_new, linking_result=linking_result
        )

        assert isinstance(result, ScoreEquatingResult)
