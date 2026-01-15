"""Tests for score equating functions."""

import numpy as np
import pytest

from mirt.equating import (
    ScoreEquatingResult,
    equipercentile_equating,
    lord_wingersky_recursion,
    observed_score_equating,
    score_to_theta,
    theta_to_score,
    true_score_equating,
)
from mirt.models.dichotomous import TwoParameterLogistic


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

    def test_true_score_same_form_identity(self, simple_model):
        """Test that equating same form gives identity."""
        result = true_score_equating(simple_model, simple_model)

        np.testing.assert_allclose(result.old_scores, result.new_scores, atol=0.5)

    def test_true_score_with_item_subset(self, simple_model):
        """Test equating with item subsets."""
        result = true_score_equating(
            simple_model,
            simple_model,
            items_old=[0, 1, 2, 3, 4],
            items_new=[5, 6, 7, 8, 9],
        )

        assert len(result.old_scores) == 6


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

        np.testing.assert_allclose(equated, np.arange(5), atol=0.01)

    def test_equipercentile_shifted_dist(self):
        """Test with shifted distribution."""
        dist_old = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        dist_new = np.array([0.05, 0.1, 0.2, 0.4, 0.2, 0.05])

        equated = equipercentile_equating(dist_old, dist_new)

        assert len(equated) == len(dist_old)
        assert equated[0] >= 0
        assert equated[-1] <= len(dist_new) - 1

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
