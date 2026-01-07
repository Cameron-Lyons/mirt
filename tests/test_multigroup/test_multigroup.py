"""Tests for multigroup IRT analysis."""

import numpy as np
import pytest

from mirt.multigroup.multigroup import compare_invariance, fit_multigroup


class TestMultigroupAnalysis:
    """Tests for multigroup IRT functions."""

    def test_configural_invariance(self, rng):
        """Test configural invariance fitting."""
        n_persons_per_group = 100
        n_items = 10

        theta1 = rng.standard_normal(n_persons_per_group)
        difficulty1 = rng.normal(0, 1, n_items)
        probs1 = 1 / (1 + np.exp(-(theta1[:, None] - difficulty1)))
        responses1 = (rng.random((n_persons_per_group, n_items)) < probs1).astype(int)

        theta2 = rng.standard_normal(n_persons_per_group)
        difficulty2 = difficulty1 + 0.5
        probs2 = 1 / (1 + np.exp(-(theta2[:, None] - difficulty2)))
        responses2 = (rng.random((n_persons_per_group, n_items)) < probs2).astype(int)

        responses = np.vstack([responses1, responses2])
        groups = np.array([0] * n_persons_per_group + [1] * n_persons_per_group)

        result = fit_multigroup(
            responses,
            groups,
            model="2PL",
            invariance="configural",
            n_quadpts=15,
            max_iter=50,
            verbose=False,
        )

        assert result is not None
        assert result.model.is_fitted

    def test_minimum_two_groups(self, rng):
        """Test that at least 2 groups are required."""
        n_persons = 100
        n_items = 5

        responses = rng.integers(0, 2, size=(n_persons, n_items))
        groups = np.zeros(n_persons)

        with pytest.raises(ValueError, match="At least 2 groups"):
            fit_multigroup(responses, groups, model="2PL")

    def test_metric_invariance(self, rng):
        """Test metric invariance fitting."""
        n_persons = 100
        n_items = 5

        responses = rng.integers(0, 2, size=(n_persons, n_items))
        groups = np.array([0] * 50 + [1] * 50)

        result = fit_multigroup(
            responses,
            groups,
            model="2PL",
            invariance="metric",
            n_quadpts=15,
            max_iter=30,
            verbose=False,
        )

        assert result is not None
        assert result.model.is_fitted

    def test_scalar_invariance(self, rng):
        """Test scalar invariance fitting."""
        n_persons = 100
        n_items = 5

        responses = rng.integers(0, 2, size=(n_persons, n_items))
        groups = np.array([0] * 50 + [1] * 50)

        result = fit_multigroup(
            responses,
            groups,
            model="2PL",
            invariance="scalar",
            n_quadpts=15,
            max_iter=30,
            verbose=False,
        )

        assert result is not None
        assert result.model.is_fitted

    def test_strict_invariance(self, rng):
        """Test strict invariance fitting."""
        n_persons = 100
        n_items = 5

        responses = rng.integers(0, 2, size=(n_persons, n_items))
        groups = np.array([0] * 50 + [1] * 50)

        result = fit_multigroup(
            responses,
            groups,
            model="2PL",
            invariance="strict",
            n_quadpts=15,
            max_iter=30,
            verbose=False,
        )

        assert result is not None
        assert result.model.is_fitted

    def test_three_groups(self, rng):
        """Test with three groups."""
        n_per_group = 50
        n_items = 5

        responses = rng.integers(0, 2, size=(n_per_group * 3, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group + [2] * n_per_group)

        result = fit_multigroup(
            responses,
            groups,
            model="2PL",
            invariance="configural",
            n_quadpts=15,
            max_iter=30,
            verbose=False,
        )

        assert result is not None

    def test_string_group_labels(self, rng):
        """Test with string group labels."""
        n_persons = 60
        n_items = 5

        responses = rng.integers(0, 2, size=(n_persons, n_items))
        groups = np.array(["A"] * 30 + ["B"] * 30)

        result = fit_multigroup(
            responses,
            groups,
            model="2PL",
            invariance="configural",
            n_quadpts=15,
            max_iter=30,
            verbose=False,
        )

        assert result is not None

    def test_1pl_model(self, rng):
        """Test multigroup with 1PL model."""
        n_persons = 60
        n_items = 5

        responses = rng.integers(0, 2, size=(n_persons, n_items))
        groups = np.array([0] * 30 + [1] * 30)

        result = fit_multigroup(
            responses,
            groups,
            model="1PL",
            invariance="configural",
            n_quadpts=15,
            max_iter=30,
            verbose=False,
        )

        assert result is not None
        assert result.model.model_name == "1PL"

    def test_polytomous_grm(self, rng):
        """Test multigroup with GRM model."""
        n_persons = 80
        n_items = 5
        n_categories = 4

        responses = rng.integers(0, n_categories, size=(n_persons, n_items))
        groups = np.array([0] * 40 + [1] * 40)

        result = fit_multigroup(
            responses,
            groups,
            model="GRM",
            n_categories=n_categories,
            invariance="configural",
            n_quadpts=15,
            max_iter=30,
            verbose=False,
        )

        assert result is not None
        assert result.model.model_name == "GRM"

    def test_compare_invariance(self, rng):
        """Test comparing different invariance levels."""
        n_persons = 80
        n_items = 5

        responses = rng.integers(0, 2, size=(n_persons, n_items))
        groups = np.array([0] * 40 + [1] * 40)

        results = compare_invariance(
            responses,
            groups,
            model="2PL",
            n_quadpts=15,
            max_iter=30,
            verbose=False,
        )

        assert "configural" in results
        assert "metric" in results
        assert "scalar" in results
        assert "strict" in results

        for inv, result in results.items():
            assert result.model.is_fitted
