"""Tests for Mixture IRT model."""

import numpy as np
import pytest

from mirt import MixtureIRT, fit_mixture_irt


class TestMixtureIRT:
    """Tests for MixtureIRT model."""

    def test_init(self):
        """Test MixtureIRT initialization."""
        model = MixtureIRT(n_items=10, n_classes=2)
        assert model.n_items == 10
        assert model.n_classes == 2
        assert model.base_model == "2PL"
        assert model.model_name == "MixtureIRT"

    def test_invalid_n_classes(self):
        """Test that n_classes < 2 raises error."""
        with pytest.raises(ValueError):
            MixtureIRT(n_items=10, n_classes=1)

    def test_class_proportions(self):
        """Test class proportion initialization."""
        model = MixtureIRT(n_items=10, n_classes=3)
        model._initialize_parameters()

        props = model.class_proportions
        assert len(props) == 3
        assert np.isclose(props.sum(), 1.0)

    def test_class_parameters(self):
        """Test getting class-specific parameters."""
        model = MixtureIRT(n_items=10, n_classes=2, base_model="3PL")
        model._initialize_parameters()

        params = model.get_class_parameters(0)
        assert "discrimination" in params
        assert "difficulty" in params
        assert "guessing" in params
        assert len(params["difficulty"]) == 10

    def test_probability(self):
        """Test marginal probability computation."""
        model = MixtureIRT(n_items=10, n_classes=2)
        model._initialize_parameters()

        theta = np.array([[0.0], [1.0], [-1.0]])
        probs = model.probability(theta)

        assert probs.shape == (3, 10)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_class_posterior(self, dichotomous_responses):
        """Test class posterior computation."""
        responses = dichotomous_responses["responses"]
        n_items = dichotomous_responses["n_items"]
        model = MixtureIRT(n_items=n_items, n_classes=2)
        model._initialize_parameters()

        theta = np.zeros((len(responses), 1))
        posterior = model.class_posterior(responses, theta)

        assert posterior.shape == (len(responses), 2)
        assert np.allclose(posterior.sum(axis=1), 1.0)

    def test_classify_persons(self, dichotomous_responses):
        """Test person classification."""
        responses = dichotomous_responses["responses"]
        n_items = dichotomous_responses["n_items"]
        model = MixtureIRT(n_items=n_items, n_classes=2)
        model._initialize_parameters()

        theta = np.zeros((len(responses), 1))
        classes = model.classify_persons(responses, theta)

        assert classes.shape == (len(responses),)
        assert set(classes).issubset({0, 1})


class TestFitMixtureIRT:
    """Tests for mixture IRT fitting."""

    def test_fit_mixture(self, dichotomous_responses):
        """Test fitting mixture IRT model."""
        responses = dichotomous_responses["responses"]

        model, posteriors = fit_mixture_irt(
            responses=responses,
            n_classes=2,
            base_model="2PL",
            max_iter=20,
        )

        assert model._is_fitted
        assert posteriors.shape == (len(responses), 2)
        assert np.allclose(posteriors.sum(axis=1), 1.0)

    def test_fit_with_3pl(self, dichotomous_responses):
        """Test fitting mixture 3PL model."""
        responses = dichotomous_responses["responses"]

        model, _ = fit_mixture_irt(
            responses=responses,
            n_classes=2,
            base_model="3PL",
            max_iter=10,
        )

        assert model._is_fitted
        params = model.get_class_parameters(0)
        assert "guessing" in params
