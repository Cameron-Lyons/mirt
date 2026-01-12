"""Tests for Advanced Growth Models."""

import numpy as np
import pytest

from mirt.models.dynamic import (
    GrowthMixtureModel,
    NonlinearGrowthModel,
    PiecewiseGrowthModel,
)


class TestPiecewiseGrowthModel:
    """Tests for PiecewiseGrowthModel."""

    def test_init_single_piece(self):
        """Test initialization with single piece (linear)."""
        model = PiecewiseGrowthModel(n_pieces=1)

        assert model.n_pieces == 1
        assert len(model.changepoints) == 0

    def test_init_two_pieces(self):
        """Test initialization with two pieces."""
        model = PiecewiseGrowthModel(
            n_pieces=2,
            changepoints=np.array([2.0]),
        )

        assert model.n_pieces == 2
        assert len(model.changepoints) == 1

    def test_init_auto_changepoints(self):
        """Test automatic changepoint initialization."""
        model = PiecewiseGrowthModel(n_pieces=3)

        assert model.n_pieces == 3
        assert len(model.changepoints) == 2

    def test_init_changepoints_mismatch(self):
        """Test error on changepoint mismatch."""
        with pytest.raises(ValueError, match="changepoints length"):
            PiecewiseGrowthModel(
                n_pieces=3,
                changepoints=np.array([1.0]),
            )

    def test_compute_theta_single_piece(self):
        """Test theta computation for single piece."""
        model = PiecewiseGrowthModel(n_pieces=1)
        time_values = np.array([0.0, 1.0, 2.0, 3.0])
        intercept = 0.0
        slopes = np.array([[0.5]])

        theta = model.compute_theta(time_values, intercept, slopes)

        expected = np.array([0.0, 0.5, 1.0, 1.5])
        np.testing.assert_array_almost_equal(theta, expected)

    def test_compute_theta_two_pieces(self):
        """Test theta computation for two pieces."""
        model = PiecewiseGrowthModel(
            n_pieces=2,
            changepoints=np.array([2.0]),
        )
        time_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        intercept = 0.0
        slopes = np.array([[0.5, 0.1]])

        theta = model.compute_theta(time_values, intercept, slopes)

        assert theta[0] == 0.0
        assert theta[2] == pytest.approx(1.0)
        assert theta[4] < 1.5

    def test_simulate(self):
        """Test simulation."""
        model = PiecewiseGrowthModel(n_pieces=2, changepoints=np.array([3.0]))
        time_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])

        theta, intercepts, slopes = model.simulate(
            n_persons=50, time_values=time_values, seed=42
        )

        assert theta.shape == (50, 6)
        assert intercepts.shape == (50,)
        assert slopes.shape == (50, 2)

    def test_detect_changepoints_linear(self):
        """Test changepoint detection with linear data."""
        model = PiecewiseGrowthModel(n_pieces=1)
        time_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        observations = np.array([[0.0, 0.5, 1.0, 1.5, 2.0]])

        changepoints = model.detect_changepoints(time_values, observations)

        assert len(changepoints) == 0

    def test_detect_changepoints_piecewise(self):
        """Test changepoint detection with piecewise data."""
        model = PiecewiseGrowthModel(n_pieces=1)
        time_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        observations = np.array(
            [
                [0.0, 0.5, 1.0, 1.0, 1.0, 1.0],
            ]
        )

        changepoints = model.detect_changepoints(
            time_values, observations, max_changepoints=1
        )

        assert len(changepoints) <= 1


class TestNonlinearGrowthModel:
    """Tests for NonlinearGrowthModel."""

    def test_init_logistic(self):
        """Test initialization with logistic growth."""
        model = NonlinearGrowthModel(
            growth_type="logistic",
            asymptote=1.0,
            rate=1.0,
            inflection=0.0,
        )

        assert model.growth_type == "logistic"
        assert model.asymptote == 1.0

    def test_init_exponential(self):
        """Test initialization with exponential growth."""
        model = NonlinearGrowthModel(growth_type="exponential")

        assert model.growth_type == "exponential"

    def test_init_gompertz(self):
        """Test initialization with Gompertz growth."""
        model = NonlinearGrowthModel(growth_type="gompertz")

        assert model.growth_type == "gompertz"

    def test_compute_theta_logistic(self):
        """Test logistic growth computation."""
        model = NonlinearGrowthModel(
            growth_type="logistic",
            asymptote=1.0,
            rate=1.0,
            inflection=0.0,
        )
        time_values = np.array([-2.0, 0.0, 2.0])

        theta = model.compute_theta(time_values)

        assert theta[1] == pytest.approx(0.5)
        assert theta[0] < theta[1] < theta[2]

    def test_compute_theta_exponential(self):
        """Test exponential growth computation."""
        model = NonlinearGrowthModel(
            growth_type="exponential",
            asymptote=1.0,
            rate=1.0,
        )
        time_values = np.array([0.0, 1.0, 2.0, 10.0])

        theta = model.compute_theta(time_values)

        assert theta[0] == pytest.approx(0.0)
        assert theta[-1] == pytest.approx(1.0, abs=0.01)

    def test_compute_theta_gompertz(self):
        """Test Gompertz growth computation."""
        model = NonlinearGrowthModel(
            growth_type="gompertz",
            asymptote=1.0,
            rate=1.0,
            inflection=0.0,
        )
        time_values = np.array([-2.0, 0.0, 2.0])

        theta = model.compute_theta(time_values)

        assert theta[0] < theta[1] < theta[2]
        assert np.all(theta <= model.asymptote)

    def test_growth_velocity_logistic(self):
        """Test growth velocity computation for logistic."""
        model = NonlinearGrowthModel(
            growth_type="logistic",
            asymptote=1.0,
            rate=1.0,
            inflection=0.0,
        )
        time_values = np.array([-2.0, 0.0, 2.0])

        velocity = model.growth_velocity(time_values)

        assert velocity[1] > velocity[0]
        assert velocity[1] > velocity[2]

    def test_growth_velocity_exponential(self):
        """Test growth velocity computation for exponential."""
        model = NonlinearGrowthModel(
            growth_type="exponential",
            asymptote=1.0,
            rate=1.0,
        )
        time_values = np.array([0.0, 1.0, 2.0])

        velocity = model.growth_velocity(time_values)

        assert velocity[0] > velocity[1] > velocity[2]

    def test_simulate(self):
        """Test simulation."""
        model = NonlinearGrowthModel(growth_type="logistic")
        time_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        theta, params = model.simulate(n_persons=50, time_values=time_values, seed=42)

        assert theta.shape == (50, 5)
        assert "asymptote" in params
        assert "rate" in params
        assert "inflection" in params
        assert len(params["asymptote"]) == 50

    def test_fit_individual(self):
        """Test individual fitting."""
        model = NonlinearGrowthModel(
            growth_type="logistic",
            asymptote=1.0,
            rate=1.0,
            inflection=2.0,
        )
        time_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        true_theta = model.compute_theta(time_values)

        params = model.fit_individual(time_values, true_theta)

        assert "asymptote" in params
        assert "rate" in params
        assert "inflection" in params


class TestGrowthMixtureModel:
    """Tests for GrowthMixtureModel."""

    def test_init_basic(self):
        """Test basic initialization."""
        model = GrowthMixtureModel(n_classes=3)

        assert model.n_classes == 3
        assert len(model.class_proportions) == 3
        assert len(model.class_intercepts) == 3
        assert len(model.class_slopes) == 3

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        model = GrowthMixtureModel(
            n_classes=2,
            class_proportions=np.array([0.7, 0.3]),
            class_intercepts=np.array([0.0, 1.0]),
            class_slopes=np.array([0.1, 0.5]),
        )

        np.testing.assert_array_equal(model.class_proportions, [0.7, 0.3])
        np.testing.assert_array_equal(model.class_intercepts, [0.0, 1.0])

    def test_init_quadratic(self):
        """Test initialization with quadratic growth."""
        model = GrowthMixtureModel(n_classes=2, growth_type="quadratic")

        assert model.growth_type == "quadratic"
        assert len(model.class_quadratics) == 2

    def test_compute_class_trajectory(self):
        """Test class trajectory computation."""
        model = GrowthMixtureModel(
            n_classes=2,
            class_intercepts=np.array([0.0, 1.0]),
            class_slopes=np.array([0.5, 0.2]),
        )
        time_values = np.array([0.0, 1.0, 2.0])

        traj_0 = model.compute_class_trajectory(0, time_values)
        traj_1 = model.compute_class_trajectory(1, time_values)

        np.testing.assert_array_almost_equal(traj_0, [0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(traj_1, [1.0, 1.2, 1.4])

    def test_class_likelihood(self):
        """Test class likelihood computation."""
        model = GrowthMixtureModel(n_classes=2)
        observations = np.array(
            [
                [0.0, 0.1, 0.2, 0.3, 0.4],
            ]
        )
        time_values = np.arange(5, dtype=np.float64)

        likelihoods = model.class_likelihood(observations, time_values)

        assert likelihoods.shape == (1, 2)
        assert np.all(likelihoods >= 0)

    def test_classify(self):
        """Test classification."""
        model = GrowthMixtureModel(
            n_classes=2,
            class_intercepts=np.array([-1.0, 1.0]),
            class_slopes=np.array([0.5, 0.5]),
        )
        observations = np.array(
            [
                [-1.0, -0.5, 0.0, 0.5, 1.0],
                [1.0, 1.5, 2.0, 2.5, 3.0],
            ]
        )
        time_values = np.arange(5, dtype=np.float64)

        classes = model.classify(observations, time_values)

        assert len(classes) == 2
        assert classes[0] == 0
        assert classes[1] == 1

    def test_posterior_probabilities(self):
        """Test posterior probability computation."""
        np.random.seed(42)
        model = GrowthMixtureModel(n_classes=2)
        observations = np.random.randn(10, 5)
        time_values = np.arange(5, dtype=np.float64)

        posteriors = model.posterior_probabilities(observations, time_values)

        assert posteriors.shape == (10, 2)
        np.testing.assert_array_almost_equal(
            posteriors.sum(axis=1), np.ones(10), decimal=3
        )

    def test_simulate(self):
        """Test simulation."""
        model = GrowthMixtureModel(n_classes=3)
        time_values = np.arange(5, dtype=np.float64)

        observations, true_classes = model.simulate(
            n_persons=100, time_values=time_values, seed=42
        )

        assert observations.shape == (100, 5)
        assert true_classes.shape == (100,)
        assert set(true_classes).issubset({0, 1, 2})

    def test_fit_em(self):
        """Test EM fitting."""
        model = GrowthMixtureModel(
            n_classes=2,
            class_intercepts=np.array([-1.0, 1.0]),
            class_slopes=np.array([0.3, 0.3]),
        )
        time_values = np.arange(5, dtype=np.float64)

        observations, _ = model.simulate(
            n_persons=100, time_values=time_values, seed=42
        )

        result = model.fit_em(observations, time_values, max_iter=20)

        assert "classifications" in result
        assert "posteriors" in result
        assert "log_likelihood" in result
        assert "n_iterations" in result
        assert "converged" in result

    def test_entropy(self):
        """Test entropy computation."""
        model = GrowthMixtureModel(
            n_classes=2,
            class_intercepts=np.array([-2.0, 2.0]),
            class_slopes=np.array([0.3, 0.3]),
        )
        time_values = np.arange(5, dtype=np.float64)
        observations, _ = model.simulate(n_persons=50, time_values=time_values, seed=42)

        entropy = model.entropy(observations, time_values)

        assert 0 <= entropy


class TestGrowthModelIntegration:
    """Integration tests for growth models."""

    def test_piecewise_simulate_and_detect(self):
        """Test piecewise simulation and detection."""
        model = PiecewiseGrowthModel(
            n_pieces=2,
            changepoints=np.array([3.0]),
            slope_means=np.array([0.5, 0.1]),
        )
        time_values = np.linspace(0, 6, 7)

        theta, _, _ = model.simulate(n_persons=100, time_values=time_values, seed=42)

        changepoints = model.detect_changepoints(time_values, theta, max_changepoints=2)

        assert len(changepoints) <= 2

    def test_nonlinear_round_trip(self):
        """Test nonlinear model round trip."""
        model = NonlinearGrowthModel(
            growth_type="logistic",
            asymptote=2.0,
            rate=0.5,
            inflection=3.0,
        )
        time_values = np.linspace(0, 10, 11)

        theta, params = model.simulate(n_persons=50, time_values=time_values, seed=42)

        assert theta.shape == (50, 11)
        max_expected = (
            model.asymptote
            + 3 * np.sqrt(model.asymptote_var)
            + 3 * np.sqrt(model.residual_variance)
        )
        assert np.mean(theta) < max_expected

    def test_mixture_class_recovery(self):
        """Test mixture model class recovery."""
        np.random.seed(42)
        model = GrowthMixtureModel(
            n_classes=2,
            class_proportions=np.array([0.5, 0.5]),
            class_intercepts=np.array([-2.0, 2.0]),
            class_slopes=np.array([0.3, 0.3]),
            intercept_var=0.1,
            slope_var=0.01,
            residual_variance=0.1,
        )
        time_values = np.arange(5, dtype=np.float64)

        observations, true_classes = model.simulate(
            n_persons=100, time_values=time_values, seed=42
        )

        result = model.fit_em(observations, time_values, max_iter=50)

        accuracy = np.mean(result["classifications"] == true_classes)
        if accuracy < 0.5:
            accuracy = 1 - accuracy

        assert accuracy > 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
