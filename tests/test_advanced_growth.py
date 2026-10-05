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

        assert set(params) == {"asymptote", "rate", "inflection", "converged", "sse"}
        assert all(type(params[key]) is float for key in params if key != "converged")
        assert params["converged"] is True
        assert params["asymptote"] == pytest.approx(1.0, abs=1e-6)
        assert params["rate"] == pytest.approx(1.0, abs=1e-6)
        assert params["inflection"] == pytest.approx(2.0, abs=1e-6)

    @pytest.mark.parametrize("growth_type", ["logistic", "gompertz", "exponential"])
    def test_fit_individual_recovers_noise_free_curve(self, growth_type):
        """Regression: the rate and inflection must move away from the defaults."""
        model = NonlinearGrowthModel(
            growth_type=growth_type,
            asymptote=2.0,
            rate=1.5,
            inflection=3.0,
        )
        time_values = np.linspace(0.0, 8.0, 17)
        observations = model.compute_theta(
            time_values, asymptote=3.0, rate=0.8, inflection=5.0
        )

        params = model.fit_individual(time_values, observations)

        assert params["converged"] is True
        assert params["asymptote"] == pytest.approx(3.0, abs=1e-6)
        assert params["rate"] == pytest.approx(0.8, abs=1e-6)
        expected_inflection = 3.0 if growth_type == "exponential" else 5.0
        assert params["inflection"] == pytest.approx(expected_inflection, abs=1e-6)
        assert params["sse"] < 1e-12

    @pytest.mark.parametrize("growth_type", ["logistic", "gompertz", "exponential"])
    def test_fit_individual_recovers_noisy_curve(self, growth_type):
        """Noisy trajectories are fitted close to the generating curve."""
        model = NonlinearGrowthModel(growth_type=growth_type, rate=1.0)
        rng = np.random.default_rng(20261005)
        time_values = np.linspace(0.0, 10.0, 25)
        truth = model.compute_theta(
            time_values, asymptote=2.5, rate=0.7, inflection=4.0
        )
        observations = truth + rng.normal(0.0, 0.05, time_values.size)

        params = model.fit_individual(time_values, observations)

        assert params["converged"] is True
        assert params["asymptote"] == pytest.approx(2.5, abs=0.1)
        assert params["rate"] == pytest.approx(0.7, abs=0.1)
        if growth_type != "exponential":
            assert params["inflection"] == pytest.approx(4.0, abs=0.1)
        residuals = observations - model.compute_theta(
            time_values,
            asymptote=params["asymptote"],
            rate=params["rate"],
            inflection=params["inflection"],
        )
        assert params["sse"] == pytest.approx(float(residuals @ residuals))
        assert params["sse"] <= float((observations - truth) @ (observations - truth))

    def test_fit_individual_allows_declining_trajectories(self):
        """A negative asymptote is estimated rather than clamped."""
        model = NonlinearGrowthModel(growth_type="logistic")
        time_values = np.linspace(0.0, 8.0, 12)
        observations = model.compute_theta(
            time_values, asymptote=-2.0, rate=0.5, inflection=4.0
        )

        params = model.fit_individual(time_values, observations)

        assert params["asymptote"] == pytest.approx(-2.0, abs=1e-6)
        assert params["rate"] == pytest.approx(0.5, abs=1e-6)
        assert params["inflection"] == pytest.approx(4.0, abs=1e-6)

    def test_fit_individual_reports_nonconvergence_at_iteration_limit(self):
        """Exhausting max_iter returns finite estimates flagged as unconverged."""
        model = NonlinearGrowthModel(growth_type="logistic")
        time_values = np.linspace(0.0, 8.0, 17)

        params = model.fit_individual(time_values, np.full(17, 1.3), max_iter=5)

        assert params["converged"] is False
        assert np.isfinite([params[key] for key in ("asymptote", "rate", "sse")]).all()

    def test_fit_individual_reports_rate_stuck_at_bound(self):
        """A rate pinned to its bound is not reported as converged."""
        model = NonlinearGrowthModel(growth_type="logistic", rate=1e12)
        time_values = np.linspace(0.0, 8.0, 17)
        observations = model.compute_theta(
            time_values, asymptote=2.0, rate=0.5, inflection=3.0
        )

        params = model.fit_individual(time_values, observations)

        assert params["converged"] is False
        assert params["rate"] == pytest.approx(1e8)

    def test_fit_individual_rejects_curves_overflowing_at_start(self):
        """Overflowing starting curves raise a clear error instead of scipy's."""
        model = NonlinearGrowthModel(growth_type="exponential")

        with pytest.raises(ValueError, match="overflows at the starting values"):
            model.fit_individual(np.linspace(-800.0, 0.0, 9), np.linspace(0, 1, 9))

    @pytest.mark.parametrize(
        ("time_values", "observations", "kwargs", "message"),
        [
            (np.zeros((2, 2)), np.zeros(4), {}, "1D"),
            (np.arange(5.0), np.zeros(4), {}, "same length"),
            (np.arange(5.0), np.array([0.0, 1.0, np.nan, 1.0, 1.0]), {}, "finite"),
            (np.arange(3.0), np.zeros(3), {}, "at least 4 observations"),
            (np.arange(5.0), np.zeros(5), {"max_iter": 0}, "max_iter"),
            (np.arange(5.0), np.zeros(5), {"max_iter": 2.5}, "max_iter"),
        ],
    )
    def test_fit_individual_rejects_invalid_inputs(
        self, time_values, observations, kwargs, message
    ):
        """Malformed trajectories are rejected before optimization."""
        model = NonlinearGrowthModel(growth_type="logistic")

        with pytest.raises(ValueError, match=message):
            model.fit_individual(time_values, observations, **kwargs)

    def test_fit_individual_exponential_needs_three_observations(self):
        """Exponential curves have two free parameters."""
        model = NonlinearGrowthModel(growth_type="exponential")

        with pytest.raises(ValueError, match="at least 3 observations"):
            model.fit_individual(np.arange(2.0), np.zeros(2))
        params = model.fit_individual(
            np.arange(3.0), model.compute_theta(np.arange(3.0), 1.5, 0.4)
        )
        assert params["asymptote"] == pytest.approx(1.5, abs=1e-6)
        assert params["rate"] == pytest.approx(0.4, abs=1e-6)


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


def _dense_growth_mixture_reference(model, observations, time_values, prediction_times):
    """Per-person dense-covariance reference for the masked Woodbury algebra."""
    n_persons = observations.shape[0]
    trajectories = np.vstack(
        [model.compute_class_trajectory(k, time_values) for k in range(model.n_classes)]
    )
    prediction_trajectories = np.vstack(
        [
            model.compute_class_trajectory(k, prediction_times)
            for k in range(model.n_classes)
        ]
    )
    design = np.column_stack([np.ones_like(time_values), time_values])
    if model.growth_type == "quadratic":
        design = np.column_stack([design, time_values**2])
    elif model.growth_type == "piecewise":
        hinge = np.maximum(time_values - model.changepoint, 0.0)
        design = np.column_stack([design, hinge])

    log_likelihoods = np.empty((n_persons, model.n_classes))
    class_means = np.empty((n_persons, model.n_classes, prediction_times.size))
    conditional_variances = np.empty((n_persons, prediction_times.size))
    precision_grams = np.empty((n_persons, design.shape[1], design.shape[1]))
    precision_observations = np.empty((n_persons, design.shape[1]))
    for person, row in enumerate(observations):
        observed = ~np.isnan(row)
        times = time_values[observed]
        covariance = (
            model.intercept_var * np.ones((times.size, times.size))
            + model.slope_var * np.outer(times, times)
            + model.residual_variance * np.eye(times.size)
        )
        cross = model.intercept_var + model.slope_var * np.outer(
            prediction_times, times
        )
        _, log_determinant = np.linalg.slogdet(covariance)
        for k in range(model.n_classes):
            residual = row[observed] - trajectories[k, observed]
            log_likelihoods[person, k] = -0.5 * (
                residual @ np.linalg.solve(covariance, residual)
                + times.size * np.log(2.0 * np.pi)
                + log_determinant
            )
            class_means[person, k] = prediction_trajectories[k] + cross @ (
                np.linalg.solve(covariance, residual)
            )
        conditional_variances[person] = (
            model.intercept_var
            + model.slope_var * prediction_times**2
            - np.einsum("ij,ji->i", cross, np.linalg.solve(covariance, cross.T))
        )
        observed_design = design[observed]
        precision_grams[person] = observed_design.T @ np.linalg.solve(
            covariance, observed_design
        )
        precision_observations[person] = observed_design.T @ np.linalg.solve(
            covariance, row[observed]
        )

    log_joint = log_likelihoods + np.log(model.class_proportions)
    posteriors = np.exp(log_joint - log_joint.max(axis=1, keepdims=True))
    posteriors /= posteriors.sum(axis=1, keepdims=True)
    means = np.einsum("nk,nkp->np", posteriors, class_means)
    variances = conditional_variances + np.einsum(
        "nk,nkp->np", posteriors, (class_means - means[:, None, :]) ** 2
    )
    normal_matrices = np.einsum("nk,nab->kab", posteriors, precision_grams)
    right_hand_sides = posteriors.T @ precision_observations
    coefficients = np.linalg.solve(normal_matrices, right_hand_sides[..., None])[..., 0]
    return log_likelihoods, posteriors, means, variances, coefficients


class TestGrowthMixtureMaskedLikelihood:
    """Vectorized masked-Woodbury algebra against dense per-person references."""

    @pytest.mark.parametrize("growth_type", ["linear", "quadratic", "piecewise"])
    @pytest.mark.parametrize(
        ("intercept_var", "slope_var"),
        [(0.5, 0.1), (0.0, 0.1), (0.5, 0.0), (0.0, 0.0)],
    )
    @pytest.mark.parametrize("missing_rate", [0.0, 0.15, 0.4])
    def test_matches_dense_reference(
        self, growth_type, intercept_var, slope_var, missing_rate
    ):
        """Likelihoods, predictions and one EM update match dense algebra."""
        model = GrowthMixtureModel(
            n_classes=3,
            growth_type=growth_type,
            class_proportions=np.array([0.3, 0.45, 0.25]),
            class_intercepts=np.array([-1.5, 0.0, 1.5]),
            class_slopes=np.array([0.3, -0.2, 0.1]),
            class_quadratics=np.array([0.04, -0.02, 0.01]),
            class_post_slopes=np.array([-0.3, 0.4, 0.2]),
            changepoint=1.5,
            intercept_var=intercept_var,
            slope_var=slope_var,
            residual_variance=0.25,
        )
        time_values = np.linspace(-1.0, 4.0, 8)
        rng = np.random.default_rng(
            [len(growth_type), int(intercept_var * 10), int(slope_var * 10)]
        )
        observations, _ = model.simulate(60, time_values, seed=rng)
        if missing_rate:
            observations[rng.random(observations.shape) < missing_rate] = np.nan
            observations[:3] = np.nan
            observations[:3, [1, 4, 6]] = [0.2, -0.1, 0.6]
            observations[3, :] = np.nan
            observations[3, 2] = 0.4
            observations[4, :] = np.nan
            observations[4, [0, 7]] = [-1.0, 1.0]
            observations[5] = model.simulate(1, time_values, seed=rng)[0][0]
            observations = observations[np.isfinite(observations).any(axis=1)]
        prediction_times = np.array([-2.0, 0.5, 3.0, 6.0])

        (
            expected_log_likelihoods,
            expected_posteriors,
            expected_means,
            expected_variances,
            expected_coefficients,
        ) = _dense_growth_mixture_reference(
            model, observations, time_values, prediction_times
        )

        np.testing.assert_allclose(
            model.class_log_likelihood(observations, time_values),
            expected_log_likelihoods,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            model.posterior_probabilities(observations, time_values),
            expected_posteriors,
            rtol=1e-10,
            atol=1e-12,
        )
        means, variances = model.predict_trajectory_moments(
            observations, time_values, prediction_times
        )
        np.testing.assert_allclose(means, expected_means, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(variances, expected_variances, rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(
            model.predict_trajectories(observations, time_values, prediction_times),
            expected_means,
            rtol=1e-10,
            atol=1e-10,
        )
        _, residual_variances = model.predict_trajectory_moments(
            observations, time_values, prediction_times, include_residual=True
        )
        np.testing.assert_allclose(
            residual_variances,
            expected_variances + model.residual_variance,
            rtol=1e-9,
            atol=1e-10,
        )

        model.fit_em(observations, time_values, max_iter=1)

        np.testing.assert_allclose(
            model.class_proportions, expected_posteriors.mean(axis=0), rtol=1e-10
        )
        np.testing.assert_allclose(
            model.class_intercepts, expected_coefficients[:, 0], rtol=1e-9, atol=1e-10
        )
        np.testing.assert_allclose(
            model.class_slopes, expected_coefficients[:, 1], rtol=1e-9, atol=1e-10
        )
        if growth_type == "quadratic":
            np.testing.assert_allclose(
                model.class_quadratics,
                expected_coefficients[:, 2],
                rtol=1e-9,
                atol=1e-10,
            )
        elif growth_type == "piecewise":
            np.testing.assert_allclose(
                model.class_post_slopes,
                expected_coefficients[:, 1] + expected_coefficients[:, 2],
                rtol=1e-9,
                atol=1e-10,
            )

    def test_fit_does_not_depend_on_row_order_or_pattern_count(self):
        """Every row has its own pattern yet the fit equals the permuted fit."""
        source = GrowthMixtureModel(
            n_classes=2,
            class_intercepts=np.array([-1.0, 1.0]),
            class_slopes=np.array([0.4, -0.2]),
            intercept_var=0.3,
            slope_var=0.05,
            residual_variance=0.2,
        )
        time_values = np.arange(14.0)
        observations, _ = source.simulate(300, time_values, seed=11)
        rng = np.random.default_rng(12)
        observations[rng.random(observations.shape) < 0.3] = np.nan
        observations[:, 0] = rng.normal(size=300)
        assert np.unique(np.isnan(observations), axis=0).shape[0] > 250
        order = rng.permutation(300)
        first = GrowthMixtureModel(
            n_classes=2,
            class_intercepts=np.array([-0.5, 0.5]),
            intercept_var=0.3,
            slope_var=0.05,
            residual_variance=0.2,
        )
        second = GrowthMixtureModel(
            n_classes=2,
            class_intercepts=np.array([-0.5, 0.5]),
            intercept_var=0.3,
            slope_var=0.05,
            residual_variance=0.2,
        )

        result = first.fit(observations, time_values, max_iter=200, tol=1e-10)
        permuted = second.fit(observations[order], time_values, max_iter=200, tol=1e-10)

        assert result.converged and permuted.converged
        np.testing.assert_allclose(first.class_intercepts, second.class_intercepts)
        np.testing.assert_allclose(first.class_slopes, second.class_slopes)
        np.testing.assert_allclose(result.posteriors[order], permuted.posteriors)
        assert result.log_likelihood == pytest.approx(permuted.log_likelihood)

    def test_complete_data_without_random_effects_is_unchanged(self):
        """The cdist fast path still produces the closed-form normal density."""
        model = GrowthMixtureModel(
            n_classes=2, intercept_var=0.0, slope_var=0.0, residual_variance=0.4
        )
        time_values = np.arange(5.0)
        observations = np.random.default_rng(3).normal(size=(7, 5))
        trajectories = np.vstack(
            [model.compute_class_trajectory(k, time_values) for k in range(2)]
        )
        squared = ((observations[:, None, :] - trajectories[None]) ** 2).sum(axis=2)
        expected = -0.5 * (squared / 0.4 + 5 * np.log(2.0 * np.pi) + 5 * np.log(0.4))

        np.testing.assert_allclose(
            model.class_log_likelihood(observations, time_values),
            expected,
            rtol=1e-14,
        )


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
