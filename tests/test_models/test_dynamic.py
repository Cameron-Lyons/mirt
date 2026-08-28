"""Tests for dynamic IRT models module."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.models.dynamic import (
    BKTModel,
    BKTResult,
    GrowthMixtureModel,
    GrowthMixtureResult,
    LongitudinalIRTModel,
    LongitudinalResult,
    NonlinearGrowthModel,
    PiecewiseGrowthModel,
    StateSpaceIRT,
)


class TestBKTModel:
    def test_default_initialization(self):
        model = BKTModel(n_skills=3)
        assert model.n_skills == 3
        assert len(model.skill_names) == 3
        assert model.p_init.shape == (3,)
        assert model.p_learn.shape == (3,)
        assert model.p_slip.shape == (3,)
        assert model.p_guess.shape == (3,)
        assert_allclose(model.p_forget, np.zeros(3))

    def test_allow_forgetting(self):
        model = BKTModel(n_skills=2, allow_forgetting=True)
        assert np.all(model.p_forget > 0)

    def test_custom_skill_names(self):
        model = BKTModel(n_skills=2, skill_names=["Add", "Subtract"])
        assert model.skill_names == ["Add", "Subtract"]

    def test_transition_matrix(self):
        model = BKTModel(n_skills=1)
        T = model.transition_matrix(0)
        assert T.shape == (2, 2)
        assert_allclose(T.sum(axis=1), [1.0, 1.0])

    def test_transition_matrix_with_forgetting(self):
        model = BKTModel(n_skills=1, allow_forgetting=True)
        T = model.transition_matrix(0)
        assert T[1, 0] > 0

    def test_emission_probability(self):
        model = BKTModel(n_skills=1)
        p_correct_learned = model.emission_probability(1, 1, 0)
        p_correct_unlearned = model.emission_probability(1, 0, 0)
        assert p_correct_learned == 1 - model.p_slip[0]
        assert p_correct_unlearned == model.p_guess[0]

    def test_emission_probability_incorrect(self):
        model = BKTModel(n_skills=1)
        p_incorrect_learned = model.emission_probability(0, 1, 0)
        p_incorrect_unlearned = model.emission_probability(0, 0, 0)
        assert p_incorrect_learned == model.p_slip[0]
        assert p_incorrect_unlearned == 1 - model.p_guess[0]

    def test_forward(self):
        model = BKTModel(n_skills=1)
        responses = np.array([1, 1, 0, 1, 1])
        skills = np.zeros(5, dtype=int)
        alpha, scaling = model.forward(responses, skills)
        assert alpha.shape == (5, 2)
        assert scaling.shape == (5,)
        assert np.all(scaling > 0)
        assert_allclose(alpha.sum(axis=1), np.ones(5), atol=1e-10)

    def test_backward(self):
        model = BKTModel(n_skills=1)
        responses = np.array([1, 0, 1, 1])
        skills = np.zeros(4, dtype=int)
        _, scaling = model.forward(responses, skills)
        beta = model.backward(responses, skills, scaling)
        assert beta.shape == (4, 2)
        assert_allclose(beta[-1], [1.0, 1.0])

    def test_forward_backward(self):
        model = BKTModel(n_skills=1)
        responses = np.array([1, 1, 0, 1, 1])
        skills = np.zeros(5, dtype=int)
        gamma, log_lik = model.forward_backward(responses, skills)
        assert gamma.shape == (5, 2)
        assert_allclose(gamma.sum(axis=1), np.ones(5), atol=1e-10)
        assert np.isfinite(log_lik)

    def test_mastery_increases_with_correct(self):
        model = BKTModel(n_skills=1)
        responses = np.array([1, 1, 1, 1, 1])
        skills = np.zeros(5, dtype=int)
        gamma, _ = model.forward_backward(responses, skills)
        for t in range(1, 5):
            assert gamma[t, 1] >= gamma[t - 1, 1] - 0.01

    def test_viterbi(self):
        model = BKTModel(n_skills=1)
        responses = np.array([0, 0, 1, 1, 1])
        skills = np.zeros(5, dtype=int)
        path = model.viterbi(responses, skills)
        assert path.shape == (5,)
        assert set(path.tolist()).issubset({0, 1})

    def test_predict_mastery(self):
        model = BKTModel(n_skills=1)
        responses = np.array([1, 1, 1, 1, 1])
        skills = np.zeros(5, dtype=int)
        mastery = model.predict_mastery(responses, skills)
        assert 0 <= mastery <= 1

    def test_predict_mastery_high_after_correct(self):
        model = BKTModel(n_skills=1, p_learn=np.array([0.3]))
        responses = np.ones(20, dtype=int)
        skills = np.zeros(20, dtype=int)
        mastery = model.predict_mastery(responses, skills)
        assert mastery > 0.5

    def test_simulate(self):
        n_skills, n_persons, n_trials_per_skill = 2, 10, 5
        model = BKTModel(n_skills=n_skills)
        responses, skills, states = model.simulate(
            n_persons, n_trials_per_skill, seed=42
        )
        n_total_trials = n_skills * n_trials_per_skill
        assert responses.shape == (n_persons, n_total_trials)
        assert skills.shape == (n_total_trials,)
        assert states.shape == (n_persons, n_total_trials)
        assert set(np.unique(responses)).issubset({0, 1})
        assert set(np.unique(states)).issubset({0, 1})

    def test_simulate_skill_assignments(self):
        n_skills, n_trials_per_skill = 3, 4
        model = BKTModel(n_skills=n_skills)
        _, skills, _ = model.simulate(5, n_trials_per_skill, seed=42)
        assert len(skills) == n_skills * n_trials_per_skill
        for j in range(n_skills):
            start = j * n_trials_per_skill
            end = start + n_trials_per_skill
            assert_allclose(skills[start:end], np.full(n_trials_per_skill, j))

    def test_summary(self):
        model = BKTModel(n_skills=2)
        summary = model.summary()
        assert "BKT Model Summary" in summary
        assert "Skill_0" in summary


class TestLongitudinalIRTModel:
    def test_default_initialization(self):
        model = LongitudinalIRTModel(n_items=5, n_timepoints=4)
        assert model.n_items == 5
        assert model.n_timepoints == 4
        assert model.discrimination.shape == (5,)
        assert model.difficulty.shape == (5,)
        assert model.n_growth_factors == 2

    def test_quadratic_growth(self):
        model = LongitudinalIRTModel(
            n_items=5, n_timepoints=4, growth_model="quadratic"
        )
        assert model.n_growth_factors == 3
        assert model.growth_mean.shape == (3,)
        assert model.growth_cov.shape == (3, 3)

    def test_compute_theta_linear(self):
        model = LongitudinalIRTModel(n_items=5, n_timepoints=4)
        growth_factors = np.array([[0.0, 0.5], [1.0, -0.3]])
        theta = model.compute_theta(growth_factors)
        assert theta.shape == (2, 4)
        assert_allclose(theta[0, 0], 0.0)
        assert_allclose(theta[0, 1], 0.5)
        assert_allclose(theta[0, 2], 1.0)

    def test_compute_theta_quadratic(self):
        model = LongitudinalIRTModel(
            n_items=5, n_timepoints=4, growth_model="quadratic"
        )
        growth_factors = np.array([[0.0, 1.0, -0.1]])
        theta = model.compute_theta(growth_factors)
        assert theta.shape == (1, 4)
        t = np.arange(4.0)
        expected = 0.0 + 1.0 * t + (-0.1) * t**2
        assert_allclose(theta[0], expected)

    def test_compute_theta_with_residuals(self):
        model = LongitudinalIRTModel(n_items=5, n_timepoints=3)
        gf = np.array([[0.0, 0.5]])
        residuals = np.array([[0.1, -0.1, 0.2]])
        theta = model.compute_theta(gf, residuals=residuals)
        theta_no_resid = model.compute_theta(gf)
        assert_allclose(theta, theta_no_resid + residuals)

    def test_probability_single_item(self):
        model = LongitudinalIRTModel(n_items=5, n_timepoints=3)
        theta = np.array([0.0, 1.0, -1.0])
        p = model.probability(theta, item_idx=0)
        assert p.shape == (3,)
        assert np.all(p >= 0) and np.all(p <= 1)

    def test_probability_all_items(self):
        model = LongitudinalIRTModel(n_items=5, n_timepoints=3)
        theta = np.array([0.0, 1.0])
        p = model.probability(theta)
        assert p.shape == (2, 5)
        assert np.all(p >= 0) and np.all(p <= 1)

    def test_simulate(self):
        model = LongitudinalIRTModel(n_items=5, n_timepoints=4)
        n_persons = 20
        responses, theta, gf = model.simulate(n_persons, seed=42)
        assert responses.shape == (n_persons, model.n_timepoints, model.n_items)
        assert theta.shape == (n_persons, model.n_timepoints)
        assert gf.shape == (n_persons, model.n_growth_factors)
        assert set(np.unique(responses)).issubset({0, 1})

    def test_summary(self):
        model = LongitudinalIRTModel(n_items=5, n_timepoints=4)
        summary = model.summary()
        assert "Longitudinal IRT" in summary
        assert "Intercept" in summary
        assert "Slope" in summary


class TestStateSpaceIRT:
    @staticmethod
    def _reference_simulation(model, n_persons, seed):
        rng = np.random.default_rng(seed)
        transition = model.transition_matrix[0, 0]
        process_noise = model.process_noise[0, 0]
        theta = np.zeros((n_persons, model.n_timepoints))
        responses = np.zeros(
            (n_persons, model.n_timepoints, model.n_items),
            dtype=np.int32,
        )
        theta[:, 0] = rng.normal(
            model.initial_mean,
            np.sqrt(model.initial_var),
            n_persons,
        )
        for time in range(1, model.n_timepoints):
            theta[:, time] = transition * theta[:, time - 1] + rng.normal(
                0,
                np.sqrt(process_noise),
                n_persons,
            )
        for person in range(n_persons):
            for time in range(model.n_timepoints):
                logits = model.discrimination * (theta[person, time] - model.difficulty)
                probabilities = 1.0 / (1.0 + np.exp(-logits))
                if model.base_model == "3PL":
                    probabilities = (
                        model.guessing + (1.0 - model.guessing) * probabilities
                    )
                responses[person, time] = rng.random(model.n_items) < probabilities
        return responses, theta

    def test_default_initialization(self):
        model = StateSpaceIRT(n_items=5, n_timepoints=4)
        assert model.n_items == 5
        assert model.n_timepoints == 4
        assert model.transition_matrix.shape == (1, 1)
        assert model.process_noise.shape == (1, 1)
        assert model.guessing is None

    def test_3pl_initialization(self):
        model = StateSpaceIRT(n_items=5, n_timepoints=4, base_model="3PL")
        assert model.guessing is not None
        assert model.guessing.shape == (5,)

    def test_extended_kalman_filter(self):
        model = StateSpaceIRT(n_items=5, n_timepoints=4)
        rng = np.random.default_rng(42)
        responses = rng.integers(0, 2, (4, 5))
        means, vars = model.extended_kalman_filter(responses)
        assert means.shape == (4,)
        assert vars.shape == (4,)
        assert np.all(vars > 0)

    def test_ekf_with_missing_data(self):
        model = StateSpaceIRT(n_items=5, n_timepoints=3)
        responses = np.array(
            [
                [1, 0, -1, 1, 0],
                [-1, -1, -1, -1, -1],
                [1, 1, 0, 1, 1],
            ]
        )
        means, vars = model.extended_kalman_filter(responses)
        assert means.shape == (3,)
        assert np.all(np.isfinite(means))

    def test_simulate(self):
        model = StateSpaceIRT(n_items=5, n_timepoints=6)
        n_persons = 20
        responses, theta = model.simulate(n_persons, seed=42)
        assert responses.shape == (n_persons, model.n_timepoints, model.n_items)
        assert theta.shape == (n_persons, model.n_timepoints)
        assert set(np.unique(responses)).issubset({0, 1})

    @pytest.mark.parametrize("base_model", ["2PL", "3PL"])
    def test_vectorized_simulation_preserves_seeded_draws(
        self,
        base_model,
        monkeypatch,
    ):
        """Chunked broadcasting preserves the prior seeded response stream."""
        from mirt.models import dynamic as dynamic_module

        model = StateSpaceIRT(
            n_items=7,
            n_timepoints=5,
            base_model=base_model,
            discrimination=np.linspace(0.6, 1.8, 7),
            difficulty=np.linspace(-1.0, 1.0, 7),
        )
        monkeypatch.setattr(
            dynamic_module,
            "_LONGITUDINAL_MAX_PROBABILITY_VALUES",
            17,
        )
        expected_responses, expected_theta = self._reference_simulation(model, 13, 42)

        responses, theta = model.simulate(13, seed=42)

        np.testing.assert_array_equal(responses, expected_responses)
        np.testing.assert_array_equal(theta, expected_theta)

    @pytest.mark.parametrize("n_persons", [0, -1, True, 1.5])
    def test_simulate_requires_positive_person_count(self, n_persons):
        model = StateSpaceIRT(n_items=3, n_timepoints=2)

        with pytest.raises(ValueError, match="n_persons"):
            model.simulate(n_persons)

    def test_simulate_theta_autocorrelation(self):
        model = StateSpaceIRT(
            n_items=5,
            n_timepoints=10,
            process_noise=np.array([[0.01]]),
        )
        _, theta = model.simulate(100, seed=42)
        corr = np.corrcoef(theta[:, :-1].ravel(), theta[:, 1:].ravel())[0, 1]
        assert corr > 0.5

    def test_summary(self):
        model = StateSpaceIRT(n_items=5, n_timepoints=4)
        summary = model.summary()
        assert "State-Space IRT" in summary


class TestPiecewiseGrowthModel:
    def test_single_piece(self):
        model = PiecewiseGrowthModel(n_pieces=1)
        assert len(model.changepoints) == 0
        assert len(model.slope_means) == 1

    def test_two_pieces(self):
        model = PiecewiseGrowthModel(n_pieces=2)
        assert len(model.changepoints) == 1
        assert len(model.slope_means) == 2

    def test_invalid_changepoints(self):
        with pytest.raises(ValueError, match="changepoints length"):
            PiecewiseGrowthModel(n_pieces=3, changepoints=np.array([1.0]))

    def test_compute_theta_single_piece(self):
        model = PiecewiseGrowthModel(n_pieces=1)
        time_values = np.array([0.0, 1.0, 2.0, 3.0])
        theta = model.compute_theta(
            time_values, intercept=1.0, slopes=np.array([[0.5]])
        )
        assert theta.shape == (4,)
        expected = 1.0 + 0.5 * time_values
        assert_allclose(theta, expected)

    def test_compute_theta_two_pieces(self):
        model = PiecewiseGrowthModel(n_pieces=2, changepoints=np.array([2.0]))
        time_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        theta = model.compute_theta(
            time_values,
            intercept=0.0,
            slopes=np.array([[1.0, 0.5]]),
        )
        assert theta.shape == (5,)
        assert_allclose(theta[0], 0.0)
        assert_allclose(theta[2], 2.0)

    def test_simulate(self):
        model = PiecewiseGrowthModel(n_pieces=2, changepoints=np.array([3.0]))
        time_values = np.arange(6.0)
        theta, intercepts, slopes = model.simulate(10, time_values, seed=42)
        assert theta.shape == (10, 6)
        assert intercepts.shape == (10,)
        assert slopes.shape == (10, 2)


class TestNonlinearGrowthModel:
    def test_logistic_default(self):
        model = NonlinearGrowthModel()
        assert model.growth_type == "logistic"
        assert model.asymptote == 1.0

    def test_exponential(self):
        model = NonlinearGrowthModel(growth_type="exponential")
        t = np.array([0.0])
        theta = model.compute_theta(t)
        assert_allclose(theta, 0.0, atol=1e-10)

    def test_logistic_shape(self):
        model = NonlinearGrowthModel(
            growth_type="logistic", asymptote=2.0, rate=1.0, inflection=5.0
        )
        t = np.linspace(0, 10, 50)
        theta = model.compute_theta(t)
        assert theta.shape == (50,)
        assert theta[-1] > theta[0]
        assert_allclose(theta[np.argmin(np.abs(t - 5.0))], 1.0, atol=0.1)

    def test_gompertz(self):
        model = NonlinearGrowthModel(growth_type="gompertz", asymptote=1.0)
        t = np.linspace(0, 10, 20)
        theta = model.compute_theta(t)
        assert np.all(theta >= 0)
        assert np.all(theta <= model.asymptote + 0.01)

    def test_compute_theta_multiple_persons(self):
        model = NonlinearGrowthModel(growth_type="logistic")
        t = np.linspace(0, 5, 10)
        asymptotes = np.array([1.0, 2.0])
        theta = model.compute_theta(t, asymptote=asymptotes)
        assert theta.shape == (2, 10)

    def test_growth_velocity_logistic(self):
        model = NonlinearGrowthModel(
            growth_type="logistic", asymptote=1.0, rate=1.0, inflection=5.0
        )
        t = np.linspace(0, 10, 50)
        velocity = model.growth_velocity(t)
        assert velocity.shape == (50,)
        assert np.all(velocity >= 0)
        assert np.argmax(velocity) == np.argmin(np.abs(t - 5.0))

    def test_growth_velocity_exponential(self):
        model = NonlinearGrowthModel(growth_type="exponential", asymptote=1.0, rate=0.5)
        t = np.array([0.0, 1.0, 5.0])
        velocity = model.growth_velocity(t)
        assert velocity[0] > velocity[1] > velocity[2]

    def test_simulate(self):
        model = NonlinearGrowthModel(growth_type="logistic")
        t = np.linspace(0, 10, 20)
        theta, params = model.simulate(15, t, seed=42)
        assert theta.shape == (15, 20)
        assert "asymptote" in params
        assert "rate" in params
        assert "inflection" in params
        assert params["asymptote"].shape == (15,)


class TestGrowthMixtureModel:
    def test_default_initialization(self):
        model = GrowthMixtureModel(n_classes=3)
        assert model.n_classes == 3
        assert_allclose(model.class_proportions.sum(), 1.0)
        assert model.class_intercepts.shape == (3,)
        assert model.class_slopes.shape == (3,)

    def test_quadratic_initialization(self):
        model = GrowthMixtureModel(n_classes=2, growth_type="quadratic")
        assert model.class_quadratics.shape == (2,)

    def test_piecewise_initialization(self):
        model = GrowthMixtureModel(n_classes=2, growth_type="piecewise")

        assert_allclose(model.class_post_slopes, model.class_slopes)

    def test_compute_class_trajectory(self):
        model = GrowthMixtureModel(n_classes=2, n_timepoints=5)
        t = np.arange(5.0)
        traj = model.compute_class_trajectory(0, t)
        assert traj.shape == (5,)

    def test_class_trajectories_differ(self):
        model = GrowthMixtureModel(n_classes=2)
        t = np.arange(5.0)
        traj0 = model.compute_class_trajectory(0, t)
        traj1 = model.compute_class_trajectory(1, t)
        assert not np.allclose(traj0, traj1)

    def test_piecewise_trajectory_is_continuous_with_distinct_slopes(self):
        model = GrowthMixtureModel(
            n_classes=1,
            growth_type="piecewise",
            class_intercepts=np.array([1.0]),
            class_slopes=np.array([0.5]),
            class_post_slopes=np.array([-0.25]),
            changepoint=2.0,
        )
        time_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        trajectory = model.compute_class_trajectory(0, time_values)

        assert_allclose(trajectory, np.array([1.0, 1.5, 2.0, 1.75, 1.5]))

    def test_piecewise_trajectory_uses_time_range_midpoint_by_default(self):
        model = GrowthMixtureModel(
            n_classes=1,
            growth_type="piecewise",
            class_intercepts=np.array([0.0]),
            class_slopes=np.array([1.0]),
            class_post_slopes=np.array([2.0]),
        )
        time_values = np.array([10.0, 12.0, 14.0])

        trajectory = model.compute_class_trajectory(0, time_values)

        assert_allclose(trajectory, np.array([10.0, 12.0, 16.0]))

    def test_class_likelihood(self):
        model = GrowthMixtureModel(n_classes=2, n_timepoints=5)
        rng = np.random.default_rng(42)
        obs = rng.standard_normal((10, 5))
        t = np.arange(5.0)
        lik = model.class_likelihood(obs, t)
        assert lik.shape == (10, 2)
        assert np.all(lik >= 0)

    @pytest.mark.parametrize("growth_type", ["linear", "quadratic", "piecewise"])
    def test_class_log_likelihood_matches_scalar_reference(self, growth_type):
        model = GrowthMixtureModel(
            n_classes=3,
            growth_type=growth_type,
            class_intercepts=np.array([-0.8, 0.2, 1.1]),
            class_slopes=np.array([0.1, -0.3, 0.5]),
            class_quadratics=np.array([0.04, -0.02, 0.01]),
            class_post_slopes=np.array([0.6, 0.2, -0.1]),
            changepoint=0.5,
            intercept_var=0.4,
            residual_variance=0.2,
        )
        rng = np.random.default_rng(123)
        observations = rng.normal(size=(12, 7))
        time_values = np.linspace(-1.0, 2.0, 7)
        covariance = (
            model.intercept_var * np.ones((len(time_values), len(time_values)))
            + model.slope_var * np.outer(time_values, time_values)
            + model.residual_variance * np.eye(len(time_values))
        )
        _, log_determinant = np.linalg.slogdet(covariance)

        expected = np.empty((len(observations), model.n_classes))
        for person_index, observation in enumerate(observations):
            for class_index in range(model.n_classes):
                residual = observation - model.compute_class_trajectory(
                    class_index, time_values
                )
                expected[person_index, class_index] = -0.5 * (
                    residual @ np.linalg.solve(covariance, residual)
                    + len(time_values) * np.log(2.0 * np.pi)
                    + log_determinant
                )

        actual = model.class_log_likelihood(observations, time_values)

        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        assert_allclose(
            model.class_likelihood(observations, time_values), np.exp(expected)
        )

    def test_long_trajectory_posteriors_remain_normalized(self):
        model = GrowthMixtureModel(
            n_classes=2,
            class_proportions=np.array([0.4, 0.6]),
            class_intercepts=np.array([-1.0, 1.0]),
            class_slopes=np.array([0.1, 0.2]),
        )
        time_values = np.arange(2_000, dtype=np.float64)
        observation = 100.0 * (-1.0) ** time_values

        likelihoods = model.class_likelihood(observation, time_values)
        posteriors = model.posterior_probabilities(observation, time_values)

        assert likelihoods.shape == (1, 2)
        assert np.all(likelihoods == 0.0)
        assert np.all(np.isfinite(posteriors))
        assert_allclose(posteriors.sum(axis=1), 1.0)
        assert model.classify(observation, time_values)[0] == np.argmax(posteriors[0])

    def test_class_log_likelihood_preserves_small_large_offset_residuals(self):
        model = GrowthMixtureModel(
            n_classes=1,
            class_intercepts=np.array([1e8]),
            class_slopes=np.array([0.0]),
        )
        observations = np.array([[1e8, 1e8 + 1.0, 1e8 - 1.0]])
        time_values = np.arange(3.0)
        covariance = (
            model.intercept_var * np.ones((len(time_values), len(time_values)))
            + model.slope_var * np.outer(time_values, time_values)
            + model.residual_variance * np.eye(len(time_values))
        )
        residual = observations[0] - model.compute_class_trajectory(0, time_values)
        _, log_determinant = np.linalg.slogdet(covariance)
        expected = -0.5 * (
            residual @ np.linalg.solve(covariance, residual)
            + len(time_values) * np.log(2.0 * np.pi)
            + log_determinant
        )

        actual = model.class_log_likelihood(observations, time_values)

        assert actual[0, 0] == pytest.approx(expected)

    def test_class_log_likelihood_uses_slope_variance(self):
        observations = np.array([[0.0, 0.4, -0.2, 0.8]])
        time_values = np.arange(4.0)
        without_random_slopes = GrowthMixtureModel(n_classes=2, slope_var=0.0)
        with_random_slopes = GrowthMixtureModel(n_classes=2, slope_var=2.0)

        without = without_random_slopes.class_log_likelihood(
            observations,
            time_values,
        )
        with_slope = with_random_slopes.class_log_likelihood(
            observations,
            time_values,
        )

        assert not np.allclose(without, with_slope)

    def test_class_log_likelihood_without_random_effects_matches_independent_normal(
        self,
    ):
        model = GrowthMixtureModel(
            n_classes=2,
            intercept_var=0.0,
            slope_var=0.0,
            residual_variance=0.3,
        )
        observations = np.array([[0.2, -0.1, 0.7], [-0.4, 0.3, 1.2]])
        time_values = np.arange(3.0)
        expected = np.empty((2, 2))
        for person_index, observation in enumerate(observations):
            for class_index in range(model.n_classes):
                residual = observation - model.compute_class_trajectory(
                    class_index,
                    time_values,
                )
                expected[person_index, class_index] = -0.5 * (
                    np.sum(residual**2) / model.residual_variance
                    + len(time_values) * np.log(2.0 * np.pi * model.residual_variance)
                )

        actual = model.class_log_likelihood(observations, time_values)

        assert_allclose(actual, expected)

    def test_class_log_likelihood_requires_positive_residual_variance(self):
        model = GrowthMixtureModel(
            n_classes=2,
            residual_variance=0.0,
        )

        with pytest.raises(ValueError, match="residual_variance must be positive"):
            model.class_log_likelihood(np.zeros((2, 3)), np.arange(3.0))

    @pytest.mark.parametrize(
        "time_values",
        [np.array([2.0]), np.ones(4)],
    )
    def test_class_log_likelihood_handles_rank_deficient_random_effect_basis(
        self,
        time_values,
    ):
        model = GrowthMixtureModel(
            n_classes=2,
            intercept_var=0.4,
            slope_var=0.3,
            residual_variance=0.2,
        )
        observations = np.random.default_rng(6).normal(size=(5, len(time_values)))
        covariance = (
            model.intercept_var * np.ones((len(time_values), len(time_values)))
            + model.slope_var * np.outer(time_values, time_values)
            + model.residual_variance * np.eye(len(time_values))
        )
        _, log_determinant = np.linalg.slogdet(covariance)
        expected = np.empty((len(observations), model.n_classes))
        for person_index, observation in enumerate(observations):
            for class_index in range(model.n_classes):
                residual = observation - model.compute_class_trajectory(
                    class_index,
                    time_values,
                )
                expected[person_index, class_index] = -0.5 * (
                    residual @ np.linalg.solve(covariance, residual)
                    + len(time_values) * np.log(2.0 * np.pi)
                    + log_determinant
                )

        actual = model.class_log_likelihood(observations, time_values)

        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_classify(self):
        model = GrowthMixtureModel(n_classes=2, n_timepoints=5)
        rng = np.random.default_rng(42)
        obs = rng.standard_normal((10, 5))
        t = np.arange(5.0)
        classes = model.classify(obs, t)
        assert classes.shape == (10,)
        assert set(np.unique(classes)).issubset({0, 1})

    def test_posterior_probabilities(self):
        model = GrowthMixtureModel(n_classes=3, n_timepoints=5)
        rng = np.random.default_rng(42)
        obs = rng.standard_normal((10, 5))
        t = np.arange(5.0)
        posteriors = model.posterior_probabilities(obs, t)
        assert posteriors.shape == (10, 3)
        assert_allclose(posteriors.sum(axis=1), np.ones(10), atol=1e-6)

    def test_simulate(self):
        model = GrowthMixtureModel(n_classes=2, n_timepoints=5)
        obs, classes = model.simulate(30, seed=42)
        assert obs.shape == (30, 5)
        assert classes.shape == (30,)
        assert set(np.unique(classes)).issubset({0, 1})

    def test_simulate_custom_time(self):
        model = GrowthMixtureModel(n_classes=2, n_timepoints=5)
        t = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
        obs, classes = model.simulate(10, time_values=t, seed=42)
        assert obs.shape == (10, 5)

    @pytest.mark.parametrize("growth_type", ["linear", "quadratic", "piecewise"])
    def test_simulate_matches_seeded_scalar_reference(self, growth_type):
        model = GrowthMixtureModel(
            n_classes=3,
            growth_type=growth_type,
            class_proportions=np.array([0.2, 0.3, 0.5]),
            class_intercepts=np.array([-0.8, 0.2, 1.1]),
            class_slopes=np.array([0.1, -0.3, 0.5]),
            class_quadratics=np.array([0.04, -0.02, 0.01]),
            class_post_slopes=np.array([0.6, 0.2, -0.1]),
            changepoint=0.5,
            intercept_var=0.4,
            slope_var=0.15,
            residual_variance=0.2,
        )
        n_persons = 37
        time_values = np.linspace(-1.0, 2.0, 7)
        seed = 987
        rng = np.random.default_rng(seed)
        expected_classes = rng.choice(
            model.n_classes,
            size=n_persons,
            p=model.class_proportions,
        )
        expected_observations = np.empty((n_persons, len(time_values)))
        for person_index, class_index in enumerate(expected_classes):
            mean = model.compute_class_trajectory(class_index, time_values)
            intercept_deviation = rng.normal(0, np.sqrt(model.intercept_var))
            slope_deviation = rng.normal(0, np.sqrt(model.slope_var))
            expected_observations[person_index] = (
                mean
                + intercept_deviation
                + slope_deviation * time_values
                + rng.normal(
                    0,
                    np.sqrt(model.residual_variance),
                    len(time_values),
                )
            )

        observations, classes = model.simulate(
            n_persons,
            time_values,
            seed=seed,
        )

        assert np.array_equal(classes, expected_classes)
        assert np.array_equal(observations, expected_observations)

    def test_simulate_is_seeded_independently_of_internal_chunks(self, monkeypatch):
        model = GrowthMixtureModel(n_classes=3)
        time_values = np.linspace(0.0, 4.0, 9)
        expected_observations, expected_classes = model.simulate(
            53,
            time_values,
            seed=321,
        )
        monkeypatch.setattr(
            "mirt.models.dynamic._GROWTH_MIXTURE_MAX_RANDOM_VALUES",
            23,
        )

        observations, classes = model.simulate(53, time_values, seed=321)

        assert np.array_equal(classes, expected_classes)
        assert np.array_equal(observations, expected_observations)

    def test_simulate_normalizes_class_weights(self):
        model = GrowthMixtureModel(
            n_classes=2,
            class_proportions=np.array([2.0, 1.0]),
        )
        rng = np.random.default_rng(44)
        expected_classes = rng.choice(2, size=20, p=np.array([2.0, 1.0]) / 3.0)

        _, classes = model.simulate(20, seed=44)

        assert np.array_equal(classes, expected_classes)

    def test_simulate_supports_deterministic_zero_variances(self):
        model = GrowthMixtureModel(
            n_classes=2,
            intercept_var=0.0,
            slope_var=0.0,
            residual_variance=0.0,
        )
        time_values = np.arange(5.0)

        observations, classes = model.simulate(12, time_values, seed=8)

        expected = np.vstack(
            [
                model.compute_class_trajectory(class_index, time_values)
                for class_index in classes
            ]
        )
        assert np.array_equal(observations, expected)

    @pytest.mark.parametrize("n_persons", [0, -1, True, 1.5])
    def test_simulate_validates_n_persons(self, n_persons):
        model = GrowthMixtureModel(n_classes=2)

        with pytest.raises(ValueError, match="positive integer"):
            model.simulate(n_persons)

    @pytest.mark.parametrize("n_timepoints", [0, -1, True, 1.5])
    def test_simulate_validates_default_n_timepoints(self, n_timepoints):
        model = GrowthMixtureModel(n_classes=2, n_timepoints=n_timepoints)

        with pytest.raises(ValueError, match="positive integer"):
            model.simulate(3)

    @pytest.mark.parametrize(
        "time_values",
        [
            np.array([]),
            np.zeros((2, 2)),
            np.array([0.0, np.nan]),
            np.array(["invalid"]),
        ],
    )
    def test_simulate_validates_time_values(self, time_values):
        model = GrowthMixtureModel(n_classes=2)

        with pytest.raises(ValueError):
            model.simulate(3, time_values)

    @pytest.mark.parametrize(
        ("variance_name", "invalid_value"),
        [
            ("intercept_var", -0.1),
            ("slope_var", np.inf),
            ("residual_variance", "invalid"),
        ],
    )
    def test_simulate_validates_variance_components(
        self,
        variance_name,
        invalid_value,
    ):
        model = GrowthMixtureModel(n_classes=2)
        setattr(model, variance_name, invalid_value)

        with pytest.raises(ValueError, match="variance components"):
            model.simulate(3)

    def test_entropy(self):
        model = GrowthMixtureModel(n_classes=2, n_timepoints=5)
        rng = np.random.default_rng(42)
        obs = rng.standard_normal((20, 5))
        t = np.arange(5.0)
        ent = model.entropy(obs, t)
        assert isinstance(ent, float)
        assert ent >= 0

    def test_fit_em(self):
        model = GrowthMixtureModel(n_classes=2, n_timepoints=5)
        obs, _ = model.simulate(50, seed=42)
        t = np.arange(5.0)
        result = model.fit_em(obs, t, max_iter=20)
        assert "classifications" in result
        assert "posteriors" in result
        assert "log_likelihood" in result
        assert "converged" in result

    def test_fit_em_vectorized_quadratic_update_matches_scalar_reference(self):
        model = GrowthMixtureModel(
            n_classes=2,
            growth_type="quadratic",
            class_proportions=np.array([0.45, 0.55]),
            class_intercepts=np.array([-0.7, 0.8]),
            class_slopes=np.array([0.2, -0.1]),
            class_quadratics=np.array([0.03, -0.02]),
        )
        rng = np.random.default_rng(9)
        time_values = np.linspace(-2.0, 2.0, 8)
        observations = rng.normal(size=(30, len(time_values)))
        posteriors = model.posterior_probabilities(observations, time_values)
        design = np.column_stack(
            [np.ones(len(time_values)), time_values, time_values**2]
        )
        expected_coefficients = np.empty((model.n_classes, design.shape[1]))
        for class_index in range(model.n_classes):
            weights = posteriors[:, class_index]
            weighted_mean = weights @ observations / weights.sum()
            expected_coefficients[class_index] = np.linalg.solve(
                design.T @ design,
                design.T @ weighted_mean,
            )

        result = model.fit_em(observations, time_values, max_iter=1)

        assert_allclose(model.class_proportions, posteriors.mean(axis=0))
        assert_allclose(model.class_intercepts, expected_coefficients[:, 0])
        assert_allclose(model.class_slopes, expected_coefficients[:, 1])
        assert_allclose(model.class_quadratics, expected_coefficients[:, 2])
        assert np.isfinite(result["log_likelihood"])

    def test_fit_em_vectorized_piecewise_update_matches_scalar_reference(self):
        model = GrowthMixtureModel(
            n_classes=2,
            growth_type="piecewise",
            class_proportions=np.array([0.45, 0.55]),
            class_intercepts=np.array([-0.7, 0.8]),
            class_slopes=np.array([0.2, -0.1]),
            class_post_slopes=np.array([0.5, -0.4]),
            changepoint=0.25,
        )
        rng = np.random.default_rng(19)
        time_values = np.linspace(-2.0, 2.0, 8)
        observations = rng.normal(size=(30, len(time_values)))
        posteriors = model.posterior_probabilities(observations, time_values)
        hinge = np.maximum(time_values - model.changepoint, 0.0)
        design = np.column_stack([np.ones(len(time_values)), time_values, hinge])
        expected_coefficients = np.empty((model.n_classes, design.shape[1]))
        for class_index in range(model.n_classes):
            weights = posteriors[:, class_index]
            weighted_mean = weights @ observations / weights.sum()
            expected_coefficients[class_index] = np.linalg.solve(
                design.T @ design,
                design.T @ weighted_mean,
            )

        result = model.fit_em(observations, time_values, max_iter=1)

        assert_allclose(model.class_proportions, posteriors.mean(axis=0))
        assert_allclose(model.class_intercepts, expected_coefficients[:, 0])
        assert_allclose(model.class_slopes, expected_coefficients[:, 1])
        assert_allclose(
            model.class_post_slopes,
            expected_coefficients[:, 1] + expected_coefficients[:, 2],
        )
        assert np.isfinite(result["log_likelihood"])

    def test_piecewise_fit_recovers_segment_slopes_and_classes(self):
        time_values = np.arange(6.0)
        source = GrowthMixtureModel(
            n_classes=2,
            growth_type="piecewise",
            class_proportions=np.array([0.5, 0.5]),
            class_intercepts=np.array([-1.0, 1.0]),
            class_slopes=np.array([0.2, -0.2]),
            class_post_slopes=np.array([0.8, -0.8]),
            changepoint=2.0,
            intercept_var=0.05,
            slope_var=0.01,
            residual_variance=0.05,
        )
        observations, expected_classes = source.simulate(
            500,
            time_values,
            seed=42,
        )
        model = GrowthMixtureModel(
            n_classes=2,
            growth_type="piecewise",
            class_intercepts=np.array([-0.7, 0.7]),
            class_slopes=np.array([0.1, -0.1]),
            class_post_slopes=np.array([0.5, -0.5]),
            changepoint=2.0,
            intercept_var=0.05,
            slope_var=0.01,
            residual_variance=0.05,
        )

        result = model.fit(observations, time_values, max_iter=50)

        assert np.mean(result.classifications == expected_classes) > 0.95
        assert_allclose(model.class_intercepts, source.class_intercepts, atol=0.05)
        assert_allclose(model.class_slopes, source.class_slopes, atol=0.05)
        assert_allclose(model.class_post_slopes, source.class_post_slopes, atol=0.05)
        assert result.converged is True

    def test_fit_em_reports_convergence_on_last_allowed_iteration(self):
        model = GrowthMixtureModel(n_classes=2)
        observations = np.zeros((4, 5))

        result = model.fit_em(
            observations,
            np.arange(5.0),
            max_iter=1,
            tol=1e9,
        )

        assert result["converged"] is True

    @pytest.mark.parametrize(
        ("growth_type", "expected"),
        [("linear", 8), ("piecewise", 11), ("quadratic", 11)],
    )
    def test_n_fitted_parameters(self, growth_type, expected):
        model = GrowthMixtureModel(n_classes=3, growth_type=growth_type)

        assert model.n_fitted_parameters == expected

    def test_fit_returns_structured_diagnostics(self):
        source = GrowthMixtureModel(
            n_classes=2,
            class_intercepts=np.array([-1.0, 1.0]),
            class_slopes=np.array([0.2, 0.4]),
        )
        time_values = np.arange(5.0)
        observations, _ = source.simulate(40, time_values, seed=14)
        model = GrowthMixtureModel(n_classes=2)

        result = model.fit(observations, time_values, max_iter=5)

        assert isinstance(result, GrowthMixtureResult)
        assert result.model is model
        assert result.classifications.shape == (40,)
        assert result.posteriors.shape == (40, 2)
        assert result.n_observations == 40
        assert result.n_parameters == model.n_fitted_parameters == 5
        assert result.aic == pytest.approx(
            2 * result.n_parameters - 2 * result.log_likelihood
        )
        assert result.bic == pytest.approx(
            np.log(result.n_observations) * result.n_parameters
            - 2 * result.log_likelihood
        )
        assert result.entropy == pytest.approx(model.entropy(observations, time_values))
        assert result.n_iterations <= 5

    def test_fit_matches_fit_em_final_state(self):
        time_values = np.arange(5.0)
        observations = np.random.default_rng(22).normal(size=(25, 5))
        mapping_model = GrowthMixtureModel(n_classes=2)
        result_model = GrowthMixtureModel(n_classes=2)

        mapping = mapping_model.fit_em(
            observations,
            time_values,
            max_iter=4,
        )
        result = result_model.fit(
            observations,
            time_values,
            max_iter=4,
        )

        assert np.array_equal(result.classifications, mapping["classifications"])
        assert_allclose(result.posteriors, mapping["posteriors"])
        assert result.log_likelihood == pytest.approx(mapping["log_likelihood"])
        assert result.converged is mapping["converged"]
        assert result.n_iterations == mapping["n_iterations"]
        assert_allclose(result_model.class_proportions, mapping_model.class_proportions)
        assert_allclose(result_model.class_intercepts, mapping_model.class_intercepts)
        assert_allclose(result_model.class_slopes, mapping_model.class_slopes)

    @pytest.mark.parametrize(
        ("observations", "time_values", "message"),
        [
            (np.empty((0, 3)), np.arange(3.0), "non-empty"),
            (np.zeros((2, 3)), np.arange(2.0), "one value per observation"),
            (np.array([[0.0, np.nan]]), np.arange(2.0), "finite"),
            (np.zeros((2, 2)), np.array([0.0, np.inf]), "finite"),
        ],
    )
    def test_likelihood_validates_trajectory_data(
        self, observations, time_values, message
    ):
        model = GrowthMixtureModel(n_classes=2)

        with pytest.raises(ValueError, match=message):
            model.class_log_likelihood(observations, time_values)

    @pytest.mark.parametrize(
        "fit_options",
        [
            {"max_iter": 0},
            {"max_iter": True},
            {"max_iter": 1.5},
            {"tol": 0.0},
            {"tol": np.nan},
            {"tol": "invalid"},
        ],
    )
    def test_fit_em_validates_controls(self, fit_options):
        model = GrowthMixtureModel(n_classes=2)

        with pytest.raises(ValueError):
            model.fit_em(np.zeros((3, 5)), np.arange(5.0), **fit_options)

    def test_fit_em_rejects_rank_deficient_time_values(self):
        model = GrowthMixtureModel(n_classes=2)

        with pytest.raises(ValueError, match="full-rank"):
            model.fit_em(np.zeros((3, 5)), np.ones(5))

    @pytest.mark.parametrize("changepoint", [np.nan, np.inf, "invalid"])
    def test_piecewise_model_validates_changepoint(self, changepoint):
        model = GrowthMixtureModel(
            n_classes=2,
            growth_type="piecewise",
            changepoint=changepoint,
        )

        with pytest.raises(ValueError, match="changepoint"):
            model.class_log_likelihood(np.zeros((2, 5)), np.arange(5.0))

    @pytest.mark.parametrize(
        "post_slopes",
        [np.array([0.2]), np.array([0.2, np.nan]), np.array(["a", "b"])],
    )
    def test_piecewise_model_validates_post_slopes(self, post_slopes):
        model = GrowthMixtureModel(
            n_classes=2,
            growth_type="piecewise",
            class_post_slopes=post_slopes,
        )

        with pytest.raises(ValueError, match="post slopes"):
            model.class_log_likelihood(np.zeros((2, 5)), np.arange(5.0))

    def test_posterior_probabilities_validate_class_proportions(self):
        model = GrowthMixtureModel(
            n_classes=2,
            class_proportions=np.zeros(2),
        )

        with pytest.raises(ValueError, match="positive value"):
            model.posterior_probabilities(np.zeros((2, 5)), np.arange(5.0))


class TestBKTResult:
    def test_summary(self):
        model = BKTModel(n_skills=2)
        result = BKTResult(
            model=model,
            learning_curves=np.zeros((10, 10)),
            skill_mastery=np.random.default_rng(42).random((10, 2)),
            log_likelihood=-100.0,
            aic=210.0,
            bic=220.0,
            n_observations=100,
            n_parameters=5,
            converged=True,
        )
        summary = result.summary()
        assert "BKT Estimation" in summary
        assert "Log-Likelihood" in summary
        assert "Converged" in summary


class TestLongitudinalResult:
    def test_summary(self):
        model = LongitudinalIRTModel(n_items=5, n_timepoints=4)
        result = LongitudinalResult(
            model=model,
            growth_factors=np.random.default_rng(42).standard_normal((20, 2)),
            theta_trajectories=np.random.default_rng(42).standard_normal((20, 4)),
            growth_factor_se=np.full((20, 2), 0.1),
            log_likelihood=-200.0,
            aic=410.0,
            bic=420.0,
            converged=True,
            n_iterations=50,
        )
        summary = result.summary()
        assert "Longitudinal IRT" in summary
        assert "Intercept" in summary
        assert "Slope" in summary


class TestGrowthMixtureResult:
    def test_summary(self):
        model = GrowthMixtureModel(n_classes=2)
        rng = np.random.default_rng(42)
        result = GrowthMixtureResult(
            model=model,
            classifications=rng.integers(0, 2, 30),
            posteriors=rng.dirichlet([1, 1], 30),
            log_likelihood=-300.0,
            aic=610.0,
            bic=620.0,
            entropy=0.5,
            converged=True,
            n_iterations=20,
        )
        summary = result.summary()
        assert "Growth Mixture" in summary
        assert "Class 0" in summary
        assert "Class 1" in summary
        assert "Entropy" in summary

    def test_class_counts_and_shares(self):
        model = GrowthMixtureModel(n_classes=3)
        result = GrowthMixtureResult(
            model=model,
            classifications=np.array([0, 2, 1, 2, 2]),
            posteriors=np.full((5, 3), 1 / 3),
            log_likelihood=-20.0,
            aic=50.0,
            bic=55.0,
            entropy=0.8,
            converged=True,
            n_iterations=7,
        )

        assert np.array_equal(result.class_counts, np.array([1, 1, 3]))
        assert_allclose(result.class_shares, np.array([0.2, 0.2, 0.6]))
        assert result.n_observations == 5
        assert result.n_parameters == 8

    def test_quadratic_summary_includes_curvature_and_iterations(self):
        model = GrowthMixtureModel(
            n_classes=2,
            growth_type="quadratic",
            class_quadratics=np.array([-0.2, 0.3]),
        )
        result = GrowthMixtureResult(
            model=model,
            classifications=np.array([0, 1, 1]),
            posteriors=np.array([[0.8, 0.2], [0.1, 0.9], [0.2, 0.8]]),
            log_likelihood=-10.0,
            aic=34.0,
            bic=31.0,
            entropy=0.4,
            converged=False,
            n_iterations=12,
        )

        summary = result.summary()

        assert "Observations:       3" in summary
        assert "Fitted Parameters:  7" in summary
        assert "Iterations:         12" in summary
        assert "Quadratic=-0.200" in summary
        assert "Quadratic=0.300" in summary

    def test_piecewise_summary_includes_changepoint_and_segment_slopes(self):
        model = GrowthMixtureModel(
            n_classes=2,
            growth_type="piecewise",
            class_slopes=np.array([0.1, 0.2]),
            class_post_slopes=np.array([0.8, -0.3]),
            changepoint=2.5,
        )
        result = GrowthMixtureResult(
            model=model,
            classifications=np.array([0, 1, 1]),
            posteriors=np.array([[0.8, 0.2], [0.1, 0.9], [0.2, 0.8]]),
            log_likelihood=-10.0,
            aic=34.0,
            bic=31.0,
            entropy=0.4,
            converged=True,
            n_iterations=9,
        )

        summary = result.summary()

        assert "Changepoint:        2.5000" in summary
        assert "Pre-Slope=0.100" in summary
        assert "Post-Slope=0.800" in summary
        assert "Pre-Slope=0.200" in summary
        assert "Post-Slope=-0.300" in summary
