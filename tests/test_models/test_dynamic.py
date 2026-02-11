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

    def test_class_likelihood(self):
        model = GrowthMixtureModel(n_classes=2, n_timepoints=5)
        rng = np.random.default_rng(42)
        obs = rng.standard_normal((10, 5))
        t = np.arange(5.0)
        lik = model.class_likelihood(obs, t)
        assert lik.shape == (10, 2)
        assert np.all(lik >= 0)

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
