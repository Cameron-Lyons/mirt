"""Regression coverage for response-time Gibbs sampling."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats
from scipy.special import logsumexp

from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON
from mirt.estimation.rt_gibbs import (
    _TIME_DISCRIMINATION_MH_STEPS,
    ResponseTimeGibbsSampler,
    RTModelPriors,
)
from mirt.exceptions import MirtValidationError
from mirt.models.response_time import ResponseTimeModel


def _scalar_accuracy_update(
    sampler: ResponseTimeGibbsSampler,
    responses: np.ndarray,
    theta: np.ndarray,
    discrimination: np.ndarray,
    difficulty: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the scalar accuracy update; prior and walk both live on log a."""
    updated_discrimination = discrimination.copy()
    updated_difficulty = difficulty.copy()
    for item_idx in range(len(discrimination)):
        log_disc_current = np.log(discrimination[item_idx])
        log_disc_proposed = log_disc_current + rng.normal(0.0, 0.1)
        disc_proposed = np.exp(log_disc_proposed)
        diff_proposed = difficulty[item_idx] + rng.normal(0.0, 0.1)
        log_like_current = 0.0
        log_like_proposed = 0.0

        for person_idx in range(len(theta)):
            response = responses[person_idx, item_idx]
            if response < 0:
                continue
            current = np.clip(
                sigmoid(
                    discrimination[item_idx]
                    * (theta[person_idx] - difficulty[item_idx])
                ),
                PROB_EPSILON,
                1.0 - PROB_EPSILON,
            )
            proposed = np.clip(
                sigmoid(disc_proposed * (theta[person_idx] - diff_proposed)),
                PROB_EPSILON,
                1.0 - PROB_EPSILON,
            )
            if response == 1:
                log_like_current += np.log(current)
                log_like_proposed += np.log(proposed)
            else:
                log_like_current += np.log(1.0 - current)
                log_like_proposed += np.log(1.0 - proposed)

        log_prior_current = (
            -0.5
            * (log_disc_current - sampler.priors.disc_mean) ** 2
            / sampler.priors.disc_var
            - 0.5
            * (difficulty[item_idx] - sampler.priors.diff_mean) ** 2
            / sampler.priors.diff_var
        )
        log_prior_proposed = (
            -0.5
            * (log_disc_proposed - sampler.priors.disc_mean) ** 2
            / sampler.priors.disc_var
            - 0.5
            * (diff_proposed - sampler.priors.diff_mean) ** 2
            / sampler.priors.diff_var
        )
        log_acceptance = (
            log_like_proposed
            + log_prior_proposed
            - log_like_current
            - log_prior_current
        )
        if np.log(rng.random()) < log_acceptance:
            updated_discrimination[item_idx] = disc_proposed
            updated_difficulty[item_idx] = diff_proposed

    return updated_discrimination, updated_difficulty


def _scalar_guessing_update(
    responses: np.ndarray,
    theta: np.ndarray,
    discrimination: np.ndarray,
    difficulty: np.ndarray,
    guessing: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run the former scalar guessing update as a reference."""
    updated = guessing.copy()
    for item_idx in range(len(guessing)):
        proposed_guess = np.clip(guessing[item_idx] + rng.normal(0.0, 0.02), 0.01, 0.5)
        log_like_current = 0.0
        log_like_proposed = 0.0
        for person_idx in range(len(theta)):
            response = responses[person_idx, item_idx]
            if response < 0:
                continue
            logistic = sigmoid(
                discrimination[item_idx] * (theta[person_idx] - difficulty[item_idx])
            )
            current = np.clip(
                guessing[item_idx] + (1.0 - guessing[item_idx]) * logistic,
                PROB_EPSILON,
                1.0 - PROB_EPSILON,
            )
            proposed = np.clip(
                proposed_guess + (1.0 - proposed_guess) * logistic,
                PROB_EPSILON,
                1.0 - PROB_EPSILON,
            )
            if response == 1:
                log_like_current += np.log(current)
                log_like_proposed += np.log(proposed)
            else:
                log_like_current += np.log(1.0 - current)
                log_like_proposed += np.log(1.0 - proposed)
        if np.log(rng.random()) < log_like_proposed - log_like_current:
            updated[item_idx] = proposed_guess
    return updated


def test_vectorized_item_updates_match_scalar_reference() -> None:
    fixture_rng = np.random.default_rng(83)
    responses = fixture_rng.integers(0, 2, size=(45, 7), dtype=np.int32)
    responses[fixture_rng.random(responses.shape) < 0.12] = -1
    theta = fixture_rng.normal(size=responses.shape[0])
    discrimination = fixture_rng.lognormal(0.0, 0.2, size=responses.shape[1])
    difficulty = fixture_rng.normal(0.0, 0.5, size=responses.shape[1])
    guessing = fixture_rng.uniform(0.1, 0.3, size=responses.shape[1])
    sampler = ResponseTimeGibbsSampler(n_iter=4, burnin=2, use_rust=False)
    reference_rng = np.random.default_rng(191)
    vectorized_rng = np.random.default_rng(191)

    expected_disc, expected_diff = _scalar_accuracy_update(
        sampler,
        responses,
        theta,
        discrimination,
        difficulty,
        reference_rng,
    )
    expected_guess = _scalar_guessing_update(
        responses,
        theta,
        expected_disc,
        expected_diff,
        guessing,
        reference_rng,
    )
    actual_disc, actual_diff = sampler._sample_accuracy_params(
        responses,
        theta,
        discrimination,
        difficulty,
        vectorized_rng,
    )
    actual_guess = sampler._sample_guessing_params(
        responses,
        theta,
        actual_disc,
        actual_diff,
        guessing,
        vectorized_rng,
    )

    assert_array_equal(actual_disc, expected_disc)
    assert_array_equal(actual_diff, expected_diff)
    assert_array_equal(actual_guess, expected_guess)


def _reference_time_update(
    priors: RTModelPriors,
    log_rt: np.ndarray,
    tau: np.ndarray,
    time_discrimination: np.ndarray,
    time_intensity: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the time-parameter update one item at a time."""
    n_items = log_rt.shape[1]
    n_steps = _TIME_DISCRIMINATION_MH_STEPS
    intensity_noise = rng.standard_normal(n_items)
    step_noise = rng.standard_normal((n_steps, n_items))
    log_uniform = np.log(rng.random((n_steps, n_items)))
    updated_discrimination = time_discrimination.copy()
    updated_intensity = time_intensity.copy()
    for item_idx in range(n_items):
        valid = ~np.isnan(log_rt[:, item_idx])
        shifted = log_rt[valid, item_idx] + tau[valid]
        n_valid = shifted.size
        precision = time_discrimination[item_idx] ** 2
        post_var = 1.0 / (1.0 / priors.time_int_var + n_valid * precision)
        post_mean = post_var * (
            priors.time_int_mean / priors.time_int_var + precision * shifted.sum()
        )
        intensity = post_mean + np.sqrt(post_var) * intensity_noise[item_idx]
        updated_intensity[item_idx] = intensity
        residuals = shifted - intensity

        def log_target(eta: float, residuals: np.ndarray = residuals) -> float:
            log_likelihood = np.sum(stats.norm.logpdf(residuals, scale=np.exp(-eta)))
            log_prior = stats.norm.logpdf(
                eta,
                priors.time_disc_mean,
                np.sqrt(priors.time_disc_var),
            )
            return float(log_likelihood + log_prior)

        eta = np.log(time_discrimination[item_idx])
        step = 2.4 / np.sqrt(2.0 * n_valid + 1.0 / priors.time_disc_var)
        for step_idx in range(n_steps):
            eta_proposed = eta + step * step_noise[step_idx, item_idx]
            log_ratio = log_target(eta_proposed) - log_target(eta)
            if log_uniform[step_idx, item_idx] < log_ratio:
                eta = eta_proposed
                updated_discrimination[item_idx] = np.exp(eta)
    return updated_discrimination, updated_intensity


def test_time_parameter_update_matches_itemwise_reference() -> None:
    fixture_rng = np.random.default_rng(5)
    log_rt = fixture_rng.normal(0.5, 0.6, size=(40, 6))
    log_rt[fixture_rng.random(log_rt.shape) < 0.15] = np.nan
    log_rt[:, 4] = np.nan
    tau = fixture_rng.normal(size=40)
    time_discrimination = fixture_rng.lognormal(0.0, 0.3, size=6)
    time_intensity = fixture_rng.normal(0.0, 0.5, size=6)
    priors = RTModelPriors(time_disc_mean=0.2, time_disc_var=0.5)
    sampler = ResponseTimeGibbsSampler(n_iter=4, burnin=2, priors=priors)

    expected = _reference_time_update(
        priors,
        log_rt,
        tau,
        time_discrimination,
        time_intensity,
        np.random.default_rng(3),
    )
    actual = sampler._sample_time_params(
        log_rt,
        tau,
        time_discrimination,
        time_intensity,
        np.random.default_rng(3),
    )

    assert_allclose(actual[1], expected[1], rtol=1e-13, atol=1e-13)
    assert_allclose(actual[0], expected[0], rtol=1e-14, atol=0.0)
    assert np.any(actual[0] != time_discrimination)


def test_time_discrimination_prior_is_stationary_without_timing_data() -> None:
    """With every response time missing the item draws follow their priors."""
    priors = RTModelPriors(
        time_disc_mean=0.4,
        time_disc_var=0.25,
        time_int_mean=1.0,
        time_int_var=0.5,
    )
    sampler = ResponseTimeGibbsSampler(n_iter=2, burnin=1, priors=priors)
    rng = np.random.default_rng(0)
    n_items = 200
    log_rt = np.full((3, n_items), np.nan)
    tau = np.zeros(3)
    time_discrimination = np.ones(n_items)
    time_intensity = np.zeros(n_items)
    log_discrimination_draws = []
    intensity_draws = []
    for iteration in range(1500):
        time_discrimination, time_intensity = sampler._sample_time_params(
            log_rt,
            tau,
            time_discrimination,
            time_intensity,
            rng,
        )
        if iteration >= 200:
            log_discrimination_draws.append(np.log(time_discrimination))
            intensity_draws.append(time_intensity)
    log_discrimination = np.asarray(log_discrimination_draws)
    intensity = np.asarray(intensity_draws)

    assert abs(log_discrimination.mean() - 0.4) < 0.05
    assert abs(log_discrimination.var() / 0.25 - 1.0) < 0.1
    assert abs(intensity.mean() - 1.0) < 0.05
    assert abs(intensity.var() / 0.5 - 1.0) < 0.1


def test_accuracy_discrimination_prior_has_no_jacobian_shift() -> None:
    """Regression: a spurious Jacobian moved the stationary mean to m + v."""
    priors = RTModelPriors(disc_mean=0.2, disc_var=0.25)
    sampler = ResponseTimeGibbsSampler(n_iter=2, burnin=1, priors=priors)
    rng = np.random.default_rng(0)
    n_items = 20
    responses = np.full((2, n_items), -1, dtype=np.int32)
    theta = np.zeros(2)
    discrimination = np.ones(n_items)
    difficulty = np.zeros(n_items)
    log_discrimination_draws = []
    for iteration in range(1000):
        discrimination, difficulty = sampler._sample_accuracy_params(
            responses,
            theta,
            discrimination,
            difficulty,
            rng,
        )
        if iteration >= 100:
            log_discrimination_draws.append(np.log(discrimination))

    assert abs(np.mean(log_discrimination_draws) - 0.2) < 0.12


def test_time_parameter_update_ignores_covariance_prior_df() -> None:
    """Regression: the time-precision draw used sigma_df as its Gamma prior."""
    fixture_rng = np.random.default_rng(13)
    log_rt = fixture_rng.normal(0.0, 0.5, size=(25, 4))
    tau = fixture_rng.normal(size=25)
    draws = [
        ResponseTimeGibbsSampler(
            n_iter=2,
            burnin=1,
            priors=RTModelPriors(sigma_df=sigma_df),
        )._sample_time_params(
            log_rt,
            tau,
            np.ones(4),
            np.zeros(4),
            np.random.default_rng(1),
        )
        for sigma_df in (2, 50)
    ]

    assert_array_equal(draws[0][0], draws[1][0])
    assert_array_equal(draws[0][1], draws[1][1])


def test_time_discrimination_update_mixes_within_one_sweep() -> None:
    """Several cheap Metropolis steps keep successive draws nearly independent.

    A single random-walk step per sweep left a lag-one autocorrelation of
    about 0.6 here; the direct Gamma draw it replaced mixed almost perfectly.
    """
    data_rng = np.random.default_rng(2)
    alpha = np.linspace(0.8, 2.5, 8)
    tau = data_rng.normal(size=200)
    log_rt = (
        np.linspace(-0.5, 1.0, 8)
        - tau[:, None]
        + data_rng.normal(size=(200, 8)) / alpha
    )
    sampler = ResponseTimeGibbsSampler(n_iter=2, burnin=1)
    rng = np.random.default_rng(0)
    time_discrimination = np.ones(8)
    time_intensity = np.zeros(8)
    draws = []
    for iteration in range(1200):
        time_discrimination, time_intensity = sampler._sample_time_params(
            log_rt,
            tau,
            time_discrimination,
            time_intensity,
            rng,
        )
        if iteration >= 200:
            draws.append(np.log(time_discrimination))
    centered = np.asarray(draws) - np.mean(draws, axis=0)
    lag_one = np.sum(centered[1:] * centered[:-1], axis=0) / np.sum(centered**2, axis=0)

    assert np.max(lag_one) < 0.25


def test_tight_time_discrimination_prior_moves_posterior_towards_prior() -> None:
    generating_model = ResponseTimeModel(5, use_rust=False)
    responses, response_times, _, _ = generating_model.simulate(30, seed=3)

    def posterior_time_discrimination(priors: RTModelPriors) -> np.ndarray:
        result = ResponseTimeGibbsSampler(
            n_iter=400,
            burnin=200,
            priors=priors,
            seed=5,
        ).fit(responses, response_times)
        return result.model.time_discrimination

    default = posterior_time_discrimination(RTModelPriors())
    tight = posterior_time_discrimination(
        RTModelPriors(time_disc_mean=np.log(3.0), time_disc_var=0.01)
    )

    assert np.all(tight > default + 0.3)
    assert np.all(tight < 3.0)


def test_time_discrimination_is_recovered_at_default_priors() -> None:
    alpha = np.linspace(0.8, 2.5, 10)
    generating_model = ResponseTimeModel(
        10,
        time_discrimination=alpha,
        time_intensity=np.linspace(-0.5, 1.0, 10),
        use_rust=False,
    )
    responses, response_times, _, _ = generating_model.simulate(500, seed=11)

    result = ResponseTimeGibbsSampler(n_iter=400, burnin=200, seed=7).fit(
        responses,
        response_times,
    )

    assert_allclose(result.model.time_discrimination, alpha, atol=0.15)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_iter": 0}, "n_iter"),
        ({"n_iter": True}, "n_iter"),
        ({"burnin": -1}, "burnin"),
        ({"n_iter": 2, "burnin": 2}, "burnin"),
        ({"thin": 0}, "thin"),
        ({"n_chains": 0}, "n_chains"),
        ({"proposal_sd": 0.0}, "proposal_sd"),
        ({"proposal_sd": "invalid"}, "proposal_sd"),
        ({"adapt_interval": 0}, "adapt_interval"),
        ({"priors": object()}, "priors"),
        ({"verbose": 1}, "verbose"),
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
    ],
)
def test_sampler_configuration_is_validated(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(MirtValidationError, match=match):
        ResponseTimeGibbsSampler(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"disc_mean": np.nan}, "disc_mean"),
        ({"diff_mean": "invalid"}, "diff_mean"),
        ({"disc_var": 0.0}, "disc_var"),
        ({"time_int_var": np.inf}, "time_int_var"),
        ({"sigma_df": 1}, "sigma_df"),
        ({"mu_mean": np.zeros(3)}, "mu_mean"),
        ({"mu_cov": np.array([[1.0, 0.5], [0.0, 1.0]])}, "mu_cov"),
        ({"sigma_scale": np.array([[1.0, 2.0], [2.0, 1.0]])}, "sigma_scale"),
    ],
)
def test_prior_configuration_is_validated(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(MirtValidationError, match=match):
        RTModelPriors(**kwargs)


@pytest.mark.parametrize(
    ("responses", "response_times", "accuracy_model", "match"),
    [
        (np.array([1, 0]), np.array([1.0, 2.0]), "2PL", "shape"),
        (np.empty((0, 2)), np.empty((0, 2)), "2PL", "at least one"),
        (np.array([[1, 0]]), np.array([[1.0]]), "2PL", "same shape"),
        (np.array([[1, 2]]), np.array([[1.0, 2.0]]), "2PL", "0 or 1"),
        (np.array([[1, np.inf]]), np.array([[1.0, 2.0]]), "2PL", "0 or 1"),
        (np.array([[1]]), np.array([[0.0]]), "2PL", "positive"),
        (np.array([[1]]), np.array([[-2.0]]), "2PL", "positive"),
        (np.array([[1]]), np.array([[np.inf]]), "2PL", "positive"),
        (np.array([[1]]), np.array([[1.0]]), "4PL", "accuracy_model"),
        (np.array([[np.nan]]), np.array([[np.nan]]), "2PL", "observation"),
    ],
)
def test_fit_data_is_validated_before_sampling(
    responses: np.ndarray,
    response_times: np.ndarray,
    accuracy_model: str,
    match: str,
) -> None:
    sampler = ResponseTimeGibbsSampler(n_iter=2, burnin=1, use_rust=False)

    with pytest.raises(MirtValidationError, match=match):
        sampler.fit(
            responses,
            response_times,
            accuracy_model=cast(Any, accuracy_model),
        )


def test_accuracy_and_timing_missingness_are_independent_in_fit_data() -> None:
    sampler = ResponseTimeGibbsSampler(n_iter=2, burnin=1, use_rust=False)
    responses, log_rt = sampler._validate_fit_data(
        np.array([[1.0, np.nan], [-1.0, 0.0]]),
        np.array([[np.nan, 2.0], [3.0, np.nan]]),
        "2PL",
    )

    assert_array_equal(responses, [[1, -1], [-1, 0]])
    assert_allclose(log_rt[np.isfinite(log_rt)], np.log([2.0, 3.0]))
    assert_array_equal(np.isnan(log_rt), [[True, False], [False, True]])


def test_multiple_chains_and_nondivisible_thinning_are_reproducible() -> None:
    generating_model = ResponseTimeModel(2, use_rust=False)
    responses, response_times, _, _ = generating_model.simulate(6, seed=17)

    def fit():
        return ResponseTimeGibbsSampler(
            n_iter=13,
            burnin=2,
            thin=3,
            n_chains=2,
            proposal_sd=0.1,
            adapt_interval=2,
            seed=29,
            use_rust=False,
        ).fit(responses, response_times)

    first = fit()
    second = fit()

    assert first.n_chains == 2
    assert first.chains is not None
    assert second.chains is not None
    assert first.chains["difficulty"].shape == (8, 2)
    for name in first.chains:
        assert_array_equal(first.chains[name], second.chains[name])
    assert not np.array_equal(
        first.chains["difficulty"][:4], first.chains["difficulty"][4:]
    )
    assert np.all(np.isfinite([first.log_likelihood, first.dic, first.waic]))


def test_chain_diagnostics_use_chain_identity_and_handle_constants() -> None:
    rng = np.random.default_rng(47)
    well_mixed = rng.normal(size=(4, 256))
    shifted = well_mixed.copy()
    shifted[0] += 3.0
    constant = np.ones((4, 256))
    sampler = ResponseTimeGibbsSampler(n_iter=4, burnin=2, use_rust=False)

    assert sampler._compute_rhat({"x": well_mixed})["x"] < 1.05
    assert sampler._compute_rhat({"x": shifted})["x"] > 1.1
    assert sampler._compute_rhat({"x": constant})["x"] == 1.0
    assert sampler._compute_ess({"x": well_mixed})["x"] > 400.0
    assert sampler._compute_ess({"x": constant})["x"] == 1024.0


def test_waic_remains_finite_for_extreme_log_likelihoods() -> None:
    class ExtremeLikelihoodModel:
        def joint_log_likelihood(
            self,
            responses: np.ndarray,
            log_rt: np.ndarray,
            theta: np.ndarray,
            tau: np.ndarray,
        ) -> np.ndarray:
            del responses, log_rt, tau
            return np.array([-1000.0 - theta[0], -1200.0 - 2.0 * theta[0]])

    sampler = ResponseTimeGibbsSampler(n_iter=4, burnin=2, use_rust=False)
    theta_samples = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    tau_samples = np.zeros_like(theta_samples)
    log_likes = np.array([[-1000.0, -1200.0], [-1001.0, -1202.0], [-1002.0, -1204.0]])
    expected_lppd = np.sum(logsumexp(log_likes, axis=0) - np.log(3.0))
    expected = -2.0 * (expected_lppd - np.sum(np.var(log_likes, axis=0, ddof=1)))

    actual = sampler._compute_waic(
        cast(ResponseTimeModel, ExtremeLikelihoodModel()),
        np.ones((2, 1), dtype=np.int32),
        np.ones((2, 1)),
        theta_samples,
        tau_samples,
    )

    assert np.isfinite(actual)
    assert actual == pytest.approx(expected)
