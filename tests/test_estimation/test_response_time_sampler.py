"""Regression coverage for response-time Gibbs sampling."""

from __future__ import annotations

import warnings
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


def _scalar_log_probability(
    response: int,
    discrimination: float,
    difficulty: float,
    guessing: float | None,
    theta: float,
) -> float:
    """Return one clipped accuracy log probability."""
    logistic = float(sigmoid(discrimination * (theta - difficulty)))
    probability = logistic if guessing is None else guessing + (1 - guessing) * logistic
    probability = float(np.clip(probability, PROB_EPSILON, 1.0 - PROB_EPSILON))
    return np.log(probability) if response == 1 else np.log1p(-probability)


def _scalar_accuracy_update(
    sampler: ResponseTimeGibbsSampler,
    responses: np.ndarray,
    theta: np.ndarray,
    discrimination: np.ndarray,
    difficulty: np.ndarray,
    rng: np.random.Generator,
    guessing: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the accuracy update one item and person at a time.

    The prior and the walk both live on log a, and the likelihood is the 3PL
    whenever guessing parameters are supplied.
    """
    n_items = len(discrimination)
    log_disc_steps = rng.normal(0.0, 0.1, n_items)
    diff_steps = rng.normal(0.0, 0.1, n_items)
    log_uniform = np.log(rng.random(n_items))
    updated_discrimination = discrimination.copy()
    updated_difficulty = difficulty.copy()
    for item_idx in range(n_items):
        log_disc_current = np.log(discrimination[item_idx])
        log_disc_proposed = log_disc_current + log_disc_steps[item_idx]
        disc_proposed = np.exp(log_disc_proposed)
        diff_proposed = difficulty[item_idx] + diff_steps[item_idx]
        guess = None if guessing is None else guessing[item_idx]
        log_like_current = 0.0
        log_like_proposed = 0.0

        for person_idx in range(len(theta)):
            response = responses[person_idx, item_idx]
            if response < 0:
                continue
            log_like_current += _scalar_log_probability(
                response,
                discrimination[item_idx],
                difficulty[item_idx],
                guess,
                theta[person_idx],
            )
            log_like_proposed += _scalar_log_probability(
                response, disc_proposed, diff_proposed, guess, theta[person_idx]
            )

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
        if log_uniform[item_idx] < log_acceptance:
            updated_discrimination[item_idx] = disc_proposed
            updated_difficulty[item_idx] = diff_proposed

    return updated_discrimination, updated_difficulty


def _scalar_guessing_update(
    priors: RTModelPriors,
    responses: np.ndarray,
    theta: np.ndarray,
    discrimination: np.ndarray,
    difficulty: np.ndarray,
    guessing: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run the logit-scale guessing update one item at a time.

    ``eta = logit(c / U)`` moves by a symmetric walk; the target on that
    scale is the 3PL likelihood, the Beta prior density of ``c`` and the
    Jacobian ``c (U - c) / U``.
    """
    upper = priors.guess_upper
    n_items = len(guessing)
    steps = rng.normal(0.0, 0.3, n_items)
    log_uniform = np.log(rng.random(n_items))
    updated = guessing.copy()

    def log_target(item_idx: int, guess: float) -> float:
        log_likelihood = sum(
            _scalar_log_probability(
                responses[person_idx, item_idx],
                discrimination[item_idx],
                difficulty[item_idx],
                guess,
                theta[person_idx],
            )
            for person_idx in range(len(theta))
            if responses[person_idx, item_idx] >= 0
        )
        log_prior = stats.beta.logpdf(guess, priors.guess_alpha, priors.guess_beta)
        log_jacobian = np.log(guess) + np.log(upper - guess) - np.log(upper)
        return float(log_likelihood + log_prior + log_jacobian)

    for item_idx in range(n_items):
        eta = np.log(guessing[item_idx] / (upper - guessing[item_idx]))
        proposed = upper / (1.0 + np.exp(-(eta + steps[item_idx])))
        log_ratio = log_target(item_idx, proposed) - log_target(
            item_idx, guessing[item_idx]
        )
        if log_uniform[item_idx] < log_ratio:
            updated[item_idx] = proposed
    return updated


@pytest.mark.parametrize("accuracy_model", ["2PL", "3PL"])
def test_vectorized_item_updates_match_scalar_reference(accuracy_model: str) -> None:
    fixture_rng = np.random.default_rng(83)
    responses = fixture_rng.integers(0, 2, size=(45, 7), dtype=np.int32)
    responses[fixture_rng.random(responses.shape) < 0.12] = -1
    theta = fixture_rng.normal(size=responses.shape[0])
    discrimination = fixture_rng.lognormal(0.0, 0.2, size=responses.shape[1])
    difficulty = fixture_rng.normal(0.0, 0.5, size=responses.shape[1])
    guessing = fixture_rng.uniform(0.1, 0.3, size=responses.shape[1])
    accuracy_guessing = guessing if accuracy_model == "3PL" else None
    priors = RTModelPriors(guess_alpha=2.0, guess_beta=6.0, guess_upper=0.4)
    sampler = ResponseTimeGibbsSampler(
        n_iter=4, burnin=2, priors=priors, use_rust=False
    )
    reference_rng = np.random.default_rng(191)
    vectorized_rng = np.random.default_rng(191)

    expected_disc, expected_diff = _scalar_accuracy_update(
        sampler,
        responses,
        theta,
        discrimination,
        difficulty,
        reference_rng,
        accuracy_guessing,
    )
    expected_guess = _scalar_guessing_update(
        priors,
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
        guess=accuracy_guessing,
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
    assert_allclose(actual_guess, expected_guess, rtol=1e-12, atol=0.0)
    assert np.any(actual_disc != discrimination)
    assert np.any(actual_guess != guessing)


def test_3pl_accuracy_update_uses_the_3pl_likelihood() -> None:
    """Regression: the (a, b) step ignored guessing and targeted the 2PL.

    With ability and guessing held at their generating values the old step
    settled at a = [0.70, 0.95, 1.09, 1.07, 0.87] and b about 0.5 too low for
    a = [0.8, ..., 2.0] (N = 20000); the 3PL step stays at the truth.
    """
    discrimination = np.array([0.8, 1.4, 2.0])
    difficulty = np.array([-0.8, 0.0, 0.8])
    guessing = np.full(3, 0.25)
    generator = ResponseTimeModel(
        n_items=3,
        accuracy_model="3PL",
        discrimination=discrimination,
        difficulty=difficulty,
        guessing=guessing,
        use_rust=False,
    )
    responses, _, theta, _ = generator.simulate(6000, seed=3)
    sampler = ResponseTimeGibbsSampler(n_iter=2, burnin=1)
    rng = np.random.default_rng(7)
    current_disc, current_diff = discrimination.copy(), difficulty.copy()
    disc_draws, diff_draws = [], []
    for iteration in range(700):
        current_disc, current_diff = sampler._sample_accuracy_params(
            responses, theta, current_disc, current_diff, rng, guess=guessing
        )
        if iteration >= 200:
            disc_draws.append(current_disc)
            diff_draws.append(current_diff)

    assert_allclose(np.mean(disc_draws, axis=0), discrimination, atol=0.2)
    assert_allclose(np.mean(diff_draws, axis=0), difficulty, atol=0.15)


def test_guessing_update_samples_its_exact_conditional() -> None:
    """Regression: clipped proposals piled 24% of the draws on c = 0.01.

    Every item has the same generating parameters and its own responses, so
    the chains are independent replicates of a one-dimensional conditional
    that a grid evaluates exactly.
    """
    n_persons, n_items = 300, 12
    generator = ResponseTimeModel(
        n_items=n_items,
        accuracy_model="3PL",
        discrimination=np.ones(n_items),
        difficulty=np.full(n_items, 1.5),
        guessing=np.full(n_items, 0.02),
        use_rust=False,
    )
    responses, _, theta, _ = generator.simulate(n_persons, seed=11)
    sampler = ResponseTimeGibbsSampler(n_iter=2, burnin=1)
    rng = np.random.default_rng(3)
    discrimination = np.ones(n_items)
    difficulty = np.full(n_items, 1.5)
    guessing = np.full(n_items, 0.1)
    draws = []
    for iteration in range(3000):
        guessing = sampler._sample_guessing_params(
            responses, theta, discrimination, difficulty, guessing, rng
        )
        if iteration >= 300:
            draws.append(guessing)
    draws_array = np.asarray(draws)

    grid = np.linspace(0.0, 0.5, 4001)[1:-1]
    logistic = sigmoid(theta - 1.5)
    probability = grid[:, None] + (1.0 - grid[:, None]) * logistic[None, :]
    exact_means = []
    for item_idx in range(n_items):
        correct = responses[:, item_idx] == 1
        log_density = np.sum(
            np.where(correct, np.log(probability), np.log1p(-probability)), axis=1
        )
        weights = np.exp(log_density - log_density.max())
        exact_means.append(np.sum(weights * grid) / np.sum(weights))

    assert np.all((draws_array > 0.0) & (draws_array < 0.5))
    assert np.mean(draws_array == draws_array.min()) < 0.01
    assert abs(np.mean(draws_array.mean(axis=0) - exact_means)) < 0.002


def test_guessing_prior_is_stationary_without_responses() -> None:
    """With no observed accuracy the guessing draws follow the Beta prior."""
    priors = RTModelPriors(guess_alpha=3.0, guess_beta=9.0, guess_upper=0.6)
    sampler = ResponseTimeGibbsSampler(n_iter=2, burnin=1, priors=priors)
    rng = np.random.default_rng(0)
    n_items = 200
    responses = np.full((2, n_items), -1, dtype=np.int32)
    guessing = np.full(n_items, 0.2)
    draws = []
    for iteration in range(1200):
        guessing = sampler._sample_guessing_params(
            responses,
            np.zeros(2),
            np.ones(n_items),
            np.zeros(n_items),
            guessing,
            rng,
        )
        if iteration >= 200:
            draws.append(guessing)
    draws_array = np.asarray(draws)

    # Beta(3, 9) truncated to (0, 0.6), integrated on a fine grid.
    grid = np.linspace(0.0, 0.6, 6001)
    weights = stats.beta.pdf(grid, 3.0, 9.0)
    weights /= weights.sum()
    expected_mean = np.sum(weights * grid)
    expected_sd = np.sqrt(np.sum(weights * grid**2) - expected_mean**2)

    assert abs(draws_array.mean() - expected_mean) < 0.005
    assert abs(draws_array.std() - expected_sd) < 0.005


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


def test_population_draw_fixes_identification_and_targets_regression() -> None:
    """The ability-speed mean is zero and Var(theta) is one in every draw.

    The free entries follow the conjugate regression of speed on ability, so
    with many persons Cov(theta, tau) concentrates at S_xy / S_xx.
    """
    data_rng = np.random.default_rng(4)
    theta = data_rng.normal(size=4000)
    tau = -0.4 * theta + 0.5 * data_rng.normal(size=4000)
    sampler = ResponseTimeGibbsSampler(n_iter=2, burnin=1)
    rng = np.random.default_rng(1)
    draws = np.asarray(
        [sampler._sample_population_params(theta, tau, rng) for _ in range(400)]
    )

    assert np.all(draws[:, 0, 0] == 1.0)
    assert_allclose(draws[:, 0, 1], draws[:, 1, 0])
    slope = (theta @ tau) / (theta @ theta)
    residual = np.mean((tau - slope * theta) ** 2)
    assert abs(draws[:, 0, 1].mean() - slope) < 0.01
    assert abs((draws[:, 1, 1] - draws[:, 0, 1] ** 2).mean() - residual) < 0.01


def test_fit_is_identified_and_recovers_2pl_parameters() -> None:
    """Regression: free mu and Var(theta) let the chain drift without bound.

    The time intensities followed mu_tau towards the N(0, 4) prior mean
    (bias -0.39 after 3000 sweeps at N = 1500) and R-hat never converged.
    """
    rng = np.random.default_rng(1)
    n_items = 8
    discrimination = rng.uniform(0.8, 2.0, n_items)
    difficulty = rng.normal(0.0, 1.0, n_items)
    time_intensity = rng.normal(3.5, 0.4, n_items)
    generator = ResponseTimeModel(
        n_items=n_items,
        discrimination=discrimination,
        difficulty=difficulty,
        time_discrimination=rng.uniform(1.0, 2.0, n_items),
        time_intensity=time_intensity,
        ability_speed_cov=np.array([[1.0, -0.4], [-0.4, 0.25]]),
    )
    responses, response_times, _, _ = generator.simulate(300, seed=11)

    result = ResponseTimeGibbsSampler(n_iter=400, burnin=200, seed=1).fit(
        responses, response_times
    )

    assert result.chains is not None
    assert np.all(result.chains["mu_theta"] == 0.0)
    assert np.all(result.chains["mu_tau"] == 0.0)
    assert np.all(result.chains["sigma_11"] == 1.0)
    assert_array_equal(result.model.ability_speed_mean, np.zeros(2))
    assert result.model.ability_speed_cov[0, 0] == 1.0
    assert not {"mu_theta", "mu_tau", "sigma_11"} & set(result.rhat)
    assert abs(np.mean(result.model.time_intensity - time_intensity)) < 0.1
    ratio = result.model.discrimination.mean() / discrimination.mean()
    assert 0.8 < ratio < 1.2
    assert -0.55 < result.model.ability_speed_cov[0, 1] < -0.2
    assert 0.15 < result.model.ability_speed_cov[1, 1] < 0.35


def test_fit_recovers_3pl_parameters() -> None:
    """Regression: 3PL fits returned mean a 0.41 (true 1.36) and c 0.04."""
    rng = np.random.default_rng(2)
    n_items = 10
    discrimination = rng.uniform(0.8, 2.0, n_items)
    difficulty = rng.normal(0.0, 1.0, n_items)
    generator = ResponseTimeModel(
        n_items=n_items,
        accuracy_model="3PL",
        discrimination=discrimination,
        difficulty=difficulty,
        guessing=np.full(n_items, 0.2),
        time_discrimination=rng.uniform(1.0, 2.0, n_items),
        time_intensity=rng.normal(3.5, 0.4, n_items),
    )
    responses, response_times, _, _ = generator.simulate(1000, seed=12)

    result = ResponseTimeGibbsSampler(n_iter=1000, burnin=500, seed=2).fit(
        responses, response_times, accuracy_model="3PL"
    )

    model = result.model
    assert model.guessing is not None
    ratio = model.discrimination.mean() / discrimination.mean()
    assert 0.75 < ratio < 1.3
    assert abs(model.guessing.mean() - 0.2) < 0.06
    assert np.corrcoef(model.difficulty, difficulty)[0, 1] > 0.9
    assert abs(np.mean(model.difficulty - difficulty)) < 0.25


@pytest.mark.slow
@pytest.mark.parametrize("accuracy_model", ["2PL", "3PL"])
def test_long_chain_recovery_is_stationary(accuracy_model: str) -> None:
    """Longer seeded fits recover the generating parameters and converge."""
    rng = np.random.default_rng(1)
    n_items = 20
    discrimination = rng.uniform(0.8, 2.0, n_items)
    difficulty = rng.normal(0.0, 1.0, n_items)
    time_intensity = rng.normal(3.5, 0.4, n_items)
    generator = ResponseTimeModel(
        n_items=n_items,
        accuracy_model=accuracy_model,
        discrimination=discrimination,
        difficulty=difficulty,
        guessing=np.full(n_items, 0.2) if accuracy_model == "3PL" else None,
        time_discrimination=rng.uniform(1.0, 2.0, n_items),
        time_intensity=time_intensity,
        ability_speed_cov=np.array([[1.0, -0.4], [-0.4, 0.25]]),
    )
    responses, response_times, _, _ = generator.simulate(1500, seed=3)

    result = ResponseTimeGibbsSampler(n_iter=3000, burnin=1000, seed=5).fit(
        responses, response_times, accuracy_model=accuracy_model
    )

    model = result.model
    # 3PL item parameters mix slowly (effective sizes near 20 of 2000 draws),
    # so a single chain gets a looser split R-hat bound.
    assert max(result.rhat.values()) < (1.1 if accuracy_model == "2PL" else 1.2)
    assert abs(np.mean(model.time_intensity - time_intensity)) < 0.05
    assert abs(model.discrimination.mean() / discrimination.mean() - 1.0) < 0.1
    assert abs(np.mean(model.difficulty - difficulty)) < 0.1
    assert abs(model.ability_speed_cov[0, 1] + 0.4) < 0.06
    if accuracy_model == "3PL":
        assert model.guessing is not None
        assert abs(model.guessing.mean() - 0.2) < 0.04


def test_dic_and_waic_pair_draws_with_their_item_parameters() -> None:
    """Regression: every person draw was evaluated at posterior-mean items."""
    generating_model = ResponseTimeModel(5, use_rust=False)
    responses, response_times, _, _ = generating_model.simulate(60, seed=8)
    sampler = ResponseTimeGibbsSampler(n_iter=160, burnin=60, seed=9)
    result = sampler.fit(responses, response_times)
    response_codes, log_rt = sampler._validate_fit_data(
        responses, response_times, "2PL"
    )
    chains, theta, tau = sampler._run_chain(
        response_codes, log_rt, "2PL", np.random.default_rng(9), 0
    )

    pointwise = np.asarray(
        [
            ResponseTimeModel(
                5,
                discrimination=chains["discrimination"][draw],
                difficulty=chains["difficulty"][draw],
                time_discrimination=chains["time_discrimination"][draw],
                time_intensity=chains["time_intensity"][draw],
                use_rust=False,
            ).joint_log_likelihood(response_codes, log_rt, theta[draw], tau[draw])
            for draw in range(theta.shape[0])
        ]
    )
    deviance_at_mean = -2.0 * np.sum(
        result.model.joint_log_likelihood(
            response_codes, log_rt, theta.mean(axis=0), tau.mean(axis=0)
        )
    )
    mean_deviance = np.mean(-2.0 * pointwise.sum(axis=1))
    expected_dic = 2.0 * mean_deviance - deviance_at_mean
    lppd = np.sum(logsumexp(pointwise, axis=0) - np.log(theta.shape[0]))
    expected_waic = -2.0 * (lppd - np.sum(np.var(pointwise, axis=0, ddof=1)))

    assert result.dic == pytest.approx(expected_dic, rel=1e-10)
    assert result.waic == pytest.approx(expected_waic, rel=1e-10)
    assert result.log_likelihood == pytest.approx(
        np.mean(pointwise.sum(axis=1)), rel=1e-10
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"guess_alpha": 0.0}, "guess_alpha"),
        ({"guess_beta": np.nan}, "guess_beta"),
        ({"guess_upper": 0.0}, "guess_upper"),
        ({"guess_upper": 1.5}, "guess_upper"),
        ({"guess_upper": "high"}, "guess_upper"),
        ({"guess_upper": True}, "guess_upper"),
    ],
)
def test_guessing_prior_configuration_is_validated(
    kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(MirtValidationError, match=match):
        RTModelPriors(**kwargs)


def test_population_mean_priors_are_deprecated_and_ignored() -> None:
    with pytest.warns(DeprecationWarning, match="mu_mean"):
        RTModelPriors(mu_mean=np.array([0.5, 0.0]))
    with pytest.warns(DeprecationWarning, match="mu_cov"):
        RTModelPriors(mu_cov=np.eye(2))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        RTModelPriors(mu_mean=np.zeros(2), mu_cov=10.0 * np.eye(2))
