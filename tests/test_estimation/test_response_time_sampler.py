"""Regression coverage for response-time Gibbs sampling."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import logsumexp

from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON
from mirt.estimation.rt_gibbs import ResponseTimeGibbsSampler, RTModelPriors
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
    """Run the former scalar accuracy update as a reference."""
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
            + log_disc_proposed
            - log_disc_current
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
