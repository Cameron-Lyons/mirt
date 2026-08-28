"""Contracts and regression coverage for longitudinal Gibbs estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

import mirt.estimation.dynamic_gibbs as dynamic_gibbs
from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON
from mirt.estimation.dynamic_gibbs import (
    LongitudinalGibbsSampler,
    LongitudinalPriors,
)
from mirt.models.dynamic import LongitudinalIRTModel


def _reference_log_likelihood(
    responses: np.ndarray,
    theta: np.ndarray,
    model: LongitudinalIRTModel,
) -> float:
    value = 0.0
    for i in range(responses.shape[0]):
        for t in range(responses.shape[1]):
            for j in range(responses.shape[2]):
                response = responses[i, t, j]
                if response < 0:
                    continue
                probability = model.probability(np.array([theta[i, t]]), j)[0]
                probability = np.clip(
                    probability,
                    PROB_EPSILON,
                    1.0 - PROB_EPSILON,
                )
                value += (
                    np.log(probability) if response == 1 else np.log1p(-probability)
                )
    return float(value)


def _reference_sample_theta(
    responses: np.ndarray,
    model: LongitudinalIRTModel,
    growth_factors: np.ndarray,
    time_values: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    theta_pred = model.compute_theta(growth_factors, time_values)
    theta = theta_pred.copy()
    residual_sd = np.sqrt(model.residual_variance)

    for i in range(responses.shape[0]):
        for t in range(responses.shape[1]):
            theta_proposed = theta[i, t] + rng.normal(0, 0.3)
            prior_current = stats.norm.logpdf(
                theta[i, t],
                theta_pred[i, t],
                residual_sd,
            )
            prior_proposed = stats.norm.logpdf(
                theta_proposed,
                theta_pred[i, t],
                residual_sd,
            )

            current = 0.0
            proposed = 0.0
            for j in range(model.n_items):
                response = responses[i, t, j]
                if response < 0:
                    continue
                p_current = np.clip(
                    model.probability(np.array([theta[i, t]]), j)[0],
                    PROB_EPSILON,
                    1.0 - PROB_EPSILON,
                )
                p_proposed = np.clip(
                    model.probability(np.array([theta_proposed]), j)[0],
                    PROB_EPSILON,
                    1.0 - PROB_EPSILON,
                )
                if response == 1:
                    current += np.log(p_current)
                    proposed += np.log(p_proposed)
                else:
                    current += np.log1p(-p_current)
                    proposed += np.log1p(-p_proposed)

            log_alpha = proposed + prior_proposed - current - prior_current
            if np.log(rng.random()) < log_alpha:
                theta[i, t] = theta_proposed

    return theta


def _reference_sample_items(
    responses: np.ndarray,
    theta: np.ndarray,
    model: LongitudinalIRTModel,
    priors: LongitudinalPriors,
    rng: np.random.Generator,
) -> None:
    for j in range(model.n_items):
        a_current = model.discrimination[j]
        b_current = model.difficulty[j]
        a_proposed = np.clip(a_current + rng.normal(0, 0.1), 0.2, 5.0)
        b_proposed = np.clip(b_current + rng.normal(0, 0.15), -5.0, 5.0)
        current = 0.0
        proposed = 0.0

        for i in range(responses.shape[0]):
            for t in range(responses.shape[1]):
                response = responses[i, t, j]
                if response < 0:
                    continue
                current_probability = np.clip(
                    sigmoid(a_current * (theta[i, t] - b_current)),
                    PROB_EPSILON,
                    1.0 - PROB_EPSILON,
                )
                proposed_probability = np.clip(
                    sigmoid(a_proposed * (theta[i, t] - b_proposed)),
                    PROB_EPSILON,
                    1.0 - PROB_EPSILON,
                )
                if response == 1:
                    current += np.log(current_probability)
                    proposed += np.log(proposed_probability)
                else:
                    current += np.log1p(-current_probability)
                    proposed += np.log1p(-proposed_probability)

        prior_current = stats.lognorm.logpdf(
            a_current,
            s=np.sqrt(priors.discrimination_var),
            scale=priors.discrimination_mean,
        ) + stats.norm.logpdf(
            b_current,
            priors.difficulty_mean,
            np.sqrt(priors.difficulty_var),
        )
        prior_proposed = stats.lognorm.logpdf(
            a_proposed,
            s=np.sqrt(priors.discrimination_var),
            scale=priors.discrimination_mean,
        ) + stats.norm.logpdf(
            b_proposed,
            priors.difficulty_mean,
            np.sqrt(priors.difficulty_var),
        )
        if np.log(rng.random()) < (proposed + prior_proposed - current - prior_current):
            model.discrimination[j] = a_proposed
            model.difficulty[j] = b_proposed


@pytest.fixture
def longitudinal_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    responses = np.array(
        [
            [[1, 0, -1, 1], [0, 1, 1, 0], [1, -1, 0, 1]],
            [[0, 1, 1, -1], [1, 0, 1, 0], [-1, 1, 0, 0]],
            [[1, 1, 0, 0], [0, -1, 1, 1], [1, 0, 0, 1]],
        ],
        dtype=np.int32,
    )
    growth_factors = np.array(
        [[-0.2, 0.1], [0.4, -0.05], [0.1, 0.2]],
        dtype=np.float64,
    )
    time_values = np.array([0.0, 0.75, 2.0])
    return responses, growth_factors, time_values


def test_vectorized_log_likelihood_matches_scalar_reference(
    longitudinal_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses, growth_factors, time_values = longitudinal_data
    model = LongitudinalIRTModel(
        n_items=responses.shape[2],
        n_timepoints=responses.shape[1],
        discrimination=np.array([0.7, 1.1, 1.6, 0.9]),
        difficulty=np.array([-0.8, 0.2, 1.0, -0.1]),
    )
    theta = model.compute_theta(growth_factors, time_values)
    sampler = LongitudinalGibbsSampler(n_iter=2, burnin=1)
    monkeypatch.setattr(dynamic_gibbs, "_LONGITUDINAL_MAX_PROBABILITY_VALUES", 7)

    actual = sampler._compute_log_likelihood(responses, theta, model)
    expected = _reference_log_likelihood(responses, theta, model)

    assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_vectorized_theta_sampler_preserves_seeded_draws(
    longitudinal_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    responses, growth_factors, time_values = longitudinal_data
    model = LongitudinalIRTModel(
        n_items=responses.shape[2],
        n_timepoints=responses.shape[1],
        discrimination=np.array([0.7, 1.1, 1.6, 0.9]),
        difficulty=np.array([-0.8, 0.2, 1.0, -0.1]),
    )
    sampler = LongitudinalGibbsSampler(n_iter=2, burnin=1)

    expected = _reference_sample_theta(
        responses,
        model,
        growth_factors,
        time_values,
        np.random.default_rng(91),
    )
    actual = sampler._sample_theta(
        responses,
        model,
        growth_factors,
        time_values,
        np.random.default_rng(91),
    )

    assert_array_equal(actual, expected)


def test_vectorized_item_sampler_preserves_seeded_draws(
    longitudinal_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses, growth_factors, time_values = longitudinal_data
    model_kwargs = {
        "n_items": responses.shape[2],
        "n_timepoints": responses.shape[1],
        "discrimination": np.array([0.7, 1.1, 1.6, 0.9]),
        "difficulty": np.array([-0.8, 0.2, 1.0, -0.1]),
    }
    expected_model = LongitudinalIRTModel(**model_kwargs)
    actual_model = LongitudinalIRTModel(**model_kwargs)
    priors = LongitudinalPriors(
        discrimination_mean=1.2,
        discrimination_var=0.7,
        difficulty_mean=-0.1,
        difficulty_var=2.5,
    )
    sampler = LongitudinalGibbsSampler(n_iter=2, burnin=1, priors=priors)
    theta = actual_model.compute_theta(growth_factors, time_values)
    monkeypatch.setattr(dynamic_gibbs, "_LONGITUDINAL_MAX_PROBABILITY_VALUES", 7)

    _reference_sample_items(
        responses,
        theta,
        expected_model,
        priors,
        np.random.default_rng(27),
    )
    sampler._sample_item_params(
        responses,
        theta,
        actual_model,
        np.random.default_rng(27),
    )

    assert_array_equal(actual_model.discrimination, expected_model.discrimination)
    assert_array_equal(actual_model.difficulty, expected_model.difficulty)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"n_iter": 0}, ValueError, "positive integer"),
        ({"n_iter": True}, ValueError, "positive integer"),
        ({"burnin": -1}, ValueError, "non-negative integer"),
        ({"n_iter": 2, "burnin": 2}, ValueError, "less than n_iter"),
        ({"thin": 0}, ValueError, "positive integer"),
        ({"priors": object()}, TypeError, "LongitudinalPriors"),
        ({"verbose": 1}, TypeError, "boolean"),
        ({"seed": -1}, ValueError, "non-negative integer"),
        ({"seed": True}, ValueError, "non-negative integer"),
    ],
)
def test_sampler_configuration_is_validated(
    kwargs: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        LongitudinalGibbsSampler(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"discrimination_mean": 0.0},
        {"discrimination_var": np.inf},
        {"difficulty_mean": np.nan},
        {"difficulty_var": -1.0},
        {"growth_cov_prior_df": True},
        {"residual_var_prior_shape": 0.0},
        {"residual_var_prior_rate": "invalid"},
        {"growth_mean_prior_mean": [[0.0, 1.0]]},
        {"growth_mean_prior_cov": np.ones((2, 3))},
        {"growth_mean_prior_cov": np.array([[1.0, 2.0], [0.0, 1.0]])},
        {"growth_cov_prior_scale": np.array([[1.0, 0.0], [0.0, 0.0]])},
    ],
)
def test_longitudinal_priors_are_validated(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        LongitudinalPriors(**kwargs)


@pytest.mark.parametrize(
    ("priors", "growth_model", "match"),
    [
        (
            LongitudinalPriors(growth_mean_prior_mean=np.zeros(2)),
            "quadratic",
            "growth_mean_prior_mean",
        ),
        (
            LongitudinalPriors(growth_mean_prior_cov=np.eye(3)),
            "linear",
            "growth_mean_prior_cov",
        ),
        (
            LongitudinalPriors(growth_cov_prior_scale=np.eye(2)),
            "quadratic",
            "growth_cov_prior_scale",
        ),
        (
            LongitudinalPriors(growth_cov_prior_df=2.0),
            "quadratic",
            "growth_cov_prior_df",
        ),
    ],
)
def test_growth_prior_dimensions_match_model(
    priors: LongitudinalPriors,
    growth_model: str,
    match: str,
) -> None:
    sampler = LongitudinalGibbsSampler(n_iter=2, burnin=1, priors=priors)

    with pytest.raises(ValueError, match=match):
        sampler.fit(
            np.array([[[1, 0], [0, 1]]]),
            growth_model=growth_model,
        )


@pytest.mark.parametrize(
    ("responses", "fit_kwargs", "match"),
    [
        (np.array([1, 0]), {}, "shape"),
        (np.empty((0, 2, 2), dtype=int), {}, "non-empty"),
        (np.array([[[1.0, 0.0]]]), {}, "integer values"),
        (np.array([[[1, 2]]]), {}, "only -1, 0, or 1"),
        (np.array([[[-1, -1]]]), {}, "observed value"),
        (np.array([[1, 0, 1, 0]]), {}, "positive integer"),
        (np.array([[1, 0, 1]]), {"n_items": 2}, "divisible"),
        (np.array([[[1, 0]]]), {"n_items": 3}, "final response dimension"),
        (np.array([[[1, 0], [0, 1]]]), {"time_values": [0.0]}, "shape"),
        (
            np.array([[[1, 0], [0, 1]]]),
            {"time_values": [0.0, np.inf]},
            "finite",
        ),
    ],
)
def test_fit_inputs_are_validated_before_sampling(
    responses: np.ndarray,
    fit_kwargs: dict[str, Any],
    match: str,
) -> None:
    sampler = LongitudinalGibbsSampler(n_iter=2, burnin=1)

    with pytest.raises(ValueError, match=match):
        sampler.fit(responses, **fit_kwargs)


def test_flat_and_tensor_inputs_produce_same_seeded_fit(
    longitudinal_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    responses, _, time_values = longitudinal_data
    tensor_result = LongitudinalGibbsSampler(
        n_iter=4,
        burnin=2,
        seed=123,
    ).fit(responses, time_values=time_values)
    flat_result = LongitudinalGibbsSampler(
        n_iter=4,
        burnin=2,
        seed=123,
    ).fit(
        responses.reshape(responses.shape[0], -1),
        n_items=responses.shape[2],
        time_values=time_values,
    )

    assert_array_equal(
        flat_result.model.discrimination,
        tensor_result.model.discrimination,
    )
    assert_array_equal(flat_result.model.difficulty, tensor_result.model.difficulty)
    assert_array_equal(flat_result.growth_factors, tensor_result.growth_factors)
    assert_array_equal(flat_result.theta_trajectories, tensor_result.theta_trajectories)
    assert flat_result.log_likelihood == tensor_result.log_likelihood
