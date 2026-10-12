"""Contracts and regression coverage for longitudinal Gibbs estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats
from scipy.special import digamma

import mirt.estimation.dynamic_gibbs as dynamic_gibbs
from mirt._core import sigmoid
from mirt.constants import PROB_EPSILON
from mirt.estimation._identified_covariance import sample_unit_leading_covariance
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
    theta_current: np.ndarray,
    growth_factors: np.ndarray,
    time_values: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evaluate one scalar MH sweep that starts from the chain state."""
    theta_pred = model.compute_theta(growth_factors, time_values)
    theta = theta_current.copy()
    residual_sd = np.sqrt(model.residual_variance)
    steps = rng.normal(0, 0.3, size=theta.size).reshape(theta.shape)
    log_uniform = np.log(rng.random(theta.size)).reshape(theta.shape)

    for i in range(responses.shape[0]):
        for t in range(responses.shape[1]):
            theta_proposed = theta[i, t] + steps[i, t]
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
            if log_uniform[i, t] < log_alpha:
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


def _reference_sample_growth_factors(
    theta_trajectories: np.ndarray,
    model: LongitudinalIRTModel,
    time_values: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evaluate the original person-wise posterior sampler."""
    n_persons = theta_trajectories.shape[0]
    n_growth = model.n_growth_factors
    design = np.ones((time_values.size, n_growth))
    design[:, 1] = time_values
    if model.growth_model == "quadratic":
        design[:, 2] = time_values**2

    precision_prior = np.linalg.inv(model.growth_cov)
    precision_likelihood = design.T @ design / model.residual_variance
    growth_factors = np.empty((n_persons, n_growth))
    for person in range(n_persons):
        covariance = np.linalg.inv(precision_prior + precision_likelihood)
        likelihood_mean = (
            design.T @ theta_trajectories[person] / model.residual_variance
        )
        prior_mean = precision_prior @ model.growth_mean
        posterior_mean = covariance @ (prior_mean + likelihood_mean)
        growth_factors[person] = rng.multivariate_normal(
            posterior_mean,
            covariance,
        )
    return growth_factors


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


def test_vectorized_theta_sampler_matches_scalar_reference(
    longitudinal_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    responses, growth_factors, time_values = longitudinal_data
    model = LongitudinalIRTModel(
        n_items=responses.shape[2],
        n_timepoints=responses.shape[1],
        discrimination=np.array([0.7, 1.1, 1.6, 0.9]),
        difficulty=np.array([-0.8, 0.2, 1.0, -0.1]),
        residual_variance=0.4,
    )
    sampler = LongitudinalGibbsSampler(n_iter=2, burnin=1)
    theta_current = model.compute_theta(
        growth_factors, time_values
    ) + np.random.default_rng(5).normal(0.0, 0.5, size=(3, 3))

    expected = _reference_sample_theta(
        responses,
        model,
        theta_current,
        growth_factors,
        time_values,
        np.random.default_rng(91),
    )
    actual = sampler._sample_theta(
        responses,
        model,
        theta_current,
        growth_factors,
        time_values,
        np.random.default_rng(91),
    )

    assert_array_equal(actual, expected)
    assert np.any(actual != theta_current)


def test_theta_sampler_keeps_chain_state_and_targets_conditional_prior() -> None:
    """With no observed responses theta | growth is N(theta_pred, sigma^2).

    Regression: the sampler used to restart each MH step at the growth-curve
    prediction, which shrank the sampled residual variance towards zero and
    left a point mass at the prediction.
    """
    rng = np.random.default_rng(0)
    n_persons, n_timepoints, n_items = 500, 4, 3
    residual_variance = 0.7
    model = LongitudinalIRTModel(
        n_items=n_items,
        n_timepoints=n_timepoints,
        residual_variance=residual_variance,
    )
    responses = np.full((n_persons, n_timepoints, n_items), -1, dtype=np.int_)
    growth_factors = rng.normal(size=(n_persons, 2))
    time_values = np.arange(n_timepoints, dtype=np.float64)
    theta_pred = model.compute_theta(growth_factors, time_values)
    sampler = LongitudinalGibbsSampler(n_iter=2, burnin=1)

    theta = theta_pred.copy()
    deviations = []
    for sweep in range(300):
        theta = sampler._sample_theta(
            responses,
            model,
            theta,
            growth_factors,
            time_values,
            rng,
        )
        if sweep >= 100:
            deviations.append(theta - theta_pred)
    deviations_array = np.asarray(deviations)

    assert abs(deviations_array.var() / residual_variance - 1.0) < 0.05
    assert abs(deviations_array.mean()) < 0.05
    assert np.mean(deviations_array[-1] == 0.0) < 0.01


def test_fit_recovers_residual_to_intercept_variance_ratio() -> None:
    """Compare the residual variance with the intercept variance.

    The old kernel pinned the residual variance at its 0.01 floor, about a
    fifth of the generating ratio. The intercept variance is now fixed at one,
    so the ratio is the residual variance on the identified scale.
    """
    n_items = 10
    generator = LongitudinalIRTModel(
        n_items=n_items,
        n_timepoints=5,
        residual_variance=0.1,
        discrimination=np.linspace(0.8, 2.0, n_items),
        difficulty=np.linspace(-1.5, 1.5, n_items),
    )
    responses, _, _ = generator.simulate(250, seed=4)

    result = LongitudinalGibbsSampler(n_iter=150, burnin=75, seed=4).fit(responses)

    true_ratio = generator.residual_variance / generator.growth_cov[0, 0]
    ratio = result.model.residual_variance / result.model.growth_cov[0, 0]
    assert result.model.residual_variance > 0.05
    assert 0.5 * true_ratio < ratio < 2.5 * true_ratio


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


@pytest.mark.parametrize("growth_model", ["linear", "quadratic"])
def test_batched_growth_factor_sampler_matches_personwise_reference(
    growth_model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(84)
    n_persons = 37
    time_values = np.array([-0.5, 0.2, 1.7, 3.1])
    n_growth = 2 if growth_model == "linear" else 3
    covariance_root = rng.normal(size=(n_growth, n_growth))
    model = LongitudinalIRTModel(
        n_items=4,
        n_timepoints=time_values.size,
        growth_model=growth_model,
        growth_mean=rng.normal(size=n_growth),
        growth_cov=covariance_root @ covariance_root.T + 0.5 * np.eye(n_growth),
        residual_variance=0.35,
    )
    theta = rng.normal(size=(n_persons, time_values.size))
    expected_rng = np.random.default_rng(91)
    actual_rng = np.random.default_rng(91)
    expected = _reference_sample_growth_factors(
        theta,
        model,
        time_values,
        expected_rng,
    )
    sampler = LongitudinalGibbsSampler(n_iter=2, burnin=1)

    inverse_calls = 0
    original_inverse = np.linalg.inv

    def counting_inverse(values: np.ndarray) -> np.ndarray:
        nonlocal inverse_calls
        inverse_calls += 1
        return original_inverse(values)

    monkeypatch.setattr(dynamic_gibbs.np.linalg, "inv", counting_inverse)
    actual = sampler._sample_growth_factors(
        theta,
        model,
        time_values,
        actual_rng,
    )

    assert inverse_calls == 2
    assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    assert actual_rng.random() == expected_rng.random()


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses, _, time_values = longitudinal_data
    original_responses, original_times = responses.copy(), time_values.copy()
    options = dict(n_iter=4, burnin=2, seed=123)
    default_rng = np.random.default_rng
    generators = []
    seeds = []

    def record_generator(seed):
        generator = default_rng(seed)
        seeds.append(seed)
        generators.append(generator)
        return generator

    monkeypatch.setattr(dynamic_gibbs.np.random, "default_rng", record_generator)
    tensor_result = LongitudinalGibbsSampler(**options).fit(
        responses, time_values=time_values
    )
    flat_result = LongitudinalGibbsSampler(**options).fit(
        responses.reshape(responses.shape[0], -1),
        n_items=responses.shape[2],
        time_values=time_values,
    )

    assert seeds == [options["seed"], options["seed"]]
    assert generators[0].bit_generator.state == generators[1].bit_generator.state
    assert_array_equal(responses, original_responses)
    assert_array_equal(time_values, original_times)
    assert flat_result.n_iterations == tensor_result.n_iterations == options["n_iter"]
    assert flat_result.converged == tensor_result.converged
    for name in ("n_items", "n_timepoints", "n_growth_factors", "growth_model"):
        assert getattr(flat_result.model, name) == getattr(tensor_result.model, name)

    # Separate covariance solves and sampling transforms can round differently
    # at each recurrent sweep. Allow only a float64 roundoff budget per sweep.
    tolerance = 32 * options["n_iter"] * np.finfo(np.float64).eps
    for left, right, names in (
        (
            flat_result.model,
            tensor_result.model,
            (
                "discrimination",
                "difficulty",
                "growth_mean",
                "growth_cov",
                "residual_variance",
            ),
        ),
        (
            flat_result,
            tensor_result,
            (
                "growth_factors",
                "theta_trajectories",
                "growth_factor_se",
                "log_likelihood",
                "aic",
                "bic",
            ),
        ),
    ):
        for name in names:
            actual, expected = (
                np.asarray(getattr(left, name)),
                np.asarray(getattr(right, name)),
            )
            assert actual.shape == expected.shape
            assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def _growth_data(
    n_persons: int, seed: int, n_timepoints: int = 4
) -> tuple[LongitudinalIRTModel, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_items = 10
    generator = LongitudinalIRTModel(
        n_items=n_items,
        n_timepoints=n_timepoints,
        discrimination=rng.uniform(0.8, 2.0, n_items),
        difficulty=rng.normal(0.0, 1.0, n_items),
        growth_mean=np.array([0.0, 0.4]),
        growth_cov=np.array([[1.0, 0.1], [0.1, 0.09]]),
        residual_variance=0.2,
    )
    responses, _, growth_factors = generator.simulate(n_persons, seed=seed + 1)
    return generator, responses, growth_factors


def test_fit_identifies_the_latent_scale_and_recovers_parameters() -> None:
    """Regression: the latent location and scale drifted with chain length.

    growth_cov[0, 0] climbed from 1.06 to 1.91 over 6000 sweeps (truth 1),
    the residual variance from 0.21 to 0.50 and mean a fell from 1.19 to
    0.94. The intercept factor is now standard normal in every draw.
    """
    generator, responses, _ = _growth_data(300, seed=1)

    result = LongitudinalGibbsSampler(n_iter=400, burnin=200, seed=1).fit(responses)

    model = result.model
    assert model.growth_mean[0] == 0.0
    assert model.growth_cov[0, 0] == 1.0
    ratio = model.discrimination.mean() / generator.discrimination.mean()
    assert 0.8 < ratio < 1.2
    assert abs(np.mean(model.difficulty - generator.difficulty)) < 0.2
    assert abs(model.growth_mean[1] - 0.4) < 0.15
    assert 0.1 < model.residual_variance < 0.4
    n_params = 2 * 10 + 1 + 3
    assert result.aic == pytest.approx(-2.0 * result.log_likelihood + 2 * n_params)


def test_quadratic_fit_is_not_worse_than_the_nested_linear_fit() -> None:
    """Regression: the quadratic chain drifted to a 4x latent scale.

    Its plug-in log-likelihood was about 1700 below the nested linear fit on
    these data, so AIC and BIC comparisons were meaningless.
    """
    rng = np.random.default_rng(0)
    n_items = 8
    generator = LongitudinalIRTModel(
        n_items=n_items,
        n_timepoints=5,
        discrimination=rng.uniform(0.8, 2.0, n_items),
        difficulty=rng.normal(0.0, 1.0, n_items),
        growth_mean=np.array([0.0, 0.3]),
        growth_cov=np.array([[1.0, 0.0], [0.0, 0.05]]),
        residual_variance=0.2,
    )
    responses, _, _ = generator.simulate(200, seed=2)
    sampler = LongitudinalGibbsSampler(n_iter=300, burnin=150, seed=0)

    linear = sampler.fit(responses)
    quadratic = sampler.fit(responses, growth_model="quadratic")

    assert quadratic.model.growth_mean[0] == 0.0
    assert quadratic.model.growth_cov[0, 0] == 1.0
    assert quadratic.log_likelihood > linear.log_likelihood - 50.0


def _prior_state(
    sampler: LongitudinalGibbsSampler,
    n_persons: int,
    time_values: np.ndarray,
    growth_model: str,
    rng: np.random.Generator,
) -> tuple[LongitudinalIRTModel, np.ndarray, np.ndarray]:
    """Draw every latent and item quantity from the identified prior."""
    priors = sampler.priors
    model = LongitudinalIRTModel(
        n_items=3,
        n_timepoints=time_values.size,
        growth_model=growth_model,
    )
    n_growth = model.n_growth_factors
    prior_mean, prior_precision = sampler._growth_mean_prior(n_growth)
    slope_precision = prior_precision[1:, 1:]
    slope_mean = prior_mean[1:] + np.linalg.solve(
        slope_precision, prior_precision[1:, 0] * prior_mean[0]
    )
    model.growth_mean = np.concatenate(
        ([0.0], rng.multivariate_normal(slope_mean, np.linalg.inv(slope_precision)))
    )
    model.growth_cov = sample_unit_leading_covariance(
        np.empty((0, n_growth)),
        priors.growth_cov_prior_df,
        sampler._growth_cov_prior_scale(n_growth),
        rng,
    )
    model.residual_variance = float(
        stats.invgamma.rvs(
            priors.residual_var_prior_shape,
            scale=priors.residual_var_prior_rate,
            random_state=rng,
        )
    )
    model.discrimination = np.exp(
        np.log(priors.discrimination_mean)
        + np.sqrt(priors.discrimination_var) * rng.standard_normal(3)
    )
    model.difficulty = priors.difficulty_mean + np.sqrt(
        priors.difficulty_var
    ) * rng.standard_normal(3)
    growth_factors = rng.multivariate_normal(
        model.growth_mean, model.growth_cov, size=n_persons
    )
    theta = model.compute_theta(growth_factors, time_values) + np.sqrt(
        model.residual_variance
    ) * rng.standard_normal((n_persons, time_values.size))
    return model, growth_factors, theta


@pytest.mark.parametrize("growth_model", ["linear", "quadratic"])
def test_location_scale_moves_preserve_the_prior(growth_model: str) -> None:
    """Without responses the joint prior is the target, so the moves keep it.

    States drawn exactly from the prior must keep their distribution after
    repeated location and scale moves. Dropping any Jacobian term shifts the
    log discrimination, residual-variance or slope-variance moments by 0.1
    to 0.2.
    """
    n_growth = 2 if growth_model == "linear" else 3
    priors = LongitudinalPriors(
        discrimination_var=0.5,
        growth_mean_prior_cov=np.eye(n_growth),
    )
    sampler = LongitudinalGibbsSampler(n_iter=2, burnin=1, priors=priors)
    rng = np.random.default_rng(1)
    time_values = np.array([0.0, 1.0, 2.0])
    log_discrimination, difficulty, log_residual, log_slope_residual = [], [], [], []
    intercepts = []
    for _ in range(1500):
        model, growth_factors, theta = _prior_state(
            sampler, 4, time_values, growth_model, rng
        )
        for _ in range(8):
            theta, growth_factors = sampler._sample_location_scale(
                theta, growth_factors, model, rng
            )
        log_discrimination.append(np.log(model.discrimination[0]))
        difficulty.append(model.difficulty[0])
        log_residual.append(np.log(model.residual_variance))
        log_slope_residual.append(
            np.log(model.growth_cov[1, 1] - model.growth_cov[0, 1] ** 2)
        )
        intercepts.append(growth_factors[0, 0])
        assert model.growth_mean[0] == 0.0
        assert model.growth_cov[0, 0] == 1.0

    # log of an inverse-gamma(k, s) draw has mean log(s) - digamma(k).
    residual_mean = np.log(priors.residual_var_prior_rate) - digamma(
        priors.residual_var_prior_shape
    )
    slope_shape = priors.growth_cov_prior_df / 2.0
    assert abs(np.mean(log_discrimination)) < 0.06
    assert abs(np.std(log_discrimination) - np.sqrt(0.5)) < 0.05
    assert abs(np.mean(difficulty)) < 0.16
    assert abs(np.mean(log_residual) - residual_mean) < 0.07
    if growth_model == "linear":
        slope_mean = np.log(0.5) - digamma(slope_shape)
        assert abs(np.mean(log_slope_residual) - slope_mean) < 0.07
    assert abs(np.mean(intercepts)) < 0.08
    assert abs(np.var(intercepts) - 1.0) < 0.1


def test_item_step_row_likelihoods_are_reused_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The theta step reuses the item step's current-state likelihoods."""
    _, responses, growth_factors = _growth_data(40, seed=3)
    model = LongitudinalIRTModel(n_items=10, n_timepoints=4)
    theta = model.compute_theta(growth_factors)
    sampler = LongitudinalGibbsSampler(n_iter=2, burnin=1)

    rows = sampler._sample_item_params(
        responses, theta, model, np.random.default_rng(0)
    )

    assert rows is not None
    assert_array_equal(
        rows,
        dynamic_gibbs._longitudinal_log_likelihood_rows(
            responses.reshape(-1, 10), theta.reshape(-1), model
        ),
    )

    def fit() -> Any:
        return LongitudinalGibbsSampler(n_iter=30, burnin=10, seed=4).fit(responses)

    reused = fit()
    monkeypatch.setattr(dynamic_gibbs, "_LONGITUDINAL_REUSE_MAX_VALUES", 0)
    assert (
        sampler._sample_item_params(responses, theta, model, np.random.default_rng(0))
        is None
    )
    recomputed = fit()
    assert_array_equal(reused.growth_factors, recomputed.growth_factors)
    assert_array_equal(reused.model.discrimination, recomputed.model.discrimination)


def test_logit_likelihood_is_exact_in_the_tails() -> None:
    """The fused form needs no probability clipping, even far in the tails."""
    logits = np.array([[-60.0, -5.0, 0.0, 5.0, 60.0]] * 3)
    responses = np.array(
        [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [-1, -1, -1, -1, -1]], dtype=np.int_
    )

    cells = dynamic_gibbs._binary_log_likelihood_cells(logits.copy(), responses)

    assert_allclose(cells[0], -np.logaddexp(0.0, -logits[0]), rtol=1e-14)
    assert_allclose(cells[1], -np.logaddexp(0.0, logits[1]), rtol=1e-14)
    assert_array_equal(cells[2], 0.0)
    assert cells[0, 0] == pytest.approx(-60.0)


@pytest.mark.slow
def test_long_chains_are_stationary_and_agree_across_seeds() -> None:
    """Block means stay put and independent seeds agree (MA-16 trace)."""
    generator, responses, _ = _growth_data(500, seed=0)
    traces = []

    for seed in (0, 1):
        trace: list[tuple[float, float, float]] = []
        sampler = LongitudinalGibbsSampler(n_iter=3000, burnin=500, seed=seed)
        original = sampler._sample_residual_variance

        def recording(*args: Any, original: Any = original, trace: Any = trace):
            original(*args)
            model = args[2]
            trace.append(
                (
                    model.residual_variance,
                    model.growth_cov[1, 1],
                    float(np.mean(model.discrimination)),
                )
            )

        sampler._sample_residual_variance = recording  # type: ignore[method-assign]
        sampler.fit(responses)
        traces.append(np.asarray(trace)[500:])

    for trace in traces:
        blocks = trace.reshape(5, -1, 3).mean(axis=1)
        assert np.ptp(blocks[:, 0]) < 0.1
        assert np.ptp(blocks[:, 2]) < 0.15
    first, second = (trace.mean(axis=0) for trace in traces)
    assert_allclose(first, second, rtol=0.15)
    assert 0.8 < first[2] / generator.discrimination.mean() < 1.1
