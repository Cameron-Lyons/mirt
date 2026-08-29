"""Contracts for posterior predictive diagnostics and pointwise likelihoods."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mirt.diagnostics.bayesian import (
    compute_pointwise_log_lik,
    posterior_predictive_check,
)
from mirt.estimation.mcmc import MCMCResult
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel


def _dichotomous_case() -> tuple[
    TwoParameterLogistic,
    np.ndarray,
    dict[str, np.ndarray],
]:
    model = TwoParameterLogistic(3).set_parameters(
        discrimination=np.array([0.8, 1.1, 1.4]),
        difficulty=np.array([-0.7, 0.1, 0.9]),
    )
    responses = np.array(
        [
            [1, 0, -1],
            [0, 1, 1],
            [1, -1, 0],
            [0, 0, 1],
        ]
    )
    chains = {
        "discrimination": np.array(
            [
                [0.7, 1.0, 1.3],
                [0.8, 1.1, 1.4],
                [0.9, 1.2, 1.5],
            ]
        ),
        "difficulty": np.array(
            [
                [-0.9, -0.1, 0.7],
                [-0.7, 0.1, 0.9],
                [-0.5, 0.3, 1.1],
            ]
        ),
        "theta": np.array(
            [
                [[-1.0], [-0.2], [0.4], [1.1]],
                [[-0.8], [0.0], [0.6], [1.3]],
                [[-0.6], [0.2], [0.8], [1.5]],
            ]
        ),
    }
    return model, responses, chains


def _mcmc_result(
    model: TwoParameterLogistic | GradedResponseModel,
    chains: dict[str, np.ndarray],
) -> MCMCResult:
    return MCMCResult(
        model=model,
        chains=chains,
        log_likelihood=0.0,
        dic=0.0,
        waic=0.0,
        rhat={},
        ess={},
        n_iterations=len(next(iter(chains.values()))),
        burnin=0,
        thin=1,
    )


def _manual_dichotomous_log_likelihood(
    responses: np.ndarray,
    chains: dict[str, np.ndarray],
) -> np.ndarray:
    observed = responses >= 0
    result = np.zeros((chains["theta"].shape[0], *responses.shape))
    for sample_idx, theta in enumerate(chains["theta"]):
        discrimination = chains["discrimination"][sample_idx]
        difficulty = chains["difficulty"][sample_idx]
        probability = 1.0 / (
            1.0
            + np.exp(-discrimination[None, :] * (theta[:, :1] - difficulty[None, :]))
        )
        pointwise = np.where(
            responses == 1,
            np.log(probability),
            np.log1p(-probability),
        )
        result[sample_idx] = np.where(observed, pointwise, 0.0)
    return result


def test_pointwise_dichotomous_matches_manual_calculation() -> None:
    model, responses, chains = _dichotomous_case()
    expected = _manual_dichotomous_log_likelihood(responses, chains)

    by_person = compute_pointwise_log_lik(model, responses, chains)
    by_observation = compute_pointwise_log_lik(
        model,
        responses,
        chains,
        by="observation",
    )

    assert_allclose(by_person, expected.sum(axis=2))
    assert_allclose(by_observation, expected.reshape(expected.shape[0], -1))


def test_observed_layout_excludes_missing_cells() -> None:
    model, responses, chains = _dichotomous_case()
    expected = _manual_dichotomous_log_likelihood(responses, chains)
    observed = responses >= 0

    actual = compute_pointwise_log_lik(model, responses, chains, by="observed")

    assert actual.shape == (len(chains["theta"]), int(np.sum(observed)))
    assert_allclose(actual, expected[:, observed])


def test_pointwise_likelihood_restores_model_parameters() -> None:
    model, responses, chains = _dichotomous_case()
    original = model.parameters
    shifted = {**chains, "difficulty": chains["difficulty"] + 3.0}

    compute_pointwise_log_lik(model, responses, shifted)

    for name, values in original.items():
        assert_array_equal(model.parameters[name], values)


def test_native_style_unidimensional_theta_chain_is_supported() -> None:
    model, responses, chains = _dichotomous_case()
    sampled_2d = {**chains, "theta": chains["theta"][..., 0]}

    expected = compute_pointwise_log_lik(model, responses, chains)
    actual = compute_pointwise_log_lik(model, responses, sampled_2d)

    assert_allclose(actual, expected)


def test_pointwise_likelihood_broadcasts_fixed_parameters_and_theta() -> None:
    model, responses, chains = _dichotomous_case()
    n_samples = 3
    fixed_theta = chains["theta"][0]
    fixed = {
        "log_likelihood": np.zeros(n_samples),
        "theta": fixed_theta,
    }
    sampled = {
        "discrimination": np.tile(model.discrimination, (n_samples, 1)),
        "difficulty": np.tile(model.difficulty, (n_samples, 1)),
        "theta": np.tile(fixed_theta, (n_samples, 1, 1)),
    }

    expected = compute_pointwise_log_lik(model, responses, sampled, by="observed")
    actual = compute_pointwise_log_lik(model, responses, fixed, by="observed")

    assert_allclose(actual, expected)


def test_boolean_dichotomous_responses_are_supported() -> None:
    model, responses, chains = _dichotomous_case()
    complete = np.maximum(responses, 0)

    integer_result = compute_pointwise_log_lik(model, complete, chains)
    boolean_result = compute_pointwise_log_lik(model, complete.astype(bool), chains)

    assert_allclose(boolean_result, integer_result)


def test_polytomous_pointwise_likelihood_selects_response_categories() -> None:
    model = GradedResponseModel(2, n_categories=[3, 4]).set_parameters(
        discrimination=np.array([0.9, 1.2]),
        thresholds=np.array([[-1.0, 0.8, 0.0], [-1.2, -0.1, 1.1]]),
    )
    responses = np.array([[0, 3], [2, 1], [1, -1]])
    theta = np.array([[[-0.8], [0.2], [1.0]], [[-0.4], [0.5], [1.3]]])
    chains = {
        "discrimination": np.tile(model.discrimination, (2, 1)),
        "thresholds": np.tile(model.thresholds, (2, 1, 1)),
        "theta": theta,
    }
    original = model.parameters

    actual = compute_pointwise_log_lik(model, responses, chains, by="observation")

    expected = np.zeros((2, responses.size))
    for sample_idx in range(2):
        probability = model.probability(theta[sample_idx])
        selected = []
        for person_idx, item_idx in np.ndindex(responses.shape):
            response = responses[person_idx, item_idx]
            selected.append(
                0.0
                if response < 0
                else np.log(probability[person_idx, item_idx, response])
            )
        expected[sample_idx] = selected

    assert_allclose(actual, expected)
    for name, values in original.items():
        assert_array_equal(model.parameters[name], values)


def test_polytomous_predictive_check_preserves_categories_and_missingness() -> None:
    model = GradedResponseModel(2, n_categories=[3, 4])
    responses = np.array([[0, 3], [2, 1], [1, -1], [0, 2]])
    chains = {
        "discrimination": np.tile(model.discrimination, (4, 1)),
        "thresholds": np.tile(model.thresholds, (4, 1, 1)),
        "theta": np.zeros((4, 4, 1)),
    }
    captured: list[np.ndarray] = []

    def statistic(values: np.ndarray) -> float:
        captured.append(values.copy())
        return float(np.mean(values[values >= 0]))

    result = posterior_predictive_check(
        _mcmc_result(model, chains),
        responses,
        model,
        test_statistic=statistic,
        n_rep=4,
        seed=7,
    )

    assert result.test_statistic_replicated.shape == (4,)
    for replicated in captured[1:]:
        assert_array_equal(replicated < 0, responses < 0)
        assert np.all(replicated[:, 0][replicated[:, 0] >= 0] < 3)
        assert np.all(replicated[:, 1][replicated[:, 1] >= 0] < 4)


def test_predictive_check_is_reproducible_and_restores_model() -> None:
    model, responses, chains = _dichotomous_case()
    original = model.parameters
    mcmc_result = _mcmc_result(model, chains)

    first = posterior_predictive_check(
        mcmc_result,
        responses,
        model,
        n_rep=5,
        seed=17,
    )
    second = posterior_predictive_check(
        mcmc_result,
        responses,
        model,
        n_rep=5,
        seed=17,
    )

    assert_array_equal(
        first.test_statistic_replicated,
        second.test_statistic_replicated,
    )
    for name, values in original.items():
        assert_array_equal(model.parameters[name], values)


def test_predictive_check_restores_model_when_statistic_raises() -> None:
    model, responses, chains = _dichotomous_case()
    original = model.parameters
    call_count = 0

    def failing_statistic(values: np.ndarray) -> float:
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise RuntimeError("statistic failed")
        return float(np.mean(values[values >= 0]))

    with pytest.raises(RuntimeError, match="statistic failed"):
        posterior_predictive_check(
            _mcmc_result(model, chains),
            responses,
            model,
            test_statistic=failing_statistic,
            n_rep=2,
            seed=3,
        )

    for name, values in original.items():
        assert_array_equal(model.parameters[name], values)


@pytest.mark.parametrize(
    ("modify", "message"),
    [
        (lambda responses: responses[:, :2], "expected 3"),
        (lambda responses: responses.astype(float) + 0.5, "integer category"),
        (
            lambda responses: np.where(
                np.arange(responses.size).reshape(responses.shape) == 0,
                np.nan,
                responses,
            ),
            "finite integer",
        ),
        (lambda responses: np.where(responses < 0, responses, 2), "unsupported"),
    ],
)
def test_response_validation_rejects_invalid_matrices(
    modify: Callable[[np.ndarray], np.ndarray],
    message: str,
) -> None:
    model, responses, chains = _dichotomous_case()

    with pytest.raises(ValueError, match=message):
        compute_pointwise_log_lik(model, modify(responses), chains)


def test_pointwise_validation_rejects_invalid_layout_and_chain_lengths() -> None:
    model, responses, chains = _dichotomous_case()

    with pytest.raises(ValueError, match="by must be"):
        compute_pointwise_log_lik(model, responses, chains, by="typo")

    inconsistent = {**chains, "difficulty": chains["difficulty"][:2]}
    with pytest.raises(ValueError, match="same number"):
        compute_pointwise_log_lik(model, responses, inconsistent)

    nonfinite = chains.copy()
    nonfinite["difficulty"] = chains["difficulty"].copy()
    nonfinite["difficulty"][0, 0] = np.nan
    with pytest.raises(ValueError, match="only finite"):
        compute_pointwise_log_lik(model, responses, nonfinite)


@pytest.mark.parametrize("n_rep", [0, -1, 1.5, True])
def test_predictive_check_requires_positive_integer_replications(
    n_rep: object,
) -> None:
    model, responses, chains = _dichotomous_case()

    with pytest.raises(ValueError, match="positive integer"):
        posterior_predictive_check(
            _mcmc_result(model, chains),
            responses,
            model,
            n_rep=n_rep,  # type: ignore[arg-type]
        )
