"""Regression tests for survey-weighted EM estimation."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from mirt.estimation.weighted import (
    WeightedEMEstimator,
    compute_design_effect,
    compute_effective_sample_size,
)
from mirt.models.dichotomous import TwoParameterLogistic


@pytest.fixture
def responses() -> np.ndarray:
    return np.array([[0, 1], [1, 0], [1, 1], [0, 0]])


@pytest.mark.parametrize("normalize_weights", [True, False])
@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (np.zeros(4), "at least one positive"),
        (np.array([1.0, np.nan, 1.0, 1.0]), "finite"),
        (np.array([1.0, np.inf, 1.0, 1.0]), "finite"),
        (np.array([1.0, -1.0, 1.0, 1.0]), "non-negative"),
    ],
)
def test_fit_rejects_invalid_weights(
    responses: np.ndarray,
    normalize_weights: bool,
    weights: np.ndarray,
    message: str,
) -> None:
    estimator = WeightedEMEstimator(
        n_quadpts=5,
        max_iter=2,
        normalize_weights=normalize_weights,
    )

    with pytest.raises(ValueError, match=message):
        estimator.fit(TwoParameterLogistic(2), responses, weights=weights)


@pytest.mark.parametrize(
    "function", [compute_effective_sample_size, compute_design_effect]
)
@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (np.array([]), "at least one positive"),
        (np.zeros(3), "at least one positive"),
        (np.array([1.0, np.nan]), "finite"),
        (np.array([1.0, np.inf]), "finite"),
        (np.array([1.0, -1.0]), "non-negative"),
    ],
)
def test_weight_summaries_reject_invalid_values(
    function: Callable[[np.ndarray], float],
    weights: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        function(weights)


@pytest.mark.parametrize("normalize_weights", [None, 1, "yes"])
def test_normalize_weights_requires_boolean(normalize_weights: Any) -> None:
    with pytest.raises(ValueError, match="normalize_weights must be a boolean"):
        WeightedEMEstimator(normalize_weights=normalize_weights)


def test_weight_summary_values() -> None:
    weights = np.array([1.0, 2.0, 1.0])

    assert compute_effective_sample_size(weights) == pytest.approx(8.0 / 3.0)
    assert compute_design_effect(weights) == pytest.approx(9.0 / 8.0)


def test_extreme_finite_weights_do_not_overflow(responses: np.ndarray) -> None:
    weights = np.full(4, np.finfo(np.float64).max)

    assert compute_effective_sample_size(weights) == pytest.approx(4.0)
    assert compute_design_effect(weights) == pytest.approx(1.0)

    estimator = WeightedEMEstimator(n_quadpts=5, max_iter=1)
    estimator.fit(TwoParameterLogistic(2), responses, weights=weights)

    assert np.all(np.isfinite(estimator._weights))
    assert estimator._weights.sum() == pytest.approx(4.0)


def test_exhausted_fit_statistics_describe_returned_model(
    responses: np.ndarray,
) -> None:
    survey_weights = np.array([0.5, 1.0, 1.5, 2.0])
    estimator = WeightedEMEstimator(
        n_quadpts=7,
        max_iter=1,
        normalize_weights=False,
    )
    result = estimator.fit(TwoParameterLogistic(2), responses, weights=survey_weights)

    _, marginal = estimator._e_step_weighted(
        result.model,
        responses,
        np.zeros(1),
        np.eye(1),
        survey_weights,
    )
    expected_ll = float(np.sum(survey_weights * np.log(marginal)))
    effective_n = survey_weights.sum() ** 2 / np.sum(survey_weights**2)

    assert result.log_likelihood == pytest.approx(expected_ll, abs=1e-12)
    assert result.aic == pytest.approx(-2 * expected_ll + 2 * result.n_parameters)
    assert result.bic == pytest.approx(
        -2 * expected_ll + np.log(effective_n) * result.n_parameters
    )
    assert estimator.convergence_history[-1] == pytest.approx(expected_ll)


def test_convergence_on_last_allowed_iteration_is_reported(
    responses: np.ndarray,
) -> None:
    result = WeightedEMEstimator(
        n_quadpts=7,
        max_iter=2,
        tol=1e9,
    ).fit(TwoParameterLogistic(2), responses)

    assert result.n_iterations == 2
    assert result.converged is True
