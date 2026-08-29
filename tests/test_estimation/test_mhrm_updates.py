"""Regression coverage for Python MHRM parameter updates."""

import numpy as np
import pytest

import mirt.estimation.mcmc as mcmc_module
from mirt import MHRMEstimator, TwoParameterLogistic
from mirt.constants import PROB_EPSILON


class _CountingTwoPL(TwoParameterLogistic):
    def _initialize_parameters(self) -> None:
        super()._initialize_parameters()
        self.probability_calls = 0

    def probability(self, theta, item_idx=None):
        self.probability_calls += 1
        return super().probability(theta, item_idx)


def _reference_update(
    model: TwoParameterLogistic,
    responses: np.ndarray,
    theta: np.ndarray,
    gain: float,
) -> None:
    discrimination = model._parameters["discrimination"]
    difficulty = model._parameters["difficulty"]
    for item_idx in range(model.n_items):
        valid = responses[:, item_idx] >= 0
        if not np.any(valid):
            continue
        probability = np.clip(
            model.probability(theta[valid], item_idx),
            PROB_EPSILON,
            1 - PROB_EPSILON,
        )
        residual = responses[valid, item_idx] - probability
        discrimination[item_idx] = np.clip(
            discrimination[item_idx] + gain * np.mean(residual * theta[valid].ravel()),
            0.1,
            5.0,
        )
        difficulty[item_idx] = np.clip(
            difficulty[item_idx] - gain * np.mean(residual),
            -6.0,
            6.0,
        )


@pytest.mark.parametrize(
    ("probability_budget", "expected_calls"),
    [(10_000, 1), (0, 3)],
)
def test_updates_match_itemwise_reference_and_respect_probability_budget(
    monkeypatch: pytest.MonkeyPatch,
    probability_budget: int,
    expected_calls: int,
) -> None:
    responses = np.array(
        [
            [1, 0, -1, -1],
            [0, 1, 1, -1],
            [1, 1, 0, -1],
            [0, -1, 1, -1],
            [1, 0, 1, -1],
        ],
        dtype=np.int64,
    )
    theta = np.array([[-1.2], [-0.4], [0.1], [0.8], [1.5]])
    initial = {
        "discrimination": np.array([0.8, 1.1, 1.4, 0.9]),
        "difficulty": np.array([-0.6, 0.2, 0.7, -0.1]),
    }
    model = _CountingTwoPL(4).set_parameters(**initial)
    reference = TwoParameterLogistic(4).set_parameters(**initial)
    estimator = MHRMEstimator(use_rust=False)
    rng = np.random.default_rng(8)

    monkeypatch.setattr(
        mcmc_module,
        "_MHRM_MAX_PROBABILITY_VALUES",
        probability_budget,
    )
    estimator._update_parameters(model, responses, theta, 0.15, rng)
    _reference_update(reference, responses, theta, 0.15)

    assert model.probability_calls == expected_calls
    np.testing.assert_allclose(
        model.parameters["discrimination"],
        reference.parameters["discrimination"],
    )
    np.testing.assert_allclose(
        model.parameters["difficulty"],
        reference.parameters["difficulty"],
    )
    assert model.parameters["discrimination"][3] == initial["discrimination"][3]
    assert model.parameters["difficulty"][3] == initial["difficulty"][3]


def test_python_fit_persists_parameter_updates() -> None:
    responses = np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 1, 1],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.int64,
    )
    model = TwoParameterLogistic(3)
    initial = {name: values.copy() for name, values in model.parameters.items()}

    result = MHRMEstimator(
        n_cycles=6,
        burnin=2,
        use_rust=False,
        seed=23,
    ).fit(model, responses)

    assert any(
        not np.array_equal(initial[name], values)
        for name, values in result.model.parameters.items()
    )
