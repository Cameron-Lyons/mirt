"""Regression coverage for shared multigroup parameter optimization."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.polytomous import GeneralizedPartialCredit
from mirt.multigroup.estimator import MultigroupEMEstimator
from mirt.multigroup.model import MultigroupModel

QUADRATURE = np.array([[-2.0], [-0.5], [0.5], [2.0]])
POSTERIOR_WEIGHTS = [
    np.array(
        [
            [0.70, 0.20, 0.08, 0.02],
            [0.40, 0.30, 0.20, 0.10],
            [0.10, 0.20, 0.30, 0.40],
            [0.02, 0.08, 0.20, 0.70],
        ]
    ),
    np.array(
        [
            [0.60, 0.25, 0.10, 0.05],
            [0.20, 0.30, 0.30, 0.20],
            [0.05, 0.15, 0.30, 0.50],
            [0.01, 0.04, 0.15, 0.80],
        ]
    ),
]


def _joint_optimum(objective: Callable[[float], float]) -> float:
    result = minimize_scalar(
        lambda value: -objective(float(value)),
        bounds=(0.1, 5.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    assert result.success
    return float(result.x)


def _optimize_shared_discrimination(
    model: MultigroupModel,
    responses: list[NDArray[np.int_]],
) -> float:
    estimator = MultigroupEMEstimator(
        item_optim_maxiter=200,
        item_optim_ftol=1e-12,
    )
    estimator._optimize_shared_item_param(
        model,
        item_idx=0,
        param_name="discrimination",
        responses=responses,
        posterior_weights=POSTERIOR_WEIGHTS,
        quad_points=QUADRATURE,
    )
    fitted = [
        float(group_model.parameters["discrimination"][0])
        for group_model in model.group_models
    ]
    assert fitted[0] == pytest.approx(fitted[1], abs=1e-12)
    return fitted[0]


def test_shared_dichotomous_parameter_uses_each_group_context() -> None:
    responses = [
        np.array([[0], [0], [1], [1]]),
        np.array([[0], [1], [1], [1]]),
    ]
    model = MultigroupModel(TwoParameterLogistic(1), n_groups=2)
    for group_model, difficulty in zip(model.group_models, (-1.75, 1.25), strict=True):
        group_model.set_parameters(
            discrimination=np.array([1.0]),
            difficulty=np.array([difficulty]),
        )
    model.set_shared_parameter("discrimination")

    def expected_log_likelihood(discrimination: float) -> float:
        total = 0.0
        for group_model, group_responses, weights in zip(
            model.group_models, responses, POSTERIOR_WEIGHTS, strict=True
        ):
            probability = 1.0 / (
                1.0
                + np.exp(
                    -discrimination * (QUADRATURE[:, 0] - group_model.difficulty[0])
                )
            )
            n_k = weights.sum(axis=0)
            r_k = (group_responses[:, 0, None] * weights).sum(axis=0)
            total += float(
                np.sum(r_k * np.log(probability) + (n_k - r_k) * np.log1p(-probability))
            )
        return total

    expected = _joint_optimum(expected_log_likelihood)
    actual = _optimize_shared_discrimination(model, responses)

    assert actual == pytest.approx(expected, abs=1e-6)


def test_shared_polytomous_parameter_uses_each_group_context() -> None:
    responses = [
        np.array([[0], [0], [1], [2]]),
        np.array([[0], [1], [2], [2]]),
    ]
    model = MultigroupModel(GeneralizedPartialCredit(1, 3), n_groups=2)
    group_steps = (np.array([[-1.8, 0.2]]), np.array([[-0.2, 1.8]]))
    for group_model, steps in zip(model.group_models, group_steps, strict=True):
        group_model.set_parameters(
            discrimination=np.array([1.0]),
            steps=steps,
        )
    model.set_shared_parameter("discrimination")

    def expected_log_likelihood(discrimination: float) -> float:
        total = 0.0
        for group_model, group_responses, weights in zip(
            model.group_models, responses, POSTERIOR_WEIGHTS, strict=True
        ):
            group_model._parameters["discrimination"][0] = discrimination
            probabilities = group_model.probability(QUADRATURE, item_idx=0)
            for category in range(3):
                category_weights = weights[group_responses[:, 0] == category]
                total += float(
                    np.sum(
                        category_weights.sum(axis=0)
                        * np.log(probabilities[:, category])
                    )
                )
        return total

    expected = _joint_optimum(expected_log_likelihood)
    for group_model in model.group_models:
        group_model._parameters["discrimination"][0] = 1.0
    actual = _optimize_shared_discrimination(model, responses)

    assert actual == pytest.approx(expected, abs=1e-6)
