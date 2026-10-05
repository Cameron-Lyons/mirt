"""Batched Newton M-step for built-in 1PL/2PL multigroup items."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.special import expit

from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.models.dichotomous import (
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.multigroup import InvarianceSpec, MultigroupEMEstimator, MultigroupModel


class ItemwiseEstimator(MultigroupEMEstimator):
    """Reference: the joint L-BFGS-B update of one item at a time."""

    def _uses_newton_m_step(self, model):
        return False


SPECS = {
    "configural": InvarianceSpec("configural"),
    "strict": InvarianceSpec("strict"),
    # Items 1 and 3 are group-specific, the others shared.
    "partial": InvarianceSpec(
        "strict", free_discrimination=[1, 3], free_intercepts=[1, 3]
    ),
}
MODELS = {
    "1PL": lambda: OneParameterLogistic(5),
    "2PL": lambda: TwoParameterLogistic(5),
    "2PL-2D": lambda: TwoParameterLogistic(5, n_factors=2),
}


def _m_step_inputs(factory, spec, seed=8):
    rng = np.random.default_rng(seed)
    model = MultigroupModel(factory(), 2)
    spec.apply_to_model(model)
    n_factors = model.n_factors
    quadrature = GaussHermiteQuadrature(9 if n_factors == 1 else 5, n_factors)
    responses, posterior = [], []
    for shift in (0.0, 0.4):
        theta = rng.normal(shift, 1.0, (400, n_factors))
        slopes = rng.uniform(0.8, 1.8, (5, n_factors))
        p = expit(theta @ slopes.T - rng.normal(0.0, 0.8, 5))
        data = (rng.random(p.shape) < p).astype(int)
        data[rng.random(data.shape) < 0.1] = -1
        responses.append(data)
        distance = ((theta[:, None, :] - quadrature.nodes[None]) ** 2).sum(axis=2)
        weights = np.exp(-distance)
        posterior.append(weights / weights.sum(axis=1, keepdims=True))
    for group, difficulty in zip(model.group_models, (0.2, -0.3), strict=True):
        group.set_parameters(difficulty=np.full(5, difficulty))
    model.synchronize_shared_parameters()
    return model, responses, posterior, quadrature


def _run_m_step(estimator, model, responses, posterior, quadrature, spec):
    estimator._quadrature = quadrature
    estimator._m_step(model, responses, posterior, spec)
    return model


@pytest.mark.parametrize("invariance", list(SPECS))
@pytest.mark.parametrize("kind", list(MODELS))
def test_newton_m_step_matches_the_itemwise_optimizer(kind, invariance):
    spec = SPECS[invariance]
    model, responses, posterior, quadrature = _m_step_inputs(MODELS[kind], spec)
    newton = _run_m_step(
        MultigroupEMEstimator(),
        deepcopy(model),
        responses,
        posterior,
        quadrature,
        spec,
    )
    reference = _run_m_step(
        ItemwiseEstimator(item_optim_ftol=1e-15, item_optim_maxiter=2000),
        deepcopy(model),
        responses,
        posterior,
        quadrature,
        spec,
    )
    for fitted, expected in zip(
        newton.group_models, reference.group_models, strict=True
    ):
        for name, values in expected.parameters.items():
            np.testing.assert_allclose(fitted.parameters[name], values, atol=2e-5)
        assert np.any(
            fitted.parameters["difficulty"] != model.get_group_model(0).difficulty
        )


def test_partly_shared_and_fixed_items_take_the_itemwise_path():
    spec = InvarianceSpec("metric")
    model, responses, posterior, quadrature = _m_step_inputs(
        MODELS["2PL"], spec, seed=2
    )
    estimator = MultigroupEMEstimator()
    estimator._quadrature = quadrature
    counts = [
        estimator._all_item_counts(group, data, weights)
        for group, data, weights in zip(
            model.group_models, responses, posterior, strict=True
        )
    ]
    masks = [model.effective_free_parameter_masks(g) for g in range(2)]
    # Metric invariance shares slopes but not difficulties.
    assert estimator._newton_m_step(model, counts, masks, quadrature.nodes) == list(
        range(5)
    )

    strict = MultigroupModel(TwoParameterLogistic(5), 2)
    InvarianceSpec("strict").apply_to_model(strict)
    strict.fix_item_parameters({"difficulty": {2: 0.5}})
    masks = [strict.effective_free_parameter_masks(g) for g in range(2)]
    assert estimator._newton_m_step(strict, counts, masks, quadrature.nodes) == [2]
    for group in strict.group_models:
        assert group.parameters["difficulty"][2] == 0.5


def test_newton_m_step_applies_only_to_unrestricted_builtin_items():
    estimator = MultigroupEMEstimator()
    assert estimator._uses_newton_m_step(MultigroupModel(TwoParameterLogistic(3), 2))
    assert estimator._uses_newton_m_step(MultigroupModel(OneParameterLogistic(3), 2))
    assert not estimator._uses_newton_m_step(
        MultigroupModel(ThreeParameterLogistic(3), 2)
    )
    restricted = TwoParameterLogistic(3)
    restricted.set_free_parameter_masks({"difficulty": np.array([True, False, True])})
    assert not estimator._uses_newton_m_step(MultigroupModel(restricted, 2))
    assert not MultigroupEMEstimator(prob_epsilon=1e-6)._uses_newton_m_step(
        MultigroupModel(TwoParameterLogistic(3), 2)
    )

    class Custom(MultigroupEMEstimator):
        def _optimize_item(self, *args, **kwargs):
            return super()._optimize_item(*args, **kwargs)

    assert not Custom()._uses_newton_m_step(MultigroupModel(TwoParameterLogistic(3), 2))
