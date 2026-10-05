"""Joint item M-step for built-in binary multigroup models."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import expit

from mirt.models.dichotomous import (
    FourParameterLogistic,
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.polytomous import GradedResponseModel
from mirt.multigroup import _prepare_multigroup
from mirt.multigroup.estimator import MultigroupEMEstimator
from mirt.multigroup.model import MultigroupModel

NODES = np.linspace(-3.0, 3.0, 9)[:, None]


class PerParameterEstimator(MultigroupEMEstimator):
    """Reference M-step: one parameter name at a time, as before the joint path.

    Overriding ``_optimize_item`` also disables the batched Newton M-step.
    """

    def _optimize_item(
        self,
        model,
        item_idx,
        responses,
        posterior_weights,
        quad_points,
        *,
        masks=None,
        counts=None,
    ):
        groups = model.group_models
        if counts is None:
            counts = [
                self._expected_item_counts(group, item_idx, data, posterior)
                for group, data, posterior in zip(
                    groups, responses, posterior_weights, strict=True
                )
            ]
        if masks is None:
            masks = [
                model.effective_free_parameter_masks(g) for g in range(len(groups))
            ]
        for name in model.parameter_names:
            name_masks = [group_masks[name][item_idx].ravel() for group_masks in masks]
            if model.is_item_parameter_shared(name, item_idx):
                blocks = [(groups, name_masks, counts, {name})]
            else:
                blocks = [
                    ([group], [mask], [group_counts], set())
                    for group, mask, group_counts in zip(
                        groups, name_masks, counts, strict=True
                    )
                ]
            for block_groups, block_masks, block_counts, shared in blocks:
                if not np.any(block_masks):
                    continue
                if not self._optimize_binary_block(
                    block_groups,
                    item_idx,
                    {name: block_masks},
                    shared,
                    block_counts,
                    quad_points,
                ):
                    self._optimize_parameter_block(
                        block_groups,
                        block_masks,
                        block_counts,
                        item_idx,
                        name,
                        quad_points,
                    )


def _simulate(model: str, n_items: int, n_persons: int, seed: int):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.8, 2.0, n_items)
    b = rng.normal(0.0, 1.0, n_items)
    guessing = 0.15 if model == "3PL" else 0.0
    blocks = []
    for mean in (0.0, 0.5):
        theta = rng.normal(mean, 1.0, n_persons)
        p = guessing + (1 - guessing) * expit(a * (theta[:, None] - b))
        blocks.append((rng.random(p.shape) < p).astype(int))
    data = np.vstack(blocks)
    groups = np.repeat([0, 1], n_persons)
    return data, groups


def _fit(estimator_class, model, invariance, data, groups, **kwargs):
    mg_model, responses, reference = _prepare_multigroup(
        data, groups, model, n_categories=None, reference_group=0, item_names=None
    )
    result = estimator_class(n_quadpts=15, **kwargs).fit(
        mg_model, responses, invariance, reference
    )
    return mg_model, result


def _stacked_parameters(model: MultigroupModel) -> np.ndarray:
    return np.concatenate(
        [
            np.concatenate([np.ravel(value) for value in group.parameters.values()])
            for group in model.group_models
        ]
    )


@pytest.mark.parametrize("invariance", ["configural", "metric", "scalar"])
def test_joint_m_step_reaches_the_per_parameter_maximum(invariance):
    data, groups = _simulate("2PL", n_items=5, n_persons=300, seed=31)
    settings = {"max_iter": 3000, "tol": 1e-7, "item_optim_ftol": 1e-12}

    joint_model, joint = _fit(
        MultigroupEMEstimator, "2PL", invariance, data, groups, **settings
    )
    reference_model, reference = _fit(
        PerParameterEstimator, "2PL", invariance, data, groups, **settings
    )

    assert joint.converged and reference.converged
    assert joint.log_likelihood == pytest.approx(reference.log_likelihood, abs=1e-5)
    np.testing.assert_allclose(
        _stacked_parameters(joint_model),
        _stacked_parameters(reference_model),
        atol=2e-3,
    )
    np.testing.assert_allclose(
        joint.latent_distributions[1].mean,
        reference.latent_distributions[1].mean,
        atol=2e-3,
    )


def _informative_item(rng, difficulty, guessing, n_persons=150):
    """Responses to one item and posteriors centred on the true abilities."""
    theta = rng.normal(0.0, 1.0, n_persons)
    p = guessing + (1 - guessing) * expit(1.4 * (theta - difficulty))
    responses = (rng.random(n_persons) < p).astype(int)[:, None]
    log_weights = -((NODES[:, 0] - theta[:, None]) ** 2)
    posterior = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
    return responses, posterior / posterior.sum(axis=1, keepdims=True)


SHARED = {
    "configural": (),
    "metric": ("discrimination",),
    "scalar": ("discrimination", "difficulty"),
}


@pytest.mark.parametrize("invariance", ["configural", "metric", "scalar"])
@pytest.mark.parametrize("model_class", [TwoParameterLogistic, ThreeParameterLogistic])
def test_item_update_is_the_joint_expected_likelihood_maximum(model_class, invariance):
    rng = np.random.default_rng(56)
    guessing = 0.2 if model_class is ThreeParameterLogistic else 0.0
    data = [_informative_item(rng, b, guessing) for b in (-0.5, 0.7)]
    responses = [block[0] for block in data]
    posterior = [block[1] for block in data]
    model = MultigroupModel(model_class(1), 2)
    for name in SHARED[invariance]:
        model.set_shared_parameter(name)
    if invariance != "scalar":
        for group, difficulty in zip(model.group_models, (-0.4, 0.6), strict=True):
            group.set_parameters(difficulty=np.array([difficulty]))

    # Oracle variables: one per shared name, otherwise one per group.
    names = list(model.parameter_names)
    layout = []
    for name in names:
        layout.append(
            [(name, None)] if name in SHARED[invariance] else [(name, 0), (name, 1)]
        )
    variables = [slot for block in layout for slot in block]
    start = np.array(
        [model.get_group_model(g or 0).parameters[name][0] for name, g in variables]
    )
    bounds = {"discrimination": (0.1, 5.0), "difficulty": (-6, 6), "guessing": (0, 0.5)}
    counts = [
        (weights.sum(axis=0), (r[:, 0, None] * weights).sum(axis=0))
        for r, weights in zip(responses, posterior, strict=True)
    ]

    def group_values(params, g):
        return {
            name: params[i]
            for i, (name, owner) in enumerate(variables)
            if owner in (None, g)
        }

    def loss(params):
        total = 0.0
        for g, (observed, correct) in enumerate(counts):
            values = group_values(params, g)
            c = values.get("guessing", 0.0)
            curve = expit(
                values["discrimination"] * (NODES[:, 0] - values["difficulty"])
            )
            p = c + (1 - c) * curve
            total -= correct @ np.log(p) + (observed - correct) @ np.log1p(-p)
        return total

    oracle = minimize(
        loss,
        x0=start,
        method="L-BFGS-B",
        bounds=[bounds[name] for name, _ in variables],
        options={"ftol": 1e-15, "gtol": 1e-9, "maxiter": 2000},
    )
    MultigroupEMEstimator(
        item_optim_ftol=1e-15, item_optim_maxiter=2000
    )._optimize_item(model, 0, responses, posterior, NODES)
    fitted = np.array(
        [model.get_group_model(g or 0).parameters[name][0] for name, g in variables]
    )
    for name in SHARED[invariance]:
        first, second = (group.parameters[name][0] for group in model.group_models)
        assert first == second
    assert loss(fitted) == pytest.approx(oracle.fun, abs=1e-7)
    np.testing.assert_allclose(fitted, oracle.x, atol=1e-3)

    if invariance == "metric":
        # One coordinate sweep of the previous M-step stops short of the optimum.
        sweep = MultigroupModel(model_class(1), 2)
        sweep.set_shared_parameter("discrimination")
        for group, difficulty in zip(sweep.group_models, (-0.4, 0.6), strict=True):
            group.set_parameters(difficulty=np.array([difficulty]))
        PerParameterEstimator(item_optim_ftol=1e-15)._optimize_item(
            sweep, 0, responses, posterior, NODES
        )
        swept = np.array(
            [sweep.get_group_model(g or 0).parameters[name][0] for name, g in variables]
        )
        assert loss(fitted) < loss(swept) - 1e-6


@pytest.mark.parametrize(
    "model_class, seed", [(ThreeParameterLogistic, 3), (FourParameterLogistic, 2)]
)
def test_default_item_update_reaches_the_joint_optimum(model_class, seed):
    # Item losses are large sums: L-BFGS-B's default relative ftol stopped
    # these flat lower/upper-asymptote solves about 0.03 short of the optimum.
    nodes = np.linspace(-4.0, 4.0, 15)[:, None]
    rng = np.random.default_rng(seed)
    upper = 0.92 if model_class is FourParameterLogistic else 1.0
    responses, posterior = [], []
    for difficulty in (-0.5, 0.7):
        theta = rng.normal(0.0, 1.0, 1500)
        p = 0.2 + (upper - 0.2) * expit(1.4 * (theta - difficulty))
        responses.append((rng.random(theta.size) < p).astype(int)[:, None])
        log_weights = -((nodes[:, 0] - theta[:, None]) ** 2)
        weights = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
        posterior.append(weights / weights.sum(axis=1, keepdims=True))
    model = MultigroupModel(model_class(1), 2)
    model.set_shared_parameter("discrimination")
    names = [name for name in model.parameter_names if name != "discrimination"]
    variables = [("discrimination", 0)] + [(n, g) for n in names for g in (0, 1)]
    bounds = {
        "discrimination": (0.1, 5.0),
        "difficulty": (-6.0, 6.0),
        "guessing": (0.0, 0.5),
        "upper": (0.5, 1.0),
    }
    counts = [
        (weights.sum(axis=0), (r[:, 0, None] * weights).sum(axis=0))
        for r, weights in zip(responses, posterior, strict=True)
    ]

    def loss(params):
        total = 0.0
        for g, (observed, correct) in enumerate(counts):
            value = {
                name: params[i]
                for i, (name, owner) in enumerate(variables)
                if name == "discrimination" or owner == g
            }
            c, d = value["guessing"], value.get("upper", 1.0)
            curve = expit(value["discrimination"] * (nodes[:, 0] - value["difficulty"]))
            p = c + (d - c) * curve
            total -= correct @ np.log(p) + (observed - correct) @ np.log1p(-p)
        return total

    def current():
        return np.array(
            [model.get_group_model(g).parameters[name][0] for name, g in variables]
        )

    oracle = minimize(
        loss,
        x0=current(),
        method="L-BFGS-B",
        bounds=[bounds[name] for name, _ in variables],
        options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 5000},
    )
    MultigroupEMEstimator()._optimize_item(model, 0, responses, posterior, nodes)
    assert loss(current()) == pytest.approx(oracle.fun, abs=1e-6)
    np.testing.assert_allclose(current(), oracle.x, atol=1e-3)


def test_groups_with_custom_curves_are_not_pooled_into_the_builtin_kernel():
    # Equal shared parameters used to pool every group's counts into the
    # built-in kernel, silently ignoring the other group's probability hook.
    from types import MethodType

    rng = np.random.default_rng(4)
    data = [_informative_item(rng, b, 0.0) for b in (-0.3, 0.4)]
    responses = [block[0] for block in data]
    posterior = [block[1] for block in data]
    model = MultigroupModel(TwoParameterLogistic(1), 2)
    for name in ("discrimination", "difficulty"):
        model.set_shared_parameter(name)
    builtin = TwoParameterLogistic.probability
    shifted = model.get_group_model(1)
    shifted.probability = MethodType(
        lambda self, theta, item_idx=None: builtin(self, theta + 1.0, item_idx),
        shifted,
    )
    counts = [
        (weights.sum(axis=0), (r[:, 0, None] * weights).sum(axis=0))
        for r, weights in zip(responses, posterior, strict=True)
    ]

    def loss(params):
        total = 0.0
        for shift, (observed, correct) in zip((0.0, 1.0), counts, strict=True):
            p = expit(params[0] * (NODES[:, 0] + shift - params[1]))
            total -= correct @ np.log(p) + (observed - correct) @ np.log1p(-p)
        return total

    oracle = minimize(
        loss,
        x0=[1.0, 0.0],
        method="L-BFGS-B",
        bounds=[(0.1, 5.0), (-6.0, 6.0)],
        options={"ftol": 1e-15, "gtol": 1e-10},
    )
    # The hook sends the item to the per-parameter path; repeat its sweeps.
    estimator = MultigroupEMEstimator(item_optim_ftol=1e-12, item_optim_maxiter=500)
    for _ in range(60):
        estimator._optimize_item(model, 0, responses, posterior, NODES)
    for group in model.group_models:
        fitted = [
            group.parameters[name][0] for name in ("discrimination", "difficulty")
        ]
        assert loss(fitted) == pytest.approx(oracle.fun, abs=1e-6)


@pytest.mark.parametrize("invariance", ["configural", "metric", "scalar"])
def test_fixed_and_structural_coordinates_stay_bitwise_unchanged(invariance):
    data, groups = _simulate("3PL", n_items=4, n_persons=150, seed=7)
    mg_model, responses, reference = _prepare_multigroup(
        data, groups, "3PL", n_categories=None, reference_group=0, item_names=None
    )
    fixed = {
        "discrimination": {0: 1.37},
        "difficulty": {1: -0.6312},
        "guessing": {2: 0.0731},
    }
    MultigroupEMEstimator(n_quadpts=11, max_iter=6).fit(
        mg_model, responses, invariance, reference, fixed_parameters=fixed
    )
    for group in mg_model.group_models:
        for name, items in fixed.items():
            for item, value in items.items():
                assert group.parameters[name][item] == value
        assert np.any(group.parameters["discrimination"][1:] != 1.0)


def test_one_parameter_slopes_never_move():
    model = MultigroupModel(OneParameterLogistic(3), 2)
    rng = np.random.default_rng(12)
    responses = [rng.integers(0, 2, (50, 3)) for _ in range(2)]
    posterior = [rng.dirichlet(np.ones(NODES.shape[0]), 50) for _ in range(2)]
    estimator = MultigroupEMEstimator()
    for item in range(3):
        estimator._optimize_item(model, item, responses, posterior, NODES)
    for group in model.group_models:
        np.testing.assert_array_equal(group.parameters["discrimination"], 1.0)
        assert np.all(group.parameters["difficulty"] != 0.0)


def test_unadministered_item_keeps_its_group_specific_parameters():
    rng = np.random.default_rng(3)
    responses = [rng.integers(0, 2, (40, 2)), rng.integers(0, 2, (40, 2))]
    responses[1][:, 1] = -1
    posterior = [rng.dirichlet(np.ones(NODES.shape[0]), 40) for _ in range(2)]
    model = MultigroupModel(ThreeParameterLogistic(2), 2)
    model.set_shared_parameter("discrimination")
    before = model.get_group_model(1).parameters
    MultigroupEMEstimator()._optimize_item(model, 1, responses, posterior, NODES)

    first, second = model.group_models
    for name in ("difficulty", "guessing"):
        assert second.parameters[name][1] == before[name][1]
        assert first.parameters[name][1] != before[name][1]
    # The shared slope is estimated from the administering group and kept equal.
    assert (
        second.parameters["discrimination"][1]
        == (first.parameters["discrimination"][1])
    )


def test_custom_and_polytomous_items_use_the_per_parameter_path(monkeypatch):
    calls = []
    original = MultigroupEMEstimator._optimize_parameter_block

    def record(self, models, masks, counts, item_idx, param_name, quad_points):
        calls.append(param_name)
        return original(self, models, masks, counts, item_idx, param_name, quad_points)

    monkeypatch.setattr(MultigroupEMEstimator, "_optimize_parameter_block", record)
    rng = np.random.default_rng(5)
    binary = [rng.integers(0, 2, (30, 1)) for _ in range(2)]
    posterior = [rng.dirichlet(np.ones(NODES.shape[0]), 30) for _ in range(2)]

    builtin = MultigroupModel(TwoParameterLogistic(1), 2)
    MultigroupEMEstimator()._optimize_item(builtin, 0, binary, posterior, NODES)
    assert calls == []

    restricted = TwoParameterLogistic(1)
    restricted.set_free_parameter_masks({"discrimination": np.array([False])})
    MultigroupEMEstimator()._optimize_item(
        MultigroupModel(restricted, 2), 0, binary, posterior, NODES
    )
    assert calls == ["discrimination", "discrimination", "difficulty", "difficulty"]

    calls.clear()
    graded = [rng.integers(0, 3, (30, 1)) for _ in range(2)]
    MultigroupEMEstimator()._optimize_item(
        MultigroupModel(GradedResponseModel(1, 3), 2), 0, graded, posterior, NODES
    )
    assert set(calls) == {"discrimination", "thresholds"}
