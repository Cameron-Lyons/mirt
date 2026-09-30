"""Prepared and native fitting respects changed model hooks."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import mirt.backends.rust.polytomous_mstep as native_polytomous_module
import mirt.estimation.bl as bl_module
import mirt.estimation.em as em_module
import mirt.estimation.mcem as mc_module
from mirt._model_defaults import (
    original_model_hook,
    uses_builtin_model_hooks,
    uses_original_model_hook,
)
from mirt.backends.rust.polytomous_mstep import try_polytomous_m_step
from mirt.constants import PROB_EPSILON
from mirt.estimation._affine_objective import prepare_affine_objective
from mirt.estimation._bl_objective import prepare_bl_objective
from mirt.estimation._dichotomous_objective import prepare_dichotomous_objective
from mirt.estimation._em_context import EMFitContext
from mirt.estimation._item_information import item_standard_errors
from mirt.estimation._patterns import supports_pattern_compression
from mirt.estimation._polytomous_objective import prepare_polytomous_objective
from mirt.estimation.bl import BLEstimator
from mirt.estimation.em import EMEstimator
from mirt.estimation.latent_density import GaussianDensity
from mirt.estimation.mcem import MCEMEstimator, QMCEMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.models.base import BaseItemModel
from mirt.models.bifactor import BifactorModel
from mirt.models.dichotomous import (
    FiveParameterLogistic,
    FourParameterLogistic,
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.multidimensional import MultidimensionalModel
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
)

_KINDS = ("1pl", "2pl", "3pl", "4pl", "grm", "gpcm", "pcm", "nrm", "mirt", "bifactor")


def _model(kind):
    if kind == "mirt":
        return MultidimensionalModel(2, 2)
    if kind == "bifactor":
        return BifactorModel(2, [0, 1])
    cls = {
        "1pl": OneParameterLogistic,
        "2pl": TwoParameterLogistic,
        "3pl": ThreeParameterLogistic,
        "4pl": FourParameterLogistic,
        "grm": GradedResponseModel,
        "gpcm": GeneralizedPartialCredit,
        "pcm": PartialCreditModel,
        "nrm": NominalResponseModel,
    }[kind]
    return (
        cls(2, n_categories=[3, 4]) if kind in ("grm", "gpcm", "pcm", "nrm") else cls(2)
    )


def _problem(model):
    rng = np.random.default_rng(988)
    categories = model.n_categories if model.is_polytomous else [2] * model.n_items
    data = np.column_stack([rng.integers(-1, k, 11) for k in categories])
    quadrature = GaussHermiteQuadrature(5, model.n_factors)
    posterior = rng.uniform(size=(len(data), len(quadrature.nodes)))
    posterior /= posterior.sum(axis=1, keepdims=True)
    return data, quadrature, posterior


def _prepare_item(model, points):
    if model.is_polytomous:
        return prepare_polytomous_objective(
            model,
            0,
            points,
            np.ones((len(points), model.n_categories[0])),
            PROB_EPSILON,
        )
    _, bounds = EMEstimator()._get_item_params_and_bounds(model, 0)
    correct = np.ones(len(points))
    objective = prepare_dichotomous_objective(
        model, 0, points, correct * 2, correct, PROB_EPSILON, bounds
    )
    return objective or prepare_affine_objective(
        model, 0, points, correct * 2, correct, PROB_EPSILON, bounds
    )


def _changed_curve(model, monkeypatch, binding="class"):
    original = type(model).probability

    def curve(self, theta, item_idx=None):
        probability = original(self, theta, item_idx)
        return probability[..., ::-1] if self.is_polytomous else probability**1.2

    if binding == "instance":
        monkeypatch.setattr(model, "probability", curve.__get__(model))
    elif binding == "subclass":
        model.__class__ = type("ChangedCurve", (type(model),), {"probability": curve})
    else:
        monkeypatch.setattr(type(model), "probability", curve)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize(
    "hook", ["probability", "_ensure_theta_2d", "set_item_parameter"]
)
def test_class_hook_changes_disable_prepared_item_kernels(kind, hook, monkeypatch):
    model = _model(kind)
    _, quadrature, _ = _problem(model)
    assert _prepare_item(model, quadrature.nodes) is not None
    original = getattr(type(model), hook)

    def changed(self, *args, **kwargs):
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(model), hook, changed)
    assert _prepare_item(model, quadrature.nodes) is None


@pytest.mark.parametrize("kind", _KINDS[:4])
def test_custom_parameter_domains_disable_pure_item_kernels(kind, monkeypatch):
    model = _model(kind)
    _, quadrature, _ = _problem(model)
    original = type(model)._validate_parameter_state
    monkeypatch.setattr(
        type(model),
        "_validate_parameter_state",
        lambda self, values: original(self, values),
    )
    assert _prepare_item(model, quadrature.nodes) is None


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("hook", ["parameters", "free_parameter_masks"])
def test_changed_parent_properties_disable_prepared_kernels(kind, hook, monkeypatch):
    model = _model(kind)
    _, quadrature, _ = _problem(model)
    original = getattr(BaseItemModel, hook)
    monkeypatch.setattr(BaseItemModel, hook, property(lambda self: original.fget(self)))
    assert not uses_builtin_model_hooks(model)
    assert _prepare_item(model, quadrature.nodes) is None


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("hook", ["log_likelihood", "log_likelihood_batch"])
def test_class_likelihood_changes_disable_joint_kernels_and_compression(
    kind, hook, monkeypatch
):
    model = _model(kind)
    data, quadrature, _ = _problem(model)
    original = getattr(type(model), hook)
    monkeypatch.setattr(
        type(model), hook, lambda self, *args: original(self, *args) * 1.2
    )
    estimator = BLEstimator(n_quadpts=5)
    _, bounds, structure = estimator._flatten_parameters(model)
    with EMFitContext(data) as context:
        assert (
            prepare_bl_objective(
                model,
                context,
                quadrature.nodes,
                np.log(quadrature.weights),
                structure,
                bounds,
                estimator._unflatten_parameters,
            )
            is None
        )
    assert not supports_pattern_compression(model)
    # Conditional item objectives do not call a joint likelihood hook.
    assert _prepare_item(model, quadrature.nodes) is not None


@pytest.mark.parametrize("kind", _KINDS[:7])
def test_class_curve_changes_disable_exact_em_information(kind, monkeypatch):
    model = _model(kind)
    data, quadrature, posterior = _problem(model)
    assert (
        item_standard_errors(model, data, posterior, quadrature.nodes, PROB_EPSILON)
        is not None
    )
    _changed_curve(model, monkeypatch)
    assert (
        item_standard_errors(model, data, posterior, quadrature.nodes, PROB_EPSILON)
        is None
    )


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("method", ["em", "mc", "qmc"])
def test_class_curve_changes_use_public_weighted_item_objectives(
    kind, method, monkeypatch
):
    model = _model(kind)
    data, quadrature, posterior = _problem(model)
    original = model.parameters
    _changed_curve(model, monkeypatch)
    count = 50
    samples = np.broadcast_to(
        np.random.default_rng(987).normal(size=(count, model.n_factors)),
        (len(data), count, model.n_factors),
    )
    weights = np.full((len(data), count), 1 / count)
    estimator = (
        EMEstimator(use_rust=False, use_gpu=False)
        if method == "em"
        else (
            MCEMEstimator(n_samples=count)
            if method == "mc"
            else QMCEMEstimator(n_samples=count)
        )
    )
    estimator._quadrature = quadrature
    calls = 0

    def optimize(objective, x0, *, jac, **kwargs):
        nonlocal calls
        calls += 1
        assert not jac
        bounds = np.asarray(kwargs["bounds"])
        trial = np.clip(x0 + 0.02, bounds[:, 0], bounds[:, 1])
        loss = objective(trial)
        item = calls - 1
        valid = data[:, item] >= 0
        points = quadrature.nodes if method == "em" else samples[0]
        probability = np.clip(
            model.probability(points, item),
            PROB_EPSILON,
            1 - PROB_EPSILON if method == "em" or not model.is_polytomous else 1,
        )
        if model.is_polytomous:
            logs = np.log(probability[:, data[valid, item]].T)
        else:
            decisions = data[valid, item, None]
            logs = decisions * np.log(probability) + (1 - decisions) * np.log1p(
                -probability
            )
        mass = posterior[valid] if method == "em" else weights[valid]
        np.testing.assert_allclose(loss, -np.sum(mass * logs), rtol=1e-12)
        return SimpleNamespace(x=x0, fun=loss, success=True)

    monkeypatch.setattr(
        em_module if method == "em" else mc_module, "minimize", optimize
    )
    if method == "em":
        estimator._m_step(model, data, posterior)
    else:
        estimator._m_step_mc(model, data, samples, weights)
    assert calls == model.n_items
    for name in original:
        np.testing.assert_array_equal(model.parameters[name], original[name])


@pytest.mark.parametrize("kind", _KINDS)
def test_joint_fitting_uses_the_modified_public_likelihood(kind, monkeypatch):
    model = _model(kind)
    data, _, _ = _problem(model)
    _changed_curve(model, monkeypatch)
    calls = 0
    estimator = BLEstimator(n_quadpts=5, max_iter=2)

    def optimize(objective, x0, *, jac, bounds, **kwargs):
        nonlocal calls
        calls += 1
        assert not jac
        limits = np.asarray(bounds)
        trial = np.clip(x0 + 0.02, limits[:, 0], limits[:, 1])
        value = objective(trial)
        expected = -estimator._compute_marginal_log_likelihood(model, data)
        np.testing.assert_allclose(value, expected, rtol=1e-12)
        return SimpleNamespace(x=trial, fun=value, success=True, nit=1)

    monkeypatch.setattr(bl_module, "minimize", optimize)
    monkeypatch.setattr(
        estimator,
        "_compute_standard_errors",
        lambda model, *args: {
            name: np.zeros_like(value) for name, value in model.parameters.items()
        },
    )
    result = estimator.fit(model, data)
    assert calls == 1
    np.testing.assert_allclose(
        result.log_likelihood,
        estimator._compute_marginal_log_likelihood(model, data),
        rtol=1e-12,
    )


@pytest.mark.parametrize("shared", [False, True])
def test_bifactor_mc_gradient_curvature_matches_public_numerical_reference(shared):
    model = _model("bifactor")
    data, _, _ = _problem(model)
    samples = np.random.default_rng(982).normal(size=(len(data), 50, model.n_factors))
    if shared:
        samples = np.broadcast_to(samples[:1], samples.shape)
    weights = np.full((len(data), 50), 0.02)
    original = model.parameters
    analytic = MCEMEstimator(n_samples=50, compute_standard_errors=True)
    numerical = MCEMEstimator(
        n_samples=50, compute_standard_errors=True, se_step_size=1e-3
    )
    numerical._item_expected_log_likelihood = numerical._item_expected_log_likelihood
    actual = analytic._compute_standard_errors_mc(model, data, samples, weights)
    expected = numerical._compute_standard_errors_mc(model, data, samples, weights)
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name], rtol=2e-6, atol=1e-7)
        np.testing.assert_array_equal(model.parameters[name], original[name])


@pytest.mark.parametrize("kind", ["grm", "gpcm", "pcm", "nrm"])
@pytest.mark.parametrize("binding", ["instance", "class", "subclass"])
def test_polytomous_likelihoods_respect_public_curves(kind, binding, monkeypatch):
    model = _model(kind)
    data, quadrature, _ = _problem(model)
    _changed_curve(model, monkeypatch, binding)
    expected = np.zeros((len(data), len(quadrature.nodes)))
    for item in range(model.n_items):
        observed = data[:, item] >= 0
        probability = np.clip(
            model.probability(quadrature.nodes, item), PROB_EPSILON, 1 - PROB_EPSILON
        )
        expected[observed] += np.log(probability[:, data[observed, item]].T)
    np.testing.assert_allclose(
        model.log_likelihood_batch(data, quadrature.nodes), expected, rtol=1e-12
    )
    repeats = max(1, (50 + len(quadrature.nodes) - 1) // len(quadrature.nodes))
    grid = np.tile(quadrature.nodes, (repeats, 1))
    points = np.broadcast_to(grid, (len(data), *grid.shape))
    actual = MCEMEstimator(n_samples=len(grid))._sample_log_likelihoods(
        model, data, points
    )
    np.testing.assert_allclose(actual, np.tile(expected, (1, repeats)), rtol=1e-12)
    for index, points in enumerate(quadrature.nodes):
        np.testing.assert_allclose(
            model.log_likelihood(data, points[None]), expected[:, index], rtol=1e-12
        )


@pytest.mark.parametrize("kind", ["grm", "gpcm", "pcm", "nrm"])
@pytest.mark.parametrize("binding", ["instance", "class", "subclass"])
def test_polytomous_likelihoods_pass_original_points_to_custom_validation(
    kind, binding, monkeypatch
):
    model = _model(kind)
    data, quadrature, _ = _problem(model)
    original = type(model)._ensure_theta_2d

    def changed(self, points):
        return original(self, points) * 1.15 + 0.1

    if binding == "instance":
        monkeypatch.setattr(model, "_ensure_theta_2d", changed.__get__(model))
    elif binding == "subclass":
        model.__class__ = type(
            "ChangedValidation", (type(model),), {"_ensure_theta_2d": changed}
        )
    else:
        monkeypatch.setattr(type(model), "_ensure_theta_2d", changed)
    expected = np.zeros((len(data), len(quadrature.nodes)))
    for item in range(model.n_items):
        valid = data[:, item] >= 0
        probability = np.clip(
            model.probability(quadrature.nodes, item), PROB_EPSILON, 1 - PROB_EPSILON
        )
        expected[valid] += np.log(probability[:, data[valid, item]].T)
    np.testing.assert_allclose(
        model.log_likelihood_batch(data, quadrature.nodes), expected, rtol=1e-12
    )
    for index, point in enumerate(quadrature.nodes):
        np.testing.assert_allclose(
            model.log_likelihood(data, point[None]), expected[:, index], rtol=1e-12
        )


@pytest.mark.parametrize("kind", ["grm", "gpcm", "pcm"])
def test_native_polytomous_updates_decline_class_curve_changes(kind, monkeypatch):
    model = _model(kind)
    data, quadrature, posterior = _problem(model)
    _changed_curve(model, monkeypatch)
    monkeypatch.setattr(native_polytomous_module, "rust_enabled", lambda: True)
    assert not try_polytomous_m_step(
        model,
        data,
        posterior,
        quadrature.nodes,
        max_iter=5,
        ftol=1e-6,
        epsilon=PROB_EPSILON,
        n_jobs=1,
    )


@pytest.mark.parametrize(
    "hook", ["probability", "log_likelihood_batch", "_ensure_theta_2d"]
)
def test_native_3pl_iteration_declines_class_hooks(hook, monkeypatch):
    model = _model("3pl")
    data, _, _ = _problem(model)
    estimator = EMEstimator(n_quadpts=5, use_gpu=False)
    estimator._latent_density = GaussianDensity()
    monkeypatch.setattr(em_module, "RUST_AVAILABLE", True)
    monkeypatch.setattr(em_module, "should_use_rust", lambda use: True)
    assert estimator._can_use_rust_3pl(model, data)
    original = getattr(type(model), hook)
    monkeypatch.setattr(type(model), hook, lambda self, *args: original(self, *args))
    assert not estimator._can_use_rust_3pl(model, data)


@pytest.mark.parametrize("kind", ["2pl", "3pl", "grm", "gpcm"])
def test_gpu_dispatch_declines_class_curve_changes(kind, monkeypatch):
    model = _model(kind)
    data, quadrature, _ = _problem(model)
    _changed_curve(model, monkeypatch)
    monkeypatch.setattr(EMEstimator, "_should_use_gpu", property(lambda self: True))

    def fail(*args):
        raise AssertionError("Changed model must retain its public likelihood")

    monkeypatch.setattr(EMEstimator, "_compute_log_likelihoods_gpu", fail)
    actual = EMEstimator()._compute_log_likelihoods(model, data, quadrature.nodes)
    np.testing.assert_array_equal(
        actual, model.log_likelihood_batch(data, quadrature.nodes)
    )


def test_model_hooks_are_recorded_before_estimator_imports():
    script = """
import sys
import numpy as np
from mirt.models.dichotomous import TwoParameterLogistic
assert 'mirt.estimation' not in sys.modules
original = TwoParameterLogistic.probability
TwoParameterLogistic.probability = lambda self, *a: original(self, *a) ** 1.2
from mirt.estimation._dichotomous_objective import prepare_dichotomous_objective
from mirt.estimation.mcem import MCEMEstimator
model = TwoParameterLogistic(1)
assert not MCEMEstimator._uses_default_information_model(model)
assert prepare_dichotomous_objective(model, 0, np.zeros((2, 1)), np.ones(2), np.ones(2), 1e-10) is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "base", ["BaseItemModel", "DichotomousItemModel", "PolytomousItemModel"]
)
def test_base_hooks_are_recorded_before_concrete_model_imports(base):
    script = """
import sys
from mirt.models import base
assert 'mirt.models.dichotomous' not in sys.modules
assert 'mirt.models.polytomous' not in sys.modules
name = sys.argv[1]
cls = getattr(base, name)
hook = '_ensure_theta_2d' if name == 'BaseItemModel' else 'log_likelihood_batch'
original = getattr(cls, hook)
setattr(cls, hook, lambda self, *a: original(self, *a))
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel
from mirt._model_defaults import uses_builtin_model_hooks
model = GradedResponseModel(1, 3) if name == 'PolytomousItemModel' else TwoParameterLogistic(1)
assert not uses_builtin_model_hooks(model, likelihood=True)
from mirt.estimation.mcem import QMCEMEstimator
assert not QMCEMEstimator()._uses_default_sampling_methods(model)
if name == 'BaseItemModel':
    from mirt.estimation._mc_likelihood import uses_default_sample_likelihood
    assert not uses_default_sample_likelihood(model)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, base], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("poly", [False, True])
def test_original_likelihoods_precede_estimator_imports(poly):
    script = """
import sys
from mirt.models.base import DichotomousItemModel, PolytomousItemModel
cls = PolytomousItemModel if sys.argv[1] == 'True' else DichotomousItemModel
original = cls.log_likelihood
cls.log_likelihood = lambda self, *a: original(self, *a) * 1.2
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel
from mirt.estimation._mc_likelihood import uses_default_sample_likelihood
model = GradedResponseModel(1, 3) if sys.argv[1] == 'True' else TwoParameterLogistic(1)
assert not uses_default_sample_likelihood(model)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(poly)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_unknown_original_model_hook_is_rejected():
    with pytest.raises(KeyError, match="No original model hook"):
        original_model_hook(TwoParameterLogistic, "custom_method")
    assert not uses_original_model_hook(object(), "custom_method")


def test_original_likelihood_hooks_do_not_require_prepared_item_support():
    model = FiveParameterLogistic(2)
    assert not uses_builtin_model_hooks(model)
    assert uses_original_model_hook(model, "log_likelihood_batch")
    assert QMCEMEstimator()._uses_default_sampling_methods(model)
