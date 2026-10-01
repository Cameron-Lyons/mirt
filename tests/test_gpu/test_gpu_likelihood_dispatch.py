"""Model dispatch, category padding, and real tensor likelihood parity."""

import numpy as np
import pytest
from scipy.special import logsumexp

import mirt._gpu_backend as backend
import mirt.estimation.em as em_module
from mirt.constants import PROB_EPSILON
from mirt.estimation.em import EMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.models.dichotomous import (
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    PartialCreditModel,
)

_KINDS = (
    "1pl",
    "2pl",
    "3pl",
    "grm",
    "grm_mixed",
    "gpcm",
    "gpcm_mixed",
    "pcm",
    "pcm_mixed",
)
_CATEGORY_KERNELS = (
    backend.compute_log_likelihoods_grm_gpu,
    backend.compute_log_likelihoods_gpcm_gpu,
)


def _model(kind, items=7):
    cls = {
        "1pl": OneParameterLogistic,
        "2pl": TwoParameterLogistic,
        "3pl": ThreeParameterLogistic,
        "grm": GradedResponseModel,
        "gpcm": GeneralizedPartialCredit,
        "pcm": PartialCreditModel,
    }[kind.split("_")[0]]
    return (
        cls(
            items,
            n_categories=[2 + j % 4 for j in range(items)] if "mixed" in kind else 5,
        )
        if kind.startswith(("grm", "gpcm", "pcm"))
        else cls(items)
    )


def _data(model, persons=41):
    rng = np.random.default_rng(981)
    counts = model.n_categories if model.is_polytomous else [2] * model.n_items
    data = np.column_stack([rng.integers(0, count, persons) for count in counts])
    data[rng.random(data.shape) < 0.2] = -1
    data[0] = -10
    return data


@pytest.fixture
def cpu_tensor_runtime(monkeypatch):
    if not backend.is_torch_available():
        pytest.skip("PyTorch not installed")
    import torch

    monkeypatch.setattr(
        backend, "_load_torch_runtime", lambda: (torch, torch.device("cpu"))
    )
    monkeypatch.setattr(em_module, "is_gpu_available", lambda: True)
    return torch


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("renamed", [False, True])
def test_dispatch_passes_model_parameters_and_active_categories(
    kind, renamed, monkeypatch
):
    model = _model(kind)
    if renamed:
        model.model_name = "Survey item"
    data = _data(model)
    points = np.linspace(-2, 2, 11)[:, None]
    params = model.parameters
    name = (
        "gpcm"
        if kind.startswith(("gpcm", "pcm"))
        else "grm"
        if kind.startswith("grm")
        else "3pl"
        if kind == "3pl"
        else "2pl"
    )
    calls = 0

    def kernel(responses, grid, discrimination, locations, *extra, **kwargs):
        nonlocal calls
        calls += 1
        np.testing.assert_array_equal(responses, data)
        np.testing.assert_array_equal(grid, points[:, 0])
        np.testing.assert_array_equal(discrimination, params["discrimination"])
        key = (
            "steps"
            if name == "gpcm"
            else "thresholds"
            if name == "grm"
            else "difficulty"
        )
        np.testing.assert_array_equal(locations, params[key])
        if model.is_polytomous:
            assert kwargs == {"n_categories": model.n_categories}
        elif name == "3pl":
            np.testing.assert_array_equal(extra[0], params["guessing"])
        return np.zeros((len(data), len(points)))

    monkeypatch.setattr(em_module, "is_gpu_available", lambda: True)
    monkeypatch.setattr(em_module, f"compute_log_likelihoods_{name}_gpu", kernel)
    actual = EMEstimator(use_gpu=True)._compute_log_likelihoods(model, data, points)
    assert calls == 1
    np.testing.assert_array_equal(actual, np.zeros((len(data), len(points))))


@pytest.mark.parametrize("kind", ["1pl", "pcm_mixed"])
@pytest.mark.parametrize("binding", ["class", "instance"])
def test_new_dispatch_preserves_custom_curves(kind, binding, monkeypatch):
    model = _model(kind)
    original = type(model).probability

    def changed(self, *args):
        values = original(self, *args)
        return values[..., ::-1] if self.is_polytomous else values**1.1

    monkeypatch.setattr(
        type(model), "probability", changed
    ) if binding == "class" else monkeypatch.setattr(
        model, "probability", changed.__get__(model)
    )
    monkeypatch.setattr(em_module, "is_gpu_available", lambda: True)

    def fail(*args, **kwargs):
        raise AssertionError("Custom probability curves must be used")

    monkeypatch.setattr(em_module, "compute_log_likelihoods_2pl_gpu", fail)
    monkeypatch.setattr(em_module, "compute_log_likelihoods_gpcm_gpu", fail)
    data, points = _data(model), np.linspace(-2, 2, 11)[:, None]
    actual = EMEstimator(use_gpu=True)._compute_log_likelihoods(model, data, points)
    np.testing.assert_array_equal(actual, model.log_likelihood_batch(data, points))


@pytest.mark.parametrize("kind", _KINDS)
def test_real_tensor_dispatch_matches_public_likelihood(kind, cpu_tensor_runtime):
    model = _model(kind)
    data, points = _data(model), np.linspace(-3, 3, 11)[:, None]
    original = data.copy()
    actual = EMEstimator(use_gpu=True)._compute_log_likelihoods(model, data, points)
    np.testing.assert_allclose(
        actual, model.log_likelihood_batch(data, points), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("kind", ["grm_mixed", "gpcm_mixed"])
@pytest.mark.parametrize("padding", [0.0, np.nan, np.inf])
def test_real_category_kernels_ignore_inactive_padding(
    kind, padding, cpu_tensor_runtime
):
    model = _model(kind)
    data, points = _data(model), np.linspace(-3, 3, 11)
    params = model.parameters
    key = "thresholds" if kind.startswith("grm") else "steps"
    values = params[key]
    for item, count in enumerate(model.n_categories):
        values[item, count - 1 :] = padding
    kernel = _CATEGORY_KERNELS[0 if kind.startswith("grm") else 1]
    actual = kernel(
        data, points, params["discrimination"], values, n_categories=model.n_categories
    )
    np.testing.assert_allclose(
        actual,
        model.log_likelihood_batch(data, points[:, None]),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("kernel", _CATEGORY_KERNELS)
def test_explicit_uniform_counts_preserve_legacy_calls(kernel, cpu_tensor_runtime):
    data = np.array([[1, -1], [2, 0]])
    points = np.linspace(-3, 3, 11)
    discrimination = np.array([0.8, 1.2])
    thresholds = np.array([[-1.0, 1.0], [-0.5, 0.5]])
    actual = kernel(data, points, discrimination, thresholds, n_categories=[3, 3])
    np.testing.assert_array_equal(
        actual, kernel(data, points, discrimination, thresholds)
    )


@pytest.mark.parametrize("kernel", _CATEGORY_KERNELS)
@pytest.mark.parametrize(
    "counts", [[], [3], [[3, 3]], [True, True], [2.0, 3.0], [1, 3], [4, 3], [-1, 3]]
)
def test_category_metadata_is_validated_before_loading_runtime(
    kernel, counts, monkeypatch
):
    def fail():
        raise AssertionError("Invalid metadata must not load PyTorch")

    monkeypatch.setattr(backend, "_load_torch_runtime", fail)
    with pytest.raises(ValueError, match="n_categories must contain"):
        kernel(
            np.array([[1, 2]]),
            np.array([0.0]),
            np.ones(2),
            np.array([[-1.0, 1.0]] * 2),
            n_categories=counts,
        )


@pytest.mark.parametrize("kernel", _CATEGORY_KERNELS)
def test_categories_are_checked_per_item_before_loading_runtime(kernel, monkeypatch):
    monkeypatch.setattr(
        backend, "_load_torch_runtime", lambda: pytest.fail("Loaded PyTorch")
    )
    with pytest.raises(ValueError, match="item 0.*between 0 and 1"):
        kernel(
            np.array([[2, 3]]),
            np.array([0.0]),
            np.ones(2),
            np.array([[-1.0, 0.0, 1.0]] * 2),
            n_categories=[2, 4],
        )


@pytest.mark.parametrize("kernel", _CATEGORY_KERNELS)
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_active_thresholds_remain_finite(kernel, value, monkeypatch):
    monkeypatch.setattr(
        backend, "_load_torch_runtime", lambda: pytest.fail("Loaded PyTorch")
    )
    with pytest.raises(ValueError, match="must be finite"):
        kernel(
            np.array([[1, 2]]),
            np.array([0.0]),
            np.ones(2),
            np.array([[value, 0.0], [-1.0, 1.0]]),
            n_categories=[2, 3],
        )


@pytest.mark.parametrize("kind", ["2pl", "3pl", "mirt"])
@pytest.mark.parametrize("general_responses", [False, True])
def test_binary_tensor_reduction_matches_public_curves(
    kind, general_responses, cpu_tensor_runtime
):
    model = TwoParameterLogistic(7, n_factors=3) if kind == "mirt" else _model(kind)
    rng = np.random.default_rng(983)
    values = model.parameters
    values["discrimination"] = rng.uniform(0.4, 2.0, values["discrimination"].shape)
    values["difficulty"] = np.linspace(-2, 2, model.n_items)
    if kind == "3pl":
        values["guessing"] = np.linspace(0.05, 0.3, model.n_items)
    model.set_parameters(**values)
    data = _data(model).astype(np.float64)
    if general_responses:
        data[1] = [-2, 0.25, 0.5, 1.5, 2, 0, 1]
    points = np.random.default_rng(982).normal(size=(11, model.n_factors)) * 5
    points[0] = -1000
    points[-1] = 1000
    probabilities = np.clip(model.probability(points), PROB_EPSILON, 1 - PROB_EPSILON)
    terms = data[:, None, :] * np.log(probabilities)[None, :, :]
    terms += (1 - data[:, None, :]) * np.log1p(-probabilities)[None, :, :]
    expected = np.where((data >= 0)[:, None, :], terms, 0.0).sum(axis=2)
    params = model.parameters
    args = [
        data,
        points if kind == "mirt" else points[:, 0],
        params["discrimination"],
        params["difficulty"],
    ]
    if kind == "3pl":
        args.append(params["guessing"])
    actual = getattr(backend, f"compute_log_likelihoods_{kind}_gpu")(*args)
    # Near-unit probabilities amplify the existing tensor/NumPy sigmoid
    # rounding difference, including in the former broadcast reduction.
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)


def test_matrix_reduction_preserves_extreme_curves_and_response_buffers(
    cpu_tensor_runtime,
):
    torch = cpu_tensor_runtime
    rng = np.random.default_rng(984)
    data = rng.choice([-10.0, -1.0, 0.0, 0.25, 1.0, 1.5, 2.0], size=(41, 7))
    data[0] = -10
    probabilities = rng.uniform(0.001, 0.999, (11, 7))
    probabilities[0] = PROB_EPSILON
    probabilities[-1] = 1 - PROB_EPSILON
    original = data.copy()
    terms = data[:, None, :] * np.log(probabilities)[None, :, :]
    terms += (1 - data[:, None, :]) * np.log1p(-probabilities)[None, :, :]
    expected = np.where((data >= 0)[:, None, :], terms, 0.0).sum(axis=2)
    actual = backend._aggregate_dichotomous_log_likelihoods(
        torch, torch.as_tensor(data), torch.as_tensor(probabilities)
    ).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("mean,variance", [(0.0, 1.0), (0.4, 1.7)])
def test_complete_tensor_e_step_preserves_prior_and_missing_rows(
    mean, variance, cpu_tensor_runtime
):
    model = _model("2pl")
    data = _data(model)
    grid = GaussHermiteQuadrature(15)
    points = grid.nodes[:, 0]
    log_mass = np.log(grid.weights) - 0.5 * (
        np.log(variance) + (points - mean) ** 2 / variance - points**2
    )
    log_mass -= logsumexp(log_mass)
    joint = model.log_likelihood_batch(data, grid.nodes) + log_mass
    normalizer = logsumexp(joint, axis=1)
    params = model.parameters
    posterior, marginal = backend.e_step_complete_gpu(
        data,
        points,
        grid.weights,
        params["discrimination"],
        params["difficulty"],
        mean,
        variance,
    )
    np.testing.assert_allclose(
        posterior, np.exp(joint - normalizer[:, None]), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(marginal, np.exp(normalizer), rtol=1e-12, atol=1e-12)
    assert marginal[0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kind", ["1pl", "2pl", "3pl", "grm_mixed", "gpcm_mixed", "pcm_mixed"]
)
def test_seeded_tensor_fits_match_cpu_fits(kind, cpu_tensor_runtime):
    model = _model(kind)
    data = _data(model, 150)
    options = dict(
        n_quadpts=11, max_iter=2, use_rust=False, compute_standard_errors=False
    )
    cpu = EMEstimator(**options, use_gpu=False)
    tensor = EMEstimator(**options, use_gpu=True)
    expected, actual = cpu.fit(model.copy(), data), tensor.fit(model.copy(), data)
    np.testing.assert_allclose(
        tensor.convergence_history, cpu.convergence_history, rtol=1e-10, atol=1e-8
    )
    for name in model.parameters:
        np.testing.assert_allclose(
            actual.model.parameters[name],
            expected.model.parameters[name],
            rtol=1e-6,
            atol=1e-7,
        )
