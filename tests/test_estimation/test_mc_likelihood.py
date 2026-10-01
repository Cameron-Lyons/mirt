"""Bounded Monte Carlo likelihoods and preservation of model callbacks."""

import tracemalloc
from types import MethodType

import numpy as np
import pytest

from mirt.estimation import _mc_likelihood as likelihood_module
from mirt.estimation.mcem import MCEMEstimator, QMCEMEstimator, StochasticEMEstimator
from mirt.models.base import BaseItemModel, DichotomousItemModel, PolytomousItemModel
from mirt.models.bifactor import BifactorModel
from mirt.models.dichotomous import (
    ComplementaryLogLog,
    FiveParameterLogistic,
    FourParameterLogistic,
    NegativeLogLog,
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
    UnipolarLogLogistic,
)
from mirt.models.multidimensional import MultidimensionalModel
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
)


def _model(kind):
    if kind == "2pl_3d":
        return TwoParameterLogistic(3, n_factors=3)
    if kind == "mirt":
        return MultidimensionalModel(3, n_factors=3)
    if kind == "bifactor":
        return BifactorModel(3, [2, 5, 2])
    if kind in ("grm", "gpcm", "pcm", "nrm"):
        cls = {
            "grm": GradedResponseModel,
            "gpcm": GeneralizedPartialCredit,
            "pcm": PartialCreditModel,
            "nrm": NominalResponseModel,
        }[kind]
        return cls(3, n_categories=[2, 4, 3], n_factors=1 if kind == "pcm" else 2)
    return {
        "1pl": OneParameterLogistic,
        "2pl": TwoParameterLogistic,
        "3pl": ThreeParameterLogistic,
        "4pl": FourParameterLogistic,
        "5pl": FiveParameterLogistic,
        "ull": UnipolarLogLogistic,
        "cll": ComplementaryLogLog,
        "nll": NegativeLogLog,
    }[kind](3)


def _problem(model, n_samples=50):
    rng = np.random.default_rng(831)
    responses = np.column_stack(
        [
            rng.integers(0, model.n_categories[j] if model.is_polytomous else 2, 17)
            for j in range(model.n_items)
        ]
    )
    responses[::4, 0] = -1
    responses[0] = -1
    responses[:, 2] = -1
    samples = rng.normal(size=(17, n_samples, model.n_factors))
    return responses, samples


def _expanded(self, model, responses, samples):
    values = model.log_likelihood(
        np.repeat(responses, samples.shape[1], axis=0),
        np.asarray(samples, dtype=np.float64).reshape(-1, model.n_factors),
    )
    return values.reshape(samples.shape[:2])


@pytest.mark.parametrize(
    "kind",
    [
        "1pl",
        "2pl",
        "3pl",
        "4pl",
        "5pl",
        "ull",
        "cll",
        "nll",
        "2pl_3d",
        "mirt",
        "bifactor",
        "grm",
        "gpcm",
        "pcm",
        "nrm",
    ],
)
@pytest.mark.parametrize("blocked", [False, True])
def test_sampled_likelihood_matches_public_expansion(kind, blocked, monkeypatch):
    model = _model(kind)
    responses, samples = _problem(model)
    expected = _expanded(None, model, responses, samples)
    responses.setflags(write=False)
    samples.setflags(write=False)
    if blocked:
        monkeypatch.setattr(likelihood_module, "_MAX_MC_LIKELIHOOD_ELEMENTS", 128)
    actual = MCEMEstimator(n_samples=50)._sample_log_likelihoods(
        model, responses, samples
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(actual[0], np.zeros(50))
    assert actual.dtype == np.float64 and actual.flags.writeable
    assert not np.shares_memory(actual, samples)


@pytest.mark.parametrize("kind", ["2pl_3d", "gpcm"])
@pytest.mark.parametrize("layout", ["float32", "strided", "reversed", "shared", "list"])
def test_sampled_likelihood_handles_borrowed_sample_layouts(kind, layout, monkeypatch):
    model = _model(kind)
    responses, samples = _problem(model)
    if layout == "float32":
        samples = samples.astype(np.float32)
    elif layout == "strided":
        storage = np.zeros((17, 100, model.n_factors))
        storage[:, ::2] = samples
        samples = storage[:, ::2]
    elif layout == "reversed":
        samples = samples[:, ::-1, ::-1]
    elif layout == "shared":
        samples = np.broadcast_to(samples[0], samples.shape)
    original = samples.copy()
    expected = _expanded(None, model, responses, samples)
    samples.setflags(write=False)
    if layout == "list":
        samples = samples.tolist()
    monkeypatch.setattr(likelihood_module, "_MAX_MC_LIKELIHOOD_ELEMENTS", 128)
    actual = MCEMEstimator(n_samples=50)._sample_log_likelihoods(
        model, responses, samples
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(samples, original)


@pytest.mark.parametrize("kind", ["2pl_3d", "grm"])
def test_probability_callbacks_receive_only_bounded_points(kind, monkeypatch):
    model = _model(kind)
    responses, samples = _problem(model)
    expected = _expanded(None, model, responses, samples)
    method = "_category_probabilities" if model.is_polytomous else "probability"
    probability = getattr(model, method)
    sizes, cached, originals = [], [], []
    monkeypatch.setattr(likelihood_module, "_MAX_MC_LIKELIHOOD_ELEMENTS", 128)

    def curve(points, *args):
        sizes.append(len(points))
        values = probability(points, *args)
        originals.append(values.copy())
        values.setflags(write=False)
        cached.append(values)
        return values

    monkeypatch.setattr(model, method, curve)
    actual = MCEMEstimator(n_samples=50)._sample_log_likelihoods(
        model, responses, samples
    )
    assert len(sizes) > 1
    width = max(
        model.n_items,
        model.n_factors,
        max(model.n_categories) if model.is_polytomous else 2,
    )
    assert max(sizes) * width <= 128
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    for values, original in zip(cached, originals, strict=True):
        np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize("override", ["instance", "class", "subclass"])
@pytest.mark.parametrize(
    ("kind", "method"),
    [
        ("2pl", "log_likelihood"),
        ("grm", "log_likelihood"),
        ("2pl", "_ensure_theta_2d"),
        ("2pl", "_validate_dichotomous_responses"),
        ("grm", "_ensure_theta_2d"),
        ("grm", "_validate_polytomous_responses"),
    ],
)
def test_custom_likelihood_and_validation_callbacks_keep_model_path(
    kind, method, override, monkeypatch
):
    model = _model(kind)
    owner = (
        BaseItemModel
        if method == "_ensure_theta_2d"
        else PolytomousItemModel
        if model.is_polytomous
        else DichotomousItemModel
    )
    original = getattr(owner, method)
    calls = []

    def custom(self, *args):
        calls.append(args[0].shape)
        value = original(self, *args)
        return value - 0.125 if method == "log_likelihood" else value

    if override == "instance":
        setattr(model, method, MethodType(custom, model))
    elif override == "class":
        monkeypatch.setattr(owner, method, custom)
    else:
        model.__class__ = type("CustomModel", (type(model),), {method: custom})
    responses, samples = _problem(model)
    expected = _expanded(None, model, responses, samples)
    calls.clear()

    def unexpected(*args):
        pytest.fail("custom model likelihood/validation must keep its evaluation path")

    monkeypatch.setattr(likelihood_module, "_binary_log_likelihoods", unexpected)
    monkeypatch.setattr(likelihood_module, "_category_log_likelihoods", unexpected)
    actual = MCEMEstimator(n_samples=50)._sample_log_likelihoods(
        model, responses, samples
    )
    np.testing.assert_array_equal(actual, expected)
    assert calls and all(shape[0] == 17 * 50 for shape in calls)


@pytest.mark.parametrize("kind", ["2pl", "grm"])
@pytest.mark.parametrize("dtype", [np.int64, np.float64, np.float32])
def test_broadcast_custom_probabilities_preserve_clipping_and_missing_values(
    kind, dtype, monkeypatch
):
    model = _model(kind)
    responses, samples = _problem(model)
    cached = (
        np.array([[0, 1, 0]], dtype=dtype)
        if not model.is_polytomous
        else np.array([[0, 0, 0, 1]], dtype=dtype)
    )
    if dtype == np.float32:
        cached = np.array(
            [[0.2, 0.8, 0.4]] if not model.is_polytomous else [[0.2, 0.1, 0.5, 0.2]],
            dtype=dtype,
        )
    original = cached.copy()
    cached.setflags(write=False)
    method = "_category_probabilities" if model.is_polytomous else "probability"
    monkeypatch.setattr(model, method, lambda *args: cached)
    expected = _expanded(None, model, responses, samples)
    actual = MCEMEstimator(n_samples=50)._sample_log_likelihoods(
        model, responses, samples
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(cached, original)


@pytest.mark.parametrize("kind", ["2pl", "grm"])
def test_missing_items_suppress_undefined_probability_callbacks(kind, monkeypatch):
    model = _model(kind)
    responses, samples = _problem(model)
    method = "_category_probabilities" if model.is_polytomous else "probability"
    probability = getattr(model, method)

    def curve(points, *args):
        values = probability(points, *args)
        if not model.is_polytomous:
            values[:, 2] = np.nan
        elif args[0] == 2:
            values[:] = np.nan
        return values

    monkeypatch.setattr(model, method, curve)
    expected = _expanded(None, model, responses, samples)
    actual = MCEMEstimator(n_samples=50)._sample_log_likelihoods(
        model, responses, samples
    )
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_general_binary_response_values_keep_public_reduction(dtype):
    model = _model("2pl")
    responses, samples = _problem(model)
    responses = responses.astype(dtype)
    responses[2, 0] = 2
    expected = _expanded(None, model, responses, samples)
    actual = MCEMEstimator(n_samples=50)._sample_log_likelihoods(
        model, responses, samples
    )
    np.testing.assert_array_equal(actual, expected)


def test_general_binary_response_values_do_not_overflow_separate_reductions(
    monkeypatch,
):
    model = _model("2pl")
    responses, samples = _problem(model)
    responses = np.full(responses.shape, 1e308)
    monkeypatch.setattr(model, "probability", lambda *args: np.full((1, 3), 0.5))
    expected = _expanded(None, model, responses, samples)
    actual = MCEMEstimator(n_samples=50)._sample_log_likelihoods(
        model, responses, samples
    )
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind", ["2pl_3d", "grm"])
def test_all_samples_are_validated_before_probability_callbacks(kind, monkeypatch):
    model = _model(kind)
    responses, samples = _problem(model)
    samples[-1, -1, -1] = np.inf
    monkeypatch.setattr(likelihood_module, "_MAX_MC_LIKELIHOOD_ELEMENTS", 128)

    def unexpected(*args):
        pytest.fail("invalid draws must be rejected before evaluating probabilities")

    monkeypatch.setattr(model, "probability", unexpected)
    monkeypatch.setattr(model, "_category_probabilities", unexpected, raising=False)
    with pytest.raises(ValueError, match="theta_samples must contain only finite"):
        MCEMEstimator(n_samples=50)._sample_log_likelihoods(model, responses, samples)


@pytest.mark.parametrize("code", [1.5, 4.0, np.inf])
def test_category_response_validation_matches_public_likelihood(code):
    model = _model("grm")
    responses, samples = _problem(model)
    responses = responses.astype(np.float64)
    responses[1, 1] = code
    with pytest.raises(ValueError) as reference:
        _expanded(None, model, responses, samples)
    with pytest.raises(type(reference.value), match=str(reference.value)):
        MCEMEstimator(n_samples=50)._sample_log_likelihoods(model, responses, samples)


@pytest.mark.parametrize("kind", ["2pl", "grm"])
def test_observed_undefined_probabilities_are_rejected(kind, monkeypatch):
    model = _model(kind)
    responses, samples = _problem(model)
    method = "_category_probabilities" if model.is_polytomous else "probability"
    probability = getattr(model, method)

    def invalid(points, *args):
        values = probability(points, *args)
        values[:] = np.nan
        return values

    monkeypatch.setattr(model, method, invalid)
    with pytest.raises(ValueError, match="invalid sampled values"):
        MCEMEstimator(n_samples=50)._sample_log_likelihoods(model, responses, samples)


@pytest.mark.parametrize("kind", ["binary", "category"])
def test_peak_excludes_full_sample_conversion_and_response_expansion(kind):
    model = (
        TwoParameterLogistic(12, n_factors=20)
        if kind == "binary"
        else GradedResponseModel(12, n_categories=4, n_factors=20)
    )
    rng = np.random.default_rng(835)
    responses = rng.integers(-1, 2 if kind == "binary" else 4, (2000, 12))
    samples = rng.normal(size=(2000, 256, 20)).astype(np.float32)
    responses.setflags(write=False)
    samples.setflags(write=False)
    tracemalloc.start()
    try:
        actual = MCEMEstimator(n_samples=256)._sample_log_likelihoods(
            model, responses, samples
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert np.isfinite(actual).all() and actual.shape == (2000, 256)
    assert peak < actual.nbytes + 6 * likelihood_module._MAX_MC_LIKELIHOOD_ELEMENTS * 8


@pytest.mark.parametrize("kind", ["2pl_3d", "grm", "nrm"])
@pytest.mark.parametrize("method", ["importance", "posterior", "stochastic"])
def test_seeded_complete_fits_match_expanded_likelihoods(kind, method):
    model = _model(kind)
    responses, _ = _problem(model)
    options = (
        {"n_chains": 5}
        if method == "stochastic"
        else {"n_samples": 50, "importance_sampling": method == "importance"}
    )
    cls = StochasticEMEstimator if method == "stochastic" else MCEMEstimator
    actual = cls(**options, max_iter=2, seed=839)
    expanded = cls(**options, max_iter=2, seed=839)
    expanded._sample_log_likelihoods = MethodType(_expanded, expanded)
    result = actual.fit(model.copy(), responses)
    reference = expanded.fit(model.copy(), responses)
    np.testing.assert_allclose(
        actual.convergence_history, expanded.convergence_history, rtol=2e-8, atol=2e-7
    )
    np.testing.assert_allclose(
        result.log_likelihood, reference.log_likelihood, rtol=2e-8, atol=2e-7
    )
    assert (
        result.n_iterations == reference.n_iterations
        and result.converged == reference.converged
    )
    for name, values in result.model.parameters.items():
        np.testing.assert_allclose(
            values, reference.model.parameters[name], rtol=2e-5, atol=2e-5
        )


@pytest.mark.parametrize("kind", ["2pl_3d", "grm"])
@pytest.mark.parametrize("method", ["posterior", "stochastic"])
def test_seeded_posterior_samples_match_expanded_likelihoods(kind, method):
    model = _model(kind)
    responses, _ = _problem(model)
    if method == "posterior":
        actual = MCEMEstimator(n_samples=50, importance_sampling=False, seed=845)
        expanded = MCEMEstimator(n_samples=50, importance_sampling=False, seed=845)
    else:
        actual = StochasticEMEstimator(n_chains=5, seed=845)
        expanded = StochasticEMEstimator(n_chains=5, seed=845)
    expanded._sample_log_likelihoods = MethodType(_expanded, expanded)
    actual._rng = np.random.default_rng(845)
    expanded._rng = np.random.default_rng(845)
    arguments = (
        model,
        responses,
        np.zeros(model.n_factors),
        np.eye(model.n_factors),
        model.n_factors,
    )
    samples, weights = actual._e_step_mc(*arguments)
    expected_samples, expected_weights = expanded._e_step_mc(*arguments)
    np.testing.assert_array_equal(samples, expected_samples)
    np.testing.assert_array_equal(weights, expected_weights)


def test_qmc_independent_draws_use_bounded_reduction():
    model = _model("2pl_3d")
    responses, samples = _problem(model)
    expected = _expanded(None, model, responses, samples)
    actual = QMCEMEstimator(n_samples=50)._sample_log_likelihoods(
        model, responses, samples
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
