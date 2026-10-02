"""Optional Monte Carlo uncertainty matches conditional item curvature."""

import tracemalloc

import numpy as np
import pytest

import mirt.estimation._mc_information as information_module
import mirt.estimation._mc_objective as objective_module
import mirt.estimation._polytomous_information as polytomous_information_module
from mirt.constants import PROB_EPSILON
from mirt.estimation.mcem import MCEMEstimator, QMCEMEstimator, StochasticEMEstimator
from mirt.models.dichotomous import (
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

_FACTORIES = (
    lambda **kw: MCEMEstimator(n_samples=50, **kw),
    lambda **kw: QMCEMEstimator(n_samples=50, **kw),
    lambda **kw: StochasticEMEstimator(n_chains=50, **kw),
)
_MODELS = ("1pl", "2pl", "2pl_3d", "3pl", "4pl", "mirt", "grm", "gpcm", "pcm", "nrm")


def _model(kind):
    constructors = {
        "1pl": lambda: OneParameterLogistic(2),
        "2pl": lambda: TwoParameterLogistic(2),
        "2pl_3d": lambda: TwoParameterLogistic(2, n_factors=3),
        "3pl": lambda: ThreeParameterLogistic(2),
        "4pl": lambda: FourParameterLogistic(2),
        "mirt": lambda: MultidimensionalModel(2, n_factors=3),
        "grm": lambda: GradedResponseModel(2, n_categories=[3, 4], n_factors=2),
        "gpcm": lambda: GeneralizedPartialCredit(2, n_categories=[3, 4], n_factors=2),
        "pcm": lambda: PartialCreditModel(2, n_categories=[3, 4]),
        "nrm": lambda: NominalResponseModel(2, n_categories=[3, 4], n_factors=2),
    }
    model = constructors[kind]()
    if kind == "4pl":
        model.set_parameters(upper=np.full(2, 0.85))
    return model


def _inputs(model, layout="ordinary", persons=13):
    rng = np.random.default_rng(941)
    responses = np.column_stack(
        [
            rng.integers(
                0, model.n_categories[j] if model.is_polytomous else 2, persons
            )
            for j in range(2)
        ]
    )
    responses[rng.random(responses.shape) < 0.2] = -1
    samples = rng.normal(size=(persons, 50, model.n_factors))
    if layout == "shared":
        samples = np.broadcast_to(samples[:1], samples.shape)
    elif layout == "float32":
        samples = samples.astype(np.float32)
    elif layout == "strided":
        storage = np.empty((persons, 100, model.n_factors))
        storage[:, ::2] = samples
        samples = storage[:, ::2]
    weights = rng.uniform(size=(persons, 50))
    weights /= weights.sum(axis=1, keepdims=True)
    for array in (responses, samples, weights):
        array.setflags(write=False)
    return responses, samples, weights


def _direct_item_log_likelihood(model, item, responses, samples, weights):
    valid = responses[:, item] >= 0
    points = samples[valid].reshape(-1, model.n_factors)
    decisions = np.repeat(responses[valid, item], samples.shape[1])
    probability = model.probability(points, item)
    if model.is_polytomous:
        selected = probability[np.arange(len(points)), decisions]
        logs = np.log(np.clip(selected, PROB_EPSILON, 1.0))
    else:
        p = np.clip(probability.ravel(), PROB_EPSILON, 1 - PROB_EPSILON)
        logs = decisions * np.log(p) + (1 - decisions) * np.log1p(-p)
    return float(weights[valid].ravel() @ logs)


def _reference(model, responses, samples, weights):
    parameters, masks = model.parameters, model.free_parameter_masks
    result = {name: np.zeros_like(value) for name, value in parameters.items()}
    for item in range(model.n_items):
        center = _direct_item_log_likelihood(model, item, responses, samples, weights)
        for name, values in parameters.items():
            current = np.asarray(values[item]).copy()
            output = result[name][item : item + 1].reshape(-1)
            for index in np.flatnonzero(np.asarray(masks[name][item]).reshape(-1)):
                losses = []
                for offset in (-1e-3, 1e-3):
                    candidate = current.copy()
                    candidate.reshape(-1)[index] += offset
                    try:
                        model.set_item_parameter(item, name, candidate)
                        losses.append(
                            _direct_item_log_likelihood(
                                model, item, responses, samples, weights
                            )
                        )
                    finally:
                        model.set_item_parameter(item, name, current)
                curvature = (losses[0] - 2 * center + losses[1]) / 1e-6
                output[index] = np.sqrt(-1 / curvature) if curvature < 0 else np.nan
    return result


@pytest.mark.parametrize("factory", _FACTORIES)
@pytest.mark.parametrize("value", [True, np.nan, np.inf, -1, 0, "small", None])
def test_invalid_se_step_sizes_are_rejected(factory, value):
    with pytest.raises(ValueError, match="se_step_size"):
        factory(se_step_size=value)


@pytest.mark.parametrize("factory", _FACTORIES)
@pytest.mark.parametrize("value", [1, None, "yes"])
def test_standard_error_switch_requires_a_boolean(factory, value):
    with pytest.raises(TypeError, match="compute_standard_errors"):
        factory(compute_standard_errors=value)


@pytest.mark.parametrize("kind", _MODELS)
@pytest.mark.parametrize(
    "layout", ["ordinary", "shared", "float32", "strided", "blocked"]
)
def test_standard_errors_match_direct_weighted_likelihood_curvature(
    kind, layout, monkeypatch
):
    model = _model(kind)
    responses, samples, weights = _inputs(model, layout)
    originals = [array.copy() for array in (responses, samples, weights)]
    parameters = model.parameters
    expected = _reference(model, responses, samples, weights)
    if layout == "blocked":
        monkeypatch.setattr(objective_module, "_MAX_MC_OBJECTIVE_ENTRIES", 17)
    estimator = MCEMEstimator(n_samples=50, compute_standard_errors=True)
    actual = estimator._compute_standard_errors_mc(model, responses, samples, weights)
    for name in parameters:
        np.testing.assert_allclose(actual[name], expected[name], rtol=3e-5, atol=1e-7)
        np.testing.assert_array_equal(model.parameters[name], parameters[name])
        np.testing.assert_array_equal(
            actual[name][~model.free_parameter_masks[name]], 0
        )
    for array, original in zip((responses, samples, weights), originals, strict=True):
        np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize("kind", ["grm", "gpcm", "nrm"])
@pytest.mark.parametrize("n_factors", [1, 2, 4])
@pytest.mark.parametrize("layout", ["ordinary", "shared", "blocked"])
def test_exact_polytomous_curvature_handles_nondefault_multidimensional_parameters(
    kind, n_factors, layout
):
    constructors = {
        "grm": GradedResponseModel,
        "gpcm": GeneralizedPartialCredit,
        "nrm": NominalResponseModel,
    }
    model = constructors[kind](2, n_categories=[3, 4], n_factors=n_factors)
    rng = np.random.default_rng(978)
    parameters = model.parameters
    for name, values in parameters.items():
        values[:] = (
            rng.uniform(0.4, 1.0, values.shape)
            if name == "discrimination"
            else rng.normal(scale=0.5, size=values.shape)
        )
        if name == "thresholds":
            values.sort(axis=1)
    model.set_parameters(**parameters)
    responses, samples, weights = _inputs(model, layout)
    samples = samples * 0.4
    if layout == "shared":
        samples = np.broadcast_to(samples[:1], samples.shape)
    expected = _reference(model, responses, samples, weights)
    if layout == "blocked":
        samples = np.tile(samples, (20, 1, 1))
        responses = np.tile(responses, (20, 1))
        weights = np.tile(weights, (20, 1)) / 20.0
    actual = MCEMEstimator(
        n_samples=50, compute_standard_errors=True
    )._compute_standard_errors_mc(model, responses, samples, weights)
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("kind", ["grm", "gpcm", "nrm"])
def test_exact_polytomous_curvature_preserves_model_after_failure(kind, monkeypatch):
    model = _model(kind)
    responses, samples, weights = _inputs(model)
    parameters = model.parameters

    def fail(*args, **kwargs):
        raise RuntimeError("polytomous curvature failed")

    monkeypatch.setattr(information_module, "polytomous_item_curvature", fail)
    with pytest.raises(RuntimeError, match="polytomous curvature failed"):
        MCEMEstimator(
            n_samples=50, compute_standard_errors=True
        )._compute_standard_errors_mc(model, responses, samples, weights)
    for name in parameters:
        np.testing.assert_array_equal(model.parameters[name], parameters[name])


@pytest.mark.parametrize("kind", ["grm", "gpcm", "nrm"])
def test_exact_polytomous_curvature_streams_bounded_blocks(kind, monkeypatch):
    model = _model(kind)
    responses, samples, weights = _inputs(model, persons=1000)
    largest = 0
    original = information_module.polytomous_item_curvature

    def counted(model, item, points, counts, epsilon):
        nonlocal largest
        largest = max(largest, points.size, counts.size)
        return original(model, item, points, counts, epsilon)

    monkeypatch.setattr(information_module, "polytomous_item_curvature", counted)
    errors = MCEMEstimator(
        n_samples=50, compute_standard_errors=True
    )._compute_standard_errors_mc(model, responses, samples, weights)
    assert 0 < largest <= 32_768
    for name, mask in model.free_parameter_masks.items():
        assert np.isfinite(errors[name][mask]).all()


def test_multidimensional_partial_credit_zero_slopes_keep_slope_information():
    model = _model("gpcm")
    model.set_parameters(discrimination=np.zeros_like(model.discrimination))
    responses, samples, weights = _inputs(model)
    errors = MCEMEstimator(
        n_samples=50, compute_standard_errors=True
    )._compute_standard_errors_mc(model, responses, samples, weights)
    assert np.isfinite(errors["discrimination"]).all()
    assert (errors["discrimination"] > 0).all()
    assert np.isnan(errors["steps"][model.free_parameter_masks["steps"]]).all()


@pytest.mark.parametrize("kind", ["grm", "gpcm", "nrm"])
@pytest.mark.parametrize("layout", ["ordinary", "shared"])
@pytest.mark.parametrize("condition", ["clipped", "zero_weights", "missing"])
def test_exact_polytomous_undefined_curvature_leaves_fixed_coordinates_zero(
    kind, layout, condition
):
    model = _model(kind)
    responses, samples, weights = _inputs(model, layout)
    if condition == "clipped":
        samples = np.full_like(samples, 1e6)
    elif condition == "zero_weights":
        weights = np.zeros_like(weights)
    else:
        responses = np.full_like(responses, -1)
    if layout == "shared":
        samples = np.broadcast_to(samples[:1], samples.shape)
    errors = MCEMEstimator(
        n_samples=50, compute_standard_errors=True
    )._compute_standard_errors_mc(model, responses, samples, weights)
    for name, mask in model.free_parameter_masks.items():
        assert np.isnan(errors[name][mask]).all()
        np.testing.assert_array_equal(errors[name][~mask], 0.0)


def test_unsupported_polytomous_curvature_model_is_rejected():
    model = _model("2pl")
    with pytest.raises(TypeError, match="Unsupported"):
        polytomous_information_module.polytomous_item_curvature(
            model, 0, np.zeros((2, 1)), np.ones((2, 2)), PROB_EPSILON
        )


@pytest.mark.parametrize("kind", _MODELS)
@pytest.mark.parametrize("binding", ["instance", "class", "subclass"])
def test_custom_probability_uses_numerical_curvature_and_restores_parameters(
    kind, binding, monkeypatch
):
    model = _model(kind)
    responses, samples, weights = _inputs(model)
    expected = _reference(model, responses, samples, weights)
    original, parameters = model.probability, model.parameters
    calls = 0

    def custom(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    if binding == "instance":
        monkeypatch.setattr(model, "probability", custom)
    else:

        def class_custom(self, *args, **kwargs):
            return custom(*args, **kwargs)

        if binding == "class":
            monkeypatch.setattr(type(model), "probability", class_custom)
        else:
            model.__class__ = type(
                "CustomCurve", (type(model),), {"probability": class_custom}
            )
    estimator = MCEMEstimator(
        n_samples=50, compute_standard_errors=True, se_step_size=1e-3
    )
    actual = estimator._compute_standard_errors_mc(model, responses, samples, weights)
    assert calls > 0
    for name in parameters:
        np.testing.assert_allclose(actual[name], expected[name], rtol=2e-7, atol=1e-7)
        np.testing.assert_array_equal(model.parameters[name], parameters[name])


@pytest.mark.parametrize("binding", ["instance", "class", "subclass"])
def test_custom_affine_curve_parameters_remain_authoritative(binding, monkeypatch):
    model = _model("mirt")
    responses, samples, weights = _inputs(model)
    original = model._curve_parameters
    calls = 0

    def custom(item):
        nonlocal calls
        calls += 1
        slopes, intercepts = original(item)
        return slopes * 0.75, intercepts + 0.1

    if binding == "instance":
        monkeypatch.setattr(model, "_curve_parameters", custom)
    else:

        def class_custom(self, item):
            return custom(item)

        if binding == "class":
            monkeypatch.setattr(type(model), "_curve_parameters", class_custom)
        else:
            model.__class__ = type(
                "CustomAffine", (type(model),), {"_curve_parameters": class_custom}
            )
    expected = _reference(model, responses, samples, weights)
    calls = 0
    actual = MCEMEstimator(
        n_samples=50, compute_standard_errors=True, se_step_size=1e-3
    )._compute_standard_errors_mc(model, responses, samples, weights)
    assert calls > 0
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name], rtol=2e-7, atol=1e-7)


@pytest.mark.parametrize("factory", _FACTORIES)
def test_default_fit_retains_placeholder_errors(factory):
    estimator = factory()
    model = _model("1pl")
    responses, samples, weights = _inputs(model)
    actual = estimator._compute_standard_errors_mc(model, responses, samples, weights)
    np.testing.assert_array_equal(actual["discrimination"], 0)
    assert np.isnan(actual["difficulty"]).all()


@pytest.mark.parametrize(
    "method", ["importance", "posterior", "sobol", "halton", "stochastic"]
)
def test_opt_in_fit_reports_errors_without_changing_seeded_fit(method):
    options = {"n_samples": 50, "max_iter": 2, "seed": 945}
    if method in ("sobol", "halton"):
        cls, options = QMCEMEstimator, {**options, "sequence": method}
    elif method == "stochastic":
        cls, options = (
            StochasticEMEstimator,
            {"n_chains": 5, "max_iter": 2, "seed": 945},
        )
    else:
        cls, options = (
            MCEMEstimator,
            {**options, "importance_sampling": method == "importance"},
        )
    model = _model("2pl")
    responses, _, _ = _inputs(model, persons=61)
    ordinary, enabled = cls(**options), cls(**options, compute_standard_errors=True)
    expected = ordinary.fit(model.copy(), responses)
    actual = enabled.fit(model.copy(), responses)
    assert actual.log_likelihood == expected.log_likelihood
    assert actual.n_iterations == expected.n_iterations
    assert enabled.convergence_history == ordinary.convergence_history
    for name in actual.model.parameters:
        np.testing.assert_array_equal(
            actual.model.parameters[name], expected.model.parameters[name]
        )
        assert np.isfinite(actual.standard_errors[name]).all()


def test_missing_items_zero_weight_draws_and_clipped_curvature_are_undefined():
    estimator = MCEMEstimator(n_samples=50, compute_standard_errors=True)
    model = _model("1pl")
    responses, samples, weights = _inputs(model)
    for data, points, mass in (
        (np.full_like(responses, -1), samples, weights),
        (responses, samples, np.zeros_like(weights)),
        (responses, np.full_like(samples, 1e6), weights),
    ):
        errors = estimator._compute_standard_errors_mc(model, data, points, mass)
        assert np.isnan(errors["difficulty"]).all()
        np.testing.assert_array_equal(errors["discrimination"], 0)


def test_borrowed_draw_peak_excludes_a_full_observed_person_copy():
    estimator = MCEMEstimator(n_samples=50, compute_standard_errors=True)
    model = _model("2pl_3d")
    responses, samples, weights = _inputs(model, persons=5000)
    tracemalloc.start()
    try:
        errors = estimator._compute_standard_errors_mc(
            model, responses, samples, weights
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert np.isfinite(errors["discrimination"]).all()
    assert peak < samples.nbytes / 2


def test_close_thresholds_match_numerical_curvature(monkeypatch):
    model = GradedResponseModel(2, n_categories=3)
    model.set_parameters(thresholds=np.array([[0.0, 1e-6], [0.0, 1e-6]]))
    responses, samples, weights = _inputs(model)
    analytic = MCEMEstimator(n_samples=50, compute_standard_errors=True)
    expected_estimator = MCEMEstimator(
        n_samples=50, compute_standard_errors=True, se_step_size=1e-9
    )
    actual = analytic._compute_standard_errors_mc(model, responses, samples, weights)
    monkeypatch.setattr(model, "probability", model.probability)
    expected = expected_estimator._compute_standard_errors_mc(
        model, responses, samples, weights
    )
    np.testing.assert_allclose(actual["thresholds"], expected["thresholds"], rtol=1e-4)


@pytest.mark.parametrize("kind", ["3pl", "4pl"])
def test_boundary_parameters_use_valid_one_sided_stencils(kind, monkeypatch):
    model = _model(kind)
    model.set_parameters(guessing=np.zeros(2))
    if kind == "4pl":
        model.set_parameters(upper=np.ones(2))
    responses, samples, weights = _inputs(model)
    estimator = MCEMEstimator(n_samples=50, compute_standard_errors=True)
    actual = estimator._compute_standard_errors_mc(model, responses, samples, weights)
    monkeypatch.setattr(model, "probability", model.probability)
    expected = MCEMEstimator(
        n_samples=50, compute_standard_errors=True, se_step_size=1e-4
    )._compute_standard_errors_mc(model, responses, samples, weights)
    for name in actual:
        np.testing.assert_allclose(actual[name], expected[name], rtol=2e-3, atol=1e-7)


def test_rasch_errors_match_closed_form_weighted_information():
    model = _model("1pl")
    responses, samples, weights = _inputs(model)
    estimator = MCEMEstimator(n_samples=50, compute_standard_errors=True)
    actual = estimator._compute_standard_errors_mc(model, responses, samples, weights)
    for item in range(2):
        p = model.probability(samples.reshape(-1, 1), item).reshape(weights.shape)
        info = np.sum(weights * p * (1 - p) * (responses[:, item] >= 0)[:, None])
        np.testing.assert_allclose(
            actual["difficulty"][item], 1 / np.sqrt(info), rtol=1e-9
        )


@pytest.mark.parametrize("layout", ["ordinary", "shared"])
@pytest.mark.parametrize("kind", ["2pl_3d", "grm", "gpcm", "nrm"])
def test_weight_scaling_matches_information_scaling(kind, layout):
    model = _model(kind)
    responses, samples, weights = _inputs(model, layout)
    estimator = MCEMEstimator(n_samples=50, compute_standard_errors=True)
    errors = estimator._compute_standard_errors_mc(model, responses, samples, weights)
    scaled = estimator._compute_standard_errors_mc(
        model, responses, samples, 4 * weights
    )
    for name in errors:
        np.testing.assert_allclose(scaled[name], errors[name] / 2, rtol=1e-11)


@pytest.mark.parametrize("callback", ["probability", "objective"])
def test_callback_failure_restores_every_item_parameter(callback, monkeypatch):
    estimator = MCEMEstimator(n_samples=50, compute_standard_errors=True)
    model = _model("2pl_3d")
    responses, samples, weights = _inputs(model)
    parameters = model.parameters

    def fail(*args, **kwargs):
        raise RuntimeError("uncertainty callback failed")

    target, name = (
        (model, "probability")
        if callback == "probability"
        else (estimator, "_item_expected_log_likelihood")
    )
    monkeypatch.setattr(target, name, fail)
    with pytest.raises(RuntimeError, match="uncertainty callback failed"):
        estimator._compute_standard_errors_mc(model, responses, samples, weights)
    for name in parameters:
        np.testing.assert_array_equal(model.parameters[name], parameters[name])


def test_prepared_gradient_failure_restores_every_item_parameter(monkeypatch):
    estimator = MCEMEstimator(n_samples=50, compute_standard_errors=True)
    model = _model("2pl_3d")
    responses, samples, weights = _inputs(model)
    parameters = model.parameters

    def fail(*args, **kwargs):
        raise RuntimeError("prepared gradient failed")

    monkeypatch.setattr(information_module, "prepare_mc_objective", lambda *args: fail)
    with pytest.raises(RuntimeError, match="prepared gradient failed"):
        estimator._compute_standard_errors_mc(model, responses, samples, weights)
    for name in parameters:
        np.testing.assert_array_equal(model.parameters[name], parameters[name])


@pytest.mark.parametrize(
    "invalid",
    [
        "samples_shape",
        "weights_shape",
        "negative_weights",
        "nonfinite_weights",
        "nonfinite_samples",
        "nonfinite_grid",
    ],
)
def test_invalid_uncertainty_inputs_are_rejected(invalid):
    estimator = MCEMEstimator(n_samples=50, compute_standard_errors=True)
    model = _model("2pl")
    responses, samples, weights = _inputs(model)
    if invalid == "samples_shape":
        samples = samples[:, :-1]
    elif invalid == "weights_shape":
        weights = weights[:, :-1]
    elif invalid.startswith("nonfinite_") and invalid != "nonfinite_weights":
        samples = samples.copy()
        samples[:, 0] = np.nan
        if invalid == "nonfinite_grid":
            samples = np.broadcast_to(samples[:1], samples.shape)
    else:
        weights = weights.copy()
        weights[0, 0] = -1 if invalid == "negative_weights" else np.nan
    with pytest.raises(ValueError, match="shape|weights|finite"):
        estimator._compute_standard_errors_mc(model, responses, samples, weights)
