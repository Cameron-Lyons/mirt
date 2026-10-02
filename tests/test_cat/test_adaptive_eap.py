"""Compare adaptive scores with independently reduced binary posteriors."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit, roots_hermite

from mirt.cat import CATEngine, MCATEngine
from mirt.models import (
    BifactorModel,
    FourParameterLogistic,
    MultidimensionalModel,
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.scoring import ability_posterior


def _model(cls=TwoParameterLogistic, n_factors=1):
    rng = np.random.default_rng(632)
    if cls is BifactorModel:
        # Sparse external labels must map to separate ability columns; these
        # labels are deliberately larger than the number of factors.
        model = cls(n_items=8, specific_factors=[10, 20, 10, 20, 10, 20, 20, 10])
        model.set_parameters(
            general_loadings=rng.uniform(0.3, 1.8, 8),
            specific_loadings=rng.uniform(0.3, 1.8, 8),
            intercepts=rng.normal(size=8),
        )
    else:
        model = cls(n_items=8, n_factors=n_factors)
    if cls is MultidimensionalModel:
        model.set_parameters(
            slopes=rng.uniform(0.3, 1.8, (8, n_factors)),
            intercepts=rng.normal(size=8),
        )
    elif cls is not BifactorModel:
        params = {"difficulty": rng.normal(size=8)}
        if cls is not OneParameterLogistic:
            shape = 8 if n_factors == 1 else (8, n_factors)
            params["discrimination"] = rng.uniform(0.3, 1.8, shape)
        if cls in (ThreeParameterLogistic, FourParameterLogistic):
            params["guessing"] = rng.uniform(0.05, 0.25, 8)
        if cls is FourParameterLogistic:
            params["upper"] = rng.uniform(0.8, 0.95, 8)
        model.set_parameters(**params)
    model._is_fitted = True
    return model


def _reference(model, items, responses, n_quadpts):
    nodes, weights = roots_hermite(n_quadpts)
    nodes *= np.sqrt(2)
    weights /= np.sqrt(np.pi)
    points = np.column_stack(
        [v.ravel() for v in np.meshgrid(*([nodes] * model.n_factors), indexing="ij")]
    )
    mass = np.prod(
        np.stack(np.meshgrid(*([weights] * model.n_factors), indexing="ij")), axis=0
    ).ravel()
    params = model.parameters
    for item, response in zip(items, responses, strict=True):
        if "slopes" in params:
            p = expit(points @ params["slopes"][item] + params["intercepts"][item])
        elif "general_loadings" in params:
            labels = np.unique(model.specific_factors)
            specific_column = (
                1 + np.flatnonzero(labels == model.specific_factors[item])[0]
            )
            p = expit(
                points[:, 0] * params["general_loadings"][item]
                + points[:, specific_column] * params["specific_loadings"][item]
                + params["intercepts"][item]
            )
        else:
            a = np.atleast_1d(params["discrimination"][item])
            # The difficulty is subtracted from every ability coordinate in
            # this parameterization, so its intercept is -b * sum(a).
            p = expit(points @ a - params["difficulty"][item] * a.sum())
            lower = params.get("guessing", np.zeros(model.n_items))[item]
            upper = params.get("upper", np.ones(model.n_items))[item]
            p = lower + (upper - lower) * p
        p = np.clip(p, 1e-10, 1 - 1e-10)
        mass *= p if response else (1 - p)
    mass /= mass.sum()
    mean = mass @ points
    residuals = points - mean
    covariance = np.einsum("q,qi,qj->ij", mass, residuals, residuals)
    return mean, covariance


@pytest.mark.parametrize(
    "cls",
    [
        OneParameterLogistic,
        TwoParameterLogistic,
        ThreeParameterLogistic,
        FourParameterLogistic,
    ],
)
def test_cat_history_matches_independent_posterior(cls):
    engine = CATEngine(_model(cls), n_quadpts=15, min_items=5, max_items=5)
    for response in [1, 0, 0, 1, 1]:
        state = engine.administer_item(response)
        mean, covariance = _reference(
            engine.model, state.items_administered, state.responses, 15
        )
        assert_allclose(state.theta, mean[0], atol=2e-14)
        assert_allclose(state.standard_error**2, covariance[0, 0], atol=2e-14)
    result = engine.get_result()
    assert_allclose(result.theta_history[-1], mean[0], atol=2e-14)


@pytest.mark.parametrize(
    ("cls", "n_factors"),
    [
        (TwoParameterLogistic, 2),
        (MultidimensionalModel, 2),
        (MultidimensionalModel, 3),
        (BifactorModel, 3),
    ],
)
@pytest.mark.parametrize("chunk_size", [3, 131_072])
def test_mcat_covariance_matches_independent_posterior(
    monkeypatch, cls, n_factors, chunk_size
):
    monkeypatch.setattr("mirt.cat._eap._MAX_CURVE_VALUES", chunk_size)
    engine = MCATEngine(_model(cls, n_factors), n_quadpts=7, min_items=3, max_items=3)
    for response in [1, 0, 1]:
        state = engine.administer_item(response)
        mean, covariance = _reference(
            engine.model, state.items_administered, state.responses, 7
        )
        assert_allclose(state.theta, mean, atol=2e-14)
        assert_allclose(state.covariance, covariance, atol=2e-14)
        assert_allclose(state.standard_error**2, np.diag(covariance), atol=2e-14)


@pytest.mark.parametrize(
    ("cls", "n_factors"), [(TwoParameterLogistic, 2), (BifactorModel, 3)]
)
@pytest.mark.parametrize("chunk_size", [3, 131_072])
def test_mutable_multidimensional_parameters_refresh_prior_response_evidence(
    monkeypatch, cls, n_factors, chunk_size
):
    monkeypatch.setattr("mirt.cat._eap._MAX_CURVE_VALUES", chunk_size)
    model = _model(cls, n_factors)
    engine = MCATEngine(model, n_quadpts=7, min_items=3, max_items=3)
    state = engine.administer_item(1)
    original_theta = state.theta.copy()
    item = state.items_administered[0]

    # These public array properties can be changed without set_parameters().
    # Rescoring the same history must use the updated parameter values.
    if cls is BifactorModel:
        model.general_loadings[item] *= 1.7
        model.specific_loadings[item] *= 0.6
        model.intercepts[item] += 0.8
    else:
        model.discrimination[item] *= np.array([1.7, 0.6])
        model.difficulty[item] += 0.8
    engine._update_theta()
    state = engine.get_current_state()
    mean, covariance = _reference(model, state.items_administered, state.responses, 7)

    assert np.max(np.abs(mean - original_theta)) > 0.01
    assert_allclose(state.theta, mean, atol=2e-14)
    assert_allclose(state.covariance, covariance, atol=2e-14)
    assert_allclose(state.standard_error**2, np.diag(covariance), atol=2e-14)


def test_quadrature_is_reused_and_probability_work_excludes_unused_items(monkeypatch):
    from mirt.models import dichotomous
    from mirt.scoring import _common

    original_quad = _common.build_quadrature
    original_probability = dichotomous._logistic_probability
    quad_calls = []
    shapes = []

    def quad(**kwargs):
        quad_calls.append(kwargs)
        return original_quad(**kwargs)

    def probability(logits, *args):
        shapes.append(logits.shape)
        return original_probability(logits, *args)

    monkeypatch.setattr(_common, "build_quadrature", quad)
    monkeypatch.setattr(dichotomous, "_logistic_probability", probability)
    engine = CATEngine(_model(), n_quadpts=11, min_items=3, max_items=3)
    for response in [1, 0, 1]:
        engine.administer_item(response)
    assert len(quad_calls) == 1
    assert shapes == [(11,)] * 6  # Histories of length 1, 2, then 3.


def test_parameter_updates_reset_and_grid_changes_have_no_stale_evidence():
    engine = CATEngine(_model(), n_quadpts=11, min_items=3, max_items=3)
    engine.administer_item(1)
    engine.model.set_parameters(difficulty=np.linspace(-2, 2, 8))
    engine.n_quadpts = 17
    state = engine.administer_item(0)
    mean, covariance = _reference(
        engine.model, state.items_administered, state.responses, 17
    )
    assert_allclose(state.theta, mean[0], atol=2e-14)
    assert_allclose(state.standard_error**2, covariance[0, 0], atol=2e-14)

    engine.reset()
    state = engine.administer_item(0)
    mean, covariance = _reference(
        engine.model, state.items_administered, state.responses, 17
    )
    assert_allclose(state.theta, mean[0], atol=2e-14)
    assert_allclose(state.standard_error**2, covariance[0, 0], atol=2e-14)


@pytest.mark.parametrize("invalid_count", [11.0, True, 0])
def test_cached_grid_preserves_quadrature_validation(invalid_count):
    from mirt.cat._eap import score_binary_eap

    engine = CATEngine(_model(), n_quadpts=11, min_items=3, max_items=3)
    engine.administer_item(1)
    engine.n_quadpts = invalid_count
    with pytest.raises(ValueError, match="n_quadpts"):
        score_binary_eap(engine)


def test_cached_grid_preserves_fitted_model_requirement():
    from mirt.cat._eap import score_binary_eap

    engine = CATEngine(_model(), n_quadpts=11, min_items=3, max_items=3)
    engine.administer_item(1)
    engine.model._is_fitted = False
    with pytest.raises(ValueError, match="fitted"):
        score_binary_eap(engine)


def test_nonfinite_posterior_keeps_last_mcat_state_and_can_recover():
    model = _model(MultidimensionalModel, 2)
    engine = MCATEngine(model, n_quadpts=7, min_items=3, max_items=3)
    engine._items_administered = [0, 1]
    engine._responses = [1, 0]
    engine._update_theta()
    original_theta = engine._current_theta.copy()
    original_covariance = engine._current_covariance.copy()

    # Public slope arrays are mutable. Invalid evidence must leave both moments
    # intact, and correcting the parameters must reevaluate the full history.
    model.slopes[0, 0] = np.nan
    engine._update_theta()
    assert_allclose(engine._current_theta, original_theta)
    assert_allclose(engine._current_covariance, original_covariance)

    model.slopes[0, 0] = 0.7
    engine._update_theta()
    mean, covariance = _reference(model, [0, 1], [1, 0], 7)
    assert_allclose(engine._current_theta, mean, atol=2e-14)
    assert_allclose(engine._current_covariance, covariance, atol=2e-14)


@pytest.mark.parametrize("hook", ["probability", "log_likelihood_batch"])
def test_custom_hooks_keep_full_public_scoring(monkeypatch, hook):
    model = _model()
    original = getattr(model, hook)
    calls = []

    def customized(*args, **kwargs):
        calls.append(1)
        values = original(*args, **kwargs)
        return 0.1 + 0.8 * values if hook == "probability" else 0.6 * values

    monkeypatch.setattr(model, hook, customized)
    engine = CATEngine(model, n_quadpts=11, min_items=3, max_items=3)
    state = engine.administer_item(1)
    assert calls
    assert not hasattr(engine, "_eap_quadrature")
    responses = np.full((1, 8), -1)
    responses[0, state.items_administered] = state.responses
    posterior = ability_posterior(model, responses, n_quadpts=11)
    assert_allclose(state.theta, posterior.mean[0], atol=2e-14)
    assert_allclose(state.standard_error, posterior.standard_error[0], atol=2e-14)
