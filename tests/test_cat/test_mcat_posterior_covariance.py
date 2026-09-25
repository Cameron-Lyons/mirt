"""MCAT must retain cross-factor uncertainty after scoring responses."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.cat import CompositeClassificationStop, MCATEngine
from mirt.models import MultidimensionalModel
from mirt.scoring import ability_posterior, fscores


def _model():
    model = MultidimensionalModel(n_items=4, n_factors=2)
    model.set_parameters(
        slopes=np.array([[1.4, 1.1], [0.7, 1.5], [1.6, 0.8], [0.9, 1.2]]),
        intercepts=np.array([-0.3, 0.2, 0.5, -0.6]),
    )
    model._is_fitted = True
    return model


def _posterior_reference(model, responses, n_quadpts):
    posterior = ability_posterior(model, responses, n_quadpts=n_quadpts)
    mean = np.average(posterior.points, weights=posterior.weights[0], axis=0)
    residuals = posterior.points - mean
    covariance = np.einsum("q,qi,qj->ij", posterior.weights[0], residuals, residuals)
    return mean, covariance


def test_eap_state_and_history_retain_full_posterior_covariance():
    model = _model()
    engine = MCATEngine(model, n_quadpts=11, min_items=3, max_items=3)
    responses = np.full((1, model.n_items), -1)
    expected_history = []

    for response in [1, 0, 1]:
        item = engine.select_next_item()
        responses[0, item] = response
        state = engine.administer_item(response)
        mean, covariance = _posterior_reference(model, responses, 11)
        expected_history.append(covariance)

        assert abs(covariance[0, 1]) > 0.01
        assert_allclose(state.theta, mean, atol=1e-14)
        assert_allclose(state.covariance, covariance, atol=1e-14)
        assert_allclose(state.standard_error**2, np.diag(covariance), atol=1e-14)

    result = engine.get_result()
    assert_allclose(result.covariance, expected_history[-1], atol=1e-14)
    assert_allclose(result.covariance_history, expected_history, atol=1e-14)
    engine.reset()
    assert_allclose(engine.get_current_state().covariance, np.eye(2))


def test_composite_stopping_uses_correlations_from_actual_responses():
    model = _model()
    engine = MCATEngine(model, n_quadpts=15, min_items=4, max_items=4)
    state = engine.administer_item(1)
    weights = np.array([0.5, 0.5])
    full_se = np.sqrt(weights @ state.covariance @ weights)
    diagonal_se = np.sqrt(np.sum(weights**2 * np.diag(state.covariance)))
    assert full_se < diagonal_se

    # Place the decision boundary between the full and diagonal uncertainty
    # margins. A one-sided 95% normal cutoff is about 1.645 standard errors.
    cut = weights @ state.theta - 1.6448536269514722 * (full_se + diagonal_se) / 2
    stopping = CompositeClassificationStop(weights, cut_score=cut, confidence=0.95)
    assert stopping.should_stop(state)
    state.covariance = np.diag(np.diag(state.covariance))
    assert not stopping.should_stop(state)


def test_map_keeps_its_existing_standard_error_approximation():
    model = _model()
    engine = MCATEngine(model, scoring_method="MAP", min_items=2, max_items=2)
    item = engine.select_next_item()
    state = engine.administer_item(1)
    responses = np.full((1, model.n_items), -1)
    responses[0, item] = 1
    scores = fscores(model, responses, method="MAP")

    assert_allclose(state.theta, scores.theta.ravel())
    assert_allclose(state.covariance, np.diag(scores.standard_error.ravel() ** 2))


def test_failed_covariance_update_preserves_previous_state(monkeypatch):
    engine = MCATEngine(_model())
    original_theta = engine._current_theta.copy()
    original_covariance = engine._current_covariance.copy()

    def fail(*args, **kwargs):
        raise ValueError("posterior unavailable")

    monkeypatch.setattr("mirt.scoring.ability_posterior", fail)
    engine._update_theta()

    assert_allclose(engine._current_theta, original_theta)
    assert_allclose(engine._current_covariance, original_covariance)


@pytest.mark.parametrize("method", ["EAP", "MAP"])
def test_covariance_update_evaluates_responses_once(monkeypatch, method):
    model = _model()
    engine = MCATEngine(model, scoring_method=method, min_items=2, max_items=2)
    # Instrument the public scoring entry point, not each optimizer evaluation.
    name = "ability_posterior" if method == "EAP" else "fscores"
    original = ability_posterior if method == "EAP" else fscores
    calls = []

    def capture(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(f"mirt.scoring.{name}", capture)
    engine.administer_item(1)
    assert len(calls) == 1
