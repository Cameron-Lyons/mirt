"""Fit-local Monte Carlo evidence reuse and draw lifetime regressions."""

import weakref
from types import MethodType

import numpy as np
import pytest

import mirt.estimation.mcem as mcem_module
from mirt.estimation.mcem import MCEMEstimator, QMCEMEstimator, StochasticEMEstimator
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.polytomous import GeneralizedPartialCredit, GradedResponseModel

_METHODS = ("importance", "posterior", "stochastic", "sobol", "halton")


def _estimator(method, **kwargs):
    if method in ("sobol", "halton"):
        return QMCEMEstimator(n_samples=64, sequence=method, seed=931, **kwargs)
    if method == "stochastic":
        return StochasticEMEstimator(n_chains=5, seed=931, **kwargs)
    return MCEMEstimator(
        n_samples=50, importance_sampling=method == "importance", seed=931, **kwargs
    )


def _input(kind):
    rng = np.random.default_rng(932)
    if kind == "2pl":
        model = TwoParameterLogistic(3, n_factors=3)
    elif kind == "grm":
        model = GradedResponseModel(3, n_categories=4, n_factors=2)
    else:
        model = GeneralizedPartialCredit(3, n_categories=4, n_factors=2)
    responses = rng.integers(0, 4 if model.is_polytomous else 2, (17, 3))
    responses[rng.random(responses.shape) < 0.2] = -1
    responses[0] = -1
    mean = np.linspace(-0.3, 0.5, model.n_factors)
    covariance = np.eye(model.n_factors) + 0.2
    for value in (responses, mean, covariance):
        value.setflags(write=False)
    return model, responses, mean, covariance


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("kind", ["2pl", "grm", "gpcm"])
@pytest.mark.parametrize("stop", ["one", "limit", "converged"])
def test_fit_matches_separate_e_step_and_likelihood_evaluation(method, kind, stop):
    options = {
        "max_iter": 1 if stop == "one" else 3,
        "tol": 1e9 if stop == "converged" else 1e-12,
    }
    optimized, separate = _estimator(method, **options), _estimator(method, **options)
    # Keep the same item optimizer, while invoking the protected reporting hook.
    separate._estimate_marginal_ll = separate._estimate_marginal_ll
    model, responses, mean, covariance = _input(kind)
    originals = [value.copy() for value in (responses, mean, covariance)]
    actual = optimized.fit(model.copy(), responses, mean, covariance)
    expected = separate.fit(model.copy(), responses, mean, covariance)
    assert actual.log_likelihood == expected.log_likelihood
    assert actual.aic == expected.aic and actual.bic == expected.bic
    assert actual.n_iterations == expected.n_iterations
    assert actual.converged == expected.converged
    np.testing.assert_array_equal(
        optimized.convergence_history, separate.convergence_history
    )
    for name, parameters in actual.model.parameters.items():
        np.testing.assert_array_equal(parameters, expected.model.parameters[name])
    for name, errors in actual.standard_errors.items():
        np.testing.assert_array_equal(errors, expected.standard_errors[name])
    for value, original in zip((responses, mean, covariance), originals, strict=True):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("method", _METHODS)
def test_fit_evaluates_and_normalizes_each_importance_draw_once(method, monkeypatch):
    estimator = _estimator(method, max_iter=3)
    model = TwoParameterLogistic(2, n_factors=2)
    responses = np.random.default_rng(933).integers(-1, 2, (17, 2))
    probability, normalize = model.probability, mcem_module.normalize_log_posterior
    calls = {"probability": 0, "normalization": 0}

    def counted_probability(*args, **kwargs):
        calls["probability"] += 1
        return probability(*args, **kwargs)

    def counted_normalization(*args, **kwargs):
        calls["normalization"] += 1
        return normalize(*args, **kwargs)

    monkeypatch.setattr(model, "probability", counted_probability)
    monkeypatch.setattr(mcem_module, "normalize_log_posterior", counted_normalization)
    monkeypatch.setattr(estimator, "_m_step_mc", lambda *args: None)
    monkeypatch.setattr(estimator, "_check_convergence", lambda *args: False)
    monkeypatch.setattr(estimator, "_monte_carlo_converged", lambda *args: False)
    result = estimator.fit(model, responses)
    assert result.n_iterations == 3
    importance = method not in ("posterior", "stochastic")
    # MCEM's ascent check evaluates the draws of iterations two and three at
    # both the current and the previous iterate, without normalizing them.
    ascent = 4 if method in ("importance", "posterior") else 0
    assert calls["probability"] == (4 if importance else 64) + ascent
    assert calls["normalization"] == (4 if importance else 0)


@pytest.mark.parametrize("method", _METHODS)
def test_fit_releases_previous_draw_and_keeps_final_draw_for_errors(
    method, monkeypatch
):
    estimator = _estimator(method, max_iter=3)
    model = TwoParameterLogistic(2, n_factors=2)
    responses = np.random.default_rng(934).integers(-1, 2, (17, 2))
    probability = model.probability
    draws = []
    final_weights = None

    def check_previous_draw(*args, **kwargs):
        if draws and len(draws) < 3:
            assert all(reference() is None for reference in draws[-1])
        return probability(*args, **kwargs)

    def remember_draw(model, responses, samples, weights):
        nonlocal final_weights
        draws.append((weakref.ref(samples), weakref.ref(weights)))
        if len(draws) == 3:
            final_weights = weights.copy()

    def check_final_draw(model, responses, samples, weights):
        assert draws[-1][0]() is samples
        # Importance weights are refreshed after the last item update.
        np.testing.assert_array_equal(weights, final_weights)
        return {}

    monkeypatch.setattr(model, "probability", check_previous_draw)
    monkeypatch.setattr(estimator, "_m_step_mc", remember_draw)
    monkeypatch.setattr(estimator, "_check_convergence", lambda *args: False)
    monkeypatch.setattr(estimator, "_monte_carlo_converged", lambda *args: False)
    monkeypatch.setattr(estimator, "_compute_standard_errors_mc", check_final_draw)
    result = estimator.fit(model, responses)
    assert result.n_iterations == len(draws) == 3
    assert all(reference() is None for draw in draws for reference in draw)


@pytest.mark.parametrize("method", _METHODS)
def test_repeated_fits_reset_draws_weights_and_likelihood_history(method):
    estimator = _estimator(method, max_iter=2, tol=1e-12)
    model, responses, mean, covariance = _input("2pl")
    first = estimator.fit(model.copy(), responses, mean, covariance)
    history = estimator.convergence_history
    second = estimator.fit(model.copy(), responses, mean, covariance)
    assert first.log_likelihood == second.log_likelihood
    assert history == estimator.convergence_history
    for name, parameters in first.model.parameters.items():
        np.testing.assert_array_equal(parameters, second.model.parameters[name])


@pytest.mark.parametrize("method", _METHODS)
def test_interrupted_fit_releases_previous_draw_and_can_be_retried(method, monkeypatch):
    estimator = _estimator(method, max_iter=2, tol=1e-12)
    model, responses, mean, covariance = _input("2pl")
    probability, m_step = model.probability, estimator._m_step_mc
    previous = []

    def interrupt_next_draw(*args, **kwargs):
        if previous:
            assert all(reference() is None for reference in previous)
            raise RuntimeError("interrupted E-step")
        return probability(*args, **kwargs)

    def remember_m_step(model, responses, samples, weights):
        m_step(model, responses, samples, weights)
        previous.extend((weakref.ref(samples), weakref.ref(weights)))

    monkeypatch.setattr(model, "probability", interrupt_next_draw)
    monkeypatch.setattr(estimator, "_m_step_mc", remember_m_step)
    with pytest.raises(RuntimeError, match="interrupted E-step"):
        estimator.fit(model, responses, mean, covariance)
    assert all(reference() is None for reference in previous)
    monkeypatch.setattr(estimator, "_m_step_mc", m_step)
    fresh_model, _, _, _ = _input("2pl")
    actual = estimator.fit(fresh_model.copy(), responses, mean, covariance)
    expected = _estimator(method, max_iter=2, tol=1e-12).fit(
        fresh_model.copy(), responses, mean, covariance
    )
    assert actual.log_likelihood == expected.log_likelihood
    for name, parameters in actual.model.parameters.items():
        np.testing.assert_array_equal(parameters, expected.model.parameters[name])


_HOOK_CASES = (
    [
        (method, hook, expected)
        for method in _METHODS
        for hook, expected in (
            ("_e_step_mc", 1),
            ("_estimate_marginal_ll", 1),
            ("_refresh_mc_state", 2),
            (
                "_sample_log_likelihoods",
                2
                if method in ("sobol", "halton")
                else 23
                if method in ("posterior", "stochastic")
                else 3,
            ),
        )
    ]
    + [
        (method, "_normalized_importance_weights", 3)
        for method in ("importance", "sobol", "halton")
    ]
    + [
        (method, hook, expected)
        for method in ("posterior", "stochastic")
        for hook, expected in (
            ("_draw_posterior_samples", 1),
            ("_gaussian_log_kernel", 21),
        )
    ]
)


@pytest.mark.parametrize(("method", "hook", "expected"), _HOOK_CASES)
@pytest.mark.parametrize("binding", ["instance", "class", "subclass"])
def test_custom_estimator_hooks_keep_separate_evaluation(
    method, hook, expected, binding, monkeypatch
):
    estimator = _estimator(method, max_iter=1)
    cls = type(estimator)
    original = getattr(cls, hook)
    static = hook in ("_normalized_importance_weights", "_gaussian_log_kernel")
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    if binding == "subclass":
        estimator.__class__ = type(
            "CustomFit", (cls,), {hook: staticmethod(counted) if static else counted}
        )
    elif binding == "class":
        monkeypatch.setattr(cls, hook, staticmethod(counted) if static else counted)
    else:
        monkeypatch.setattr(
            estimator, hook, counted if static else MethodType(counted, estimator)
        )
    model = TwoParameterLogistic(1)
    responses = np.array([[0], [1], [-1]])
    monkeypatch.setattr(estimator, "_m_step_mc", lambda *args: None)
    estimator.fit(model, responses)
    assert calls == expected


@pytest.mark.parametrize("method", _METHODS)
@pytest.mark.parametrize("binding", ["instance", "class", "subclass"])
def test_custom_model_likelihoods_keep_fresh_evaluation(method, binding, monkeypatch):
    estimator = _estimator(method, max_iter=1)
    model = TwoParameterLogistic(1)
    shared = method in ("sobol", "halton")
    hook = "log_likelihood_batch" if shared else "log_likelihood"
    calls = 0

    def changing_likelihood(self, responses, theta):
        nonlocal calls
        calls += 1
        shape = (len(responses), len(theta)) if shared else (len(theta),)
        values = np.full(shape, -float(calls))
        values.setflags(write=False)
        return values

    if binding == "subclass":
        model.__class__ = type(
            "CustomLikelihood", (type(model),), {hook: changing_likelihood}
        )
    elif binding == "class":
        monkeypatch.setattr(type(model), hook, changing_likelihood)
    else:
        monkeypatch.setattr(model, hook, MethodType(changing_likelihood, model))
    monkeypatch.setattr(estimator, "_m_step_mc", lambda *args: None)
    result = estimator.fit(model, np.array([[0], [1], [-1]]))
    expected_calls = 23 if method in ("posterior", "stochastic") else 3
    assert calls == expected_calls
    assert result.log_likelihood == -3.0 * expected_calls


def test_qmcem_keeps_a_changed_base_sample_callback(monkeypatch):
    estimator = _estimator("sobol", max_iter=1)
    calls = 0
    original = MCEMEstimator._sample_log_likelihoods

    def counted(self, *args):
        nonlocal calls
        calls += 1
        return original(self, *args)

    monkeypatch.setattr(MCEMEstimator, "_sample_log_likelihoods", counted)
    monkeypatch.setattr(estimator, "_m_step_mc", lambda *args: None)
    estimator.fit(TwoParameterLogistic(1), np.array([[0], [1], [-1]]))
    assert calls == 2
