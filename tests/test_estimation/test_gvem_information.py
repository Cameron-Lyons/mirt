"""Independent curvature references and contracts for GVEM uncertainty."""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest

import mirt
from mirt.estimation import _gvem_information
from mirt.estimation.gvem import GVEMEstimator


def _state(kind="mirt"):
    rng = np.random.default_rng(427)
    factors = 3 if kind == "mirt" else 1
    model = (
        mirt.OneParameterLogistic(3)
        if kind == "1pl"
        else mirt.TwoParameterLogistic(3, n_factors=factors)
    )
    if kind != "1pl":
        a = rng.uniform(0.4, 1.6, (3, factors))
        model.set_parameters(discrimination=a[:, 0] if factors == 1 else a)
    model.set_parameters(difficulty=[-0.7, 0.3, 0.2])
    responses = rng.integers(-1, 2, (18, 6))[::2, ::2]
    responses[0] = -9
    responses[:, -1] = -1
    estimator = GVEMEstimator(use_gpu=False)
    estimator._mu = rng.normal(size=(18, factors))[::2]
    matrices = rng.normal(size=(9, factors, factors))
    estimator._sigma = matrices @ matrices.transpose(0, 2, 1) + np.eye(factors)
    estimator._xi = rng.uniform(0.0, 2.0, (18, 6))[::2, ::2]
    return model, responses, estimator


def _decimal_bound(model, responses, estimator, params):
    """Evaluate the expected logistic bound directly with 60-digit arithmetic."""
    D = Decimal.from_float
    a = params["discrimination"].reshape(model.n_items, model.n_factors)
    b = params["difficulty"]
    value = Decimal(0)
    for person, item in np.argwhere(responses >= 0):
        x = abs(D(float(estimator._xi[person, item])))
        exp_x = x.exp()
        lam = (
            Decimal("0.125")
            if x < Decimal("1e-6")
            else (exp_x - 1) / ((exp_x + 1) * 4 * x)
        )
        eta = sum(
            D(float(a[item, f]))
            * (D(float(estimator._mu[person, f])) - D(float(b[item])))
            for f in range(model.n_factors)
        )
        variance = sum(
            D(float(a[item, f]))
            * D(float(estimator._sigma[person, f, g]))
            * D(float(a[item, g]))
            for f in range(model.n_factors)
            for g in range(model.n_factors)
        )
        value += (
            Decimal(int(responses[person, item])) - Decimal("0.5")
        ) * eta - lam * (eta * eta + variance)
    return value


@pytest.mark.parametrize("kind", ["1pl", "2pl", "mirt"])
@pytest.mark.parametrize("budget", [1, 262_144])
def test_standard_errors_match_high_precision_bound(monkeypatch, kind, budget):
    monkeypatch.setattr(_gvem_information, "_MAX_INFORMATION_ELEMENTS", budget)
    model, responses, estimator = _state(kind)
    before = model.parameters
    arrays = [
        responses,
        estimator._mu,
        estimator._sigma,
        estimator._xi,
        *model._parameters.values(),
    ]
    snapshots = [array.copy() for array in arrays]
    for array in arrays:
        array.setflags(write=False)
    actual = estimator._compute_standard_errors(
        model, responses, np.zeros(model.n_factors), np.eye(model.n_factors)
    )
    with localcontext() as context:
        context.prec = 60
        center = _decimal_bound(model, responses, estimator, before)
        for name, values in before.items():
            for index in np.ndindex(values.shape):
                if not model.free_parameter_masks[name][index]:
                    assert actual[name][index] == 0.0
                    continue
                # Move by one representable float in each direction. Unequal
                # float spacings are handled by the divided-difference formula.
                plus = {k: v.copy() for k, v in before.items()}
                minus = {k: v.copy() for k, v in before.items()}
                plus[name][index] = np.nextafter(values[index], np.inf)
                minus[name][index] = np.nextafter(values[index], -np.inf)
                forward = Decimal.from_float(
                    float(plus[name][index])
                ) - Decimal.from_float(float(values[index]))
                backward = Decimal.from_float(
                    float(values[index])
                ) - Decimal.from_float(float(minus[name][index]))
                high = _decimal_bound(model, responses, estimator, plus)
                low = _decimal_bound(model, responses, estimator, minus)
                second = (
                    2
                    * ((high - center) / forward - (center - low) / backward)
                    / (forward + backward)
                )
                expected = float((-second).sqrt() ** -1) if second < 0 else np.nan
                np.testing.assert_allclose(
                    actual[name][index], expected, rtol=1e-12, atol=1e-13
                )
    for array, snapshot in zip(arrays, snapshots, strict=True):
        np.testing.assert_array_equal(array, snapshot)


def test_large_constant_in_bound_does_not_erase_curvature():
    model = mirt.TwoParameterLogistic(1).set_parameters(difficulty=[1e12])
    estimator = GVEMEstimator(use_gpu=False)
    estimator._mu = np.full((8, 1), 1e12)
    estimator._sigma = np.ones((8, 1, 1))
    estimator._xi = np.ones((8, 1))
    actual = estimator._compute_standard_errors(
        model, np.zeros((8, 1), dtype=int), np.zeros(1), np.eye(1)
    )
    expected = 1.0 / np.sqrt(4 * np.tanh(0.5))
    np.testing.assert_allclose(actual["discrimination"], expected, rtol=1e-14)
    np.testing.assert_allclose(actual["difficulty"], expected, rtol=1e-14)


@pytest.mark.parametrize("slope", [1e-200, 1e200])
def test_difficulty_standard_error_does_not_square_extreme_loading(slope):
    model = mirt.TwoParameterLogistic(1).set_parameters(discrimination=[slope])
    estimator = GVEMEstimator(use_gpu=False)
    estimator._mu = np.zeros((1, 1))
    estimator._sigma = np.ones((1, 1, 1))
    estimator._xi = np.zeros((1, 1))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        actual = estimator._compute_standard_errors(
            model, np.ones((1, 1), dtype=int), np.zeros(1), np.eye(1)
        )
    np.testing.assert_allclose(actual["difficulty"], 2.0 / slope, rtol=1e-14, atol=0.0)


def test_cancelling_loadings_leave_difficulty_unidentified():
    model = mirt.TwoParameterLogistic(1, n_factors=2).set_parameters(
        discrimination=[[1.0, -1.0]]
    )
    estimator = GVEMEstimator(use_gpu=False)
    estimator._mu = np.zeros((4, 2))
    estimator._sigma = np.broadcast_to(np.eye(2), (4, 2, 2))
    estimator._xi = np.ones((4, 1))
    actual = estimator._compute_standard_errors(
        model, np.ones((4, 1), dtype=int), np.zeros(2), np.eye(2)
    )
    assert np.isnan(actual["difficulty"][0])
    assert np.all(np.isfinite(actual["discrimination"]))


def test_curvature_lambda_work_is_bounded(monkeypatch):
    monkeypatch.setattr(_gvem_information, "_MAX_INFORMATION_ELEMENTS", 30)
    model, responses, estimator = _state()
    seen = []

    def bounded_lambda(xi):
        assert xi.ndim == 1
        seen.append(xi.size)
        return estimator._lambda(xi)

    _gvem_information.gvem_standard_errors(
        model, responses, estimator._mu, estimator._sigma, estimator._xi, bounded_lambda
    )
    assert max(seen) <= 30 // (3 * model.n_factors + 5)
    assert sum(seen) == np.count_nonzero(responses >= 0)


@pytest.mark.parametrize("override", ["instance", "subclass", "python", "step"])
def test_custom_objectives_and_steps_keep_numerical_fallback(monkeypatch, override):
    model, responses, estimator = _state()
    calls = []

    def quadratic(self, model, responses, prior_mean, prior_cov):
        calls.append((prior_mean.copy(), prior_cov.copy()))
        return -sum(float(np.sum(values**2)) for values in model.parameters.values())

    if override == "subclass":

        class CustomEstimator(GVEMEstimator):
            _compute_elbo = quadratic

        estimator = CustomEstimator(use_gpu=False)
    elif override == "python":
        monkeypatch.setattr(
            estimator, "_compute_elbo_python", quadratic.__get__(estimator)
        )
        import mirt.estimation.gvem as module

        monkeypatch.setattr(module, "_rust_gvem_compute_elbo", lambda *args: None)
    elif override == "step":
        estimator.se_step_size = 1e-3
        # Record objective calls without changing the estimator's methods.
        import mirt.estimation.gvem as module

        def record(*args):
            calls.append(args)
            return None

        monkeypatch.setattr(module, "_rust_gvem_compute_elbo", record)
    else:
        monkeypatch.setattr(estimator, "_compute_elbo", quadratic.__get__(estimator))

    mean, cov = np.array([0.2, -0.3, 0.1]), np.eye(3)
    result = estimator._compute_standard_errors(model, responses, mean, cov)
    assert len(calls) == 2 * model.n_parameters + 1
    if override != "step":
        for value in result.values():
            np.testing.assert_allclose(value, np.sqrt(0.5), rtol=1e-4)
        assert all(np.array_equal(m, mean) and np.array_equal(c, cov) for m, c in calls)


@pytest.mark.parametrize("kind", ["2pl", "mirt"])
def test_numerical_failure_restores_parameters_and_internal_predictors(
    monkeypatch, kind
):
    model, responses, estimator = _state(kind)
    before = model.parameters
    estimator._convert_to_slope_intercept(model)
    old_slopes, old_intercepts = estimator._slopes.copy(), estimator._intercepts.copy()
    calls = 0

    def broken(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("trial objective failed")
        return -sum(float(np.sum(v**2)) for v in model.parameters.values())

    monkeypatch.setattr(estimator, "_compute_elbo", broken)
    with pytest.raises(RuntimeError, match="trial objective failed"):
        estimator._compute_standard_errors(
            model, responses, np.zeros(model.n_factors), np.eye(model.n_factors)
        )
    for name, values in before.items():
        np.testing.assert_array_equal(model.parameters[name], values)
    np.testing.assert_array_equal(estimator._slopes, old_slopes)
    np.testing.assert_array_equal(estimator._intercepts, old_intercepts)


@pytest.mark.parametrize("backend", ["numpy", "rust"])
@pytest.mark.parametrize("factors", [1, 3])
def test_complete_fit_matches_numerical_uncertainty(backend, factors):
    if backend == "rust" and not mirt.is_rust_available():
        pytest.skip("native backend unavailable")
    previous = mirt.get_backend()
    mirt.set_backend(backend)
    try:
        data = mirt.simdata(n_persons=100, n_items=5, n_factors=factors, seed=4)
        data[::7, 0] = -1
        options = dict(max_iter=3, use_gpu=False)
        analytic = GVEMEstimator(**options).fit(
            mirt.TwoParameterLogistic(5, n_factors=factors), data
        )
        numerical = GVEMEstimator(**options, se_step_size=1e-3).fit(
            mirt.TwoParameterLogistic(5, n_factors=factors), data
        )
        assert analytic.log_likelihood == numerical.log_likelihood
        assert analytic.n_iterations == numerical.n_iterations
        for name, values in analytic.model.parameters.items():
            np.testing.assert_array_equal(values, numerical.model.parameters[name])
            np.testing.assert_allclose(
                analytic.standard_errors[name],
                numerical.standard_errors[name],
                rtol=1e-5,
                atol=1e-6,
            )
    finally:
        mirt.set_backend(previous)
