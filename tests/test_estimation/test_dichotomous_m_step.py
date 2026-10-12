"""Check EM gradients against the public item likelihood."""

from copy import deepcopy
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import expit, xlog1py, xlogy

import mirt
from mirt.estimation import em as em_module
from mirt.estimation._dichotomous_objective import (
    guessing_items_objective,
    prepare_dichotomous_objective,
    prepare_logistic_objective,
)
from mirt.estimation.em import EMEstimator
from mirt.estimation.latent_density import GaussianDensity
from mirt.estimation.priors import NormalPrior
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.models.dichotomous import (
    FourParameterLogistic,
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)

_KINDS = ["1PL", "2PL", "2PL-multi", "3PL", "4PL"]


def _model(kind):
    if kind == "1PL":
        return OneParameterLogistic(1).set_parameters(difficulty=np.array([0.25]))
    if kind == "2PL-multi":
        return TwoParameterLogistic(1, n_factors=3).set_parameters(
            discrimination=np.array([[0.8, 1.2, 0.6]]), difficulty=np.array([0.25])
        )
    cls = {
        "2PL": TwoParameterLogistic,
        "3PL": ThreeParameterLogistic,
        "4PL": FourParameterLogistic,
    }[kind]
    params = dict(discrimination=np.array([1.2]), difficulty=np.array([0.25]))
    if kind in ("3PL", "4PL"):
        params["guessing"] = np.array([0.05])
    if kind == "4PL":
        params["upper"] = np.array([0.95])
    return cls(1).set_parameters(**params)


def _public_objective(model, estimator, points, observed, correct, params):
    working = deepcopy(model)
    estimator._set_item_params(working, 0, params)
    p = np.clip(
        working.probability(points, 0),
        estimator.prob_epsilon,
        1 - estimator.prob_epsilon,
    )
    return -np.sum(xlogy(correct, p) + xlog1py(observed - correct, -p))


def _kernel_objective(model, estimator, points, observed, correct, params):
    """The public likelihood, clipped only near a free asymptote."""
    working = deepcopy(model)
    estimator._set_item_params(working, 0, params)
    p = working.probability(points, 0)
    if "guessing" in working.parameters:
        p = np.maximum(p, estimator.prob_epsilon)
    if "upper" in working.parameters:
        p = np.minimum(p, 1 - estimator.prob_epsilon)
    return -np.sum(xlogy(correct, p) + xlog1py(observed - correct, -p))


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("epsilon", [1e-10, 0.1])
@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dichotomous_gradient_matches_public_likelihood(kind, epsilon, bounded, dtype):
    model = _model(kind)
    original = model.parameters
    rng = np.random.default_rng(914)
    points = (rng.normal(size=(41, model.n_factors)) * 3).astype(dtype)
    observed = rng.uniform(1, 10, 41)
    correct = observed * rng.random(41)
    observed[::5] = correct[::5] = 0
    points.flags.writeable = observed.flags.writeable = correct.flags.writeable = False
    estimator = EMEstimator(prob_epsilon=epsilon)
    params, bounds = estimator._get_item_params_and_bounds(model, 0)
    objective = prepare_dichotomous_objective(
        model, 0, points, observed, correct, epsilon, bounds if bounded else None
    )
    assert objective is not None
    actual, gradient = objective(params)
    # Nothing saturates at the default clip, where the value is unchanged.
    # The wide clip compares exact tails with a reference that loses
    # precision as probabilities approach one.
    expected = _kernel_objective(model, estimator, points, observed, correct, params)
    np.testing.assert_allclose(
        actual, expected, rtol=1e-12 if epsilon < 1e-6 else 1e-10
    )
    if epsilon < 1e-6:
        clipped = _public_objective(model, estimator, points, observed, correct, params)
        assert actual == pytest.approx(clipped, rel=1e-12)
    numerical = np.empty_like(params)
    for index in range(params.size):
        offset = np.zeros_like(params)
        offset[index] = 1e-5
        high = _kernel_objective(
            model, estimator, points, observed, correct, params + offset
        )
        low = _kernel_objective(
            model, estimator, points, observed, correct, params - offset
        )
        numerical[index] = (high - low) / 2e-5
    np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=3e-7)
    for name, value in original.items():
        np.testing.assert_array_equal(model.parameters[name], value)


@pytest.mark.parametrize("kind", _KINDS)
def test_prepared_dichotomous_optimizer_matches_numerical_likelihood(kind):
    model = _model(kind)
    rng = np.random.default_rng(621)
    points = rng.normal(size=(53, model.n_factors)) * 2.0
    observed = rng.uniform(1.0, 15.0, len(points))
    correct = observed * model.probability(points, 0)
    estimator = EMEstimator()
    params, bounds = estimator._get_item_params_and_bounds(model, 0)
    initial = params + 0.04
    if kind == "4PL":
        initial[-1] = params[-1] - 0.04
    objective = prepare_dichotomous_objective(
        model, 0, points, observed, correct, estimator.prob_epsilon, bounds
    )
    assert objective is not None

    def numerical(trial):
        return _public_objective(model, estimator, points, observed, correct, trial)

    options = dict(
        method="L-BFGS-B", bounds=bounds, options={"ftol": 1e-12, "gtol": 1e-8}
    )
    analytic_fit = minimize(objective, initial, jac=True, **options)
    numeric_fit = minimize(numerical, initial, **options)
    np.testing.assert_allclose(analytic_fit.fun, numeric_fit.fun, rtol=1e-10)
    np.testing.assert_allclose(analytic_fit.x, numeric_fit.x, rtol=2e-4, atol=2e-5)
    assert analytic_fit.nfev < numeric_fit.nfev


def _saturated_tails(model):
    points = np.zeros((4, model.n_factors))
    points[:, 0] = [-1000, -30, 30, 1000]
    observed = np.array([3.0, 5.0, 2.0, 4.0])
    correct = np.array([2.0, 4.0, 1.0, 2.0])
    estimator = EMEstimator(prob_epsilon=1e-8)
    params, _ = estimator._get_item_params_and_bounds(model, 0)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        loss, gradient = estimator._neg_expected_loglik_with_grad_dichotomous(
            model, 0, points, observed, correct, params
        )
    return estimator, points, observed, correct, params, loss, gradient


@pytest.mark.parametrize("kind", ["1PL", "2PL", "2PL-multi"])
def test_saturated_logistic_tails_keep_the_exact_likelihood(kind):
    # Regression: clipped probabilities left these tails flat, so the
    # itemwise optimizer stopped on items whose expected responses saturate.
    model = _model(kind)
    _, points, observed, correct, _, loss, gradient = _saturated_tails(model)
    slopes = model.parameters["discrimination"].reshape(-1)
    location = model.parameters["difficulty"][0]
    logits = points @ slopes - slopes.sum() * location
    expected = np.sum(
        correct * np.logaddexp(0.0, -logits)
        + (observed - correct) * np.logaddexp(0.0, logits)
    )
    np.testing.assert_allclose(loss, expected, rtol=1e-13)
    residual = observed * expit(logits) - correct
    expected_gradient = [-slopes.sum() * residual.sum()]
    if kind != "1PL":
        expected_gradient = [*((points - location).T @ residual), *expected_gradient]
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-13)


@pytest.mark.parametrize("kind", ["3PL", "4PL"])
def test_guessing_tails_clip_only_near_free_asymptotes(kind):
    # The exact derivative in a free asymptote is 1/p beyond the clip, which
    # L-BFGS-B cannot step against, so only the 3PL's fixed one stays exact.
    model = _model(kind).set_parameters(guessing=np.array([0.0]))
    if kind == "4PL":
        model.set_parameters(upper=np.array([1.0]))
    estimator, points, observed, correct, params, loss, gradient = _saturated_tails(
        model
    )
    clipped = _public_objective(model, estimator, points, observed, correct, params)
    if kind == "4PL":
        np.testing.assert_allclose(loss, clipped, rtol=1e-14)
        np.testing.assert_array_equal(gradient, 0.0)
        return
    # Only the node at 30 approaches the upper asymptote with a representable
    # failure probability; the one at 1000 underflows and keeps the clip.
    logit = 1.2 * (30.0 - 0.25)
    failure = expit(-logit)
    exact = -(
        correct[2] * np.log1p(-failure) + (observed[2] - correct[2]) * np.log(failure)
    )
    high = 1.0 - estimator.prob_epsilon
    clipped_node = -(xlogy(correct[2], high) + xlog1py(observed[2] - correct[2], -high))
    np.testing.assert_allclose(loss, clipped - clipped_node + exact, rtol=1e-14)
    score = (observed[2] - correct[2]) / failure - correct[2] / (1.0 - failure)
    common = score * failure * (1.0 - failure)
    np.testing.assert_allclose(
        gradient, [common * (30.0 - 0.25), -1.2 * common, failure * score], rtol=1e-12
    )


@pytest.mark.parametrize("observed", [0.0, 3.0])
@pytest.mark.parametrize("kind", _KINDS)
def test_saturated_probabilities_and_zero_counts_do_not_produce_nan(kind, observed):
    model = _model(kind)
    if "upper" in model.parameters:
        model.set_parameters(upper=np.array([1.0]))
    estimator = EMEstimator(prob_epsilon=1e-20)
    params, _ = estimator._get_item_params_and_bounds(model, 0)
    points = np.full((1, model.n_factors), 1000.0)
    counts = np.array([observed])
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        loss, gradient = estimator._neg_expected_loglik_with_grad_dichotomous(
            model, 0, points, counts, counts, params
        )
    assert loss == 0.0
    np.testing.assert_array_equal(gradient, 0.0)


def test_itemwise_3pl_m_step_leaves_saturated_easy_starts():
    # Regression: a 3PL item started where every incorrect response
    # saturates had a flat clipped objective, and stopped there. Its upper
    # asymptote is fixed, so that tail keeps the exact likelihood.
    rng = np.random.default_rng(5)
    responses = rng.integers(0, 2, (356, 3))
    responses[:, 0] = 1
    responses[rng.choice(356, 6, replace=False), 0] = 0
    model = ThreeParameterLogistic(3)
    estimator = EMEstimator(use_rust=False, use_gpu=False)
    estimator._quadrature = GaussHermiteQuadrature(41)
    estimator._latent_density = GaussianDensity()
    posterior, _ = estimator._e_step(model, responses)
    points = estimator._quadrature.nodes
    optimum = estimator._optimize_item_params(model, 0, responses, posterior, points)
    start = deepcopy(model)
    estimator._set_item_params(start, 0, np.array([5.0, -30.0, 0.2]))
    estimate = estimator._optimize_item_params(start, 0, responses, posterior, points)
    _, _, objective, _ = estimator._item_objective(
        model, 0, responses, posterior, points
    )
    assert objective(estimate)[0] == pytest.approx(objective(optimum)[0], rel=1e-6)


def test_batched_3pl_objective_matches_the_itemwise_one():
    theta = np.linspace(-6.0, 6.0, 41)
    rng = np.random.default_rng(12)
    observed = rng.uniform(1.0, 10.0, (3, 41))
    correct = observed * rng.random((3, 41))
    # Items saturating toward the fixed upper asymptote, toward the free
    # guessing one, and neither.
    x = np.array([[4.0, -9.0, 0.2], [3.0, 8.0, 0.0], [1.0, 0.0, 0.1]])
    batched = guessing_items_objective(theta, correct, observed, 1e-10)
    loss, gradient, information = batched(np.arange(3), x, True)
    np.testing.assert_array_equal(batched(np.arange(3), x, False)[0], loss)
    assert np.isfinite(information).all()
    for item in range(3):
        itemwise = prepare_logistic_objective(
            theta[:, None], observed[item], correct[item], 1e-10, has_guessing=True
        )
        value, item_gradient = itemwise(x[item])
        assert loss[item] == pytest.approx(value, rel=1e-14)
        np.testing.assert_allclose(gradient[item], item_gradient, rtol=1e-12)


def test_four_parameter_positive_tail_retains_representable_gradient():
    model = FourParameterLogistic(1).set_parameters(
        guessing=np.array([0.2]), upper=np.array([0.9])
    )
    estimator = EMEstimator()
    params, _ = estimator._get_item_params_and_bounds(model, 0)
    _, gradient = estimator._neg_expected_loglik_with_grad_dichotomous(
        model, 0, np.array([[40.0]]), np.array([1.0]), np.array([0.0]), params
    )
    # The upper asymptote is interior, so clipping does not erase this tail.
    tail = np.exp(-40.0) / (1 + np.exp(-40.0))
    score = 1 / (1 - 0.9)
    common = score * 0.7 * tail * (1 - tail)
    np.testing.assert_allclose(
        gradient, [40 * common, -common, score * tail, score * (1 - tail)], rtol=1e-14
    )


@pytest.mark.parametrize("bounded", [False, True])
def test_multidimensional_item_objective_recovers_finite_cancelled_logits(bounded):
    model = TwoParameterLogistic(1, n_factors=2).set_parameters(
        discrimination=np.array([[5.0, 5.0]]), difficulty=np.array([0.3])
    )
    estimator = EMEstimator()
    params, bounds = estimator._get_item_params_and_bounds(model, 0)
    objective = prepare_dichotomous_objective(
        model,
        0,
        np.array([[1e308, -1e308]]),
        np.array([1.0]),
        np.array([0.0]),
        estimator.prob_epsilon,
        bounds if bounded else None,
    )
    assert objective is not None
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        loss, gradient = objective(params)
    # The exact centered dot product is 5 * (-0.3) + 5 * (-0.3).
    probability = 1 / (1 + np.exp(3.0))
    np.testing.assert_allclose(loss, -np.log1p(-probability), rtol=1e-14)
    np.testing.assert_allclose(
        gradient,
        [probability * 1e308, -probability * 1e308, -10 * probability],
        rtol=1e-14,
    )


@pytest.mark.parametrize(
    "change", ["subclass", "instance", "parameter_order", "private_theta"]
)
def test_custom_dichotomous_models_use_their_public_probability(monkeypatch, change):
    class PowerModel(TwoParameterLogistic):
        def probability(self, theta, item_idx=None):
            return super().probability(theta, item_idx) ** 2

    model = PowerModel(1) if change == "subclass" else TwoParameterLogistic(1)
    if change == "instance":
        model.probability = lambda theta, item_idx=None: np.full(len(theta), 0.37)
    elif change == "parameter_order":
        model._parameters = dict(reversed(list(model._parameters.items())))
    elif change == "private_theta":

        def transform(self, theta):
            return TwoParameterLogistic._ensure_theta_2d(self, theta) * 1.2 + 0.8

        model._ensure_theta_2d = MethodType(transform, model)
    estimator = EMEstimator(use_rust=False, use_gpu=False)
    points = np.array([[-2.0], [-0.5], [1.0], [3.0]])
    responses = np.array([[0], [1], [-1]])
    posterior = np.full((3, 4), 0.25)

    def minimize(objective, x0, *, jac=False, **kwargs):
        trial = x0 + 0.2
        result = objective(trial)
        value = result[0] if jac else result
        expected = _public_objective(
            model, estimator, points, np.full(4, 0.5), np.full(4, 0.25), trial
        )
        np.testing.assert_allclose(value, expected, rtol=1e-13)
        return SimpleNamespace(x=trial)

    monkeypatch.setattr(em_module, "minimize", minimize)
    estimator._optimize_item_params(model, 0, responses, posterior, points)


@pytest.fixture(scope="module")
def responses_3pl():
    rng = np.random.default_rng(21)
    n_persons, n_items = 1000, 12
    theta = rng.standard_normal(n_persons)
    a = rng.uniform(0.8, 2.0, n_items)
    b = rng.normal(0.0, 1.0, n_items)
    c = rng.uniform(0.05, 0.25, n_items)
    probability = c + (1 - c) / (1 + np.exp(-a * (theta[:, None] - b)))
    responses = (rng.random(probability.shape) < probability).astype(int)
    responses[rng.random(responses.shape) < 0.03] = -1
    return responses


@pytest.fixture(scope="module")
def native_3pl(responses_3pl):
    return EMEstimator(compute_standard_errors=False).fit(
        ThreeParameterLogistic(12), responses_3pl
    )


def _assert_same_fit(result, reference):
    assert result.converged
    assert result.log_likelihood == pytest.approx(reference.log_likelihood, abs=1e-3)
    assert result.n_iterations <= reference.n_iterations + 5
    for name, values in reference.model.parameters.items():
        np.testing.assert_allclose(result.model.parameters[name], values, atol=1e-2)


@pytest.mark.parametrize(
    "options",
    [
        {"n_jobs": 2},
        {"use_rust": False},
        {"item_priors": {"difficulty": NormalPrior(0.0, 1e4)}},
    ],
    ids=["threads", "numpy", "flat-prior"],
)
def test_generic_3pl_m_steps_reach_the_native_optimum(
    responses_3pl, native_3pl, options
):
    # Loosely solved 3PL items jittered between M-steps, which stopped EM
    # hundreds of iterations later at estimates up to 0.7 away.
    result = EMEstimator(compute_standard_errors=False, **options).fit(
        ThreeParameterLogistic(12), responses_3pl
    )
    _assert_same_fit(result, native_3pl)


def test_coordinate_fixed_at_its_estimate_reproduces_the_fit(responses_3pl, native_3pl):
    fixed = np.zeros(12, dtype=bool)
    fixed[0] = True
    guessing = native_3pl.model.parameters["guessing"]
    result = mirt.fit_mirt(
        responses_3pl,
        model="3PL",
        compute_standard_errors=False,
        start_values={"guessing": guessing},
        fixed={"guessing": fixed},
    )
    assert result.model.parameters["guessing"][0] == guessing[0]
    _assert_same_fit(result, native_3pl)


def _item_tolerances(monkeypatch, model, estimator):
    tolerances = []

    def minimize(objective, x0, **kwargs):
        tolerances.append(kwargs["options"]["ftol"])
        return SimpleNamespace(x=x0)

    monkeypatch.setattr(em_module, "minimize", minimize)
    points = np.array([[-1.0], [0.0], [1.0]])
    responses = np.array([[0], [1], [1]])
    estimator._optimize_item_params(model, 0, responses, np.full((3, 3), 1 / 3), points)
    return tolerances


def test_analytic_dichotomous_items_use_the_precise_tolerance(monkeypatch):
    class Custom(ThreeParameterLogistic):
        def probability(self, theta, item_idx=None):
            return super().probability(theta, item_idx)

    estimator = EMEstimator(item_optim_ftol=1e-6)
    restricted = ThreeParameterLogistic(1)
    restricted.set_free_parameter_masks({"guessing": np.array([False])})
    for model in (ThreeParameterLogistic(1), restricted):
        assert _item_tolerances(monkeypatch, model, estimator) == [1e-10]
    # Numerical objectives keep the configured tolerance.
    assert _item_tolerances(monkeypatch, Custom(1), estimator) == [1e-6]


def test_fixed_coordinates_keep_the_analytic_item_objective():
    values = {
        "discrimination": np.array([1.3, 0.9]),
        "difficulty": np.array([0.2, -0.4]),
        "guessing": np.array([0.15, 0.1]),
    }
    free = {"difficulty": np.array([True, False]), "guessing": np.array([False, True])}
    model = ThreeParameterLogistic(2).set_parameters(**values)
    model.set_free_parameter_masks(free)
    full = ThreeParameterLogistic(2).set_parameters(**values)
    rng = np.random.default_rng(7)
    points = np.linspace(-3.0, 3.0, 9)[:, None]
    observed = rng.uniform(5.0, 20.0, 9)
    correct = observed * rng.uniform(0.1, 0.9, 9)
    responses = np.zeros((1, 2), dtype=int)
    estimator = EMEstimator()
    for item in range(2):
        start, _, objective, analytic = estimator._item_objective(
            model, item, responses, None, points, r_k=correct, n_k_valid=observed
        )
        assert analytic and start.size == 2
        trial = start + 0.05
        value, gradient = objective(trial)
        # The unrestricted objective at the same point, with the fixed
        # coordinate at its value.
        mask = np.array([True, free["difficulty"][item], free["guessing"][item]])
        point = np.array([values[name][item] for name in values])
        point[mask] = trial
        _, _, unrestricted, _ = estimator._item_objective(
            full, item, responses, None, points, r_k=correct, n_k_valid=observed
        )
        expected_value, expected_gradient = unrestricted(point)
        assert value == pytest.approx(expected_value, rel=1e-14)
        np.testing.assert_allclose(gradient, expected_gradient[mask], rtol=1e-14)
