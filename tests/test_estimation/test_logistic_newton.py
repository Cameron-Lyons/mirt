"""The batched Newton M-step for built-in 1PL, 2PL and MIRT items."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.optimize import minimize

from mirt.estimation._em_context import EMFitContext
from mirt.estimation._logistic_newton import newton_logistic_items
from mirt.estimation.em import EMEstimator
from mirt.estimation.latent_density import GaussianDensity
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.models.dichotomous import OneParameterLogistic, TwoParameterLogistic
from mirt.models.multidimensional import MultidimensionalModel


class _ItemwiseEM(EMEstimator):
    """Reference estimator: overriding an item hook keeps the itemwise scipy path."""

    def _optimize_item_params(self, *args, **kwargs):
        return super()._optimize_item_params(*args, **kwargs)


def _responses(n_items, n_factors=1, seed=0, n_persons=400):
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(n_persons, n_factors))
    slopes = rng.uniform(0.6, 2.0, (n_items, n_factors))
    locations = rng.normal(0.0, 0.8, n_items)
    logits = theta @ slopes.T - slopes.sum(axis=1) * locations
    probability = 1.0 / (1.0 + np.exp(-logits))
    responses = (rng.random(logits.shape) < probability).astype(np.int_)
    responses[rng.random(responses.shape) < 0.1] = -1
    return responses


def _model(kind, n_items):
    if kind == "1PL":
        model = OneParameterLogistic(n_items)
        return model.set_parameters(difficulty=np.linspace(-0.5, 0.5, n_items))
    model = TwoParameterLogistic(n_items, n_factors=int(kind[-1]))
    return model.set_parameters(
        discrimination=np.full(model.parameters["discrimination"].shape, 0.9),
        difficulty=np.linspace(-0.5, 0.5, n_items),
    )


def _prepare(estimator, model, responses, n_quadpts, compress=False):
    """Install quadrature and density, returning the context and posterior."""
    estimator._quadrature = GaussHermiteQuadrature(n_quadpts, model.n_factors)
    estimator._latent_density = GaussianDensity(n_dimensions=model.n_factors)
    context = EMFitContext(responses, compress=compress)
    posterior, _ = estimator._e_step(model, context.responses)
    if context.frequencies is not None:
        posterior = posterior * context.frequencies[:, None]
    return context, posterior


def _itemwise_reference(model, responses, posterior, quadrature):
    reference = deepcopy(model)
    tight = _ItemwiseEM(item_optim_ftol=1e-15, item_optim_maxiter=2000)
    tight._quadrature = quadrature
    assert not tight._uses_newton_logistic_m_step(reference)
    tight._m_step(reference, responses, posterior)
    return reference


@pytest.mark.parametrize("kind", ["1PL", "2PL-1", "2PL-2", "2PL-3"])
@pytest.mark.parametrize("compress", [False, True])
def test_newton_m_step_matches_tight_itemwise_optimizer(kind, compress):
    model = _model(kind, 6)
    responses = _responses(6, model.n_factors, seed=11)
    # Repeated patterns exercise frequency-weighted posteriors.
    responses = np.vstack([responses, responses[:150]])
    estimator = EMEstimator(use_gpu=False)
    n_quadpts = 7 if model.n_factors > 1 else 15
    context, posterior = _prepare(estimator, model, responses, n_quadpts, compress)
    reference = _itemwise_reference(
        model, context.responses, posterior, estimator._quadrature
    )

    assert estimator._uses_newton_logistic_m_step(model)
    estimator._fit_context = context
    estimator._m_step(model, context.responses, posterior)

    for name, values in reference.parameters.items():
        np.testing.assert_allclose(model.parameters[name], values, atol=2e-6)


def test_boundary_and_unobserved_items_keep_itemwise_semantics(monkeypatch):
    rng = np.random.default_rng(3)
    theta = rng.normal(size=600)
    slopes = np.array([1.2, 1.0, -1.5, 1.0, 1.4, 0.8, 1.1])
    probability = 1.0 / (1.0 + np.exp(-slopes * (theta[:, None] - 0.2)))
    responses = (rng.random(probability.shape) < probability).astype(np.int_)
    responses[rng.random(responses.shape) < 0.1] = -1
    responses[:, 1] = np.where(responses[:, 1] >= 0, 1, -1)
    responses[:, 3] = -1
    # Informative starting values make the reversed item's update leave the box.
    model = TwoParameterLogistic(7).set_parameters(
        discrimination=np.where(slopes > 0, 1.5 * slopes, 0.2),
        difficulty=np.full(7, 0.2),
    )
    estimator = EMEstimator(
        item_optim_ftol=1e-15, item_optim_maxiter=2000, use_gpu=False
    )
    _, posterior = _prepare(estimator, model, responses, 15)
    reference = _itemwise_reference(model, responses, posterior, estimator._quadrature)
    before = model.parameters

    fallback = []
    optimize = EMEstimator._optimize_item

    def record(self, model, item_idx, *args, **kwargs):
        fallback.append(item_idx)
        return optimize(self, model, item_idx, *args, **kwargs)

    monkeypatch.setattr(EMEstimator, "_optimize_item", record)
    assert estimator._uses_newton_logistic_m_step(model)
    estimator._m_step(model, responses, posterior)

    # The all-correct item's maximum lies at infinity and the reversed item
    # needs a slope below the lower bound. The unobserved item never moves.
    assert sorted(fallback) == [1, 2]
    assert model.parameters["discrimination"][2] == pytest.approx(0.1)
    for name, values in before.items():
        assert model.parameters[name][3] == values[3]
    for name, values in reference.parameters.items():
        np.testing.assert_allclose(model.parameters[name], values, atol=1e-6)


@pytest.mark.parametrize(
    ("kind", "start"),
    [
        ("2PL-1", [8.0, 8.0]),
        ("2PL-2", [6.0, -6.0, 5.0]),
        ("1PL", [30.0]),
        ("mirt", [0.5, 0.5, -30.0]),
    ],
)
def test_itemwise_m_step_leaves_saturated_starts(kind, start):
    # Regression: the clipped item objective was flat where every expected
    # response of an item saturated, so the itemwise optimizer (also the
    # fallback of unconverged Newton steps) stopped at such a start. For an
    # item with 6 of 356 correct responses that left 138 against about 29.5.
    model = MultidimensionalModel(4, 2) if kind == "mirt" else _model(kind, 4)
    responses = _responses(4, model.n_factors, seed=5, n_persons=356)
    responses[:, 0] = 0
    responses[np.random.default_rng(5).choice(356, 6, replace=False), 0] = 1
    estimator = EMEstimator(use_gpu=False)
    _, posterior = _prepare(estimator, model, responses, 21 if kind[-1] == "2" else 41)
    reference = deepcopy(model)
    estimator._m_step(reference, responses, posterior)
    optimum, _ = estimator._get_item_params_and_bounds(reference, 0)

    itemwise = _ItemwiseEM(use_gpu=False)
    itemwise._quadrature = estimator._quadrature
    itemwise._set_item_params(model, 0, np.array(start))
    _, _, objective, _ = itemwise._item_objective(
        model, 0, responses, posterior, estimator._quadrature.nodes
    )
    estimate = itemwise._optimize_item_params(
        model, 0, responses, posterior, estimator._quadrature.nodes
    )
    assert objective(estimate)[0] == pytest.approx(objective(optimum)[0], rel=1e-8)
    np.testing.assert_allclose(estimate, optimum, rtol=1e-3, atol=1e-3)


def test_newton_m_step_is_reserved_for_builtin_logistic_items():
    estimator = EMEstimator()
    model = TwoParameterLogistic(3)
    assert estimator._uses_newton_logistic_m_step(model)

    # Fixed coordinates stay in the batched step.
    restricted = TwoParameterLogistic(3).set_free_parameter_masks(
        {"discrimination": np.array([True, False, True])}
    )
    assert estimator._uses_newton_logistic_m_step(restricted)
    assert estimator._uses_newton_logistic_m_step(MultidimensionalModel(3, 2))

    class Custom(TwoParameterLogistic):
        def probability(self, theta, item_idx=None):
            return super().probability(theta, item_idx) ** 2

    assert not estimator._uses_newton_logistic_m_step(Custom(3))
    assert not EMEstimator(prob_epsilon=0.01)._uses_newton_logistic_m_step(model)
    assert not _ItemwiseEM()._uses_newton_logistic_m_step(model)
    patched = EMEstimator()
    patched._set_item_params = lambda *args: None
    assert not patched._uses_newton_logistic_m_step(model)


def _scipy_logistic(points, correct, observed, start, offset=None):
    """Minimize one item's unclipped binomial objective with scipy."""
    design = np.column_stack((points, np.ones(len(points))))
    if offset is not None:
        design = design[:, -1:]
    base = np.zeros(len(points)) if offset is None else offset

    def objective(x):
        z = base + design @ x
        loss = np.sum(
            correct * np.logaddexp(0, -z) + (observed - correct) * np.logaddexp(0, z)
        )
        p = 1.0 / (1.0 + np.exp(-z))
        return loss, design.T @ (observed * p - correct)

    result = minimize(
        objective,
        start,
        jac=True,
        method="BFGS",
        options={"gtol": 1e-11, "maxiter": 10_000},
    )
    return result.x


@pytest.mark.parametrize("n_factors", [1, 2])
def test_newton_solver_matches_unconstrained_optimum(n_factors):
    rng = np.random.default_rng(42)
    points = GaussHermiteQuadrature(9, n_factors).nodes
    observed = rng.uniform(5.0, 50.0, (5, len(points)))
    truth = rng.normal(0.0, 1.0, (5, n_factors + 1))
    probability = 1.0 / (1.0 + np.exp(-(truth[:, :-1] @ points.T + truth[:, -1:])))
    correct = observed * np.clip(
        probability + rng.normal(0, 0.05, probability.shape), 0.01, 0.99
    )
    slopes, intercepts, converged = newton_logistic_items(
        points, correct, observed, np.ones((5, n_factors)), np.zeros(5)
    )
    assert converged.all()
    for item in range(5):
        expected = _scipy_logistic(
            points, correct[item], observed[item], np.zeros(n_factors + 1)
        )
        np.testing.assert_allclose(slopes[item], expected[:-1], atol=1e-7)
        assert intercepts[item] == pytest.approx(expected[-1], abs=1e-7)

    fixed = np.full((5, n_factors), 0.7)
    kept, intercepts, converged = newton_logistic_items(
        points, correct, observed, fixed, np.zeros(5), estimate_slopes=False
    )
    assert converged.all()
    np.testing.assert_array_equal(kept, fixed)
    for item in range(5):
        expected = _scipy_logistic(
            points,
            correct[item],
            observed[item],
            np.zeros(1),
            offset=points @ fixed[item],
        )
        assert intercepts[item] == pytest.approx(expected[0], abs=1e-7)


def test_newton_solver_flags_items_without_finite_maximum():
    points = GaussHermiteQuadrature(7).nodes
    observed = np.full((2, 7), 10.0)
    correct = np.vstack((observed[0], 0.5 * observed[1]))
    _, intercepts, converged = newton_logistic_items(
        points, correct, observed, np.ones((2, 1)), np.zeros(2)
    )
    np.testing.assert_array_equal(converged, [False, True])
    assert np.isfinite(intercepts).all()


@pytest.mark.parametrize("kind", ["1PL", "2PL-1", "2PL-2"])
def test_newton_em_fits_reach_the_itemwise_fixed_point(kind):
    responses = _responses(5, int(kind[-1]) if kind != "1PL" else 1, seed=7)
    options = dict(n_quadpts=7, tol=1e-9, max_iter=2000, use_gpu=False)
    newton = _model(kind, 5)
    itemwise = _model(kind, 5)
    # Both fits start from the model's values, which give the two factors
    # equal slopes. Default starts would stagger them, and this five-item
    # two-factor model then creeps toward slopes at their bounds without
    # meeting tol=1e-9 within max_iter.
    newton_fit = EMEstimator(compute_standard_errors=False, **options).fit(
        newton, responses, start="model"
    )
    itemwise_fit = _ItemwiseEM(
        item_optim_ftol=1e-15, item_optim_maxiter=2000, **options
    ).fit(itemwise, responses, start="model")
    assert newton_fit.converged and itemwise_fit.converged
    assert newton_fit.log_likelihood == pytest.approx(
        itemwise_fit.log_likelihood, abs=1e-6
    )
    for name, values in itemwise.parameters.items():
        np.testing.assert_allclose(newton.parameters[name], values, atol=1e-4)


@pytest.mark.parametrize("n_factors", [1, 2])
def test_newton_solver_holds_fixed_coefficients(n_factors):
    rng = np.random.default_rng(8)
    points = GaussHermiteQuadrature(9, n_factors).nodes
    observed = rng.uniform(5.0, 50.0, (4, len(points)))
    truth = rng.normal(0.0, 1.0, (4, n_factors + 1))
    probability = 1.0 / (1.0 + np.exp(-(truth[:, :-1] @ points.T + truth[:, -1:])))
    correct = observed * np.clip(
        probability + rng.normal(0, 0.05, probability.shape), 0.01, 0.99
    )
    start = rng.normal(0.0, 0.5, (4, n_factors + 1))
    free = np.ones((4, n_factors + 1), dtype=bool)
    free[0, 0] = False  # a fixed slope
    free[1, -1] = False  # a fixed intercept
    free[2] = False  # nothing free
    slopes, intercepts, converged = newton_logistic_items(
        points, correct, observed, start[:, :-1], start[:, -1], free=free
    )
    assert converged.all()
    estimate = np.column_stack((slopes, intercepts))
    np.testing.assert_array_equal(estimate[~free], start[~free])
    design = np.column_stack((points, np.ones(len(points))))
    for item in (0, 1, 3):
        mask = free[item]

        def objective(x, item=item, mask=mask):
            coefficients = start[item].copy()
            coefficients[mask] = x
            z = design @ coefficients
            loss = np.sum(
                correct[item] * np.logaddexp(0, -z)
                + (observed[item] - correct[item]) * np.logaddexp(0, z)
            )
            p = 1.0 / (1.0 + np.exp(-z))
            return loss, (design.T @ (observed[item] * p - correct[item]))[mask]

        expected = minimize(
            objective,
            start[item, mask],
            jac=True,
            method="BFGS",
            options={"gtol": 1e-11, "maxiter": 10_000},
        ).x
        np.testing.assert_allclose(estimate[item, mask], expected, atol=1e-7)


def _masked_model(kind):
    """Return a model with fixed coordinates and its expected fallback items."""
    if kind == "mirt":
        pattern = np.array([[1, 0], [1, 0], [1, 1], [0, 1], [0, 1], [0, 1]])
        model = MultidimensionalModel(
            6, 2, model_type="confirmatory", loading_pattern=pattern
        )
        model.set_parameters(slopes=0.9 * pattern, intercepts=np.linspace(-1, 1, 6))
        model.set_free_parameter_masks({"intercepts": np.arange(6) != 4})
        return model, []
    model = _model(kind, 6)
    slopes = model.parameters["discrimination"]
    masks = {"discrimination": np.ones(slopes.shape, dtype=bool)}
    if kind == "2PL-2":
        # R mirt's exploratory identification and one more fixed slope.
        masks["discrimination"][0, 1] = masks["discrimination"][3, 0] = False
        slopes[0, 1] = 0.0
        model.set_parameters(discrimination=slopes)
        return model.set_free_parameter_masks(masks), []
    masks["discrimination"][1] = False
    # A fixed difficulty with a free slope ties the intercept to the slope.
    masks["difficulty"] = np.arange(6) != 4
    return model.set_free_parameter_masks(masks), [4]


@pytest.mark.parametrize("kind", ["2PL-1", "2PL-2", "mirt"])
def test_fixed_coordinates_stay_in_the_newton_m_step(kind, monkeypatch):
    # Regression: one fixed coordinate, or a confirmatory loading pattern,
    # sent every item to the itemwise optimizer, 3-6 times slower per
    # iteration.
    model, expected_fallback = _masked_model(kind)
    responses = _responses(6, model.n_factors, seed=11)
    estimator = EMEstimator(use_gpu=False)
    _, posterior = _prepare(estimator, model, responses, 9)
    reference = _itemwise_reference(model, responses, posterior, estimator._quadrature)
    before = model.parameters

    fallback = []
    optimize = EMEstimator._optimize_item

    def record(self, model, item_idx, *args, **kwargs):
        fallback.append(item_idx)
        return optimize(self, model, item_idx, *args, **kwargs)

    monkeypatch.setattr(EMEstimator, "_optimize_item", record)
    estimator._m_step(model, responses, posterior)

    assert fallback == expected_fallback
    for name, mask in model.free_parameter_masks.items():
        np.testing.assert_array_equal(
            model.parameters[name][~mask], before[name][~mask]
        )
    for name, values in reference.parameters.items():
        np.testing.assert_allclose(model.parameters[name], values, atol=2e-6)


@pytest.mark.parametrize("n_factors", [1, 2])
@pytest.mark.parametrize("difficulty", [True, False])
@pytest.mark.parametrize("estimate_slopes", [True, False])
def test_newton_solver_maximizes_the_log_posterior(
    n_factors, difficulty, estimate_slopes
):
    # The difficulty prior acts on b = -d / sum(a), so its derivatives in the
    # slope-intercept coordinates come from the chain rule.
    from mirt.estimation.priors import LogNormalPrior, NormalPrior

    rng = np.random.default_rng(23)
    points = GaussHermiteQuadrature(9, n_factors).nodes
    observed = rng.uniform(5.0, 40.0, (4, len(points)))
    truth = np.column_stack(
        (rng.uniform(0.6, 1.8, (4, n_factors)), rng.normal(0.0, 1.0, 4))
    )
    probability = 1.0 / (1.0 + np.exp(-(truth[:, :-1] @ points.T + truth[:, -1:])))
    correct = observed * np.clip(
        probability + rng.normal(0, 0.05, probability.shape), 0.01, 0.99
    )
    slope_prior, location_prior = LogNormalPrior(0.2, 0.3), NormalPrior(0.5, 0.4)
    start = np.column_stack((np.full((4, n_factors), 0.8), np.zeros(4)))
    free = np.ones((4, n_factors + 1), dtype=bool)
    if estimate_slopes and n_factors == 2:
        free[0, 1] = False  # a fixed slope still enters sum(a)
    slopes, intercepts, converged = newton_logistic_items(
        points,
        correct,
        observed,
        start[:, :-1],
        start[:, -1],
        estimate_slopes=estimate_slopes,
        free=free if estimate_slopes else None,
        slope_prior=slope_prior,
        location_prior=location_prior,
        difficulty=difficulty,
    )
    assert converged.all()
    estimate = np.column_stack((slopes, intercepts))
    design = np.column_stack((points, np.ones(len(points))))
    for item in range(4):
        mask = free[item] if estimate_slopes else np.arange(n_factors + 1) == n_factors

        def objective(x, item=item, mask=mask):
            coefficients = start[item].copy()
            coefficients[mask] = x
            z = design @ coefficients
            loss = np.sum(
                correct[item] * np.logaddexp(0, -z)
                + (observed[item] - correct[item]) * np.logaddexp(0, z)
            )
            if estimate_slopes:
                loss -= np.sum(slope_prior.log_pdf(coefficients[:-1][mask[:-1]]))
            location = (
                -coefficients[-1] / coefficients[:-1].sum()
                if difficulty
                else coefficients[-1]
            )
            return loss - float(location_prior.log_pdf(np.array([location]))[0])

        expected = minimize(
            objective,
            start[item, mask],
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20_000},
        ).x
        np.testing.assert_allclose(estimate[item, mask], expected, atol=1e-5)
        np.testing.assert_array_equal(estimate[item, ~mask], start[item, ~mask])


def test_newton_solver_steps_through_indefinite_posterior_curvature(monkeypatch):
    # A steep start where the log-normal prior is convex makes the
    # log-posterior's Hessian indefinite; scaled gradient steps reach the
    # region where Newton steps converge.
    import mirt.estimation._logistic_newton as logistic_newton
    from mirt.estimation.priors import LogNormalPrior

    definite = []
    steps = logistic_newton._definite_steps

    def record(gradient, curvature):
        step, positive = steps(gradient, curvature)
        definite.extend(positive)
        return step, positive

    monkeypatch.setattr(logistic_newton, "_definite_steps", record)
    points = GaussHermiteQuadrature(9).nodes
    observed = np.full((1, len(points)), 30.0)
    probability = 1.0 / (1.0 + np.exp(-(1.2 * points[:, 0] - 0.3)))
    correct = observed * probability
    prior = LogNormalPrior(np.log(6.0), 0.05)
    slopes, intercepts, converged = newton_logistic_items(
        points,
        correct,
        observed,
        np.full((1, 1), 18.0),
        np.zeros(1),
        slope_prior=prior,
    )
    assert converged.all()
    assert not definite[0] and definite[-1]
    design = np.column_stack((points, np.ones(len(points))))

    def objective(x):
        z = design @ x
        loss = np.sum(
            correct[0] * np.logaddexp(0, -z)
            + (observed[0] - correct[0]) * np.logaddexp(0, z)
        )
        return loss - float(prior.log_pdf(x[:1])[0])

    expected = minimize(
        objective,
        [4.0, 0.0],
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12},
    ).x
    np.testing.assert_allclose([slopes[0, 0], intercepts[0]], expected, atol=1e-5)


def test_newton_solver_leaves_items_without_prior_density():
    from mirt.estimation.priors import UniformPrior

    points = GaussHermiteQuadrature(7).nodes
    observed = np.full((2, 7), 10.0)
    correct = observed / (1.0 + np.exp(-2.5 * points[:, 0]))
    _, intercepts, converged = newton_logistic_items(
        points,
        correct,
        observed,
        np.array([[1.0], [3.0]]),
        np.zeros(2),
        slope_prior=UniformPrior(2.0, 4.0),
    )
    # The first item starts outside the prior's support.
    np.testing.assert_array_equal(converged, [False, True])
    assert intercepts[0] == 0.0
