"""SQUAREM acceleration of the EM estimator."""

import numpy as np
import pytest

from mirt import simdata
from mirt.estimation import _acceleration
from mirt.estimation import em as em_module
from mirt.estimation._acceleration import (
    FreeItemParameters,
    squarem_point,
    squarem_step_length,
)
from mirt.estimation.em import EMEstimator
from mirt.estimation.latent_density import EmpiricalHistogram, GaussianDensity
from mirt.exceptions import MirtValidationError
from mirt.models.dichotomous import (
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
)

CASES = {
    "1PL": (lambda: OneParameterLogistic(8), dict(model="2PL", n_items=8), 15),
    "2PL": (lambda: TwoParameterLogistic(8), dict(model="2PL", n_items=8), 15),
    "2PL-2D": (
        lambda: TwoParameterLogistic(8, n_factors=2),
        dict(model="2PL", n_items=8, n_factors=2),
        7,
    ),
    "GRM": (
        lambda: GradedResponseModel(6, n_categories=4),
        dict(model="GRM", n_items=6, n_categories=4),
        15,
    ),
    "GPCM": (
        lambda: GeneralizedPartialCredit(6, n_categories=4),
        dict(model="GPCM", n_items=6, n_categories=4),
        15,
    ),
    "NRM": (
        lambda: NominalResponseModel(6, n_categories=3),
        dict(model="GRM", n_items=6, n_categories=3),
        15,
    ),
}


def _fit(factory, responses, n_quadpts, **options):
    model = factory()
    estimator = EMEstimator(
        n_quadpts=n_quadpts,
        tol=1e-6,
        max_iter=3000,
        use_gpu=False,
        compute_standard_errors=False,
        **options,
    )
    return model, estimator.fit(model, responses), estimator


@pytest.mark.parametrize("kind", list(CASES))
def test_squarem_reaches_the_plain_em_optimum_in_fewer_e_steps(kind):
    factory, simulation, n_quadpts = CASES[kind]
    responses = simdata(n_persons=500, seed=3, **simulation)
    # Plain EM with precise item optimizers is the reference fixed point.
    plain, plain_fit, _ = _fit(
        factory, responses, n_quadpts, use_rust=False, item_optim_ftol=1e-10
    )
    model, fit, estimator = _fit(factory, responses, n_quadpts, accelerate="squarem")

    assert plain_fit.converged and fit.converged
    assert fit.n_iterations < plain_fit.n_iterations
    assert fit.log_likelihood == pytest.approx(plain_fit.log_likelihood, abs=1e-5)
    for name, values in plain.parameters.items():
        np.testing.assert_allclose(model.parameters[name], values, atol=1e-3)
    history = np.asarray(estimator.convergence_history)
    assert history[-1] == fit.log_likelihood
    assert np.all(np.diff(history) >= -1e-8)


def test_squarem_3pl_improves_on_plain_em_without_the_fused_iteration(monkeypatch):
    responses = simdata(model="3PL", n_items=8, n_persons=3000, seed=5)
    plain, plain_fit, _ = _fit(
        lambda: ThreeParameterLogistic(8),
        responses,
        15,
        use_rust=False,
        item_optim_ftol=1e-10,
    )

    def fused(*args, **kwargs):
        raise AssertionError("SQUAREM must use the generic E- and M-steps")

    monkeypatch.setattr(em_module, "em_iteration_3pl", fused)
    model, fit, _ = _fit(
        lambda: ThreeParameterLogistic(8), responses, 15, accelerate="squarem"
    )

    # The 3PL likelihood is flat, so plain EM stops short of the optimum.
    assert fit.converged
    assert fit.n_iterations < plain_fit.n_iterations
    assert fit.log_likelihood >= plain_fit.log_likelihood - 1e-6
    for name, values in plain.parameters.items():
        np.testing.assert_allclose(model.parameters[name], values, atol=2e-2)


def test_squarem_keeps_graded_thresholds_ordered(monkeypatch):
    responses = simdata(model="GRM", n_items=6, n_categories=5, n_persons=400, seed=8)
    proposals = []
    point = _acceleration.squarem_point

    def record(*args):
        proposals.append(point(*args))
        return proposals[-1]

    monkeypatch.setattr(_acceleration, "squarem_point", record)
    model, fit, _ = _fit(
        lambda: GradedResponseModel(6, n_categories=5),
        responses,
        11,
        accelerate="squarem",
        use_rust=False,
    )
    assert fit.converged and proposals
    assert np.all(np.diff(model.parameters["thresholds"], axis=1) > 0)


@pytest.mark.parametrize("ordered", [True, False])
def test_rejected_extrapolations_fall_back_to_the_em_step(monkeypatch, ordered):
    responses = simdata(model="GRM", n_items=6, n_categories=4, n_persons=500, seed=3)
    reference, reference_fit, _ = _fit(
        CASES["GRM"][0], responses, 15, accelerate="squarem"
    )

    def far(start, first, second, alpha, lower, upper):
        if ordered:
            # Steep slopes with ordered thresholds lower the likelihood.
            return np.where(np.arange(start.size) < 6, upper, second)
        # Alternating threshold bounds violate the GRM ordering.
        return np.where(np.arange(start.size) % 2, lower, upper)

    monkeypatch.setattr(_acceleration, "squarem_point", far)
    evaluated = []
    evaluate = EMEstimator._evaluate

    def record(self, *args):
        result = evaluate(self, *args)
        evaluated.append(result[1])
        return result

    monkeypatch.setattr(EMEstimator, "_evaluate", record)
    model, fit, estimator = _fit(CASES["GRM"][0], responses, 15, accelerate="squarem")

    assert fit.converged
    # Worse ordered points are evaluated, then rejected; disordered points
    # are refused unevaluated. Neither enters the recorded history.
    assert (min(evaluated[1:]) < evaluated[0]) == ordered
    assert set(estimator.convergence_history) <= set(evaluated)
    assert np.all(np.diff(estimator.convergence_history) >= -1e-8)
    assert fit.log_likelihood == pytest.approx(reference_fit.log_likelihood, abs=1e-5)
    for name, values in reference.parameters.items():
        np.testing.assert_allclose(model.parameters[name], values, atol=1e-3)


# Caps cover every exit point of the first SQUAREM cycles.
@pytest.mark.parametrize("max_iter", range(1, 9))
def test_squarem_counts_e_steps_and_bounds_m_steps(monkeypatch, max_iter):
    responses = simdata(model="2PL", n_items=8, n_persons=300, seed=4)
    estimator = EMEstimator(
        n_quadpts=11,
        max_iter=max_iter,
        tol=1e-12,
        use_gpu=False,
        compute_standard_errors=False,
        accelerate="squarem",
    )
    calls = {"e": 0, "m": 0}
    e_step, m_step = estimator._e_step, estimator._m_step

    def counted_e_step(*args):
        calls["e"] += 1
        return e_step(*args)

    def counted_m_step(*args):
        calls["m"] += 1
        return m_step(*args)

    monkeypatch.setattr(estimator, "_e_step", counted_e_step)
    monkeypatch.setattr(estimator, "_m_step", counted_m_step)
    model = TwoParameterLogistic(8)
    fit = estimator.fit(model, responses)

    assert not fit.converged
    assert calls["m"] == max_iter
    assert fit.n_iterations == calls["e"] > calls["m"]
    assert not estimator._precise_m_steps
    # Every exit point reports the likelihood of the parameters it returns.
    _, log_marginal = e_step(model, responses)
    assert fit.log_likelihood == pytest.approx(np.sum(log_marginal), rel=1e-12)
    assert estimator.convergence_history[-1] == fit.log_likelihood


@pytest.mark.parametrize(
    "density",
    [EmpiricalHistogram, lambda: GaussianDensity(n_dimensions=1, estimate_mean=True)],
)
def test_estimated_latent_densities_fall_back_to_plain_em(density):
    responses = simdata(model="2PL", n_items=6, n_persons=300, seed=2)

    def fit(accelerate):
        model = TwoParameterLogistic(6)
        estimator = EMEstimator(
            n_quadpts=11,
            max_iter=20,
            latent_density=density(),
            use_gpu=False,
            compute_standard_errors=False,
            accelerate=accelerate,
        )
        return model, estimator.fit(model, responses)

    plain, plain_fit = fit("none")
    with pytest.warns(UserWarning, match="fixed Gaussian latent density"):
        model, result = fit("squarem")
    assert result.n_iterations == plain_fit.n_iterations
    assert result.log_likelihood == plain_fit.log_likelihood
    for name, values in plain.parameters.items():
        np.testing.assert_array_equal(model.parameters[name], values)


def test_accelerate_none_is_the_default_plain_loop():
    responses = simdata(model="GRM", n_items=5, n_categories=3, n_persons=300, seed=6)
    fits = []
    for options in ({}, {"accelerate": "none"}):
        model = GradedResponseModel(5, n_categories=3)
        estimator = EMEstimator(
            n_quadpts=11, use_gpu=False, compute_standard_errors=False, **options
        )
        fits.append((model, estimator.fit(model, responses), estimator))
    (first, first_fit, first_estimator), (second, second_fit, second_estimator) = fits
    assert first_fit.n_iterations == second_fit.n_iterations
    assert first_estimator.convergence_history == second_estimator.convergence_history
    for name, values in first.parameters.items():
        np.testing.assert_array_equal(second.parameters[name], values)


@pytest.mark.parametrize("value", ["fast", None, 1, "SQUAREM"])
def test_accelerate_is_validated(value):
    with pytest.raises(MirtValidationError, match="accelerate"):
        EMEstimator(accelerate=value)


def test_step_length_solves_a_linear_contraction():
    fixed_point = np.array([0.5, -1.0, 2.0])
    start = np.array([1.5, 0.0, 1.0])

    def contraction(x):
        return fixed_point + 0.8 * (x - fixed_point)

    first = contraction(start)
    second = contraction(first)
    alpha = squarem_step_length(start, first, second, step_max=100.0)
    lower, upper = np.full(3, -6.0), np.full(3, 6.0)

    # One SqS3 step is exact when every coordinate contracts at one rate.
    assert alpha == pytest.approx(5.0)
    np.testing.assert_allclose(
        squarem_point(start, first, second, alpha, lower, upper), fixed_point
    )
    assert squarem_step_length(start, first, second, step_max=2.0) == 2.0
    np.testing.assert_array_equal(
        squarem_point(start, first, second, 1.0, lower, upper), second
    )
    np.testing.assert_allclose(
        squarem_point(start, first, second, alpha, lower, np.full(3, 1.0)),
        [0.5, -1.0, 1.0],
    )
    # Without curvature between the EM steps there is nothing to extrapolate.
    assert squarem_step_length(start, first, 2 * first - start, 100.0) == 1.0


def test_free_item_parameters_pack_only_free_coordinates():
    model = GradedResponseModel(3, n_categories=[3, 5, 4])
    model.set_free_parameter_masks({"discrimination": np.array([True, False, True])})
    parameters = FreeItemParameters(model)
    vector = parameters.get(model)
    before = model.parameters

    assert vector.size == model.n_parameters
    np.testing.assert_array_equal(parameters.lower[:2], [0.1, 0.1])
    np.testing.assert_array_equal(parameters.upper[2:], np.full(vector.size - 2, 6.0))

    assert parameters.set(model, vector + 0.25, check_order=True)
    after = model.parameters
    assert after["discrimination"][1] == before["discrimination"][1]
    np.testing.assert_allclose(after["discrimination"][[0, 2]], [1.25, 1.25])
    masks = model.free_parameter_masks["thresholds"]
    np.testing.assert_array_equal(
        after["thresholds"][~masks], before["thresholds"][~masks]
    )
    np.testing.assert_allclose(parameters.get(model), vector + 0.25)

    # Reversing thresholds is refused and leaves the model unchanged.
    reversed_vector = parameters.get(model)
    reversed_vector[2:] = reversed_vector[2:][::-1]
    current = model.parameters
    assert not parameters.set(model, reversed_vector, check_order=True)
    for name, values in current.items():
        np.testing.assert_array_equal(model.parameters[name], values)

    rasch = FreeItemParameters(OneParameterLogistic(4))
    assert rasch.get(OneParameterLogistic(4)).size == 4
    np.testing.assert_array_equal(rasch.lower, np.full(4, -6.0))
