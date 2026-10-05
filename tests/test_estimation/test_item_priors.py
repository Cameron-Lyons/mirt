"""Bayes modal EM with item-parameter priors."""

import numpy as np
import pytest

import mirt
import mirt.backends.rust.estimation as rust_estimation
import mirt.backends.rust.polytomous_mstep as polytomous_mstep
import mirt.estimation.em as em_module
from mirt.estimation._item_priors import ItemPriorPenalty, resolve_item_priors
from mirt.estimation.em import EMEstimator
from mirt.estimation.priors import (
    BetaPrior,
    CustomPrior,
    GammaPrior,
    LogNormalPrior,
    NormalPrior,
    Prior,
    PriorSpecification,
    TruncatedNormalPrior,
    UniformPrior,
)
from mirt.exceptions import MirtValidationError
from mirt.models.dichotomous import ThreeParameterLogistic, TwoParameterLogistic
from mirt.models.polytomous import GradedResponseModel
from mirt.results.fit_result import FitResult

_PRIORS_AND_POINTS = [
    (NormalPrior(0.3, 1.7), [-2.0, 0.1, 0.3, 4.0]),
    (TruncatedNormalPrior(0.5, 2.0, lower=-1.0, upper=3.0), [-0.5, 0.0, 2.5]),
    (LogNormalPrior(0.2, 0.4), [0.3, 1.0, 2.5]),
    (BetaPrior(5.0, 17.0), [0.05, 0.2, 0.6]),
    (BetaPrior(0.7, 1.0), [0.1, 0.5, 0.9]),
    (UniformPrior(-1.0, 2.0), [-0.5, 0.0, 1.5]),
    (GammaPrior(2.5, 1.5), [0.2, 1.0, 3.0]),
    (
        CustomPrior(
            lambda x: -(np.abs(np.asarray(x)) ** 3),
            lambda size, rng: np.zeros(size),
        ),
        [-1.2, 0.4, 2.0],
    ),
    (
        CustomPrior(
            lambda x: -(np.asarray(x) ** 4),
            lambda size, rng: np.zeros(size),
            grad_log_pdf_fn=lambda x: -4 * np.asarray(x) ** 3,
        ),
        [-1.0, 0.5, 1.5],
    ),
]


@pytest.mark.parametrize(("prior", "points"), _PRIORS_AND_POINTS)
def test_grad_log_pdf_matches_central_differences(prior, points):
    x = np.asarray(points)
    step = 1e-6
    numerical = (prior.log_pdf(x + step) - prior.log_pdf(x - step)) / (2 * step)
    np.testing.assert_allclose(prior.grad_log_pdf(x), numerical, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("prior", "outside"),
    [
        (TruncatedNormalPrior(0.0, 1.0, lower=0.0, upper=1.0), [-0.1, 1.1]),
        (LogNormalPrior(), [-1.0, 0.0]),
        (BetaPrior(2.0, 3.0), [-0.1, 1.1]),
        (UniformPrior(0.0, 1.0), [-0.1, 1.1]),
        (GammaPrior(2.0, 1.0), [-1.0]),
    ],
)
def test_grad_log_pdf_is_nan_outside_support(prior, outside):
    assert np.all(np.isnan(prior.grad_log_pdf(np.asarray(outside))))


def test_grad_log_pdf_handles_unit_shape_boundaries():
    np.testing.assert_allclose(BetaPrior(1.0, 3.0).grad_log_pdf(np.array([0.0])), -2.0)
    np.testing.assert_allclose(GammaPrior(1.0, 2.5).grad_log_pdf(np.array([0.0])), -2.5)


def test_prior_subclasses_inherit_numerical_gradient():
    class Quartic(Prior):
        def log_pdf(self, x):
            return -(np.asarray(x, dtype=float) ** 4)

        def sample(self, size, rng=None):
            return np.zeros(size)

        mean = 0.0
        variance = 1.0

    np.testing.assert_allclose(
        Quartic().grad_log_pdf(np.array([-1.0, 0.5])), [4.0, -0.5], rtol=1e-6
    )


@pytest.fixture(scope="module")
def responses_2pl():
    return mirt.simdata("2PL", n_persons=400, n_items=8, seed=3)


def test_diffuse_priors_reproduce_the_maximum_likelihood_estimate(responses_2pl):
    options = dict(tol=1e-9, item_optim_ftol=1e-12, compute_standard_errors=False)
    mle = EMEstimator(**options).fit(TwoParameterLogistic(8), responses_2pl)
    diffuse = {
        "discrimination": NormalPrior(0.0, 1e4),
        "difficulty": NormalPrior(0.0, 1e4),
    }
    estimator = EMEstimator(**options, item_priors=diffuse)
    map_fit = estimator.fit(TwoParameterLogistic(8), responses_2pl)
    for name, values in mle.model.parameters.items():
        np.testing.assert_allclose(map_fit.model.parameters[name], values, atol=1e-4)
    assert map_fit.log_likelihood == pytest.approx(mle.log_likelihood, abs=1e-6)
    assert mle.log_posterior is None
    assert map_fit.log_posterior < map_fit.log_likelihood


def test_beta_prior_keeps_guessing_off_the_bounds():
    responses = mirt.simdata("3PL", n_persons=500, n_items=20, seed=100)
    mle = EMEstimator(compute_standard_errors=False).fit(
        ThreeParameterLogistic(20), responses
    )
    prior = {"guessing": BetaPrior(5.0, 17.0)}
    estimator = EMEstimator(compute_standard_errors=False, item_priors=prior)
    result = estimator.fit(ThreeParameterLogistic(20), responses)

    def at_bounds(values):
        return int(np.sum((values < 1e-4) | (values > 0.5 - 1e-4)))

    assert at_bounds(mle.model.parameters["guessing"]) > 0
    assert at_bounds(result.model.parameters["guessing"]) == 0
    assert result.converged

    guessing = result.model.parameters["guessing"]
    log_prior = float(np.sum(BetaPrior(5.0, 17.0).log_pdf(guessing)))
    assert result.log_posterior == pytest.approx(
        result.log_likelihood + log_prior, rel=1e-12
    )
    # Convergence is judged on the log-posterior, which MAP-EM never decreases.
    history = np.asarray(estimator.convergence_history)
    assert history[-1] == pytest.approx(result.log_posterior)
    assert np.all(np.diff(history) > -1e-6)
    assert result.aic == pytest.approx(
        -2 * result.log_likelihood + 2 * result.n_parameters
    )


def test_priors_bypass_native_and_batched_m_steps(responses_2pl, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("fast path cannot apply item priors")

    monkeypatch.setattr(em_module, "em_iteration_3pl", forbidden)
    monkeypatch.setattr(polytomous_mstep, "try_polytomous_m_step", forbidden)
    monkeypatch.setattr(EMEstimator, "_newton_logistic_m_step", forbidden)
    monkeypatch.setattr(rust_estimation, "_em_fit_2pl_prepared", forbidden)
    prior = {"discrimination": LogNormalPrior(0.0, 0.5)}

    mirt.fit_mirt(responses_2pl, priors=prior, max_iter=3)
    data_3pl = mirt.simdata("3PL", n_persons=200, n_items=5, seed=8)
    EMEstimator(max_iter=3, item_priors=prior).fit(ThreeParameterLogistic(5), data_3pl)
    data_grm = mirt.simdata("GRM", n_persons=200, n_items=4, n_categories=4, seed=9)
    EMEstimator(max_iter=3, item_priors=prior).fit(
        GradedResponseModel(4, n_categories=4), data_grm
    )


def test_threaded_and_constrained_m_steps_apply_priors():
    data = mirt.simdata("GRM", n_persons=300, n_items=4, n_categories=4, seed=11)
    priors = {
        "discrimination": LogNormalPrior(0.0, 0.25),
        "thresholds": NormalPrior(0.0, 0.5),
    }
    options = dict(max_iter=40, compute_standard_errors=False, use_rust=False)
    plain = EMEstimator(**options).fit(GradedResponseModel(4, n_categories=4), data)
    serial = EMEstimator(**options, item_priors=priors).fit(
        GradedResponseModel(4, n_categories=4), data
    )
    threaded = EMEstimator(**options, n_jobs=2, item_priors=priors).fit(
        GradedResponseModel(4, n_categories=4), data
    )
    for name, values in serial.model.parameters.items():
        np.testing.assert_allclose(threaded.model.parameters[name], values, atol=1e-8)
    # The thresholds shrink toward zero and stay ordered.
    shrunk = np.abs(serial.model.parameters["thresholds"])
    assert np.sum(shrunk) < np.sum(np.abs(plain.model.parameters["thresholds"]))
    assert np.all(np.diff(serial.model.parameters["thresholds"], axis=1) > 0)


def test_fixed_coordinates_carry_no_prior(responses_2pl):
    model = TwoParameterLogistic(8)
    model.set_parameters(discrimination=np.full(8, 1.4))
    mask = np.ones(8, dtype=bool)
    mask[:3] = False
    model.set_free_parameter_masks({"discrimination": mask})
    prior = LogNormalPrior(0.0, 0.3)
    result = EMEstimator(
        compute_standard_errors=False, item_priors={"discrimination": prior}
    ).fit(model, responses_2pl, start="model")
    discrimination = result.model.parameters["discrimination"]
    np.testing.assert_array_equal(discrimination[:3], 1.4)
    log_prior = float(np.sum(prior.log_pdf(discrimination[3:])))
    assert result.log_posterior - result.log_likelihood == pytest.approx(log_prior)


def test_penalized_objective_gradient_matches_finite_differences():
    model = ThreeParameterLogistic(2)
    penalty = ItemPriorPenalty(
        {"discrimination": LogNormalPrior(0.1, 0.4), "guessing": BetaPrior(5, 17)}
    )

    def objective(params):
        return float(params @ params), 2.0 * params

    bounds = [(0.1, 5.0), (-6.0, 6.0), (0.0, 0.5)]
    penalized, shrunk = penalty.penalize(model, 1, objective, bounds, analytic=True)
    assert shrunk[0] == bounds[0] and shrunk[1] == bounds[1]
    assert 0.0 < shrunk[2][0] < 1e-6 and shrunk[2][1] == 0.5
    x = np.array([1.3, -0.4, 0.2])
    value, gradient = penalized(x)
    numerical = np.zeros(3)
    for index in range(3):
        delta = np.zeros(3)
        delta[index] = 1e-6
        numerical[index] = (penalized(x + delta)[0] - penalized(x - delta)[0]) / 2e-6
    np.testing.assert_allclose(gradient, numerical, rtol=1e-6)
    scalar, _ = penalty.penalize(
        model, 1, lambda params: objective(params)[0], bounds, analytic=False
    )
    assert scalar(x) == pytest.approx(value)


def test_prior_specification_resolution():
    specification = PriorSpecification(guessing=BetaPrior(2, 8))
    resolved = resolve_item_priors(specification, ThreeParameterLogistic(3))
    assert set(resolved) == {"discrimination", "difficulty", "guessing"}
    graded = resolve_item_priors(specification, GradedResponseModel(3, n_categories=3))
    assert set(graded) == {"discrimination"}
    assert resolve_item_priors(None, TwoParameterLogistic(2)) == {}


@pytest.mark.parametrize(
    ("priors", "message"),
    [
        ([NormalPrior()], "PriorSpecification or a mapping"),
        ({"difficulty": "normal"}, "Prior objects"),
    ],
)
def test_invalid_item_priors_are_rejected(priors, message):
    with pytest.raises(MirtValidationError, match=message):
        EMEstimator(item_priors=priors)


def test_unknown_or_unsupported_prior_parameters_are_rejected(responses_2pl):
    estimator = EMEstimator(max_iter=2, item_priors={"guessing": BetaPrior()})
    with pytest.raises(MirtValidationError, match="not an item parameter"):
        estimator.fit(TwoParameterLogistic(8), responses_2pl)
    narrow = EMEstimator(max_iter=2, item_priors={"guessing": UniformPrior(0.0, 0.3)})
    data = mirt.simdata("3PL", n_persons=100, n_items=4, seed=1)
    with pytest.raises(MirtValidationError, match="positive density"):
        narrow.fit(ThreeParameterLogistic(4), data)


def test_empty_prior_mapping_is_plain_maximum_likelihood(responses_2pl):
    estimator = EMEstimator(item_priors={}, compute_standard_errors=False)
    assert estimator.item_priors is None
    assert estimator.fit(TwoParameterLogistic(8), responses_2pl).log_posterior is None


def test_log_posterior_round_trips_through_fit_result(responses_2pl):
    result = mirt.fit_mirt(
        responses_2pl,
        priors={"discrimination": LogNormalPrior()},
        compute_standard_errors=False,
    )
    statistics = result.fit_statistics()
    assert statistics["log_posterior"] == result.log_posterior
    restored = FitResult.from_dict(result.to_dict())
    assert restored.log_posterior == result.log_posterior
    assert "Log-Posterior" in result.summary()
    plain = mirt.fit_mirt(responses_2pl, compute_standard_errors=False)
    assert "log_posterior" not in plain.fit_statistics()
    assert FitResult.from_dict(plain.to_dict()).log_posterior is None


def test_fit_mirt_rejects_priors_for_samplers(responses_2pl):
    with pytest.raises(MirtValidationError, match="estimation='EM'"):
        mirt.fit_mirt(
            responses_2pl, estimation="MHRM", priors={"difficulty": NormalPrior()}
        )


def test_squarem_reaches_the_same_posterior_mode():
    responses = mirt.simdata("3PL", n_persons=300, n_items=6, seed=100)
    priors = {"guessing": BetaPrior(5, 17), "discrimination": LogNormalPrior(0, 0.5)}
    options = dict(tol=1e-7, compute_standard_errors=False, item_priors=priors)
    plain = EMEstimator(**options).fit(ThreeParameterLogistic(6), responses)
    accelerated = EMEstimator(**options, accelerate="squarem").fit(
        ThreeParameterLogistic(6), responses
    )
    assert plain.converged and accelerated.converged
    assert accelerated.log_posterior == pytest.approx(plain.log_posterior, abs=1e-4)
    for name, values in plain.model.parameters.items():
        np.testing.assert_allclose(
            accelerated.model.parameters[name], values, atol=2e-3
        )


@pytest.mark.parametrize(("prior", "points"), _PRIORS_AND_POINTS)
def test_hess_log_pdf_matches_central_differences(prior, points):
    x = np.asarray(points)
    step = 1e-5
    numerical = (prior.grad_log_pdf(x + step) - prior.grad_log_pdf(x - step)) / (
        2 * step
    )
    np.testing.assert_allclose(prior.hess_log_pdf(x), numerical, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("prior", "outside"),
    [
        (TruncatedNormalPrior(0.0, 1.0, lower=0.0, upper=1.0), [-0.1, 1.1]),
        (LogNormalPrior(), [-1.0, 0.0]),
        (BetaPrior(2.0, 3.0), [-0.1, 1.1]),
        (UniformPrior(0.0, 1.0), [-0.1, 1.1]),
        (GammaPrior(2.0, 1.0), [-1.0]),
    ],
)
def test_hess_log_pdf_is_nan_outside_support(prior, outside):
    assert np.all(np.isnan(prior.hess_log_pdf(np.asarray(outside))))


_SE_PRIORS = {
    "discrimination": LogNormalPrior(0.0, 0.3),
    "difficulty": NormalPrior(0.0, 0.8),
}


def _map_fit(responses, se_method):
    estimator = EMEstimator(
        n_quadpts=15,
        tol=1e-10,
        item_optim_ftol=1e-13,
        item_priors=_SE_PRIORS,
        se_method=se_method,
    )
    return estimator.fit(TwoParameterLogistic(5), responses)


def test_bayes_modal_errors_invert_the_log_posterior_hessian():
    from mirt.estimation.quadrature import GaussHermiteQuadrature
    from mirt.estimation.standard_errors import (
        _finite_difference_information,
        _flatten_parameters,
    )

    responses = mirt.simdata("2PL", n_persons=300, n_items=5, seed=21)
    result = _map_fit(responses, "oakes")
    model = result.model
    quadrature = GaussHermiteQuadrature(15)
    mass = quadrature.weights / quadrature.weights.sum()
    likelihood, _ = _finite_difference_information(
        model, responses, quadrature, mass, 1e-4
    )
    values, layouts = _flatten_parameters(model)
    # Second differences of the log-prior, independent of hess_log_pdf.
    step = 1e-4
    prior = np.empty_like(values)
    offset = 0
    for name, layout in layouts.items():
        size = layout.free_indices.size
        x = values[offset : offset + size]
        log_pdf = _SE_PRIORS[name].log_pdf
        prior[offset : offset + size] = (
            -(log_pdf(x + step) - 2 * log_pdf(x) + log_pdf(x - step)) / step**2
        )
        offset += size
    covariance = np.linalg.inv(likelihood + np.diag(prior))
    np.testing.assert_allclose(result.vcov, covariance, rtol=1e-4, atol=1e-8)
    expected = np.sqrt(np.diag(covariance))
    np.testing.assert_allclose(
        np.concatenate([result.standard_errors[name] for name in layouts]),
        expected,
        rtol=1e-4,
    )
    # The prior's curvature matters at this sample size.
    likelihood_only = np.sqrt(np.diag(np.linalg.inv(likelihood)))
    assert np.all(expected < likelihood_only * 0.999)


def test_prior_curvature_enters_crossprod_and_complete_data_errors():
    from mirt.estimation._item_information import item_standard_errors
    from mirt.estimation.quadrature import GaussHermiteQuadrature
    from mirt.estimation.standard_errors import (
        _information_and_meat,
        _posterior_from_model,
    )

    responses = mirt.simdata("2PL", n_persons=300, n_items=5, seed=22)
    crossprod = _map_fit(responses, "crossprod")
    complete = _map_fit(responses, "complete_data")
    model = crossprod.model
    for name, values in model.parameters.items():
        np.testing.assert_allclose(complete.model.parameters[name], values, atol=1e-8)
    curvature = ItemPriorPenalty(_SE_PRIORS).information(model)

    quadrature = GaussHermiteQuadrature(15)
    mass = quadrature.weights / quadrature.weights.sum()
    _, meat, layouts = _information_and_meat(
        model, responses, quadrature, mass, 1e-5, observed=False
    )
    prior = np.concatenate([curvature[name] for name in layouts])
    np.testing.assert_allclose(
        crossprod.vcov, np.linalg.inv(meat + np.diag(prior)), rtol=1e-6, atol=1e-10
    )

    posterior = _posterior_from_model(model, responses, quadrature)
    likelihood = item_standard_errors(
        model, responses, posterior, quadrature.nodes, 1e-10
    )
    for name, errors in likelihood.items():
        np.testing.assert_allclose(
            complete.standard_errors[name],
            1.0 / np.sqrt(errors**-2.0 + curvature[name]),
            rtol=1e-6,
        )
