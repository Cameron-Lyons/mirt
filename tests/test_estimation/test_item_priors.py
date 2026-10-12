"""Bayes modal EM with item-parameter priors."""

import warnings

import numpy as np
import pytest

import mirt
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
from mirt.models.polytomous import GradedResponseModel, NominalResponseModel
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


def test_priors_take_batched_m_steps_instead_of_native_kernels(
    responses_2pl, monkeypatch
):
    # Regression: any item prior sent every item to its own scipy optimizer,
    # 4-8 times slower per EM cycle than the batched and native M-steps.
    def forbidden(*args, **kwargs):
        raise AssertionError("native kernels cannot apply item priors")

    monkeypatch.setattr(em_module, "em_iteration_3pl", forbidden)
    monkeypatch.setattr(polytomous_mstep, "try_polytomous_m_step", forbidden)
    batched = []
    for name in (
        "_newton_logistic_m_step",
        "_newton_guessing_m_step",
        "_newton_polytomous_m_step",
    ):
        step = getattr(EMEstimator, name)

        def record(self, model, *args, _step=step, **kwargs):
            batched.append(model.model_name)
            return _step(self, model, *args, **kwargs)

        monkeypatch.setattr(EMEstimator, name, record)
    prior = {"discrimination": LogNormalPrior(0.0, 0.5)}

    mirt.fit_mirt(responses_2pl, priors=prior, max_iter=3)
    data_3pl = mirt.simdata("3PL", n_persons=200, n_items=5, seed=8)
    # The native 3PL iteration applies these priors; without it the batched
    # step does.
    EMEstimator(max_iter=3, item_priors=prior, use_rust=False).fit(
        ThreeParameterLogistic(5), data_3pl
    )
    data_grm = mirt.simdata("GRM", n_persons=200, n_items=4, n_categories=4, seed=9)
    EMEstimator(max_iter=3, item_priors=prior).fit(
        GradedResponseModel(4, n_categories=4), data_grm
    )
    assert batched == ["2PL"] * 3 + ["3PL"] * 3 + ["GRM"] * 3


class _ItemwiseEM(EMEstimator):
    """Reference estimator: overriding an item hook keeps the itemwise scipy path."""

    def _optimize_item_params(self, *args, **kwargs):
        return super()._optimize_item_params(*args, **kwargs)


def _graded_responses(n_categories, seed, n_persons=600):
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(n_persons)
    columns = []
    for k in n_categories:
        cuts = np.sort(rng.normal(0.0, 1.0, k - 1))
        latent = rng.uniform(0.8, 1.8) * theta + rng.logistic(size=n_persons)
        columns.append(np.searchsorted(cuts, latent))
    responses = np.column_stack(columns)
    responses[rng.random(responses.shape) < 0.05] = -1
    return responses


def _prior_m_step_case(kind):
    from mirt.models.dichotomous import OneParameterLogistic
    from mirt.models.multidimensional import MultidimensionalModel
    from mirt.models.polytomous import GeneralizedPartialCredit, PartialCreditModel

    slope, location = LogNormalPrior(0.0, 0.5), NormalPrior(0.0, 1.0)
    if kind in ("1PL", "2PL", "2PL-2", "MIRT"):
        responses = mirt.simdata("2PL", n_persons=600, n_items=6, seed=14)
        if kind == "1PL":
            model = OneParameterLogistic(6)
            return model, responses, {"difficulty": location}
        if kind == "MIRT":
            model = MultidimensionalModel(6, 2)
            model.set_parameters(slopes=np.full((6, 2), 0.7))
            return model, responses, {"slopes": slope, "intercepts": location}
        model = TwoParameterLogistic(6, n_factors=2 if kind == "2PL-2" else 1)
        model.set_parameters(
            discrimination=np.full(model.parameters["discrimination"].shape, 0.9)
        )
        return model, responses, {"discrimination": slope, "difficulty": location}
    if kind == "3PL":
        responses = mirt.simdata("3PL", n_persons=600, n_items=6, seed=15)
        priors = {"discrimination": slope, "guessing": BetaPrior(5.0, 17.0)}
        return ThreeParameterLogistic(6), responses, priors
    n_categories = [3, 4, 4]
    responses = _graded_responses(n_categories, seed=16)
    if kind == "GRM":
        model = GradedResponseModel(3, n_categories=n_categories)
        return model, responses, {"discrimination": slope, "thresholds": location}
    if kind == "GPCM":
        model = GeneralizedPartialCredit(3, n_categories=n_categories)
        return model, responses, {"discrimination": slope, "steps": location}
    if kind == "PCM":
        return (
            PartialCreditModel(3, n_categories=n_categories),
            responses,
            {"steps": location},
        )
    model = NominalResponseModel(3, n_categories=n_categories)
    return model, responses, {"slopes": NormalPrior(0.5, 1.0), "intercepts": location}


@pytest.mark.parametrize(
    "kind", ["1PL", "2PL", "2PL-2", "MIRT", "3PL", "GRM", "GPCM", "PCM", "NRM"]
)
def test_batched_prior_m_steps_reach_the_itemwise_posterior_mode(kind, monkeypatch):
    from copy import deepcopy

    from mirt.estimation.latent_density import GaussianDensity
    from mirt.estimation.quadrature import GaussHermiteQuadrature

    model, responses, priors = _prior_m_step_case(kind)
    penalty = ItemPriorPenalty(resolve_item_priors(priors, model))
    batched = EMEstimator(use_gpu=False)
    reference = _ItemwiseEM(item_optim_ftol=1e-15, item_optim_maxiter=2000)
    quadrature = GaussHermiteQuadrature(
        9 if model.n_factors > 1 else 21, model.n_factors
    )
    for estimator in (batched, reference):
        estimator._quadrature = quadrature
        estimator._latent_density = GaussianDensity(n_dimensions=model.n_factors)
        estimator._prior_penalty = penalty
    posterior, _ = batched._e_step(model, responses)
    expected = deepcopy(model)
    reference._m_step(expected, responses, posterior)

    itemwise = []
    optimize = EMEstimator._optimize_item

    def record(self, model, item_idx, *args, **kwargs):
        itemwise.append(item_idx)
        return optimize(self, model, item_idx, *args, **kwargs)

    monkeypatch.setattr(EMEstimator, "_optimize_item", record)
    batched._m_step(model, responses, posterior)

    # Every item is solved by the batched step, at the itemwise optimum.
    assert itemwise == []
    for name, values in expected.parameters.items():
        np.testing.assert_allclose(model.parameters[name], values, atol=2e-5)

    def objective(fitted):
        total = 0.0
        for item in range(fitted.n_items):
            start, _, item_objective, _ = reference._item_objective(
                fitted, item, responses, posterior, quadrature.nodes
            )
            value = item_objective(start)
            total += float(value[0] if isinstance(value, tuple) else value)
        return total

    assert objective(model) <= objective(expected) + 1e-7 * abs(objective(expected))


@pytest.mark.parametrize("kind", ["1PL", "PCM"])
def test_common_slope_step_applies_priors(kind, monkeypatch):
    # Regression: priors made the 1PL and PCM skip the conditional common-slope
    # step for one joint scipy optimization over every item.
    from mirt.models.dichotomous import OneParameterLogistic
    from mirt.models.polytomous import PartialCreditModel

    if kind == "1PL":
        responses = mirt.simdata("2PL", n_persons=500, n_items=6, seed=17)
        location = "difficulty"

        def factory():
            return OneParameterLogistic(6, estimate_discrimination=True)

    else:
        responses = _graded_responses([3, 4, 4], seed=18, n_persons=500)
        location = "steps"

        def factory():
            return PartialCreditModel(
                3, n_categories=[3, 4, 4], estimate_discrimination=True
            )

    priors = {"discrimination": LogNormalPrior(0.0, 0.5), location: NormalPrior()}
    options = dict(
        tol=1e-10, max_iter=3000, compute_standard_errors=False, item_priors=priors
    )
    # The itemwise reference optimizes the tied items jointly.
    joint = _ItemwiseEM(item_optim_ftol=1e-15, item_optim_maxiter=2000, **options).fit(
        factory(), responses
    )
    common = []
    step = EMEstimator._m_step_common_slope

    def record(self, *args, **kwargs):
        common.append(step(self, *args, **kwargs))
        return common[-1]

    monkeypatch.setattr(EMEstimator, "_m_step_common_slope", record)
    fit = EMEstimator(**options).fit(factory(), responses)

    assert common and all(common)
    assert fit.converged and joint.converged
    assert fit.log_posterior == pytest.approx(joint.log_posterior, abs=1e-6)
    for name, values in joint.model.parameters.items():
        np.testing.assert_allclose(fit.model.parameters[name], values, atol=1e-4)


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


def test_prior_specification_must_reach_the_fitted_model():
    # Regression: fit_mirt and EMEstimator silently ignored a specification
    # with no matching parameter, which fit_mirt(spec=...) rejected.
    data = mirt.simdata("GRM", n_persons=150, n_items=3, n_categories=3, seed=4)
    specification = PriorSpecification(difficulty=NormalPrior(5.0, 0.01))
    with pytest.raises(MirtValidationError, match="sets no prior on the param"):
        EMEstimator(max_iter=2, item_priors=specification).fit(
            NominalResponseModel(3, n_categories=3), data
        )
    with pytest.raises(MirtValidationError, match="sets no prior on the param"):
        mirt.fit_mirt(data, model="NRM", priors=specification, max_iter=2)


def test_explicit_priors_on_missing_parameters_warn():
    data = mirt.simdata("GRM", n_persons=150, n_items=3, n_categories=3, seed=4)
    explicit = PriorSpecification(difficulty=NormalPrior(5.0, 0.01))
    with pytest.warns(UserWarning, match="difficulty prior is ignored"):
        result = EMEstimator(
            max_iter=2, item_priors=explicit, compute_standard_errors=False
        ).fit(GradedResponseModel(3, n_categories=3), data)
    # The default discrimination prior still applies.
    assert result.log_posterior is not None

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Defaults, and guessing priors documented for 3PL and 4PL models.
        for specification in (
            PriorSpecification(),
            PriorSpecification(guessing=BetaPrior(2.0, 8.0)),
        ):
            EMEstimator(
                max_iter=2, item_priors=specification, compute_standard_errors=False
            ).fit(GradedResponseModel(3, n_categories=3), data)


def test_mixed_format_specification_needs_one_matching_component():
    from mirt.estimation.mixed_format_em import MixedFormatEMEstimator
    from mirt.models.mixed_format import MixedItemModel

    binary = mirt.simdata("2PL", n_persons=150, n_items=2, seed=5)
    graded = mirt.simdata("GRM", n_persons=150, n_items=2, n_categories=3, seed=6)
    data = np.column_stack([binary, graded])
    model = MixedItemModel.from_itemtypes(["2PL", "2PL", "NRM", "NRM"], n_categories=3)
    result = MixedFormatEMEstimator(
        max_iter=2,
        item_priors=PriorSpecification(),
        compute_standard_errors=False,
    ).fit(model, data)
    assert result.log_posterior is not None

    nominal = MixedItemModel.from_itemtypes(["NRM", "NRM"], n_categories=[3, 4])
    with pytest.raises(MirtValidationError, match="sets no prior on the param"):
        MixedFormatEMEstimator(max_iter=2, item_priors=PriorSpecification()).fit(
            nominal, graded
        )


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
    data = mirt.simdata("3PL", n_persons=100, n_items=4, seed=1)
    nowhere = EMEstimator(max_iter=2, item_priors={"guessing": UniformPrior(0.6, 0.9)})
    with pytest.raises(MirtValidationError, match="positive density"):
        nowhere.fit(ThreeParameterLogistic(4), data)


def test_optimizer_boxes_are_intersected_with_the_prior_support():
    data = mirt.simdata("3PL", n_persons=300, n_items=4, seed=1)
    narrow = EMEstimator(
        max_iter=20,
        item_priors={"guessing": UniformPrior(0.0, 0.1)},
        compute_standard_errors=False,
    )
    fit = narrow.fit(ThreeParameterLogistic(4), data)
    guessing = fit.model.guessing
    # The support caps the guessing box [0, 0.5] at 0.1. Its lower end 0 has
    # positive density, and an estimate may rest there under a flat prior.
    assert np.all((guessing >= 0.0) & (guessing < 0.1))
    assert np.isfinite(fit.log_posterior)
    # A log-normal slope prior removes the negative loadings of a
    # multidimensional box.
    responses = mirt.simdata("2PL", n_persons=300, n_items=6, n_factors=2, seed=3)
    fitted = EMEstimator(
        max_iter=5,
        item_priors={"discrimination": LogNormalPrior(0.0, 0.5)},
        compute_standard_errors=False,
    ).fit(TwoParameterLogistic(6, n_factors=2), responses)
    assert np.all(fitted.model.discrimination > 0.0)


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
        quadrature="gauss-hermite",
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


@pytest.mark.skipif(not mirt.is_rust_available(), reason="native backend unavailable")
@pytest.mark.parametrize(
    "priors",
    [
        {"guessing": BetaPrior(5.0, 17.0)},
        {
            "discrimination": LogNormalPrior(0.0, 0.5),
            "difficulty": NormalPrior(0.0, 2.0),
            "guessing": BetaPrior(2.0, 8.0),
        },
    ],
)
def test_native_3pl_iteration_reaches_the_numpy_posterior_mode(monkeypatch, priors):
    # The fused native 3PL iteration used to refuse item priors, so these
    # fits ran the slower batched NumPy step.
    calls = []
    native = em_module.em_iteration_3pl

    def record(*args, **kwargs):
        calls.append(kwargs["priors"])
        return native(*args, **kwargs)

    data = mirt.simdata("3PL", n_persons=1500, n_items=8, seed=12)
    options = {"item_priors": priors, "tol": 1e-9, "max_iter": 2000}
    expected = EMEstimator(use_rust=False, **options).fit(
        ThreeParameterLogistic(8), data
    )
    monkeypatch.setattr(em_module, "em_iteration_3pl", record)
    actual = EMEstimator(**options).fit(ThreeParameterLogistic(8), data)

    assert calls and all(call is not None for call in calls)
    assert actual.log_posterior == pytest.approx(expected.log_posterior, abs=1e-6)
    assert actual.log_likelihood == pytest.approx(expected.log_likelihood, abs=1e-3)
    for name, values in expected.model.parameters.items():
        np.testing.assert_allclose(actual.model.parameters[name], values, atol=2e-3)
    for name in expected.standard_errors:
        np.testing.assert_allclose(
            actual.standard_errors[name], expected.standard_errors[name], rtol=2e-2
        )


def test_unsupported_priors_keep_the_numpy_3pl_step(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the native iteration cannot apply a gamma prior")

    monkeypatch.setattr(em_module, "em_iteration_3pl", forbidden)
    data = mirt.simdata("3PL", n_persons=200, n_items=5, seed=8)
    result = EMEstimator(
        max_iter=3, item_priors={"discrimination": GammaPrior(2.0, 2.0)}
    ).fit(ThreeParameterLogistic(5), data)
    assert np.isfinite(result.log_posterior)
