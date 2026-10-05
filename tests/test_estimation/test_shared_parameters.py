"""Estimation of item parameters shared by every item (RSM and GRSM)."""

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from mirt.estimation._shared_step import (
    SharedParameters,
    _graded_rating_scale_objective,
    _numerical_objective,
    _rating_scale_objective,
)
from mirt.estimation.base import _free_shared_parameters
from mirt.estimation.em import EMEstimator
from mirt.estimation.mcem import MCEMEstimator, QMCEMEstimator, StochasticEMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.se_methods import compute_se
from mirt.estimation.standard_errors import (
    _finite_difference_information,
    _flatten_parameters,
)
from mirt.estimation.weighted import WeightedEMEstimator
from mirt.exceptions import MirtModelError, MirtValidationError
from mirt.models.base import DichotomousItemModel
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedRatingScaleModel,
    RatingScaleModel,
)

THRESHOLDS = np.array([0.0, 1.5, 4.0])


def _simulate(model, n_persons, seed):
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal((n_persons, 1))
    probabilities = model.probability(theta)
    draws = rng.random((n_persons, model.n_items, 1))
    responses = (draws > probabilities.cumsum(axis=2)).sum(axis=2)
    return np.minimum(responses, probabilities.shape[2] - 1).astype(np.int32)


def _generating_model(kind, n_items=10, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "RSM":
        model = RatingScaleModel(n_items, 4)
        model.set_parameters(
            difficulty=rng.normal(0.0, 0.7, n_items), thresholds=THRESHOLDS
        )
    else:
        model = GradedRatingScaleModel(n_items, 4)
        model.set_parameters(
            discrimination=1.7,
            difficulty=rng.normal(0.0, 0.5, n_items),
            thresholds=THRESHOLDS,
        )
    return model


FACTORIES = {"RSM": RatingScaleModel, "GRSM": GradedRatingScaleModel}


@pytest.fixture(scope="module")
def rating_data():
    return {
        kind: (model := _generating_model(kind), _simulate(model, 3000, seed=1))
        for kind in FACTORIES
    }


def test_rating_scale_models_declare_their_shared_parameters():
    assert TwoParameterLogistic._shared_parameters == frozenset()
    assert RatingScaleModel._shared_parameters == {"thresholds"}
    assert GradedRatingScaleModel._shared_parameters == {"discrimination", "thresholds"}
    assert _free_shared_parameters(RatingScaleModel(4, 3)) == ("thresholds",)
    assert _free_shared_parameters(GeneralizedPartialCredit(4, 3)) == ()


def test_shared_parameters_stay_whole_when_their_length_equals_n_items():
    # Three thresholds for three items used to be read as one per item.
    model = RatingScaleModel(3, 4)
    item = model.get_item_parameters(1)
    np.testing.assert_array_equal(item["thresholds"], model.thresholds)
    assert isinstance(item["difficulty"], float)
    graded = GradedRatingScaleModel(1, 3)
    np.testing.assert_array_equal(
        graded.get_item_parameters(0)["discrimination"], [1.0]
    )
    with pytest.raises(MirtValidationError, match="shared by all items"):
        model.set_item_parameter(0, "thresholds", 0.5)
    with pytest.raises(MirtValidationError, match="shared by all items"):
        graded.set_item_parameter(0, "discrimination", 1.5)
    model.set_item_parameter(2, "difficulty", 0.25)
    assert model.difficulty[2] == 0.25


@pytest.mark.parametrize("factory", list(FACTORIES.values()))
def test_set_parameters_does_not_mark_rating_scale_models_fitted(factory):
    model = factory(4, 3)
    model.set_parameters(difficulty=np.linspace(-1.0, 1.0, 4))
    assert not model.is_fitted
    assert not GeneralizedPartialCredit(4, 3).is_fitted


@pytest.mark.parametrize("kind", list(FACTORIES))
def test_em_recovers_shared_parameters(kind, rating_data):
    truth, responses = rating_data[kind]
    start = FACTORIES[kind](10, 4).parameters
    result = EMEstimator().fit(FACTORIES[kind](10, 4), responses)
    estimates = result.model.parameters

    assert result.converged
    np.testing.assert_allclose(estimates["thresholds"], THRESHOLDS, atol=0.15)
    np.testing.assert_allclose(
        estimates["difficulty"], truth.parameters["difficulty"], atol=0.15
    )
    if kind == "GRSM":
        assert estimates["discrimination"][0] == pytest.approx(1.7, abs=0.15)
    # Shared parameters leave their starting values, where they used to stay.
    for name in FACTORIES[kind]._shared_parameters:
        assert not np.allclose(estimates[name], start[name], atol=0.3)
    # The observed information covers the shared coordinates.
    assert result.se_method == "oakes"
    assert result.vcov.shape == (result.model.n_parameters,) * 2
    free = result.model.free_parameter_masks
    for name, errors in result.standard_errors.items():
        assert np.all(np.isfinite(errors[free[name]]))
        assert np.all(errors[free[name]] > 0)
        np.testing.assert_array_equal(errors[~free[name]], 0.0)


@pytest.mark.parametrize("kind", list(FACTORIES))
def test_shared_m_step_keeps_em_monotone(kind, rating_data):
    _, responses = rating_data[kind]
    estimator = EMEstimator(tol=1e-9, max_iter=60, compute_standard_errors=False)
    estimator.fit(FACTORIES[kind](10, 4), responses)
    history = np.asarray(estimator.convergence_history)
    assert np.all(np.diff(history) >= -1e-8)


@pytest.mark.parametrize("kind", list(FACTORIES))
def test_squarem_extrapolates_shared_parameters(kind, rating_data):
    _, responses = rating_data[kind]
    options = dict(tol=1e-8, max_iter=2000, compute_standard_errors=False)
    plain = EMEstimator(**options).fit(FACTORIES[kind](10, 4), responses)
    accelerated = EMEstimator(**options, accelerate="squarem").fit(
        FACTORIES[kind](10, 4), responses
    )
    assert accelerated.n_iterations < plain.n_iterations
    assert accelerated.log_likelihood == pytest.approx(plain.log_likelihood, abs=1e-5)
    for name, values in plain.model.parameters.items():
        np.testing.assert_allclose(
            accelerated.model.parameters[name], values, atol=1e-3
        )


def test_rating_scale_with_as_many_thresholds_as_items_fits():
    truth = RatingScaleModel(3, 4)
    truth.set_parameters(difficulty=np.array([-0.5, 0.0, 0.6]), thresholds=THRESHOLDS)
    responses = _simulate(truth, 2000, seed=4)
    result = EMEstimator(n_quadpts=15).fit(RatingScaleModel(3, 4), responses)
    assert result.model.thresholds.shape == (3,)
    np.testing.assert_allclose(result.model.thresholds, THRESHOLDS, atol=0.25)
    assert np.all(np.isfinite(result.standard_errors["thresholds"][1:]))


def test_results_label_shared_parameters_by_position():
    # Three thresholds of three items were labeled and tabulated as one per item.
    truth = RatingScaleModel(3, 4)
    truth.set_parameters(difficulty=np.array([-0.5, 0.0, 0.6]), thresholds=THRESHOLDS)
    responses = _simulate(truth, 300, seed=11)
    result = EMEstimator(n_quadpts=11).fit(RatingScaleModel(3, 4), responses)

    assert result.vcov_labels == [
        "difficulty[Item_0]",
        "difficulty[Item_1]",
        "difficulty[Item_2]",
        "thresholds[1]",
        "thresholds[2]",
    ]
    summary = result.summary()
    assert "thresholds[2]" in summary
    assert summary.count("Item_2") == 1
    with pytest.raises(MirtValidationError, match="global parameters"):
        result.coef()


def test_fixed_shared_parameters_are_held(rating_data):
    _, responses = rating_data["GRSM"]
    model = GradedRatingScaleModel(10, 4)
    model.set_parameters(discrimination=1.7)
    model.set_free_parameter_masks({"discrimination": np.array([False])})
    result = EMEstimator(n_quadpts=15).fit(model, responses)
    assert result.model.parameters["discrimination"][0] == 1.7
    assert result.standard_errors["discrimination"][0] == 0.0
    np.testing.assert_allclose(result.model.thresholds, THRESHOLDS, atol=0.15)


def test_graded_rating_scale_thresholds_stay_ordered():
    truth = GradedRatingScaleModel(6, 4)
    truth.set_parameters(
        discrimination=1.2,
        difficulty=np.linspace(-1.0, 1.0, 6),
        thresholds=np.array([0.0, 0.01, 0.02]),
    )
    responses = _simulate(truth, 800, seed=5)
    result = EMEstimator(n_quadpts=15, compute_standard_errors=False).fit(
        GradedRatingScaleModel(6, 4), responses
    )
    assert np.all(np.diff(result.model.thresholds) >= 1e-6 - 1e-9)


class _CustomRatingScale(RatingScaleModel):
    """Same curves through an overridden hook, so no closed forms apply."""

    def probability(self, theta, item_idx=None):
        return super().probability(theta, item_idx)


def test_custom_curves_use_the_numerical_shared_step(rating_data):
    _, responses = rating_data["RSM"]
    responses = responses[:1000]
    options = dict(n_quadpts=11, tol=1e-7, compute_standard_errors=False)
    exact = EMEstimator(**options).fit(RatingScaleModel(10, 4), responses)
    custom = EMEstimator(**options).fit(_CustomRatingScale(10, 4), responses)
    for name, values in exact.model.parameters.items():
        np.testing.assert_allclose(custom.model.parameters[name], values, atol=1e-3)


def test_custom_curves_use_marginal_differences_for_shared_errors(rating_data):
    _, responses = rating_data["RSM"]
    responses = responses[:300]
    fits = [
        EMEstimator(n_quadpts=11, tol=1e-7).fit(factory(10, 4), responses)
        for factory in (RatingScaleModel, _CustomRatingScale)
    ]
    assert fits[0].se_method == "oakes"
    # Custom curves resolve to complete-data curvature unless asked.
    assert fits[1].se_method == "complete_data"
    differenced = EMEstimator(n_quadpts=11, tol=1e-7, se_method="oakes").fit(
        _CustomRatingScale(10, 4), responses
    )
    for name, errors in fits[0].standard_errors.items():
        np.testing.assert_allclose(
            differenced.standard_errors[name], errors, rtol=2e-3, atol=1e-6
        )


@pytest.mark.parametrize("kind", list(FACTORIES))
def test_shared_objective_gradients_match_finite_differences(kind):
    model = _generating_model(kind, n_items=5, seed=2)
    nodes = GaussHermiteQuadrature(11).nodes
    rng = np.random.default_rng(3)
    counts = [rng.uniform(0.0, 10.0, (11, 4)) for _ in range(5)]
    layout = SharedParameters(model)
    exact = {"RSM": _rating_scale_objective, "GRSM": _graded_rating_scale_objective}
    objective = exact[kind](model, layout, nodes[:, 0], np.stack(counts, axis=1), 1e-10)
    numerical = _numerical_objective(model, layout, nodes, counts, 1e-10)
    point = layout.get(model) + 0.1
    original = dict(model._parameters)
    try:
        value, gradient = objective(point)
        assert value == pytest.approx(numerical(point), rel=1e-12)
        reference = approx_fprime(point, numerical, 1e-7)
    finally:
        model._parameters.update(original)
    np.testing.assert_allclose(gradient, reference, rtol=1e-5, atol=1e-3)


@pytest.fixture(scope="module")
def complete_data_grsm(rating_data):
    responses = rating_data["GRSM"][1][:800]
    fitted = EMEstimator(n_quadpts=15, se_method="complete_data").fit(
        GradedRatingScaleModel(10, 4), responses
    )
    return fitted, responses


@pytest.mark.parametrize("method", ["numerical", "forward", "richardson"])
def test_itemwise_curvature_covers_shared_parameters(method, complete_data_grsm):
    fitted, responses = complete_data_grsm
    quadrature = GaussHermiteQuadrature(15)
    nodes = quadrature.nodes
    log_joint = fitted.model.log_likelihood_batch(responses, nodes)
    log_joint += np.log(quadrature.weights)
    posterior = np.exp(log_joint - log_joint.max(axis=1, keepdims=True))
    posterior /= posterior.sum(axis=1, keepdims=True)
    errors = compute_se(fitted.model, responses, quadrature, posterior, method=method)
    # Differences of the summed objective agree with the closed form.
    for name in ("discrimination", "thresholds"):
        np.testing.assert_allclose(
            errors[name], fitted.standard_errors[name], rtol=2e-3, atol=1e-8
        )
    assert errors["thresholds"][0] == 0.0
    assert np.all(errors["thresholds"][1:] > 0)


def test_complete_data_shared_errors_match_summed_item_curvature(rating_data):
    _, responses = rating_data["RSM"]
    responses = responses[:800]
    estimator = EMEstimator(n_quadpts=15, se_method="complete_data")
    result = estimator.fit(RatingScaleModel(10, 4), responses)
    model = result.model
    quadrature = GaussHermiteQuadrature(15)
    log_joint = model.log_likelihood_batch(responses, quadrature.nodes)
    log_joint += np.log(quadrature.weights)
    posterior = np.exp(log_joint - log_joint.max(axis=1, keepdims=True))
    posterior /= posterior.sum(axis=1, keepdims=True)
    valid = responses >= 0
    counts = np.stack(
        [
            posterior.T @ ((responses[:, j, None] == np.arange(4)) & valid[:, j, None])
            for j in range(10)
        ],
        axis=1,
    )

    def expected_log_likelihood(threshold, index):
        thresholds = model.thresholds.copy()
        thresholds[index] = threshold
        increments = (
            quadrature.nodes[:, 0, None, None]
            - model.difficulty[None, :, None]
            - thresholds[None, None, :]
        )
        logits = np.concatenate(
            (np.zeros((15, 10, 1)), np.cumsum(increments, axis=2)), axis=2
        )
        logits -= logits.max(axis=2, keepdims=True)
        log_p = logits - np.log(np.exp(logits).sum(axis=2, keepdims=True))
        return float(np.sum(counts * log_p))

    for index in (1, 2):
        h = 1e-4
        center = model.thresholds[index]
        curvature = (
            expected_log_likelihood(center + h, index)
            - 2 * expected_log_likelihood(center, index)
            + expected_log_likelihood(center - h, index)
        ) / h**2
        assert result.standard_errors["thresholds"][index] == pytest.approx(
            np.sqrt(-1.0 / curvature), rel=1e-4
        )


@pytest.mark.parametrize(
    "estimator",
    [
        MCEMEstimator(n_samples=50, max_iter=2, seed=1),
        QMCEMEstimator(n_samples=64, max_iter=2, seed=1),
        StochasticEMEstimator(max_iter=2, seed=1),
    ],
    ids=["MCEM", "QMCEM", "StochasticEM"],
)
def test_estimators_without_a_shared_step_refuse_free_shared_parameters(
    estimator, rating_data
):
    _, responses = rating_data["RSM"]
    responses = responses[:200]
    with pytest.raises(MirtModelError, match="cannot estimate thresholds"):
        estimator.fit(RatingScaleModel(10, 4), responses)
    held = RatingScaleModel(10, 4)
    held.set_free_parameter_masks({"thresholds": np.zeros(3, dtype=bool)})
    result = estimator.fit(held, responses)
    np.testing.assert_array_equal(result.model.thresholds, [0.0, 1.0, 2.0])


def test_monte_carlo_errors_leave_held_shared_parameters_whole():
    # Monte Carlo curvature restored three thresholds of three items one per
    # item, which the shared parameter refuses.
    truth = RatingScaleModel(3, 4)
    truth.set_parameters(difficulty=np.array([-0.5, 0.0, 0.6]), thresholds=THRESHOLDS)
    responses = _simulate(truth, 200, seed=10)
    held = RatingScaleModel(3, 4)
    held.set_parameters(thresholds=THRESHOLDS)
    held.set_free_parameter_masks({"thresholds": np.zeros(3, dtype=bool)})
    estimator = MCEMEstimator(
        n_samples=50, max_iter=2, seed=1, compute_standard_errors=True
    )
    errors = estimator.fit(held, responses).standard_errors
    np.testing.assert_array_equal(errors["thresholds"], 0.0)
    assert np.all(errors["difficulty"] > 0)


def test_weighted_em_estimates_shared_parameters(rating_data):
    _, responses = rating_data["GRSM"]
    responses = responses[:500]
    options = dict(n_quadpts=11, tol=1e-5)
    plain = EMEstimator(**options, compute_standard_errors=False).fit(
        GradedRatingScaleModel(10, 4), responses
    )
    weighted = WeightedEMEstimator(**options).fit(
        GradedRatingScaleModel(10, 4), responses, weights=np.ones(500)
    )
    for name, values in plain.model.parameters.items():
        np.testing.assert_allclose(weighted.model.parameters[name], values, atol=1e-3)
    errors = weighted.standard_errors
    assert np.all(np.isfinite(errors["discrimination"]))
    assert np.all(errors["thresholds"][1:] > 0)


def test_rating_scale_louis_information_matches_marginal_differences(rating_data):
    from mirt.estimation._louis_information import louis_information

    _, responses = rating_data["GRSM"]
    model = _generating_model("GRSM", seed=0)
    responses = responses[:300].copy()
    responses[::7, 2] = -1
    quadrature = GaussHermiteQuadrature(11)
    mass = quadrature.weights / quadrature.weights.sum()
    _, layouts = _flatten_parameters(model)
    exact = louis_information(model, responses, quadrature.nodes, mass, layouts)
    differenced, _ = _finite_difference_information(
        model, responses, quadrature, mass, 1e-4
    )
    np.testing.assert_allclose(
        exact.information, differenced, rtol=0, atol=1e-6 * np.abs(differenced).max()
    )


def test_differenced_item_terms_cover_shared_coordinates(monkeypatch):
    from mirt.estimation import _louis_information as louis

    model = _generating_model("RSM", n_items=4, seed=6)
    responses = _simulate(model, 200, seed=7)
    quadrature = GaussHermiteQuadrature(9)
    mass = quadrature.weights / quadrature.weights.sum()
    _, layouts = _flatten_parameters(model)
    exact = louis.louis_information(model, responses, quadrature.nodes, mass, layouts)
    monkeypatch.setattr(louis, "has_analytic_item_derivatives", lambda model: False)
    differenced = louis.louis_information(
        model, responses, quadrature.nodes, mass, layouts
    )
    np.testing.assert_allclose(
        differenced.information, exact.information, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        differenced.score_crossproduct, exact.score_crossproduct, rtol=1e-5, atol=1e-6
    )


class _CommonSlopeLogistic(DichotomousItemModel):
    """Logistic items with one slope shared by every item."""

    model_name = "common-slope 2PL"
    _shared_parameters = frozenset({"discrimination"})

    def _initialize_parameters(self):
        self._parameters["discrimination"] = np.ones(1)
        self._parameters["difficulty"] = np.zeros(self.n_items)

    def probability(self, theta, item_idx=None):
        theta = self._ensure_theta_2d(theta)[:, 0]
        slope = self._parameters["discrimination"][0]
        difficulty = self._parameters["difficulty"]
        if item_idx is not None:
            return 1.0 / (1.0 + np.exp(-slope * (theta - difficulty[item_idx])))
        return 1.0 / (1.0 + np.exp(-slope * (theta[:, None] - difficulty)))

    def information(self, theta, item_idx=None):
        probability = self.probability(theta, item_idx)
        return (
            self._parameters["discrimination"][0] ** 2
            * probability
            * (1.0 - probability)
        )


def test_dichotomous_items_can_share_a_slope():
    truth = TwoParameterLogistic(8)
    truth.set_parameters(
        discrimination=np.full(8, 1.6), difficulty=np.linspace(-1.5, 1.5, 8)
    )
    rng = np.random.default_rng(8)
    theta = rng.standard_normal((2000, 1))
    responses = (rng.random((2000, 8)) < truth.probability(theta)).astype(np.int32)
    result = EMEstimator(n_quadpts=15).fit(_CommonSlopeLogistic(8), responses)
    assert result.converged
    assert result.model.parameters["discrimination"][0] == pytest.approx(1.6, abs=0.15)
    np.testing.assert_allclose(
        result.model.parameters["difficulty"], np.linspace(-1.5, 1.5, 8), atol=0.15
    )
    # Custom curves fall back to complete-data curvature for every coordinate.
    assert result.se_method == "complete_data"
    assert 0.0 < result.standard_errors["discrimination"][0] < 0.1


def test_shared_parameters_take_no_item_priors():
    from mirt.estimation.priors import NormalPrior, PriorSpecification

    responses = _simulate(_generating_model("RSM", n_items=3), 100, seed=9)
    estimator = EMEstimator(max_iter=2, item_priors={"thresholds": NormalPrior()})
    with pytest.raises(MirtValidationError, match="not an item parameter"):
        estimator.fit(RatingScaleModel(3, 4), responses)
    # A specification's slope prior skips the shared GRSM slope.
    fitted = EMEstimator(
        max_iter=2, item_priors=PriorSpecification(), compute_standard_errors=False
    ).fit(GradedRatingScaleModel(1, 4), responses[:, :1])
    difficulty_prior = PriorSpecification().difficulty.log_pdf(
        fitted.model.parameters["difficulty"]
    )
    assert fitted.log_posterior - fitted.log_likelihood == pytest.approx(
        float(difficulty_prior.sum())
    )
