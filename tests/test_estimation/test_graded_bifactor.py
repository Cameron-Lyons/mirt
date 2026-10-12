"""Graded bifactor models: curves, dimension-reduction EM, inference and scoring."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import logsumexp

import mirt
from mirt.estimation import bifactor_em
from mirt.estimation._em_context import EMFitContext
from mirt.estimation._graded_order import THRESHOLD_GAP, graded_order
from mirt.estimation._louis_information import (
    louis_information,
    supports_louis_information,
)
from mirt.estimation._refit import em_estimator_for
from mirt.estimation.bifactor_em import BifactorEMEstimator, bfactor
from mirt.estimation.em import EMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature, RectangularQuadrature
from mirt.estimation.standard_errors import _flatten_parameters
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models.bifactor import BifactorModel, GradedBifactorModel
from mirt.results.fit_result import FitResult
from mirt.scoring.eap import EAPScorer

_LABELS = [4, 4, 4, 9, 9, 9, 9]
_CATEGORIES = [3, 4, 2, 4, 3, 2, 5]


def _likert(seed, n_persons, labels, categories, *, missing=0.0):
    """Draw graded bifactor responses from seeded generating parameters."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n_items = labels.size
    model = GradedBifactorModel(n_items, labels, list(categories))
    intercepts = np.zeros((n_items, max(categories) - 1))
    for item, count in enumerate(categories):
        intercepts[item, : count - 1] = np.linspace(1.8, -1.8, count - 1)
        intercepts[item] += rng.normal(0.0, 0.4)
    model.set_parameters(
        general_loadings=rng.uniform(0.9, 1.7, n_items),
        specific_loadings=rng.uniform(0.6, 1.3, n_items),
        intercepts=intercepts,
    )
    theta = rng.standard_normal((n_persons, model.n_factors))
    responses = model.simulate(theta, seed=seed + 1).astype(int)
    responses[rng.random(responses.shape) < missing] = -1
    return model, responses


def _order(problem, model):
    """Item-major coordinates of the free parameters in ``layouts`` order."""
    _, layouts = _flatten_parameters(model)
    first = problem.layout.first
    width = model.max_categories - 1
    flat = layouts["intercepts"].free_indices
    return layouts, np.concatenate(
        (
            first[layouts["general_loadings"].free_indices],
            first[layouts["specific_loadings"].free_indices] + 1,
            first[flat // width] + 2 + flat % width,
        )
    )


def _product_log_marginals(model, responses, quadrature):
    log_likelihood = model.log_likelihood_batch(responses, quadrature.nodes)
    weights = quadrature.weights / quadrature.weights.sum()
    return logsumexp(log_likelihood + np.log(weights), axis=1)


def test_graded_curves_reduce_to_the_dichotomous_bifactor_item() -> None:
    model, _ = _likert(0, 10, _LABELS, _CATEGORIES)
    theta = np.random.default_rng(1).standard_normal((25, model.n_factors))
    probability = model.probability(theta)

    np.testing.assert_allclose(probability.sum(axis=2), 1.0, atol=1e-12)
    assert probability.shape == (25, 7, 5)
    assert np.all(probability[:, 0, 3:] == 0.0)
    for item in range(model.n_items):
        np.testing.assert_allclose(
            model.probability(theta, item), probability[:, item, : _CATEGORIES[item]]
        )
    # Two-category items are BifactorModel items with the same coefficients.
    dichotomous = BifactorModel(7, _LABELS)
    dichotomous.set_parameters(
        general_loadings=model.general_loadings,
        specific_loadings=model.specific_loadings,
        intercepts=model.intercepts[:, 0],
    )
    for item in (2, 5):
        np.testing.assert_allclose(
            probability[:, item, 1], dichotomous.probability(theta)[:, item], rtol=1e-13
        )
    # The scalar information is the trace of the exact Fisher matrix.
    matrices = np.stack(
        [model.item_information_matrix(theta, item) for item in range(7)], axis=1
    )
    np.testing.assert_allclose(
        np.trace(matrices, axis1=2, axis2=3).sum(axis=1), model.information(theta)
    )
    np.testing.assert_allclose(
        model.test_information_matrix(theta), matrices.sum(axis=1), rtol=1e-12
    )
    assert model.n_parameters == 2 * 7 + sum(count - 1 for count in _CATEGORIES)
    assert model.get_loading_matrix().shape == (7, 3)
    assert np.isfinite(model.omega_hierarchical())


def test_graded_model_copy_simulate_and_validation() -> None:
    model, _ = _likert(2, 10, _LABELS, _CATEGORIES)
    model.set_free_parameter_masks({"specific_loadings": np.arange(7) != 3})
    copy = model.copy()
    copy.set_parameters(general_loadings=np.zeros(7))

    assert type(copy) is GradedBifactorModel
    assert copy.n_categories == _CATEGORIES
    assert np.array_equal(copy.specific_factors, _LABELS)
    assert not copy.free_parameter_masks["specific_loadings"][3]
    assert np.all(model.general_loadings > 0.0)
    theta = np.random.default_rng(3).standard_normal((400, 3))
    responses = model.simulate(theta, seed=4)
    assert np.array_equal(responses, model.simulate(theta, seed=4, chunk_size=7))
    assert np.all(responses.max(axis=0) < np.array(_CATEGORIES))
    # Padding is ignored, active intercepts must be finite.
    padded = model.intercepts.copy()
    padded[0, 2:] = np.nan
    model.set_parameters(intercepts=padded)
    padded[0, 0] = np.inf
    with pytest.raises(MirtValidationError, match="intercepts"):
        model.set_parameters(intercepts=padded)
    with pytest.raises(ValueError, match="specific_factors"):
        GradedBifactorModel(3, [0, 1], 3)
    with pytest.raises(MirtValidationError, match="n_categories"):
        GradedBifactorModel(3, [0, 0, 1], [3, 1, 3])


@pytest.mark.parametrize("rule", ["gauss-hermite", "rectangular"])
def test_reduced_log_likelihood_and_eap_match_the_product_grid(rule) -> None:
    model, responses = _likert(5, 300, _LABELS, _CATEGORIES, missing=0.1)
    problem = bifactor_em._GradedBifactorProblem(model)
    grid = problem.grid(7, model, rule)
    curves = bifactor_em._GradedCurves.evaluate(grid, model, problem.layout)
    with EMFitContext(responses) as context:
        blocks = list(
            bifactor_em._graded_posterior_blocks(
                grid, curves, problem.layout, context, 37
            )
        )
    reduced = np.concatenate([block.log_marginal for block in blocks])
    quadrature = (
        GaussHermiteQuadrature(7, 3)
        if rule == "gauss-hermite"
        else RectangularQuadrature(7, 3)
    )

    np.testing.assert_allclose(
        reduced, _product_log_marginals(model, responses, quadrature), atol=1e-10
    )

    model._is_fitted = True
    scores = mirt.fscores(model, responses, method="EAP", n_quadpts=9)
    product = EAPScorer(n_quadpts=9)
    product._bifactor_grid = lambda _: None  # force the product grid
    reference = product.score(model, responses)
    np.testing.assert_allclose(scores.theta, reference.theta, atol=1e-12)
    np.testing.assert_allclose(
        scores.standard_error, reference.standard_error, atol=1e-12
    )


def test_observed_information_matches_product_grid_louis() -> None:
    model, responses = _likert(6, 300, _LABELS, _CATEGORIES, missing=0.1)
    problem = bifactor_em._GradedBifactorProblem(model)
    grid = problem.grid(7, model)
    with EMFitContext(responses, compress=True) as context:
        terms = problem.louis_terms(model, grid, context)
        scores_only = problem.louis_terms(model, grid, context, observed=False)
    layouts, order = _order(problem, model)
    quadrature = GaussHermiteQuadrature(7, 3)
    mass = quadrature.weights / quadrature.weights.sum()
    reference = louis_information(model, responses, quadrature.nodes, mass, layouts)
    block = np.ix_(order, order)

    assert supports_louis_information(model)
    # The product-grid reference differences item curves numerically.
    for reduced, product in (
        (terms.information[block], reference.information),
        (terms.score_crossproduct[block], reference.score_crossproduct),
    ):
        scale = np.abs(product).max()
        np.testing.assert_allclose(reduced, product, rtol=1e-4, atol=1e-6 * scale)
    np.testing.assert_array_equal(
        scores_only.score_crossproduct, terms.score_crossproduct
    )
    np.testing.assert_array_equal(scores_only.complete, terms.complete)


def test_m_step_solves_each_item_with_decreasing_intercepts() -> None:
    model, responses = _likert(7, 400, _LABELS, _CATEGORIES)
    # A nearly empty middle category pulls its two intercepts together.
    responses[(responses[:, 1] == 2) & (np.arange(400) % 10 != 0), 1] = 1
    problem = bifactor_em._GradedBifactorProblem(model)
    grid = problem.grid(7, model)
    with EMFitContext(responses, compress=True) as context:
        _, (counts,) = problem.expected_counts(model, grid, context)
    problem.m_step(model, grid, counts)

    layout = problem.layout
    for item, count in enumerate(_CATEGORIES):
        item_counts = counts[layout.offsets[item] : layout.offsets[item] + count]
        estimate = np.r_[
            model.general_loadings[item],
            model.specific_loadings[item],
            model.intercepts[item, : count - 1],
        ]

        def objective(params, item_counts=item_counts):
            terms = bifactor_em._graded_terms(
                grid.design, item_counts[None], params[None], second=False
            )
            return terms.loss[0], terms.gradient[0]

        reference = minimize(
            objective,
            estimate + 0.05,
            jac=True,
            method="SLSQP",
            constraints=[{"type": "ineq", "fun": lambda x: -np.diff(x[2:]) - 1e-6}],
            options={"ftol": 1e-14, "maxiter": 500},
        )
        assert objective(estimate)[0] <= reference.fun + 1e-8
        assert np.all(np.diff(estimate[2:]) < 0.0)


def test_m_step_holds_fixed_coordinates_and_orders_disordered_starts() -> None:
    model, responses = _likert(8, 300, _LABELS, _CATEGORIES)
    intercepts = model.intercepts.copy()
    intercepts[1, :3] = [-1.0, 0.5, 1.0]  # increasing: disordered
    model.set_parameters(intercepts=intercepts)
    free = model.free_parameter_masks
    free["general_loadings"][0] = False
    free["intercepts"][3, 1] = False
    model.set_free_parameter_masks(free)
    held = (model.general_loadings[0], model.intercepts[3, 1])
    problem = bifactor_em._GradedBifactorProblem(model)
    grid = problem.grid(7, model)
    with EMFitContext(responses, compress=True) as context:
        _, (counts,) = problem.expected_counts(model, grid, context)
    problem.m_step(model, grid, counts)

    assert (model.general_loadings[0], model.intercepts[3, 1]) == held
    for item, count in enumerate(_CATEGORIES):
        assert np.all(np.diff(model.intercepts[item, : count - 1]) < 0.0)


def test_em_estimator_reduces_graded_fits_and_orders_product_grid_intercepts() -> None:
    model, responses = _likert(24, 400, _LABELS, _CATEGORIES)
    options = {"max_iter": 5, "tol": 1e-12, "compute_standard_errors": False}
    em = EMEstimator(n_quadpts=7, **options)
    delegated = em.fit(GradedBifactorModel(7, _LABELS, _CATEGORIES), responses)
    reduced = BifactorEMEstimator(n_quadpts=7, **options)
    reduced.fit(GradedBifactorModel(7, _LABELS, _CATEGORIES), responses)

    assert delegated.refit_recipe.estimator is BifactorEMEstimator
    np.testing.assert_allclose(
        em.convergence_history, reduced.convergence_history, rtol=1e-12
    )

    # Regression: product-grid M-steps left graded bifactor intercepts
    # disordered, whose clipped curves sum to more than one.
    disordered = model.intercepts.copy()
    for item, count in enumerate(_CATEGORIES):
        disordered[item, : count - 1] = disordered[item, : count - 1][::-1]
    order = graded_order(model.copy().set_parameters(intercepts=disordered), 6, 2 + 4)
    assert order is not None and order.sign == -1.0
    assert not order.satisfied(np.r_[1.0, 1.0, disordered[6]])
    projected = order.project(
        np.r_[1.0, 1.0, disordered[6]], np.full(6, -30.0), np.full(6, 30.0)
    )
    assert np.all(-np.diff(projected[2:]) >= THRESHOLD_GAP * (1.0 - 1e-9))
    # A rectangular grid takes the product grid.
    product = EMEstimator(n_quadpts=7, quadrature="rectangular", **options)
    result = product.fit(
        GradedBifactorModel(7, _LABELS, _CATEGORIES),
        responses,
        start={"intercepts": disordered},
    )
    assert result.refit_recipe.estimator is EMEstimator
    for item, count in enumerate(_CATEGORIES):
        assert np.all(np.diff(result.model.intercepts[item, : count - 1]) < 0.0)


def test_two_category_graded_fit_equals_the_dichotomous_bifactor_fit() -> None:
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    _, responses = _likert(4, 500, labels, [2] * 8)
    options = {"n_quadpts": 9, "tol": 1e-10, "max_iter": 2000}
    graded = bfactor(responses, labels, model="graded", **options)
    dichotomous = bfactor(responses, labels, **options)

    assert type(graded.model) is GradedBifactorModel
    assert type(dichotomous.model) is BifactorModel
    assert graded.converged and dichotomous.converged
    assert graded.log_likelihood == pytest.approx(dichotomous.log_likelihood, abs=1e-7)
    assert graded.n_parameters == dichotomous.n_parameters == 24
    for name in ("general_loadings", "specific_loadings"):
        np.testing.assert_allclose(
            graded.model.parameters[name],
            dichotomous.model.parameters[name],
            atol=1e-4,
        )
        np.testing.assert_allclose(
            graded.standard_errors[name], dichotomous.standard_errors[name], rtol=1e-3
        )
    np.testing.assert_allclose(
        graded.model.intercepts[:, 0], dichotomous.model.intercepts, atol=1e-4
    )


# R mirt 1.48: bfactor(data, c(1, 1, 1, 2, 2, 2), itemtype = "graded",
# SE.type = "Oakes", TOL = 1e-8) on _likert(2022, 500, ...) data with R's
# grid of 31 equally spaced points on [-6, 6]; logLik -3106.785440, nest 27.
_R_LOG_LIKELIHOOD = -3106.785440
_R_ESTIMATES = {
    "general_loadings": [1.040263, 0.965831, 0.791137, 1.301904, 1.490708, 1.266081],
    "specific_loadings": [1.504507, 0.772517, 0.388335, 1.47949, 0.872388, 0.991484],
    "intercepts": [
        [3.269096, 1.089692, -0.959147],
        [1.59885, -0.245733, -2.24151],
        [2.136151, 0.0, 0.0],
        [1.009474, -0.92055, -2.627703],
        [1.779521, -1.592202, 0.0],
        [1.76996, -0.168906, -1.947815],
    ],
}
_R_STANDARD_ERRORS = {
    "general_loadings": [0.7604, 0.49431, 0.48042, 0.84121, 0.86742, 0.60497],
    "specific_loadings": [1.13339, 0.71232, 0.60727, 0.60109, 1.08695, 0.90056],
    "intercepts": [
        [1.14005, 0.42144, 0.3753],
        [0.20397, 0.12039, 0.26216],
        [0.20237, 0.0, 0.0],
        [0.22413, 0.21307, 0.45676],
        [0.21523, 0.20044, 0.0],
        [0.18733, 0.13276, 0.19965],
    ],
}


def test_bfactor_graded_matches_r_mirt() -> None:
    # Regression: bfactor() raised MirtDataError for any response above 1.
    labels = [0, 0, 0, 1, 1, 1]
    _, responses = _likert(2022, 500, labels, [4, 4, 2, 4, 3, 4], missing=0.03)
    estimator = BifactorEMEstimator(n_quadpts=31, quadrature="rectangular", tol=1e-7)
    model = GradedBifactorModel(6, labels, [4, 4, 2, 4, 3, 4])
    result = estimator.fit(model, responses, start=_R_ESTIMATES)

    # R's estimates attain R's log-likelihood on R's grid and are the MLE.
    assert estimator.convergence_history[0] == pytest.approx(
        _R_LOG_LIKELIHOOD, abs=1e-5
    )
    assert result.converged and result.n_iterations <= 3
    assert result.n_parameters == 27
    assert result.se_method == "oakes"
    masks = result.model.free_parameter_masks
    for name, values in _R_ESTIMATES.items():
        np.testing.assert_allclose(
            result.model.parameters[name][masks[name]],
            np.asarray(values)[masks[name]],
            rtol=0.0,
            atol=1e-5,
        )
    # Oakes standard errors agree with R's to its printed precision.
    for name, values in _R_STANDARD_ERRORS.items():
        np.testing.assert_allclose(
            result.standard_errors[name][masks[name]],
            np.asarray(values)[masks[name]],
            rtol=2e-4,
        )
    # bfactor() picks the graded model for these Likert responses.
    fitted = mirt.bfactor(
        responses,
        labels,
        n_quadpts=31,
        quadrature="rectangular",
        max_iter=2,
        start_values=_R_ESTIMATES,
        compute_standard_errors=False,
    )
    assert type(fitted.model) is GradedBifactorModel
    assert fitted.model.n_categories == [4, 4, 2, 4, 3, 4]
    assert fitted.refit_recipe.options["quadrature"] == "rectangular"


def test_bfactor_chooses_the_item_type_and_validates_it() -> None:
    _, responses = _likert(10, 200, [0, 0, 1, 1], [4, 3, 4, 3])
    options = {"n_quadpts": 7, "max_iter": 3, "compute_standard_errors": False}

    assert type(bfactor(responses, [0, 0, 1, 1], **options).model) is (
        GradedBifactorModel
    )
    assert type(bfactor(responses, [0, 0, 1, 1], model="GRM", **options).model) is (
        GradedBifactorModel
    )
    with pytest.raises(MirtDataError, match="dichotomous"):
        bfactor(responses, [0, 0, 1, 1], model="2PL", **options)
    with pytest.raises(MirtValidationError, match="model") as error:
        bfactor(responses, [0, 0, 1, 1], model="gpcm", **options)
    assert error.value.context["parameter"] == "model"
    with pytest.raises(MirtValidationError, match="specific_factors"):
        bfactor(responses, [0, 1, 0], **options)
    single = responses.copy()
    single[:, 2] = 1
    with pytest.raises(MirtDataError, match="one observed response category"):
        bfactor(single, [0, 0, 1, 1], **options)
    with pytest.raises(MirtDataError, match="must be below"):
        BifactorEMEstimator(n_quadpts=5).fit(
            GradedBifactorModel(4, [0, 0, 1, 1], 3), responses
        )


def test_em_increases_the_likelihood_and_errors_invert_the_information() -> None:
    _, responses = _likert(24, 400, _LABELS, _CATEGORIES, missing=0.05)
    estimator = BifactorEMEstimator(n_quadpts=9, tol=1e-6, max_iter=1000)
    result = estimator.fit(GradedBifactorModel(7, _LABELS, _CATEGORIES), responses)

    history = np.asarray(estimator.convergence_history)
    assert np.all(np.diff(history) > -1e-8)
    assert result.converged and result.se_method == "oakes"
    # The covariance inverts the product-grid observed information.
    problem = bifactor_em._GradedBifactorProblem(result.model)
    layouts, _ = _order(problem, result.model)
    quadrature = GaussHermiteQuadrature(9, 3)
    mass = quadrature.weights / quadrature.weights.sum()
    reference = louis_information(
        result.model, responses, quadrature.nodes, mass, layouts
    )
    np.testing.assert_allclose(
        result.vcov, np.linalg.inv(reference.information), rtol=1e-4, atol=1e-8
    )
    assert result.vcov_labels[0] == "general_loadings[Item_0]"
    assert "intercepts[Item_6,3]" in result.vcov_labels
    for method in ("crossprod", "sandwich", "complete_data"):
        errors = BifactorEMEstimator(n_quadpts=9, se_method=method, max_iter=1).fit(
            result.model.copy(), responses, start="model"
        )
        assert errors.se_method == method
        assert np.all(
            np.isfinite(
                errors.standard_errors["intercepts"][
                    result.model.free_parameter_masks["intercepts"]
                ]
            )
        )


def test_graded_fit_round_trips_and_reports_coefficients() -> None:
    _, responses = _likert(12, 250, _LABELS, _CATEGORIES)
    free = GradedBifactorModel(7, _LABELS, _CATEGORIES).free_parameter_masks
    result = bfactor(
        responses,
        _LABELS,
        n_quadpts=7,
        max_iter=50,
        fixed={"specific_loadings": ~free["specific_loadings"] | (np.arange(7) == 6)},
        start_values={"specific_loadings": np.r_[np.full(6, 0.5), 0.0]},
    )

    assert result.model.specific_loadings[6] == 0.0
    restored = FitResult.from_json(result.to_json())
    payload = json.loads(result.to_json())
    assert payload["model"]["name"] == "GradedBifactor"
    assert payload["model"]["n_categories"] == _CATEGORIES
    assert payload["model"]["specific_factors"] == _LABELS
    assert type(restored.model) is GradedBifactorModel
    assert restored.model.n_categories == _CATEGORIES
    assert not restored.model.free_parameter_masks["specific_loadings"][6]
    for name, values in result.model.parameters.items():
        np.testing.assert_array_equal(restored.model.parameters[name], values)
    assert restored.vcov_labels == result.vcov_labels
    np.testing.assert_array_equal(
        mirt.fscores(restored, responses).theta, mirt.fscores(result, responses).theta
    )
    coefficients = result.coef_with_se()
    columns = [
        name
        for name in getattr(coefficients, "columns", coefficients)
        if name != "item"
    ]
    assert columns[:3] == [
        "general_loadings",
        "general_loadings_se",
        "specific_loadings",
    ]
    assert "intercepts_4" in columns
    assert "GradedBifactor" in result.summary()
    # Refits without a recipe, as in the bootstrap, keep the reduced grids.
    assert type(em_estimator_for(restored, max_iter=2)) is BifactorEMEstimator
    with pytest.raises(MirtValidationError, match="n_categories"):
        broken = result.to_dict()
        broken["model"]["n_categories"] = None
        FitResult.from_dict(broken)
