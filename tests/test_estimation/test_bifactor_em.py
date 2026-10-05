"""Exact bifactor EM by dimension reduction, its inference and entry points."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

import mirt
from mirt.constants import PROB_EPSILON
from mirt.estimation import bifactor_em
from mirt.estimation import em as em_module
from mirt.estimation._em_context import EMFitContext
from mirt.estimation._louis_information import louis_information
from mirt.estimation.bifactor_em import BifactorEMEstimator, bfactor
from mirt.estimation.em import EMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.standard_errors import _flatten_parameters, _posterior_from_model
from mirt.exceptions import MirtDataError, MirtModelError, MirtValidationError
from mirt.models.bifactor import BifactorModel
from mirt.models.multidimensional import MultidimensionalModel


def _simulate(seed, n_persons, labels, *, missing=0.0, general_only=()):
    """Draw bifactor responses; ``general_only`` items have no specific loading."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    _, factors = np.unique(labels, return_inverse=True)
    n_items = labels.size
    model = BifactorModel(n_items, labels)
    specific = rng.uniform(0.6, 1.3, n_items)
    specific[list(general_only)] = 0.0
    model.set_parameters(
        general_loadings=rng.uniform(0.8, 1.6, n_items),
        specific_loadings=specific,
        intercepts=rng.normal(0.0, 0.8, n_items),
    )
    theta = rng.standard_normal((n_persons, model.n_factors))
    responses = (rng.random((n_persons, n_items)) < model.probability(theta)).astype(
        int
    )
    responses[rng.random(responses.shape) < missing] = -1
    assert np.array_equal(model._specific_factor_indices, factors)
    return model, responses


def _product_log_marginals(model, responses, n_quadpts):
    quadrature = GaussHermiteQuadrature(n_quadpts, model.n_factors)
    log_likelihood = model.log_likelihood_batch(responses, quadrature.nodes)
    return logsumexp(log_likelihood + np.log(quadrature.weights), axis=1)


def _reduced_terms(model, responses, n_quadpts, *, observed=True):
    grid = bifactor_em._Grid.build(n_quadpts, model)
    with EMFitContext(responses, compress=True) as context:
        terms = bifactor_em._louis_information(model, grid, context, observed=observed)
        frequencies = context.frequencies
    _, layouts = _flatten_parameters(model)
    order = np.concatenate(
        [
            3 * layouts[name].free_indices + column
            for column, name in enumerate(bifactor_em._PARAMETERS)
        ]
    )
    return terms, np.ix_(order, order), order, frequencies


def _product_louis(model, responses, n_quadpts):
    quadrature = GaussHermiteQuadrature(n_quadpts, model.n_factors)
    _, layouts = _flatten_parameters(model)
    mass = quadrature.weights / quadrature.weights.sum()
    return louis_information(model, responses, quadrature.nodes, mass, layouts)


def _exact_product_louis(model, responses, n_quadpts):
    """Louis terms on the product grid with analytic logistic derivatives.

    Coordinate ``c`` of item ``j`` is ``3j + c``. Derivatives vanish where
    the likelihood's probability clipping binds.
    """
    quadrature = GaussHermiteQuadrature(n_quadpts, model.n_factors)
    nodes = quadrature.nodes
    n_nodes, n_items = nodes.shape[0], model.n_items
    design = np.stack(
        (
            np.repeat(nodes[:, :1], n_items, axis=1),
            nodes[:, 1 + model._specific_factor_indices],
            np.ones((n_nodes, n_items)),
        ),
        axis=-1,
    )
    log_posterior = model.log_likelihood_batch(responses, nodes) + np.log(
        quadrature.weights / quadrature.weights.sum()
    )
    posterior = np.exp(log_posterior - logsumexp(log_posterior, axis=1)[:, None])
    probability = model.probability(nodes)
    active = (probability > PROB_EPSILON) & (probability < 1.0 - PROB_EPSILON)
    observed = responses >= 0
    correct = np.where(observed, responses, 0)
    residual = (correct[:, None] - observed[:, None] * probability) * active
    scores = (residual[..., None] * design).reshape(len(responses), n_nodes, -1)
    means = np.einsum("ng,ngp->np", posterior, scores)
    second_moment = np.einsum("ng,ngp,ngq->pq", posterior, scores, scores)
    curvature = (posterior.T @ observed) * probability * (1.0 - probability) * active
    blocks = np.einsum("gj,gjc,gjd->jcd", curvature, design, design)
    information = means.T @ means - second_moment
    for item in range(n_items):
        information[3 * item : 3 * item + 3, 3 * item : 3 * item + 3] += blocks[item]
    return information, means.T @ means


def test_reduced_log_likelihood_matches_product_grid() -> None:
    model, responses = _simulate(0, 240, [5, 2, 5, 2, 5, 2, 2, 5], missing=0.1)
    grid = bifactor_em._Grid.build(9, model)
    curves = bifactor_em._Curves.evaluate(grid, bifactor_em._coefficients(model))
    with EMFitContext(responses) as context:
        blocks = list(bifactor_em._posterior_blocks(grid, curves, context, 37))
    reduced = np.concatenate([block.log_marginal for block in blocks])

    product = _product_log_marginals(model, responses, 9)

    np.testing.assert_allclose(reduced, product, rtol=0.0, atol=1e-10)
    for block in blocks:
        np.testing.assert_allclose(block.general.sum(axis=1), 1.0, atol=1e-12)
        for conditional in block.conditionals:
            np.testing.assert_allclose(conditional.sum(axis=2), 1.0, atol=1e-12)


def test_fit_reports_the_product_grid_log_likelihood() -> None:
    model, responses = _simulate(1, 300, [0, 0, 0, 1, 1, 1])
    start = model.parameters
    estimator = BifactorEMEstimator(
        n_quadpts=9, max_iter=1, tol=1e-12, compute_standard_errors=False
    )
    result = estimator.fit(model.copy(), responses, start=start)
    initial = _product_log_marginals(model, responses, 9).sum()
    final = _product_log_marginals(result.model, responses, 9).sum()

    np.testing.assert_allclose(estimator.convergence_history[0], initial, rtol=1e-12)
    np.testing.assert_allclose(result.log_likelihood, final, rtol=1e-12)
    assert result.log_likelihood > initial
    assert result.n_iterations == 1
    assert not result.converged


def test_em_iterations_match_product_grid_em() -> None:
    labels = np.array([4, 1, 4, 1, 4, 1, 4, 1])
    _, responses = _simulate(5, 400, labels, missing=0.05)
    product = EMEstimator(
        n_quadpts=11,
        max_iter=5,
        tol=1e-12,
        use_rust=False,
        use_gpu=False,
        compute_standard_errors=False,
        item_optim_ftol=1e-15,
        item_optim_maxiter=1000,
    )
    reference = product.fit(BifactorModel(8, labels), responses)
    reduced = BifactorEMEstimator(
        n_quadpts=11, max_iter=5, tol=1e-12, compute_standard_errors=False
    )
    result = reduced.fit(BifactorModel(8, labels), responses)

    np.testing.assert_allclose(
        reduced.convergence_history, product.convergence_history, rtol=1e-9
    )
    for name, values in reference.model.parameters.items():
        np.testing.assert_allclose(
            result.model.parameters[name], values, rtol=0.0, atol=1e-6
        )
    assert result.n_iterations == reference.n_iterations == 5


def test_seeded_recovery_with_four_specific_factors() -> None:
    rng = np.random.default_rng(1)
    labels = np.repeat([10, 20, 30, 40], 5)
    n_items, n_persons = labels.size, 2000
    truth = {
        "general_loadings": rng.uniform(1.0, 1.8, n_items),
        "specific_loadings": rng.uniform(0.8, 1.4, n_items),
        "intercepts": rng.normal(0.0, 0.8, n_items),
    }
    theta = rng.standard_normal((n_persons, 5))
    logits = (
        truth["general_loadings"] * theta[:, :1]
        + truth["specific_loadings"] * theta[:, 1 + np.repeat(np.arange(4), 5)]
        + truth["intercepts"]
    )
    responses = (rng.random(logits.shape) < expit(logits)).astype(int)

    result = bfactor(responses, labels)

    # The product grid would need 21**5 = 4,084,101 nodes per person.
    assert result.model.n_factors == 5
    assert result.converged
    assert result.se_method == "oakes"
    limits = {"general_loadings": 0.2, "specific_loadings": 0.25, "intercepts": 0.12}
    for name, true in truth.items():
        estimate = result.model.parameters[name]
        error = result.standard_errors[name]
        assert np.sqrt(np.mean((estimate - true) ** 2)) < limits[name]
        assert np.all(np.isfinite(error)) and np.all(error > 0.0)
        assert np.max(np.abs(estimate - true) / error) < 4.0


def test_relabelled_specific_factors_fit_identically() -> None:
    contiguous = np.array([0, 1, 0, 1, 2, 2, 0, 1])
    relabelled = np.array([7, 3, 7, 3, 12, 12, 7, 3])
    _, responses = _simulate(2, 350, contiguous, missing=0.05)

    # Two-item specific factors converge slowly, so a fixed iteration count
    # keeps the comparison fast.
    def fit(labels):
        return BifactorEMEstimator(n_quadpts=9, max_iter=40, tol=1e-12).fit(
            BifactorModel(8, labels), responses
        )

    first, second = fit(contiguous), fit(relabelled)

    np.testing.assert_allclose(second.log_likelihood, first.log_likelihood, rtol=1e-12)
    for name, values in first.model.parameters.items():
        np.testing.assert_allclose(second.model.parameters[name], values, atol=1e-8)
        np.testing.assert_allclose(
            second.standard_errors[name], first.standard_errors[name], rtol=1e-6
        )
    assert second.model.get_factor_structure() == {
        3: [1, 3, 7],
        7: [0, 2, 6],
        12: [4, 5],
    }


def test_items_without_specific_loading_use_the_general_factor_only() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1, 1])
    general_only = (2, 6)
    _, responses = _simulate(3, 500, labels, general_only=general_only)
    fixed = np.zeros(7, dtype=bool)
    fixed[list(general_only)] = True
    start = {"specific_loadings": np.where(fixed, 0.0, 0.5)}

    model = BifactorModel(7, labels).set_free_parameter_masks(
        {"specific_loadings": ~fixed}
    )
    result = BifactorEMEstimator(n_quadpts=11, max_iter=40, tol=1e-12).fit(
        model, responses, start=start
    )
    via_entry = bfactor(
        responses,
        labels,
        n_quadpts=11,
        max_iter=40,
        tol=1e-12,
        start_values=start,
        fixed={"specific_loadings": fixed},
    )

    specific = result.model.parameters["specific_loadings"]
    assert np.all(specific[fixed] == 0.0)
    assert np.all(result.standard_errors["specific_loadings"][fixed] == 0.0)
    assert result.vcov.shape == (19, 19)
    assert result.n_parameters == 19
    for name, values in result.model.parameters.items():
        np.testing.assert_allclose(via_entry.model.parameters[name], values)
    np.testing.assert_allclose(
        result.log_likelihood,
        _product_log_marginals(result.model, responses, 11).sum(),
        rtol=1e-12,
    )
    terms, block, _, _ = _reduced_terms(result.model, responses, 7)
    reference = _product_louis(result.model, responses, 7)
    scale = np.abs(reference.information).max()
    np.testing.assert_allclose(
        terms.information[block], reference.information, rtol=0.0, atol=1e-7 * scale
    )


def test_observed_information_matches_product_grid_louis() -> None:
    model, responses = _simulate(4, 900, [1, 6, 1, 6, 1, 6], missing=0.05)
    terms, block, _, frequencies = _reduced_terms(model, responses, 7)
    reference = _product_louis(model, responses, 7)

    # Repeated patterns are compressed, so the reduced terms use frequencies.
    # The product-grid reference differences item curves numerically.
    assert frequencies is not None
    for mine, theirs in (
        (terms.information, reference.information),
        (terms.score_crossproduct, reference.score_crossproduct),
    ):
        scale = np.abs(theirs).max()
        np.testing.assert_allclose(mine[block], theirs, rtol=0.0, atol=1e-7 * scale)
    information, crossproduct = _exact_product_louis(model, responses, 7)
    for mine, theirs in (
        (terms.information, information),
        (terms.score_crossproduct, crossproduct),
    ):
        scale = np.abs(theirs).max()
        np.testing.assert_allclose(mine, theirs, rtol=0.0, atol=1e-11 * scale)
    crossproduct_only, _, _, _ = _reduced_terms(model, responses, 7, observed=False)
    assert crossproduct_only.information is None
    np.testing.assert_allclose(
        crossproduct_only.score_crossproduct, terms.score_crossproduct, rtol=1e-12
    )


def test_reduced_terms_follow_probability_clipping_at_extreme_parameters() -> None:
    model, responses = _simulate(12, 200, [0, 0, 1, 1, 1, 2], missing=0.1)
    model.set_parameters(
        general_loadings=np.array([5.9, -4.0, 3.0, 0.2, 5.5, 1.0]),
        specific_loadings=np.array([5.0, 4.0, -5.8, 0.0, 3.0, 2.0]),
        intercepts=np.array([5.9, -5.9, 0.0, 2.0, -3.0, 0.5]),
    )
    grid = bifactor_em._Grid.build(7, model)
    curves = bifactor_em._Curves.evaluate(grid, bifactor_em._coefficients(model))
    with EMFitContext(responses) as context:
        blocks = list(bifactor_em._posterior_blocks(grid, curves, context, 64))
    reduced = np.concatenate([block.log_marginal for block in blocks])
    terms, _, _, _ = _reduced_terms(model, responses, 7)
    information, crossproduct = _exact_product_louis(model, responses, 7)

    assert not np.all(curves.active)
    np.testing.assert_allclose(
        reduced, _product_log_marginals(model, responses, 7), rtol=0.0, atol=1e-10
    )
    for mine, theirs in (
        (terms.information, information),
        (terms.score_crossproduct, crossproduct),
    ):
        scale = np.abs(theirs).max()
        np.testing.assert_allclose(mine, theirs, rtol=0.0, atol=1e-11 * scale)


@pytest.mark.parametrize("method", ["auto", "oakes", "crossprod", "sandwich"])
def test_standard_errors_invert_the_exact_information(method: str) -> None:
    model, responses = _simulate(6, 800, [0, 1, 0, 1, 0, 1, 0, 1])
    result = BifactorEMEstimator(n_quadpts=7, tol=1e-8, se_method=method).fit(
        model.copy(), responses, start=model.parameters
    )
    reference = _product_louis(result.model, responses, 7)
    bread = np.linalg.inv(reference.information)
    expected = {
        "auto": bread,
        "oakes": bread,
        "crossprod": np.linalg.inv(reference.score_crossproduct),
        "sandwich": bread @ reference.score_crossproduct @ bread,
    }[method]

    assert result.se_method == ("oakes" if method == "auto" else method)
    # The product-grid reference differences item curves numerically.
    scale = np.abs(expected).max()
    np.testing.assert_allclose(result.vcov, expected, rtol=0.0, atol=1e-4 * scale)
    errors = np.concatenate(
        [result.standard_errors[name] for name in bifactor_em._PARAMETERS]
    )
    np.testing.assert_allclose(errors, np.sqrt(np.diag(expected)), rtol=1e-4)
    assert result.vcov_labels[0] == f"general_loadings[{model.item_names[0]}]"


def test_complete_data_standard_errors_match_product_grid_curvature() -> None:
    model, responses = _simulate(7, 250, [0, 1, 0, 1, 1])
    result = BifactorEMEstimator(n_quadpts=7, tol=1e-6, se_method="complete_data").fit(
        model.copy(), responses
    )
    fitted = result.model
    quadrature = GaussHermiteQuadrature(7, fitted.n_factors)
    posterior = _posterior_from_model(fitted, responses, quadrature)
    nodes = quadrature.nodes
    observed = (responses >= 0).astype(float).T @ posterior
    probability = fitted.probability(nodes)
    for item in range(fitted.n_items):
        design = np.column_stack(
            (
                nodes[:, 0],
                nodes[:, 1 + fitted._specific_factor_indices[item]],
                np.ones(nodes.shape[0]),
            )
        )
        weight = observed[item] * probability[:, item] * (1.0 - probability[:, item])
        curvature = np.einsum("q,qc,qc->c", weight, design, design)
        for column, name in enumerate(bifactor_em._PARAMETERS):
            np.testing.assert_allclose(
                result.standard_errors[name][item],
                1.0 / np.sqrt(curvature[column]),
                rtol=1e-9,
            )
    assert result.se_method == "complete_data"
    assert result.vcov is None


def test_m_step_matches_bounded_item_maximization() -> None:
    model, _ = _simulate(8, 10, [0, 0, 1, 1, 1])
    grid = bifactor_em._Grid.build(7, model)
    rng = np.random.default_rng(8)
    observed = rng.uniform(1.0, 20.0, (5, grid.design.shape[0]))
    correct = observed * rng.uniform(0.05, 0.95, observed.shape)
    # The last item is answered correctly everywhere, so its optimum lies
    # on the intercept bound.
    correct[-1] = observed[-1]
    fixed = np.array([False, True, False, False, False])
    model.set_free_parameter_masks({"general_loadings": ~fixed})
    start = bifactor_em._coefficients(model)

    bifactor_em._m_step(model, grid, correct, observed)
    updated = bifactor_em._coefficients(model)

    def reference(item):
        free = np.array([not fixed[item], True, True])

        def objective(params):
            trial = start[item].copy()
            trial[free] = params
            logits = grid.design @ trial
            return np.sum(
                correct[item] * np.logaddexp(0.0, -logits)
                + (observed[item] - correct[item]) * np.logaddexp(0.0, logits)
            )

        solution = minimize(
            objective,
            start[item][free],
            method="L-BFGS-B",
            bounds=[(-6.0, 6.0)] * int(free.sum()),
            options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 1000},
        )
        trial = start[item].copy()
        trial[free] = solution.x
        return trial

    for item in range(5):
        np.testing.assert_allclose(updated[item], reference(item), atol=2e-4)
    assert updated[1, 0] == start[1, 0]
    assert updated[-1, 2] == pytest.approx(6.0)


def test_bfactor_entry_point_and_lazy_exports() -> None:
    pd = pytest.importorskip("pandas")
    _, responses = _simulate(9, 200, [0, 0, 1, 1])
    frame = pd.DataFrame(responses, columns=["a", "b", "c", "d"])

    result = mirt.bfactor(
        frame, [3, 3, 8, 8], n_quadpts=7, compute_standard_errors=False
    )

    assert mirt.bfactor is bfactor
    assert mirt.BifactorEMEstimator is BifactorEMEstimator
    assert mirt.estimation.bfactor is bfactor
    assert mirt.estimation.BifactorEMEstimator is BifactorEMEstimator
    assert isinstance(result.model, BifactorModel)
    assert result.model.item_names == ["a", "b", "c", "d"]
    assert result.standard_errors == {}
    assert result.se_method is None
    with pytest.raises(MirtValidationError, match="specific_factors") as error:
        bfactor(responses, [0, 1, 0])
    assert error.value.context["parameter"] == "specific_factors"
    with pytest.raises(MirtValidationError, match="item_names") as error:
        bfactor(responses, [0, 0, 1, 1], item_names=["a", "b"])
    assert error.value.context["parameter"] == "item_names"


def test_estimator_rejects_unsupported_models_and_data() -> None:
    model, responses = _simulate(10, 50, [0, 0, 1, 1])
    estimator = BifactorEMEstimator(n_quadpts=5, max_iter=2)

    with pytest.raises(MirtModelError, match="BifactorModel"):
        estimator.fit(MultidimensionalModel(4, 2), responses)
    polytomous = responses.copy()
    polytomous[0, 0] = 2
    with pytest.raises(MirtDataError, match="dichotomous"):
        estimator.fit(model.copy(), polytomous)
    for value in (4, 7.0, True):
        with pytest.raises(MirtValidationError, match="n_quadpts"):
            BifactorEMEstimator(n_quadpts=value)
    with pytest.raises(MirtValidationError, match="se_method"):
        BifactorEMEstimator(se_method="louis")


def test_missing_rows_and_items_without_responses_keep_starting_values() -> None:
    model, responses = _simulate(11, 120, [0, 0, 1, 1, 1])
    responses[:, 4] = -1
    responses[:5] = -1
    start = model.parameters

    result = BifactorEMEstimator(n_quadpts=7, tol=1e-6).fit(
        model.copy(), responses, start=start
    )

    for name in bifactor_em._PARAMETERS:
        assert result.model.parameters[name][4] == start[name][4]
        assert np.isnan(result.standard_errors[name][4])
    assert np.isfinite(result.log_likelihood)
    np.testing.assert_allclose(
        result.log_likelihood,
        _product_log_marginals(result.model, responses, 7).sum(),
        rtol=1e-12,
    )


def test_em_warns_before_building_a_large_product_grid(monkeypatch) -> None:
    class Stop(Exception):
        pass

    def stop(*args, **kwargs):
        raise Stop

    monkeypatch.setattr(em_module, "GaussHermiteQuadrature", stop)
    responses = np.zeros((4, 8), dtype=int)

    with pytest.warns(RuntimeWarning, match=r"4,084,101 nodes.*mirt\.bfactor"):
        with pytest.raises(Stop):
            EMEstimator().fit(BifactorModel(8, np.arange(8) % 4), responses)
    with pytest.warns(RuntimeWarning, match=r"1,048,576 nodes.*reduce n_quadpts"):
        with pytest.raises(Stop):
            EMEstimator(n_quadpts=16).fit(MultidimensionalModel(8, 5), responses)
    # 21 ** 15 wraps to a negative value in 64-bit NumPy integer arithmetic.
    with pytest.warns(RuntimeWarning, match=r"68,122,318,582,951,682,301 nodes"):
        with pytest.raises(Stop):
            EMEstimator(n_quadpts=np.int64(21)).fit(
                MultidimensionalModel(8, 15), responses
            )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(Stop):
            EMEstimator().fit(BifactorModel(8, np.arange(8) % 3), responses)
