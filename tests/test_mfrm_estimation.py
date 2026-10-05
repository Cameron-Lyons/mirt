"""Marginal maximum likelihood estimation of many-facet Rasch models."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mirt
from mirt.estimation import fit_mfrm
from mirt.estimation.mfrm_mml import (
    _build_design,
    _MarginalLikelihood,
    _natural_covariance,
    _parameter_map,
    _starting_values,
)
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models.mfrm import Facet, ManyFacetRaschModel, MFRMResult, PolytomousMFRM

ITEMS = np.linspace(-0.8, 0.8, 6)
RATERS = np.array([-0.5, -0.1, 0.2, 0.4])
TASKS = np.array([0.8, 0.2])
RATING_SCALE = np.array([-1.0, -0.2, 0.3, 0.9])
PARTIAL_CREDIT = np.array(
    [
        [-1.2, 0.1, 0.4, 0.7],
        [-0.8, -0.3, 0.5, 0.6],
        [-1.0, 0.0, 0.2, 0.8],
        [-0.6, -0.4, 0.3, 0.7],
        [-1.1, 0.2, 0.3, 0.6],
        [-0.9, -0.1, 0.4, 0.6],
    ]
)


def _facets() -> list[Facet]:
    return [Facet("rater", 4), Facet("task", 2, anchor_value=0.5)]


def _model(structure: str) -> ManyFacetRaschModel:
    if structure == "binary":
        return ManyFacetRaschModel(len(ITEMS), _facets())
    return PolytomousMFRM(len(ITEMS), 5, _facets(), category_structure=structure)


def _true_model(structure: str) -> ManyFacetRaschModel:
    model = _model(structure)
    model.set_item_difficulty(ITEMS)
    model.set_facet_parameters("rater", RATERS)
    model.set_facet_parameters("task", TASKS)
    if isinstance(model, PolytomousMFRM):
        model.set_thresholds(
            RATING_SCALE if structure == "rating_scale" else PARTIAL_CREDIT
        )
    return model


def _wide_data(
    structure: str, n_persons: int, seed: int
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 1.2, n_persons)
    assignments = {
        "rater": rng.integers(0, len(RATERS), (n_persons, len(ITEMS))),
        "task": rng.integers(0, len(TASKS), n_persons),
    }
    responses = _true_model(structure).simulate(theta, assignments, seed=seed + 1)
    return responses, assignments, theta


def _repeated_rater_data(
    structure: str, n_persons: int, seed: int
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Two distinct raters score every person on every item."""
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 1.2, n_persons)
    first = rng.integers(0, 4, (n_persons, len(ITEMS)))
    second = (first + rng.integers(1, 4, first.shape)) % 4
    raters = np.stack((first, second), axis=2).reshape(n_persons, -1)
    item_indices = np.repeat(np.arange(len(ITEMS)), 2)
    task = rng.integers(0, 2, n_persons)
    truth = _true_model(structure)
    responses = np.empty(raters.shape, dtype=np.int64)
    for column, item in enumerate(item_indices):
        simulated = truth.simulate(
            theta, {"rater": raters[:, column], "task": task}, seed=seed + column
        )
        responses[:, column] = simulated[:, item]
    return responses, {"rater": raters, "task": task}, item_indices


def _within(estimate, truth, standard_error, z: float = 4.0) -> None:
    deviation = np.abs(np.asarray(estimate) - np.asarray(truth))
    assert np.all(deviation <= z * np.asarray(standard_error)), (
        deviation,
        standard_error,
    )


def test_binary_rater_by_item_design_recovers_parameters() -> None:
    responses, assignments, theta = _wide_data("binary", 1000, seed=3)
    model = _model("binary")

    result = fit_mfrm(model, responses, assignments)

    assert isinstance(result, MFRMResult)
    assert result.converged
    assert result.thresholds is None and result.threshold_se is None
    _within(result.item_difficulty, ITEMS, result.item_se)
    _within(result.facet_parameters["rater"], RATERS, result.facet_se["rater"])
    _within(result.facet_parameters["task"], TASKS, result.facet_se["task"])
    assert abs(result.sigma - 1.2) < 4 * result.sigma_se
    assert np.all(result.facet_se["rater"] > 0) and result.sigma_se > 0
    assert np.corrcoef(result.theta, theta)[0, 1] > 0.75
    assert np.all(result.theta_se > 0) and np.all(result.theta_se < result.sigma)
    assert result.n_observations == responses.size
    assert result.n_parameters == len(ITEMS) + 3 + 1 + 1
    # Six binary ratings per person leave wide posteriors: no refinement.
    assert result.n_quadpts == 41


@pytest.mark.parametrize("structure", ["rating_scale", "partial_credit"])
def test_polytomous_thresholds_are_recovered(structure: str) -> None:
    responses, assignments, _ = _wide_data(structure, 800, seed=11)

    result = fit_mfrm(_model(structure), responses, assignments)

    truth = RATING_SCALE if structure == "rating_scale" else PARTIAL_CREDIT
    assert result.converged
    assert result.thresholds.shape == truth.shape
    _within(result.thresholds, truth, result.threshold_se)
    _within(result.item_difficulty, ITEMS, result.item_se)
    _within(result.facet_parameters["rater"], RATERS, result.facet_se["rater"])
    assert_allclose(result.thresholds.sum(axis=-1), 0.0, atol=1e-10)


@pytest.mark.parametrize("structure", ["binary", "rating_scale", "partial_credit"])
def test_repeated_rater_long_layout_recovers_parameters(structure: str) -> None:
    responses, assignments, item_indices = _repeated_rater_data(structure, 600, seed=5)

    result = fit_mfrm(
        _model(structure), responses, assignments, item_indices=item_indices
    )

    assert result.converged
    assert result.n_observations == responses.size
    _within(result.item_difficulty, ITEMS, result.item_se)
    _within(result.facet_parameters["rater"], RATERS, result.facet_se["rater"])
    _within(result.facet_parameters["task"], TASKS, result.facet_se["task"])


def test_long_layout_matches_wide_layout() -> None:
    responses, assignments, _ = _wide_data("rating_scale", 300, seed=21)
    wide = fit_mfrm(_model("rating_scale"), responses, assignments)

    rng = np.random.default_rng(0)
    order = rng.permutation(len(ITEMS))
    long = fit_mfrm(
        _model("rating_scale"),
        responses[:, order],
        {"rater": assignments["rater"][:, order], "task": assignments["task"]},
        item_indices=np.tile(order, (300, 1)),
    )

    assert_allclose(long.log_likelihood, wide.log_likelihood, rtol=1e-12)
    assert_allclose(long.item_difficulty, wide.item_difficulty, atol=1e-6)
    assert_allclose(long.facet_se["rater"], wide.facet_se["rater"], rtol=1e-4)
    assert_allclose(long.outfit["rater"], wide.outfit["rater"], rtol=1e-6)


def test_indices_at_missing_ratings_are_ignored() -> None:
    responses, assignments, item_indices = _repeated_rater_data(
        "rating_scale", 300, seed=8
    )
    rng = np.random.default_rng(1)
    missing = rng.random(responses.shape) < 0.15
    filled = np.where(missing, -1, responses)
    reference = fit_mfrm(
        _model("rating_scale"), filled, assignments, item_indices=item_indices
    )

    padded_raters = np.where(missing, -7, assignments["rater"])
    padded_items = np.where(missing, -1, np.tile(item_indices, (300, 1)))
    result = fit_mfrm(
        _model("rating_scale"),
        np.where(missing, np.nan, responses.astype(float)),
        {"rater": padded_raters, "task": assignments["task"]},
        item_indices=padded_items,
    )

    assert result.n_observations == int((~missing).sum())
    assert_allclose(result.log_likelihood, reference.log_likelihood, rtol=1e-12)
    assert_allclose(
        result.facet_parameters["rater"], reference.facet_parameters["rater"]
    )


@pytest.mark.parametrize("structure", ["binary", "rating_scale", "partial_credit"])
@pytest.mark.parametrize("estimate_sd", [True, False])
def test_analytic_gradient_matches_finite_differences(
    structure: str, estimate_sd: bool
) -> None:
    responses, assignments, item_indices = _repeated_rater_data(structure, 150, seed=2)
    model = _model(structure)
    n_categories = 2 if structure == "binary" else 5
    design = _build_design(model, responses, assignments, item_indices)
    parameter_map = _parameter_map(model, n_categories, estimate_sd)
    likelihood = _MarginalLikelihood(design, parameter_map, 11)
    start = parameter_map.free(_starting_values(model, design, parameter_map))
    point = start + np.random.default_rng(4).normal(0.0, 0.3, start.size)

    _, gradient, _ = likelihood.evaluate(point)

    step = 1e-5
    numerical = np.empty_like(point)
    for index in range(point.size):
        shift = np.zeros_like(point)
        shift[index] = step
        upper, _, _ = likelihood.evaluate(point + shift, gradient=False)
        lower, _, _ = likelihood.evaluate(point - shift, gradient=False)
        numerical[index] = (upper - lower) / (2 * step)
    assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("structure", ["binary", "rating_scale", "partial_credit"])
@pytest.mark.parametrize("estimate_sd", [True, False])
def test_observed_information_matches_finite_differences(
    structure: str, estimate_sd: bool
) -> None:
    responses, assignments, item_indices = _repeated_rater_data(structure, 150, seed=2)
    responses = np.where(
        np.random.default_rng(3).random(responses.shape) < 0.2, -1, responses
    )
    model = _model(structure)
    n_categories = 2 if structure == "binary" else 5
    design = _build_design(model, responses, assignments, item_indices)
    parameter_map = _parameter_map(model, n_categories, estimate_sd)
    likelihood = _MarginalLikelihood(design, parameter_map, 21)
    start = parameter_map.free(_starting_values(model, design, parameter_map))
    # Away from the optimum, so the score terms of Louis' identity matter.
    point = start + np.random.default_rng(4).normal(0.0, 0.3, start.size)

    information = likelihood.information(point)

    # Reference: central differences of the (separately tested) gradient.
    numerical = np.empty((point.size, point.size))
    for index in range(point.size):
        step = 1e-4 * max(1.0, abs(point[index]))
        shift = np.zeros_like(point)
        shift[index] = step
        _, upper, _ = likelihood.evaluate(point + shift)
        _, lower, _ = likelihood.evaluate(point - shift)
        numerical[:, index] = (lower - upper) / (2 * step)
    assert_allclose(information, information.T, rtol=1e-12, atol=1e-9)
    assert_allclose(
        information, numerical, rtol=1e-6, atol=1e-6 * np.abs(numerical).max()
    )


def test_facet_free_binary_model_matches_em_one_parameter_logistic() -> None:
    rng = np.random.default_rng(2)
    truth = ManyFacetRaschModel(5, []).set_item_difficulty(np.linspace(-1, 1, 5))
    responses = truth.simulate(rng.normal(size=600), seed=3)

    result = fit_mfrm(ManyFacetRaschModel(5, []), responses, estimate_sd=False)
    reference = mirt.fit_mirt(responses, model="1PL", tol=1e-6)

    assert result.sigma == 1.0 and result.sigma_se == 0.0
    assert_allclose(result.log_likelihood, reference.log_likelihood, atol=1e-6)
    assert_allclose(result.item_difficulty, reference.model.difficulty, atol=1e-4)
    assert_allclose(result.item_se, reference.standard_errors["difficulty"], rtol=1e-4)


def test_facet_free_partial_credit_model_matches_em_pcm() -> None:
    rng = np.random.default_rng(2)
    truth = PolytomousMFRM(5, 4, [], category_structure="partial_credit")
    truth.set_item_difficulty(np.linspace(-0.6, 0.6, 5))
    truth.set_thresholds(np.tile([-0.8, 0.1, 0.7], (5, 1)))
    responses = truth.simulate(rng.normal(size=600), seed=3)

    result = fit_mfrm(
        PolytomousMFRM(5, 4, [], category_structure="partial_credit"),
        responses,
        estimate_sd=False,
    )
    # 21 Gauss-Hermite nodes are 6e-3 off this log-likelihood; 61 resolve it.
    reference = mirt.fit_mirt(responses, model="PCM", tol=1e-6, n_quadpts=61)

    # Adjacent-category steps of the PCM are b_i + tau_ij here.
    steps = result.item_difficulty[:, None] + result.thresholds
    assert_allclose(result.log_likelihood, reference.log_likelihood, atol=1e-6)
    assert_allclose(steps, reference.model.steps, atol=1e-4)


def test_fit_statistics_center_on_one_and_flag_an_erratic_rater() -> None:
    responses, assignments, _ = _wide_data("rating_scale", 1000, seed=13)
    consistent = fit_mfrm(_model("rating_scale"), responses, assignments)

    for statistics in (
        consistent.infit["rater"],
        consistent.outfit["rater"],
        consistent.item_infit,
        consistent.item_outfit,
    ):
        assert np.all(np.abs(statistics - 1.0) < 0.15), statistics

    rng = np.random.default_rng(14)
    erratic = assignments["rater"] == 3
    noisy = responses.copy()
    noisy[erratic] = rng.integers(0, 5, int(erratic.sum()))
    flagged = fit_mfrm(_model("rating_scale"), noisy, assignments)

    assert flagged.infit["rater"][3] > 1.3
    assert flagged.outfit["rater"][3] > 1.3
    assert np.all(flagged.outfit["rater"][:3] < 1.15)


def test_posterior_fit_statistics_match_a_direct_reference() -> None:
    responses, assignments, _ = _wide_data("partial_credit", 120, seed=4)
    result = fit_mfrm(_model("partial_credit"), responses, assignments)

    model = result.model
    # The person grid: equally spaced over +-6 sigma with normal weights.
    nodes = np.linspace(-6.0, 6.0, result.n_quadpts)
    weights = np.exp(-0.5 * nodes**2) / np.exp(-0.5 * nodes**2).sum()
    nodes = result.sigma * nodes
    log_posterior = np.log(weights) + np.zeros((len(responses), 1))
    probabilities = []
    for column, node in enumerate(nodes):
        values = model.probability(np.full(len(responses), node), None, assignments)
        probabilities.append(values)
        chosen = np.take_along_axis(values, responses[..., None], axis=2)[..., 0]
        log_posterior[:, column] += np.log(chosen).sum(axis=1)
    posterior = np.exp(log_posterior - log_posterior.max(axis=1, keepdims=True))
    posterior /= posterior.sum(axis=1, keepdims=True)
    categories = np.arange(5)
    squared = np.zeros(responses.shape)
    standardized = np.zeros(responses.shape)
    variance = np.zeros(responses.shape)
    for column, values in enumerate(probabilities):
        mean = values @ categories
        node_variance = values @ categories**2 - mean**2
        weight = posterior[:, column, None]
        squared += weight * (responses - mean) ** 2
        standardized += weight * (responses - mean) ** 2 / node_variance
        variance += weight * node_variance
    raters = assignments["rater"].ravel()
    infit = np.bincount(raters, squared.ravel()) / np.bincount(raters, variance.ravel())
    outfit = np.bincount(raters, standardized.ravel()) / np.bincount(raters)

    assert_allclose(result.infit["rater"], infit, rtol=1e-8)
    assert_allclose(result.outfit["rater"], outfit, rtol=1e-8)
    assert_allclose(result.theta, posterior @ nodes, atol=1e-10)


def test_estimates_are_stored_on_the_fitted_model() -> None:
    responses, assignments, _ = _wide_data("partial_credit", 300, seed=6)
    model = _model("partial_credit")

    result = fit_mfrm(model, responses, assignments)

    assert result.model is model
    assert model._is_fitted and model.copy()._is_fitted
    assert_array_equal(model.item_difficulty, result.item_difficulty)
    assert_array_equal(model.thresholds, result.thresholds)
    for name, values in result.facet_parameters.items():
        assert_allclose(model.facet_parameters[name], values)
    assert_allclose(result.facet_parameters["rater"].mean(), 0.0, atol=1e-10)
    assert_allclose(result.facet_parameters["task"].mean(), 0.5, atol=1e-10)
    assert result.aic == pytest.approx(
        -2 * result.log_likelihood + 2 * result.n_parameters
    )
    assert result.bic == pytest.approx(
        -2 * result.log_likelihood + result.n_parameters * np.log(300)
    )
    summary = result.summary()
    for label in ("Item_5", "rater_3", "task_1", "Infit", "Outfit"):
        assert label in summary


def test_anchoring_constraint_standard_errors_cover_every_level() -> None:
    responses, assignments, _ = _wide_data("binary", 400, seed=9)

    result = fit_mfrm(_model("binary"), responses, assignments)

    # With two levels anchored at their mean, both share one standard error.
    task_se = result.facet_se["task"]
    assert task_se.shape == (2,)
    assert_allclose(task_se[0], task_se[1], rtol=1e-10)
    assert np.all(np.isfinite(result.facet_se["rater"]))


def test_iteration_limit_is_reported_as_not_converged() -> None:
    responses, assignments, _ = _wide_data("rating_scale", 200, seed=10)

    result = fit_mfrm(_model("rating_scale"), responses, assignments, max_iter=1)

    assert not result.converged
    assert result.n_iterations == 1
    assert np.all(np.isfinite(result.facet_parameters["rater"]))


def _informative_data() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Twelve five-category ratings per person with sigma = 1.5."""
    rng = np.random.default_rng(7)
    truth = PolytomousMFRM(12, 5, [Facet("rater", 3)])
    truth.set_item_difficulty(np.linspace(-1.0, 1.0, 12))
    truth.set_facet_parameters("rater", np.array([-0.3, 0.0, 0.3]))
    truth.set_thresholds(RATING_SCALE)
    assignments = {"rater": rng.integers(0, 3, (300, 12))}
    responses = truth.simulate(rng.normal(0.0, 1.5, 300), assignments, seed=8)
    return responses, assignments


def _fit_informative(**options: object) -> MFRMResult:
    responses, assignments = _informative_data()
    model = PolytomousMFRM(12, 5, [Facet("rater", 3)])
    return fit_mfrm(model, responses, assignments, **options)


def test_default_grid_resolves_narrow_person_posteriors() -> None:
    # 21 Gauss-Hermite nodes scaled by sigma put sigma at 1.45 instead of 1.61
    # here and shifted every item difficulty by about 0.06.
    default = _fit_informative()
    fine = _fit_informative(n_quadpts=401)

    assert 41 < default.n_quadpts < 401
    assert default.sigma == pytest.approx(fine.sigma, abs=1e-5)
    assert_allclose(default.item_difficulty, fine.item_difficulty, atol=1e-5)
    assert_allclose(default.item_se, fine.item_se, rtol=1e-4)
    assert default.log_likelihood == pytest.approx(fine.log_likelihood, abs=1e-5)


def test_explicit_coarse_grid_is_kept_with_a_warning() -> None:
    with pytest.warns(RuntimeWarning, match="coarser than the person posteriors"):
        result = _fit_informative(n_quadpts=21)

    assert result.n_quadpts == 21


def test_grid_refinement_shares_the_iteration_budget() -> None:
    with pytest.warns(RuntimeWarning, match="coarser"):
        first_grid = _fit_informative(n_quadpts=41)

    result = _fit_informative(max_iter=first_grid.n_iterations + 2)

    assert result.n_quadpts > 41
    assert result.n_iterations == first_grid.n_iterations + 2
    assert not result.converged


def test_sigma_without_person_variance_stays_on_its_bound() -> None:
    rng = np.random.default_rng(0)
    responses = rng.integers(0, 2, (300, 6))
    raters = rng.integers(0, 3, (300, 6))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = fit_mfrm(
            ManyFacetRaschModel(6, [Facet("rater", 3)]), responses, {"rater": raters}
        )

    assert result.sigma == pytest.approx(1e-4)
    assert np.isnan(result.sigma_se)
    assert np.all(np.isfinite(result.item_se))


def test_non_positive_definite_information_gives_nan_standard_errors() -> None:
    model = ManyFacetRaschModel(2, [Facet("rater", 3)])
    parameter_map = _parameter_map(model, 2, True)

    covariance = _natural_covariance(
        -np.eye(parameter_map.matrix.shape[1]), parameter_map
    )

    assert covariance.shape == (len(parameter_map.offset),) * 2
    assert np.all(np.isnan(covariance))


def test_minimal_result_summary_keeps_backward_compatible_defaults() -> None:
    model = ManyFacetRaschModel(2, [Facet("rater", 2)])
    result = MFRMResult(
        model=model,
        facet_parameters={"rater": np.zeros(2)},
        facet_se={"rater": np.ones(2)},
        infit={"rater": np.ones(2)},
        outfit={"rater": np.ones(2)},
        log_likelihood=-10.0,
        n_iterations=1,
        converged=True,
    )

    assert result.item_difficulty is None and result.theta is None
    summary = result.summary()
    assert "rater_1" in summary and "Item_0" not in summary


def test_top_level_export_resolves_to_estimator() -> None:
    assert mirt.fit_mfrm is fit_mfrm
    assert "fit_mfrm" in mirt.__all__


def test_unanchored_facet_is_rejected() -> None:
    model = ManyFacetRaschModel(4, [Facet("rater", 2, is_anchored=False)])
    responses = np.random.default_rng(0).integers(0, 2, (50, 4))

    with pytest.raises(MirtValidationError, match="not anchored"):
        fit_mfrm(model, responses, {"rater": np.zeros((50, 4), dtype=int)})


def test_facet_confounded_with_items_is_rejected() -> None:
    model = ManyFacetRaschModel(4, [Facet("rater", 2)])
    responses = np.random.default_rng(0).integers(0, 2, (50, 4))
    confounded = np.tile([0, 0, 1, 1], (50, 1))

    with pytest.raises(MirtValidationError, match="not identified"):
        fit_mfrm(model, responses, {"rater": confounded})


def _binary_inputs() -> tuple[ManyFacetRaschModel, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return (
        ManyFacetRaschModel(4, [Facet("rater", 2)]),
        rng.integers(0, 2, (60, 4)),
        rng.integers(0, 2, (60, 4)),
    )


@pytest.mark.parametrize(
    ("change", "error", "match"),
    [
        (lambda y, r: (y, np.zeros_like(r)), MirtDataError, "'rater_1' has no"),
        (
            lambda y, r: (np.where(np.arange(4) == 0, 1, y), r),
            MirtDataError,
            "'Item_0' has only extreme",
        ),
        (lambda y, r: (2 * y, r), MirtDataError, "coded 0 to 1"),
        (lambda y, r: (-np.ones_like(y), r), MirtDataError, "no observed"),
        (lambda y, r: (y, r.astype(float)), MirtValidationError, "integers"),
        (lambda y, r: (y, r[:, :3]), MirtValidationError, "shape"),
        (lambda y, r: (y, r + 1), MirtValidationError, r"\[0, 2\)"),
    ],
)
def test_invalid_ratings_and_assignments_are_rejected(
    change: Callable, error: type[Exception], match: str
) -> None:
    model, responses, raters = _binary_inputs()
    responses, raters = change(responses, raters)

    with pytest.raises(error, match=match):
        fit_mfrm(model, responses, {"rater": raters})


def test_facet_assignment_keys_are_validated() -> None:
    model, responses, raters = _binary_inputs()

    with pytest.raises(MirtValidationError, match="Missing facet"):
        fit_mfrm(model, responses)
    with pytest.raises(MirtValidationError, match="Unknown facet"):
        fit_mfrm(model, responses, {"rater": raters, "task": 0})


def test_empty_categories_are_rejected() -> None:
    rng = np.random.default_rng(0)
    raters = rng.integers(0, 2, (60, 4))
    rating_scale = PolytomousMFRM(4, 4, [Facet("rater", 2)])
    with pytest.raises(MirtDataError, match="category 3 is never used"):
        fit_mfrm(rating_scale, rng.integers(0, 3, (60, 4)), {"rater": raters})

    responses = rng.integers(0, 3, (60, 4))
    responses[:, 2] = np.where(responses[:, 2] == 1, 2, responses[:, 2])
    partial_credit = PolytomousMFRM(
        4, 3, [Facet("rater", 2)], category_structure="partial_credit"
    )
    with pytest.raises(MirtDataError, match="category 1 is never used for item"):
        fit_mfrm(partial_credit, responses, {"rater": raters})


def test_sigma_requires_persons_with_two_ratings() -> None:
    rng = np.random.default_rng(0)
    responses = -np.ones((80, 4), dtype=int)
    responses[np.arange(80), np.arange(80) % 4] = rng.integers(0, 2, 80)
    raters = (np.arange(80) // 4) % 2
    model = ManyFacetRaschModel(4, [Facet("rater", 2)])

    with pytest.raises(MirtDataError, match="estimate_sd=False"):
        fit_mfrm(model, responses, {"rater": raters})
    result = fit_mfrm(model, responses, {"rater": raters}, estimate_sd=False)
    assert result.sigma == 1.0 and result.sigma_se == 0.0


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("n_quadpts", 3),
        ("n_quadpts", 21.0),
        ("max_iter", 0),
        ("tol", 0.0),
        ("tol", float("nan")),
        ("estimate_sd", "yes"),
    ],
)
def test_estimation_arguments_are_validated(keyword: str, value: object) -> None:
    model, responses, raters = _binary_inputs()

    with pytest.raises(MirtValidationError, match=keyword):
        fit_mfrm(model, responses, {"rater": raters}, **{keyword: value})


def test_model_type_is_validated() -> None:
    with pytest.raises(MirtValidationError, match="ManyFacetRaschModel"):
        fit_mfrm(mirt.TwoParameterLogistic(3), np.zeros((5, 3), dtype=int))
