"""EM estimation under equality constraints across items."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import OptimizeResult, minimize

import mirt
from mirt.estimation._acceleration import FreeItemParameters
from mirt.estimation._graded_order import graded_threshold_constraint
from mirt.estimation._item_information import item_standard_errors
from mirt.estimation._shared_step import (
    EqualityGroup,
    TiedCoordinates,
    TiedItemObjective,
    optimize_tied_items,
    resolve_equality_constraints,
    validate_equality_constraints,
)
from mirt.estimation.constraints import EqualityConstraint, FixedConstraint
from mirt.estimation.em import EMEstimator
from mirt.estimation.latent_density import GaussianDensity
from mirt.estimation.mixed_format_em import MixedFormatEMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.standard_errors import (
    _flatten_parameters,
    _marginal_log_likelihoods,
)
from mirt.exceptions import MirtModelError, MirtValidationError
from mirt.models.dichotomous import (
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.mixed_format import MixedItemModel
from mirt.models.polytomous import GradedResponseModel, RatingScaleModel

N_ITEMS = 6
ALL_SLOPES = [{"parameter": "discrimination", "items": list(range(N_ITEMS))}]


def _logistic_data(seed: int, n_persons: int, slope: float = 1.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal((n_persons, 1))
    difficulty = np.linspace(-1.2, 1.2, N_ITEMS)
    probability = 1.0 / (1.0 + np.exp(-slope * (theta - difficulty)))
    return (rng.random(probability.shape) < probability).astype(np.int32)


def _graded_data(seed: int, n_persons: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    model = GradedResponseModel(N_ITEMS, n_categories=4)
    gaps = rng.uniform(0.7, 1.3, (N_ITEMS, 3))
    model.set_parameters(
        discrimination=np.r_[np.full(3, 1.5), rng.uniform(0.8, 2.0, 3)],
        thresholds=np.cumsum(gaps, axis=1) - gaps.sum(axis=1)[:, None] / 2,
    )
    return model.simulate(rng.standard_normal((n_persons, 1)), seed=seed + 1)


@pytest.fixture(scope="module")
def binary() -> np.ndarray:
    return _logistic_data(seed=0, n_persons=800)


@pytest.fixture(scope="module")
def equal_slopes(binary):
    return mirt.fit_mirt(binary, "2PL", constraints=ALL_SLOPES, tol=1e-9)


@pytest.fixture(scope="module")
def graded() -> np.ndarray:
    return _graded_data(seed=2, n_persons=800)


def _marginal(model, responses, n_quadpts: int = 21) -> float:
    quadrature = GaussHermiteQuadrature(n_quadpts, 1)
    mass = quadrature.weights / quadrature.weights.sum()
    return float(_marginal_log_likelihoods(model, responses, quadrature, mass).sum())


def _equal_slope_log_likelihood(responses, vector) -> float:
    """Marginal log-likelihood of a 2PL with one common slope."""
    model = TwoParameterLogistic(N_ITEMS)
    model.set_parameters(
        discrimination=np.full(N_ITEMS, vector[0]), difficulty=vector[1:]
    )
    return _marginal(model, responses)


# Estimation


def test_equal_slopes_maximize_the_constrained_marginal_likelihood(
    binary, equal_slopes
) -> None:
    reference = minimize(
        lambda vector: -_equal_slope_log_likelihood(binary, vector),
        np.r_[1.0, np.zeros(N_ITEMS)],
        method="BFGS",
        options={"gtol": 1e-6},
    )
    parameters = equal_slopes.model.parameters
    estimate = np.r_[parameters["discrimination"][0], parameters["difficulty"]]

    assert equal_slopes.converged
    assert_allclose(estimate, reference.x, atol=2e-4)
    assert equal_slopes.log_likelihood == pytest.approx(-reference.fun, abs=1e-6)


def test_equal_slopes_match_a_rasch_fit_with_estimated_variance(
    binary, equal_slopes
) -> None:
    density = GaussianDensity(estimate_cov=True)
    rasch = EMEstimator(latent_density=density, tol=1e-9).fit(
        OneParameterLogistic(N_ITEMS), binary
    )
    slope = equal_slopes.model.parameters["discrimination"][0]

    # Both models have one slope (or variance) and a location per item.
    assert equal_slopes.n_parameters == rasch.n_parameters == N_ITEMS + 1
    assert np.sqrt(density.cov[0, 0]) == pytest.approx(slope, rel=0.02)
    assert_allclose(
        rasch.model.parameters["difficulty"] / slope,
        equal_slopes.model.parameters["difficulty"],
        atol=0.03,
    )
    assert equal_slopes.log_likelihood == pytest.approx(rasch.log_likelihood, abs=0.05)


def test_tied_coordinates_stay_bitwise_equal(binary, graded) -> None:
    result = mirt.fit_mirt(
        binary, "2PL", constraints=[("discrimination", [0, 2, 4])], max_iter=30
    )
    slopes = result.model.parameters["discrimination"]
    assert np.unique(slopes[[0, 2, 4]]).size == 1
    assert np.unique(slopes).size == 4
    assert np.unique(result.standard_errors["discrimination"][[0, 2, 4]]).size == 1

    result = mirt.fit_mirt(
        graded,
        "GRM",
        constraints=[("thresholds", [0, 1, 2], 1), ("discrimination", [3, 4])],
        max_iter=30,
    )
    thresholds = result.model.parameters["thresholds"]
    assert np.unique(thresholds[:3, 1]).size == 1
    assert np.unique(result.model.parameters["discrimination"][3:5]).size == 1
    assert np.all(np.diff(thresholds, axis=1) > 0)


def test_constrained_graded_slopes_reach_a_stationary_point(graded) -> None:
    result = mirt.fit_mirt(
        graded, "GRM", constraints=[("discrimination", [0, 1, 2])], tol=1e-9
    )
    model = result.model
    free = mirt.fit_mirt(graded, "GRM", tol=1e-9)

    def derivative(name, index, step=1e-5):
        values = model.parameters[name]
        shifted = []
        for sign in (1.0, -1.0):
            trial = values.copy()
            trial[index] += sign * step
            model._parameters[name] = trial
            shifted.append(_marginal(model, graded))
        model._parameters[name] = values
        return (shifted[0] - shifted[1]) / (2 * step)

    assert np.unique(model.parameters["discrimination"][:3]).size == 1
    # The group moves as one coordinate; its derivative is the sum of its
    # members', which vanishes at the constrained maximum.
    assert abs(derivative("discrimination", [0, 1, 2])) < 2e-3
    assert abs(derivative("discrimination", 4)) < 2e-3
    assert abs(derivative("thresholds", (1, 0))) < 2e-3
    assert abs(derivative("thresholds", (0, 2))) < 2e-3
    assert result.log_likelihood < free.log_likelihood
    assert free.n_parameters - result.n_parameters == 2


def test_constrained_fit_is_monotone_and_accepts_item_names_and_objects(
    binary,
) -> None:
    estimator = EMEstimator(
        constraints=[EqualityConstraint("discrimination", [0, 1, 2])], max_iter=40
    )
    by_object = estimator.fit(TwoParameterLogistic(N_ITEMS), binary)
    history = np.asarray(estimator.convergence_history)
    assert np.all(np.diff(history) >= -1e-8)

    by_name = mirt.fit_mirt(
        binary,
        "2PL",
        item_names=[f"Q{index}" for index in range(N_ITEMS)],
        constraints=[("discrimination", ["Q0", "Q1", "Q2"])],
        max_iter=40,
    )
    assert_allclose(
        by_name.model.parameters["discrimination"],
        by_object.model.parameters["discrimination"],
        rtol=1e-6,
    )


def test_starting_values_of_a_group_are_averaged(binary) -> None:
    start = np.array([0.5, 1.5, 1.0, 1.0, 1.0, 1.0])
    model = TwoParameterLogistic(N_ITEMS)
    estimator = EMEstimator(constraints=[("discrimination", [0, 1])], max_iter=1)
    estimator.fit(model, binary, start={"discrimination": start})
    assert estimator.convergence_history[0] == pytest.approx(
        _marginal(
            TwoParameterLogistic(N_ITEMS).set_parameters(
                discrimination=np.r_[1.0, 1.0, start[2:]]
            ),
            binary,
        ),
        abs=1e-8,
    )


@pytest.mark.parametrize("use_rust", [False, True])
@pytest.mark.parametrize("round_objectives", [False, True])
def test_disordered_starting_thresholds_do_not_stall_the_joint_step(
    monkeypatch, use_rust, round_objectives
) -> None:
    if round_objectives:
        original = EMEstimator._item_objective
        weights = 1.0 + np.random.default_rng(35).normal(size=2) * 1e-14

        def rounded(self, model, item, *args, **kwargs):
            start, bounds, objective, analytic = original(
                self, model, item, *args, **kwargs
            )
            if item >= len(weights) or objective is None or not analytic:
                return start, bounds, objective, analytic

            def perturbed(vector):
                value, gradient = objective(vector)
                return weights[item] * value, weights[item] * gradient

            return start, bounds, perturbed, analytic

        # Backend rounding must not make SLSQP accept a collapsed category.
        monkeypatch.setattr(EMEstimator, "_item_objective", rounded)

    rng = np.random.default_rng(3)
    thresholds = np.array(
        [[-2.5, -2.3, -2.0], [1.8, 2.0, 2.4], [-0.5, 0.0, 0.5], [-1.0, 0.0, 1.0]]
    )
    true = GradedResponseModel(4, n_categories=4)
    true.set_parameters(
        discrimination=np.array([1.5, 1.2, 1.4, 1.0]), thresholds=thresholds
    )
    responses = true.simulate(rng.standard_normal((600, 1)), seed=4)
    constraints = [("thresholds", [0, 1], 0)]
    # Rows out of order at the bounds, where SLSQP cannot start.
    start = thresholds.copy()
    start[:2] = [[6.0, -6.0, -6.0], [6.0, -6.0, -0.2]]

    model = GradedResponseModel(4, n_categories=4)
    EMEstimator(
        constraints=constraints,
        max_iter=1,
        compute_standard_errors=False,
        use_rust=use_rust,
    ).fit(model, responses, start={"thresholds": start})
    # The first M-step already orders the tied items.
    estimate = model.parameters["thresholds"]
    assert np.all(np.diff(estimate, axis=1) > 0)
    assert estimate[0, 0] == estimate[1, 0]

    result = EMEstimator(constraints=constraints, use_rust=use_rust).fit(
        GradedResponseModel(4, n_categories=4),
        responses,
        start={"thresholds": start},
    )
    reference = mirt.fit_mirt(
        responses, "GRM", constraints=constraints, use_rust=use_rust
    )
    assert result.converged and reference.converged
    assert result.log_likelihood == pytest.approx(reference.log_likelihood, abs=1e-3)


@pytest.mark.parametrize("first_outcome", ["success", "failure", "nonfinite"])
def test_tied_step_recovers_from_a_collapsed_optimizer_endpoint(
    monkeypatch, first_outcome
) -> None:
    import mirt.estimation._shared_step as shared_step

    model = GradedResponseModel(2, n_categories=4).set_parameters(
        thresholds=np.tile([-3.0, 0.0, 3.0], (2, 1))
    )
    tied = resolve_equality_constraints([("thresholds", [0, 1], 0)], model)
    assert tied is not None
    target = np.array([1.0, -1.0, 0.0, 1.0])

    def objective(vector):
        delta = vector - target
        if first_outcome == "nonfinite" and np.all(np.abs(vector[1:]) < 1e-5):
            return np.nan, delta
        return float(0.5 * np.sum(delta**2)), delta

    estimator = EMEstimator()
    parts = []
    for item in range(model.n_items):
        start, bounds = estimator._get_item_params_and_bounds(model, item)
        parts.append(
            TiedItemObjective(
                item,
                start,
                bounds,
                objective,
                graded_threshold_constraint(model, item, start.size),
            )
        )
    original = shared_step.minimize
    first = True

    def collapse_once(*args, **kwargs):
        nonlocal first
        if not first:
            return original(*args, **kwargs)
        first = False
        constraint = kwargs["constraints"][0]
        collapsed = kwargs["x0"].copy()
        movable = np.any(constraint.A != 0.0, axis=0)
        # Return an ordered but collapsed candidate, just as SLSQP can on a
        # clipped category's flat objective, including a false success flag.
        collapsed[movable] = np.linalg.lstsq(
            constraint.A[:, movable], constraint.lb, rcond=None
        )[0]
        return OptimizeResult(x=collapsed, success=first_outcome != "failure")

    monkeypatch.setattr(shared_step, "minimize", collapse_once)
    result = optimize_tied_items(model, tied, parts, max_iter=50, ftol=1e-10)

    assert result is not None
    for estimate in result:
        assert objective(estimate)[0] < 1e-9
        assert np.all(np.diff(estimate[1:]) > 0.0)
    assert result[0][1] == result[1][1]


def test_numerical_item_objectives_match_the_analytic_fit(binary) -> None:
    analytic = mirt.fit_mirt(
        binary, "2PL", constraints=[("discrimination", [0, 1, 2])], tol=1e-8
    )
    # Fixed guessing restricts the masks, which disables the prepared
    # objectives, so the joint step differences the item likelihoods.
    numerical = mirt.fit_mirt(
        binary,
        "3PL",
        start_values={"guessing": np.zeros(N_ITEMS)},
        fixed={"guessing": True},
        constraints=[("discrimination", [0, 1, 2])],
        tol=1e-8,
    )
    for name in ("discrimination", "difficulty"):
        assert_allclose(
            numerical.model.parameters[name],
            analytic.model.parameters[name],
            atol=2e-4,
        )


def test_constraints_bypass_native_full_iterations(monkeypatch, binary) -> None:
    import mirt.backends.rust.estimation as native_estimation
    import mirt.estimation.em as em

    def refuse(*args, **kwargs):
        raise AssertionError("native fast path used with constraints")

    monkeypatch.setattr(native_estimation, "_em_fit_2pl_prepared", refuse)
    monkeypatch.setattr(em, "em_iteration_3pl", refuse)

    mirt.fit_mirt(binary, "2PL", constraints=ALL_SLOPES, max_iter=3)
    mirt.fit_mirt(binary, "3PL", constraints=ALL_SLOPES, max_iter=3)


@pytest.mark.skipif(not mirt.is_rust_available(), reason="native backend unavailable")
def test_native_polytomous_step_leaves_tied_items_to_the_joint_step(
    monkeypatch, graded
) -> None:
    import mirt.backends.rust.polytomous_mstep as native_polytomous

    handled = []
    native = native_polytomous.try_polytomous_m_step

    def spy(*args, **kwargs):
        handled.append(native(*args, **kwargs))
        return handled[-1]

    monkeypatch.setattr(native_polytomous, "try_polytomous_m_step", spy)
    constraints = [("discrimination", [0, 1, 2]), ("thresholds", [3, 4], 1)]
    fitted = mirt.fit_mirt(graded, "GRM", constraints=constraints, tol=1e-9)
    assert handled and all(handled)

    generic = mirt.fit_mirt(
        graded, "GRM", constraints=constraints, tol=1e-9, use_rust=False
    )
    parameters = fitted.model.parameters
    assert np.unique(parameters["discrimination"][:3]).size == 1
    assert parameters["thresholds"][3, 1] == parameters["thresholds"][4, 1]
    for name, values in generic.model.parameters.items():
        assert_allclose(parameters[name], values, atol=5e-4)
    assert fitted.log_likelihood == pytest.approx(generic.log_likelihood, abs=1e-5)


def test_batched_newton_step_updates_only_untied_items(monkeypatch, binary) -> None:
    constraints = [("discrimination", [1, 3]), ("difficulty", [4, 5])]
    skipped = []
    newton = EMEstimator._newton_logistic_m_step

    def spy(self, model, correct, observed, skip=()):
        skipped.append(tuple(skip))
        return newton(self, model, correct, observed, skip=skip)

    monkeypatch.setattr(EMEstimator, "_newton_logistic_m_step", spy)
    batched = mirt.fit_mirt(binary, "2PL", constraints=constraints, tol=1e-10)
    assert skipped and set(skipped) == {(1, 3, 4, 5)}

    # Itemwise optimization of the untied items reaches the same estimates.
    monkeypatch.setattr(EMEstimator, "_uses_newton_logistic_m_step", lambda *_: False)
    itemwise = mirt.fit_mirt(binary, "2PL", constraints=constraints, tol=1e-10)
    for name, values in itemwise.model.parameters.items():
        assert_allclose(batched.model.parameters[name], values, atol=2e-4)
    assert batched.log_likelihood == pytest.approx(itemwise.log_likelihood, abs=1e-6)
    parameters = batched.model.parameters
    assert parameters["discrimination"][1] == parameters["discrimination"][3]
    assert parameters["difficulty"][4] == parameters["difficulty"][5]


def test_squarem_extrapolates_each_group_as_one_coordinate(binary, equal_slopes):
    model = equal_slopes.model
    tied = resolve_equality_constraints(ALL_SLOPES, model)
    packed = FreeItemParameters(model, tied)
    vector = packed.get(model)

    assert vector.size == packed.lower.size == N_ITEMS + 1
    assert packed.set(model, vector + 0.01)
    assert np.unique(model.parameters["discrimination"]).size == 1
    packed.set(model, vector)

    accelerated = mirt.fit_mirt(
        binary, "2PL", constraints=ALL_SLOPES, accelerate="squarem", tol=1e-9
    )
    assert accelerated.log_likelihood == pytest.approx(
        equal_slopes.log_likelihood, abs=1e-5
    )
    assert np.unique(accelerated.model.parameters["discrimination"]).size == 1


# Standard errors


def test_observed_information_errors_match_constrained_differences(
    binary, equal_slopes
) -> None:
    parameters = equal_slopes.model.parameters
    estimate = np.r_[parameters["discrimination"][0], parameters["difficulty"]]
    size, step = estimate.size, 1e-4
    hessian = np.empty((size, size))
    basis = np.eye(size) * step
    for row in range(size):
        for column in range(size):
            hessian[row, column] = (
                _equal_slope_log_likelihood(
                    binary, estimate + basis[row] + basis[column]
                )
                - _equal_slope_log_likelihood(
                    binary, estimate + basis[row] - basis[column]
                )
                - _equal_slope_log_likelihood(
                    binary, estimate - basis[row] + basis[column]
                )
                + _equal_slope_log_likelihood(
                    binary, estimate - basis[row] - basis[column]
                )
            ) / (4 * step**2)
    covariance = np.linalg.inv(-hessian)
    errors = equal_slopes.standard_errors

    assert equal_slopes.se_method == "oakes"
    assert_allclose(errors["discrimination"], np.sqrt(covariance[0, 0]), rtol=1e-4)
    assert_allclose(errors["difficulty"], np.sqrt(np.diag(covariance)[1:]), rtol=1e-4)

    # The covariance repeats the group's row for every tied coordinate.
    vcov = equal_slopes.vcov
    assert vcov.shape == (2 * N_ITEMS, 2 * N_ITEMS)
    assert equal_slopes.vcov_labels[:2] == [
        "discrimination[Item_1]",
        "discrimination[Item_2]",
    ]
    assert_allclose(vcov[:N_ITEMS, :N_ITEMS], covariance[0, 0], rtol=1e-4)
    assert_allclose(
        vcov[:N_ITEMS, N_ITEMS:],
        np.tile(covariance[0, 1:], (N_ITEMS, 1)),
        rtol=1e-3,
        atol=1e-8,
    )
    assert_allclose(vcov[N_ITEMS:, N_ITEMS:], covariance[1:, 1:], rtol=1e-3, atol=1e-8)


@pytest.mark.parametrize("method", ["crossprod", "sandwich"])
def test_matrix_errors_reduce_the_tied_information(binary, method) -> None:
    result = mirt.fit_mirt(binary, "2PL", constraints=ALL_SLOPES, se_method=method)
    errors = result.standard_errors["discrimination"]
    assert result.se_method == method
    assert np.unique(errors).size == 1
    assert np.isfinite(errors[0]) and errors[0] > 0
    # A shared slope is far better determined than any single item's.
    free = mirt.fit_mirt(binary, "2PL", se_method=method)
    assert errors[0] < free.standard_errors["discrimination"].min()


def test_complete_data_errors_sum_the_member_curvatures(binary, equal_slopes) -> None:
    result = mirt.fit_mirt(
        binary, "2PL", constraints=ALL_SLOPES, se_method="complete_data", tol=1e-9
    )
    model = result.model
    quadrature = GaussHermiteQuadrature(21, 1)
    log_likelihood = model.log_likelihood_batch(binary, quadrature.nodes)
    log_joint = log_likelihood + np.log(quadrature.weights)
    posterior = np.exp(log_joint - log_joint.max(axis=1, keepdims=True))
    posterior /= posterior.sum(axis=1, keepdims=True)
    members = item_standard_errors(model, binary, posterior, quadrature.nodes, 1e-10)[
        "discrimination"
    ]

    expected = np.sum(members**-2.0) ** -0.5
    assert result.se_method == "complete_data"
    assert_allclose(result.standard_errors["discrimination"], expected, rtol=1e-6)
    assert_allclose(
        result.standard_errors["difficulty"],
        item_standard_errors(model, binary, posterior, quadrature.nodes, 1e-10)[
            "difficulty"
        ],
        rtol=1e-6,
    )


def test_tied_combination_of_complete_data_errors() -> None:
    tied = TiedCoordinates([("discrimination", np.array([0, 2]))], [[0, 2]])
    combined = tied.combine_standard_errors(
        {"discrimination": np.array([0.3, 0.5, 0.4]), "difficulty": np.ones(3)}
    )
    assert_allclose(combined["discrimination"], [0.24, 0.5, 0.24])
    assert_allclose(combined["difficulty"], 1.0)
    missing = tied.combine_standard_errors({"discrimination": np.full(3, np.nan)})
    assert np.isnan(missing["discrimination"][[0, 2]]).all()


def test_counts_and_criteria_use_one_parameter_per_group(binary, equal_slopes) -> None:
    free = mirt.fit_mirt(binary, "2PL")
    assert free.n_parameters == 2 * N_ITEMS
    assert equal_slopes.n_parameters == N_ITEMS + 1
    assert equal_slopes.aic == pytest.approx(
        -2 * equal_slopes.log_likelihood + 2 * (N_ITEMS + 1)
    )


def test_priors_on_tied_coordinates(binary) -> None:
    from mirt.estimation.priors import LogNormalPrior

    result = mirt.fit_mirt(
        binary,
        "2PL",
        constraints=ALL_SLOPES,
        priors={"discrimination": LogNormalPrior(0.0, 0.5)},
    )
    slopes = result.model.parameters["discrimination"]
    assert np.unique(slopes).size == 1
    assert result.log_posterior is not None
    assert np.unique(result.standard_errors["discrimination"]).size == 1


# Validation


def test_structure_validation_accepts_the_documented_forms() -> None:
    groups = validate_equality_constraints(
        [
            {"parameter": "discrimination", "items": [0, 1]},
            ("thresholds", np.arange(3), np.int64(1)),
            ["difficulty", ["Q1", "Q2"]],
            EqualityConstraint("difficulty"),
            # An empty item list means every item, as in EqualityConstraint.apply.
            EqualityConstraint("difficulty", []),
        ]
    )
    assert groups == (
        EqualityGroup("discrimination", (0, 1)),
        EqualityGroup("thresholds", (0, 1, 2), 1),
        EqualityGroup("difficulty", ("Q1", "Q2")),
        EqualityGroup("difficulty"),
        EqualityGroup("difficulty"),
    )
    assert validate_equality_constraints(None) == ()


@pytest.mark.parametrize(
    ("constraints", "message"),
    [
        ({"parameter": "discrimination"}, "sequence of constraint groups"),
        ("discrimination", "sequence of constraint groups"),
        ([("discrimination",)], "write each constraint"),
        ([{"items": [0, 1]}], "needs 'parameter'"),
        ([{"parameter": "a", "item": [0, 1]}], "unknown keys item"),
        ([(3, [0, 1])], "stored parameter name"),
        ([("discrimination", [0])], "at least two items"),
        ([("discrimination", [0, 0])], "more than once"),
        ([("discrimination", [0, -1])], "zero-based"),
        ([("discrimination", [0, 1.0])], "zero-based"),
        ([("discrimination", [0, True])], "zero-based"),
        ([("discrimination", [0, np.True_])], "zero-based"),
        ([("discrimination", 3)], "sequence of item positions"),
        ([("thresholds", [0, 1], -1)], "non-negative integer"),
        ([FixedConstraint("discrimination", [0], 1.0)], "not an equality"),
    ],
)
def test_malformed_constraints_are_rejected(constraints, message) -> None:
    with pytest.raises(MirtValidationError, match=message):
        validate_equality_constraints(constraints)
    with pytest.raises(MirtValidationError, match=message):
        EMEstimator(constraints=constraints)


@pytest.mark.parametrize(
    ("model", "constraints", "message"),
    [
        (TwoParameterLogistic(4), [("slopes", [0, 1])], "unknown parameter 'slopes'"),
        (OneParameterLogistic(4), [("discrimination", [0, 1])], "is fixed"),
        (RatingScaleModel(4, 3), [("thresholds", [0, 1])], "already shared"),
        (TwoParameterLogistic(4), [("difficulty", [0, 4])], "out of range"),
        (TwoParameterLogistic(4), [("difficulty", [0, "Q9"])], "'Q9'"),
        (TwoParameterLogistic(4), [("difficulty", [0, 1], 0)], "omit column"),
        (
            GradedResponseModel(4, n_categories=3),
            [("thresholds", [0, 1], 2)],
            "column 2 does not exist",
        ),
        (
            GradedResponseModel(3, n_categories=[3, 4, 4]),
            [("thresholds", [0, 1])],
            "give a column",
        ),
        (
            TwoParameterLogistic(4),
            [("difficulty", [0, 1]), ("difficulty", [1, 2])],
            r"also tied by constraints\[0\]",
        ),
        (TwoParameterLogistic(4), [("difficulty", ["Item_0", 0])], "more than once"),
    ],
)
def test_constraints_must_match_the_model(model, constraints, message) -> None:
    with pytest.raises(MirtValidationError, match=message):
        resolve_equality_constraints(constraints, model)


def test_constraints_reject_fixed_coordinates_and_other_estimators(binary) -> None:
    fixed = np.zeros(N_ITEMS, dtype=bool)
    fixed[1] = True
    with pytest.raises(MirtValidationError, match="Item_2 is fixed"):
        mirt.fit_mirt(
            binary,
            "2PL",
            fixed={"discrimination": fixed},
            constraints=[("discrimination", [0, 1])],
        )
    with pytest.raises(MirtValidationError, match="only to EM"):
        mirt.fit_mirt(binary, "2PL", estimation="MHRM", constraints=ALL_SLOPES)


def test_mixed_format_models_reject_constraints(binary) -> None:
    model = MixedItemModel(
        [(TwoParameterLogistic(3), [0, 1, 2]), (ThreeParameterLogistic(3), [3, 4, 5])]
    )
    estimator = MixedFormatEMEstimator(
        constraints=[("2PL.discrimination", [0, 1])], max_iter=2
    )
    with pytest.raises(MirtModelError, match="mixed-format"):
        estimator.fit(model, binary)
    with pytest.raises(MirtModelError, match="mixed-format"):
        mirt.fit_mirt(
            binary,
            ["2PL"] * 3 + ["3PL"] * 3,
            constraints=[("discrimination", [0, 1])],
        )


def test_tying_maps_packed_coordinates_to_groups() -> None:
    model = GradedResponseModel(3, n_categories=3)
    tied = resolve_equality_constraints(
        [("discrimination", [0, 2]), ("thresholds", [1, 2], 0)], model
    )
    assert tied is not None
    assert tied.items == (0, 1, 2)
    assert tied.n_redundant == 2
    _, layouts = _flatten_parameters(model)
    tying = tied.tying({name: layout.free_indices for name, layout in layouts.items()})
    # discrimination 0, 1, 2 then thresholds (0,0), (0,1), (1,0), (1,1), ...
    assert tying.tolist() == [0, 1, 0, 2, 3, 4, 5, 4, 6]


def test_linked_items_form_independent_components() -> None:
    model = GradedResponseModel(7, n_categories=3)
    tied = resolve_equality_constraints(
        [
            ("discrimination", [4, 0]),
            ("thresholds", [2, 3], 1),
            ("thresholds", [6, 4], 0),
            ("discrimination", [5, 2]),
        ],
        model,
    )
    assert tied is not None
    # Items 0, 4 and 6 are linked through item 4; 2, 3 and 5 through item 2.
    assert tied.components == ((0, 4, 6), (2, 3, 5))
    assert tied.items == (0, 2, 3, 4, 5, 6)
