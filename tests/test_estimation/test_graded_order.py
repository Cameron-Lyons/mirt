"""Ordering constraints of graded thresholds in the itemwise EM layout."""

import numpy as np
import pytest
from scipy.optimize import minimize

from mirt.estimation._graded_order import THRESHOLD_GAP, graded_order
from mirt.estimation.em import EMEstimator, _ordered_step
from mirt.exceptions import MirtValidationError
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.polytomous import GeneralizedPartialCredit, GradedResponseModel


def _order(free_thresholds):
    model = GradedResponseModel(1, n_categories=6)
    model.set_parameters(thresholds=np.array([[-1.0, -0.2, 0.0, 0.1, 0.4]]))
    model.set_free_parameter_masks({"thresholds": np.array([free_thresholds])})
    params, bounds = EMEstimator()._get_item_params_and_bounds(model, 0)
    order = graded_order(model, 0, params.size)
    assert order is not None
    lower, upper = np.asarray(bounds).T
    return order, lower, upper


def _is_ordered(order, point):
    constraint = order.constraint
    return bool(np.all(constraint.A @ point >= constraint.lb - 1e-12))


def _reference_projection(order, point, lower, upper):
    """Euclidean projection onto the ordered box by a generic QP solver."""
    constraint = order.constraint
    result = minimize(
        lambda x: (0.5 * np.sum((x - point) ** 2), x - point),
        x0=np.clip(point, lower, upper),
        jac=True,
        method="SLSQP",
        bounds=list(zip(lower, upper, strict=True)),
        constraints=[
            {
                "type": "ineq",
                "fun": lambda x: constraint.A @ x - constraint.lb,
                "jac": lambda x: constraint.A,
            }
        ],
        options={"ftol": 1e-15, "maxiter": 500},
    )
    return result.x


@pytest.mark.parametrize(
    "free",
    [
        [True] * 5,
        [True, True, False, True, True],
        [False, True, True, True, False],
        [True, False, True, False, True],
    ],
)
def test_projection_is_the_nearest_ordered_point_in_the_box(free):
    order, lower, upper = _order(free)
    rng = np.random.default_rng(sum(free))
    for _ in range(25):
        point = rng.normal(0.0, 2.0, lower.size)
        point[rng.random(lower.size) < 0.2] = 8.0
        projected = order.project(point, lower, upper)
        assert _is_ordered(order, projected)
        assert np.all((projected >= lower) & (projected <= upper))
        expected = _reference_projection(order, point, lower, upper)
        np.testing.assert_allclose(projected, expected, atol=1e-7)


def test_ordered_points_inside_the_box_are_kept():
    order, lower, upper = _order([True] * 5)
    point = np.array([1.0, -1.0, -0.5, 0.0, 0.1, 2.0])
    assert order.satisfied(point)
    np.testing.assert_array_equal(order.project(point, lower, upper), point)


def test_tied_thresholds_keep_the_minimum_gap_inside_the_box():
    order, lower, upper = _order([True] * 5)
    point = np.array([1.0, 0.3, 0.2, 0.2, 0.1, 6.5])
    projected = order.project(point, lower, upper)
    # The first four thresholds pool at their mean; the last one stops
    # exactly at the upper bound.
    np.testing.assert_allclose(np.diff(projected[1:5]), THRESHOLD_GAP, rtol=1e-6)
    assert projected[1] == pytest.approx(0.2 - 1.5 * THRESHOLD_GAP)
    assert projected[-1] == upper[-1]


def test_layout_skips_fixed_thresholds_and_rejects_fixed_disorder():
    model = GradedResponseModel(2, n_categories=4)
    model.set_parameters(thresholds=np.array([[-1.0, 0.25, 1.0], [0.5, -0.5, 1.0]]))
    model.set_free_parameter_masks(
        {"thresholds": np.array([[True, False, True], [False, False, True]])}
    )
    order = graded_order(model, 0, 3)
    assert order is not None
    np.testing.assert_array_equal(order.positions, [1, -1, 2])
    np.testing.assert_allclose(order.constraint.A, [[0, -1, 0], [0, 0, 1]])
    np.testing.assert_allclose(
        order.constraint.lb, [THRESHOLD_GAP - 0.25, THRESHOLD_GAP + 0.25]
    )
    with pytest.raises(MirtValidationError, match="must be ordered"):
        graded_order(model, 1, 2)
    assert graded_order(GradedResponseModel(1, n_categories=2), 0, 2) is None
    assert graded_order(GeneralizedPartialCredit(1, n_categories=4), 0, 4) is None
    assert graded_order(TwoParameterLogistic(1), 0, 2) is None


def test_ordered_step_never_raises_the_objective():
    order, lower, upper = _order([True] * 5)
    bounds = list(zip(lower, upper, strict=True))
    target = np.array([1.0, -1.0, 0.3, 0.2, 0.25, 1.0])
    start = np.array([1.0, -1.0, -0.5, 0.0, 0.5, 1.0])

    def objective(x):
        return float(np.sum((x - target) ** 2)), 2.0 * (x - target)

    # A proposal whose tied thresholds are slightly disordered, as SLSQP
    # leaves them, is projected.
    proposal = np.array([1.0, -1.0, 0.25, 0.25 - 1e-9, 0.25 + 1e-9, 1.0])
    step = _ordered_step(order, objective, bounds, start, proposal)
    assert _is_ordered(order, step)
    np.testing.assert_array_equal(step, order.project(proposal, lower, upper))
    # Ordered proposals are returned as they are.
    feasible = np.array([1.0, -1.0, 0.2, 0.25, 0.3, 1.0])
    np.testing.assert_array_equal(
        _ordered_step(order, objective, bounds, start, feasible), feasible
    )
    # A proposal whose projection is worse than the start, or one that is not
    # finite, keeps the start.
    worse = np.array([1.0, 3.0, 2.0, 1.0, 0.0, 1.0])
    np.testing.assert_array_equal(
        _ordered_step(order, objective, bounds, start, worse), start
    )
    np.testing.assert_array_equal(
        _ordered_step(order, objective, bounds, start, np.full(6, np.nan)), start
    )


def _fixed_layout(thresholds, free):
    model = GradedResponseModel(1, n_categories=len(thresholds) + 1)
    model.set_parameters(thresholds=np.array([thresholds]))
    model.set_free_parameter_masks({"thresholds": np.array([free])})
    params, bounds = EMEstimator()._get_item_params_and_bounds(model, 0)
    lower, upper = np.asarray(bounds).T
    return model, graded_order(model, 0, params.size), params, lower, upper


@pytest.mark.parametrize(
    ("thresholds", "free"),
    [
        # A threshold fixed beyond the box leaves the next one no room.
        ([-1.0, 6.5, 7.0], [True, False, True]),
        # Fixed neighbours closer than two gaps.
        ([-1.0, 0.0, -1.0 + 1e-7], [False, True, False]),
    ],
)
def test_fixed_thresholds_without_ordered_room_are_rejected(thresholds, free):
    model, order, params, lower, upper = _fixed_layout(thresholds, free)
    with pytest.raises(MirtValidationError, match="no room"):
        order.project(params, lower, upper)
    # The generic M-step used to return disordered thresholds silently.
    responses = np.random.default_rng(3).integers(0, len(thresholds) + 1, (200, 1))
    estimator = EMEstimator(use_rust=False, max_iter=5, compute_standard_errors=False)
    with pytest.raises(MirtValidationError, match="GRM item 0"):
        estimator.fit(model, responses, start="model")


def test_fixed_thresholds_exactly_two_gaps_apart_hold_the_free_one_between():
    _, order, params, lower, upper = _fixed_layout(
        [-1.0, 0.5, -1.0 + 2 * THRESHOLD_GAP], [False, True, False]
    )
    projected = order.project(params, lower, upper)
    assert projected[1] == pytest.approx(-1.0 + THRESHOLD_GAP, abs=1e-15)
