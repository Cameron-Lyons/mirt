"""Regression coverage for exact linear parameter constraints."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.estimation.constraints import LinearConstraint
from mirt.models.dichotomous import TwoParameterLogistic


@pytest.fixture
def model() -> TwoParameterLogistic:
    fitted = TwoParameterLogistic(n_items=5)
    fitted.set_parameters(
        discrimination=np.ones(5),
        difficulty=np.array([-1.0, -0.5, 0.0, 0.5, 1.0]),
    )
    return fitted


@pytest.mark.parametrize(
    ("constraint_type", "target", "reducer"),
    [
        ("mean", 1.0, np.mean),
        ("sum", 2.0, np.sum),
    ],
)
def test_nonzero_subset_constraint_is_satisfied_exactly(
    model: TwoParameterLogistic,
    constraint_type: str,
    target: float,
    reducer,
) -> None:
    original = model.difficulty.copy()
    constraint = LinearConstraint(
        "difficulty",
        item_indices=[0, 1, 2],
        target=target,
        constraint_type=constraint_type,
    )

    constraint.apply(model)

    assert reducer(model.difficulty[:3]) == pytest.approx(target, abs=1e-12)
    assert_allclose(model.difficulty[3:], original[3:])
    assert constraint.is_satisfied(model)
    assert constraint.penalty(model) == pytest.approx(0.0, abs=1e-24)


def test_weighted_constraint_uses_minimum_norm_projection(
    model: TwoParameterLogistic,
) -> None:
    original = model.difficulty.copy()
    coefficients = np.array([1.0, 2.0, -1.0])
    target = 1.0
    expected = original[:3] + (
        (target - coefficients @ original[:3])
        * coefficients
        / (coefficients @ coefficients)
    )
    constraint = LinearConstraint(
        "difficulty",
        item_indices=[0, 1, 2],
        target=target,
        coefficients=coefficients,
    )

    constraint.apply(model)

    assert_allclose(model.difficulty[:3], expected, atol=1e-12)
    assert coefficients @ model.difficulty[:3] == pytest.approx(target, abs=1e-12)
    assert_allclose(model.difficulty[3:], original[3:])


def test_multidimensional_constraint_changes_only_requested_factor() -> None:
    model = TwoParameterLogistic(n_items=4, n_factors=2)
    discrimination = np.array([[0.5, -1.0], [1.0, -0.5], [1.5, 0.5], [2.0, 1.0]])
    model.set_parameters(
        discrimination=discrimination,
        difficulty=np.zeros(4),
    )
    constraint = LinearConstraint(
        "discrimination",
        item_indices=[0, 2, 3],
        target=3.0,
        constraint_type="sum",
        factor=1,
    )

    constraint.apply(model)

    assert model.discrimination[[0, 2, 3], 1].sum() == pytest.approx(3.0)
    assert_allclose(model.discrimination[:, 0], discrimination[:, 0])
    assert model.discrimination[1, 1] == discrimination[1, 1]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"constraint_type": "median"}, "constraint_type"),
        ({"target": np.inf}, "target"),
        ({"item_indices": []}, "item_indices"),
        ({"item_indices": [0, 0]}, "duplicates"),
        ({"item_indices": [-1]}, "non-negative"),
        ({"factor": -1}, "factor"),
        ({"coefficients": np.array([0.0, 0.0])}, "nonzero"),
        ({"coefficients": np.array([1.0, np.nan])}, "finite"),
    ],
)
def test_invalid_constraint_specification_is_rejected(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        LinearConstraint("difficulty", **kwargs)


@pytest.mark.parametrize(
    ("constraint", "error", "match"),
    [
        (
            LinearConstraint("difficulty", item_indices=[0, 5]),
            IndexError,
            "out of range",
        ),
        (
            LinearConstraint("difficulty", coefficients=np.ones(2)),
            ValueError,
            "coefficients length",
        ),
        (
            LinearConstraint("difficulty", factor=1),
            ValueError,
            "factor is out of range",
        ),
    ],
)
def test_model_dependent_validation(
    model: TwoParameterLogistic,
    constraint: LinearConstraint,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        constraint.apply(model)


@pytest.mark.parametrize("tol", [-1.0, np.inf, np.nan])
def test_satisfaction_tolerance_must_be_valid(
    model: TwoParameterLogistic,
    tol: float,
) -> None:
    constraint = LinearConstraint("difficulty")
    with pytest.raises(ValueError, match="tol"):
        constraint.is_satisfied(model, tol=tol)
