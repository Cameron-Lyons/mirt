"""Shared model-state contract tests."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mirt.exceptions import MirtValidationError
from mirt.models.dichotomous import (
    OneParameterLogistic,
    TwoParameterLogistic,
)


def test_parameter_updates_detach_caller_owned_arrays() -> None:
    model = TwoParameterLogistic(n_items=2)
    difficulty = np.array([-0.5, 0.5])

    model.set_parameters(difficulty=difficulty)
    difficulty[0] = 99.0

    assert_array_equal(model.parameters["difficulty"], [-0.5, 0.5])


def test_multi_parameter_update_is_atomic() -> None:
    model = TwoParameterLogistic(n_items=2)
    before = model.parameters

    with pytest.raises(MirtValidationError, match="Shape mismatch"):
        model.set_parameters(
            difficulty=np.array([-1.0, 1.0]),
            discrimination=np.ones(3),
        )

    for name, values in before.items():
        assert_array_equal(model.parameters[name], values)


@pytest.mark.parametrize("item_idx", [True, np.bool_(False), 0.0])
def test_item_parameter_access_requires_integer_indices(item_idx: object) -> None:
    model = TwoParameterLogistic(n_items=2)

    with pytest.raises(IndexError, match="must be an integer"):
        model.get_item_parameters(item_idx)  # type: ignore[arg-type]
    with pytest.raises(IndexError, match="must be an integer"):
        model.set_item_parameter(  # type: ignore[arg-type]
            item_idx,
            "difficulty",
            1.0,
        )


def test_item_update_preserves_subclass_parameter_rules() -> None:
    model = OneParameterLogistic(n_items=2)
    before = model.parameters["discrimination"]

    model.set_item_parameter(0, "discrimination", 1.0)
    with pytest.raises(ValueError, match="Cannot set discrimination"):
        model.set_item_parameter(0, "discrimination", 2.0)

    assert_array_equal(model.parameters["discrimination"], before)
