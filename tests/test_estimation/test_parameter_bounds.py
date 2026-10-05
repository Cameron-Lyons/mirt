"""The shared item-parameter boxes used by EM item optimizers."""

from types import SimpleNamespace

import numpy as np
import pytest

import mirt
from mirt.backends.rust.polytomous_mstep import try_polytomous_m_step
from mirt.estimation.base import _parameter_bounds
from mirt.estimation.em import EMEstimator
from mirt.models.dichotomous import FourParameterLogistic
from mirt.models.polytomous import GradedResponseModel, NominalResponseModel


@pytest.mark.parametrize(
    ("model_name", "name", "expected"),
    [
        ("4PL", "discrimination", (0.1, 5.0)),
        ("4PL", "difficulty", (-6.0, 6.0)),
        ("4PL", "guessing", (0.0, 0.5)),
        ("4PL", "upper", (0.5, 1.0)),
        ("5PL", "asymmetry", (0.1, 5.0)),
        ("GRM", "thresholds", (-6.0, 6.0)),
        ("GPCM", "steps", (-6.0, 6.0)),
        ("NRM", "slopes", (-5.0, 5.0)),
        ("NRM", "intercepts", (-6.0, 6.0)),
        ("Bifactor", "general_slopes", (0.1, 5.0)),
        ("MIRT", "specific_discrimination", (0.1, 5.0)),
        ("Custom", "location", (-6.0, 6.0)),
    ],
)
def test_parameter_bounds_by_name(model_name, name, expected):
    assert _parameter_bounds(SimpleNamespace(model_name=model_name), name) == expected


@pytest.mark.parametrize(
    "model",
    [FourParameterLogistic(2), NominalResponseModel(2, [3, 4], n_factors=2)],
)
def test_item_optimizer_boxes_follow_parameter_bounds(model):
    _, bounds = EMEstimator()._get_item_params_and_bounds(model, 1)
    masks = model.free_parameter_masks
    expected = []
    for name in model.parameters:
        count = int(np.count_nonzero(masks[name][1]))
        expected += [_parameter_bounds(model, name)] * count
    assert bounds == expected


@pytest.mark.skipif(not mirt.is_rust_available(), reason="native backend unavailable")
@pytest.mark.parametrize(
    ("name", "value", "accepted"),
    [
        ("discrimination", 5.0, True),
        ("discrimination", 5.5, False),
        ("thresholds", -6.5, False),
    ],
)
def test_native_polytomous_m_step_requires_starts_inside_the_boxes(
    name, value, accepted
):
    previous = mirt.get_backend()
    mirt.set_backend("rust")
    try:
        model = GradedResponseModel(2, n_categories=3)
        values = model.parameters[name]
        values.flat[0] = value
        model.set_parameters(**{name: values})
        responses = np.array([[0, 2], [1, 1], [2, 0], [1, 2]])
        points = np.linspace(-2.0, 2.0, 5)[:, None]
        posterior = np.full((4, 5), 0.2)
        assert (
            try_polytomous_m_step(
                model,
                responses,
                posterior,
                points,
                max_iter=20,
                ftol=1e-10,
                epsilon=1e-10,
                n_jobs=1,
            )
            is accepted
        )
    finally:
        mirt.set_backend(previous)
