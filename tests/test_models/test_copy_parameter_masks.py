"""Explicit fixed coordinates survive every item-model copy implementation."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mirt.models.base import BaseItemModel
from mirt.models.bifactor import BifactorModel
from mirt.models.cdm import DINA, DINO
from mirt.models.cdm_advanced import GDINA, HigherOrderCDM
from mirt.models.custom import CustomItemModel, get_standard_item_type
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.explanatory import LLTM, ExplanatoryIRT, RaschLLTM
from mirt.models.mixture import MixtureIRT
from mirt.models.multidimensional import MultidimensionalModel
from mirt.models.multilevel import (
    CrossedRandomEffectsModel,
    MultilevelIRTModel,
    ThreeLevelIRTModel,
)
from mirt.models.nested import (
    FourPLNestedLogit,
    ThreePLNestedLogit,
    TwoPLNestedLogit,
)
from mirt.models.nonparametric import (
    KernelSmoothingModel,
    MonotonicPolynomialModel,
    MonotonicSplineModel,
)
from mirt.models.polytomous import GradedResponseModel
from mirt.models.testlet import (
    BifactorTestletModel,
    RandomTestletEffectsModel,
    TestletModel,
)
from mirt.models.unfolding import (
    GeneralizedGradedUnfolding,
    HyperbolicCosineModel,
    IdealPointModel,
)

_Q = np.array([[1, 0], [0, 1], [1, 1]])
_FEATURES = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])


@pytest.mark.parametrize(
    "factory, parameter",
    [
        pytest.param(lambda: TwoParameterLogistic(3), "difficulty", id="base"),
        pytest.param(
            lambda: GradedResponseModel(3, [2, 4, 3]),
            "discrimination",
            id="polytomous_base",
        ),
        pytest.param(
            lambda: MultidimensionalModel(
                3,
                2,
                model_type="confirmatory",
                loading_pattern=np.array([[1, 0], [1, 1], [0, 1]]),
            ),
            "intercepts",
            id="multidimensional",
        ),
        pytest.param(
            lambda: BifactorModel(4, [0, 0, 1, 1]), "intercepts", id="bifactor"
        ),
        pytest.param(lambda: DINA(3, 2, _Q), "slip", id="DINA"),
        pytest.param(lambda: DINO(3, 2, _Q), "slip", id="DINO"),
        pytest.param(lambda: GDINA(3, 2, _Q), "delta", id="GDINA"),
        pytest.param(
            lambda: HigherOrderCDM(3, 2, _Q), "loadings", id="higher_order_CDM"
        ),
        pytest.param(
            lambda: CustomItemModel(3, get_standard_item_type("STANDARD_2PL")),
            "b",
            id="custom",
        ),
        pytest.param(lambda: LLTM(3, _FEATURES), "feature_weights", id="LLTM"),
        pytest.param(
            lambda: RaschLLTM(3, _FEATURES), "feature_weights", id="Rasch_LLTM"
        ),
        pytest.param(
            lambda: ExplanatoryIRT(3, _FEATURES, 2), "feature_weights", id="explanatory"
        ),
        pytest.param(lambda: MixtureIRT(3), "difficulty_class0", id="mixture"),
        pytest.param(
            lambda: TwoPLNestedLogit(3, [2, 4, 3], correct_response=[1, 2, 0]),
            "difficulty",
            id="nested_2PL",
        ),
        pytest.param(
            lambda: ThreePLNestedLogit(3, [2, 4, 3], correct_response=[1, 2, 0]),
            "difficulty",
            id="nested_3PL",
        ),
        pytest.param(
            lambda: FourPLNestedLogit(3, [2, 4, 3], correct_response=[1, 2, 0]),
            "difficulty",
            id="nested_4PL",
        ),
        pytest.param(
            lambda: MonotonicSplineModel(3, n_knots=3, degree=2),
            "lower",
            id="spline",
        ),
        pytest.param(
            lambda: MonotonicPolynomialModel(3, degree=3), "location", id="polynomial"
        ),
        pytest.param(lambda: TestletModel(4, [0, 0, 1, 1]), "difficulty", id="testlet"),
        pytest.param(
            lambda: BifactorTestletModel(4, [0, 0, 1, 1]),
            "difficulty",
            id="bifactor_testlet",
        ),
        pytest.param(
            lambda: RandomTestletEffectsModel(4, [0, 0, 1, 1]),
            "difficulty",
            id="random_testlet",
        ),
        pytest.param(
            lambda: GeneralizedGradedUnfolding(3, [3, 5, 4]),
            "location",
            id="graded_unfolding",
        ),
        pytest.param(lambda: IdealPointModel(3), "location", id="ideal_point"),
        pytest.param(
            lambda: HyperbolicCosineModel(3), "location", id="hyperbolic_cosine"
        ),
    ],
)
def test_model_copy_preserves_fixed_counts_and_independent_state(
    factory: Callable[[], BaseItemModel], parameter: str
) -> None:
    model = factory()
    original_count = model.n_parameters
    original_parameters = model.parameters
    mask = model.free_parameter_masks[parameter]
    assert mask.flat[0] and mask.flat[1]
    mask.flat[0] = False
    model.set_free_parameter_masks({parameter: mask})
    model._is_fitted = True

    copied = model.copy()

    assert type(copied) is type(model)
    assert copied.is_fitted
    assert copied.item_names == model.item_names
    assert model.n_parameters == copied.n_parameters == original_count - 1
    for name, values in original_parameters.items():
        assert_array_equal(model.parameters[name], values)
        assert_array_equal(copied.parameters[name], values)
    n_traits = 1 if isinstance(model, HigherOrderCDM) else model.n_factors
    theta = np.vstack((np.zeros(n_traits), np.ones(n_traits)))
    assert_array_equal(copied.probability(theta), model.probability(theta))

    # Caller-owned masks and returned masks must not undo either restriction.
    mask.flat[0] = True
    copied.free_parameter_masks[parameter].flat[0] = True
    assert model.n_parameters == copied.n_parameters == original_count - 1
    # Copying must also detach the stored Boolean buffers, not only the dict.
    assert not np.shares_memory(
        model._free_parameter_restrictions[parameter],
        copied._free_parameter_restrictions[parameter],
    )

    copied_mask = copied.free_parameter_masks[parameter]
    copied_mask.flat[1] = False
    copied.set_free_parameter_masks({parameter: copied_mask})
    assert copied.n_parameters == original_count - 2
    assert model.n_parameters == original_count - 1

    updated = copied.parameters[parameter]
    updated.flat[0] += 0.01
    copied.set_parameters(**{parameter: updated})
    assert_array_equal(model.parameters[parameter], original_parameters[parameter])
    assert copied.parameters[parameter].flat[0] == updated.flat[0]

    model.set_free_parameter_masks(None)
    assert model.n_parameters == original_count
    assert copied.n_parameters == original_count - 2
    copied.set_free_parameter_masks(None)
    assert copied.n_parameters == original_count


def test_nonparametric_kernel_copy_preserves_zero_parameter_contract_and_calibration():
    theta = np.array([-1.0, 0.0, 1.0])
    responses = np.array([[0, 0], [0, 1], [1, 1]])
    model = KernelSmoothingModel(2, theta_grid=theta).calibrate(responses, theta)
    model.set_free_parameter_masks({})
    copied = model.copy()

    # This empirical family stores curves, with no parametric coordinates to fix.
    assert model.n_parameters == copied.n_parameters == 0
    assert model.free_parameter_masks == copied.free_parameter_masks == {}
    assert_array_equal(copied.probability(theta), model.probability(theta))
    copied.calibrate(1 - responses, theta)
    assert_array_equal(model.calibration_counts, [3, 3])
    assert not np.array_equal(copied.probability(theta), model.probability(theta))


@pytest.mark.parametrize("wrapper", ["two_level", "three_level", "crossed"])
def test_multilevel_wrapper_copies_preserve_restricted_base_model(wrapper):
    base = TestletModel(4, [0, 0, 1, 1])
    mask = np.array([False, True, True, True])
    base.set_free_parameter_masks({"difficulty": mask})
    if wrapper == "two_level":
        model = MultilevelIRTModel(base, np.array([0, 0, 1, 1]))
    elif wrapper == "three_level":
        model = ThreeLevelIRTModel(base, np.array([0, 0, 1, 1]), np.array([0, 1]))
    else:
        model = CrossedRandomEffectsModel(base, 2)

    copied = model.copy()

    assert base.n_parameters == model.base_model.n_parameters == 13
    assert copied.base_model.n_parameters == 13
    copied.base_model.set_free_parameter_masks(None)
    assert copied.base_model.n_parameters == 14
    assert base.n_parameters == model.base_model.n_parameters == 13


def test_higher_order_copy_preserves_nested_item_restrictions():
    model = HigherOrderCDM(3, 2, _Q)
    mask = model._base_cdm.free_parameter_masks["delta"]
    mask[2, 0] = False
    model._base_cdm.set_free_parameter_masks({"delta": mask})
    copied = model.copy()

    assert model._base_cdm.n_parameters == copied._base_cdm.n_parameters == 7
    copied._base_cdm.set_free_parameter_masks(None)
    assert copied._base_cdm.n_parameters == 8
    assert model._base_cdm.n_parameters == 7


@pytest.mark.parametrize("shape", [(), (2,), (2, 4)])
def test_base_copy_preserves_scalar_and_global_storage_masks(shape):
    model = TwoParameterLogistic(3)
    # Generic storage may contain global arrays unrelated to the item count.
    # Exercise the inherited Base copy directly without introducing a subclass.
    model._parameters["global"] = np.zeros(shape)
    original_count = 6 + int(np.prod(shape))
    assert model.n_parameters == original_count
    model.set_free_parameter_masks({"global": np.zeros(shape, dtype=bool)})

    copied = model.copy()

    assert model.n_parameters == copied.n_parameters == 6
    assert copied.free_parameter_masks["global"].shape == shape
    assert_array_equal(copied.parameters["global"], np.zeros(shape))
    assert not np.shares_memory(
        model._free_parameter_restrictions["global"],
        copied._free_parameter_restrictions["global"],
    )
    copied.set_free_parameter_masks(None)
    assert copied.n_parameters == original_count
    assert model.n_parameters == 6
