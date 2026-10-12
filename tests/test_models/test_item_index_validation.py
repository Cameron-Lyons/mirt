"""One item-index contract for every model family."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mirt.exceptions import MirtIndexError, MirtValidationError
from mirt.models.base import BaseItemModel
from mirt.models.bifactor import BifactorModel
from mirt.models.cdm import DINA, BaseCDM
from mirt.models.cdm_advanced import HigherOrderCDM
from mirt.models.compensatory import (
    DisjunctiveModel,
    NoncompensatoryModel,
    PartiallyCompensatoryModel,
)
from mirt.models.custom import CustomItemModel, get_standard_item_type
from mirt.models.dichotomous import (
    ComplementaryLogLog,
    FiveParameterLogistic,
    FourParameterLogistic,
    NegativeLogLog,
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
    UnipolarLogLogistic,
)
from mirt.models.dynamic import LongitudinalIRTModel
from mirt.models.explanatory import LLTM, ExplanatoryIRT, RaschLLTM
from mirt.models.irtree import IRTreeModel
from mirt.models.mixed_format import MixedItemModel
from mirt.models.mixture import MixtureIRT
from mirt.models.multidimensional import MultidimensionalModel
from mirt.models.nested import TwoPLNestedLogit
from mirt.models.nonparametric import (
    KernelSmoothingModel,
    MonotonicPolynomialModel,
    MonotonicSplineModel,
)
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedRatingScaleModel,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
    RatingScaleModel,
)
from mirt.models.response_time import ResponseTimeModel
from mirt.models.sequential import AdjacentCategoryModel, SequentialResponseModel
from mirt.models.testlet import (
    BifactorTestletModel,
    RandomTestletEffectsModel,
    TestletModel,
)
from mirt.models.unfolding import GeneralizedGradedUnfolding, IdealPointModel
from mirt.models.zeroinflated import HurdleIRT, ZeroInflated2PL

_FEATURES = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def _kernel_model() -> KernelSmoothingModel:
    rng = np.random.default_rng(0)
    theta = rng.normal(size=40)
    responses = (rng.random((40, 3)) < 0.5).astype(int)
    return KernelSmoothingModel(3).calibrate(responses, theta)


_MODELS: dict[str, Callable[[], BaseItemModel]] = {
    "2PL": lambda: TwoParameterLogistic(3),
    "2PL-2D": lambda: TwoParameterLogistic(3, n_factors=2),
    "1PL": lambda: OneParameterLogistic(3),
    "3PL": lambda: ThreeParameterLogistic(3),
    "4PL": lambda: FourParameterLogistic(3),
    "5PL": lambda: FiveParameterLogistic(3),
    "ULL": lambda: UnipolarLogLogistic(3),
    "CLL": lambda: ComplementaryLogLog(3),
    "NLL": lambda: NegativeLogLog(3),
    "GRM": lambda: GradedResponseModel(3, 4),
    "GRM-2D": lambda: GradedResponseModel(3, 4, n_factors=2),
    "GPCM": lambda: GeneralizedPartialCredit(3, 4),
    "PCM": lambda: PartialCreditModel(3, 4),
    "RSM": lambda: RatingScaleModel(3, 4),
    "GRSM": lambda: GradedRatingScaleModel(3, 4),
    "NRM": lambda: NominalResponseModel(3, 4),
    "nested": lambda: TwoPLNestedLogit(3, 4),
    "sequential": lambda: SequentialResponseModel(3, 4),
    "adjacent": lambda: AdjacentCategoryModel(3, 4),
    "ZI-2PL": lambda: ZeroInflated2PL(3),
    "hurdle": lambda: HurdleIRT(3),
    "MIRT": lambda: MultidimensionalModel(3, 2),
    "bifactor": lambda: BifactorModel(3, [0, 0, 1]),
    "partially-compensatory": lambda: PartiallyCompensatoryModel(3),
    "noncompensatory": lambda: NoncompensatoryModel(3),
    "disjunctive": lambda: DisjunctiveModel(3),
    "DINA": lambda: DINA(3, 2, np.array([[1, 0], [0, 1], [1, 1]])),
    "mixture": lambda: MixtureIRT(3),
    "GGUM": lambda: GeneralizedGradedUnfolding(3, 3),
    "ideal-point": lambda: IdealPointModel(3),
    "testlet": lambda: TestletModel(3, [0, 0, 1]),
    "LLTM": lambda: LLTM(3, _FEATURES),
    "RaschLLTM": lambda: RaschLLTM(3, _FEATURES),
    "explanatory": lambda: ExplanatoryIRT(3, _FEATURES, n_person_covariates=1),
    "spline": lambda: MonotonicSplineModel(3),
    "polynomial": lambda: MonotonicPolynomialModel(3),
    "kernel": _kernel_model,
    "custom": lambda: CustomItemModel(3, get_standard_item_type("STANDARD_2PL")),
}
_INVALID = [-1, 3, 1.0, True, np.bool_(True), np.float64(1.0), "1"]


def _points(model: BaseItemModel) -> np.ndarray:
    if isinstance(model, BaseCDM):
        return np.zeros((2, model.n_attributes), dtype=np.int_)
    return np.linspace(-1.0, 1.0, 2 * model.n_factors).reshape(2, model.n_factors)


@pytest.mark.parametrize("item_idx", _INVALID, ids=repr)
@pytest.mark.parametrize("name", _MODELS)
def test_probability_rejects_invalid_item_indices(name: str, item_idx: object) -> None:
    model = _MODELS[name]()

    with pytest.raises(MirtIndexError, match="item_idx") as caught:
        model.probability(_points(model), item_idx)  # type: ignore[arg-type]

    assert isinstance(caught.value, IndexError)
    assert isinstance(caught.value, ValueError)


@pytest.mark.parametrize("item_idx", [-1, 3, True, 0.0])
@pytest.mark.parametrize(
    "name", [name for name in _MODELS if name not in {"DINA", "mixture"}]
)
def test_information_rejects_invalid_item_indices(name: str, item_idx: object) -> None:
    model = _MODELS[name]()

    with pytest.raises(MirtIndexError, match="item_idx"):
        model.information(_points(model), item_idx)  # type: ignore[arg-type]


@pytest.mark.parametrize("item_idx", [-1, 3, True, 0.0])
@pytest.mark.parametrize("name", _MODELS)
def test_item_parameter_access_rejects_invalid_indices(
    name: str, item_idx: object
) -> None:
    with pytest.raises(MirtIndexError, match="item_idx"):
        _MODELS[name]().get_item_parameters(item_idx)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "item_idx", [np.int64(2), np.intp(2), np.uint8(2), np.array(2)], ids=repr
)
@pytest.mark.parametrize("name", _MODELS)
def test_integer_like_indices_match_python_integers(
    name: str, item_idx: object
) -> None:
    model = _MODELS[name]()
    points = _points(model)

    assert_array_equal(
        model.probability(points, item_idx),  # type: ignore[arg-type]
        model.probability(points, 2),
    )


def test_negative_index_no_longer_aliases_the_last_item() -> None:
    model = TwoParameterLogistic(3).set_parameters(difficulty=np.array([0.0, 1.0, 2.0]))

    with pytest.raises(IndexError, match=r"item_idx -1 out of range \[0, 3\)"):
        model.probability(np.zeros((2, 1)), -1)


@pytest.mark.parametrize("item_idx", [-1, True])
@pytest.mark.parametrize("name", ["GRM", "GPCM", "RSM", "GRSM", "NRM"])
def test_polytomous_single_item_methods_validate_indices(
    name: str, item_idx: object
) -> None:
    model = _MODELS[name]()
    theta = np.zeros((2, 1))

    for method in (
        lambda: model.expected_score(theta, item_idx),
        lambda: model.category_response_curves(theta, item_idx),
        lambda: model.category_probability(theta, item_idx, 0),
    ):
        with pytest.raises(MirtIndexError, match="item_idx"):
            method()
    if hasattr(model, "cumulative_probability"):
        with pytest.raises(MirtIndexError, match="item_idx"):
            model.cumulative_probability(theta, item_idx, 0)


def _mixed_model() -> MixedItemModel:
    return MixedItemModel(
        [(TwoParameterLogistic(2), [0, 2]), (GradedResponseModel(1, 4), [1])]
    )


@pytest.mark.parametrize(
    "category", [-1, 4, 7, True, np.bool_(False), 1.0, "1"], ids=repr
)
@pytest.mark.parametrize(
    "name",
    ["GRM", "GRM-2D", "GPCM", "PCM", "RSM", "GRSM", "NRM", "sequential", "adjacent"],
)
def test_category_probability_rejects_invalid_categories(
    name: str, category: object
) -> None:
    model = _MODELS[name]()

    with pytest.raises(MirtIndexError, match="category") as caught:
        model.category_probability(_points(model), 0, category)  # type: ignore[arg-type]

    assert isinstance(caught.value, IndexError)
    assert isinstance(caught.value, ValueError)


@pytest.mark.parametrize("category", [-1, 2, True, 1.0], ids=repr)
def test_mixed_format_category_probability_uses_each_items_categories(
    category: object,
) -> None:
    model = _mixed_model()

    with pytest.raises(MirtIndexError, match="category"):
        model.category_probability(np.zeros(2), 0, category)  # type: ignore[arg-type]
    assert_array_equal(
        model.category_probability(np.zeros(2), 1, np.int64(3)),
        model.probability(np.zeros(2), 1)[:, 3],
    )


@pytest.mark.parametrize("threshold_idx", [-1, 1, 3, True, 0.0], ids=repr)
def test_graded_cumulative_probability_rejects_padded_thresholds(
    threshold_idx: object,
) -> None:
    """Item 0 has a single threshold; padding and wraparound gave sigmoid(a*theta)."""
    model = GradedResponseModel(2, n_categories=[2, 5])

    with pytest.raises(MirtIndexError, match="threshold_idx"):
        model.cumulative_probability(np.zeros(2), 0, threshold_idx)  # type: ignore[arg-type]
    assert model.cumulative_probability(np.zeros(2), 1, np.int64(3)).shape == (2,)


@pytest.mark.parametrize("threshold_idx", [-1, 3, True, 1.0], ids=repr)
def test_graded_rating_scale_cumulative_probability_rejects_invalid_thresholds(
    threshold_idx: object,
) -> None:
    model = GradedRatingScaleModel(2, 4)

    with pytest.raises(MirtIndexError, match="threshold_idx"):
        model.cumulative_probability(np.zeros(2), 0, threshold_idx)  # type: ignore[arg-type]
    assert model.cumulative_probability(np.zeros(2), 0, 2).shape == (2,)


@pytest.mark.parametrize("item_idx", [-1, True, np.bool_(False)], ids=repr)
@pytest.mark.parametrize(
    "name",
    ["MIRT", "bifactor", "GRM-2D", "GPCM", "NRM", "noncompensatory", "disjunctive"],
)
def test_item_information_matrices_reject_boolean_and_negative_indices(
    name: str, item_idx: object
) -> None:
    model = _MODELS[name]()

    with pytest.raises(MirtIndexError, match="item_idx"):
        model.item_information_matrix(_points(model), item_idx)  # type: ignore[attr-defined]


@pytest.mark.parametrize("name", ["ZI-2PL", "sequential"])
def test_formerly_value_error_families_stay_catchable_as_value_errors(
    name: str,
) -> None:
    model = _MODELS[name]()

    with pytest.raises(MirtValidationError, match="item_idx must be an integer"):
        model.set_item_parameter(1.5, "discrimination", 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="out of range"):
        model.probability(np.zeros(2), 3)


@pytest.mark.parametrize("item_idx", [-1, 3, True, 1.0])
def test_unfitted_kernel_model_validates_the_index_first(item_idx: object) -> None:
    model = KernelSmoothingModel(3)

    for method in (model.probability, model.information):
        with pytest.raises(MirtIndexError, match="item_idx"):
            method(np.zeros(2), item_idx)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="calibrated"):
        model.probability(np.zeros(2), 1)


@pytest.mark.parametrize("item_idx", [-1, 2, True, 1.0])
def test_item_families_outside_base_model_raise_mirt_index_error(
    item_idx: object,
) -> None:
    tree = IRTreeModel(n_items=2, tree_spec="direction_intensity")
    timing = ResponseTimeModel(n_items=2)
    growth = LongitudinalIRTModel(n_items=2, n_timepoints=3)

    for call in (
        lambda: tree.get_item_parameters(item_idx),  # type: ignore[arg-type]
        lambda: timing.response_time_information(item_idx),  # type: ignore[arg-type]
        lambda: growth.probability(np.zeros(2), item_idx),  # type: ignore[arg-type]
    ):
        with pytest.raises(MirtIndexError, match="item_idx"):
            call()


# Families whose model-level parameters can have one entry per item by
# coincidence: three items with three features, testlets, attributes or
# latent classes.
_GLOBAL_PARAMETER_MODELS: dict[str, Callable[[], BaseItemModel]] = {
    "LLTM": lambda: LLTM(3, np.eye(3)),
    "RaschLLTM": lambda: RaschLLTM(3, np.eye(3)),
    "explanatory": lambda: ExplanatoryIRT(3, np.eye(3), n_person_covariates=1),
    "mixture": lambda: MixtureIRT(3, n_classes=3),
    "testlet": lambda: TestletModel(3, [0, 1, 2]),
    "bifactor-testlet": lambda: BifactorTestletModel(3, [0, 1, 2]),
    "random-testlet": lambda: RandomTestletEffectsModel(3, [0, 1, 2]),
    "higher-order-CDM": lambda: HigherOrderCDM(3, 3, np.eye(3, dtype=int)),
}


@pytest.mark.parametrize("name", _GLOBAL_PARAMETER_MODELS)
def test_global_parameters_are_never_item_indexed(name: str) -> None:
    from mirt.results.fit_result import _indexed_by_item
    from mirt.utils.extraction import extract_item

    model = _GLOBAL_PARAMETER_MODELS[name]()
    extracted = extract_item(model, 0).parameters

    assert model._global_parameters
    for parameter in model._global_parameters:
        values = model.parameters[parameter]
        assert values.shape[0] == model.n_items
        assert not model._item_indexed(parameter)
        assert not _indexed_by_item(model, parameter, values.shape)
        assert_array_equal(model.get_item_parameters(0)[parameter], values)
        assert_array_equal(extracted[parameter], values)
        with pytest.raises(MirtValidationError, match="per-item"):
            model.set_item_parameter(1, parameter, 0.5)
        assert_array_equal(model.parameters[parameter], values)
