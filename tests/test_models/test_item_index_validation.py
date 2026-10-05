"""One item-index contract for every model family."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mirt.exceptions import MirtIndexError, MirtValidationError
from mirt.models.base import BaseItemModel
from mirt.models.bifactor import BifactorModel
from mirt.models.cdm import DINA, BaseCDM
from mirt.models.compensatory import (
    DisjunctiveModel,
    NoncompensatoryModel,
    PartiallyCompensatoryModel,
)
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
from mirt.models.mixture import MixtureIRT
from mirt.models.multidimensional import MultidimensionalModel
from mirt.models.nested import TwoPLNestedLogit
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedRatingScaleModel,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
    RatingScaleModel,
)
from mirt.models.sequential import AdjacentCategoryModel, SequentialResponseModel
from mirt.models.testlet import TestletModel
from mirt.models.unfolding import GeneralizedGradedUnfolding, IdealPointModel
from mirt.models.zeroinflated import HurdleIRT, ZeroInflated2PL

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
