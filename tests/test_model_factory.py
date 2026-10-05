"""Contracts for the shared built-in model factory and its public callers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mirt import fit_mirt
from mirt.exceptions import MirtDataError, MirtModelError, MirtValidationError
from mirt.models._factory import (
    ITEM_MODEL_FAMILIES,
    build_item_model,
    item_model_class,
    validate_n_factors,
)
from mirt.multigroup import fit_multigroup


@pytest.fixture(scope="module")
def dichotomous() -> np.ndarray:
    rng = np.random.default_rng(11)
    return (rng.random((120, 5)) < 0.55).astype(np.int_)


@pytest.fixture(scope="module")
def polytomous() -> np.ndarray:
    rng = np.random.default_rng(12)
    return rng.integers(0, 3, size=(120, 5))


def _data(model: str, dichotomous: np.ndarray, polytomous: np.ndarray) -> np.ndarray:
    return polytomous if model in {"GRM", "GPCM", "PCM", "NRM"} else dichotomous


@pytest.mark.parametrize("model", ["1PL", "3PL", "4PL", "PCM"])
def test_fit_mirt_rejects_factors_for_unidimensional_families(
    model: str, dichotomous: np.ndarray, polytomous: np.ndarray
) -> None:
    # These families used to fit one factor silently when n_factors=2.
    with pytest.raises(MirtModelError) as error:
        fit_mirt(
            _data(model, dichotomous, polytomous),
            model=model,
            n_factors=2,
            max_iter=2,
            use_rust=False,
        )

    assert error.value.context["model_type"] == model
    assert error.value.context["n_factors"] == 2


@pytest.mark.parametrize("n_factors", [0, -1, 2.5, True, np.bool_(True), "2", None])
@pytest.mark.parametrize("model", ["1PL", "2PL", "GRM", "PCM"])
def test_fit_mirt_rejects_invalid_factor_counts(
    model: str,
    n_factors: Any,
    dichotomous: np.ndarray,
    polytomous: np.ndarray,
) -> None:
    with pytest.raises(MirtValidationError) as error:
        fit_mirt(
            _data(model, dichotomous, polytomous),
            model=model,
            n_factors=n_factors,
            max_iter=2,
        )

    assert error.value.context["parameter"] == "n_factors"


@pytest.mark.parametrize("model", ["2PL", "GRM", "GPCM", "NRM"])
def test_fit_mirt_builds_multidimensional_supported_families(
    model: str, dichotomous: np.ndarray, polytomous: np.ndarray
) -> None:
    result = fit_mirt(
        _data(model, dichotomous, polytomous),
        model=model,
        n_factors=np.int64(2),
        n_quadpts=5,
        max_iter=2,
        compute_standard_errors=False,
    )

    assert result.model.n_factors == 2
    assert type(result.model.n_factors) is int


def test_native_2pl_path_receives_a_normalized_factor_count(
    dichotomous: np.ndarray,
) -> None:
    result = fit_mirt(dichotomous, model="2PL", n_factors=np.int64(1), max_iter=5)

    assert result.model.n_factors == 1
    assert type(result.model.n_factors) is int


def test_fit_mirt_unknown_model_is_a_model_error(dichotomous: np.ndarray) -> None:
    with pytest.raises(MirtModelError, match="Unknown model"):
        fit_mirt(dichotomous, model="6PL")  # type: ignore[arg-type]


def test_factory_covers_every_fit_mirt_family() -> None:
    for name in ITEM_MODEL_FAMILIES:
        model = build_item_model(name, 3, n_categories=3)
        assert isinstance(model, item_model_class(name))
        assert model.model_name == name


def test_factory_validates_codes_against_inferred_categories() -> None:
    responses = np.array([[0, 2, -1], [1, 0, 3], [1, 1, 0]])

    model = build_item_model("GPCM", 3, responses=responses)
    assert model.n_categories == [2, 3, 4]  # type: ignore[attr-defined]

    with pytest.raises(MirtDataError, match="below n_categories"):
        build_item_model("GPCM", 3, n_categories=3, responses=responses)
    with pytest.raises(MirtDataError, match="coded as 0 or 1"):
        build_item_model("2PL", 3, responses=responses)
    with pytest.raises(MirtValidationError, match="n_categories is required"):
        build_item_model("GRM", 3)


def test_factory_copies_item_names() -> None:
    names = ["a", "b"]
    model = build_item_model("2PL", 2, item_names=names)
    names.append("c")

    assert model.item_names == ["a", "b"]


@pytest.mark.parametrize("value", [1, 3, np.int32(2)])
def test_validate_n_factors_normalizes_integers(value: Any) -> None:
    assert validate_n_factors(value) == int(value)
    assert type(validate_n_factors(value)) is int


@pytest.fixture(scope="module")
def unused_category_data() -> tuple[np.ndarray, np.ndarray]:
    from mirt import simdata

    data = simdata(model="GRM", n_persons=400, n_items=5, n_categories=4, seed=2)
    data = np.asarray(data).copy()
    data[:, 0] = np.minimum(data[:, 0], 2)
    groups = np.repeat([0, 1], 200)
    return data, groups


def test_fit_multigroup_infers_categories_per_item(
    unused_category_data: tuple[np.ndarray, np.ndarray],
) -> None:
    # A single global category count used to give item 0 a phantom category
    # whose threshold was pushed to the optimizer bound.
    data, groups = unused_category_data
    result = fit_multigroup(data, groups, model="GRM", invariance="scalar", max_iter=15)
    reference = fit_mirt(data, model="GRM", max_iter=2)

    for group in range(2):
        group_model = result.model.get_group_model(group)
        assert group_model.n_categories == [3, 4, 4, 4, 4]  # type: ignore[attr-defined]
        assert group_model.n_categories == reference.model.n_categories  # type: ignore[attr-defined]
        thresholds = group_model.parameters["thresholds"][0, :2]
        assert np.all(np.isfinite(thresholds))
        assert np.all(np.abs(thresholds) < 5.0)


def test_fit_multigroup_accepts_item_names_and_dataframe_columns(
    dichotomous: np.ndarray,
) -> None:
    pd = pytest.importorskip("pandas")
    groups = np.repeat([0, 1], 60)
    names = [f"q{i}" for i in range(dichotomous.shape[1])]

    explicit = fit_multigroup(dichotomous, groups, item_names=names, max_iter=2)
    frame = fit_multigroup(pd.DataFrame(dichotomous, columns=names), groups, max_iter=2)

    assert explicit.model.item_names == names
    assert frame.model.item_names == names
    assert frame.model.get_group_model(1).item_names == names


def test_fit_multigroup_rejects_codes_outside_dichotomous_range(
    dichotomous: np.ndarray,
) -> None:
    data = dichotomous.copy()
    data[0, 0] = 2

    with pytest.raises(MirtDataError, match="coded as 0 or 1"):
        fit_multigroup(data, np.repeat([0, 1], 60), max_iter=2)
