"""Contracts for the shared built-in model factory and its public callers."""

from __future__ import annotations

import warnings
from typing import Any, get_args

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


def _literal_values(alias: Any) -> set[str]:
    values = set()
    for argument in get_args(alias):
        values |= _literal_values(argument) if get_args(argument) else {argument}
    return values


def test_all_model_type_names_only_name_fittable_models() -> None:
    from mirt.typing import AdvancedModelType, AllModelType, CDMModelType

    assert _literal_values(AllModelType) == set(ITEM_MODEL_FAMILIES) | (
        _literal_values(CDMModelType)
    )
    advanced = _literal_values(AdvancedModelType)
    assert advanced.isdisjoint(_literal_values(AllModelType))
    for name in advanced:
        with pytest.raises(MirtModelError, match="Unknown model"):
            item_model_class(name)


def test_factory_validates_codes_against_inferred_categories() -> None:
    responses = np.array([[0, 2, -1], [1, 0, 3], [1, 1, 0]])

    with pytest.warns(UserWarning, match="Item_3 never uses categories 1, 2"):
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


def test_factory_rejects_family_sequences_with_a_clear_error() -> None:
    # Regression: an unhashable list was reported as "Unknown model: [...]".
    for value in (["2PL", "2PL"], ("GRM",), np.array(["2PL"])):
        with pytest.raises(MirtValidationError, match="per-item family sequences"):
            item_model_class(value)  # type: ignore[arg-type]
    with pytest.raises(MirtModelError, match="Unknown model"):
        item_model_class("6PL")
    # A nested sequence inside fit_mirt's per-item list is an unknown family.
    data = np.zeros((10, 2), dtype=int)
    with pytest.raises(MirtModelError, match=r"Unknown model: \['2PL'\]"):
        fit_mirt(data, model=[["2PL"], ["2PL"]])  # type: ignore[list-item]


def test_fit_multigroup_accepts_single_family_sequences(
    dichotomous: np.ndarray,
) -> None:
    # Regression: a per-item sequence reached the factory as "Unknown model".
    groups = np.repeat([0, 1], 60)
    named = fit_multigroup(dichotomous, groups, model="2PL", max_iter=3)
    listed = fit_multigroup(dichotomous, groups, model=["2PL"] * 5, max_iter=3)

    assert listed.log_likelihood == pytest.approx(named.log_likelihood)
    for group in range(2):
        expected = named.model.get_group_model(group).parameters
        for name, values in listed.model.get_group_model(group).parameters.items():
            np.testing.assert_allclose(values, expected[name])
    with pytest.raises(MirtValidationError, match="does not support mixed item"):
        fit_multigroup(dichotomous, groups, model=["2PL"] * 4 + ["3PL"])
    with pytest.raises(MirtValidationError, match="names 4 items"):
        fit_multigroup(dichotomous, groups, model=["2PL"] * 4)


def test_factory_rejects_items_with_one_observed_category_like_r() -> None:
    # Regression: an item answered 1 by everyone was fitted silently to
    # a = 4.6, b = -6 (the bound) with SE(a) = 133; R mirt stops instead.
    responses = np.array([[1, 0, 2], [1, 1, -1], [1, 0, 0], [-1, 1, 1]])
    names = ["Q1", "Q2", "Q3"]

    for family in ("2PL", "GRM"):
        codes = responses if family == "GRM" else np.minimum(responses, 1)
        with pytest.raises(
            MirtDataError,
            match=r"one observed response category and cannot be estimated: Q1 \(",
        ):
            build_item_model(family, 3, item_names=names, responses=codes)
    with pytest.raises(MirtDataError, match=r"estimated: Item_1, Item_3 \("):
        fit_mirt(np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]]), model="2PL")
    # An item without any observed response keeps its starting values.
    unanswered = np.array([[0, -1], [1, -1], [1, -1]])
    assert build_item_model("2PL", 2, responses=unanswered).n_items == 2


def test_fit_warns_about_unused_interior_categories_and_recoding_matches_r() -> None:
    # Regression: category 1 of the first item is never used. GRM thresholds
    # collapsed to [-0.979, -0.979, 0.954] with NaN standard errors, GPCM steps
    # ran to the bound, and neither warned. R mirt recodes the item to three
    # categories; recoding as the warning suggests reproduces R mirt 1.48.
    from mirt.utils import recode_responses

    rng = np.random.default_rng(11)
    theta = rng.normal(size=1500)
    cumulative = 1 / (1 + np.exp(-1.5 * (theta[:, None, None] - np.array([-1, 0, 1]))))
    responses = (rng.random((1500, 6))[:, :, None] < cumulative).sum(axis=2)
    responses[responses[:, 0] == 1, 0] = 2

    message = (
        r"Item_1 never uses category 1 below its highest observed category 3; "
        r".*recode_responses\(data, \{2: 1, 3: 2\}, items=\[0\]\)"
    )
    with pytest.warns(UserWarning, match=message) as caught:
        fit_mirt(responses, model="GRM", max_iter=1, compute_standard_errors=False)
    assert len(caught) == 1
    assert caught[0].filename == __file__

    recoded = recode_responses(responses, {2: 1, 3: 2}, items=[0])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = fit_mirt(recoded, model="GRM", compute_standard_errors=False)
    slope = result.model.discrimination[0]
    assert result.model.n_categories[0] == 3
    # R: coef(mirt(data, 1, "graded"))$I1 = a1 1.4647, d1 1.5943, d2 -1.4797.
    assert slope == pytest.approx(1.4647, abs=5e-3)
    np.testing.assert_allclose(
        -slope * result.model.thresholds[0, :2], [1.5943, -1.4797], atol=5e-3
    )
