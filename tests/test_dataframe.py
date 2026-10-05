from __future__ import annotations

from collections.abc import Generator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

import mirt
import mirt.utils.dataframe as dataframe_module
from mirt.exceptions import MirtValidationError
from mirt.utils import get_dataframe_backend, set_dataframe_backend
from mirt.utils.dataframe import create_dataframe


@pytest.fixture(autouse=True)
def restore_dataframe_backend() -> Generator[None, None, None]:
    set_dataframe_backend("auto")
    yield
    set_dataframe_backend("auto")


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_explicit_backend_selection(backend: str) -> None:
    set_dataframe_backend(backend)

    assert get_dataframe_backend() == backend


@pytest.mark.parametrize("automatic", ["auto", None])
def test_automatic_backend_prefers_polars(automatic: str | None) -> None:
    set_dataframe_backend("pandas")
    set_dataframe_backend(automatic)

    assert get_dataframe_backend() == "polars"


def test_invalid_backend_raises_package_validation_error() -> None:
    with pytest.raises(MirtValidationError, match="Invalid DataFrame backend"):
        set_dataframe_backend("arrow")


def test_explicit_unavailable_backend_has_install_hint(monkeypatch: Any) -> None:
    def unavailable(name: str) -> Any:
        if name == "polars":
            raise ImportError("missing")
        return pd

    monkeypatch.setattr(dataframe_module, "import_module", unavailable)

    with pytest.raises(ImportError, match=r"pip install mirt\[polars\]"):
        set_dataframe_backend("polars")


def test_automatic_selection_falls_back_to_pandas(monkeypatch: Any) -> None:
    def pandas_only(name: str) -> Any:
        if name == "polars":
            raise ImportError("missing")
        return pd

    monkeypatch.setattr(dataframe_module, "import_module", pandas_only)

    assert get_dataframe_backend() == "pandas"


def test_automatic_selection_requires_an_optional_backend(monkeypatch: Any) -> None:
    def unavailable(name: str) -> Any:
        raise ImportError(f"{name} missing")

    monkeypatch.setattr(dataframe_module, "import_module", unavailable)

    with pytest.raises(ImportError, match="DataFrame output requires"):
        get_dataframe_backend()


def test_polars_named_default_index_is_materialized() -> None:
    set_dataframe_backend("polars")

    result = create_dataframe({"score": [1, 2]}, index_name="person")

    assert result.columns == ["person", "score"]
    assert result["person"].to_list() == [0, 1]


def test_polars_existing_identifier_is_preserved_and_moved_first() -> None:
    set_dataframe_backend("polars")

    result = create_dataframe(
        {"score": [1, 2], "person": ["p1", "p2"]}, index_name="person"
    )

    assert result.columns == ["person", "score"]
    assert result["person"].to_list() == ["p1", "p2"]


def test_polars_numpy_index_is_inserted_without_changing_values() -> None:
    set_dataframe_backend("polars")
    index = np.array([10, 11], dtype=np.int64)

    result = create_dataframe({"score": [1, 2]}, index=index, index_name="person")

    assert result.columns == ["person", "score"]
    assert result["person"].to_numpy().dtype == index.dtype
    np.testing.assert_array_equal(result["person"].to_numpy(), index)


def test_polars_index_collision_is_rejected_without_overwriting_data() -> None:
    set_dataframe_backend("polars")

    with pytest.raises(MirtValidationError, match="conflicts with a data column"):
        create_dataframe({"person": [99, 88]}, index=[0, 1], index_name="person")


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_index_length_must_match_rows(backend: str) -> None:
    set_dataframe_backend(backend)

    with pytest.raises(MirtValidationError, match="length must match"):
        create_dataframe({"score": [1, 2]}, index=[0])


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_numpy_index_must_be_one_dimensional(backend: str) -> None:
    set_dataframe_backend(backend)

    with pytest.raises(MirtValidationError, match="one-dimensional"):
        create_dataframe({"score": [1, 2]}, index=np.array([[0], [1]]))


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_record_rows_are_supported(backend: str) -> None:
    set_dataframe_backend(backend)

    result = create_dataframe([{"score": 1}, {"score": 2}])

    assert result.shape == (2, 1)


def test_pandas_named_default_index_remains_an_index() -> None:
    set_dataframe_backend("pandas")

    result = create_dataframe({"score": [1, 2]}, index_name="person")

    assert isinstance(result, pd.DataFrame)
    assert result.index.name == "person"
    assert result.columns.tolist() == ["score"]


def test_pandas_explicit_index_retains_values_and_name() -> None:
    set_dataframe_backend("pandas")

    result = create_dataframe(
        {"score": [1, 2]}, index=["p1", "p2"], index_name="person"
    )

    assert result.index.tolist() == ["p1", "p2"]
    assert result.index.name == "person"


def test_backend_getter_is_public() -> None:
    assert mirt.get_dataframe_backend is get_dataframe_backend
    assert "get_dataframe_backend" in mirt.__all__
    assert "get_dataframe_backend" in mirt.utils.__all__


def test_personfit_materializes_person_identifiers_for_polars(monkeypatch: Any) -> None:
    from mirt.diagnostics import personfit as personfit_module

    set_dataframe_backend("polars")
    monkeypatch.setattr(
        personfit_module,
        "compute_personfit",
        lambda *args, **kwargs: {"outfit": np.array([1.0, 1.1])},
    )

    result = mirt.personfit(
        SimpleNamespace(model=object()),
        np.array([[0], [1]]),
        theta=np.array([0.0, 1.0]),
    )

    assert result.columns == ["person", "outfit"]
    assert result["person"].to_list() == [0, 1]


def test_personfit_forwards_significance_options(monkeypatch: Any) -> None:
    from mirt.diagnostics import personfit as personfit_module

    captured: dict[str, Any] = {}

    def fake_compute(*args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        captured.update(kwargs)
        return {
            "p_value": np.array([0.01, 0.5]),
            "p_value_adjusted": np.array([0.02, 0.5]),
            "aberrant": np.array([True, False]),
        }

    set_dataframe_backend("polars")
    monkeypatch.setattr(personfit_module, "compute_personfit", fake_compute)

    result = mirt.personfit(
        SimpleNamespace(model=object()),
        np.array([[0], [1]]),
        theta=np.array([0.0, 1.0]),
        p_adjust="holm",
        alpha=0.01,
        alternative="two-sided",
    )

    assert captured == {
        "p_adjust": "holm",
        "alpha": 0.01,
        "alternative": "two-sided",
    }
    assert result.columns == [
        "person",
        "p_value",
        "p_value_adjusted",
        "aberrant",
    ]


def test_dif_materializes_item_identifiers_for_polars(monkeypatch: Any) -> None:
    from mirt.diagnostics import dif as dif_module

    set_dataframe_backend("polars")
    monkeypatch.setattr(
        dif_module,
        "compute_dif",
        lambda **kwargs: {"statistic": np.array([2.0, 3.0])},
    )

    result = mirt.dif(
        np.array([[0, 1], [1, 0]]),
        np.array(["reference", "focal"]),
    )

    assert result.columns == ["item", "statistic"]
    assert result["item"].to_list() == [0, 1]


def test_default_backend_creates_polars_dataframe() -> None:
    assert isinstance(create_dataframe({"score": [1]}), pl.DataFrame)


def _binary_responses() -> np.ndarray:
    rng = np.random.default_rng(21)
    return (rng.random((150, 4)) < 0.6).astype(np.int_)


@pytest.mark.parametrize("frame_type", ["pandas", "polars"])
def test_fit_mirt_uses_dataframe_columns_as_item_names(frame_type: str) -> None:
    names = ["q_alpha", "q_beta", "q_gamma", "q_delta"]
    responses = _binary_responses()
    frame = (
        pd.DataFrame(responses, columns=names)
        if frame_type == "pandas"
        else pl.DataFrame(responses, schema=names, orient="row")
    )

    result = mirt.fit_mirt(frame, max_iter=5)
    set_dataframe_backend("pandas")

    assert result.model.item_names == names
    assert list(result.coef().index) == names
    assert list(mirt.itemfit(result, responses).index) == names


def test_explicit_item_names_take_precedence_over_columns() -> None:
    frame = pd.DataFrame(_binary_responses(), columns=["a", "b", "c", "d"])

    result = mirt.fit_mirt(frame, item_names=["w", "x", "y", "z"], max_iter=2)

    assert result.model.item_names == ["w", "x", "y", "z"]


@pytest.mark.parametrize(
    "columns",
    [None, ["a", "a", "b", "c"]],
    ids=["positional", "duplicate"],
)
def test_unnamed_or_ambiguous_columns_keep_default_item_names(
    columns: list[str] | None,
) -> None:
    frame = pd.DataFrame(_binary_responses(), columns=columns)

    result = mirt.fit_mirt(frame, max_iter=2)

    assert result.model.item_names == ["Item_1", "Item_2", "Item_3", "Item_4"]


def test_nan_missing_values_fit_like_negative_missing_codes() -> None:
    responses = _binary_responses()
    rng = np.random.default_rng(22)
    missing = rng.random(responses.shape) < 0.1
    coded = np.where(missing, -1, responses)
    with_nan = np.where(missing, np.nan, responses.astype(float))

    expected = mirt.fit_mirt(coded, max_iter=50)
    for data in (with_nan, pd.DataFrame(with_nan, columns=list("abcd"))):
        actual = mirt.fit_mirt(data, max_iter=50)
        assert actual.log_likelihood == pytest.approx(
            expected.log_likelihood, abs=1e-10
        )
        for name, values in expected.model.parameters.items():
            np.testing.assert_allclose(
                actual.model.parameters[name], values, rtol=0, atol=1e-10
            )


def test_nullable_pandas_columns_treat_na_as_missing() -> None:
    responses = _binary_responses()
    frame = pd.DataFrame(responses, columns=list("abcd")).astype("Int64")
    frame.iloc[0, 1] = pd.NA

    validated = mirt.validate_responses(frame)

    expected = responses.copy()
    expected[0, 1] = -1
    np.testing.assert_array_equal(validated, expected)
    assert mirt.fit_mirt(frame, max_iter=2).model.item_names == list("abcd")


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"a": ["yes", "no"], "b": [1, 0]}),
        # Numeric text is rejected as it is for plain arrays, not parsed.
        pd.DataFrame({"a": ["1", "0"], "b": [1, 0]}),
        pd.DataFrame({"a": pd.array(["1", None], dtype="string"), "b": [1, 0]}),
    ],
    ids=["text", "numeric-text", "string-dtype"],
)
def test_non_numeric_dataframe_is_still_rejected(frame: pd.DataFrame) -> None:
    with pytest.raises(mirt.MirtDataError, match="numeric"):
        mirt.validate_responses(frame)


def test_polars_default_column_names_keep_default_item_names() -> None:
    frame = pl.DataFrame(_binary_responses())
    assert frame.columns[0] == "column_0"

    result = mirt.fit_mirt(frame, max_iter=2)

    assert result.model.item_names == ["Item_1", "Item_2", "Item_3", "Item_4"]
