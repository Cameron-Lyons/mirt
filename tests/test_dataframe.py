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
