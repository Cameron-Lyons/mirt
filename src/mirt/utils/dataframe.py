from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from mirt.exceptions import MirtValidationError

DataFrameBackend: TypeAlias = Literal["auto", "pandas", "polars"]
DataFrameInput: TypeAlias = Mapping[str, Any] | Sequence[Mapping[str, Any]]

_DATAFRAME_BACKEND: Literal["pandas", "polars"] | None = None


def _load_backend(backend: Literal["pandas", "polars"]) -> Any:
    try:
        return import_module(backend)
    except ImportError as exc:
        raise ImportError(
            f"The {backend} DataFrame backend is not installed. "
            f"Install it with `pip install mirt[{backend}]`."
        ) from exc


def set_dataframe_backend(backend: DataFrameBackend | None) -> None:
    """Select the DataFrame implementation used for tabular results.

    Pass ``"auto"`` or ``None`` to restore automatic selection, which prefers
    Polars when both optional dependencies are installed.
    """
    global _DATAFRAME_BACKEND

    if backend in (None, "auto"):
        _DATAFRAME_BACKEND = None
        return
    if backend not in ("pandas", "polars"):
        raise MirtValidationError(
            "Invalid DataFrame backend",
            parameter="backend",
            value=backend,
            expected="'auto', 'pandas', 'polars', or None",
        )

    _load_backend(backend)
    _DATAFRAME_BACKEND = backend


def get_dataframe_backend() -> Literal["pandas", "polars"]:
    """Return the selected or automatically detected DataFrame backend."""
    global _DATAFRAME_BACKEND

    if _DATAFRAME_BACKEND is not None:
        return _DATAFRAME_BACKEND

    for backend in ("polars", "pandas"):
        try:
            _load_backend(backend)
        except ImportError:
            continue
        _DATAFRAME_BACKEND = backend
        return backend

    raise ImportError(
        "DataFrame output requires pandas or polars. Install one with "
        "`pip install mirt[pandas]` or `pip install mirt[polars]`."
    )


def _validate_index(index: Sequence[Any] | NDArray[Any], n_rows: int) -> None:
    if isinstance(index, np.ndarray) and index.ndim != 1:
        raise MirtValidationError(
            "DataFrame index must be one-dimensional",
            parameter="index",
            value=f"shape={index.shape}",
            expected="one-dimensional sequence",
        )
    if len(index) != n_rows:
        raise MirtValidationError(
            "DataFrame index length must match the number of rows",
            parameter="index",
            value=len(index),
            expected=str(n_rows),
        )


def create_dataframe(
    data: DataFrameInput,
    index: Sequence[Any] | NDArray[Any] | None = None,
    index_name: str | None = None,
) -> Any:
    """Create a pandas or Polars DataFrame with consistent row identifiers."""
    backend = get_dataframe_backend()

    if backend == "polars":
        pl = _load_backend("polars")

        df = pl.DataFrame(data)
        if index is not None:
            index_col = index_name or "index"
            _validate_index(index, df.height)
            if index_col in df.columns:
                raise MirtValidationError(
                    "DataFrame index name conflicts with a data column",
                    parameter="index_name",
                    value=index_col,
                    expected="a unique column name",
                )
            df.insert_column(0, pl.Series(index_col, index))
        elif index_name is not None:
            if index_name in df.columns:
                if df.columns[0] != index_name:
                    df = df.select(
                        [index_name, *[c for c in df.columns if c != index_name]]
                    )
            else:
                row_index = np.arange(df.height, dtype=np.int64)
                df.insert_column(0, pl.Series(index_name, row_index))
        return df
    else:
        pd = _load_backend("pandas")

        df = pd.DataFrame(data)
        if index is not None:
            _validate_index(index, len(df))
            df.index = index
        if index_name is not None:
            df.index.name = index_name
        return df
