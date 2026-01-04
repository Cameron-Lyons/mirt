from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

_DATAFRAME_BACKEND: str | None = None


def set_dataframe_backend(backend: str) -> None:
    global _DATAFRAME_BACKEND
    if backend not in ("pandas", "polars"):
        raise ValueError(f"backend must be 'pandas' or 'polars', got {backend}")
    _DATAFRAME_BACKEND = backend


def get_dataframe_backend() -> str:
    global _DATAFRAME_BACKEND
    if _DATAFRAME_BACKEND is not None:
        return _DATAFRAME_BACKEND

    try:
        import polars  # noqa: F401

        return "polars"
    except ImportError:
        pass

    try:
        import pandas  # noqa: F401

        return "pandas"
    except ImportError:
        pass

    raise ImportError("Either pandas or polars must be installed")


def create_dataframe(
    data: dict[str, NDArray[Any] | list[Any]],
    index: list[str] | NDArray[Any] | None = None,
    index_name: str | None = None,
) -> Any:
    backend = get_dataframe_backend()

    if backend == "polars":
        import polars as pl

        df = pl.DataFrame(data)
        if index is not None:
            index_col = index_name or "index"
            if isinstance(index, np.ndarray):
                index = index.tolist()
            df = df.with_columns(pl.Series(index_col, index))
            cols = [index_col] + [c for c in df.columns if c != index_col]
            df = df.select(cols)
        return df
    else:
        import pandas as pd

        df = pd.DataFrame(data)
        if index is not None:
            df.index = index
        if index_name is not None:
            df.index.name = index_name
        return df
