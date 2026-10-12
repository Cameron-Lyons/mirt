from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mirt.exceptions import MirtDataError


def validate_responses(
    responses: Any,
    n_items: int | None = None,
    allow_missing: bool = True,
    missing_code: int = -1,
    *,
    nan_as_missing: bool = True,
) -> NDArray[np.int_]:
    """Validate and convert response data for IRT analysis.

    Performs input validation on response matrices, checking dimensions,
    response codes, and handling of missing data.

    Parameters
    ----------
    responses : array-like of shape (n_persons, n_items)
        Response data to validate. Can be a list, numpy array, pandas or
        polars DataFrame, or any array-like object. Nullable DataFrame columns
        are read with their missing entries as ``NaN``.
    n_items : int, optional
        Expected number of items. If provided, validates that responses
        have this many columns.
    allow_missing : bool, default=True
        Whether to allow missing responses coded as missing_code.
    missing_code : int, default=-1
        Value used to represent missing responses in the returned array. Any
        negative input value is treated as missing and normalized to this code.
    nan_as_missing : bool, default=True
        Treat ``NaN`` entries (R's ``NA``) as missing responses when
        ``allow_missing`` is true. Infinite and non-integer values are always
        rejected.

    Returns
    -------
    ndarray of shape (n_persons, n_items)
        Validated response matrix as integer array.

    Raises
    ------
    MirtDataError
        If responses are not 2D, empty, have wrong number of items,
        or contain invalid values.

    Examples
    --------
    >>> from mirt import validate_responses
    >>> import numpy as np
    >>> data = [[1, 0, 1], [0, 1, 0]]
    >>> validated = validate_responses(data)
    >>> print(validated.dtype)
    int64

    >>> # With missing data
    >>> data_with_missing = [[1, -1, 1], [0, 1, 0]]
    >>> validated = validate_responses(data_with_missing, allow_missing=True)

    >>> # NaN is missing as well
    >>> validate_responses([[1.0, np.nan], [0.0, 1.0]]).tolist()
    [[1, -1], [0, 1]]
    """
    responses = _response_array(responses)

    if responses.ndim != 2:
        raise MirtDataError(
            f"responses must be 2D array, got {responses.ndim}D",
        )

    n_persons, n_cols = responses.shape

    if n_persons == 0:
        raise MirtDataError(
            "responses must contain at least one person",
            n_persons=0,
            n_items=n_cols,
        )

    if n_cols == 0:
        raise MirtDataError(
            "responses must contain at least one item",
            n_persons=n_persons,
            n_items=0,
        )

    if n_items is not None and n_cols != n_items:
        raise MirtDataError(
            f"responses has {n_cols} items, expected {n_items}",
            n_persons=n_persons,
            n_items=n_cols,
        )

    dtype_kind = responses.dtype.kind
    if dtype_kind not in "biuf":
        raise MirtDataError(
            "response data must be numeric",
            n_persons=n_persons,
            n_items=n_cols,
        )

    if dtype_kind == "f":
        nan_mask = np.isnan(responses)
        if nan_as_missing and np.any(nan_mask):
            if not allow_missing:
                raise MirtDataError(
                    "responses contains NaN values (missing data not allowed)",
                    n_persons=n_persons,
                    n_items=n_cols,
                )
            responses = np.where(nan_mask, missing_code, responses)
        if not np.all(np.isfinite(responses)):
            raise MirtDataError(
                "finite response codes are required; values must be finite integer values",
                n_persons=n_persons,
                n_items=n_cols,
            )
        if not np.all(responses == np.trunc(responses)):
            raise MirtDataError(
                "integer response codes are required; values must be integer-valued response codes",
                n_persons=n_persons,
                n_items=n_cols,
            )

    int_bounds = np.iinfo(np.int_)
    if dtype_kind in "fu" and (
        np.any(responses < int_bounds.min) or np.any(responses > int_bounds.max)
    ):
        raise MirtDataError(
            "response codes exceed the supported integer range",
            n_persons=n_persons,
            n_items=n_cols,
        )

    responses = responses.astype(np.int_, copy=False)

    missing_mask = responses < 0
    if np.any(missing_mask):
        if not allow_missing:
            raise MirtDataError(
                "responses contains negative values (missing data not allowed)",
                n_persons=n_persons,
                n_items=n_cols,
            )
        if np.any(responses[missing_mask] != missing_code):
            responses = responses.copy()
            responses[missing_mask] = missing_code

    return responses


def _response_array(responses: Any) -> NDArray[Any]:
    """Convert responses to an array, reading nullable DataFrame columns as floats."""
    array = np.asarray(responses)
    if array.dtype != object or not hasattr(responses, "to_numpy"):
        return array
    # pandas nullable columns (Int64, boolean) convert to objects holding pd.NA.
    # Only numeric and boolean columns are read this way, so text columns such
    # as "1" are still rejected as non-numeric, as they are for plain arrays.
    dtypes = getattr(responses, "dtypes", ())
    if not all(getattr(dtype, "kind", "O") in "biuf" for dtype in dtypes):
        return array
    try:
        return np.asarray(responses.to_numpy(dtype=np.float64, na_value=np.nan))
    except (TypeError, ValueError):
        return array


def _missing_coded_responses(responses: Any, missing_code: int = -1) -> NDArray[Any]:
    """Read responses as an array whose ``NaN`` entries are ``missing_code``.

    This is the missing-data convention of :func:`validate_responses` for
    callers that check response codes themselves: nullable DataFrame columns
    are read as floats and every ``NaN`` becomes ``missing_code``. Other
    values, including infinities and non-numeric entries, are returned
    unchanged for the caller to validate.
    """
    array = _response_array(responses)
    if array.dtype.kind == "f":
        missing = np.isnan(array)
        if np.any(missing):
            array = np.where(missing, float(missing_code), array)
    return array


def response_column_names(responses: Any) -> list[str] | None:
    """Return the column labels of a DataFrame-like response matrix.

    Returns ``None`` for arrays, for non-unique labels, and for the default
    positional labels of pandas (``0, 1, ...``) and polars (``column_0,
    column_1, ...``), which carry no item names.
    """
    if isinstance(responses, np.ndarray):
        return None
    columns = getattr(responses, "columns", None)
    if columns is None:
        return None
    try:
        labels = list(columns)
    except TypeError:
        return None
    if all(
        isinstance(label, (int, np.integer)) and not isinstance(label, bool)
        for label in labels
    ) and [int(label) for label in labels] == list(range(len(labels))):
        return None
    names = [str(label) for label in labels]
    if names == [f"column_{index}" for index in range(len(names))]:
        return None
    if len(set(names)) != len(names):
        return None
    return names


def align_response_columns(
    responses: Any,
    item_names: Sequence[str] | None,
) -> Any:
    """Select and order the columns of a named response frame by item name.

    A pandas or polars DataFrame whose columns carry item names (see
    :func:`response_column_names`) is matched to ``item_names`` by label, so
    its columns may come in any order and may include extra columns, such as
    respondent identifiers, which are dropped. Arrays, frames with default
    positional labels, and frames whose columns already equal ``item_names``
    keep positional semantics and are returned unchanged, as is everything
    when ``item_names`` is ``None`` or contains duplicates. A model whose
    item names are the defaults ``Item_1, Item_2, ...`` (or ``Item_0, ...``),
    as for a fit to an unnamed array, also reads a frame without those labels
    by position.

    Parameters
    ----------
    responses : array-like or DataFrame
        Response matrix, typically one being scored against a fitted model.
    item_names : sequence of str or None
        The model's item names in model order, such as
        ``result.model.item_names``.

    Returns
    -------
    array-like or DataFrame
        ``responses`` itself when no reordering applies; otherwise a frame of
        the same library holding exactly the ``item_names`` columns in model
        order.

    Raises
    ------
    MirtDataError
        If a frame with named columns lacks columns for some of the model's
        (non-default) item names. The message lists the missing items;
        convert the frame to an array (``frame.to_numpy()``) to score its
        columns by position instead.

    Examples
    --------
    >>> import pandas as pd
    >>> frame = pd.DataFrame({"id": [7, 8], "Q2": [1, 0], "Q1": [0, 1]})
    >>> align_response_columns(frame, ["Q1", "Q2"]).columns.tolist()
    ['Q1', 'Q2']
    """
    if item_names is None:
        return responses
    names = response_column_names(responses)
    if names is None:
        return responses
    items = [str(name) for name in item_names]
    if names == items or len(set(items)) != len(items):
        return responses

    positions = {name: index for index, name in enumerate(names)}
    missing = [name for name in items if name not in positions]
    if missing:
        n_items = len(items)
        default_labels = (
            [f"Item_{index + 1}" for index in range(n_items)],
            [f"Item_{index}" for index in range(n_items)],
        )
        if items in default_labels:
            return responses
        shown = ", ".join(missing[:10])
        if len(missing) > 10:
            shown += f", ... ({len(missing)} in total)"
        raise MirtDataError(
            f"responses has no columns for the model's items {shown}; "
            "DataFrame columns are matched to the model's item names, so "
            "pass an array (for example frame.to_numpy()) to score columns "
            "by position",
            n_items=len(names),
        )

    selected = [positions[name] for name in items]
    labels = list(responses.columns)
    try:
        return responses[[labels[index] for index in selected]]
    except (KeyError, TypeError, ValueError, IndexError):
        # Frame-like objects without list selection are read as arrays.
        return _response_array(responses)[:, selected]


def check_response_pattern(
    responses: NDArray[np.int_],
    n_categories: int | list[int] | None = None,
) -> dict[str, Any]:
    """Analyze response patterns and data quality.

    Provides summary statistics about response data including missing
    data rates, extreme response patterns, and basic descriptives.

    Parameters
    ----------
    responses : ndarray of shape (n_persons, n_items)
        Response matrix with missing data coded as negative values.
    n_categories : int or list of int, optional
        Number of response categories. If int, applies to all items.
        If list, specifies categories per item. If None, each item's maximum
        category is inferred independently from its observed responses.

    Returns
    -------
    dict
        Dictionary containing:

        - n_persons: Number of respondents
        - n_items: Number of items
        - missing_rate: Overall proportion of missing responses
        - missing_by_item: Missing rate per item
        - missing_by_person: Count of missing responses per person
        - extreme_patterns: Counts of all-minimum and all-maximum patterns

        Respondents with no observed responses are excluded from both extreme
        pattern counts.

    Raises
    ------
    ValueError
        If response data or category definitions are invalid, or an observed
        response exceeds its declared category range.

    Examples
    --------
    >>> from mirt.utils.data import check_response_pattern
    >>> import numpy as np
    >>> data = np.array([[1, 0, 1], [0, -1, 0], [1, 1, 1]])
    >>> stats = check_response_pattern(data)
    >>> print(f"Missing rate: {stats['missing_rate']:.2%}")
    """
    response_array = np.asarray(responses)
    if (
        response_array.ndim == 2
        and response_array.shape[0] > 0
        and response_array.shape[1] > 0
        and response_array.dtype.kind in "bi"
    ):
        responses = response_array.astype(np.int_, copy=False)
    else:
        responses = validate_responses(response_array)
    n_persons, n_items = responses.shape

    missing_mask = responses < 0
    missing_rate = missing_mask.mean()
    missing_by_item = missing_mask.mean(axis=0)
    missing_by_person = missing_mask.sum(axis=1)
    observed_by_person = (~missing_mask).any(axis=1)

    if n_categories is None:
        max_response = responses.max(axis=0)
    elif isinstance(n_categories, (int, np.integer)) and not isinstance(
        n_categories, bool
    ):
        if n_categories < 1:
            raise ValueError("n_categories must be positive")
        max_response = np.full(n_items, int(n_categories) - 1, dtype=np.int_)
    else:
        categories = np.asarray(n_categories)
        if categories.ndim != 1 or categories.shape[0] != n_items:
            raise ValueError(f"n_categories must contain {n_items} values")
        if categories.dtype.kind not in "iuf" or not np.all(np.isfinite(categories)):
            raise ValueError("n_categories must contain finite integers")
        if not np.all(categories == np.trunc(categories)):
            raise ValueError("n_categories must contain integers")
        categories = categories.astype(np.int_, copy=False)
        if np.any(categories < 1):
            raise ValueError("n_categories values must be positive")
        max_response = categories - 1

    if np.any((~missing_mask) & (responses > max_response[None, :])):
        raise ValueError("responses contain values outside n_categories")

    all_min = observed_by_person & np.all((responses == 0) | missing_mask, axis=1)
    all_max = observed_by_person & np.all(
        (responses == max_response[None, :]) | missing_mask,
        axis=1,
    )

    return {
        "n_persons": n_persons,
        "n_items": n_items,
        "missing_rate": float(missing_rate),
        "missing_by_item": missing_by_item.tolist(),
        "missing_by_person": missing_by_person.tolist(),
        "extreme_patterns": {
            "all_minimum": int(all_min.sum()),
            "all_maximum": int(all_max.sum()),
        },
    }
