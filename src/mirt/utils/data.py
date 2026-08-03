from typing import Any

import numpy as np
from numpy.typing import NDArray

from mirt.exceptions import MirtDataError


def validate_responses(
    responses: NDArray[Any] | list[Any],
    n_items: int | None = None,
    allow_missing: bool = True,
    missing_code: int = -1,
) -> NDArray[np.int_]:
    """Validate and convert response data for IRT analysis.

    Performs input validation on response matrices, checking dimensions,
    response codes, and handling of missing data.

    Parameters
    ----------
    responses : array-like of shape (n_persons, n_items)
        Response data to validate. Can be a list, numpy array, or any
        array-like object.
    n_items : int, optional
        Expected number of items. If provided, validates that responses
        have this many columns.
    allow_missing : bool, default=True
        Whether to allow missing responses coded as missing_code.
    missing_code : int, default=-1
        Value used to represent missing responses in the returned array. Any
        negative input value is treated as missing and normalized to this code.

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
    """
    responses = np.asarray(responses)

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


def expand_table(
    table: NDArray[Any],
    freq_col: int = -1,
) -> NDArray[np.int_]:
    """Expand frequency table to individual response records.

    Converts a summarized frequency table where each row represents a
    response pattern with a frequency count into individual response
    records suitable for IRT analysis.

    Parameters
    ----------
    table : ndarray of shape (n_patterns, n_items + 1)
        Frequency table with response patterns and counts. Each row
        contains item responses followed by (or preceded by) the frequency.
    freq_col : int, default=-1
        Column index containing frequency counts. Default is last column.

    Returns
    -------
    ndarray of shape (n_persons, n_items)
        Expanded response matrix where each pattern is repeated according
        to its frequency.

    Raises
    ------
    ValueError
        If table is not 2D.

    Examples
    --------
    >>> import numpy as np
    >>> # Table: [item1, item2, frequency]
    >>> freq_table = np.array([[1, 1, 10], [1, 0, 5], [0, 0, 3]])
    >>> data = expand_table(freq_table)
    >>> print(data.shape)
    (18, 2)
    """
    table = np.asarray(table)

    if table.ndim != 2:
        raise ValueError("table must be 2D")

    freqs = table[:, freq_col].astype(int)

    if freq_col == -1:
        patterns = table[:, :-1]
    else:
        patterns = np.delete(table, freq_col, axis=1)

    expanded = np.repeat(patterns, freqs, axis=0)

    return expanded.astype(np.int_)
