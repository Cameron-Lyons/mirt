from typing import Any

import numpy as np
from numpy.typing import NDArray


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
        Value used to represent missing responses.

    Returns
    -------
    ndarray of shape (n_persons, n_items)
        Validated response matrix as integer array.

    Raises
    ------
    ValueError
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
        raise ValueError(f"responses must be 2D array, got {responses.ndim}D")

    n_persons, n_cols = responses.shape

    if n_persons == 0:
        raise ValueError("responses cannot be empty")

    if n_items is not None and n_cols != n_items:
        raise ValueError(f"responses has {n_cols} items, expected {n_items}")

    responses = responses.astype(np.int_)

    if not allow_missing:
        if np.any(responses < 0):
            raise ValueError(
                "responses contains negative values (missing data not allowed)"
            )

    valid_mask = responses != missing_code
    if np.any(responses[valid_mask] < 0):
        raise ValueError(
            f"responses contains negative values other than missing code ({missing_code})"
        )

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
        If list, specifies categories per item. If None, inferred from data.

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

    Examples
    --------
    >>> from mirt.utils.data import check_response_pattern
    >>> import numpy as np
    >>> data = np.array([[1, 0, 1], [0, -1, 0], [1, 1, 1]])
    >>> stats = check_response_pattern(data)
    >>> print(f"Missing rate: {stats['missing_rate']:.2%}")
    """
    responses = np.asarray(responses)
    n_persons, n_items = responses.shape

    missing_mask = responses < 0
    missing_rate = missing_mask.mean()
    missing_by_item = missing_mask.mean(axis=0)
    missing_by_person = missing_mask.sum(axis=1)

    valid_responses = np.where(missing_mask, np.nan, responses)

    if n_categories is None:
        max_resp = int(np.nanmax(valid_responses))
        n_categories = max_resp + 1

    if isinstance(n_categories, int):
        max_response = n_categories - 1
    else:
        max_response = max(n_categories) - 1

    all_min = np.all((responses == 0) | (responses < 0), axis=1)
    all_max = np.all((responses == max_response) | (responses < 0), axis=1)

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
