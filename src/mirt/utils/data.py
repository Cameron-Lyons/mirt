"""Data validation and preprocessing utilities."""

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray


def validate_responses(
    responses: Union[NDArray, list],
    n_items: Optional[int] = None,
    allow_missing: bool = True,
    missing_code: int = -1,
) -> NDArray[np.int_]:
    """Validate and preprocess a response matrix.

    Parameters
    ----------
    responses : array-like of shape (n_persons, n_items)
        Response matrix to validate.
    n_items : int, optional
        Expected number of items. If provided, validates column count.
    allow_missing : bool, default=True
        Whether to allow missing values.
    missing_code : int, default=-1
        Value used to code missing responses.

    Returns
    -------
    ndarray of shape (n_persons, n_items)
        Validated response matrix with integer dtype.

    Raises
    ------
    ValueError
        If responses are invalid (wrong shape, out of range, etc.).

    Examples
    --------
    >>> responses = [[1, 0, 1], [0, 1, 0]]
    >>> validated = validate_responses(responses, n_items=3)
    >>> print(validated.dtype)
    int64
    """
    responses = np.asarray(responses)

    # Check dimensionality
    if responses.ndim != 2:
        raise ValueError(
            f"responses must be 2D array, got {responses.ndim}D"
        )

    n_persons, n_cols = responses.shape

    if n_persons == 0:
        raise ValueError("responses cannot be empty")

    if n_items is not None and n_cols != n_items:
        raise ValueError(
            f"responses has {n_cols} items, expected {n_items}"
        )

    # Convert to integer
    responses = responses.astype(np.int_)

    # Check for invalid values
    if not allow_missing:
        if np.any(responses < 0):
            raise ValueError(
                "responses contains negative values (missing data not allowed)"
            )

    # Check that non-missing responses are non-negative
    valid_mask = responses != missing_code
    if np.any(responses[valid_mask] < 0):
        raise ValueError(
            f"responses contains negative values other than missing code ({missing_code})"
        )

    return responses


def check_response_pattern(
    responses: NDArray[np.int_],
    n_categories: Optional[Union[int, list[int]]] = None,
) -> dict:
    """Analyze response patterns for quality checking.

    Parameters
    ----------
    responses : ndarray of shape (n_persons, n_items)
        Response matrix.
    n_categories : int or list of int, optional
        Number of categories per item. If not provided, inferred from data.

    Returns
    -------
    dict
        Dictionary containing:
        - n_persons: Number of respondents
        - n_items: Number of items
        - missing_rate: Overall missing data rate
        - missing_by_item: Missing rate per item
        - missing_by_person: Missing count per person
        - response_frequencies: Frequency of each response per item
        - extreme_patterns: Count of all-correct/all-incorrect patterns
    """
    responses = np.asarray(responses)
    n_persons, n_items = responses.shape

    # Missing data
    missing_mask = responses < 0
    missing_rate = missing_mask.mean()
    missing_by_item = missing_mask.mean(axis=0)
    missing_by_person = missing_mask.sum(axis=1)

    # Response frequencies
    valid_responses = np.where(missing_mask, np.nan, responses)

    if n_categories is None:
        # Infer from data
        max_resp = int(np.nanmax(valid_responses))
        n_categories = max_resp + 1

    # Count extreme patterns (all 0 or all max)
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
    table: NDArray,
    freq_col: int = -1,
) -> NDArray[np.int_]:
    """Expand a frequency table to individual response patterns.

    Parameters
    ----------
    table : ndarray
        Response pattern table with a frequency column.
    freq_col : int, default=-1
        Index of the frequency column.

    Returns
    -------
    ndarray
        Expanded response matrix (one row per observation).

    Examples
    --------
    >>> # Table: pattern + frequency
    >>> table = np.array([
    ...     [1, 0, 1, 10],  # pattern [1,0,1] appears 10 times
    ...     [0, 1, 0, 5],   # pattern [0,1,0] appears 5 times
    ... ])
    >>> expanded = expand_table(table)
    >>> print(expanded.shape)
    (15, 3)
    """
    table = np.asarray(table)

    if table.ndim != 2:
        raise ValueError("table must be 2D")

    # Extract frequencies
    freqs = table[:, freq_col].astype(int)

    # Extract patterns (all columns except frequency)
    if freq_col == -1:
        patterns = table[:, :-1]
    else:
        patterns = np.delete(table, freq_col, axis=1)

    # Expand
    expanded = np.repeat(patterns, freqs, axis=0)

    return expanded.astype(np.int_)
