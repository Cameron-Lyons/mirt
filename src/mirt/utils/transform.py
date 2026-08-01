"""Data transformation utilities for IRT.

Provides functions for transforming response data into formats
suitable for IRT analysis.
"""

import numpy as np
from numpy.typing import NDArray


def key2binary(
    responses: NDArray,
    key: list | NDArray,
    scored_values: tuple[int, int] = (0, 1),
) -> NDArray[np.float64]:
    """Score multiple choice responses using an answer key.

    Converts responses to binary (correct/incorrect) based on
    comparison with the answer key.

    Parameters
    ----------
    responses : NDArray
        Response matrix. Shape: (n_persons, n_items).
        Can contain string or numeric values.
    key : list or NDArray
        Correct answer for each item. Length: n_items.
    scored_values : tuple
        Values for (incorrect, correct). Default (0, 1).

    Returns
    -------
    NDArray[np.float64]
        Binary scored responses.

    Examples
    --------
    >>> responses = np.array([['A', 'B', 'C'],
    ...                       ['A', 'C', 'C'],
    ...                       ['B', 'B', 'A']])
    >>> key = ['A', 'B', 'C']
    >>> scored = key2binary(responses, key)
    >>> print(scored)
    [[1. 1. 1.]
     [1. 0. 1.]
     [0. 1. 0.]]
    """
    responses = np.asarray(responses)
    key = np.asarray(key)

    if responses.shape[1] != len(key):
        raise ValueError(
            f"Number of items ({responses.shape[1]}) does not match "
            f"key length ({len(key)})"
        )

    scored = np.zeros(responses.shape, dtype=np.float64)

    for j in range(responses.shape[1]):
        correct = responses[:, j] == key[j]
        scored[:, j] = np.where(correct, scored_values[1], scored_values[0])

    return scored


def poly2dich(
    responses: NDArray[np.float64],
    cutoff: int | float | list[int] | list[float] | NDArray | None = None,
    method: str = "threshold",
) -> NDArray[np.float64]:
    """Convert polytomous responses to dichotomous.

    Parameters
    ----------
    responses : NDArray[np.float64]
        Polytomous response matrix. Shape: (n_persons, n_items).
        Values should be 0, 1, 2, ..., n_categories-1.
    cutoff : scalar or array-like, optional
        Threshold for dichotomization. Responses >= cutoff become 1.
        If None, uses median split for each item.
        A scalar applies to every item; an array-like value must provide one
        cutoff per item. Only used when ``method="threshold"``.
    method : str
        Dichotomization method:
        - "threshold": Simple threshold at cutoff
        - "median": Median split per item
        - "adjacent": Create separate items for adjacent categories

    Returns
    -------
    NDArray[np.float64]
        Dichotomized responses.

    Examples
    --------
    >>> # 5-point Likert scale (0-4)
    >>> responses = np.array([[0, 2, 4], [1, 3, 3], [2, 1, 2]])
    >>> binary = poly2dich(responses, cutoff=2)
    >>> print(binary)
    [[0. 1. 1.]
     [0. 1. 1.]
     [1. 0. 1.]]
    """
    responses = np.asarray(responses, dtype=np.float64)
    if responses.ndim != 2:
        raise ValueError(f"responses must be a 2D array, got {responses.ndim}D")

    n_persons, n_items = responses.shape
    if n_persons == 0 or n_items == 0:
        raise ValueError("responses must contain at least one person and one item")
    if np.any(np.isinf(responses)):
        raise ValueError("responses must contain finite values or NaN")

    if method in ("threshold", "median"):
        if method == "median" or cutoff is None:
            cutoff_values = np.nanmedian(responses, axis=0)
        else:
            try:
                cutoff_values = np.asarray(cutoff, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("cutoff must contain numeric values") from exc

            if cutoff_values.ndim == 0:
                cutoff_values = np.full(n_items, cutoff_values.item())
            elif cutoff_values.shape != (n_items,):
                raise ValueError(
                    f"cutoff must be a scalar or one value per item; expected {n_items}"
                )
            if not np.all(np.isfinite(cutoff_values)):
                raise ValueError("cutoff must contain only finite values")

        binary = np.empty((n_persons, n_items), dtype=np.float64)
        np.greater_equal(responses, cutoff_values, out=binary)
        binary[np.isnan(responses)] = np.nan
        return binary

    if method == "adjacent":
        observed = responses[~np.isnan(responses)]
        if observed.size == 0:
            raise ValueError(
                "adjacent conversion requires at least one observed response"
            )
        if np.any(observed < 0) or np.any(observed != np.trunc(observed)):
            raise ValueError(
                "adjacent conversion requires non-negative integer category codes"
            )

        n_categories = int(observed.max()) + 1
        if n_categories < 2:
            raise ValueError("adjacent conversion requires at least two categories")

        thresholds = np.arange(1, n_categories, dtype=np.float64)
        expanded = np.empty((n_persons, n_items, n_categories - 1), dtype=np.float64)
        np.greater_equal(responses[:, :, None], thresholds, out=expanded)
        expanded[np.isnan(responses)] = np.nan
        return expanded.reshape(n_persons, -1)

    raise ValueError(f"Unknown method: {method}")


def reverse_score(
    responses: NDArray[np.float64],
    items: list[int] | None = None,
    max_score: int | None = None,
) -> NDArray[np.float64]:
    """Reverse score selected items.

    Transforms responses so that x becomes (max_score - x).

    Parameters
    ----------
    responses : NDArray[np.float64]
        Response matrix. Shape: (n_persons, n_items).
    items : list of int, optional
        Indices of items to reverse. If None, reverse all items.
    max_score : int, optional
        Maximum possible score. If None, inferred from data.

    Returns
    -------
    NDArray[np.float64]
        Responses with selected items reverse scored.

    Examples
    --------
    >>> responses = np.array([[0, 1, 2], [1, 0, 1], [2, 2, 0]])
    >>> reversed_resp = reverse_score(responses, items=[1], max_score=2)
    >>> print(reversed_resp)
    [[0. 1. 2.]
     [1. 2. 1.]
     [2. 0. 0.]]
    """
    responses = np.asarray(responses, dtype=np.float64).copy()
    n_items = responses.shape[1]

    if items is None:
        items = list(range(n_items))

    if max_score is None:
        max_score = int(np.nanmax(responses))

    for j in items:
        mask = ~np.isnan(responses[:, j])
        responses[mask, j] = max_score - responses[mask, j]

    return responses


def expand_table(
    table: NDArray[np.float64],
    freq_col: int = -1,
) -> NDArray[np.float64]:
    """Expand frequency table to full response matrix.

    Converts a table with response patterns and frequencies to
    a full response matrix with one row per person.

    Parameters
    ----------
    table : NDArray[np.float64]
        Table with response patterns and frequencies.
        Each row is a unique response pattern.
        Last column (or freq_col) contains frequencies.
    freq_col : int
        Column index containing frequencies. Default -1 (last column).

    Returns
    -------
    NDArray[np.float64]
        Expanded response matrix.

    Examples
    --------
    >>> # Table: pattern + frequency
    >>> table = np.array([[0, 0, 0, 5],   # 5 people with pattern 000
    ...                   [1, 0, 0, 3],   # 3 people with pattern 100
    ...                   [1, 1, 1, 2]])  # 2 people with pattern 111
    >>> responses = expand_table(table)
    >>> print(responses.shape)
    (10, 3)
    """
    table = np.asarray(table, dtype=np.float64)

    freqs = table[:, freq_col].astype(int)

    if freq_col == -1:
        patterns = table[:, :-1]
    else:
        patterns = np.delete(table, freq_col, axis=1)

    n_total = np.sum(freqs)
    n_items = patterns.shape[1]

    expanded = np.zeros((n_total, n_items), dtype=np.float64)

    row = 0
    for i, freq in enumerate(freqs):
        for _ in range(freq):
            expanded[row] = patterns[i]
            row += 1

    return expanded


def collapse_table(
    responses: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Collapse response matrix to frequency table.

    Inverse of expand_table. Creates table of unique patterns
    with frequencies.

    Parameters
    ----------
    responses : NDArray[np.float64]
        Response matrix. Shape: (n_persons, n_items).

    Returns
    -------
    patterns : NDArray[np.float64]
        Unique response patterns. Shape: (n_patterns, n_items).
    frequencies : NDArray[np.intp]
        Frequency of each pattern. Shape: (n_patterns,).

    Examples
    --------
    >>> responses = np.array([[0, 0], [1, 1], [0, 0], [1, 1], [1, 1]])
    >>> patterns, freqs = collapse_table(responses)
    >>> print(patterns)
    [[0. 0.]
     [1. 1.]]
    >>> print(freqs)
    [2 3]
    """
    responses = np.asarray(responses, dtype=np.float64)

    unique_patterns, indices, counts = np.unique(
        responses, axis=0, return_inverse=True, return_counts=True
    )

    return unique_patterns, counts


def recode_responses(
    responses: NDArray[np.float64],
    mapping: dict[int, int],
    items: list[int] | None = None,
) -> NDArray[np.float64]:
    """Recode response values according to a mapping.

    Parameters
    ----------
    responses : NDArray[np.float64]
        Response matrix.
    mapping : dict
        Mapping from old values to new values.
        Example: {1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
    items : list of int, optional
        Items to recode. If None, recode all items.

    Returns
    -------
    NDArray[np.float64]
        Recoded responses.

    Examples
    --------
    >>> # Collapse 5-point scale to 3-point
    >>> responses = np.array([[1, 2, 5], [3, 4, 1]])
    >>> mapping = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2}
    >>> recoded = recode_responses(responses, mapping)
    >>> print(recoded)
    [[0. 0. 2.]
     [1. 1. 0.]]
    """
    responses = np.asarray(responses, dtype=np.float64).copy()
    n_items = responses.shape[1]

    if items is None:
        items = list(range(n_items))

    for j in items:
        for old_val, new_val in mapping.items():
            mask = responses[:, j] == old_val
            responses[mask, j] = new_val

    return responses


def likert2int(
    responses: NDArray,
    labels: list[str] | None = None,
) -> NDArray[np.float64]:
    """Convert Likert scale labels to integers.

    Parameters
    ----------
    responses : NDArray
        Response matrix with string labels.
    labels : list of str, optional
        Ordered list of labels. If None, uses unique values sorted.

    Returns
    -------
    NDArray[np.float64]
        Integer-coded responses (0, 1, 2, ...).

    Examples
    --------
    >>> responses = np.array([['Agree', 'Disagree'],
    ...                       ['Strongly Agree', 'Neutral']])
    >>> labels = ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree']
    >>> coded = likert2int(responses, labels)
    >>> print(coded)
    [[3. 1.]
     [4. 2.]]
    """
    responses = np.asarray(responses)

    if labels is None:
        labels = sorted(list(set(responses.ravel())))

    label_to_int = {label: i for i, label in enumerate(labels)}

    coded = np.zeros(responses.shape, dtype=np.float64)
    for i in range(responses.shape[0]):
        for j in range(responses.shape[1]):
            val = responses[i, j]
            if val in label_to_int:
                coded[i, j] = label_to_int[val]
            else:
                coded[i, j] = np.nan

    return coded
