import numpy as np
from numpy.typing import NDArray


def validate_responses(
    responses: NDArray | list,
    n_items: int | None = None,
    allow_missing: bool = True,
    missing_code: int = -1,
) -> NDArray[np.int_]:
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
) -> dict:
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
    table: NDArray,
    freq_col: int = -1,
) -> NDArray[np.int_]:
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
