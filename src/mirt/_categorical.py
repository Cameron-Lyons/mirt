"""Internal categorical counting, likelihood, and sampling helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from mirt.constants import PROB_EPSILON
from mirt.exceptions import MirtModelError

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel

_FREQUENCY_CHUNK_ELEMENTS = 1_000_000
_MAX_FREQUENCY_ENTRIES = 5_000_000
_LIKELIHOOD_BLOCK_ELEMENTS = 1_000_000


def category_offsets(category_counts: list[int] | NDArray[np.intp]) -> NDArray[np.intp]:
    """Return each item's first row in an item-major category table."""
    counts = np.asarray(category_counts, dtype=np.intp)
    offsets = np.zeros(counts.size, dtype=np.intp)
    np.cumsum(counts[:-1], out=offsets[1:])
    return offsets


def item_category_table(
    values: NDArray[np.float64],
    category_counts: list[int] | NDArray[np.intp],
) -> NDArray[np.float64]:
    """Stack active categories of ``(n_points, n_items, width)`` values item-major.

    The result has shape ``(sum(category_counts), n_points)`` and owns its
    memory, so callers can clip and take logarithms in place.
    """
    counts = np.asarray(category_counts, dtype=np.intp)
    active = np.arange(values.shape[2]) < counts[:, None]
    return np.ascontiguousarray(values.transpose(1, 2, 0)[active])


def categorical_log_likelihood_batch(
    log_table: NDArray[np.float64],
    offsets: NDArray[np.intp],
    codes: NDArray,
    observed: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Sum the selected category log probabilities of every response pattern.

    Parameters
    ----------
    log_table : ndarray of shape (n_rows, n_points)
        Item-major category log probabilities. Item ``j`` occupies rows
        ``offsets[j]`` through ``offsets[j] + n_categories[j] - 1``.
    offsets : ndarray of shape (n_items,)
        First table row of every item.
    codes : ndarray of shape (n_persons, n_items)
        Integer-valued category codes; cells outside ``observed`` are ignored.
    observed : ndarray of shape (n_persons, n_items), optional
        Cells that contribute to the likelihood. Defaults to ``codes >= 0``.

    Returns
    -------
    ndarray of shape (n_persons, n_points)
        Log-likelihood of each response pattern at each point.

    Notes
    -----
    A sparse one-hot design multiplies the table in bounded row blocks. Each
    row accumulates its observed items in item order with unit weights, so
    the result equals sequential per-item accumulation exactly, and table
    entries that no respondent selected never enter the sums.
    """
    from scipy.sparse import csr_array

    n_persons, n_items = codes.shape
    n_rows, n_points = log_table.shape
    if observed is None:
        observed = codes >= 0
    table = np.ascontiguousarray(log_table, dtype=np.float64)
    item_offsets = np.asarray(offsets, dtype=np.intp)
    result = np.empty((n_persons, n_points), dtype=np.float64)
    rows_per_block = max(1, _LIKELIHOOD_BLOCK_ELEMENTS // max(n_items, n_points, 1))
    for start in range(0, n_persons, rows_per_block):
        stop = min(start + rows_per_block, n_persons)
        block_observed = observed[start:stop]
        columns = codes[start:stop][block_observed].astype(np.intp)
        columns += np.broadcast_to(item_offsets, block_observed.shape)[block_observed]
        row_starts = np.zeros(stop - start + 1, dtype=np.intp)
        np.cumsum(np.count_nonzero(block_observed, axis=1), out=row_starts[1:])
        design = csr_array(
            (np.ones(columns.size), columns, row_starts),
            shape=(stop - start, n_rows),
        )
        result[start:stop] = design @ table
    return result


def item_category_frequencies(
    responses: NDArray,
    valid: NDArray[np.bool_],
    *,
    max_categories: int = 32,
) -> NDArray[np.intp] | None:
    """Count validated nonnegative category codes in bounded row blocks.

    Return None for sparse or empty category ranges so callers can use their
    per-item fallback without allocating a large dense frequency table.
    """
    n_persons, n_items = responses.shape
    n_categories = int(np.max(responses, where=valid, initial=-1)) + 1
    if (
        not 0 < n_categories <= max_categories
        or n_categories * n_items > _MAX_FREQUENCY_ENTRIES
    ):
        return None

    frequencies = np.zeros((n_categories, n_items), dtype=np.intp)
    rows_per_block = max(1, _FREQUENCY_CHUNK_ELEMENTS // max(n_items, 1))
    for start in range(0, n_persons, rows_per_block):
        stop = start + rows_per_block
        block = responses[start:stop]
        block_valid = valid[start:stop]
        for category in range(n_categories):
            frequencies[category] += np.count_nonzero(
                (block == category) & block_valid, axis=0
            )
    return frequencies


def sample_categorical_rows(
    probabilities: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Draw one category from every row of a probability matrix."""
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative[:, -1] = 1.0
    uniforms = rng.random(probabilities.shape[0])
    return np.sum(uniforms[:, None] >= cumulative, axis=1).astype(np.int_)


def sample_categorical_tensor(
    probabilities: NDArray[np.float64],
    n_categories: NDArray[np.intp],
    uniforms: NDArray[np.float64],
) -> NDArray[np.intp]:
    """Draw one category for every cell of a padded probability tensor.

    Parameters
    ----------
    probabilities : ndarray of shape (n_rows, n_items, width)
        Finite, nonnegative category probabilities. Each cell must have a
        positive total; cells are normalized before sampling.
    n_categories : ndarray of shape (n_items,)
        Active categories per item, each between 2 and ``width``.
    uniforms : ndarray of shape (n_rows, n_items)
        Uniform draws on ``[0, 1)``, one per cell.

    Returns
    -------
    ndarray of shape (n_rows, n_items)
        Category codes below each item's category count. Rounding and padded
        mass are absorbed by the last active category, so padded categories
        are never drawn.
    """
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < -PROB_EPSILON):
        raise MirtModelError("Categorical probabilities must be finite and nonnegative")
    probabilities = np.maximum(probabilities, 0.0)
    totals = probabilities.sum(axis=2, keepdims=True)
    if np.any(totals <= PROB_EPSILON):
        raise MirtModelError("Each categorical probability row must have positive mass")
    cumulative = np.cumsum(probabilities / totals, axis=2)
    padded = np.arange(probabilities.shape[2]) >= (n_categories - 1)[:, None]
    np.copyto(cumulative, 1.0, where=padded[None, :, :])
    return np.sum(uniforms[:, :, None] > cumulative, axis=2, dtype=np.intp)


def _response_category_counts(model: BaseItemModel, width: int) -> NDArray[np.intp]:
    """Return validated per-item category counts for a probability tensor."""
    declared = getattr(model, "n_categories", None)
    if declared is None:
        declared = [width] * model.n_items
    elif isinstance(declared, (int, np.integer)):
        declared = [declared] * model.n_items
    if len(declared) != model.n_items:
        raise MirtModelError("Category counts must match the number of items")
    if any(
        isinstance(count, (bool, np.bool_))
        or not isinstance(count, (int, np.integer))
        or not 2 <= count <= width
        for count in declared
    ):
        raise MirtModelError("Category counts must be valid for the probability output")
    return np.asarray(declared, dtype=np.intp)


def draw_item_responses(
    model: BaseItemModel,
    theta: NDArray[np.float64],
    rng: np.random.Generator,
    *,
    chunk_size: int | None = None,
    dtype: type[np.integer] = np.int_,
) -> NDArray[np.integer]:
    """Simulate a response matrix from ``model.probability``.

    Parameters
    ----------
    model : BaseItemModel
        Item model whose ``probability(theta)`` returns binary success
        probabilities of shape ``(n, n_items)`` or padded category
        probabilities of shape ``(n, n_items, width)``.
    theta : ndarray of shape (n_persons, n_factors)
        Latent trait values.
    rng : numpy.random.Generator
        Source of one uniform draw per person and item.
    chunk_size : int, optional
        Positive maximum number of persons evaluated at once. Defaults to all
        persons.
    dtype : numpy integer type, default=numpy.int_
        Integer type of the returned codes.

    Returns
    -------
    ndarray of shape (n_persons, n_items)
        Simulated responses. A binary response is 1 when its uniform draw is
        below the success probability; categories use the inverse CDF.

    Notes
    -----
    Uniforms are drawn person-major in consecutive row blocks, so a seeded
    generator yields identical responses, and ends in the same state, for
    every chunk size.
    """
    n_persons = theta.shape[0]
    n_items = model.n_items
    step = max(1, n_persons if chunk_size is None else int(chunk_size))
    responses = np.empty((n_persons, n_items), dtype=dtype)
    category_counts: dict[int, NDArray[np.intp]] = {}
    for start in range(0, n_persons, step):
        stop = min(start + step, n_persons)
        probabilities = np.asarray(
            model.probability(theta[start:stop]), dtype=np.float64
        )
        if probabilities.ndim == 1:
            probabilities = probabilities.reshape(-1, 1)
        uniforms = rng.random((stop - start, n_items))

        if probabilities.ndim == 2:
            if probabilities.shape != uniforms.shape:
                raise MirtModelError(
                    "Binary probability output has an unexpected shape",
                    model_type=model.model_name,
                    value=probabilities.shape,
                    expected=str(uniforms.shape),
                )
            if not np.all(np.isfinite(probabilities)) or np.any(
                (probabilities < -PROB_EPSILON) | (probabilities > 1 + PROB_EPSILON)
            ):
                raise MirtModelError(
                    "Binary probabilities must be finite and within [0, 1]"
                )
            responses[start:stop] = uniforms < np.clip(probabilities, 0.0, 1.0)
            continue

        if probabilities.ndim != 3 or probabilities.shape[:2] != uniforms.shape:
            raise MirtModelError(
                "Categorical probability output has an unexpected shape",
                model_type=model.model_name,
                value=probabilities.shape,
                expected=f"({stop - start}, {n_items}, n_categories)",
            )
        width = probabilities.shape[2]
        if width not in category_counts:
            category_counts[width] = _response_category_counts(model, width)
        responses[start:stop] = sample_categorical_tensor(
            probabilities, category_counts[width], uniforms
        )
    return responses
