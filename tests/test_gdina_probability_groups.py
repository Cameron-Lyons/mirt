"""Regression coverage for grouped G-DINA probability evaluation."""

from collections.abc import Iterator

import numpy as np
import pytest
from numpy.typing import NDArray

import mirt.models.cdm_advanced as cdm_advanced
from mirt.models.cdm_advanced import GDINA, ReducedModelType

FAMILIES: tuple[ReducedModelType, ...] = (
    "DINA",
    "DINO",
    "ACDM",
    "LLM",
    "RRUM",
    "saturated",
)


@pytest.mark.parametrize("family", FAMILIES)
def test_grouped_family_probabilities_match_itemwise(
    family: ReducedModelType,
) -> None:
    model = _model_with_families([family] * 24)
    alpha = np.random.default_rng(42).integers(
        0,
        2,
        size=(257, model.n_attributes),
        dtype=np.int_,
    )

    actual = model.probability(alpha)
    expected = _itemwise_probabilities(model, alpha)

    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    assert np.all(np.isfinite(actual))
    assert np.all((actual >= 0.0) & (actual <= 1.0))


def test_grouped_mixed_probabilities_are_chunk_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    families = [FAMILIES[item_idx % len(FAMILIES)] for item_idx in range(48)]
    model = _model_with_families(families)
    alpha = np.random.default_rng(43).integers(
        0,
        2,
        size=(129, model.n_attributes),
        dtype=np.int_,
    )
    expected = model.probability(alpha)

    monkeypatch.setattr(
        cdm_advanced,
        "_MAX_GDINA_PROBABILITY_CHUNK_ENTRIES",
        500,
    )
    original_chunks = cdm_advanced._gdina_item_chunks
    chunk_sizes: list[int] = []

    def record_chunks(
        item_indices: NDArray[np.intp],
        entries_per_item: int,
    ) -> Iterator[NDArray[np.intp]]:
        for chunk in original_chunks(item_indices, entries_per_item):
            chunk_sizes.append(chunk.size)
            yield chunk

    monkeypatch.setattr(cdm_advanced, "_gdina_item_chunks", record_chunks)

    def reject_itemwise_evaluation(*args: object, **kwargs: object) -> None:
        raise AssertionError("bounded family groups should remain vectorized")

    monkeypatch.setattr(
        model, "_item_probability_from_alpha", reject_itemwise_evaluation
    )
    actual = model.probability(alpha)

    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    assert chunk_sizes
    assert max(chunk_sizes) <= 3


def test_llm_probability_handles_extreme_logits_without_overflow() -> None:
    model = GDINA(
        n_items=8,
        n_attributes=4,
        q_matrix=np.ones((8, 4), dtype=np.int_),
        reduced_models=["LLM"] * 8,
    )
    for item_idx in range(model.n_items):
        direction = -1.0 if item_idx % 2 else 1.0
        model.set_delta_parameters(
            item_idx,
            np.array([direction * 1_000.0, *([direction * 1_000.0] * 4)]),
        )
    alpha = model.attribute_patterns

    with np.errstate(over="raise", invalid="raise"):
        actual = model.probability(alpha)
        expected = _itemwise_probabilities(model, alpha)

    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(actual, expected)
    assert np.any(actual == 0.0)
    assert np.any(actual == 1.0)


@pytest.mark.parametrize("item_idx", [-1, 8, True, 1.5])
def test_gdina_probability_rejects_invalid_item_indices(item_idx: object) -> None:
    model = _model_with_families(list(FAMILIES) + ["DINA", "DINO"])

    with pytest.raises(IndexError, match="item_idx"):
        model.probability(np.zeros(model.n_attributes, dtype=np.int_), item_idx)


def test_single_item_all_probability_preserves_item_axis() -> None:
    model = _model_with_families(["saturated"])
    alpha = model.attribute_patterns

    actual = model.probability(alpha)

    assert actual.shape == (alpha.shape[0], 1)
    np.testing.assert_array_equal(actual[:, 0], model.probability(alpha, 0))


def _model_with_families(families: list[ReducedModelType]) -> GDINA:
    n_attributes = 6
    item_numbers = np.arange(len(families), dtype=np.uint64)[:, None]
    bit_numbers = np.arange(n_attributes, dtype=np.uint64)[None, :]
    q_matrix = ((item_numbers >> bit_numbers) & 1).astype(np.int_)
    model = GDINA(
        n_items=len(families),
        n_attributes=n_attributes,
        q_matrix=q_matrix,
        reduced_models=families,
    )
    for item_idx, family in enumerate(families):
        n_required = int(np.sum(q_matrix[item_idx]))
        if family in ("DINA", "DINO"):
            delta = np.array([0.05 + 0.001 * item_idx, 0.85 + 0.001 * item_idx])
        elif family == "ACDM":
            delta = np.array([0.05, *np.linspace(0.04, 0.12, n_required)])
        elif family == "LLM":
            delta = np.array([-2.5, *np.linspace(0.3, 1.2, n_required)])
        elif family == "RRUM":
            delta = np.array([0.9, *np.linspace(0.4, 0.9, n_required)])
        else:
            delta = np.linspace(0.05, 0.95, 2**n_required)
        model.set_delta_parameters(item_idx, delta)
    return model


def _itemwise_probabilities(
    model: GDINA,
    alpha: NDArray[np.int_],
) -> NDArray[np.float64]:
    return np.column_stack(
        [model.probability(alpha, item_idx) for item_idx in range(model.n_items)]
    )
