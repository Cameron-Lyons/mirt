"""Coverage for all-item callbacks in custom item models."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from mirt.exceptions import MirtValidationError
from mirt.models.custom import CustomItemModel, ItemTypeSpec, create_item_type


def _logistic(theta: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-a * (theta - b)))


def _batch_logistic(
    theta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-a[None, :] * (theta[:, None] - b[None, :])))


def _logistic_spec(
    *,
    batch_icc_function: Callable[..., np.ndarray] | None = None,
    info_function: Callable[..., np.ndarray] | None = None,
    batch_info_function: Callable[..., np.ndarray] | None = None,
) -> ItemTypeSpec:
    return create_item_type(
        "Batch logistic",
        _logistic,
        info_function=info_function,
        par_names=["a", "b"],
        par_defaults={"a": 1.0, "b": 0.0},
        batch_icc_function=batch_icc_function,
        batch_info_function=batch_info_function,
    )


def test_batch_probability_matches_fallback_and_uses_one_callback() -> None:
    calls = {"item": 0, "batch": 0}

    def item(theta: np.ndarray, a: float, b: float) -> np.ndarray:
        calls["item"] += 1
        return _logistic(theta, a, b)

    def batch(
        theta: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        calls["batch"] += 1
        result = _batch_logistic(theta, a, b)
        a[:] = -99.0
        b[:] = -99.0
        return result

    spec = create_item_type(
        "Counted",
        item,
        par_names=["a", "b"],
        par_defaults={"a": 1.0, "b": 0.0},
        batch_icc_function=batch,
    )
    model = CustomItemModel(4, spec).set_parameters(
        a=[0.7, 1.0, 1.3, 1.6],
        b=[-0.5, 0.0, 0.5, 1.0],
    )
    fallback = CustomItemModel(4, _logistic_spec()).set_parameters(**model.parameters)
    theta = np.linspace(-2.0, 2.0, 9)

    actual = model.probability(theta)
    expected = fallback.probability(theta)

    np.testing.assert_allclose(actual, expected)
    assert calls == {"item": 0, "batch": 1}
    np.testing.assert_array_equal(model.parameters["a"], [0.7, 1.0, 1.3, 1.6])
    np.testing.assert_array_equal(model.parameters["b"], [-0.5, 0.0, 0.5, 1.0])

    np.testing.assert_allclose(model.probability(theta, 2), expected[:, 2])
    assert calls == {"item": 1, "batch": 1}

    calls["batch"] = 0
    responses = np.tile([0, 1, -1, 1], (theta.size, 1))
    np.testing.assert_allclose(
        model.log_likelihood(responses, theta),
        fallback.log_likelihood(responses, theta),
    )
    assert calls["batch"] == 1


def test_polytomous_batch_callback_matches_item_fallbacks() -> None:
    def ordinal(theta: np.ndarray, shift: float) -> np.ndarray:
        eta = theta - shift
        weights = np.column_stack((np.ones_like(eta), np.exp(eta), np.exp(2.0 * eta)))
        return weights / weights.sum(axis=1, keepdims=True)

    def batch_ordinal(theta: np.ndarray, shift: np.ndarray) -> np.ndarray:
        eta = theta[:, None] - shift[None, :]
        weights = np.stack(
            (np.ones_like(eta), np.exp(eta), np.exp(2.0 * eta)),
            axis=2,
        )
        return weights / weights.sum(axis=2, keepdims=True)

    shared = {
        "par_names": ["shift"],
        "par_defaults": {"shift": 0.0},
        "n_categories": 3,
    }
    model = CustomItemModel(
        3,
        create_item_type(
            "Batched ordinal",
            ordinal,
            batch_icc_function=batch_ordinal,
            **shared,
        ),
    ).set_parameters(shift=[-0.5, 0.0, 0.75])
    fallback = CustomItemModel(
        3,
        create_item_type("Ordinal", ordinal, **shared),
    ).set_parameters(**model.parameters)
    theta = np.linspace(-1.5, 1.5, 7)
    responses = np.array([[0, 1, 2], [2, -1, 0], [1, 2, 1]])

    np.testing.assert_allclose(model.probability(theta), fallback.probability(theta))
    np.testing.assert_allclose(
        model.expected_score(theta), fallback.expected_score(theta)
    )
    np.testing.assert_allclose(
        model.log_likelihood_batch(responses, theta),
        fallback.log_likelihood_batch(responses, theta),
    )


def test_batch_probability_accelerates_numerical_information() -> None:
    calls = {"count": 0}

    def batch(
        theta: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        calls["count"] += 1
        return _batch_logistic(theta, a, b)

    model = CustomItemModel(5, _logistic_spec(batch_icc_function=batch)).set_parameters(
        a=np.linspace(0.7, 1.5, 5),
        b=np.linspace(-0.8, 0.8, 5),
    )
    fallback = CustomItemModel(5, _logistic_spec()).set_parameters(**model.parameters)
    theta = np.linspace(-2.0, 2.0, 11)

    actual = model.information(theta)
    expected = fallback.information(theta)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    assert calls["count"] == 3


def test_batch_information_callback_is_used_for_all_items() -> None:
    calls = {"item": 0, "batch": 0}

    def item_info(theta: np.ndarray, a: float, b: float) -> np.ndarray:
        calls["item"] += 1
        return np.full_like(theta, a + abs(b))

    def batch_info(
        theta: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        calls["batch"] += 1
        return np.broadcast_to(a[None, :] + np.abs(b[None, :]), (theta.size, a.size))

    model = CustomItemModel(
        3,
        _logistic_spec(
            info_function=item_info,
            batch_info_function=batch_info,
        ),
    ).set_parameters(a=[1.0, 1.5, 2.0], b=[-0.5, 0.0, 0.5])
    theta = np.array([-1.0, 0.0, 1.0])

    np.testing.assert_allclose(
        model.information(theta),
        [[1.5, 1.5, 2.5], [1.5, 1.5, 2.5], [1.5, 1.5, 2.5]],
    )
    assert calls == {"item": 0, "batch": 1}
    np.testing.assert_array_equal(model.information(theta, 1), [1.5, 1.5, 1.5])
    assert calls == {"item": 1, "batch": 1}


@pytest.mark.parametrize(
    "spec",
    [
        ItemTypeSpec(
            "Bad probabilities",
            _logistic,
            par_names=["a", "b"],
            batch_icc_function=lambda theta, a, b: np.zeros((theta.size, a.size + 1)),
        ),
        ItemTypeSpec(
            "Bad information",
            _logistic,
            par_names=["a", "b"],
            batch_info_function=lambda theta, a, b: -np.ones((theta.size, a.size)),
        ),
    ],
)
def test_batch_callback_outputs_are_validated(spec: ItemTypeSpec) -> None:
    model = CustomItemModel(2, spec)
    with pytest.raises(MirtValidationError):
        if spec.batch_icc_function is not None:
            model.probability(np.array([0.0, 1.0]))
        else:
            model.information(np.array([0.0, 1.0]))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ItemTypeSpec(
            "Bad batch probability",
            _logistic,
            batch_icc_function=1,  # type: ignore[arg-type]
        ),
        lambda: ItemTypeSpec(
            "Bad batch information",
            _logistic,
            batch_info_function=1,  # type: ignore[arg-type]
        ),
    ],
)
def test_batch_callbacks_must_be_callable(factory: Callable[[], object]) -> None:
    with pytest.raises(MirtValidationError, match="must be callable"):
        factory()
