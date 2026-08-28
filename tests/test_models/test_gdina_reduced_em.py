"""Regression tests for complete reduced-family G-DINA estimation."""

import numpy as np
import pytest

from mirt.models.cdm_advanced import GDINA, fit_gdina

_FAMILY_PARAMETERS = [
    ("DINA", np.array([0.1, 0.9])),
    ("DINO", np.array([0.1, 0.9])),
    ("ACDM", np.array([0.1, 0.2, 0.3])),
    ("LLM", np.array([-2.0, 1.2, 0.8])),
    ("RRUM", np.array([0.9, 0.4, 0.7])),
    ("saturated", np.array([0.1, 0.3, 0.5, 0.9])),
]


def _simulate_reduced_family(
    family: str,
    parameters: np.ndarray,
    seed: int,
    n_persons: int = 1500,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    q_matrix = np.vstack(
        (
            np.tile([1, 0], (3, 1)),
            np.tile([0, 1], (3, 1)),
            np.tile([1, 1], (3, 1)),
        )
    )
    reduced_models = ["DINA"] * 6 + [family] * 3
    alpha = rng.integers(0, 2, size=(n_persons, 2))

    true_model = GDINA(
        n_items=9,
        n_attributes=2,
        q_matrix=q_matrix,
        reduced_models=reduced_models,
    )
    for item_idx in range(6):
        true_model.set_delta_parameters(item_idx, np.array([0.1, 0.9]))
    for item_idx in range(6, 9):
        true_model.set_delta_parameters(item_idx, parameters)

    responses = (rng.random((n_persons, 9)) < true_model.probability(alpha)).astype(
        np.int_
    )
    target_probability = true_model.probability(true_model.attribute_patterns, 6)
    return responses, q_matrix, reduced_models, target_probability


@pytest.mark.parametrize(
    ("family", "parameters"),
    _FAMILY_PARAMETERS,
)
def test_fit_gdina_recovers_every_reduced_family(family, parameters):
    """Estimate the item curve instead of retaining initialization values."""
    seed = 30 + [name for name, _ in _FAMILY_PARAMETERS].index(family)
    responses, q_matrix, reduced_models, expected = _simulate_reduced_family(
        family,
        parameters,
        seed,
    )
    initial_model = GDINA(
        n_items=9,
        n_attributes=2,
        q_matrix=q_matrix,
        reduced_models=reduced_models,
    )

    fitted_model, class_probabilities = fit_gdina(
        responses,
        q_matrix,
        reduced_models=reduced_models,
        max_iter=100,
        tol=1e-5,
    )

    fitted_target = np.mean(
        np.column_stack(
            [
                fitted_model.probability(fitted_model.attribute_patterns, item_idx)
                for item_idx in range(6, 9)
            ]
        ),
        axis=1,
    )
    np.testing.assert_allclose(fitted_target, expected, atol=0.05)
    assert not np.array_equal(
        fitted_model.delta_parameters[6],
        initial_model.delta_parameters[6],
    )
    assert fitted_model.is_fitted
    assert np.all(class_probabilities > 0.0)
    np.testing.assert_allclose(np.sum(class_probabilities), 1.0)


def test_fit_gdina_handles_scattered_and_person_level_missingness():
    """Ignore negative responses without dropping partially observed people."""
    responses, q_matrix, reduced_models, expected = _simulate_reduced_family(
        "LLM",
        np.array([-2.0, 1.2, 0.8]),
        seed=47,
    )
    rng = np.random.default_rng(48)
    missing = rng.random(responses.shape) < 0.15
    responses = responses.copy()
    responses[missing] = -1
    responses[0, :] = -1

    fitted_model, class_probabilities = fit_gdina(
        responses,
        q_matrix,
        reduced_models=reduced_models,
        max_iter=100,
        tol=1e-5,
    )

    fitted_target = np.mean(
        np.column_stack(
            [
                fitted_model.probability(fitted_model.attribute_patterns, item_idx)
                for item_idx in range(6, 9)
            ]
        ),
        axis=1,
    )
    np.testing.assert_allclose(fitted_target, expected, atol=0.07)
    assert np.all(np.isfinite(class_probabilities))
    np.testing.assert_allclose(np.sum(class_probabilities), 1.0)


def test_fit_gdina_synchronizes_generic_parameter_cache():
    """Expose the final family-specific parameter lengths and values."""
    responses, q_matrix, reduced_models, _ = _simulate_reduced_family(
        "RRUM",
        np.array([0.9, 0.4, 0.7]),
        seed=52,
        n_persons=700,
    )

    fitted_model, _ = fit_gdina(
        responses,
        q_matrix,
        reduced_models=reduced_models,
        max_iter=50,
    )

    cache = fitted_model.parameters
    for item_idx, parameters in enumerate(fitted_model.delta_parameters):
        n_parameters = len(parameters)
        assert cache["delta_n_params"][item_idx] == n_parameters
        np.testing.assert_array_equal(
            cache["delta"][item_idx, :n_parameters],
            parameters,
        )


@pytest.mark.parametrize(
    ("max_iter", "tol", "message"),
    [
        (0, 1e-4, "max_iter"),
        (-1, 1e-4, "max_iter"),
        (2.5, 1e-4, "max_iter"),
        (True, 1e-4, "max_iter"),
        (10, 0.0, "tol"),
        (10, -1.0, "tol"),
        (10, np.nan, "tol"),
        (10, True, "tol"),
    ],
)
def test_fit_gdina_rejects_invalid_controls(max_iter, tol, message):
    """Require a positive iteration count and finite positive tolerance."""
    with pytest.raises(ValueError, match=message):
        fit_gdina(
            np.array([[1, 0], [0, 1]]),
            np.array([[1, 0], [0, 1]]),
            max_iter=max_iter,
            tol=tol,
        )


@pytest.mark.parametrize(
    ("responses", "q_matrix"),
    [
        (np.array([1, 0]), np.array([[1, 0], [0, 1]])),
        (np.empty((0, 2), dtype=np.int_), np.array([[1, 0], [0, 1]])),
        (np.empty((2, 0), dtype=np.int_), np.empty((0, 1), dtype=np.int_)),
        (np.array([[1, 0], [0, 1]]), np.array([1, 0])),
        (np.array([[1, 2], [0, 1]]), np.array([[1, 0], [0, 1]])),
        (np.array([[1.0, np.nan]]), np.array([[1, 0], [0, 1]])),
        (np.array([[1, -1], [0, -1]]), np.array([[1, 0], [0, 1]])),
    ],
)
def test_fit_gdina_rejects_invalid_data(responses, q_matrix):
    """Validate dimensions, response codes, finiteness, and item coverage."""
    with pytest.raises(ValueError):
        fit_gdina(responses, q_matrix)
