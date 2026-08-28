"""Regression tests for fitted G-DINA reduced-model selection."""

import numpy as np
import pytest

from mirt._core import sigmoid
from mirt.models.cdm_advanced import GDINA


def _linear_logistic_case(
    n_persons: int = 1500,
) -> tuple[GDINA, np.ndarray]:
    rng = np.random.default_rng(19)
    q_matrix = np.vstack(
        (
            np.tile([1, 1], (4, 1)),
            np.tile([1, 0], (6, 1)),
            np.tile([0, 1], (6, 1)),
        )
    )
    n_items = q_matrix.shape[0]
    alpha = rng.integers(0, 2, size=(n_persons, 2))

    probabilities = np.empty((n_persons, n_items), dtype=np.float64)
    probabilities[:, :4] = sigmoid(
        -2.0 + 1.2 * alpha[:, 0, None] + 1.2 * alpha[:, 1, None]
    )
    probabilities[:, 4:10] = 0.05 + 0.90 * alpha[:, 0, None]
    probabilities[:, 10:] = 0.05 + 0.90 * alpha[:, 1, None]
    responses = (rng.random((n_persons, n_items)) < probabilities).astype(np.int_)

    model = GDINA(n_items=n_items, n_attributes=2, q_matrix=q_matrix)
    patterns = model.attribute_patterns
    joint_probability = np.asarray(
        sigmoid(-2.0 + 1.2 * patterns[:, 0] + 1.2 * patterns[:, 1]),
        dtype=np.float64,
    )
    for item_idx in range(4):
        model.set_delta_parameters(item_idx, joint_probability)
    for item_idx in range(4, n_items):
        model.set_delta_parameters(item_idx, np.array([0.05, 0.95]))

    return model, responses


def test_model_selection_fits_candidates_before_comparison():
    """Recover LLM items whose fitted curve differs from fixed defaults."""
    model, responses = _linear_logistic_case()

    selected = model.model_selection(
        responses,
        candidate_models=["DINA", "LLM", "saturated"],
    )

    assert selected[:4] == ["LLM"] * 4


def test_model_selection_supports_missing_responses():
    """Use observed sample sizes and retain selection with scattered missingness."""
    model, responses = _linear_logistic_case()
    responses = responses.copy()
    responses[::11, :] = -1
    responses[1::13, 0] = -1

    selected = model.model_selection(
        responses,
        candidate_models=["DINA", "LLM", "saturated"],
    )

    assert selected[:4] == ["LLM"] * 4


def test_model_selection_preserves_all_model_state():
    """Keep item models, deltas, and the generic parameter cache unchanged."""
    model, responses = _linear_logistic_case(n_persons=500)
    reduced_models_before = model.reduced_models
    deltas_before = model.delta_parameters
    parameters_before = model.parameters

    model.model_selection(responses, candidate_models=["saturated", "DINA"])

    assert model.reduced_models == reduced_models_before
    for actual, expected in zip(model.delta_parameters, deltas_before, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert model.parameters.keys() == parameters_before.keys()
    for name, expected in parameters_before.items():
        np.testing.assert_array_equal(model.parameters[name], expected)


def test_model_selection_excludes_target_response_from_class_weights(monkeypatch):
    """Use uniform class weights when no other items provide class evidence."""
    model = GDINA(
        n_items=1,
        n_attributes=2,
        q_matrix=np.ones((1, 2), dtype=np.int_),
    )
    model.set_delta_parameters(0, np.array([0.01, 0.20, 0.80, 0.99]))
    responses = np.array([[1], [1], [0], [1], [0], [0]])
    captured = {}

    def record_statistics(model_type, group_patterns, successes, totals):
        captured["successes"] = successes.copy()
        captured["totals"] = totals.copy()
        return 0.0, group_patterns.shape[0]

    monkeypatch.setattr(model, "_fit_selection_candidate", record_statistics)

    model.model_selection(responses, candidate_models=["saturated"])

    np.testing.assert_allclose(captured["totals"], np.full(4, 1.5))
    np.testing.assert_allclose(captured["successes"], np.full(4, 0.75))


@pytest.mark.parametrize(
    ("expected", "probabilities"),
    [
        ("DINA", np.array([0.1, 0.1, 0.1, 0.9])),
        ("DINO", np.array([0.1, 0.9, 0.9, 0.9])),
        ("ACDM", np.array([0.1, 0.3, 0.4, 0.6])),
        (
            "LLM",
            np.asarray(sigmoid(np.array([-2.0, -0.8, -1.2, 0.0]))),
        ),
        ("RRUM", np.array([0.252, 0.63, 0.36, 0.9])),
    ],
)
def test_candidate_fits_recover_each_reduced_family(expected, probabilities):
    """Fit every supported reduced family to exact weighted sufficient statistics."""
    model = GDINA(n_items=1, n_attributes=2, q_matrix=np.ones((1, 2), dtype=np.int_))
    group_patterns = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    totals = np.full(4, 10_000.0)
    successes = totals * probabilities
    candidates = ["DINA", "DINO", "ACDM", "LLM", "RRUM", "saturated"]

    bic = {}
    for model_type in candidates:
        log_likelihood, n_parameters = model._fit_selection_candidate(
            model_type,
            group_patterns,
            successes,
            totals,
        )
        bic[model_type] = -2.0 * log_likelihood + n_parameters * np.log(np.sum(totals))

    assert min(bic, key=bic.get) == expected


@pytest.mark.parametrize(
    "candidate_models",
    [[], ["DINA", "DINA"], ["unknown"]],
)
def test_model_selection_rejects_invalid_candidate_lists(candidate_models):
    """Reject empty, duplicate, and unknown candidate lists."""
    model = GDINA(
        n_items=2,
        n_attributes=2,
        q_matrix=np.array([[1, 0], [0, 1]]),
    )

    with pytest.raises(ValueError):
        model.model_selection(np.array([[1, 0], [0, 1]]), candidate_models)


@pytest.mark.parametrize(
    "responses",
    [
        np.array([1, 0]),
        np.empty((0, 2), dtype=np.int_),
        np.array([[1, 0, 1]]),
        np.array([[1, 2]]),
        np.array([[1.0, np.nan]]),
        np.array([[1, -1], [0, -1]]),
    ],
)
def test_model_selection_rejects_invalid_responses(responses):
    """Require a non-empty binary-or-missing matrix with data for every item."""
    model = GDINA(
        n_items=2,
        n_attributes=2,
        q_matrix=np.array([[1, 0], [0, 1]]),
    )

    with pytest.raises(ValueError):
        model.model_selection(responses)


def test_constructor_rejects_unknown_reduced_model():
    """Fail clearly before an unknown model reaches the parameter cache."""
    with pytest.raises(ValueError, match="Unknown reduced model"):
        GDINA(
            n_items=1,
            n_attributes=1,
            q_matrix=np.ones((1, 1), dtype=np.int_),
            reduced_models=["unknown"],
        )
