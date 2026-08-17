"""Correctness and public API contracts for cognitive diagnosis models."""

from collections.abc import Callable

import numpy as np
import pytest

from mirt.models.cdm import DINA, DINO, fit_cdm

CDMFactory = Callable[..., DINA | DINO]


@pytest.fixture
def q_matrix() -> np.ndarray:
    return np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ]
    )


@pytest.mark.parametrize("factory", [DINA, DINO])
def test_structural_arrays_are_defensively_owned(
    factory: CDMFactory,
    q_matrix: np.ndarray,
) -> None:
    source = q_matrix.copy()
    model = factory(n_items=6, n_attributes=3, q_matrix=source)
    source[:] = 0
    exposed_q = model.q_matrix
    exposed_patterns = model.attribute_patterns
    exposed_q[:] = 0
    exposed_patterns[:] = 0

    np.testing.assert_array_equal(model.q_matrix, q_matrix)
    assert np.count_nonzero(model.attribute_patterns) > 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_attributes": 0, "q_matrix": np.empty((2, 0))}, "positive integer"),
        ({"n_attributes": True, "q_matrix": np.ones((2, 1))}, "positive integer"),
        ({"n_attributes": 2, "q_matrix": [[1, 0], [0, 0.5]]}, "only 0 and 1"),
        ({"n_attributes": 2, "q_matrix": [[1, 0], [0, np.nan]]}, "only 0 and 1"),
        (
            {"n_attributes": 2, "q_matrix": [["yes", "no"], ["no", "yes"]]},
            "numeric",
        ),
    ],
)
def test_constructor_validates_q_matrix(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DINA(n_items=2, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("factory", [DINA, DINO])
@pytest.mark.parametrize(
    ("alpha", "message"),
    [
        ([[[0, 1, 0]]], "one- or two-dimensional"),
        ([[0, 1]], "attributes, expected"),
        ([[0, 0.5, 1]], "only 0 and 1"),
        ([[0, np.nan, 1]], "only 0 and 1"),
        ([["no", "yes", "no"]], "numeric"),
    ],
)
def test_probability_validates_mastery_patterns(
    factory: CDMFactory,
    q_matrix: np.ndarray,
    alpha: object,
    message: str,
) -> None:
    model = factory(n_items=6, n_attributes=3, q_matrix=q_matrix)

    with pytest.raises(ValueError, match=message):
        model.probability(alpha)  # type: ignore[arg-type]


@pytest.mark.parametrize("item_idx", [-1, 6, True, 1.5])
def test_probability_validates_item_index(
    q_matrix: np.ndarray,
    item_idx: object,
) -> None:
    model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)

    with pytest.raises(IndexError, match="item_idx"):
        model.probability([[0, 0, 0]], item_idx=item_idx)  # type: ignore[arg-type]


@pytest.mark.parametrize("factory", [DINA, DINO])
def test_bulk_pattern_likelihood_matches_direct_evaluation(
    factory: CDMFactory,
    q_matrix: np.ndarray,
) -> None:
    model = factory(n_items=6, n_attributes=3, q_matrix=q_matrix)
    model.set_parameters(
        slip=np.linspace(0.05, 0.2, 6),
        guess=np.linspace(0.1, 0.25, 6),
    )
    responses = np.array(
        [
            [1, 0, 1, 0, 1, 0],
            [0, 1, -1, 1, 0, 1],
            [1, 1, 1, -9, 0, 0],
        ]
    )

    actual = model.pattern_log_likelihoods(responses)
    expected = np.column_stack(
        [
            model.log_likelihood(responses, pattern)
            for pattern in model.attribute_patterns
        ]
    )

    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("factory", [DINA, DINO])
def test_posteriors_and_attribute_marginals_are_consistent(
    factory: CDMFactory,
    q_matrix: np.ndarray,
) -> None:
    model = factory(n_items=6, n_attributes=3, q_matrix=q_matrix)
    responses = np.array([[1, 0, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0]])

    posterior = model.attribute_posteriors(responses)
    marginals = model.attribute_marginals(responses)

    assert posterior.shape == (2, 8)
    assert marginals.shape == (2, 3)
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0)
    np.testing.assert_allclose(marginals, posterior @ model.attribute_patterns)
    assert np.all((marginals >= 0.0) & (marginals <= 1.0))


def test_map_classification_uses_pattern_prior(q_matrix: np.ndarray) -> None:
    model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)
    responses = np.full((1, 6), -1)
    prior = np.zeros(8)
    prior[-1] = 3.0

    posterior = model.attribute_posteriors(responses, prior)
    mle = model.classify_respondents(responses, method="MLE")
    mapped = model.classify_respondents(
        responses,
        method="map",  # type: ignore[arg-type]
        pattern_prior=prior,
    )

    np.testing.assert_array_equal(posterior[0], prior / prior.sum())
    np.testing.assert_array_equal(mle[0], np.array([0, 0, 0]))
    np.testing.assert_array_equal(mapped[0], np.array([1, 1, 1]))


@pytest.mark.parametrize(
    ("prior", "message"),
    [
        ([1.0, 1.0], "shape"),
        ([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0], "non-negative"),
        ([0.0] * 8, "positive sum"),
        ([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan], "finite"),
    ],
)
def test_attribute_posteriors_validate_prior(
    q_matrix: np.ndarray,
    prior: object,
    message: str,
) -> None:
    model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)

    with pytest.raises(ValueError, match=message):
        model.attribute_posteriors([[0, 0, 0, 0, 0, 0]], prior)  # type: ignore[arg-type]


def test_classification_validates_method_and_prior_use(q_matrix: np.ndarray) -> None:
    model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)
    responses = np.zeros((1, 6), dtype=int)

    with pytest.raises(ValueError, match="method"):
        model.classify_respondents(responses, method="other")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="only be used"):
        model.classify_respondents(responses, method="MLE", pattern_prior=np.ones(8))


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        ([0, 1], "2D"),
        ([[0, 1]], "items, expected"),
        ([[0, 1, 0, 1, 0, 2]], "only 0, 1"),
        ([[0, 1, 0, 1, 0, np.nan]], "finite"),
        ([["yes"] * 6], "numeric"),
    ],
)
def test_pattern_likelihood_validates_responses(
    q_matrix: np.ndarray,
    responses: object,
    message: str,
) -> None:
    model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)

    with pytest.raises(ValueError, match=message):
        model.pattern_log_likelihoods(responses)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"slip": np.full(6, -0.1)}, "probabilities"),
        ({"guess": np.full(6, np.nan)}, "finite"),
        ({"slip": np.full(6, 0.6), "guess": np.full(6, 0.4)}, "less than 1"),
    ],
)
def test_noisy_gate_parameters_are_identifiable(
    q_matrix: np.ndarray,
    params: dict[str, np.ndarray],
    message: str,
) -> None:
    model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)

    with pytest.raises(ValueError, match=message):
        model.set_parameters(**params)


def test_dino_items_without_required_attributes_are_always_ideal() -> None:
    model = DINO(n_items=1, n_attributes=2, q_matrix=np.zeros((1, 2), dtype=int))

    np.testing.assert_array_equal(model.eta([[0, 0], [1, 0], [1, 1]], 0), 1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"responses": [0, 1], "q_matrix": [[1]]}, "2D"),
        ({"responses": np.empty((0, 1)), "q_matrix": [[1]]}, "at least one"),
        ({"responses": [[0]], "q_matrix": []}, "Q-matrix"),
        ({"responses": [[0]], "q_matrix": [[1]], "max_iter": 0}, "positive integer"),
        ({"responses": [[0]], "q_matrix": [[1]], "tol": 0.0}, "positive"),
        ({"responses": [[0]], "q_matrix": [[1]], "verbose": 1}, "boolean"),
    ],
)
def test_fit_cdm_validates_inputs(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fit_cdm(**kwargs)  # type: ignore[arg-type]


def test_fit_cdm_normalizes_model_name_and_missing_codes(
    q_matrix: np.ndarray,
) -> None:
    rng = np.random.default_rng(42)
    responses = rng.integers(0, 2, size=(80, 6))
    responses[::5, 0] = -9
    normalized = responses.copy()
    normalized[normalized < 0] = -1

    actual_model, actual_classes = fit_cdm(
        responses,
        q_matrix,
        model=" dina ",
        max_iter=10,
    )
    expected_model, expected_classes = fit_cdm(
        normalized,
        q_matrix,
        model="DINA",
        max_iter=10,
    )

    np.testing.assert_allclose(actual_model.slip, expected_model.slip)
    np.testing.assert_allclose(actual_model.guess, expected_model.guess)
    np.testing.assert_allclose(actual_classes, expected_classes)


@pytest.mark.parametrize("model_name", ["DINA", "DINO"])
def test_fit_cdm_preserves_monotonic_item_probabilities(
    q_matrix: np.ndarray,
    model_name: str,
) -> None:
    responses = np.random.default_rng(7).integers(0, 2, size=(100, 6))

    model, _ = fit_cdm(
        responses,
        q_matrix,
        model=model_name,
        max_iter=20,
    )

    assert np.all(1.0 - model.slip > model.guess)  # type: ignore[attr-defined]
