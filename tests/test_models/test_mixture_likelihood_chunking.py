"""Memory-bounding contracts for finite-mixture likelihood evaluation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mirt.models.mixture as mixture_module
from mirt import MixtureIRT


def _model() -> MixtureIRT:
    model = MixtureIRT(n_items=5, n_classes=3, base_model="2PL")
    model.set_parameters(
        class_proportions=np.array([0.2, 0.3, 0.5]),
        discrimination_class0=np.array([0.7, 0.8, 0.9, 1.0, 1.1]),
        difficulty_class0=np.array([-1.4, -1.0, -0.6, -0.2, 0.2]),
        discrimination_class1=np.array([1.0, 1.1, 1.2, 1.3, 1.4]),
        difficulty_class1=np.array([-0.7, -0.3, 0.1, 0.5, 0.9]),
        discrimination_class2=np.array([1.5, 1.4, 1.3, 1.2, 1.1]),
        difficulty_class2=np.array([0.0, 0.4, 0.8, 1.2, 1.6]),
    )
    return model


def _responses() -> np.ndarray:
    return np.array(
        [
            [1, 1, 0, 1, 0],
            [0, -1, 1, 0, 1],
            [1, 0, 1, -1, 1],
            [0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1],
            [-1, -1, -1, -1, -1],
            [0, 1, 0, 1, 0],
        ]
    )


def test_paired_likelihood_outputs_are_chunk_invariant() -> None:
    model = _model()
    responses = _responses()
    theta = np.linspace(-1.5, 1.5, len(responses))

    expected_likelihood = model.log_likelihood(
        responses,
        theta,
        chunk_size=len(responses),
    )
    expected_posterior = model.class_posterior(
        responses,
        theta,
        chunk_size=len(responses),
    )

    assert_allclose(
        model.log_likelihood(responses, theta, chunk_size=1),
        expected_likelihood,
        rtol=1e-14,
        atol=1e-14,
    )
    assert_allclose(
        model.class_posterior(responses, theta, chunk_size=1),
        expected_posterior,
        rtol=1e-14,
        atol=1e-14,
    )
    assert_array_equal(
        model.classify_persons(responses, theta, chunk_size=1),
        np.argmax(expected_posterior, axis=1),
    )


@pytest.mark.parametrize("base_model", ["1PL", "2PL", "3PL"])
def test_chunking_supports_every_component_family(base_model) -> None:
    model = MixtureIRT(n_items=3, n_classes=2, base_model=base_model)
    responses = np.array([[1, 0, 1], [0, -1, 1], [1, 1, 0]])
    theta = np.array([-0.8, 0.1, 1.2])

    assert_allclose(
        model.log_likelihood(responses, theta, chunk_size=1),
        model.log_likelihood(responses, theta, chunk_size=len(responses)),
        rtol=1e-14,
        atol=1e-14,
    )
    assert_allclose(
        model.log_likelihood_batch(
            responses,
            theta,
            pattern_chunk_size=1,
            theta_chunk_size=1,
        ),
        model.log_likelihood_batch(
            responses,
            theta,
            pattern_chunk_size=len(responses),
            theta_chunk_size=len(theta),
        ),
        rtol=1e-14,
        atol=1e-14,
    )


def test_paired_broadcasting_is_chunk_invariant() -> None:
    model = _model()
    responses = _responses()
    theta = np.linspace(-2.0, 2.0, len(responses))

    for response_values, theta_values in (
        (responses, np.array([0.25])),
        (responses[:1], theta),
    ):
        expected = model.class_posterior(
            response_values,
            theta_values,
            chunk_size=len(responses),
        )
        actual = model.class_posterior(
            response_values,
            theta_values,
            chunk_size=2,
        )
        assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_automatic_paired_chunks_bound_class_curve_rows(monkeypatch) -> None:
    model = _model()
    responses = np.tile(_responses(), (3, 1))
    theta = np.linspace(-2.0, 2.0, len(responses))
    expected = model.class_posterior(
        responses,
        theta,
        chunk_size=len(responses),
    )
    working_values_per_person = 2 * model.n_classes * model.n_items + 2 * model.n_items
    monkeypatch.setattr(
        mixture_module,
        "_MIXTURE_MAX_LIKELIHOOD_VALUES",
        2 * working_values_per_person,
    )
    original = model._paired_class_log_joint_chunk
    chunk_rows: list[int] = []

    def record_chunk(response_chunk, theta_chunk):
        chunk_rows.append(len(response_chunk))
        return original(response_chunk, theta_chunk)

    monkeypatch.setattr(model, "_paired_class_log_joint_chunk", record_chunk)
    actual = model.class_posterior(responses, theta)

    assert chunk_rows
    assert max(chunk_rows) <= 2
    assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_grid_likelihood_is_chunk_invariant() -> None:
    model = _model()
    responses = _responses()
    theta = np.linspace(-2.5, 2.5, 9)
    expected = model.log_likelihood_batch(
        responses,
        theta,
        pattern_chunk_size=len(responses),
        theta_chunk_size=len(theta),
    )

    actual = model.log_likelihood_batch(
        responses,
        theta,
        pattern_chunk_size=1,
        theta_chunk_size=2,
    )

    assert actual.shape == (len(responses), len(theta))
    assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_automatic_grid_chunks_bound_both_dimensions(monkeypatch) -> None:
    model = _model()
    responses = np.tile(_responses(), (2, 1))
    theta = np.linspace(-3.0, 3.0, 11)
    expected = model.log_likelihood_batch(
        responses,
        theta,
        pattern_chunk_size=len(responses),
        theta_chunk_size=len(theta),
    )
    monkeypatch.setattr(
        mixture_module,
        "_MIXTURE_MAX_LIKELIHOOD_VALUES",
        120,
    )
    original = model._grid_log_likelihood_chunk
    block_shapes: list[tuple[int, int]] = []

    def record_block(response_chunk, log_correct, log_incorrect):
        block_shapes.append((len(response_chunk), len(log_correct)))
        return original(response_chunk, log_correct, log_incorrect)

    monkeypatch.setattr(model, "_grid_log_likelihood_chunk", record_block)
    actual = model.log_likelihood_batch(responses, theta)

    assert block_shapes
    assert max(rows for rows, _ in block_shapes) <= 5
    assert max(columns for _, columns in block_shapes) <= 4
    assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_grid_likelihood_supports_an_empty_theta_grid() -> None:
    result = _model().log_likelihood_batch(_responses(), np.empty(0))

    assert result.shape == (len(_responses()), 0)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_paired_likelihoods_reject_invalid_chunk_sizes(chunk_size) -> None:
    model = _model()
    responses = _responses()[:2]
    theta = np.array([-0.5, 0.5])

    for method in (
        model.log_likelihood,
        model.class_posterior,
        model.classify_persons,
    ):
        with pytest.raises(ValueError, match="chunk_size"):
            method(responses, theta, chunk_size=chunk_size)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
@pytest.mark.parametrize("name", ["pattern_chunk_size", "theta_chunk_size"])
def test_grid_likelihood_rejects_invalid_chunk_sizes(name, chunk_size) -> None:
    kwargs = {name: chunk_size}

    with pytest.raises(ValueError, match=name):
        _model().log_likelihood_batch(
            _responses()[:2],
            np.array([-0.5, 0.5]),
            **kwargs,
        )


def test_classification_does_not_materialize_normalized_posteriors(
    monkeypatch,
) -> None:
    model = _model()

    def unexpected(*args, **kwargs):
        raise AssertionError("posterior probabilities should not be constructed")

    monkeypatch.setattr(model, "class_posterior", unexpected)
    result = model.classify_persons(
        _responses(),
        np.zeros(len(_responses())),
        chunk_size=2,
    )

    assert result.shape == (len(_responses()),)
