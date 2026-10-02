"""Posterior likelihood reuse, respondent ordering, and independent grid masses."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit, logsumexp

from mirt import get_backend, set_backend
from mirt.backends.rust._helpers import RUST_AVAILABLE
from mirt.models import GradedResponseModel, TwoParameterLogistic
from mirt.scoring.eap import EAPScorer


def _model(kind: str):
    if kind == "graded":
        model = GradedResponseModel(n_items=3, n_categories=[3, 4, 3])
        model.set_parameters(
            discrimination=np.array([0.8, 1.1, 1.4]),
            thresholds=np.array([[-1.0, 1.0, 0.0], [-1.5, 0.0, 1.5], [-0.8, 0.7, 0.0]]),
        )
        patterns = np.array([[0, 2, 2], [1, -1, 0], [2, 3, 1], [-1, -1, -1]])
    else:
        model = TwoParameterLogistic(n_items=3, n_factors=2 if kind == "2D" else 1)
        model.set_parameters(
            discrimination=(
                np.array([[1.2, 0.1], [0.2, 1.4], [0.8, 0.9]])
                if kind == "2D"
                else np.array([0.8, 1.1, 1.4])
            ),
            difficulty=np.array([-0.7, 0.1, 0.9]),
        )
        patterns = np.array([[0, 1, 0], [1, -1, 0], [1, 1, 1], [-1, -1, -1]])
    model._is_fitted = True
    return model, patterns


def _independent_grid_distribution(model, patterns):
    nodes, weights = np.polynomial.hermite.hermgauss(9)
    nodes *= np.sqrt(2)
    weights /= np.sqrt(np.pi)
    if model.n_factors == 1:
        points, prior = nodes[:, None], weights
    else:
        mesh = np.meshgrid(nodes, nodes, indexing="ij")
        points = np.column_stack([values.ravel() for values in mesh])
        prior = np.multiply.outer(weights, weights).ravel()

    log_joint = np.broadcast_to(np.log(prior), (len(patterns), len(prior))).copy()
    for item in range(model.n_items):
        if model.is_polytomous:
            cumulative = expit(
                model.discrimination[item]
                * (
                    points[:, :1]
                    - model.thresholds[item, : model.n_categories[item] - 1]
                )
            )
            probabilities = -np.diff(
                np.column_stack(
                    [np.ones(len(points)), cumulative, np.zeros(len(points))]
                ),
                axis=1,
            )
        else:
            slope = np.atleast_1d(model.discrimination[item])
            logits = (points - model.difficulty[item]) @ slope
            probabilities = np.column_stack([expit(-logits), expit(logits)])
        for person, response in enumerate(patterns[:, item]):
            if response >= 0:
                log_joint[person] += np.log(probabilities[:, response])

    log_marginal = logsumexp(log_joint, axis=1)
    posterior = np.exp(log_joint - log_marginal[:, None])
    return points, posterior, log_marginal


@pytest.mark.parametrize("kind", ["2PL", "2D", "graded"])
@pytest.mark.parametrize("batch_size", [1, 2, 7])
@pytest.mark.parametrize("backend", ["numpy", "rust"])
def test_posterior_reuses_normalized_patterns_and_preserves_row_order(
    monkeypatch: pytest.MonkeyPatch, kind: str, batch_size: int, backend: str
) -> None:
    if backend == "rust" and not RUST_AVAILABLE:
        pytest.skip("native backend is unavailable")
    previous = get_backend()
    set_backend(backend)
    try:
        model, patterns = _model(kind)
        indices = np.random.default_rng(491).integers(0, len(patterns), size=4097)
        # A strided input with multiple missing codes exercises normalization
        # before compression; the four canonical patterns occur thousands of times.
        storage = np.zeros((len(indices), model.n_items * 2), dtype=np.int_)
        responses = storage[:, ::2]
        responses[:] = patterns[indices]
        missing = responses < 0
        responses[missing] = -np.resize(
            np.array([1, 9, 999]), np.count_nonzero(missing)
        )
        original_input = responses.copy()
        original = model.log_likelihood_batch
        batches: list[np.ndarray] = []

        def capture(response_batch, theta):
            batches.append(response_batch.copy())
            return original(response_batch, theta)

        monkeypatch.setattr(model, "log_likelihood_batch", capture)
        person_ids = np.arange(len(indices)) + 10_000
        posterior = EAPScorer(n_quadpts=9, batch_size=batch_size).posterior(
            model, responses, person_ids=person_ids
        )
        points, reference_weights, reference_marginal = _independent_grid_distribution(
            model, patterns
        )
        evaluated = np.concatenate(batches)
        assert evaluated.shape == patterns.shape
        assert max(map(len, batches)) <= batch_size
        assert {tuple(row) for row in evaluated} == {tuple(row) for row in patterns}
        assert_allclose(posterior.points, points, atol=3e-15)
        assert_allclose(
            posterior.weights, reference_weights[indices], rtol=2e-13, atol=1e-15
        )
        assert_allclose(
            posterior.log_marginal_likelihood, reference_marginal[indices], atol=2e-14
        )
        assert posterior.person_ids == person_ids.tolist()
        assert_array_equal(responses, original_input)
    finally:
        set_backend(previous)


def test_posterior_skips_compression_when_many_patterns_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(391)
    model = TwoParameterLogistic(n_items=20)
    model._is_fitted = True
    responses = rng.integers(0, 2, size=(4097, 20))
    original = model.log_likelihood_batch
    call_sizes: list[int] = []

    def capture(response_batch, theta):
        call_sizes.append(len(response_batch))
        return original(response_batch, theta)

    monkeypatch.setattr(model, "log_likelihood_batch", capture)
    actual = EAPScorer(n_quadpts=9, batch_size=128).posterior(model, responses)
    assert sum(call_sizes) == len(responses)
    assert max(call_sizes) == 128
    assert_allclose(actual.weights.sum(axis=1), 1.0, atol=1e-14)


def test_repeated_posterior_rows_own_separate_output_storage() -> None:
    model, patterns = _model("2PL")
    posterior = EAPScorer(n_quadpts=9).posterior(model, patterns[[1, 1]])
    unchanged = posterior.weights[1].copy()
    posterior.weights[0, 0] = 1.0
    assert_array_equal(posterior.weights[1], unchanged)
