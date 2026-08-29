"""Performance contracts for vectorized sum-score EAP lookup construction."""

import numpy as np
from numpy.testing import assert_allclose

import mirt.scoring.eapsum as eapsum_module
from mirt import GradedResponseModel
from mirt.scoring._common import build_quadrature
from mirt.scoring.eapsum import EAPSumScorer
from mirt.utils.numeric import logsumexp


def _mixed_model() -> GradedResponseModel:
    model = GradedResponseModel(n_items=4, n_categories=[2, 3, 4, 5])
    model.set_parameters(
        discrimination=np.array([0.8, 1.0, 1.2, 1.4]),
        thresholds=np.array(
            [
                [-0.5, 0.0, 0.0, 0.0],
                [-1.0, 0.7, 0.0, 0.0],
                [-1.2, 0.0, 1.1, 0.0],
                [-1.5, -0.4, 0.5, 1.6],
            ]
        ),
    )
    model._is_fitted = True
    return model


def _scalar_distribution(
    model: GradedResponseModel,
    quad_points: np.ndarray,
    item_indices: tuple[int, ...],
) -> np.ndarray:
    max_score = sum(model._n_categories[index] - 1 for index in item_indices)
    distribution = np.full((max_score + 1, len(quad_points)), -np.inf)
    distribution[0] = 0.0

    for item_index in item_indices:
        probabilities = model.probability(quad_points, item_index)
        log_probabilities = np.log(probabilities + 1e-300)
        updated = np.full_like(distribution, -np.inf)
        for score in range(max_score + 1):
            for category in range(probabilities.shape[1]):
                if score >= category:
                    contribution = (
                        distribution[score - category] + log_probabilities[:, category]
                    )
                    updated[score] = np.logaddexp(updated[score], contribution)
        distribution = updated

    return distribution


def test_polytomous_recursion_matches_scalar_reference() -> None:
    model = _mixed_model()
    quad_points, _ = build_quadrature(
        n_quadpts=21,
        n_factors=1,
        prior_mean=None,
        prior_cov=None,
    )
    item_indices = (0, 2, 3)
    max_score = sum(model._n_categories[index] - 1 for index in item_indices)

    actual = EAPSumScorer(21)._compute_sum_score_distribution(
        model,
        quad_points,
        max_score,
        item_indices,
    )
    expected = _scalar_distribution(model, quad_points, item_indices)

    assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_polytomous_recursion_combines_whole_score_slices(monkeypatch) -> None:
    model = _mixed_model()
    quad_points, _ = build_quadrature(
        n_quadpts=17,
        n_factors=1,
        prior_mean=None,
        prior_cov=None,
    )
    item_indices = (0, 2, 3)
    call_count = 0
    original = np.logaddexp

    def counted_logaddexp(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "logaddexp", counted_logaddexp)
    EAPSumScorer(17)._compute_sum_score_distribution(
        model,
        quad_points,
        sum(model._n_categories[index] - 1 for index in item_indices),
        item_indices,
    )

    assert call_count == sum(model._n_categories[index] for index in item_indices)


def test_lookup_reduces_all_score_posteriors_together(monkeypatch) -> None:
    model = _mixed_model()
    scorer = EAPSumScorer(25)
    calls: list[tuple[int | None, bool]] = []
    original = eapsum_module.logsumexp

    def counted_logsumexp(values, axis=None, keepdims=False):
        calls.append((axis, keepdims))
        return original(values, axis=axis, keepdims=keepdims)

    monkeypatch.setattr(eapsum_module, "logsumexp", counted_logsumexp)
    lookup = scorer.get_lookup_table(model)

    assert calls == [(1, True)]

    quad_points, quad_weights = build_quadrature(
        n_quadpts=25,
        n_factors=1,
        prior_mean=None,
        prior_cov=None,
    )
    distribution = scorer._compute_sum_score_distribution(
        model,
        quad_points,
        lookup["max_score"],
    )
    log_prior = np.log(quad_weights + 1e-300)
    for score in range(lookup["max_score"] + 1):
        log_posterior = distribution[score] + log_prior
        posterior = np.exp(log_posterior - logsumexp(log_posterior))
        expected_theta = posterior @ quad_points[:, 0]
        expected_se = np.sqrt(
            np.sum(posterior * (quad_points[:, 0] - expected_theta) ** 2)
        )
        assert_allclose(lookup[score]["theta"], expected_theta, atol=1e-14)
        assert_allclose(lookup[score]["se"], expected_se, atol=1e-14)
