"""Regression coverage for stable, vectorized IRTree estimation."""

from __future__ import annotations

import numpy as np
import pytest

from mirt.estimation.irtree_em import IRTreeEMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models.irtree import IRTreeModel


def _simulate_responses(
    model: IRTreeModel,
    n_persons: int,
    seed: int = 581,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = rng.multivariate_normal(
        np.zeros(model.n_traits),
        np.eye(model.n_traits),
        size=n_persons,
    )
    probabilities = model.probability(theta)
    uniforms = rng.random((n_persons, model.n_items, 1))
    return np.sum(uniforms > np.cumsum(probabilities, axis=2), axis=2)


def test_bulk_log_likelihood_matches_category_likelihood() -> None:
    model = IRTreeModel(n_items=3, tree_spec="bockenholt")
    model.set_parameters(
        discrimination=np.array(
            [
                [0.8, 1.2, 1.4, 0.7],
                [1.1, 0.9, 1.3, 1.6],
                [1.5, 0.6, 1.0, 1.2],
            ]
        ),
        difficulty=np.array(
            [
                [0.2, -0.3, 0.5, 0.1],
                [-0.2, 0.4, 0.0, -0.6],
                [0.7, -0.1, 0.3, -0.4],
            ]
        ),
    )
    responses = np.array([[0, 4, 2], [2, 3, 1], [1, -1, 4]], dtype=int)
    theta = np.array([[0.1, -0.2, 0.3], [0.5, 0.2, -0.4], [-0.3, 0.7, 0.1]])
    pseudo, assignments, valid = model.expand_to_pseudo_items(responses)

    observed = IRTreeEMEstimator._compute_log_likelihoods(
        model, pseudo, assignments, valid, theta
    )
    expected = np.column_stack(
        [
            model.log_likelihood(
                responses,
                np.broadcast_to(point, (len(responses), model.n_traits)),
            )
            for point in theta
        ]
    )

    np.testing.assert_allclose(observed, expected, atol=1e-14)


def test_vectorized_expected_counts_match_direct_aggregation() -> None:
    model = IRTreeModel(n_items=2, tree_spec="direction_intensity")
    responses = np.array([[0, 4], [2, 3], [1, -1], [4, 0]], dtype=int)
    pseudo, _assignments, valid = model.expand_to_pseudo_items(responses)
    rng = np.random.default_rng(91)
    posterior = rng.random((len(responses), 7))
    posterior /= posterior.sum(axis=1, keepdims=True)

    observed_correct, observed_total = IRTreeEMEstimator._expected_counts(
        pseudo, valid, posterior
    )
    expected_correct = np.zeros_like(observed_correct)
    expected_total = np.zeros_like(observed_total)
    for person_idx in range(len(responses)):
        for item_idx in range(model.n_items):
            for node_idx in range(model.n_nodes):
                if valid[person_idx, item_idx, node_idx]:
                    expected_total[item_idx, node_idx] += posterior[person_idx]
                    if pseudo[person_idx, item_idx, node_idx] == 1:
                        expected_correct[item_idx, node_idx] += posterior[person_idx]

    np.testing.assert_allclose(observed_correct, expected_correct, atol=1e-15)
    np.testing.assert_allclose(observed_total, expected_total, atol=1e-15)


def test_log_marginals_remain_finite_for_long_response_vectors() -> None:
    model = IRTreeModel(n_items=1000, tree_spec="direction_intensity")
    responses = np.tile(np.arange(3, dtype=int), (2, 334))[:, : model.n_items]
    pseudo, assignments, valid = model.expand_to_pseudo_items(responses)
    estimator = IRTreeEMEstimator(n_quadpts=3)
    estimator._quadrature = GaussHermiteQuadrature(
        n_points=3, n_dimensions=model.n_traits
    )

    posterior, log_marginal = estimator._e_step(
        model,
        pseudo,
        assignments,
        valid,
        np.zeros(model.n_traits),
        np.eye(model.n_traits),
        return_log=True,
    )

    assert np.all(np.isfinite(log_marginal))
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-13)


def test_fit_reports_final_state_covariance_and_parameter_uncertainty() -> None:
    generating_model = IRTreeModel(n_items=2, tree_spec="direction_intensity")
    responses = _simulate_responses(generating_model, n_persons=120)
    model = IRTreeModel(n_items=2, tree_spec="direction_intensity")
    estimator = IRTreeEMEstimator(n_quadpts=3, max_iter=3, tol=1e-8)

    result = estimator.fit(model, responses)

    pseudo, assignments, valid = model.expand_to_pseudo_items(responses)
    _posterior, log_marginal = estimator._e_step(
        model,
        pseudo,
        assignments,
        valid,
        result.trait_means,
        result.trait_covariance,
        return_log=True,
    )
    assert result.log_likelihood == pytest.approx(float(log_marginal.sum()))
    assert estimator.convergence_history[-1] == pytest.approx(result.log_likelihood)
    assert result.trait_covariance.shape == (model.n_traits, model.n_traits)
    assert np.min(np.linalg.eigvalsh(result.trait_covariance)) > 0.0
    assert np.all(np.isfinite(result.theta_estimates))
    assert np.all(np.isfinite(result.theta_se))
    for parameter_se in result.standard_errors.values():
        assert np.all(np.isfinite(parameter_se))
        assert np.all(parameter_se > 0.0)

    expected_parameters = (
        2 * model.n_items * model.n_nodes
        + model.n_traits
        + model.n_traits * (model.n_traits + 1) // 2
    )
    assert result.n_parameters == expected_parameters
    for trait_idx in range(model.n_traits):
        assert f"Variance: {result.trait_covariance[trait_idx, trait_idx]:>8.4f}" in (
            result.trait_summary()
        )


def test_uncorrelated_model_keeps_fixed_identity_distribution() -> None:
    model = IRTreeModel(
        n_items=1,
        tree_spec="direction_intensity",
        correlated_traits=False,
    )
    responses = np.tile(np.arange(5, dtype=int), 12).reshape(-1, 1)

    result = IRTreeEMEstimator(n_quadpts=3, max_iter=1).fit(model, responses)

    np.testing.assert_array_equal(result.trait_means, np.zeros(model.n_traits))
    np.testing.assert_array_equal(result.trait_covariance, np.eye(model.n_traits))
    np.testing.assert_array_equal(result.trait_correlations, np.eye(model.n_traits))
    assert model.trait_correlations is None
    assert result.n_parameters == 2 * model.n_items * model.n_nodes


def test_convergence_is_reported_from_the_actual_tolerance_check() -> None:
    model = IRTreeModel(n_items=1, tree_spec="direction_intensity")
    responses = np.tile(np.arange(5, dtype=int), 8).reshape(-1, 1)

    result = IRTreeEMEstimator(
        n_quadpts=3,
        max_iter=3,
        tol=1e9,
        estimate_correlations=False,
    ).fit(model, responses)

    assert result.converged
    assert result.n_iterations == 2


def test_fit_preserves_response_validation_and_missing_values() -> None:
    estimator = IRTreeEMEstimator(n_quadpts=3, max_iter=1)

    with pytest.raises(MirtDataError, match="integer category codes"):
        estimator.fit(
            IRTreeModel(n_items=1, tree_spec="direction_intensity"),
            np.array([[0.5]]),
        )

    result = estimator.fit(
        IRTreeModel(n_items=1, tree_spec="direction_intensity"),
        np.array([[0.0], [np.nan], [2.0], [-1.0], [4.0]]),
    )
    assert np.isfinite(result.log_likelihood)


@pytest.mark.parametrize("n_quadpts", [0, -1, 2.5, True])
def test_quadrature_size_must_be_a_positive_integer(n_quadpts: object) -> None:
    with pytest.raises(MirtValidationError, match="positive integer"):
        IRTreeEMEstimator(n_quadpts=n_quadpts)  # type: ignore[arg-type]


def test_correlation_option_must_be_boolean() -> None:
    with pytest.raises(MirtValidationError, match="boolean"):
        IRTreeEMEstimator(estimate_correlations=1)  # type: ignore[arg-type]
