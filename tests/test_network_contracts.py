"""Contract and numerical tests for network psychometrics models."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from unittest.mock import patch

import numpy as np
import pytest

from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models.network import (
    GaussianGraphicalModel,
    IsingModel,
    compare_networks,
    fit_ggm,
    fit_ising,
)


@pytest.mark.parametrize("model_type", [IsingModel, GaussianGraphicalModel])
@pytest.mark.parametrize(
    ("n_nodes", "names"),
    [
        (1, None),
        (2.5, None),
        (2, []),
        (2, ["same", "same"]),
        (2, ["valid", ""]),
    ],
)
def test_node_metadata_is_validated(
    model_type: Callable[..., object], n_nodes: object, names: list[str] | None
) -> None:
    with pytest.raises(MirtValidationError):
        model_type(n_nodes, names)


def test_model_parameter_snapshots_are_finite_and_independent() -> None:
    thresholds = np.array([0.2, -0.3])
    ising = IsingModel(2).set_thresholds(thresholds)
    thresholds[0] = 99.0
    assert ising.thresholds[0] == 0.2

    means = np.array([1.0, -1.0])
    gaussian = GaussianGraphicalModel(2).set_means(means)
    means[0] = 99.0
    assert gaussian.means[0] == 1.0

    for action in (
        lambda: ising.set_thresholds(np.array([0.0])),
        lambda: ising.set_thresholds(np.array([0.0, np.nan])),
        lambda: ising.set_interactions(np.ones((3, 3))),
        lambda: ising.set_interactions(np.array([[0.0, np.inf], [0.0, 0.0]])),
        lambda: gaussian.set_means(np.array([0.0])),
        lambda: gaussian.set_means(np.array([0.0, np.nan])),
        lambda: gaussian.set_precision_matrix(np.ones((3, 3))),
        lambda: gaussian.set_precision_matrix(np.array([[1.0, np.nan], [np.nan, 1.0]])),
        lambda: gaussian.set_precision_matrix(np.array([[1.0, 2.0], [2.0, 1.0]])),
    ):
        with pytest.raises(MirtValidationError):
            action()


def test_conditional_probabilities_are_vectorized_and_stable() -> None:
    model = IsingModel(3)
    model.set_thresholds(np.array([-1000.0, 0.2, 1000.0]))
    model.set_interactions(
        np.array([[0.0, 0.4, -0.1], [0.4, 0.0, 0.2], [-0.1, 0.2, 0.0]])
    )
    responses = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 1]])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probabilities = model.conditional_probabilities(responses)

    assert not caught
    assert probabilities.shape == (3, 3)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    for node in range(3):
        np.testing.assert_array_equal(
            model.conditional_probability(node, responses), probabilities[:, node]
        )


@pytest.mark.parametrize(
    "responses",
    [
        np.array([0, 1]),
        np.empty((0, 2), dtype=int),
        np.array([[0, 1, 0]]),
        np.array([[0, 2]]),
        np.array([[0.0, np.nan]]),
    ],
)
def test_invalid_binary_data_is_rejected(responses: np.ndarray) -> None:
    with pytest.raises(MirtDataError):
        IsingModel(2).pseudo_likelihood(responses)
    with pytest.raises(MirtDataError):
        fit_ising(responses)


def test_invalid_node_indices_are_rejected() -> None:
    ising = IsingModel(2)
    gaussian = GaussianGraphicalModel(2)
    with pytest.raises(IndexError):
        ising.conditional_probability(2, np.array([[0, 1]]))
    with pytest.raises(MirtValidationError):
        ising.conditional_probability(True, np.array([[0, 1]]))
    with pytest.raises(IndexError):
        gaussian.conditional_variance(-1)


def test_independent_ising_pseudolikelihood_has_closed_form() -> None:
    responses = np.array([[0, 0, 1], [1, 1, 0], [1, 0, 1]])
    expected = -3.0 * np.log(2.0)
    assert IsingModel(3).pseudo_likelihood(responses) == pytest.approx(expected)


def test_exact_ising_probabilities_normalize() -> None:
    model = IsingModel(3)
    model.set_thresholds(np.array([0.2, -0.4, 0.6]))
    model.set_interactions(
        np.array([[0.0, 0.5, 0.0], [0.5, 0.0, -0.2], [0.0, -0.2, 0.0]])
    )
    state_ids = np.arange(8)[:, None]
    states = ((state_ids >> np.arange(3)) & 1).astype(int)

    probabilities = model.probability(states)
    np.testing.assert_allclose(probabilities.sum(), 1.0)
    np.testing.assert_allclose(np.log(probabilities), model.log_probability(states))
    assert np.isfinite(model.log_partition_function())

    with pytest.raises(MirtValidationError, match="Exact enumeration"):
        model.log_partition_function(max_nodes=2)


def test_ising_partition_cache_reuses_and_invalidates_exact_sum() -> None:
    model = IsingModel(4)
    model.set_thresholds(np.array([0.2, -0.4, 0.6, 0.1]))
    model.set_interactions(
        np.array(
            [
                [0.0, 0.5, 0.0, 0.1],
                [0.5, 0.0, -0.2, 0.0],
                [0.0, -0.2, 0.0, 0.3],
                [0.1, 0.0, 0.3, 0.0],
            ]
        )
    )

    with patch.object(np, "einsum", wraps=np.einsum) as einsum:
        original = model.log_partition_function()
        assert model.log_partition_function() == original
        assert einsum.call_count == 1

        model.set_thresholds(model.thresholds + 0.25)
        updated = model.log_partition_function()
        assert einsum.call_count == 2

    assert updated != original
    with pytest.raises(MirtValidationError, match="Exact enumeration"):
        model.log_partition_function(max_nodes=3)


def test_ising_copy_preserves_valid_partition_cache() -> None:
    model = IsingModel(3)
    expected = model.log_partition_function()
    copied = model.copy()

    with patch.object(np, "einsum", wraps=np.einsum) as einsum:
        assert copied.log_partition_function() == expected

    assert einsum.call_count == 0


def test_ising_sampling_supports_thinning_and_empty_output() -> None:
    model = IsingModel(2).set_interactions(np.array([[0.0, 0.8], [0.8, 0.0]]))
    first = model.sample(30, n_burnin=10, thin=3, seed=42)
    second = model.sample(30, n_burnin=10, thin=3, seed=42)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (30, 2)
    assert model.sample(0).shape == (0, 2)

    for action in (
        lambda: model.sample(-1),
        lambda: model.sample(1, n_burnin=-1),
        lambda: model.sample(1, thin=0),
    ):
        with pytest.raises(MirtValidationError):
            action()


def test_gaussian_conditionals_accept_full_or_reduced_rows() -> None:
    model = GaussianGraphicalModel(3)
    model.set_means(np.array([1.0, 2.0, 3.0]))
    model.set_precision_matrix(
        np.array([[2.0, -0.4, 0.2], [-0.4, 1.5, -0.3], [0.2, -0.3, 1.2]])
    )
    full = np.array([[1.2, 2.2, 2.8], [0.8, 1.5, 3.5]])
    reduced = full[:, [0, 2]]

    np.testing.assert_allclose(
        model.conditional_mean(1, full), model.conditional_mean(1, reduced)
    )
    assert model.conditional_variance(1) == pytest.approx(1.0 / 1.5)
    assert model.expected_influence().shape == (3,)

    with pytest.raises(MirtDataError):
        model.conditional_mean(1, np.ones((2, 1)))
    with pytest.raises(MirtDataError):
        model.conditional_mean(1, np.array([[0.0, np.nan]]))


def test_gaussian_likelihood_matches_direct_formula() -> None:
    model = GaussianGraphicalModel(2)
    means = np.array([0.5, -0.2])
    precision = np.array([[2.0, -0.4], [-0.4, 1.5]])
    model.set_means(means).set_precision_matrix(precision)
    data = np.array([[0.0, 1.0], [1.0, -1.0], [0.5, 0.5]])
    centered = data - means
    expected = (
        -0.5 * data.shape[0] * data.shape[1] * np.log(2.0 * np.pi)
        + 0.5 * data.shape[0] * np.linalg.slogdet(precision)[1]
        - 0.5 * np.sum(centered @ precision * centered)
    )
    assert model.log_likelihood(data) == pytest.approx(expected)

    with pytest.raises(MirtDataError):
        model.log_likelihood(np.ones((2, 3)))
    with pytest.raises(MirtDataError):
        model.log_likelihood(np.array([[0.0, np.inf]]))


def test_gaussian_covariance_cache_returns_copies_and_invalidates() -> None:
    model = GaussianGraphicalModel(3)
    first_precision = np.array([[2.0, -0.4, 0.2], [-0.4, 1.5, -0.3], [0.2, -0.3, 1.2]])
    model.set_precision_matrix(first_precision)

    with patch.object(np.linalg, "solve", wraps=np.linalg.solve) as solve:
        first = model.covariance_matrix
        first[0, 0] = -1.0
        second = model.covariance_matrix
        assert solve.call_count == 1
        assert second[0, 0] >= 0.0

        model.set_precision_matrix(first_precision + np.eye(3))
        updated = model.covariance_matrix
        assert solve.call_count == 2

    assert not np.array_equal(second, updated)


def test_gaussian_precision_factorization_is_reused_by_likelihood() -> None:
    model = GaussianGraphicalModel(3)
    precision = np.array([[2.0, -0.4, 0.2], [-0.4, 1.5, -0.3], [0.2, -0.3, 1.2]])
    data = np.array([[0.0, 1.0, -1.0], [0.5, -0.2, 0.8]])

    with patch.object(np.linalg, "cholesky", wraps=np.linalg.cholesky) as cholesky:
        model.set_precision_matrix(precision)
        first = model.log_likelihood(data)
        second = model.log_likelihood(data)

    assert first == second
    assert cholesky.call_count == 1


def test_gaussian_copy_preserves_valid_derived_caches() -> None:
    model = GaussianGraphicalModel(3)
    expected_covariance = model.covariance_matrix
    copied = model.copy()

    with patch.object(np.linalg, "solve", wraps=np.linalg.solve) as solve:
        np.testing.assert_array_equal(copied.covariance_matrix, expected_covariance)

    assert solve.call_count == 0


def test_fit_ising_improves_objective_and_reports_convergence() -> None:
    true_model = IsingModel(4)
    true_model.set_thresholds(np.array([0.3, -0.2, 0.1, -0.4]))
    true_model.set_interactions(
        np.array(
            [
                [0.0, 0.8, 0.0, 0.0],
                [0.8, 0.0, -0.5, 0.0],
                [0.0, -0.5, 0.0, 0.6],
                [0.0, 0.0, 0.6, 0.0],
            ]
        )
    )
    responses = true_model.sample(800, n_burnin=200, seed=42)
    fitted, pseudo_likelihood = fit_ising(
        responses,
        regularization=0.01,
        max_iter=1000,
        tol=1e-5,
        node_names=["a", "b", "c", "d"],
    )

    assert fitted.is_fitted
    assert fitted.converged
    assert fitted.n_iterations > 0
    assert fitted.node_names == ["a", "b", "c", "d"]
    assert pseudo_likelihood == pytest.approx(fitted.pseudo_likelihood(responses))
    assert np.all(np.diff(fitted.objective_history) >= -1e-12)
    assert fitted.objective_history[-1] >= fitted.objective_history[0]

    copied = fitted.copy()
    np.testing.assert_array_equal(copied.objective_history, fitted.objective_history)
    assert copied.converged == fitted.converged


@pytest.mark.parametrize(
    "kwargs",
    [
        {"regularization": -1.0},
        {"regularization": np.nan},
        {"max_iter": 0},
        {"tol": 0.0},
        {"tol": np.inf},
    ],
)
def test_fit_options_are_validated(kwargs: dict[str, float]) -> None:
    binary = np.array([[0, 1], [1, 0]])
    continuous = binary.astype(float)
    with pytest.raises(MirtValidationError):
        fit_ising(binary, **kwargs)
    with pytest.raises(MirtValidationError):
        fit_ggm(continuous, **kwargs)


def test_fit_ggm_matches_mle_for_well_conditioned_data() -> None:
    rng = np.random.default_rng(42)
    data = rng.multivariate_normal(
        [1.0, -1.0, 0.5],
        [[1.5, 0.3, 0.1], [0.3, 1.0, -0.2], [0.1, -0.2, 0.8]],
        size=500,
    )
    model, likelihood = fit_ggm(data, node_names=["a", "b", "c"])
    sample_covariance = np.cov(data, rowvar=False, ddof=0)

    np.testing.assert_allclose(model.means, data.mean(axis=0))
    np.testing.assert_allclose(model.precision_matrix, np.linalg.inv(sample_covariance))
    assert model.converged
    assert model.n_iterations == 0
    assert likelihood == pytest.approx(model.log_likelihood(data))


def test_regularized_ggm_handles_singular_data_and_tracks_objective() -> None:
    values = np.arange(20.0)
    singular = np.column_stack((values, values, 2.0 * values))

    with pytest.raises(MirtValidationError, match="use regularization"):
        fit_ggm(singular)

    model, likelihood = fit_ggm(singular, regularization=0.1, max_iter=1000, tol=1e-6)
    assert model.is_fitted
    assert model.converged
    assert model.n_iterations > 0
    assert np.isfinite(likelihood)
    assert np.all(np.linalg.eigvalsh(model.precision_matrix) > 0.0)
    assert np.all(np.isfinite(model.objective_history))

    copied = model.copy()
    assert copied.converged
    np.testing.assert_array_equal(copied.objective_history, model.objective_history)
    assert model.sample(0).shape == (0, 3)


def test_regularized_ggm_satisfies_graphical_lasso_conditions() -> None:
    rng = np.random.default_rng(7)
    data = rng.normal(size=(600, 5))
    regularization = 0.05
    model, _ = fit_ggm(data, regularization=regularization, max_iter=500, tol=1e-8)
    precision = model.precision_matrix
    sample_covariance = np.cov(data, rowvar=False, ddof=0)
    smooth_gradient = np.linalg.inv(precision) - sample_covariance
    off_diagonal = ~np.eye(5, dtype=bool)
    nonzero = off_diagonal & (np.abs(precision) > 1e-7)
    zero = off_diagonal & ~nonzero

    assert model.converged
    np.testing.assert_allclose(
        smooth_gradient[nonzero],
        regularization * np.sign(precision[nonzero]),
        atol=2e-6,
    )
    assert np.all(np.abs(smooth_gradient[zero]) <= regularization + 2e-6)


def test_compare_networks_uses_only_real_edges_and_aligns_names() -> None:
    first = IsingModel(3, ["a", "b", "c"])
    first.set_interactions(
        np.array([[0.0, 0.5, 0.0], [0.5, 0.0, -0.2], [0.0, -0.2, 0.0]])
    )
    second = IsingModel(3, ["c", "a", "b"])
    second.set_interactions(
        np.array([[0.0, 0.0, -0.1], [0.0, 0.0, 0.3], [-0.1, 0.3, 0.0]])
    )

    comparison = compare_networks(first, second)
    # Aligned edges are (0.5 vs 0.3), (0 vs 0), and (-0.2 vs -0.1).
    assert comparison["mean_edge_difference"] == pytest.approx(0.1)
    assert comparison["max_edge_difference"] == pytest.approx(0.2)
    assert comparison["edge_jaccard"] == 1.0


def test_compare_networks_validates_identity_and_threshold() -> None:
    with pytest.raises(MirtValidationError, match="same node names"):
        compare_networks(IsingModel(2, ["a", "b"]), IsingModel(2, ["x", "y"]))
    with pytest.raises(MirtValidationError, match="edge_threshold"):
        compare_networks(IsingModel(2), IsingModel(2), edge_threshold=-1.0)


def test_two_node_edge_difference_is_not_diluted_by_matrix_zeros() -> None:
    first = IsingModel(2).set_interactions(np.array([[0.0, 0.5], [0.5, 0.0]]))
    second = IsingModel(2).set_interactions(np.array([[0.0, 0.3], [0.3, 0.0]]))
    comparison = compare_networks(first, second)
    assert comparison["mean_edge_difference"] == pytest.approx(0.2)
    assert comparison["max_edge_difference"] == pytest.approx(0.2)
