"""Contracts for multigroup Gaussian latent distributions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from mirt.constants import REGULARIZATION_EPSILON
from mirt.multigroup.latent import GroupLatentDistribution, MultigroupLatentDensity


def _standard_distribution(n_factors: int = 1) -> GroupLatentDistribution:
    return GroupLatentDistribution(
        mean=np.zeros(n_factors),
        cov=np.eye(n_factors),
    )


class TestGroupLatentDistribution:
    def test_unidimensional_vectors_represent_multiple_points(self):
        distribution = _standard_distribution()
        theta = np.array([-1.0, 0.0, 1.0])

        result = distribution.log_density(theta)

        expected = -0.5 * (np.log(2.0 * np.pi) + theta**2)
        np.testing.assert_allclose(result, expected)
        assert result.shape == (3,)
        np.testing.assert_allclose(
            distribution.log_density(np.array(0.5)),
            [-0.5 * (np.log(2.0 * np.pi) + 0.25)],
        )

    def test_multidimensional_vector_represents_one_point(self):
        distribution = GroupLatentDistribution(
            mean=np.array([0.5, -0.25]),
            cov=np.array([[2.0, 0.4], [0.4, 1.5]]),
        )
        theta = np.array([1.0, 0.5])

        result = distribution.log_density(theta)

        difference = theta - distribution.mean
        _, log_det = np.linalg.slogdet(distribution.cov)
        expected = -0.5 * (
            2 * np.log(2.0 * np.pi)
            + log_det
            + difference @ np.linalg.solve(distribution.cov, difference)
        )
        np.testing.assert_allclose(result, [expected])

    @pytest.mark.parametrize(
        "mean",
        [
            np.array([]),
            np.array([[0.0]]),
            np.array([np.nan]),
            np.array([np.inf]),
        ],
    )
    def test_rejects_invalid_means(self, mean):
        with pytest.raises(ValueError, match="mean"):
            GroupLatentDistribution(mean=mean, cov=np.eye(1))

    @pytest.mark.parametrize(
        ("cov", "message"),
        [
            (np.ones(2), "shape"),
            (np.array([[np.nan]]), "finite"),
            (np.array([[1.0, 0.5], [0.0, 1.0]]), "symmetric"),
            (np.zeros((2, 2)), "positive definite"),
            (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive definite"),
        ],
    )
    def test_rejects_invalid_covariances(self, cov, message):
        mean = np.zeros(cov.shape[0]) if cov.ndim == 2 else np.zeros(1)
        with pytest.raises(ValueError, match=message):
            GroupLatentDistribution(mean=mean, cov=cov)

    @pytest.mark.parametrize(
        ("flag", "value"),
        [
            ("is_reference", 1),
            ("estimate_mean", "yes"),
            ("estimate_cov", 0),
        ],
    )
    def test_rejects_non_boolean_flags(self, flag, value):
        kwargs = {flag: value}
        with pytest.raises(TypeError, match=flag):
            GroupLatentDistribution(np.zeros(1), np.eye(1), **kwargs)

    def test_reference_distribution_has_no_free_parameters(self):
        distribution = GroupLatentDistribution(
            np.zeros(2),
            np.eye(2),
            is_reference=True,
            estimate_mean=True,
            estimate_cov=True,
        )

        assert not distribution.estimate_mean
        assert not distribution.estimate_cov
        assert distribution.n_free_parameters == 0

    def test_free_parameter_count_respects_estimation_flags(self):
        distribution = GroupLatentDistribution(
            np.zeros(3),
            np.eye(3),
            estimate_mean=True,
            estimate_cov=False,
        )

        assert distribution.n_free_parameters == 3
        distribution.estimate_cov = True
        assert distribution.n_free_parameters == 9

    def test_constructor_and_copy_own_parameter_arrays(self):
        mean = np.array([0.0])
        cov = np.array([[1.0]])
        distribution = GroupLatentDistribution(mean, cov)
        copied = distribution.copy()

        mean[0] = 9.0
        cov[0, 0] = 9.0
        copied.mean[0] = 2.0

        np.testing.assert_array_equal(distribution.mean, [0.0])
        np.testing.assert_array_equal(distribution.cov, [[1.0]])
        np.testing.assert_array_equal(copied.mean, [2.0])

    def test_direct_covariance_mutation_refreshes_cached_precision(self):
        distribution = _standard_distribution()
        baseline = distribution.log_density(np.array([1.0]))

        distribution.cov[0, 0] = 4.0
        updated = distribution.log_density(np.array([1.0]))

        assert not np.array_equal(updated, baseline)
        expected = -0.5 * (np.log(8.0 * np.pi) + 0.25)
        np.testing.assert_allclose(updated, [expected])

    def test_direct_invalid_covariance_mutation_raises(self):
        distribution = _standard_distribution(2)
        distribution.cov[0, 0] = -1.0

        with pytest.raises(ValueError, match="positive definite"):
            distribution.log_density(np.zeros((1, 2)))

    def test_direct_invalid_mean_assignment_raises(self):
        distribution = _standard_distribution()
        distribution.mean = np.array([0.0, 1.0])

        with pytest.raises(ValueError, match="shape"):
            distribution.log_density(np.array([0.0]))

    @pytest.mark.parametrize(
        "theta",
        [
            np.ones((2, 3)),
            np.ones((1, 2, 1)),
            np.array([[0.0, np.nan]]),
            np.array([[0.0, np.inf]]),
        ],
    )
    def test_rejects_invalid_theta(self, theta):
        with pytest.raises(ValueError, match="theta"):
            _standard_distribution(2).log_density(theta)

    def test_quadrature_mass_is_normalized(self):
        distribution = GroupLatentDistribution(np.array([0.5]), np.array([[1.5]]))
        theta = np.linspace(-3.0, 3.0, 13)
        quadrature_weights = np.exp(-(theta**2) / 2.0)

        log_mass = distribution.log_quadrature_mass(theta, quadrature_weights)

        np.testing.assert_allclose(np.exp(log_mass).sum(), 1.0)
        assert np.all(np.isfinite(log_mass))

    def test_multidimensional_quadrature_accepts_one_point_vector(self):
        distribution = _standard_distribution(2)

        log_mass = distribution.log_quadrature_mass(
            np.array([0.0, 0.0]), np.array([1.0])
        )

        np.testing.assert_array_equal(log_mass, [0.0])


class TestMultigroupLatentDensity:
    @pytest.mark.parametrize(
        ("kwargs", "error", "message"),
        [
            ({"n_groups": 1}, ValueError, "n_groups"),
            ({"n_groups": 2.5}, ValueError, "n_groups"),
            ({"n_groups": True}, ValueError, "n_groups"),
            ({"n_groups": 2, "n_factors": 0}, ValueError, "n_factors"),
            ({"n_groups": 2, "n_factors": 1.5}, ValueError, "n_factors"),
            ({"n_groups": 2, "reference_group": True}, TypeError, "reference_group"),
            ({"n_groups": 2, "reference_group": 2}, ValueError, "reference_group"),
        ],
    )
    def test_rejects_invalid_configuration(self, kwargs, error, message):
        with pytest.raises(error, match=message):
            MultigroupLatentDensity(**kwargs)

    @pytest.mark.parametrize("group_idx", [-1, 2, 0.5, True])
    @pytest.mark.parametrize(
        "operation",
        [
            lambda density, index: density.log_density(np.array([0.0]), index),
            lambda density, index: density.log_quadrature_mass(
                np.array([0.0]), np.array([1.0]), index
            ),
            lambda density, index: density.update(
                np.array([0.0]), np.array([1.0]), index
            ),
            lambda density, index: density.get_group_mean(index),
            lambda density, index: density.get_group_cov(index),
            lambda density, index: density.set_group_distribution(index),
        ],
    )
    def test_all_group_accessors_validate_indices(
        self,
        group_idx,
        operation: Callable[[MultigroupLatentDensity, object], object],
    ):
        density = MultigroupLatentDensity(2)
        with pytest.raises((IndexError, TypeError)):
            operation(density, group_idx)

    def test_log_density_all_aligns_points_and_groups(self):
        density = MultigroupLatentDensity(3)
        density.set_group_distribution(1, mean=np.array([1.0]))
        density.set_group_distribution(2, cov=np.array([[2.0]]))
        theta = np.array([-1.0, 0.0, 1.0])

        result = density.log_density_all(theta)

        assert result.shape == (3, 3)
        for group_idx in range(3):
            np.testing.assert_allclose(
                result[:, group_idx], density.log_density(theta, group_idx)
            )

    def test_set_distribution_is_atomic_and_copies_inputs(self):
        density = MultigroupLatentDensity(2, n_factors=2)
        mean = np.array([1.0, -1.0])
        covariance = np.array([[2.0, 0.25], [0.25, 1.5]])
        density.set_group_distribution(1, mean=mean, cov=covariance)
        before_mean = density.get_group_mean(1)
        before_cov = density.get_group_cov(1)

        mean[:] = 9.0
        covariance[:] = 9.0
        with pytest.raises(ValueError, match="shape"):
            density.set_group_distribution(1, mean=np.zeros(3))

        np.testing.assert_array_equal(density.get_group_mean(1), before_mean)
        np.testing.assert_array_equal(density.get_group_cov(1), before_cov)

    def test_reference_distribution_cannot_be_modified(self):
        density = MultigroupLatentDensity(2)
        with pytest.raises(ValueError, match="reference"):
            density.set_group_distribution(0, mean=np.array([1.0]))

    def test_update_matches_weighted_moments(self):
        density = MultigroupLatentDensity(2, n_factors=2)
        points = np.array([[-2.0, 1.0], [0.0, -1.0], [3.0, 2.0]])
        weights = np.array([1.0, 2.0, 4.0])

        density.update(points, weights, 1)

        normalized = weights / weights.sum()
        expected_mean = normalized @ points
        difference = points - expected_mean
        expected_cov = difference.T @ (normalized[:, None] * difference)
        expected_cov += np.eye(2) * REGULARIZATION_EPSILON
        np.testing.assert_allclose(density.get_group_mean(1), expected_mean)
        np.testing.assert_allclose(density.get_group_cov(1), expected_cov)

    def test_update_respects_estimation_flags(self):
        density = MultigroupLatentDensity(2)
        distribution = density.distributions[1]
        distribution.estimate_mean = False
        distribution.estimate_cov = True
        density.set_group_distribution(1, mean=np.array([2.0]))

        density.update(np.array([-1.0, 1.0]), np.ones(2), 1)

        np.testing.assert_array_equal(density.get_group_mean(1), [2.0])
        np.testing.assert_allclose(
            density.get_group_cov(1), [[5.0 + REGULARIZATION_EPSILON]]
        )

    def test_zero_weight_update_is_a_noop(self):
        density = MultigroupLatentDensity(2)
        before = density.get_latent_parameters()

        density.update(np.array([-1.0, 1.0]), np.zeros(2), 1)

        np.testing.assert_array_equal(density.get_group_mean(1), before[1]["mean"])
        np.testing.assert_array_equal(density.get_group_cov(1), before[1]["cov"])

    def test_extreme_finite_weights_are_normalized_without_overflow(self):
        density = MultigroupLatentDensity(2)

        with np.errstate(over="raise", invalid="raise"):
            density.update(
                np.array([-1.0, 1.0]),
                np.array([1e308, 1e308]),
                1,
            )

        np.testing.assert_allclose(density.get_group_mean(1), [0.0])
        np.testing.assert_allclose(
            density.get_group_cov(1), [[1.0 + REGULARIZATION_EPSILON]]
        )

    @pytest.mark.parametrize(
        ("points", "weights", "message"),
        [
            (np.ones((2, 2)), np.ones(2), "theta"),
            (np.array([0.0, np.nan]), np.ones(2), "theta"),
            (np.array([-1.0, 1.0]), np.ones((2, 1)), "weights"),
            (np.array([-1.0, 1.0]), np.ones(1), "weights"),
            (np.array([-1.0, 1.0]), np.array([1.0, -1.0]), "non-negative"),
            (np.array([-1.0, 1.0]), np.array([1.0, np.nan]), "finite"),
        ],
    )
    def test_invalid_updates_raise_without_mutation(self, points, weights, message):
        density = MultigroupLatentDensity(2)
        before_mean = density.get_group_mean(1)
        before_cov = density.get_group_cov(1)

        with pytest.raises(ValueError, match=message):
            density.update(points, weights, 1)

        np.testing.assert_array_equal(density.get_group_mean(1), before_mean)
        np.testing.assert_array_equal(density.get_group_cov(1), before_cov)

    def test_reference_updates_still_validate_inputs(self):
        density = MultigroupLatentDensity(2)
        with pytest.raises(ValueError, match="non-negative"):
            density.update(np.array([-1.0, 1.0]), np.array([1.0, -1.0]), 0)

    def test_copy_is_independent(self):
        density = MultigroupLatentDensity(2)
        density.set_group_distribution(1, mean=np.array([1.0]))

        copied = density.copy()
        copied.set_group_distribution(1, mean=np.array([3.0]))

        np.testing.assert_array_equal(density.get_group_mean(1), [1.0])
        np.testing.assert_array_equal(copied.get_group_mean(1), [3.0])

    def test_parameter_snapshot_is_independent(self):
        density = MultigroupLatentDensity(2)
        parameters = density.get_latent_parameters()

        parameters[1]["mean"][0] = 9.0
        parameters[1]["cov"][0, 0] = 9.0

        np.testing.assert_array_equal(density.get_group_mean(1), [0.0])
        np.testing.assert_array_equal(density.get_group_cov(1), [[1.0]])
