"""Tests for robust scaling methods in GRDIF."""

import numpy as np
import pytest

from mirt.diagnostics.dif import (
    _compute_robust_scale,
    compute_grdif,
)


@pytest.fixture
def two_group_data():
    """Generate synthetic data for 2-group DIF testing."""
    rng = np.random.default_rng(42)
    n_ref = 300
    n_focal = 300
    n_items = 15

    theta_ref = rng.normal(0, 1, n_ref)
    theta_focal = rng.normal(0, 1, n_focal)

    difficulty = rng.uniform(-2, 2, n_items)
    discrimination = rng.uniform(0.8, 2.0, n_items)

    data_ref = np.zeros((n_ref, n_items), dtype=np.int_)
    for j in range(n_items):
        prob = 1 / (1 + np.exp(-discrimination[j] * (theta_ref - difficulty[j])))
        data_ref[:, j] = (rng.random(n_ref) < prob).astype(int)

    difficulty_focal = difficulty.copy()
    difficulty_focal[0] += 0.8
    difficulty_focal[1] += 0.6

    data_focal = np.zeros((n_focal, n_items), dtype=np.int_)
    for j in range(n_items):
        prob = 1 / (
            1 + np.exp(-discrimination[j] * (theta_focal - difficulty_focal[j]))
        )
        data_focal[:, j] = (rng.random(n_focal) < prob).astype(int)

    data = np.vstack([data_ref, data_focal])
    groups = np.array(["ref"] * n_ref + ["focal"] * n_focal)

    return data, groups


@pytest.fixture
def data_with_outliers():
    """Generate data with outliers for robust testing."""
    rng = np.random.default_rng(123)
    n_ref = 250
    n_focal = 250
    n_items = 10

    theta_ref = rng.normal(0, 1, n_ref)
    theta_focal = rng.normal(0, 1, n_focal)

    theta_ref[:10] = rng.uniform(4, 6, 10)
    theta_focal[:10] = rng.uniform(-6, -4, 10)

    difficulty = rng.uniform(-2, 2, n_items)
    discrimination = rng.uniform(0.8, 2.0, n_items)

    data_ref = np.zeros((n_ref, n_items), dtype=np.int_)
    for j in range(n_items):
        prob = 1 / (1 + np.exp(-discrimination[j] * (theta_ref - difficulty[j])))
        data_ref[:, j] = (rng.random(n_ref) < prob).astype(int)

    data_focal = np.zeros((n_focal, n_items), dtype=np.int_)
    for j in range(n_items):
        prob = 1 / (1 + np.exp(-discrimination[j] * (theta_focal - difficulty[j])))
        data_focal[:, j] = (rng.random(n_focal) < prob).astype(int)

    data = np.vstack([data_ref, data_focal])
    groups = np.array(["ref"] * n_ref + ["focal"] * n_focal)

    return data, groups


class TestComputeRobustScale:
    """Tests for _compute_robust_scale helper function."""

    def test_mean_method_matches_variance(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 2, 100)

        result = _compute_robust_scale(data, method="mean")
        expected = np.var(data, ddof=1)

        assert np.isclose(result, expected, rtol=1e-10)

    def test_mad_method_returns_positive(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 2, 100)

        result = _compute_robust_scale(data, method="mad")

        assert result > 0

    def test_iqr_method_returns_positive(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 2, 100)

        result = _compute_robust_scale(data, method="iqr")

        assert result > 0

    def test_mad_robust_to_outliers(self):
        rng = np.random.default_rng(42)
        data_clean = rng.normal(0, 1, 100)
        data_outliers = data_clean.copy()
        data_outliers[:5] = 100.0

        var_clean = _compute_robust_scale(data_clean, method="mean")
        var_outliers = _compute_robust_scale(data_outliers, method="mean")
        mad_clean = _compute_robust_scale(data_clean, method="mad")
        mad_outliers = _compute_robust_scale(data_outliers, method="mad")

        var_ratio = var_outliers / var_clean
        mad_ratio = mad_outliers / mad_clean

        assert var_ratio > mad_ratio

    def test_iqr_robust_to_outliers(self):
        rng = np.random.default_rng(42)
        data_clean = rng.normal(0, 1, 100)
        data_outliers = data_clean.copy()
        data_outliers[:5] = 100.0

        var_clean = _compute_robust_scale(data_clean, method="mean")
        var_outliers = _compute_robust_scale(data_outliers, method="mean")
        iqr_clean = _compute_robust_scale(data_clean, method="iqr")
        iqr_outliers = _compute_robust_scale(data_outliers, method="iqr")

        var_ratio = var_outliers / var_clean
        iqr_ratio = iqr_outliers / iqr_clean

        assert var_ratio > iqr_ratio

    def test_single_element_returns_one_for_mean(self):
        data = np.array([5.0])

        result = _compute_robust_scale(data, method="mean")

        assert result == 1.0

    def test_invalid_method_raises(self):
        data = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Unknown scaling method"):
            _compute_robust_scale(data, method="invalid")


class TestGRDIFScalingMethods:
    """Tests for scaling methods in compute_grdif."""

    def test_scaling_methods_run(self, two_group_data):
        data, groups = two_group_data

        for method in ["mean", "mad", "iqr"]:
            result = compute_grdif(data, groups, model="2PL", scaling_method=method)

            assert result is not None
            assert len(result["grdif_r"]) == data.shape[1]
            assert len(result["grdif_s"]) == data.shape[1]
            assert len(result["grdif_rs"]) == data.shape[1]

    def test_default_scaling_is_mean(self, two_group_data):
        data, groups = two_group_data

        result_default = compute_grdif(data, groups, model="2PL")
        result_mean = compute_grdif(data, groups, model="2PL", scaling_method="mean")

        np.testing.assert_allclose(result_default["grdif_r"], result_mean["grdif_r"])
        np.testing.assert_allclose(result_default["grdif_s"], result_mean["grdif_s"])
        np.testing.assert_allclose(result_default["grdif_rs"], result_mean["grdif_rs"])

    def test_robust_methods_less_affected_by_outliers(self, data_with_outliers):
        data, groups = data_with_outliers

        result_mean = compute_grdif(data, groups, model="2PL", scaling_method="mean")
        result_mad = compute_grdif(data, groups, model="2PL", scaling_method="mad")
        result_iqr = compute_grdif(data, groups, model="2PL", scaling_method="iqr")

        assert result_mean is not None
        assert result_mad is not None
        assert result_iqr is not None

        assert np.all(np.isfinite(result_mean["grdif_rs"]))
        assert np.all(np.isfinite(result_mad["grdif_rs"]))
        assert np.all(np.isfinite(result_iqr["grdif_rs"]))

    def test_p_values_valid_for_all_methods(self, two_group_data):
        data, groups = two_group_data

        for method in ["mean", "mad", "iqr"]:
            result = compute_grdif(data, groups, model="2PL", scaling_method=method)

            assert np.all((result["p_value_r"] >= 0) & (result["p_value_r"] <= 1))
            assert np.all((result["p_value_s"] >= 0) & (result["p_value_s"] <= 1))
            assert np.all((result["p_value_rs"] >= 0) & (result["p_value_rs"] <= 1))

    def test_scaling_with_purification(self, two_group_data):
        data, groups = two_group_data

        for method in ["mean", "mad", "iqr"]:
            result = compute_grdif(
                data,
                groups,
                model="2PL",
                scaling_method=method,
                purify=True,
                max_purify_iter=3,
            )

            assert result is not None
            assert result["purification_history"] is not None


class TestGRDIFScalingEdgeCases:
    """Test edge cases for scaling methods."""

    def test_small_sample_sizes(self):
        rng = np.random.default_rng(999)
        n_ref = 30
        n_focal = 30
        n_items = 5

        theta_ref = rng.normal(0, 1, n_ref)
        theta_focal = rng.normal(0, 1, n_focal)
        difficulty = rng.uniform(-1, 1, n_items)

        data_ref = np.zeros((n_ref, n_items), dtype=np.int_)
        data_focal = np.zeros((n_focal, n_items), dtype=np.int_)

        for j in range(n_items):
            prob_ref = 1 / (1 + np.exp(-(theta_ref - difficulty[j])))
            prob_focal = 1 / (1 + np.exp(-(theta_focal - difficulty[j])))
            data_ref[:, j] = (rng.random(n_ref) < prob_ref).astype(int)
            data_focal[:, j] = (rng.random(n_focal) < prob_focal).astype(int)

        data = np.vstack([data_ref, data_focal])
        groups = np.array(["ref"] * n_ref + ["focal"] * n_focal)

        for method in ["mean", "mad", "iqr"]:
            result = compute_grdif(data, groups, model="2PL", scaling_method=method)
            assert result is not None
            assert np.all(np.isfinite(result["grdif_rs"]))

    def test_three_groups_with_scaling(self):
        rng = np.random.default_rng(777)
        n_per_group = 100
        n_items = 8

        data_list = []
        group_labels = []

        for g in range(3):
            theta = rng.normal(0, 1, n_per_group)
            difficulty = rng.uniform(-2, 2, n_items)
            data_g = np.zeros((n_per_group, n_items), dtype=np.int_)
            for j in range(n_items):
                prob = 1 / (1 + np.exp(-(theta - difficulty[j])))
                data_g[:, j] = (rng.random(n_per_group) < prob).astype(int)
            data_list.append(data_g)
            group_labels.extend([f"G{g}"] * n_per_group)

        data = np.vstack(data_list)
        groups = np.array(group_labels)

        for method in ["mean", "mad", "iqr"]:
            result = compute_grdif(data, groups, model="2PL", scaling_method=method)
            assert result["n_groups"] == 3
            assert np.all(np.isfinite(result["grdif_rs"]))
