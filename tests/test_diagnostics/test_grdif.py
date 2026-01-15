"""Tests for GRDIF (Generalized Residual DIF) framework."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.diagnostics.dif import (
    compute_grdif,
    compute_pairwise_rdif,
    grdif_effect_size,
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
def three_group_data():
    """Generate synthetic data for 3-group DIF testing."""
    rng = np.random.default_rng(123)
    n_per_group = 200
    n_items = 12

    thetas = [
        rng.normal(0, 1, n_per_group),
        rng.normal(0, 1, n_per_group),
        rng.normal(0, 1, n_per_group),
    ]

    difficulty = rng.uniform(-2, 2, n_items)
    discrimination = rng.uniform(0.8, 2.0, n_items)

    difficulty_shifts = [
        np.zeros(n_items),
        np.zeros(n_items),
        np.zeros(n_items),
    ]
    difficulty_shifts[1][0] = 0.7
    difficulty_shifts[2][0] = -0.5
    difficulty_shifts[2][1] = 0.6

    data_list = []
    for g, (theta_g, shift) in enumerate(zip(thetas, difficulty_shifts)):
        data_g = np.zeros((n_per_group, n_items), dtype=np.int_)
        diff_g = difficulty + shift
        for j in range(n_items):
            prob = 1 / (1 + np.exp(-discrimination[j] * (theta_g - diff_g[j])))
            data_g[:, j] = (rng.random(n_per_group) < prob).astype(int)
        data_list.append(data_g)

    data = np.vstack(data_list)
    groups = np.array(
        ["group_A"] * n_per_group
        + ["group_B"] * n_per_group
        + ["group_C"] * n_per_group
    )

    return data, groups


class TestComputeGRDIF:
    """Tests for compute_grdif function."""

    def test_two_groups_basic(self, two_group_data):
        data, groups = two_group_data

        result = compute_grdif(data, groups, model="2PL", alpha=0.05)

        assert "grdif_r" in result
        assert "grdif_s" in result
        assert "grdif_rs" in result
        assert "p_value_r" in result
        assert "p_value_s" in result
        assert "p_value_rs" in result
        assert "flagged_r" in result
        assert "flagged_s" in result
        assert "flagged_rs" in result

        assert result["n_groups"] == 2
        assert len(result["group_labels"]) == 2
        assert len(result["grdif_r"]) == data.shape[1]

    def test_three_groups_basic(self, three_group_data):
        data, groups = three_group_data

        result = compute_grdif(data, groups, model="2PL", alpha=0.05)

        assert result["n_groups"] == 3
        assert len(result["group_labels"]) == 3

        n_items = data.shape[1]
        assert len(result["grdif_r"]) == n_items
        assert len(result["grdif_s"]) == n_items
        assert len(result["grdif_rs"]) == n_items

    def test_chi_square_distribution(self, two_group_data):
        data, groups = two_group_data

        result = compute_grdif(data, groups, model="2PL")

        assert np.all(result["grdif_r"] >= 0)
        assert np.all(result["grdif_s"] >= 0)
        assert np.all(result["grdif_rs"] >= 0)

        assert np.all((result["p_value_r"] >= 0) & (result["p_value_r"] <= 1))
        assert np.all((result["p_value_s"] >= 0) & (result["p_value_s"] <= 1))
        assert np.all((result["p_value_rs"] >= 0) & (result["p_value_rs"] <= 1))

    def test_dif_items_detected(self, two_group_data):
        data, groups = two_group_data

        result = compute_grdif(data, groups, model="2PL", alpha=0.10)

        assert np.sum(result["flagged_rs"][:2]) >= 1

    def test_purification(self, two_group_data):
        data, groups = two_group_data

        result = compute_grdif(
            data, groups, model="2PL", purify=True, max_purify_iter=5
        )

        assert "purification_history" in result
        assert result["purification_history"] is not None
        assert "anchor_items" in result

    def test_different_purify_by(self, two_group_data):
        data, groups = two_group_data

        result_rs = compute_grdif(data, groups, purify=True, purify_by="grdif_rs")
        result_r = compute_grdif(data, groups, purify=True, purify_by="grdif_r")
        result_s = compute_grdif(data, groups, purify=True, purify_by="grdif_s")

        assert result_rs is not None
        assert result_r is not None
        assert result_s is not None

    def test_single_group_error(self, two_group_data):
        data, _ = two_group_data
        single_groups = np.array(["A"] * data.shape[0])

        with pytest.raises(ValueError, match="at least 2 groups"):
            compute_grdif(data, single_groups)

    def test_scoring_methods(self, two_group_data):
        data, groups = two_group_data

        for method in ["EAP", "MAP"]:
            result = compute_grdif(data, groups, scoring_method=method)
            assert result is not None
            assert len(result["grdif_r"]) == data.shape[1]


class TestPairwiseRDIF:
    """Tests for compute_pairwise_rdif function."""

    def test_three_groups_pairwise(self, three_group_data):
        data, groups = three_group_data

        result = compute_pairwise_rdif(data, groups, model="2PL")

        assert "pairs" in result
        assert len(result["pairs"]) == 3

        n_items = data.shape[1]
        assert result["rdif_r"].shape == (3, n_items)
        assert result["rdif_s"].shape == (3, n_items)
        assert result["rdif_rs"].shape == (3, n_items)

    def test_two_groups_single_pair(self, two_group_data):
        data, groups = two_group_data

        result = compute_pairwise_rdif(data, groups, model="2PL")

        assert len(result["pairs"]) == 1

        n_items = data.shape[1]
        assert result["rdif_r"].shape == (1, n_items)

    def test_pairwise_vs_grdif_consistency(self, two_group_data):
        data, groups = two_group_data

        grdif_result = compute_grdif(data, groups, model="2PL")
        pairwise_result = compute_pairwise_rdif(data, groups, model="2PL")

        assert_allclose(
            grdif_result["grdif_r"], pairwise_result["rdif_r"][0], rtol=0.1, atol=0.1
        )


class TestGRDIFEffectSize:
    """Tests for grdif_effect_size function."""

    def test_effect_size_computation(self, two_group_data):
        data, groups = two_group_data

        grdif_result = compute_grdif(data, groups, model="2PL")
        effect_sizes = grdif_effect_size(data, groups, grdif_result)

        assert len(effect_sizes) == data.shape[1]
        assert np.all(effect_sizes >= 0)

    def test_different_effect_types(self, two_group_data):
        data, groups = two_group_data

        grdif_result = compute_grdif(data, groups, model="2PL")

        es_mrr = grdif_effect_size(data, groups, grdif_result, effect_type="delta_mrr")
        es_msr = grdif_effect_size(data, groups, grdif_result, effect_type="delta_msr")
        es_max = grdif_effect_size(data, groups, grdif_result, effect_type="max_diff")

        assert len(es_mrr) == data.shape[1]
        assert len(es_msr) == data.shape[1]
        assert len(es_max) == data.shape[1]

        assert np.all(es_max >= es_mrr)
        assert np.all(es_max >= es_msr)

    def test_dif_items_have_larger_effect(self, two_group_data):
        data, groups = two_group_data

        grdif_result = compute_grdif(data, groups, model="2PL")
        effect_sizes = grdif_effect_size(data, groups, grdif_result)

        mean_dif_effect = np.mean(effect_sizes[:2])
        mean_other_effect = np.mean(effect_sizes[2:])

        assert mean_dif_effect > mean_other_effect * 0.5


class TestGRDIFEdgeCases:
    """Test edge cases and robustness."""

    def test_missing_responses(self, two_group_data):
        data, groups = two_group_data
        data_with_missing = data.copy()

        rng = np.random.default_rng(999)
        missing_mask = rng.random(data.shape) < 0.05
        data_with_missing[missing_mask] = -1

        result = compute_grdif(data_with_missing, groups, model="2PL")

        assert result is not None
        assert len(result["grdif_r"]) == data.shape[1]

    def test_unbalanced_groups(self):
        rng = np.random.default_rng(777)
        n_large = 400
        n_small = 100
        n_items = 10

        theta_large = rng.normal(0, 1, n_large)
        theta_small = rng.normal(0, 1, n_small)

        difficulty = rng.uniform(-2, 2, n_items)

        data_large = np.zeros((n_large, n_items), dtype=np.int_)
        data_small = np.zeros((n_small, n_items), dtype=np.int_)

        for j in range(n_items):
            prob_large = 1 / (1 + np.exp(-(theta_large - difficulty[j])))
            prob_small = 1 / (1 + np.exp(-(theta_small - difficulty[j])))
            data_large[:, j] = (rng.random(n_large) < prob_large).astype(int)
            data_small[:, j] = (rng.random(n_small) < prob_small).astype(int)

        data = np.vstack([data_large, data_small])
        groups = np.array(["large"] * n_large + ["small"] * n_small)

        result = compute_grdif(data, groups, model="2PL")

        assert result is not None
        assert result["group_sizes"]["large"] == n_large
        assert result["group_sizes"]["small"] == n_small

    def test_many_groups(self):
        rng = np.random.default_rng(888)
        n_groups = 5
        n_per_group = 100
        n_items = 8

        data_list = []
        group_labels = []

        for g in range(n_groups):
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

        result = compute_grdif(data, groups, model="2PL")

        assert result["n_groups"] == n_groups

        pairwise = compute_pairwise_rdif(data, groups, model="2PL")
        expected_pairs = n_groups * (n_groups - 1) // 2
        assert len(pairwise["pairs"]) == expected_pairs
