"""Tests for Differential Item Functioning (DIF) analysis."""

import numpy as np
import pytest

from mirt.diagnostics.dif import compute_dif, flag_dif_items


class TestDIF:
    """Tests for DIF detection functions."""

    def test_basic_dif_likelihood_ratio(self, rng):
        """Test basic DIF analysis with likelihood ratio method."""
        n_per_group = 100
        n_items = 5

        # Create two groups with similar ability distributions
        theta1 = rng.standard_normal(n_per_group)
        theta2 = rng.standard_normal(n_per_group)

        # Same difficulty for both groups (no DIF)
        difficulty = rng.normal(0, 1, n_items)

        # Generate responses
        probs1 = 1 / (1 + np.exp(-(theta1[:, None] - difficulty)))
        probs2 = 1 / (1 + np.exp(-(theta2[:, None] - difficulty)))

        responses1 = (rng.random((n_per_group, n_items)) < probs1).astype(int)
        responses2 = (rng.random((n_per_group, n_items)) < probs2).astype(int)

        data = np.vstack([responses1, responses2])
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        result = compute_dif(
            data,
            groups,
            model="2PL",
            method="likelihood_ratio",
            n_quadpts=11,
            max_iter=30,
        )

        assert "statistic" in result
        assert "p_value" in result
        assert "effect_size" in result
        assert "classification" in result

        assert len(result["statistic"]) == n_items
        assert len(result["p_value"]) == n_items
        assert np.all(result["p_value"] >= 0) and np.all(result["p_value"] <= 1)

    def test_dif_wald_method(self, rng):
        """Test DIF analysis with Wald method."""
        n_per_group = 80
        n_items = 5

        data = rng.integers(0, 2, size=(n_per_group * 2, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        result = compute_dif(
            data, groups, model="2PL", method="wald", n_quadpts=11, max_iter=30
        )

        assert "statistic" in result
        assert "p_value" in result
        assert len(result["statistic"]) == n_items

    def test_dif_lord_method(self, rng):
        """Test DIF analysis with Lord's chi-square method."""
        n_per_group = 80
        n_items = 5

        data = rng.integers(0, 2, size=(n_per_group * 2, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        result = compute_dif(
            data, groups, model="2PL", method="lord", n_quadpts=11, max_iter=30
        )

        assert "statistic" in result
        assert "p_value" in result

    def test_dif_raju_method(self, rng):
        """Test DIF analysis with Raju's area method."""
        n_per_group = 80
        n_items = 5

        data = rng.integers(0, 2, size=(n_per_group * 2, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        result = compute_dif(
            data, groups, model="2PL", method="raju", n_quadpts=11, max_iter=30
        )

        assert "statistic" in result
        assert "effect_size" in result
        # Raju gives signed area in effect_size

    def test_dif_detects_biased_item(self, rng):
        """Test that DIF detects an item with large difficulty difference."""
        n_per_group = 150
        n_items = 5

        theta1 = rng.standard_normal(n_per_group)
        theta2 = rng.standard_normal(n_per_group)

        # Same difficulty except item 0 has DIF
        difficulty = np.zeros(n_items)
        difficulty_group2 = difficulty.copy()
        difficulty_group2[0] = 2.0  # Large shift for item 0

        probs1 = 1 / (1 + np.exp(-(theta1[:, None] - difficulty)))
        probs2 = 1 / (1 + np.exp(-(theta2[:, None] - difficulty_group2)))

        responses1 = (rng.random((n_per_group, n_items)) < probs1).astype(int)
        responses2 = (rng.random((n_per_group, n_items)) < probs2).astype(int)

        data = np.vstack([responses1, responses2])
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        result = compute_dif(
            data,
            groups,
            model="2PL",
            method="likelihood_ratio",
            n_quadpts=11,
            max_iter=30,
        )

        # Item 0 should have larger effect size than others
        assert result["effect_size"][0] > np.mean(result["effect_size"][1:])

    def test_ets_classification(self, rng):
        """Test ETS A/B/C classification."""
        n_per_group = 100
        n_items = 5

        data = rng.integers(0, 2, size=(n_per_group * 2, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        result = compute_dif(
            data,
            groups,
            model="2PL",
            method="likelihood_ratio",
            n_quadpts=11,
            max_iter=30,
        )

        # All classifications should be A, B, or C
        valid_classes = {"A", "B", "C"}
        for c in result["classification"]:
            assert c in valid_classes

    def test_flag_dif_items(self, rng):
        """Test flag_dif_items helper function."""
        # Create mock DIF results
        dif_results = {
            "statistic": np.array([10.0, 2.0, 15.0, 1.0]),
            "p_value": np.array([0.001, 0.15, 0.0001, 0.3]),
            "effect_size": np.array([0.8, 0.2, 1.0, 0.1]),
            "classification": np.array(["C", "A", "C", "A"]),
        }

        # Flag with default settings
        flags = flag_dif_items(dif_results)
        assert flags.dtype == bool
        assert len(flags) == 4

        assert flags[0]
        assert flags[2]
        assert not flags[1]
        assert not flags[3]

    def test_flag_dif_by_classification(self, rng):
        """Test flagging by ETS classification."""
        dif_results = {
            "statistic": np.array([10.0, 5.0, 15.0, 1.0]),
            "p_value": np.array([0.001, 0.02, 0.0001, 0.3]),
            "effect_size": np.array([0.8, 0.5, 1.0, 0.1]),
            "classification": np.array(["C", "B", "C", "A"]),
        }

        # Flag only C items
        flags_c = flag_dif_items(dif_results, classification="C")
        assert np.sum(flags_c) <= np.sum(dif_results["classification"] == "C")

        # Flag B and C items
        flags_b = flag_dif_items(dif_results, classification="B")
        assert np.sum(flags_b) <= np.sum(
            (dif_results["classification"] == "B")
            | (dif_results["classification"] == "C")
        )

    def test_requires_two_groups(self, rng):
        """Test that DIF requires exactly 2 groups."""
        n_persons = 100
        n_items = 5
        data = rng.integers(0, 2, size=(n_persons, n_items))

        # Only one group
        groups = np.zeros(n_persons)
        with pytest.raises(ValueError, match="exactly 2 groups"):
            compute_dif(data, groups)

        # Three groups
        groups = np.array([0] * 33 + [1] * 33 + [2] * 34)
        with pytest.raises(ValueError, match="exactly 2 groups"):
            compute_dif(data, groups)

    def test_string_group_labels(self, rng):
        """Test DIF with string group labels."""
        n_per_group = 50
        n_items = 5

        data = rng.integers(0, 2, size=(n_per_group * 2, n_items))
        groups = np.array(["male"] * n_per_group + ["female"] * n_per_group)

        result = compute_dif(
            data, groups, model="2PL", method="wald", n_quadpts=11, max_iter=30
        )

        assert len(result["statistic"]) == n_items

    def test_focal_group_specification(self, rng):
        """Test specifying focal group."""
        n_per_group = 50
        n_items = 5

        data = rng.integers(0, 2, size=(n_per_group * 2, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        # Specify focal group
        result = compute_dif(
            data,
            groups,
            model="2PL",
            method="wald",
            focal_group=0,
            n_quadpts=11,
            max_iter=30,
        )

        assert len(result["statistic"]) == n_items

    def test_invalid_focal_group(self, rng):
        """Test error for invalid focal group."""
        n_per_group = 50
        n_items = 5

        data = rng.integers(0, 2, size=(n_per_group * 2, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        with pytest.raises(ValueError, match="not found in groups"):
            compute_dif(data, groups, focal_group=99)

    def test_1pl_model(self, rng):
        """Test DIF with 1PL model."""
        n_per_group = 50
        n_items = 5

        data = rng.integers(0, 2, size=(n_per_group * 2, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        result = compute_dif(
            data,
            groups,
            model="1PL",
            method="likelihood_ratio",
            n_quadpts=11,
            max_iter=30,
        )

        assert len(result["statistic"]) == n_items

    def test_polytomous_grm(self, rng):
        """Test DIF with polytomous GRM model."""
        n_per_group = 60
        n_items = 4
        n_categories = 4

        data = rng.integers(0, n_categories, size=(n_per_group * 2, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        result = compute_dif(
            data,
            groups,
            model="GRM",
            n_categories=n_categories,
            method="raju",
            n_quadpts=11,
            max_iter=30,
        )

        assert len(result["statistic"]) == n_items

    def test_invalid_method(self, rng):
        """Test error for invalid DIF method."""
        n_per_group = 30
        n_items = 3

        data = rng.integers(0, 2, size=(n_per_group * 2, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        with pytest.raises(ValueError, match="Unknown DIF method"):
            compute_dif(data, groups, method="invalid")


class TestDIFIntegration:
    """Integration tests for mirt.dif() function."""

    def test_dif_function_import(self):
        """Test that dif function is importable from mirt."""
        import mirt

        assert hasattr(mirt, "dif")

    def test_dif_returns_dataframe(self, rng):
        """Test that mirt.dif() returns a DataFrame."""
        import mirt

        n_per_group = 50
        n_items = 4

        data = rng.integers(0, 2, size=(n_per_group * 2, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        result = mirt.dif(
            data, groups, model="2PL", method="wald", n_quadpts=11, max_iter=30
        )

        # Should be a pandas DataFrame (or polars if configured)
        assert hasattr(result, "columns") or hasattr(result, "schema")
