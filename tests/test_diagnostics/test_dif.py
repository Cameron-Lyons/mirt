"""Tests for Differential Item Functioning (DIF) analysis."""

import numpy as np
import pytest

from mirt.diagnostics.dif import compute_dif, flag_dif_items

try:
    import pandas as pandas

    HAS_DATAFRAME = True
except ImportError:
    try:
        import polars as polars

        HAS_DATAFRAME = True
    except ImportError:
        HAS_DATAFRAME = False


class TestDIF:
    """Tests for DIF detection functions."""

    def test_basic_dif_likelihood_ratio(self, rng):
        """Test basic DIF analysis with likelihood ratio method."""
        n_per_group = 100
        n_items = 5

        theta1 = rng.standard_normal(n_per_group)
        theta2 = rng.standard_normal(n_per_group)

        difficulty = rng.normal(0, 1, n_items)

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
            p_adjust="holm",
        )

        assert "statistic" in result
        assert "p_value" in result
        assert "p_value_adjusted" in result
        assert "effect_size" in result
        assert "classification" in result
        assert "adjustment" in result

        assert len(result["statistic"]) == n_items
        assert len(result["p_value"]) == n_items
        assert np.all(result["p_value"] >= 0) and np.all(result["p_value"] <= 1)
        assert np.all(result["p_value_adjusted"] >= result["p_value"])
        assert np.all(result["adjustment"] == "holm")
        expected_classes = np.where(
            (result["p_value_adjusted"] > 0.05)
            | (np.abs(result["effect_size"]) < 0.426),
            "A",
            np.where(np.abs(result["effect_size"]) < 0.638, "B", "C"),
        )
        np.testing.assert_array_equal(result["classification"], expected_classes)

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

    def test_dif_detects_biased_item(self, rng):
        """Test that DIF detects an item with large difficulty difference."""
        n_per_group = 150
        n_items = 5

        theta1 = rng.standard_normal(n_per_group)
        theta2 = rng.standard_normal(n_per_group)

        difficulty = np.zeros(n_items)
        difficulty_group2 = difficulty.copy()
        difficulty_group2[0] = 2.0

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

        valid_classes = {"A", "B", "C"}
        for c in result["classification"]:
            assert c in valid_classes

    def test_flag_dif_items(self, rng):
        """Test flag_dif_items helper function."""
        dif_results = {
            "statistic": np.array([10.0, 2.0, 15.0, 1.0]),
            "p_value": np.array([0.001, 0.15, 0.0001, 0.3]),
            "effect_size": np.array([0.8, 0.2, 1.0, 0.1]),
            "classification": np.array(["C", "A", "C", "A"]),
        }

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

        flags_c = flag_dif_items(dif_results, classification="C")
        assert np.sum(flags_c) <= np.sum(dif_results["classification"] == "C")

        flags_b = flag_dif_items(dif_results, classification="B")
        assert np.sum(flags_b) <= np.sum(
            (dif_results["classification"] == "B")
            | (dif_results["classification"] == "C")
        )

    def test_flag_dif_applies_multiple_testing_adjustment(self):
        dif_results = {
            "p_value": np.array([0.01, 0.02, 0.04, 0.20]),
            "effect_size": np.full(4, 0.8),
            "classification": np.full(4, "C"),
        }

        unadjusted = flag_dif_items(dif_results)
        adjusted = flag_dif_items(dif_results, p_adjust="bonferroni")

        np.testing.assert_array_equal(unadjusted, [True, True, True, False])
        np.testing.assert_array_equal(adjusted, [True, False, False, False])

    def test_flag_dif_classifies_signed_effects_by_magnitude(self):
        dif_results = {
            "p_value": np.array([0.001, 0.001]),
            "effect_size": np.array([-0.8, -0.5]),
            "classification": np.array(["C", "B"]),
        }

        np.testing.assert_array_equal(
            flag_dif_items(dif_results, classification="B"),
            [True, True],
        )
        np.testing.assert_array_equal(
            flag_dif_items(dif_results, classification="C"),
            [True, False],
        )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"alpha": 0.0}, "alpha"),
            ({"alpha": "invalid"}, "alpha"),
            ({"min_effect_size": -0.1}, "min_effect_size"),
            ({"classification": "A"}, "classification"),
            ({"p_adjust": "unknown"}, "p_adjust"),
        ],
    )
    def test_flag_dif_validates_options(self, kwargs, message):
        dif_results = {
            "p_value": np.array([0.01, 0.20]),
            "effect_size": np.array([0.8, 0.2]),
            "classification": np.array(["C", "A"]),
        }

        with pytest.raises(ValueError, match=message):
            flag_dif_items(dif_results, **kwargs)

    def test_compute_dif_rejects_adjustment_before_fitting(self):
        with pytest.raises(ValueError, match="p_adjust"):
            compute_dif(
                np.zeros((1, 1), dtype=int),
                np.zeros(1, dtype=int),
                p_adjust="unknown",  # type: ignore[arg-type]
            )

    def test_requires_two_groups(self, rng):
        """Test that DIF requires exactly 2 groups."""
        n_persons = 100
        n_items = 5
        data = rng.integers(0, 2, size=(n_persons, n_items))

        groups = np.zeros(n_persons)
        with pytest.raises(ValueError, match="Expected 2 groups"):
            compute_dif(data, groups)

        groups = np.array([0] * 33 + [1] * 33 + [2] * 34)
        with pytest.raises(ValueError, match="Expected 2 groups"):
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

    @pytest.mark.skipif(not HAS_DATAFRAME, reason="Requires pandas or polars")
    def test_dif_returns_dataframe(self, rng):
        """Test that mirt.dif() returns a DataFrame."""
        import mirt

        n_per_group = 50
        n_items = 4

        data = rng.integers(0, 2, size=(n_per_group * 2, n_items))
        groups = np.array([0] * n_per_group + [1] * n_per_group)

        result = mirt.dif(
            data,
            groups,
            model="2PL",
            method="wald",
            n_quadpts=11,
            max_iter=30,
            p_adjust="fdr_bh",
        )

        assert hasattr(result, "columns") or hasattr(result, "schema")
        columns = (
            result.columns if hasattr(result, "columns") else result.schema.names()
        )
        assert "p_value_adjusted" in columns
        assert "adjustment" in columns
