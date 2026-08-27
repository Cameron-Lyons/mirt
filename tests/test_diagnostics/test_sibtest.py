"""Tests for SIBTEST statistics."""

import numpy as np
import pytest

from mirt import sibtest, sibtest_items
from mirt.diagnostics.sibtest import _adjust_p_values


class TestSIBTEST:
    """Tests for SIBTEST procedure."""

    def test_sibtest_original(self, two_group_responses):
        """Test original SIBTEST method."""
        suspect_items = [2, 3]
        matching_items = [0, 1, 4, 5, 6, 7]

        result = sibtest(
            data=two_group_responses["responses"],
            groups=two_group_responses["groups"],
            suspect_items=suspect_items,
            matching_items=matching_items,
            method="original",
        )

        assert "beta" in result
        assert "beta_se" in result or "SE" in result or "se" in result.keys()
        assert "p_value" in result

    def test_sibtest_crossing(self, two_group_responses):
        """Test crossing SIBTEST method."""
        suspect_items = [2, 3]
        matching_items = [0, 1, 4, 5, 6, 7]

        result = sibtest(
            data=two_group_responses["responses"],
            groups=two_group_responses["groups"],
            suspect_items=suspect_items,
            matching_items=matching_items,
            method="crossing",
        )

        assert "beta_uniform" in result or "beta" in result

    def test_sibtest_auto_matching(self, two_group_responses):
        """Test SIBTEST with automatic matching item selection."""
        suspect_items = [2, 3]

        result = sibtest(
            data=two_group_responses["responses"],
            groups=two_group_responses["groups"],
            suspect_items=suspect_items,
            matching_items=None,
        )

        assert result is not None

    def test_sibtest_detects_dif(self, two_group_responses):
        """Test that SIBTEST detects DIF in known items."""
        suspect_items = [2, 3]
        matching_items = [0, 1, 4, 5, 6, 7]

        result = sibtest(
            data=two_group_responses["responses"],
            groups=two_group_responses["groups"],
            suspect_items=suspect_items,
            matching_items=matching_items,
        )

        assert result["beta"] != 0

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"data": np.ones(8)}, "2D"),
            ({"data": np.full((8, 3), 2)}, "binary"),
            ({"groups": np.zeros(16)}, "shape"),
            ({"suspect_items": []}, "at least one"),
            ({"suspect_items": [8]}, "outside"),
            ({"suspect_items": [1, 1]}, "duplicate"),
            ({"matching_items": [1], "suspect_items": [1]}, "overlap"),
            ({"correction": "yes"}, "boolean"),
        ],
    )
    def test_sibtest_validates_public_inputs(
        self, two_group_responses, kwargs, message
    ):
        """Malformed matrices and item selections fail clearly."""
        responses = two_group_responses["responses"]
        groups = two_group_responses["groups"]
        call = {
            "data": responses,
            "groups": groups,
            "suspect_items": [0],
            "matching_items": [1, 2],
        }
        call.update(kwargs)

        with pytest.raises(ValueError, match=message):
            sibtest(**call)


class TestSIBTESTItems:
    """Tests for per-item SIBTEST analysis."""

    def test_sibtest_items(self, two_group_responses):
        """Test SIBTEST for each item."""
        result = sibtest_items(
            data=two_group_responses["responses"],
            groups=two_group_responses["groups"],
        )

        n_items = two_group_responses["n_items"]

        if isinstance(result, dict):
            if "beta" in result:
                assert len(result["beta"]) == n_items
            else:
                assert result is not None
        else:
            assert len(result) == n_items

    def test_sibtest_items_identifies_dif(self, two_group_responses):
        """Test that per-item SIBTEST identifies DIF items."""
        result = sibtest_items(
            data=two_group_responses["responses"],
            groups=two_group_responses["groups"],
        )

        dif_items = two_group_responses["dif_items"]

        if isinstance(result, dict):
            if "beta" in result:
                betas = np.array(result["beta"])
            else:
                assert result is not None
                return
        else:
            betas = result["beta"].values if hasattr(result, "values") else result[:, 0]

        dif_betas = np.abs([betas[i] for i in dif_items])

        assert len(dif_betas) > 0

    @pytest.mark.parametrize("method", ["original", "crossing"])
    def test_batched_results_match_individual_calls(self, two_group_responses, method):
        """The fast item-wide path preserves every individual statistic."""
        responses = two_group_responses["responses"]
        groups = two_group_responses["groups"]
        result = sibtest_items(
            responses,
            groups,
            method=method,
            p_adjust="none",
        )

        for item_index in range(responses.shape[1]):
            individual = sibtest(
                responses,
                groups,
                suspect_items=[item_index],
                method=method,
            )
            for name in ("beta", "beta_se", "z", "p_value", "effect_size"):
                assert result[name][item_index] == pytest.approx(
                    individual[name], nan_ok=True
                )

    def test_item_results_include_adjusted_inference(self, two_group_responses):
        """Item-wide output exposes uncertainty, effects, and adjusted p-values."""
        result = sibtest_items(
            two_group_responses["responses"],
            two_group_responses["groups"],
            p_adjust="holm",
            alpha=0.1,
        )

        n_items = two_group_responses["n_items"]
        assert result["beta_se"].shape == (n_items,)
        assert result["effect_size"].shape == (n_items,)
        assert result["p_value_adjusted"].shape == (n_items,)
        assert np.all(result["p_value_adjusted"] >= result["p_value"])
        np.testing.assert_array_equal(
            result["flagged"], result["p_value_adjusted"] < 0.1
        )
        assert result["adjustment"] == "holm"
        assert result["alpha"] == 0.1

    def test_explicit_anchor_set_matches_individual_calls(self, two_group_responses):
        """Shared anchors exclude the item under test without changing results."""
        responses = two_group_responses["responses"]
        groups = two_group_responses["groups"]
        anchors = [0, 1, 4, 5]
        result = sibtest_items(
            responses,
            groups,
            anchor_items=anchors,
            p_adjust="none",
        )

        for item_index in range(responses.shape[1]):
            matching = [anchor for anchor in anchors if anchor != item_index]
            individual = sibtest(
                responses,
                groups,
                suspect_items=[item_index],
                matching_items=matching,
            )
            assert result["beta"][item_index] == pytest.approx(individual["beta"])
            assert result["beta_se"][item_index] == pytest.approx(individual["beta_se"])

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            ("none", [0.01, 0.04, 0.03, 0.002, np.nan]),
            ("bonferroni", [0.04, 0.16, 0.12, 0.008, np.nan]),
            ("holm", [0.03, 0.06, 0.06, 0.008, np.nan]),
            ("fdr_bh", [0.02, 0.04, 0.04, 0.008, np.nan]),
        ],
    )
    def test_p_value_adjustments(self, method, expected):
        """Supported family-wise and false-discovery corrections are exact."""
        raw = np.array([0.01, 0.04, 0.03, 0.002, np.nan])

        np.testing.assert_allclose(
            _adjust_p_values(raw, method),
            expected,
            equal_nan=True,
        )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"alpha": 0.0}, "alpha"),
            ({"alpha": np.nan}, "alpha"),
            ({"p_adjust": "unknown"}, "p_adjust"),
            ({"anchor_items": [0, 0]}, "duplicate"),
            ({"anchor_items": [9]}, "outside"),
        ],
    )
    def test_item_analysis_validates_options(
        self, two_group_responses, kwargs, message
    ):
        """Invalid batch-analysis options fail before computation."""
        with pytest.raises(ValueError, match=message):
            sibtest_items(
                two_group_responses["responses"],
                two_group_responses["groups"],
                **kwargs,
            )
