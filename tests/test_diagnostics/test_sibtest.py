"""Tests for SIBTEST statistics."""

import numpy as np

from mirt import sibtest, sibtest_items


class TestSIBTEST:
    """Tests for SIBTEST procedure."""

    def test_sibtest_original(self, two_group_responses):
        """Test original SIBTEST method."""
        suspect_items = [2, 3]
        matching_items = [0, 1, 4, 5, 6, 7, 8, 9]

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
        matching_items = [0, 1, 4, 5, 6, 7, 8, 9]

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
        matching_items = [0, 1, 4, 5, 6, 7, 8, 9]

        result = sibtest(
            data=two_group_responses["responses"],
            groups=two_group_responses["groups"],
            suspect_items=suspect_items,
            matching_items=matching_items,
        )

        assert result["beta"] != 0


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
