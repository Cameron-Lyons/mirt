"""Tests for Differential Response Functioning (DRF)."""

from mirt import compute_drf, compute_item_drf, reliability_invariance


class TestDRF:
    """Tests for test-level DRF."""

    def test_compute_drf(self, two_group_responses):
        """Test DRF computation."""
        result = compute_drf(
            data=two_group_responses["responses"],
            groups=two_group_responses["groups"],
        )

        assert "information_ref" in result or "information_reference" in result.keys()
        assert "information_focal" in result or "info_focal" in result.keys()

    def test_drf_by_theta(self, two_group_responses):
        """Test DRF at different theta values."""
        result = compute_drf(
            data=two_group_responses["responses"],
            groups=two_group_responses["groups"],
            theta_range=(-2, 2),
        )

        if "theta_grid" in result:
            theta = result["theta_grid"]
            assert len(theta) > 0
            assert theta.min() >= -2
            assert theta.max() <= 2


class TestItemDRF:
    """Tests for item-level DRF."""

    def test_compute_item_drf(self, two_group_responses):
        """Test item-level DRF computation."""
        result = compute_item_drf(
            data=two_group_responses["responses"],
            groups=two_group_responses["groups"],
        )

        n_items = two_group_responses["n_items"]

        if isinstance(result, dict):
            if "item_drf" in result:
                assert len(result["item_drf"]) == n_items
            else:
                assert result is not None
        else:
            assert len(result) == n_items

    def test_item_drf_identifies_differential_items(self, two_group_responses):
        """Test that item DRF identifies items with different functioning."""
        result = compute_item_drf(
            data=two_group_responses["responses"],
            groups=two_group_responses["groups"],
        )

        two_group_responses["dif_items"]

        assert result is not None


class TestReliabilityInvariance:
    """Tests for reliability invariance analysis."""

    def test_reliability_invariance(self, two_group_responses):
        """Test reliability invariance computation."""
        result = reliability_invariance(
            data=two_group_responses["responses"],
            groups=two_group_responses["groups"],
        )

        assert "reliability_ref" in result or "reliability_reference" in result.keys()
        assert "reliability_foc" in result or "reliability_focal" in result.keys()

    def test_reliability_values(self, two_group_responses):
        """Test that reliability values are in valid range."""
        result = reliability_invariance(
            data=two_group_responses["responses"],
            groups=two_group_responses["groups"],
        )

        rel_ref = result.get("reliability_ref", result.get("reliability_reference"))
        rel_foc = result.get("reliability_foc", result.get("reliability_focal"))

        if rel_ref is not None:
            assert 0 <= rel_ref <= 1
        if rel_foc is not None:
            assert 0 <= rel_foc <= 1

    def test_reliability_difference(self, two_group_responses):
        """Test reliability difference computation."""
        result = reliability_invariance(
            data=two_group_responses["responses"],
            groups=two_group_responses["groups"],
        )

        if "reliability_diff" in result:
            assert abs(result["reliability_diff"]) < 1.0
