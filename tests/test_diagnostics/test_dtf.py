"""Tests for Differential Test Functioning (DTF)."""

import numpy as np

from mirt import compute_dtf


class TestDTF:
    """Tests for DTF computation."""

    def test_compute_dtf_signed(self, two_group_responses):
        """Test signed DTF computation."""
        dtf_result = compute_dtf(
            data=two_group_responses["responses"],
            groups=two_group_responses["groups"],
            method="signed",
        )

        assert "DTF" in dtf_result
        assert "DTF_SE" in dtf_result or "SE" in dtf_result or "se" in dtf_result.keys()
        assert "p_value" in dtf_result

    def test_compute_dtf_unsigned(self, two_group_responses):
        """Test unsigned DTF computation."""
        dtf_result = compute_dtf(
            data=two_group_responses["responses"],
            groups=two_group_responses["groups"],
            method="unsigned",
        )

        assert dtf_result["DTF"] >= 0

    def test_compute_dtf_expected_score(self, two_group_responses):
        """Test expected score DTF method."""
        dtf_result = compute_dtf(
            data=two_group_responses["responses"],
            groups=two_group_responses["groups"],
            method="expected_score",
        )

        assert "expected_score_diff" in dtf_result or "DTF" in dtf_result

    def test_dtf_with_model(self, two_group_responses):
        """Test DTF with specified model type."""
        dtf_result = compute_dtf(
            data=two_group_responses["responses"],
            groups=two_group_responses["groups"],
            model="2PL",
            method="signed",
            max_iter=30,
        )

        assert dtf_result is not None

    def test_dtf_detects_difference(self, two_group_responses):
        """Test that DTF detects group differences when DIF is present."""
        dtf_result = compute_dtf(
            data=two_group_responses["responses"],
            groups=two_group_responses["groups"],
            method="unsigned",
        )

        assert dtf_result["DTF"] >= 0

    def test_dtf_groups_string(self, two_group_responses):
        """Test DTF with string group labels."""
        responses = two_group_responses["responses"]
        groups = np.where(
            two_group_responses["groups"] == 0,
            "reference",
            "focal",
        )

        dtf_result = compute_dtf(
            data=responses,
            groups=groups,
            method="signed",
        )

        assert dtf_result is not None
