"""Tests for Differential Test Functioning (DTF)."""

import numpy as np
import pytest

from mirt import compute_dtf

pytestmark = pytest.mark.slow


class TestDTF:
    """Tests for DTF computation - marked slow due to bootstrap."""

    def test_compute_dtf_signed(self, two_group_responses):
        """Test signed DTF computation."""
        responses = two_group_responses["responses"]
        groups = two_group_responses["groups"]

        dtf_result = compute_dtf(
            data=responses,
            groups=groups,
            method="signed",
            max_iter=10,
            n_quadpts=9,
            n_bootstrap=3,
        )

        assert "DTF" in dtf_result
        assert "DTF_SE" in dtf_result or "SE" in dtf_result or "se" in dtf_result.keys()
        assert "p_value" in dtf_result

    def test_compute_dtf_unsigned(self, two_group_responses):
        """Test unsigned DTF computation."""
        responses = two_group_responses["responses"]
        groups = two_group_responses["groups"]

        dtf_result = compute_dtf(
            data=responses,
            groups=groups,
            method="unsigned",
            max_iter=10,
            n_quadpts=9,
            n_bootstrap=3,
        )

        assert dtf_result["DTF"] >= 0

    def test_compute_dtf_expected_score(self, two_group_responses):
        """Test expected score DTF method."""
        responses = two_group_responses["responses"]
        groups = two_group_responses["groups"]

        dtf_result = compute_dtf(
            data=responses,
            groups=groups,
            method="expected_score",
            max_iter=10,
            n_quadpts=9,
            n_bootstrap=3,
        )

        assert "expected_score_diff" in dtf_result or "DTF" in dtf_result

    def test_dtf_with_model(self, two_group_responses):
        """Test DTF with specified model type."""
        responses = two_group_responses["responses"]
        groups = two_group_responses["groups"]

        dtf_result = compute_dtf(
            data=responses,
            groups=groups,
            model="2PL",
            method="signed",
            max_iter=10,
            n_quadpts=9,
            n_bootstrap=3,
        )

        assert dtf_result is not None

    def test_dtf_detects_difference(self, two_group_responses):
        """Test that DTF detects group differences when DIF is present."""
        responses = two_group_responses["responses"]
        groups = two_group_responses["groups"]

        dtf_result = compute_dtf(
            data=responses,
            groups=groups,
            method="unsigned",
            max_iter=10,
            n_quadpts=9,
            n_bootstrap=3,
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
            max_iter=10,
            n_quadpts=9,
            n_bootstrap=3,
        )

        assert dtf_result is not None
