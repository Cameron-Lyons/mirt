"""Tests for reliable and clinically significant change utilities."""

import numpy as np
import pytest

from mirt import RCI, clinical_significance
from mirt.exceptions import MirtValidationError


class TestReliableChangeIndex:
    def test_uses_person_specific_measurement_error(self):
        result = RCI(
            [0.0, 0.0],
            [0.5, 0.5],
            sem_pre=[0.1, 1.0],
        )

        expected_se = np.sqrt(2.0) * np.array([0.1, 1.0])
        np.testing.assert_allclose(result.se_diff_by_person, expected_se)
        np.testing.assert_allclose(result.rci, 0.5 / expected_se)
        np.testing.assert_array_equal(result.significant, [True, False])
        np.testing.assert_array_equal(result.direction, ["improved", "unchanged"])
        assert result.se_diff == pytest.approx(expected_se.mean())

    def test_derives_sem_and_change_intervals_from_reliability(self):
        result = RCI(
            [0.0, 1.0],
            [1.0, 0.5],
            reliability=0.84,
            sd_theta=2.0,
            alpha=0.05,
        )

        expected_sem = 2.0 * np.sqrt(0.16)
        expected_se = np.sqrt(2.0) * expected_sem
        np.testing.assert_allclose(result.se_diff_by_person, expected_se)
        np.testing.assert_allclose(result.change, [1.0, -0.5])
        np.testing.assert_allclose(
            result.ci_lower,
            result.change - result.critical_value * expected_se,
        )
        np.testing.assert_allclose(
            result.ci_upper,
            result.change + result.critical_value * expected_se,
        )

    def test_iverson_combines_pre_and_post_error(self):
        result = RCI(
            [0.0, 0.0],
            [1.0, 1.0],
            sem_pre=[0.3, 0.4],
            sem_post=[0.4, 0.3],
            method="iverson",
        )

        np.testing.assert_allclose(result.se_diff_by_person, [0.5, 0.5])

    def test_jacobson_warns_that_post_test_error_is_ignored(self):
        pre = np.array([0.0, -1.0, 0.5])
        post = np.array([0.8, 0.2, 0.4])

        with pytest.warns(UserWarning, match="ignores sem_post"):
            result = RCI(pre, post, sem_pre=[0.3, 0.45, 0.3], sem_post=0.6)

        np.testing.assert_allclose(
            result.se_diff_by_person, np.sqrt(2.0) * np.array([0.3, 0.45, 0.3])
        )

    def test_iverson_derives_post_error_from_post_reliability(self):
        result = RCI(
            [0.0, 0.0],
            [1.0, 1.0],
            reliability=0.84,
            reliability_post=0.75,
            sd_theta=2.0,
            method="iverson",
        )

        expected = np.hypot(2.0 * np.sqrt(0.16), 2.0 * np.sqrt(0.25))
        np.testing.assert_allclose(result.se_diff_by_person, [expected, expected])

    def test_hageman_arrindell_regresses_change_by_difference_reliability(self):
        # Hageman & Arrindell (1993) RC_ID with the reliability of the
        # difference scores computed from the sample moments.
        pre = np.array([0.0, -1.0, 0.5, 1.2, -0.3])
        post = np.array([0.8, 0.2, 0.4, 1.0, 0.6])
        sem_pre = np.array([0.30, 0.45, 0.30, 0.35, 0.4])
        sem_post = np.array([0.60, 0.25, 0.30, 0.3, 0.35])
        sd_pre, sd_post = np.std(pre, ddof=1), np.std(post, ddof=1)
        r_pp = np.corrcoef(pre, post)[0, 1]
        cross = 2.0 * sd_pre * sd_post * r_pp
        r_dd = (sd_pre**2 * 0.9 + sd_post**2 * 0.8 - cross) / (
            sd_pre**2 + sd_post**2 - cross
        )
        estimate = (post - pre) * r_dd + (post.mean() - pre.mean()) * (1 - r_dd)
        expected_se = np.sqrt(r_dd * (sem_pre**2 + sem_post**2))

        result = RCI(
            pre,
            post,
            sem_pre=sem_pre,
            sem_post=sem_post,
            reliability=0.9,
            reliability_post=0.8,
            method="hageman",
        )
        iverson = RCI(pre, post, sem_pre=sem_pre, sem_post=sem_post, method="iverson")

        np.testing.assert_allclose(result.se_diff_by_person, expected_se)
        np.testing.assert_allclose(result.rci, estimate / expected_se)
        np.testing.assert_allclose(result.change, post - pre)
        np.testing.assert_allclose(
            result.ci_lower, estimate - result.critical_value * expected_se
        )
        assert not np.allclose(result.rci, iverson.rci)

    def test_hageman_arrindell_requires_reliability(self):
        with pytest.raises(MirtValidationError, match="pre-test reliability"):
            RCI(
                [0.0, -1.0, 0.5],
                [0.8, 0.2, 0.4],
                sem_pre=0.3,
                sem_post=0.3,
                method="hageman",
            )

    def test_hageman_arrindell_rejects_unreliable_differences(self):
        # Perfectly correlated pre and post scores leave no reliable
        # individual differences in change.
        with pytest.raises(MirtValidationError, match="no positive reliability"):
            RCI(
                [0.0, 1.0, 2.0],
                [0.5, 1.5, 2.5],
                reliability=0.8,
                method="hageman",
            )

    def test_lower_scores_can_represent_improvement(self):
        result = RCI(
            [2.0, 0.0],
            [0.0, 2.0],
            sem_pre=0.1,
            higher_is_better=False,
        )

        np.testing.assert_array_equal(result.direction, ["improved", "declined"])
        assert result.higher_is_better is False

    def test_result_summary_counts_classifications(self):
        result = RCI(
            [0.0, 0.0, 0.0],
            [1.0, -1.0, 0.01],
            sem_pre=0.1,
        )

        assert result.n_persons == 3
        assert result.summary() == {
            "n_persons": 3,
            "significant": 2,
            "improved": 1,
            "declined": 1,
            "unchanged": 1,
        }


class TestClinicalSignificance:
    def test_higher_is_better_recovery(self):
        result = clinical_significance(
            [-2.0, -2.0, 0.0, 0.0],
            [2.0, -0.5, 0.01, -2.0],
            cutoff=0.0,
            reliability=0.99,
        )

        np.testing.assert_array_equal(result["recovered"], [True, False, False, False])
        np.testing.assert_array_equal(result["improved"], [False, True, False, False])
        np.testing.assert_array_equal(result["unchanged"], [False, False, True, False])
        np.testing.assert_array_equal(
            result["deteriorated"], [False, False, False, True]
        )

    def test_lower_is_better_recovery(self):
        result = clinical_significance(
            [2.0, 2.0, 0.0, 0.0],
            [0.0, 1.5, -0.01, 2.0],
            cutoff=1.0,
            reliability=0.99,
            higher_is_better=False,
        )

        np.testing.assert_array_equal(result["recovered"], [True, False, False, False])
        np.testing.assert_array_equal(result["improved"], [False, True, False, False])
        np.testing.assert_array_equal(result["unchanged"], [False, False, True, False])
        np.testing.assert_array_equal(
            result["deteriorated"], [False, False, False, True]
        )

    def test_classifications_partition_every_person(self):
        result = clinical_significance(
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [2.0, -0.5, 0.0, 0.5, -2.0],
            cutoff=0.0,
            reliability=0.9,
        )

        membership = np.stack(list(result.values())).sum(axis=0)
        np.testing.assert_array_equal(membership, np.ones(5))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"theta_pre": [], "theta_post": [], "sem_pre": 0.1}, "theta_pre"),
        (
            {"theta_pre": [0.0], "theta_post": [0.0, 1.0], "sem_pre": 0.1},
            "same length",
        ),
        ({"theta_pre": [np.nan], "theta_post": [0.0], "sem_pre": 0.1}, "finite"),
        ({"theta_pre": [0.0], "theta_post": [1.0]}, "sem_pre or reliability"),
        (
            {"theta_pre": [0.0], "theta_post": [1.0], "reliability": -0.1},
            "reliability",
        ),
        (
            {"theta_pre": [0.0], "theta_post": [1.0], "reliability": 1.0},
            "reliability",
        ),
        (
            {
                "theta_pre": [0.0],
                "theta_post": [1.0],
                "reliability": 0.8,
                "sd_theta": 0,
            },
            "sd_theta",
        ),
        (
            {"theta_pre": [0.0], "theta_post": [1.0], "sem_pre": 0.0},
            "positive",
        ),
        (
            {
                "theta_pre": [0.0, 1.0],
                "theta_post": [1.0, 2.0],
                "sem_pre": [0.1, 0.2, 0.3],
            },
            "number of persons",
        ),
        (
            {"theta_pre": [0.0], "theta_post": [1.0], "sem_pre": 0.1, "alpha": 0.0},
            "alpha",
        ),
        (
            {
                "theta_pre": [0.0],
                "theta_post": [1.0],
                "sem_pre": 0.1,
                "method": "unknown",
            },
            "method",
        ),
        (
            {
                "theta_pre": [0.0],
                "theta_post": [1.0],
                "sem_pre": 0.1,
                "higher_is_better": 1,
            },
            "boolean",
        ),
    ],
)
def test_rci_rejects_invalid_inputs(kwargs, match):
    with pytest.raises(MirtValidationError, match=match):
        RCI(**kwargs)


@pytest.mark.parametrize("cutoff", [np.nan, np.inf, True])
def test_clinical_significance_rejects_invalid_cutoff(cutoff):
    with pytest.raises(MirtValidationError, match="cutoff"):
        clinical_significance(
            [0.0],
            [1.0],
            cutoff=cutoff,
            reliability=0.8,
        )
