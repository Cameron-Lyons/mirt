"""Analytical and repeated-sampling checks of SIBTEST inference."""

import numpy as np
import pytest
from scipy import stats
from scipy.special import expit

from mirt import sibtest, sibtest_items


def _cell_responses(cells, n_matching=4):
    """Construct complete binary data with independently specified score cells."""
    rows, groups = [], []
    for group, group_cells in enumerate(cells):
        for score, (count, successes) in enumerate(group_cells):
            block = np.zeros((count, n_matching + 1), dtype=int)
            block[:, :score] = 1
            block[:successes, -1] = 1
            rows.append(block)
            groups.extend([group] * count)
    return np.vstack(rows), np.asarray(groups)


def _mean_variance(count, successes):
    """Unbiased Bernoulli sample variance divided by its sample size."""
    proportion = successes / count
    return proportion * (1 - proportion) / (count - 1)


def test_pooled_stratum_effect_and_uncertainty_have_analytical_values():
    cells = [[(10, 8), (40, 20)], [(30, 9), (20, 6)]]
    data, groups = _cell_responses(cells)
    result = sibtest(data, groups, [4], correction=False)
    expected_beta = 0.4 * (0.8 - 0.3) + 0.6 * (0.5 - 0.3)
    expected_variance = 0.4**2 * (
        _mean_variance(10, 8) + _mean_variance(30, 9)
    ) + 0.6**2 * (_mean_variance(40, 20) + _mean_variance(20, 6))

    assert result["beta"] == pytest.approx(expected_beta)
    assert result["beta_se"] == pytest.approx(np.sqrt(expected_variance))
    assert result["chi2"] == pytest.approx(expected_beta**2 / expected_variance)
    assert result["p_value"] == pytest.approx(
        2 * stats.norm.sf(abs(expected_beta) / np.sqrt(expected_variance))
    )
    assert result["df"] == 1
    assert result["n_strata"] == 2


@pytest.mark.parametrize("reference_successes", [50, 70])
def test_equal_effects_across_strata_retain_sampling_uncertainty(reference_successes):
    data, groups = _cell_responses([[(100, reference_successes)] * 3, [(100, 50)] * 3])
    result = sibtest(data, groups, [4], correction=False)
    expected_variance = (
        _mean_variance(100, reference_successes) + _mean_variance(100, 50)
    ) / 3

    assert result["beta"] == pytest.approx(reference_successes / 100 - 0.5)
    assert result["beta_se"] == pytest.approx(np.sqrt(expected_variance))
    assert np.isfinite(result["p_value"])
    if reference_successes == 50:
        assert result["p_value"] == 1.0


@pytest.mark.parametrize("successes", [[10, 30, 50, 70, 90], [48, 49, 50, 51, 52]])
def test_crossing_uses_estimated_location_and_independent_region_chi_square(successes):
    cells = [
        [(100, success) for success in successes],
        [(100, 100 - success) for success in successes],
    ]
    data, groups = _cell_responses(cells)
    result = sibtest(data, groups, [4], method="crossing", correction=False)
    differences = (2 * np.asarray(successes) - 100) / 100
    mean_variances = np.array(
        [2 * _mean_variance(100, success) for success in successes]
    )
    left_effect, right_effect = differences[:3].sum() / 5, differences[3:].sum() / 5
    left_variance = mean_variances[:3].sum() / 25
    right_variance = mean_variances[3:].sum() / 25
    chi2 = left_effect**2 / left_variance + right_effect**2 / right_variance

    assert result["crossing_point"] == pytest.approx(2.0)
    assert result["beta"] == pytest.approx(abs(left_effect - right_effect))
    assert result["beta_se"] == pytest.approx(np.sqrt(left_variance + right_variance))
    assert result["chi2"] == pytest.approx(chi2)
    assert result["df"] == 2
    assert result["p_value"] == pytest.approx(stats.chi2.sf(chi2, 2))
    assert (result["p_value"] < 0.05) == (successes[0] == 10)


def test_crossing_without_a_crossing_reduces_to_one_region_test():
    data, groups = _cell_responses([[(100, 70)] * 4, [(100, 50)] * 4])
    uniform = sibtest(data, groups, [4], correction=False)
    crossing = sibtest(data, groups, [4], correction=False, method="crossing")

    for name in ("beta", "beta_se", "chi2", "p_value"):
        assert crossing[name] == pytest.approx(uniform[name])
    assert crossing["df"] == 1
    assert np.isnan(crossing["crossing_point"])


def test_true_score_correction_matches_manual_reliability_extrapolation():
    cells = [
        [(20, 2), (30, 8), (40, 20), (30, 22), (20, 18)],
        [(10, 1), (20, 4), (30, 12), (40, 24), (50, 40)],
    ]
    data, groups = _cell_responses(cells)
    means, counts, corrected = [], [], []
    true_scores = []
    reliabilities = []
    for group in (0, 1):
        anchors = data[groups == group, :4]
        scores = anchors.sum(axis=1)
        reliability = (
            4 / 3 * (1 - np.var(anchors, axis=0, ddof=1).sum() / np.var(scores, ddof=1))
        )
        reliabilities.append(reliability)
        means.append(np.array([success / count for count, success in cells[group]]))
        counts.append(np.array([count for count, _ in cells[group]]))
        true_scores.append(
            scores.mean() + reliability * (np.arange(1, 4) - scores.mean())
        )
    common_true_scores = (true_scores[0] + true_scores[1]) / 2
    for group in (0, 1):
        slopes = (means[group][2:] - means[group][:-2]) / (2 * reliabilities[group])
        corrected.append(
            means[group][1:4] + slopes * (common_true_scores - true_scores[group])
        )
    weights = (counts[0] + counts[1])[1:4]
    weights = weights / weights.sum()
    expected_beta = np.dot(weights, corrected[0] - corrected[1])
    result = sibtest(data, groups, [4])

    assert result["beta"] == pytest.approx(expected_beta)
    assert result["n_strata"] == 3
    assert np.isfinite(result["p_value"])
    assert result["beta"] != pytest.approx(
        sibtest(data, groups, [4], correction=False)["beta"]
    )


@pytest.mark.parametrize("method", ["original", "crossing"])
def test_null_repeated_sampling_controls_false_positive_rate(method):
    rng = np.random.default_rng(5829)
    p_values = []
    for _ in range(200):
        successes = rng.binomial(100, np.linspace(0.2, 0.8, 5), size=(2, 5))
        cells = [[(100, int(success)) for success in group] for group in successes]
        data, groups = _cell_responses(cells)
        p_values.append(
            sibtest(data, groups, [4], correction=False, method=method)["p_value"]
        )

    assert np.all(np.isfinite(p_values))
    # Wide deterministic tolerance around nominal 5%; old SE formulas reject
    # almost all null samples, particularly for the crossing method.
    assert 0.005 <= np.mean(np.asarray(p_values) < 0.05) <= 0.12


def test_corrected_irt_null_with_group_impact_does_not_create_systematic_dif():
    rng = np.random.default_rng(4431)
    groups = np.repeat([0, 1], 600)
    p_values, effects = [], []
    for _ in range(80):
        theta = rng.normal(size=groups.size) + groups * 0.7
        probabilities = expit(theta[:, None] - np.linspace(-1.8, 1.8, 17))
        data = (rng.random(probabilities.shape) < probabilities).astype(int)
        result = sibtest(data, groups, [8], min_cell_size=5)
        p_values.append(result["p_value"])
        effects.append(result["beta"])

    assert np.all(np.isfinite(p_values))
    assert abs(np.mean(effects)) < 0.04
    assert np.mean(np.asarray(p_values) < 0.05) <= 0.15


def test_focal_label_reversal_preserves_inference_and_reverses_uniform_effect():
    data, groups = _cell_responses([[(100, 70)] * 3, [(100, 50)] * 3])
    ordinary = sibtest(data, groups, [4], correction=False)
    reversed_groups = sibtest(data, groups, [4], correction=False, focal_group=0)
    assert reversed_groups["beta"] == pytest.approx(-ordinary["beta"])
    assert reversed_groups["p_value"] == pytest.approx(ordinary["p_value"])


@pytest.mark.parametrize("method", ["original", "crossing"])
def test_batched_inference_matches_individual_with_correction_and_sparse_cells(method):
    data, groups = _cell_responses(
        [
            [(20, 2), (30, 8), (40, 20), (30, 22), (20, 18)],
            [(10, 1), (20, 4), (30, 12), (40, 24), (50, 40)],
        ]
    )
    result = sibtest_items(data, groups, method=method, min_cell_size=5, focal_group=0)
    for item in range(data.shape[1]):
        individual = sibtest(
            data, groups, [item], method=method, min_cell_size=5, focal_group=0
        )
        for name in (
            "beta",
            "beta_se",
            "chi2",
            "df",
            "p_value",
            "crossing_point",
            "n_strata",
        ):
            assert result[name][item] == pytest.approx(individual[name], nan_ok=True)


@pytest.mark.parametrize("correction", [False, True])
@pytest.mark.parametrize("method", ["original", "crossing"])
def test_degenerate_groups_produce_unestimable_inference(correction, method):
    result = sibtest(
        np.zeros((20, 3), dtype=int),
        np.repeat([0, 1], 10),
        [2],
        correction=correction,
        method=method,
    )
    assert np.isnan(result["p_value"])
    assert np.isnan(result["z"])
    assert result["df"] == 0


def test_sparse_strata_are_excluded_by_the_requested_minimum():
    data, groups = _cell_responses([[(10, 8), (40, 20)], [(3, 1), (20, 6)]])
    result = sibtest(data, groups, [4], correction=False, min_cell_size=5)
    assert result["n_strata"] == 1
    assert result["beta"] == pytest.approx(0.2)


@pytest.mark.parametrize("minimum", [1, 2.5, True, np.nan])
def test_invalid_minimum_cell_size_is_rejected(minimum):
    data, groups = _cell_responses([[(10, 5)], [(10, 5)]])
    with pytest.raises(ValueError, match="min_cell_size"):
        sibtest(data, groups, [4], min_cell_size=minimum)


@pytest.mark.parametrize("missing", [-1, np.nan, np.inf])
def test_missing_matching_or_suspect_responses_fail_clearly(missing):
    data, groups = _cell_responses([[(10, 5)], [(10, 5)]])
    for item in (0, 4):
        incomplete = data.astype(float)
        incomplete[0, item] = missing
        with pytest.raises(ValueError, match="binary|finite"):
            sibtest(incomplete, groups, [4])


def test_correction_requires_estimable_matching_reliability():
    # All four anchor patterns appear equally often, so alpha/KR-20 is zero.
    anchors = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (20, 1))
    suspect = np.tile([0, 1, 0, 1], 20)
    data = np.tile(np.column_stack((anchors, suspect)), (2, 1))
    groups = np.repeat([0, 1], 80)
    corrected = sibtest(data, groups, [2])
    uncorrected = sibtest(data, groups, [2], correction=False)
    assert corrected["n_strata"] == 0
    assert corrected["df"] == 0
    assert np.isnan(corrected["p_value"])
    assert uncorrected["p_value"] == 1.0


def test_correction_with_one_anchor_is_unestimable():
    data = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (20, 1))
    groups = np.repeat([0, 1], 40)
    corrected = sibtest(data, groups, [1])
    assert corrected["n_strata"] == corrected["df"] == 0
    assert np.isnan(corrected["p_value"])


def test_correction_does_not_fill_absent_neighboring_score_cells_with_zero_means():
    data, groups = _cell_responses([[(40, 20), (0, 0), (40, 20), (0, 0), (40, 20)]] * 2)
    corrected = sibtest(data, groups, [4])
    assert corrected["n_strata"] == corrected["df"] == 0
    assert np.isnan(corrected["p_value"])
    assert sibtest(data, groups, [4], correction=False)["p_value"] == 1.0


def test_crossing_with_one_usable_stratum_uses_one_degree_of_freedom():
    data, groups = _cell_responses([[(10, 8), (40, 20)], [(3, 1), (20, 6)]])
    result = sibtest(
        data, groups, [4], correction=False, method="crossing", min_cell_size=5
    )
    assert result["n_strata"] == result["df"] == 1
    assert np.isnan(result["crossing_point"])
    assert np.isfinite(result["p_value"])


def test_both_groups_with_disjoint_matching_scores_are_unestimable():
    data = np.vstack(
        (np.tile([[0, 0], [0, 1]], (10, 1)), np.tile([[1, 0], [1, 1]], (10, 1)))
    )
    result = sibtest(data, np.repeat([0, 1], 20), [1], correction=False)
    assert result["n_strata"] == result["df"] == 0
    assert np.isnan(result["beta"])
    assert np.isnan(result["p_value"])


@pytest.mark.parametrize("missing", [None, np.nan])
def test_missing_group_labels_are_rejected(missing):
    data, groups = _cell_responses([[(10, 5)], [(10, 5)]])
    incomplete = groups.astype(object)
    incomplete[0] = missing
    with pytest.raises(ValueError, match="missing labels"):
        sibtest(data, incomplete, [4])
