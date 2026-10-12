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


@pytest.mark.parametrize(
    ("missing", "message"),
    [(-1, "na_rm=True"), (np.nan, "na_rm=True"), (np.inf, "finite")],
)
def test_missing_matching_or_suspect_responses_fail_clearly(missing, message):
    data, groups = _cell_responses([[(10, 5)], [(10, 5)]])
    for item in (0, 4):
        incomplete = data.astype(float)
        incomplete[0, item] = missing
        with pytest.raises(ValueError, match=message):
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


def _lcg_uniforms(shape, seed):
    """Platform-independent uniforms from a 31-bit linear congruential generator."""
    state = seed
    values = np.empty(int(np.prod(shape)))
    for index in range(values.size):
        state = (1103515245 * state + 12345) % 2**31
        values[index] = state / 2**31
    return values.reshape(shape)


def _graded_groups(n_per_group, shifts, dif, n_items=8, n_categories=3):
    """Deterministic graded responses with group impact and DIF in item 0.

    The R mirt references below were computed from these exact matrices.
    """
    thresholds = np.linspace(-1.2, 1.2, n_items)[:, None] + np.linspace(
        -0.8, 0.8, n_categories - 1
    )
    slopes = np.linspace(0.8, 1.6, n_items)
    quantiles = stats.norm.ppf((np.arange(n_per_group) + 0.5) / n_per_group)
    blocks = []
    for group, shift in enumerate(shifts):
        offsets = np.zeros(n_items)
        offsets[0] = dif[group]
        cumulative = expit(
            slopes[:, None]
            * (quantiles[:, None, None] + shift - thresholds - offsets[:, None])
        )
        uniforms = _lcg_uniforms((n_per_group, n_items, 1), 7 + group)
        blocks.append(np.sum(uniforms < cumulative, axis=2))
    return np.vstack(blocks), np.repeat(np.arange(len(shifts)), n_per_group)


# R mirt 1.48 SIBTEST (default Jmin = 5) on _graded_groups data: uniform
# (beta, SE, X2, p) and crossing (beta, X2, df, p).
_R_TWO_GROUP = {
    "polytomous": (
        [0],
        None,
        {},
        (0.105251336964041, 0.0519593238233166, 4.10325267206551, 0.0428007967675301),
        (0.105251336964041, 4.10325267206551, 1, 0.0428007967675301),
    ),
    "polytomous_crossing": (
        [3],
        None,
        {},
        (
            0.00679494792728665,
            0.0476344205288476,
            0.0203484106240151,
            0.886568300853015,
        ),
        (0.0212115212203227, 0.240968697114743, 2, 0.886490962099232),
    ),
    "polytomous_bundle": (
        [0, 1],
        [2, 3, 4, 5, 6, 7],
        {},
        (0.15448514242394, 0.074116993021869, 4.34448050177081, 0.0371290928854748),
        (0.15448514242394, 4.34448050177081, 1, 0.0371290928854748),
    ),
    "polytomous_uncorrected": (
        [0],
        None,
        {"correction": False},
        (0.145982904440789, 0.0519593238233166, 7.89363459600481, 0.00496090735868659),
        (0.145982904440789, 7.89363459600481, 1, 0.00496090735868655),
    ),
}


def _assert_matches_r(uniform, crossing, r_uniform, r_crossing):
    np.testing.assert_allclose(
        [uniform["beta"], uniform["beta_se"], uniform["chi2"], uniform["p_value"]],
        r_uniform,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        [crossing["beta"], crossing["chi2"], crossing["p_value"]],
        [r_crossing[0], r_crossing[1], r_crossing[3]],
        rtol=1e-10,
    )
    assert crossing["df"] == r_crossing[2]


@pytest.mark.parametrize("case", sorted(_R_TWO_GROUP))
def test_polytomous_sibtest_matches_r_mirt(case):
    suspect, matching, options, r_uniform, r_crossing = _R_TWO_GROUP[case]
    data, groups = _graded_groups(500, (0.0, -0.4), (0.0, 0.5))
    kwargs = {"convention": "mirt", "min_cell_size": 6, **options}
    uniform = sibtest(data, groups, suspect, matching, **kwargs)
    crossing = sibtest(data, groups, suspect, matching, method="crossing", **kwargs)
    _assert_matches_r(uniform, crossing, r_uniform, r_crossing)
    items = sibtest_items(data, groups, p_adjust="none", **kwargs)
    if len(suspect) == 1 and matching is None:
        assert items["beta"][suspect[0]] == pytest.approx(uniform["beta"], rel=1e-12)


def test_mirt_convention_reproduces_r_binary_strata():
    # R drops scores where a group's suspect response is constant; the
    # default keeps them, so only the "mirt" convention reproduces R here.
    data, groups = _graded_groups(
        300, (0.0, -0.5), (0.0, 0.6), n_items=16, n_categories=2
    )
    references = {
        0: (0.0910102620072115, 0.0396968794497025, 5.25615292613378),
        15: (0.0433949347199252, 0.0417701053947591, 1.07931182838432),
    }
    for item, expected in references.items():
        result = sibtest(data, groups, [item], convention="mirt", min_cell_size=6)
        default = sibtest(data, groups, [item], min_cell_size=6)
        np.testing.assert_allclose(
            [result["beta"], result["beta_se"], result["chi2"]], expected, rtol=1e-10
        )
        assert default["n_strata"] > result["n_strata"]
        assert default["beta_se"] != pytest.approx(expected[1], rel=1e-3)


def test_missing_rows_are_removed_like_r_na_rm():
    data, groups = _graded_groups(500, (0.0, -0.4), (0.0, 0.5))
    incomplete = data.astype(float)
    incomplete[::37, 3] = np.nan
    incomplete[5::41, 0] = -1
    complete = ~np.any(np.isnan(incomplete) | (incomplete < 0), axis=1)
    kwargs = {"convention": "mirt", "min_cell_size": 6, "na_rm": True}
    # R: SIBTEST(dat, group, suspect_set = 1 or 6, na.rm = TRUE).
    references = {
        0: (
            (
                0.100646186842685,
                0.0533519316632745,
                3.55872438976568,
                0.0592332414947274,
            ),
            (0.100646186842685, 3.55872438976568, 1, 0.0592332414947274),
        ),
        5: (
            (
                -0.030291326026365,
                0.0456824259841231,
                0.439681560249509,
                0.507276192710322,
            ),
            (0.0601706155078411, 1.77741886034685, 2, 0.411186074811173),
        ),
    }
    for item, (r_uniform, r_crossing) in references.items():
        uniform = sibtest(incomplete, groups, [item], **kwargs)
        crossing = sibtest(incomplete, groups, [item], method="crossing", **kwargs)
        _assert_matches_r(uniform, crossing, r_uniform, r_crossing)
        dropped = sibtest(
            data[complete], groups[complete], [item], **{**kwargs, "na_rm": False}
        )
        assert uniform["beta"] == dropped["beta"]
    with pytest.raises(ValueError, match="na_rm=True"):
        sibtest(incomplete, groups, [0])
    with pytest.raises(ValueError, match="na_rm=True"):
        sibtest_items(incomplete, groups)


# R mirt 1.48 GSIBTEST with the default contrasts cbind(1, -diag(2)):
# uniform (beta_1, beta_2, X2, p) and crossing (beta_1, beta_2, X2).
_R_THREE_GROUP = {
    0: (
        (
            -0.0247615683288925,
            0.246193154313565,
            25.1684627739829,
            3.42560867405695e-06,
        ),
        (0.037830728689929, 0.246193154313565, 20.3444299550076),
    ),
    2: (
        (
            -0.00261889005908144,
            -0.0873217943586553,
            3.13144722272773,
            0.208936769438345,
        ),
        (0.0315426890196253, 0.0873217943586553, 4.65851707366628),
    ),
    6: (
        (-0.0335186028637701, -0.0599616820450589, 1.61167032301687, 0.446714691809551),
        (0.0299891224186357, 0.059961682045059, 1.58637487425455),
    ),
}


def test_multigroup_gsibtest_matches_r_mirt():
    data, groups = _graded_groups(400, (0.0, 0.3, -0.3), (0.0, 0.0, 0.5))
    kwargs = {"convention": "mirt", "min_cell_size": 6}
    items = sibtest_items(data, groups, p_adjust="none", **kwargs)
    for item, (r_uniform, r_crossing) in _R_THREE_GROUP.items():
        uniform = sibtest(data, groups, [item], **kwargs)
        crossing = sibtest(data, groups, [item], method="crossing", **kwargs)
        np.testing.assert_allclose(
            [*uniform["beta"], uniform["chi2"], uniform["p_value"]],
            r_uniform,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            [*crossing["beta"], crossing["chi2"]], r_crossing, rtol=1e-10
        )
        assert uniform["df"] == 2
        # R has no asymptotic multi-group crossing p-value either.
        assert crossing["df"] == 0
        assert np.isnan(crossing["p_value"])
        np.testing.assert_allclose(items["beta"][item], uniform["beta"], rtol=1e-12)
        assert items["p_value"][item] == pytest.approx(uniform["p_value"], rel=1e-12)
    np.testing.assert_array_equal(uniform["group_labels"], [0, 1, 2])
    np.testing.assert_array_equal(uniform["contrasts"], [[1, -1, 0], [1, 0, -1]])
    np.testing.assert_allclose(
        uniform["beta_se"], np.sqrt(np.diag(uniform["beta_covariance"]))
    )

    custom = sibtest(data, groups, [0], contrasts=[[0, 1, -1]], **kwargs)
    # R: SIBTEST(dat, group, suspect_set = 1, C = matrix(c(0, 1, -1), 1)).
    np.testing.assert_allclose(
        [custom["beta"][0], custom["chi2"], custom["p_value"]],
        [0.277125841112566, 20.8083213743503, 5.07619270906368e-06],
        rtol=1e-10,
    )


def test_pairwise_tests_match_r_mirt():
    data, groups = _graded_groups(400, (0.0, 0.3, -0.3), (0.0, 0.0, 0.5))
    kwargs = {"convention": "mirt", "min_cell_size": 6}
    result = sibtest(data, groups, [0], pairwise=True, **kwargs)
    # R: SIBTEST(dat, group, suspect_set = 1, pairwise = TRUE).
    assert result["pairs"] == [(0, 1), (0, 2), (1, 2)]
    np.testing.assert_allclose(
        result["beta"],
        [-0.00332791411963066, 0.215837237759652, 0.279022215965407],
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        result["beta_se"],
        [0.0555980719525868, 0.0577180184569806, 0.0606320509358821],
        rtol=1e-10,
    )
    for index, (reference, focal) in enumerate(result["pairs"]):
        mask = np.isin(groups, [reference, focal])
        two_group = sibtest(data[mask], groups[mask], [0], focal_group=focal, **kwargs)
        assert result["p_value"][index] == two_group["p_value"]

    items = sibtest_items(data, groups, pairwise=True, p_adjust="holm", **kwargs)
    assert items["beta"].shape == items["p_value_adjusted"].shape == (8, 3)
    for index, (reference, focal) in enumerate(items["pairs"]):
        mask = np.isin(groups, [reference, focal])
        two_group = sibtest_items(
            data[mask], groups[mask], p_adjust="holm", focal_group=focal, **kwargs
        )
        np.testing.assert_array_equal(
            items["p_value_adjusted"][:, index], two_group["p_value_adjusted"]
        )


def test_multigroup_uncorrected_contrasts_have_analytical_values():
    cells = [
        [(20, 10), (30, 21)],
        [(25, 5), (25, 10)],
        [(40, 12), (10, 2)],
    ]
    data, groups = _cell_responses(cells)
    result = sibtest(data, groups, [4], correction=False)
    counts = np.array([[count for count, _ in group] for group in cells])
    means = np.array([[s / c for c, s in group] for group in cells])
    variances = np.array([[_mean_variance(c, s) for c, s in group] for group in cells])
    weights = counts.sum(axis=0) / counts.sum()
    contrasts = np.array([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]])
    beta = contrasts @ means @ weights
    covariance = contrasts @ np.diag(variances @ weights**2) @ contrasts.T

    np.testing.assert_allclose(result["beta"], beta)
    np.testing.assert_allclose(result["beta_covariance"], covariance)
    assert result["chi2"] == pytest.approx(beta @ np.linalg.solve(covariance, beta))
    assert result["df"] == 2
    assert result["p_value"] == pytest.approx(
        stats.chi2.sf(result["chi2"], 2), rel=1e-12
    )


def test_crossing_permutation_p_value_matches_r_within_monte_carlo_error():
    data, groups = _graded_groups(500, (0.0, -0.4), (0.0, 0.5))
    kwargs = {"convention": "mirt", "min_cell_size": 6, "method": "crossing"}
    asymptotic = sibtest(data, groups, [3], **kwargs)
    permuted = sibtest(data, groups, [3], n_permutations=2000, random_state=0, **kwargs)
    # R: set.seed(1); SIBTEST(..., suspect_set = 4, randomize = TRUE,
    # permute = 10000) gives p = 0.8584.
    assert permuted["p_value"] == pytest.approx(0.8584, abs=0.04)
    assert permuted["beta"] == asymptotic["beta"]
    assert permuted["beta_se"] == asymptotic["beta_se"]
    assert np.isnan(permuted["chi2"])
    assert permuted["df"] == 0
    repeated = sibtest(data, groups, [3], n_permutations=2000, random_state=0, **kwargs)
    assert repeated["p_value"] == permuted["p_value"]

    data, groups = _graded_groups(400, (0.0, 0.3, -0.3), (0.0, 0.0, 0.5))
    permuted = sibtest(data, groups, [2], n_permutations=2000, random_state=0, **kwargs)
    # R with permute = 20000 gives 0.344 to 0.354 over seeds.
    assert permuted["p_value"] == pytest.approx(0.35, abs=0.04)
    assert permuted["chi2"] == pytest.approx(_R_THREE_GROUP[2][1][2], rel=1e-10)
