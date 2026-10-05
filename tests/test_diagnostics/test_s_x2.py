"""Independent category-pattern references and real IRT S-X2 calibration."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import chi2

import mirt.diagnostics.itemfit as itemfit_module
from mirt.diagnostics.itemfit import compute_itemfit, compute_s_x2


class GridProbabilityModel:
    """Discrete latent mixture with an explicit, externally known response law."""

    def __init__(self, probabilities, categories, n_factors=1):
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.n_categories = categories
        self.n_items = len(categories)
        self.n_factors = n_factors
        self.is_polytomous = max(categories) > 2
        self.item_names = [f"item_{item}" for item in range(self.n_items)]
        self.free_parameter_masks = {}
        self.batch_sizes = []

    def probability(self, theta):
        self.batch_sizes.append(len(theta))
        values = self.probabilities[np.asarray(theta[:, 0], dtype=int)]
        return values if self.is_polytomous else values[:, :, 1]


def enumerated_joint(probabilities, categories, weights):
    """Enumerate response patterns directly; do not use score recursion."""
    patterns = np.asarray(list(product(*(range(count) for count in categories))))
    masses = []
    for pattern in patterns:
        point_likelihoods = []
        for point in probabilities:
            likelihood = 1.0
            for item, category in enumerate(pattern):
                likelihood *= point[item, category]
            point_likelihoods.append(likelihood)
        masses.append(np.dot(weights, point_likelihoods))
    masses = np.asarray(masses)
    scores = patterns.sum(axis=1)
    marginal = np.bincount(scores, weights=masses)
    joint = []
    for item, count in enumerate(categories):
        table = np.zeros((len(marginal), count))
        for pattern, mass, score in zip(patterns, masses, scores, strict=True):
            table[score, pattern[item]] += mass
        joint.append(table)
    return patterns, joint, marginal


def enumerated_statistic(responses, joint, marginal, categories):
    """Original Pearson count statistic with the published ordinal tail groups."""
    totals = responses.sum(axis=1)
    maximum = len(marginal) - 1
    statistics = []
    contrasts = []
    for item, count in enumerate(categories):
        # Kang-Chen: pool score 1..z into z and F-z..F-1 into F-z.
        groups = {}
        for score in range(1, maximum):
            group = min(max(score, count - 1), maximum - count + 1)
            groups.setdefault(group, []).append(score)
        statistic = 0.0
        degrees = 0
        for scores in groups.values():
            observed = np.zeros(count)
            expected = np.zeros(count)
            for score in scores:
                selected = responses[totals == score, item]
                observed += np.bincount(selected, minlength=count)
                expected += len(selected) * joint[item][score] / marginal[score]
            if observed.sum() > 0:
                positive = expected > 0
                statistic += np.sum(
                    (observed[positive] - expected[positive]) ** 2 / expected[positive]
                )
                degrees += np.count_nonzero(positive) - 1
        statistics.append(statistic)
        contrasts.append(degrees)
    return np.asarray(statistics), np.asarray(contrasts)


@pytest.mark.parametrize("categories", [[2] * 5, [2, 3, 4, 2]])
@pytest.mark.parametrize("block_rows", [1, 2, 1000])
def test_s_x2_matches_exhaustive_latent_mixture_patterns(
    monkeypatch, categories, block_rows
):
    rng = np.random.default_rng(5412)
    probabilities = np.zeros((3, len(categories), max(categories)))
    for item, count in enumerate(categories):
        values = rng.uniform(0.05, 1.0, size=(3, count))
        probabilities[:, item, :count] = values / values.sum(axis=1, keepdims=True)
    weights = np.array([0.2, 0.3, 0.5])
    model = GridProbabilityModel(probabilities, categories)
    patterns, joint, marginal = enumerated_joint(probabilities, categories, weights)
    frequencies = (np.arange(len(patterns)) * 7) % 11 + 2
    responses = np.repeat(patterns, frequencies, axis=0)
    responses.setflags(write=False)
    expected_statistic, expected_df = enumerated_statistic(
        responses, joint, marginal, categories
    )
    monkeypatch.setattr(
        itemfit_module,
        "_ITEMFIT_TARGET_CHUNK_ELEMENTS",
        block_rows * max(len(categories) * max(categories), len(marginal) * 3),
    )
    monkeypatch.setattr(itemfit_module, "_SX2_TARGET_CHUNK_ELEMENTS", 9)

    result = compute_s_x2(
        model,
        responses,
        min_expected=0,
        quadrature_points=np.arange(3),
        quadrature_weights=weights,
    )

    assert_allclose(result["S_X2"], expected_statistic, rtol=3e-13, atol=3e-13)
    assert_allclose(result["df"], expected_df)
    assert_allclose(result["p_value"], chi2.sf(expected_statistic, expected_df))
    assert max(model.batch_sizes) <= block_rows
    assert sum(model.batch_sizes) == 3


def test_known_binary_contingency_table_and_item_parameter_degrees():
    # All eight response patterns with fixed counts define an independently
    # calculable two-row table (scores 1 and 2) for each of the three items.
    probabilities = np.array([[[0.8, 0.2], [0.6, 0.4], [0.3, 0.7]]])
    patterns, joint, marginal = enumerated_joint(probabilities, [2] * 3, [1.0])
    responses = np.repeat(patterns, [2, 17, 8, 13, 7, 11, 5, 3], axis=0)
    expected_statistic, _ = enumerated_statistic(responses, joint, marginal, [2] * 3)
    assert_allclose(
        expected_statistic, [15.839073332877213, 2.8427485565972783, 9.843268351619642]
    )
    model = GridProbabilityModel(probabilities, [2] * 3)

    result = compute_s_x2(
        model,
        responses,
        min_expected=0,
        quadrature_points=[0],
        quadrature_weights=[1],
        item_parameter_counts=[1, 2, 3],
    )

    assert_allclose(result["S_X2"], expected_statistic)
    assert_allclose(result["df"], [1, 0, 0])
    assert result["p_value"][0] == pytest.approx(chi2.sf(expected_statistic[0], 1))
    assert np.all(np.isnan(result["p_value"][1:]))


def test_binary_sparse_score_rows_sum_expected_counts_before_testing():
    # Uniform items imply P(X_j=1 | S=s)=s/4. Both two-person tail
    # groups must merge with the 100-person score-2 group at min_expected=1.
    patterns = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 1]])
    responses = np.repeat(patterns, [2, 100, 2], axis=0)
    model = GridProbabilityModel(np.full((1, 4, 2), 0.5), [2] * 4)
    result = compute_s_x2(
        model, responses, quadrature_points=[0], quadrature_weights=[1]
    )
    observed_ones = responses.sum(axis=0)
    expected = (observed_ones - 52) ** 2 / 52 + (104 - observed_ones - 52) ** 2 / 52
    assert_allclose(result["S_X2"], expected)
    assert_allclose(result["df"], np.ones(4))
    assert_allclose(result["p_value"], chi2.sf(expected, 1))


def test_adjacent_ordinal_category_pooling_and_degrees():
    # One nondeterministic total-score group; merge the first two ordinal
    # categories, with expected [0.25, 5.75, 4] -> [6,4] and O -> [3,7].
    observed = np.array(
        [[0, 0, 0], [0, 0, 0], [0, 3, 7], [0, 0, 0], [0, 0, 0]], dtype=float
    )
    expected = np.array([[0, 0, 0], [0, 0, 0], [0.25, 5.75, 4], [0, 0, 0], [0, 0, 0]])
    statistic, degrees, p_value = itemfit_module._sx2_from_tables(
        observed, expected, 0, 1
    )
    assert statistic == pytest.approx(3.75)
    assert degrees == 1
    assert p_value == pytest.approx(chi2.sf(3.75, 1))


def test_ordinal_item_larger_than_rest_score_range_is_unestimable():
    probabilities = np.array([[[0.5, 0.5, 0.0, 0.0], [0.25, 0.25, 0.25, 0.25]]])
    model = GridProbabilityModel(probabilities, [2, 4])
    responses = np.repeat(np.asarray(list(product(range(2), range(4)))), 10, axis=0)
    result = compute_s_x2(
        model, responses, quadrature_points=[0], quadrature_weights=[1]
    )
    assert np.isfinite(result["S_X2"][0])
    assert np.isnan(result["S_X2"][1])
    assert result["df"][1] == 0
    assert np.isnan(result["p_value"][1])


def test_standard_normal_quadrature_matches_independent_pattern_integrals():
    from mirt.models import TwoParameterLogistic

    model = TwoParameterLogistic(5)
    model.set_parameters(
        discrimination=np.array([0.7, 1.1, 1.6, 0.9, 1.3]),
        difficulty=np.array([-1.5, -0.3, 0.8, 1.6, 0.1]),
    )
    # Independent high-order Gaussian integration plus pattern enumeration.
    nodes, masses = np.polynomial.hermite.hermgauss(121)
    positive = model.probability((np.sqrt(2) * nodes)[:, None])
    probabilities = np.stack([1 - positive, positive], axis=2)
    patterns, joint, marginal = enumerated_joint(
        probabilities, [2] * 5, masses / np.sqrt(np.pi)
    )
    responses = np.repeat(patterns, np.arange(1, len(patterns) + 1), axis=0)
    expected, contrasts = enumerated_statistic(responses, joint, marginal, [2] * 5)

    result = compute_s_x2(model, responses, min_expected=0, n_quadpts=81)

    assert_allclose(result["S_X2"], expected, rtol=2e-10)
    assert_allclose(result["df"], contrasts - 2)


def test_s_x2_uses_latent_distribution_and_combines_with_mean_squares(monkeypatch):
    import mirt.scoring
    from mirt.models import TwoParameterLogistic

    model = TwoParameterLogistic(4)
    model.set_parameters(
        discrimination=np.array([0.8, 1.0, 1.4, 1.7]),
        difficulty=np.array([-1.5, -0.3, 0.4, 1.1]),
    )
    patterns = np.asarray(list(product(range(2), repeat=4)))
    responses = np.repeat(patterns, 10, axis=0)
    monkeypatch.setattr(
        mirt.scoring,
        "fscores",
        lambda *args, **kwargs: pytest.fail("S-X2 must not score respondents"),
    )
    result = compute_s_x2(model, responses)
    for abilities in [-3.0, 2.0]:
        compared = compute_s_x2(
            model, responses, theta=np.full(len(responses), abilities)
        )
        for key in result:
            assert_allclose(compared[key], result[key])
    combined = compute_itemfit(
        model,
        responses,
        statistics=["S_X2", "infit", "outfit"],
        theta=np.zeros(len(responses)),
    )
    for key in result:
        assert_allclose(combined[key], result[key])
    assert set(combined) == {"S_X2", "df", "p_value", "infit", "outfit"}
    shifted = compute_s_x2(
        model,
        responses,
        quadrature_points=[-1.0, 0.0, 1.0],
        quadrature_weights=[0.7, 0.2, 0.1],
    )
    assert not np.allclose(shifted["S_X2"], result["S_X2"])


def test_missing_rows_are_explicitly_removed_only_for_s_x2():
    from mirt.models import TwoParameterLogistic

    model = TwoParameterLogistic(4)
    patterns = np.asarray(list(product(range(2), repeat=4)))
    responses = np.repeat(patterns, 10, axis=0).astype(float)
    responses[0, 0] = np.nan
    responses[1, 1] = -9
    responses.setflags(write=False)
    with pytest.raises(ValueError, match="complete responses"):
        compute_s_x2(model, responses)
    expected = compute_s_x2(model, responses[2:])
    removed = compute_s_x2(model, responses, na_rm=True)
    for key in expected:
        assert_allclose(removed[key], expected[key])
    # Mean squares use the documented negative missing-value convention.
    responses = responses.copy()
    responses[0, 0] = -1
    responses.setflags(write=False)
    combined = compute_itemfit(
        model,
        responses,
        statistics=["S_X2", "infit", "outfit"],
        theta=np.zeros(len(responses)),
        na_rm=True,
    )
    for key in expected:
        assert_allclose(combined[key], expected[key])
    mean_squares = compute_itemfit(model, responses, theta=np.zeros(len(responses)))
    assert_allclose(combined["infit"], mean_squares["infit"])
    assert_allclose(combined["outfit"], mean_squares["outfit"])
    with pytest.raises(ValueError, match="no complete persons"):
        compute_s_x2(model, np.full((10, 4), -1), na_rm=True)


def test_n_groups_deprecation_preserves_exact_score_statistic():
    from mirt.models import TwoParameterLogistic

    model = TwoParameterLogistic(4)
    responses = np.repeat(np.asarray(list(product(range(2), repeat=4))), 20, axis=0)
    expected = compute_s_x2(model, responses)
    with pytest.warns(DeprecationWarning, match="exact total scores"):
        result = compute_s_x2(model, responses, n_groups=3)
    for key in result:
        assert_allclose(result[key], expected[key])


def test_top_level_itemfit_forwards_explicit_quadrature_and_parameter_counts():
    from mirt import itemfit

    model = GridProbabilityModel(np.full((1, 4, 2), 0.5), [2] * 4)
    responses = np.repeat(np.asarray(list(product(range(2), repeat=4))), 10, axis=0)
    result = itemfit(
        SimpleNamespace(model=model),
        responses,
        statistics=["S_X2"],
        quadrature_points=np.array([0]),
        quadrature_weights=np.array([2.0]),
        item_parameter_counts=np.zeros(4, dtype=int),
        min_expected=0,
    )
    assert_allclose(np.asarray(result["S_X2"]), 0.0, atol=1e-25)
    assert_allclose(np.asarray(result["df"]), 3)
    assert_allclose(np.asarray(result["p_value"]), 1.0)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"min_expected": -1}, "min_expected"),
        ({"min_expected": np.inf}, "min_expected"),
        ({"min_expected": True}, "min_expected"),
        ({"min_expected": "invalid"}, "min_expected"),
        ({"n_quadpts": True}, "n_quadpts"),
        ({"n_quadpts": 1}, "n_quadpts"),
        ({"n_quadpts": 2.5}, "n_quadpts"),
        ({"na_rm": "yes"}, "na_rm"),
        ({"quadrature_points": [0]}, "supplied together"),
        ({"quadrature_weights": [1]}, "supplied together"),
        ({"quadrature_points": [[0, 1]], "quadrature_weights": [1]}, "one column"),
        ({"quadrature_points": [np.nan], "quadrature_weights": [1]}, "finite rows"),
        ({"quadrature_points": [], "quadrature_weights": []}, "finite rows"),
        ({"quadrature_points": [0], "quadrature_weights": [0]}, "positive total"),
        ({"quadrature_points": [0], "quadrature_weights": [-1]}, "nonnegative"),
        (
            {"quadrature_points": [0], "quadrature_weights": [np.inf]},
            "finite nonnegative",
        ),
        ({"quadrature_points": [0], "quadrature_weights": [1, 2]}, "one finite"),
        ({"item_parameter_counts": [-1] * 4}, "nonnegative integer"),
        ({"item_parameter_counts": [1.5] * 4}, "nonnegative integer"),
        ({"item_parameter_counts": [1, 2]}, "one nonnegative"),
    ],
)
def test_s_x2_rejects_invalid_inference_controls(options, message):
    from mirt.models import TwoParameterLogistic

    with pytest.raises(ValueError, match=message):
        compute_s_x2(TwoParameterLogistic(4), np.ones((10, 4)), **options)


@pytest.mark.parametrize("value", [np.inf, 0.5, 2])
def test_s_x2_rejects_invalid_observed_categories(value):
    from mirt.models import TwoParameterLogistic

    responses = np.zeros((10, 4))
    responses[0, 0] = value
    with pytest.raises(ValueError, match="infinite|integer category codes"):
        compute_s_x2(TwoParameterLogistic(4), responses, na_rm=True)


def test_unestimable_tables_do_not_invent_degrees_of_freedom():
    model = GridProbabilityModel(np.full((1, 3, 2), 0.5), [2] * 3)
    responses = np.array([[0, 0, 0], [1, 1, 1]])
    controls = {"quadrature_points": [0], "quadrature_weights": [1]}
    deterministic = compute_s_x2(model, responses, **controls)
    assert_allclose(deterministic["S_X2"], 0)
    assert_allclose(deterministic["df"], 0)
    assert np.all(np.isnan(deterministic["p_value"]))
    # Pooling cannot rescue a single one-person row at a five-person threshold.
    sparse = compute_s_x2(model, np.array([[0, 1, 0]]), min_expected=5, **controls)
    assert_allclose(sparse["df"], 1)
    assert np.all(np.isnan(sparse["p_value"]))


def test_impossible_model_total_scores_and_category_cells():
    model = GridProbabilityModel(
        np.array([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]), [2] * 3
    )
    controls = {"quadrature_points": [0], "quadrature_weights": [1], "min_expected": 0}
    with pytest.raises(ValueError, match="zero probability"):
        compute_s_x2(model, np.ones((10, 3)), **controls)
    # The total score is possible, while these item categories are impossible.
    result = compute_s_x2(model, np.tile([1, 0, 0], (10, 1)), **controls)
    assert np.isinf(result["S_X2"][[0, 2]]).all()
    assert_allclose(result["df"], 0)
    assert np.isnan(result["p_value"]).all()


@pytest.mark.parametrize("bad_probability", [np.nan, -0.01, 1.01])
def test_s_x2_rejects_invalid_custom_model_probability_curves(bad_probability):
    probabilities = np.full((1, 4, 2), 0.5)
    probabilities[0, 0, 1] = bad_probability
    model = GridProbabilityModel(probabilities, [2] * 4)
    with pytest.raises(ValueError, match="finite and between"):
        compute_s_x2(
            model, np.ones((10, 4)), quadrature_points=[0], quadrature_weights=[1]
        )


@pytest.mark.parametrize("bad_padding", [False, True])
def test_s_x2_rejects_invalid_ordinal_probability_normalization(bad_padding):
    probabilities = np.array([[[0.5, 0.5, 0.0], [0.3, 0.4, 0.3]]])
    if bad_padding:
        probabilities[0, 0] = [0.3, 0.4, 0.3]
    else:
        probabilities[0, 1, 1] = 0.3
    model = GridProbabilityModel(probabilities, [2, 3])
    with pytest.raises(ValueError, match="sum to one with zero padding"):
        compute_s_x2(
            model, np.ones((10, 2)), quadrature_points=[0], quadrature_weights=[1]
        )


def test_item_parameter_masks_count_fixed_and_heterogeneous_parameters():
    from mirt.models import (
        GradedResponseModel,
        OneParameterLogistic,
        PartialCreditModel,
    )
    from mirt.models.polytomous import RatingScaleModel

    assert_allclose(
        itemfit_module._sx2_parameter_counts(OneParameterLogistic(4), None), 1
    )
    assert_allclose(
        itemfit_module._sx2_parameter_counts(PartialCreditModel(4, [2, 3, 4, 2]), None),
        [1, 2, 3, 1],
    )
    assert_allclose(
        itemfit_module._sx2_parameter_counts(
            GradedResponseModel(4, [2, 3, 4, 2]), None
        ),
        [2, 3, 4, 2],
    )
    # Shared thresholds must be explicit even when their length equals n_items.
    model = RatingScaleModel(4, 5)
    with pytest.raises(ValueError, match="shared parameters"):
        itemfit_module._sx2_parameter_counts(model, None)
    assert_allclose(itemfit_module._sx2_parameter_counts(model, [1, 1, 1, 1]), 1)


@pytest.mark.parametrize(
    "family", ["LLTM", "RaschLLTM", "Explanatory", "Testlet", "Mixture"]
)
def test_shared_parameter_counts_require_explicit_allocation_despite_shape_coincidence(
    family,
):
    from mirt.models.explanatory import LLTM, ExplanatoryIRT, RaschLLTM
    from mirt.models.mixture import MixtureIRT
    from mirt.models.testlet import TestletModel

    # These global arrays deliberately have n_items entries, so shape alone
    # cannot distinguish an item coefficient from a shared design coefficient.
    features = np.array([[1.0, 2.0], [3.0, 1.0]])
    factories = {
        "LLTM": lambda: LLTM(2, features),
        "RaschLLTM": lambda: RaschLLTM(2, features),
        "Explanatory": lambda: ExplanatoryIRT(2, features, 1),
        "Testlet": lambda: TestletModel(2, [0, 1]),
        "Mixture": lambda: MixtureIRT(2, n_classes=2),
    }
    model = factories[family]()
    with pytest.raises(ValueError, match="shared parameters"):
        itemfit_module._sx2_parameter_counts(model, None)
    assert_allclose(itemfit_module._sx2_parameter_counts(model, [1, 2]), [1, 2])


def test_multidimensional_latent_mixture_and_scaled_probability_masses():
    probabilities = np.array(
        [[[0.7, 0.3], [0.4, 0.6], [0.1, 0.9]], [[0.2, 0.8], [0.7, 0.3], [0.6, 0.4]]]
    )
    model = GridProbabilityModel(probabilities, [2] * 3, n_factors=2)
    patterns, joint, marginal = enumerated_joint(probabilities, [2] * 3, [0.25, 0.75])
    responses = np.repeat(patterns, np.arange(1, 9), axis=0)
    expected, degrees = enumerated_statistic(responses, joint, marginal, [2] * 3)
    result = compute_s_x2(
        model,
        responses,
        min_expected=0,
        quadrature_points=[[0, -1], [1, 2]],
        quadrature_weights=[1e308 / 3, 1e308],
    )
    assert_allclose(result["S_X2"], expected)
    assert_allclose(result["df"], degrees)
    model.n_factors = 4
    with pytest.raises(ValueError, match="bounded explicit quadrature"):
        compute_s_x2(model, responses)


def test_s_x2_bounds_response_counting_and_evaluates_only_quadrature_nodes(monkeypatch):
    model = GridProbabilityModel(np.full((41, 4, 2), 0.5), [2] * 4)
    responses = np.repeat(np.asarray(list(product(range(2), repeat=4))), 1000, axis=0)
    original_bincount = np.bincount
    counted_rows = []

    def bounded_bincount(values, **kwargs):
        counted_rows.append(len(values))
        assert len(values) <= 17
        return original_bincount(values, **kwargs)

    monkeypatch.setattr(itemfit_module.np, "bincount", bounded_bincount)
    monkeypatch.setattr(itemfit_module, "_SX2_TARGET_CHUNK_ELEMENTS", 17 * 4)
    monkeypatch.setattr(itemfit_module, "_ITEMFIT_TARGET_CHUNK_ELEMENTS", 2 * 15)
    result = compute_s_x2(
        model,
        responses,
        quadrature_points=np.arange(41),
        quadrature_weights=np.ones(41),
    )
    assert sum(counted_rows) == len(responses) * (model.n_items + 1)
    assert sum(model.batch_sizes) == 41
    # Probability batches hold at most 2 * 15 node-item-category values.
    assert max(model.batch_sizes) * model.n_items * 2 <= 2 * 15
    assert_allclose(result["S_X2"], 0, atol=1e-20)
    assert_allclose(result["df"], 3)


@pytest.mark.slow
def test_repeated_actual_2pl_fits_have_calibrated_s_x2_false_positive_rate():
    from scipy.special import expit

    from mirt import fit_mirt

    rng = np.random.default_rng(220049)
    n_persons, n_items = 1000, 12
    discrimination = np.linspace(0.8, 1.6, n_items)
    difficulty = np.linspace(-1.5, 1.5, n_items)
    p_values = []
    for _ in range(24):
        theta = rng.normal(size=n_persons)
        probabilities = expit((theta[:, None] - difficulty) * discrimination)
        responses = (rng.random((n_persons, n_items)) < probabilities).astype(int)
        fitted = fit_mirt(
            responses, n_quadpts=31, max_iter=200, compute_standard_errors=False
        )
        assert fitted.converged
        p_values.extend(compute_s_x2(fitted.model, responses)["p_value"])
    assert np.all(np.isfinite(p_values))
    # 288 item tests, with within-test dependence: bounds tolerate Monte Carlo
    # variation but reject both a powerless statistic and inflated type-I error.
    assert 0.015 <= np.mean(np.asarray(p_values) < 0.05) <= 0.10
    assert 0.40 <= np.mean(p_values) <= 0.60


@pytest.mark.slow
def test_repeated_heterogeneous_grm_and_gpcm_fits_calibrate_generalized_s_x2():
    from mirt.estimation import EMEstimator
    from mirt.models import GeneralizedPartialCredit, GradedResponseModel

    rng = np.random.default_rng(33091)
    categories = [2, 3, 4, 3, 2, 4, 3, 4]
    for model_type in [GradedResponseModel, GeneralizedPartialCredit]:
        generating_model = model_type(8, n_categories=categories)
        p_values = []
        for _ in range(12):
            theta = rng.normal(size=(1200, 1))
            probabilities = generating_model.probability(theta)
            responses = (
                rng.random(probabilities.shape[:2])[:, :, None]
                > probabilities.cumsum(axis=2)
            ).sum(axis=2)
            fitted = EMEstimator(
                n_quadpts=31, max_iter=300, compute_standard_errors=False
            ).fit(model_type(8, n_categories=categories), responses)
            assert fitted.converged
            p_values.extend(compute_s_x2(fitted.model, responses)["p_value"])
        assert np.all(np.isfinite(p_values))
        assert 0.005 <= np.mean(np.asarray(p_values) < 0.05) <= 0.15
        assert 0.35 <= np.mean(p_values) <= 0.65


def test_fitted_2pl_s_x2_detects_nonmonotone_item_response_misfit():
    from scipy.special import expit

    from mirt import fit_mirt

    rng = np.random.default_rng(341219)
    n_persons, n_items = 1800, 12
    theta = rng.normal(size=n_persons)
    probabilities = expit(
        (theta[:, None] - np.linspace(-1.5, 1.5, n_items))
        * np.linspace(0.8, 1.6, n_items)
    )
    probabilities[:, 0] = expit(2 * theta**2 - 1)
    responses = (rng.random((n_persons, n_items)) < probabilities).astype(int)
    fitted = fit_mirt(
        responses, n_quadpts=31, max_iter=300, compute_standard_errors=False
    )
    assert fitted.converged
    result = compute_s_x2(fitted.model, responses)
    assert result["p_value"][0] < 1e-15
    assert result["S_X2"][0] > 100
    assert np.count_nonzero(result["p_value"][1:] < 0.01) <= 2


def test_mixture_class_dependence_requires_joint_score_integration():
    from mirt.models.mixture import MixtureIRT

    model = MixtureIRT(3)
    model.set_parameters(
        difficulty_class0=np.full(3, -np.log(4)),
        difficulty_class1=np.full(3, np.log(4)),
    )
    theta = np.zeros((1, 1))
    np.testing.assert_allclose(model.probability(theta), 0.5)
    # Classes belong to the whole response pattern: .5*.8^3 + .5*.2^3=.26,
    # whereas multiplying class-marginal item curves would incorrectly give .125.
    np.testing.assert_allclose(
        np.exp(model.log_likelihood(np.ones((1, 3), dtype=int), theta)), 0.26
    )
    responses = np.array(list(product([0, 1], repeat=3)))
    for compute in (
        compute_s_x2,
        lambda model, data: compute_itemfit(model, data, statistics=["S_X2"]),
    ):
        with pytest.raises(ValueError, match="latent class integration"):
            compute(model, responses)
    result = compute_itemfit(
        model, responses, statistics=["infit"], theta=np.zeros((8, 1))
    )
    assert np.all(np.isfinite(result["infit"]))


def _reference_full_recursion_tables(probabilities, categories, weights):
    """Previous algorithm: one full Lord-Wingersky recursion per left-out item."""

    def distribution(skip):
        values = np.ones((probabilities.shape[0], 1))
        for item, count in enumerate(categories):
            if item == skip:
                continue
            updated = np.zeros((len(values), values.shape[1] + count - 1))
            for category in range(count):
                updated[:, category : category + values.shape[1]] += (
                    values * probabilities[:, item, category, None]
                )
            values = updated
        return values

    marginal = weights @ distribution(-1)
    joint = []
    for item, count in enumerate(categories):
        rest = distribution(item)
        table = np.zeros((len(marginal), count))
        for category in range(count):
            table[category : category + rest.shape[1], category] = (
                weights * probabilities[:, item, category]
            ) @ rest
        joint.append(table / marginal[:, None])
    return joint, marginal


@pytest.mark.parametrize("categories", [[2] * 7, [3] * 5, [2, 5, 3, 2, 4, 6]])
@pytest.mark.parametrize("chunk_elements", [1, 200, 262_144])
def test_prefix_suffix_tables_match_full_recursion_reference(
    monkeypatch, categories, chunk_elements
):
    rng = np.random.default_rng(len(categories) + chunk_elements)
    n_nodes, width = 23, max(categories)
    probabilities = rng.uniform(0.05, 1.0, size=(n_nodes, len(categories), width))
    probabilities *= np.arange(width) < np.asarray(categories)[:, None]
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    weights = rng.uniform(0.1, 1.0, n_nodes)
    weights /= weights.sum()
    model = GridProbabilityModel(probabilities, categories)
    # Bound both probability batches and the recursion's node blocks.
    monkeypatch.setattr(
        itemfit_module, "_ITEMFIT_TARGET_CHUNK_ELEMENTS", chunk_elements
    )
    monkeypatch.setattr(itemfit_module, "_SX2_TARGET_CHUNK_ELEMENTS", chunk_elements)
    recursion_blocks = []
    original_add = itemfit_module._add_item_to_distribution

    def spy_add(distribution, item_probabilities):
        recursion_blocks.append(len(distribution))
        return original_add(distribution, item_probabilities)

    monkeypatch.setattr(itemfit_module, "_add_item_to_distribution", spy_add)

    conditional, marginal = itemfit_module._conditional_category_probabilities(
        model,
        np.asarray(categories),
        np.arange(n_nodes, dtype=float)[:, None],
        weights,
    )

    expected_joint, expected_marginal = _reference_full_recursion_tables(
        probabilities, categories, weights
    )
    assert_allclose(marginal, expected_marginal, rtol=1e-12, atol=1e-15)
    for actual, expected in zip(conditional, expected_joint, strict=True):
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
    n_scores = sum(categories) - len(categories) + 1
    block_rows = max(1, chunk_elements // (len(categories) * n_scores))
    assert max(recursion_blocks) == min(block_rows, n_nodes)
    assert max(model.batch_sizes) <= block_rows


def test_antidiagonal_sums_match_direct_reduction():
    rng = np.random.default_rng(9)
    for shape in [(3, 4, 7), (2, 7, 4), (1, 1, 5), (2, 6, 1)]:
        matrices = rng.normal(size=shape)
        expected = np.zeros((shape[0], shape[1] + shape[2] - 1))
        for row in range(shape[1]):
            for column in range(shape[2]):
                expected[:, row + column] += matrices[:, row, column]
        assert_allclose(
            itemfit_module._antidiagonal_sums(matrices), expected, rtol=1e-14
        )


def _reference_pooled_statistic(observed, expected, n_parameters, minimum):
    """Row-by-row pooling with array deletion, as before vectorization."""
    # Expected counts that equal min_expected up to rounding are not sparse.
    minimum *= 1.0 - 1e-10
    n_categories, n_scores = observed.shape[1], observed.shape[0]
    observed, expected = observed[1:-1].copy(), expected[1:-1].copy()
    if n_categories > 2 and len(observed):
        high, low = n_scores - n_categories - 1, n_categories - 2
        if high < low:
            return np.nan, 0, np.nan
        observed[low] += observed[:low].sum(axis=0)
        expected[low] += expected[:low].sum(axis=0)
        observed[high] += observed[high + 1 :].sum(axis=0)
        expected[high] += expected[high + 1 :].sum(axis=0)
        observed, expected = observed[low : high + 1], expected[low : high + 1]
    populated = observed.sum(axis=1) > 0
    observed, expected = observed[populated], expected[populated]
    while n_categories == 2 and minimum > 0 and len(expected) > 1:
        sparse_rows = np.flatnonzero(np.min(expected, axis=1) < minimum)
        if sparse_rows.size == 0:
            break
        row = int(sparse_rows[0])
        if row == 0:
            neighbor = 1
        elif row == len(expected) - 1:
            neighbor = row - 1
        elif observed[row - 1].sum() <= observed[row + 1].sum():
            neighbor = row - 1
        else:
            neighbor = row + 1
        observed[neighbor] += observed[row]
        expected[neighbor] += expected[row]
        observed = np.delete(observed, row, axis=0)
        expected = np.delete(expected, row, axis=0)
    statistic, contrasts, sparse = 0.0, 0, False
    for row_observed, row_expected in zip(observed, expected, strict=True):
        while (
            n_categories > 2
            and minimum > 0
            and len(row_expected) > 1
            and np.min(row_expected) < minimum
        ):
            category = int(np.argmin(row_expected))
            if category == 0:
                neighbor = 1
            elif category == len(row_expected) - 1:
                neighbor = category - 1
            elif row_expected[category - 1] <= row_expected[category + 1]:
                neighbor = category - 1
            else:
                neighbor = category + 1
            row_observed = row_observed.copy()
            row_expected = row_expected.copy()
            row_observed[neighbor] += row_observed[category]
            row_expected[neighbor] += row_expected[category]
            row_observed = np.delete(row_observed, category)
            row_expected = np.delete(row_expected, category)
        positive = row_expected > 0
        contrasts += max(int(positive.sum()) - 1, 0)
        if np.any(~positive & (row_observed > 0)):
            statistic = np.inf
        statistic += np.sum(
            (row_observed[positive] - row_expected[positive]) ** 2
            / row_expected[positive]
        )
        sparse |= bool(np.any(row_expected[positive] < minimum))
    degrees = max(contrasts - n_parameters, 0)
    p_value = chi2.sf(statistic, degrees) if degrees > 0 and not sparse else np.nan
    return statistic, degrees, p_value


@pytest.mark.parametrize("n_categories", [2, 3, 5])
@pytest.mark.parametrize("minimum", [0.0, 1.0, 4.0])
def test_vectorized_pooling_matches_rowwise_reference(n_categories, minimum):
    rng = np.random.default_rng(n_categories * 10 + int(minimum))
    for _ in range(25):
        n_scores = int(rng.integers(n_categories + 3, 30))
        expected = rng.gamma(0.7, 3.0, size=(n_scores, n_categories))
        expected[rng.random(expected.shape) < 0.05] = 0.0
        observed = rng.poisson(expected + 0.1).astype(float)
        # As in S-X2, expected rows redistribute each score group's persons.
        totals = expected.sum(axis=1, keepdims=True)
        expected *= np.divide(
            observed.sum(axis=1, keepdims=True),
            totals,
            out=np.zeros_like(totals),
            where=totals > 0,
        )
        actual = itemfit_module._sx2_from_tables(observed, expected, 1, minimum)
        reference = _reference_pooled_statistic(observed, expected, 1, minimum)
        assert_allclose(actual[0], reference[0], rtol=1e-12)
        assert actual[1] == reference[1]
        assert_allclose(actual[2], reference[2], rtol=1e-10, equal_nan=True)


def test_one_person_score_row_at_threshold_keeps_p_value():
    # Score row 2 holds one person; pooling all its categories leaves one cell
    # whose expected count is 1 up to rounding. It used to be judged sparse
    # only when the rounding fell below 1, which suppressed the p-value.
    observed = np.zeros((7, 3))
    expected = np.zeros((7, 3))
    observed[2] = [0, 1, 0]
    expected[2] = [0.25, 0.5, 0.25 - 1e-16]
    observed[3:5] = [[10, 20, 10], [15, 10, 15]]
    expected[3:5] = [[12.0, 17.0, 11.0], [14.0, 12.0, 14.0]]
    assert expected[2].sum() < 1.0

    statistic, degrees, p_value = itemfit_module._sx2_from_tables(
        observed, expected, 0, 1.0
    )

    reference = sum(
        np.sum((observed[row] - expected[row]) ** 2 / expected[row]) for row in (3, 4)
    )
    assert_allclose(statistic, reference + 0.0)
    assert degrees == 4
    assert_allclose(p_value, chi2.sf(reference, 4))


def test_binary_row_pooling_breaks_population_ties_toward_lower_scores():
    # Rows 0 and 2 both hold three persons, but rounding leaves the expected
    # total of row 0 one unit in the last place above row 2. The sparse middle
    # row must still join its lower-score neighbor, as documented.
    observed = np.array([[2, 1], [0, 1], [1, 2]], dtype=float)
    expected = np.array([[1.5, 1.5000000000000004], [0.4, 0.6], [1.5, 1.5]])
    assert expected[0].sum() > expected[2].sum()

    pooled_observed, pooled_expected = itemfit_module._pool_score_rows(
        observed, expected, 1.0
    )

    assert_allclose(pooled_observed, [[2, 2], [1, 2]])
    assert_allclose(pooled_expected, [[1.9, 2.1], [1.5, 1.5]])


def test_ordinal_s_x2_p_values_survive_single_person_score_rows():
    # Long five-category forms leave one-person score rows whose pooled
    # expected count is exactly one person up to rounding. Every p-value used
    # to be NaN because that rounding could fall below min_expected=1.
    from mirt.models import GradedResponseModel

    rng = np.random.default_rng(2)
    model = GradedResponseModel(n_items=20, n_categories=5)
    theta = rng.normal(size=(1000, 1))
    probabilities = model.probability(theta)
    draws = rng.random((1000, 20, 1))
    responses = np.minimum((draws > probabilities.cumsum(axis=2)).sum(axis=2), 4)

    result = compute_s_x2(model, responses)

    assert np.all(np.isfinite(result["p_value"]))
    assert np.all(result["df"] > 0)
