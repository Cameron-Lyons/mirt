"""Response-based vertical calibration and physical-item design oracles."""

import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit, logsumexp

from mirt._model_defaults import uses_builtin_model_hooks
from mirt.diagnostics.itemfit import compute_itemfit
from mirt.diagnostics.modelfit import compute_m2
from mirt.equating._vertical_calibration import _union_item_design
from mirt.equating.vertical import (
    GradeData,
    compute_vertical_diagnostics,
    vertical_scale,
    vertical_scale_summary,
)
from mirt.models.dichotomous import (
    FourParameterLogistic,
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
)


def _grade_fixture(n_persons=600, seed=1934):
    """Generate different forms from one physical bank and known populations."""
    rng = np.random.default_rng(seed)
    slopes = np.linspace(0.65, 1.7, 13)
    locations = np.linspace(-1.5, 1.5, 13)
    forms = [np.arange(8), np.array([8, 4, 5, 6, 7, 9, 10, 11, 12])]
    population_means = [0.0, 0.65]
    population_sds = [1.0, 1.1]
    grade_data, models = [], []
    for grade, columns in enumerate(forms):
        mean, sd = population_means[grade], population_sds[grade]
        theta = rng.normal(mean, sd, n_persons)
        probability = expit(slopes[columns] * (theta[:, None] - locations[columns]))
        responses = (rng.random(probability.shape) < probability).astype(int)
        model = TwoParameterLogistic(len(columns))
        # Separate grade calibrations standardize each population to N(0, 1).
        model.set_parameters(
            discrimination=sd * slopes[columns],
            difficulty=(locations[columns] - mean) / sd,
        )
        model._is_fitted = True
        models.append(model)
        grade_data.append(
            GradeData(
                f"g{grade}",
                responses,
                anchor_items_below=[1, 2, 3, 4] if grade else None,
                anchor_items_above=[4, 5, 6, 7] if not grade else None,
            )
        )
    return grade_data, models, np.array(population_means), np.array(population_sds)


def _three_grade_bridge_fixture():
    """Use disjoint adjacent anchor sets so the upper form needs a real bridge."""
    rng = np.random.default_rng(7148)
    slopes = np.linspace(0.75, 1.65, 11)
    locations = np.linspace(-1.5, 1.5, 11)
    forms = [np.arange(6), np.arange(2, 9), np.arange(6, 11)]
    means = np.array([-0.5, 0.0, 0.7])
    sds = np.array([1.0, 0.95, 1.1])
    grades, models = [], []
    for grade, columns in enumerate(forms):
        theta = rng.normal(means[grade], sds[grade], 600)
        probability = expit(slopes[columns] * (theta[:, None] - locations[columns]))
        responses = (rng.random(probability.shape) < probability).astype(int)
        model = TwoParameterLogistic(len(columns))
        model.set_parameters(
            discrimination=sds[grade] * slopes[columns],
            difficulty=(locations[columns] - means[grade]) / sds[grade],
        )
        model._is_fitted = True
        models.append(model)
        grades.append(
            GradeData(
                f"g{grade}",
                responses,
                anchor_items_below=[0, 1, 2, 3]
                if grade == 1
                else ([0, 1, 2] if grade == 2 else None),
                anchor_items_above=[2, 3, 4, 5]
                if grade == 0
                else ([4, 5, 6] if grade == 1 else None),
            )
        )
    return grades, models, forms, means, sds


def _binary_eap_oracle(model, responses, mean, variance, n_quadpts):
    """Integrate a Gaussian prior directly, without the scoring implementation."""
    nodes, weights = hermgauss(n_quadpts)
    theta = mean + np.sqrt(2.0 * variance) * nodes
    logits = model.discrimination * (theta[:, None] - model.difficulty)
    log_probability = -np.logaddexp(0.0, -logits)
    log_complement = -np.logaddexp(0.0, logits)
    log_joint = (
        (responses == 1) @ log_probability.T
        + (responses == 0) @ log_complement.T
        + np.log(weights / np.sqrt(np.pi))
    )
    posterior = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))
    expected_theta = posterior @ theta
    expected_variance = np.sum(
        posterior * (theta - expected_theta[:, None]) ** 2, axis=1
    )
    return expected_theta, np.sqrt(expected_variance)


def _joint_log_likelihood_oracle(result, grade_data, n_quadpts):
    """Independently evaluate the fitted EM quadrature and missing-item likelihood."""
    nodes, weights = hermgauss(n_quadpts)
    theta = np.sqrt(2.0) * nodes
    base_log_weights = np.log(weights / np.sqrt(np.pi))
    total = 0.0
    for gd in grade_data:
        model = result.calibrated_models[gd.grade_label]
        distribution = result.latent_distributions[gd.grade_label]
        mean, variance = distribution.mean[0], distribution.cov[0, 0]
        log_mass = base_log_weights - 0.5 * (
            (theta - mean) ** 2 / variance - theta**2 + np.log(variance)
        )
        log_mass -= logsumexp(log_mass)
        logits = model.discrimination * (theta[:, None] - model.difficulty)
        log_probability = -np.logaddexp(0.0, -logits)
        log_complement = -np.logaddexp(0.0, logits)
        log_joint = (
            (gd.responses == 1) @ log_probability.T
            + (gd.responses == 0) @ log_complement.T
            + log_mass
        )
        total += np.sum(logsumexp(log_joint, axis=1))
    return float(total)


def test_union_item_design_preserves_positions_and_only_merges_declared_anchors():
    original = [
        np.array([[0, 1, 0, 1], [1, 0, 1, 0]]),
        np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]]),
        np.array([[0, 1, 0], [1, 0, 1]]),
    ]
    grades = [
        GradeData("g0", original[0], anchor_items_above=[3, 1]),
        GradeData(
            "g1", original[1], anchor_items_below=[0, 4], anchor_items_above=[2, 1]
        ),
        GradeData("g2", original[2], anchor_items_below=[1, 2]),
    ]

    design = _union_item_design(grades)

    assert design.n_items == 8
    expected_maps = [[0, 1, 2, 3], [3, 4, 5, 6, 1], [7, 5, 4]]
    for responses, mapping, expected, source in zip(
        design.responses, design.item_maps, expected_maps, original, strict=True
    ):
        assert_array_equal(mapping, expected)
        padded = np.full((2, 8), -1)
        padded[:, expected] = source
        assert_array_equal(responses, padded)
        assert_array_equal(source, responses[:, mapping])
    assert design.members[1] == [(0, 1), (1, 4)]
    assert design.members[4] == [(1, 1), (2, 2)]


@pytest.mark.parametrize("method", ["fixed_anchor", "floating_anchor"])
@pytest.mark.parametrize("reference_grade", [0, 1])
def test_joint_calibration_recovers_population_and_matches_likelihood_and_score_oracles(
    method, reference_grade
):
    grades, models, true_means, true_sds = _grade_fixture()
    original_parameters = [model.parameters for model in models]
    n_quadpts = 31

    result = vertical_scale(
        grades,
        models=models,
        method=method,
        reference_grade=reference_grade,
        enforce_monotonicity=False,
        n_quadpts=n_quadpts,
    )

    expected_means = (true_means - true_means[reference_grade]) / true_sds[
        reference_grade
    ]
    expected_sds = true_sds / true_sds[reference_grade]
    assert result.grade_transformations == {}
    assert result.calibration_result.converged
    assert result.calibration_result.model.n_items == 13
    assert_allclose(list(result.grade_means.values()), expected_means, atol=0.2)
    assert_allclose(list(result.grade_sds.values()), expected_sds, atol=0.2)
    assert result.grade_means[f"g{reference_grade}"] == 0.0
    assert result.grade_sds[f"g{reference_grade}"] == 1.0
    assert_array_equal(result.item_maps["g0"], np.arange(8))
    assert_array_equal(result.item_maps["g1"], [8, 4, 5, 6, 7, 9, 10, 11, 12])

    lower, upper = result.calibrated_models["g0"], result.calibrated_models["g1"]
    for name in lower.parameters:
        assert_array_equal(lower.parameters[name][4:8], upper.parameters[name][1:5])
        if method == "fixed_anchor":
            reference_anchors = [4, 5, 6, 7] if reference_grade == 0 else [1, 2, 3, 4]
            assert_array_equal(
                result.calibrated_models[f"g{reference_grade}"].parameters[name][
                    reference_anchors
                ],
                original_parameters[reference_grade][name][reference_anchors],
            )
    expected_n_parameters = 28 if method == "floating_anchor" else 20
    assert result.calibration_result.n_parameters == expected_n_parameters
    for grade, gd in enumerate(grades):
        fitted_model = result.calibrated_models[gd.grade_label]
        distribution = result.latent_distributions[gd.grade_label]
        theta, standard_error = _binary_eap_oracle(
            fitted_model,
            gd.responses,
            distribution.mean[0],
            distribution.cov[0, 0],
            n_quadpts,
        )
        assert_allclose(result.scores[gd.grade_label].theta, theta, atol=1e-10)
        assert_allclose(
            result.scores[gd.grade_label].standard_error, standard_error, atol=1e-10
        )
        for name, values in models[grade].parameters.items():
            assert_array_equal(values, original_parameters[grade][name])
        assert fitted_model is not models[grade]
        assert fitted_model.item_names == models[grade].item_names
        assert fitted_model.item_names is not models[grade].item_names
        for name, mask in result.free_parameter_masks[gd.grade_label].items():
            expected = fitted_model.free_parameter_masks[name].copy()
            if method == "fixed_anchor":
                expected[[4, 5, 6, 7] if grade == 0 else [1, 2, 3, 4]] = False
            assert_array_equal(mask, expected)

    assert result.calibration_result.log_likelihood == pytest.approx(
        _joint_log_likelihood_oracle(result, grades, n_quadpts), abs=1e-7
    )
    assert result.linking_results[0].fit_statistics.weighted_rmse < 1e-12
    assert (
        compute_vertical_diagnostics(result, grades).anchor_stability[("g0", "g1")]
        < 1e-12
    )
    summary = vertical_scale_summary(result)
    assert "Joint Anchor Calibration" in summary
    assert "Physical items: 13" in summary


def test_floating_anchors_are_learned_from_responses():
    grades, models, _, _ = _grade_fixture(n_persons=400)
    for model in models:
        model.set_parameters(
            discrimination=1.2 * model.discrimination,
            difficulty=model.difficulty + 0.25,
        )
    original = models[0].parameters

    result = vertical_scale(
        grades, models=models, method="floating_anchor", enforce_monotonicity=False
    )

    assert (
        np.max(
            np.abs(
                result.calibrated_models["g0"].difficulty[4:8]
                - original["difficulty"][4:8]
            )
        )
        > 0.05
    )
    assert (
        np.max(
            np.abs(
                result.calibrated_models["g0"].discrimination[4:8]
                - original["discrimination"][4:8]
            )
        )
        > 0.05
    )


@pytest.mark.parametrize("method", ["fixed_anchor", "floating_anchor"])
@pytest.mark.parametrize("reference_grade", [0, 1])
def test_three_grade_disjoint_bridge_anchors_define_one_identified_item_bank(
    method, reference_grade
):
    grades, models, forms, means, sds = _three_grade_bridge_fixture()
    original_parameters = [model.parameters for model in models]

    result = vertical_scale(
        grades,
        models=models,
        method=method,
        reference_grade=reference_grade,
        enforce_monotonicity=False,
    )

    assert result.calibration_result.model.n_items == 11
    assert_allclose(
        list(result.grade_means.values()),
        (means - means[reference_grade]) / sds[reference_grade],
        atol=0.25,
    )
    assert_allclose(
        list(result.grade_sds.values()), sds / sds[reference_grade], atol=0.2
    )
    for grade, form in enumerate(forms):
        assert_array_equal(result.item_maps[f"g{grade}"], form)
    for grade in range(2):
        overlap = np.intersect1d(forms[grade], forms[grade + 1])
        left = np.flatnonzero(np.isin(forms[grade], overlap))
        right = np.flatnonzero(np.isin(forms[grade + 1], overlap))
        for name in models[grade].parameters:
            assert_array_equal(
                result.calibrated_models[f"g{grade}"].parameters[name][left],
                result.calibrated_models[f"g{grade + 1}"].parameters[name][right],
            )
    expected_fixed_items = 4 if reference_grade == 0 else 7
    expected_count = (
        26 if method == "floating_anchor" else 26 - 2 * expected_fixed_items
    )
    assert result.calibration_result.n_parameters == expected_count
    if method == "fixed_anchor":
        reference_anchors = [2, 3, 4, 5] if reference_grade == 0 else np.arange(7)
        for name in models[reference_grade].parameters:
            assert_array_equal(
                result.calibrated_models[f"g{reference_grade}"].parameters[name][
                    reference_anchors
                ],
                original_parameters[reference_grade][name][reference_anchors],
            )
    assert result.calibration_result.log_likelihood == pytest.approx(
        _joint_log_likelihood_oracle(result, grades, 31), abs=1e-7
    )


def test_administered_missing_responses_are_ignored_in_calibration_and_scoring():
    grades, models, _, _ = _grade_fixture(n_persons=350)
    rng = np.random.default_rng(872)
    for gd in grades:
        gd.responses[rng.random(gd.responses.shape) < 0.15] = -2
    originals = [gd.responses.copy() for gd in grades]

    result = vertical_scale(
        grades, models=models, method="fixed_anchor", enforce_monotonicity=False
    )

    assert result.calibration_result.log_likelihood == pytest.approx(
        _joint_log_likelihood_oracle(result, grades, 31), abs=1e-7
    )
    for gd, original in zip(grades, originals, strict=True):
        distribution = result.latent_distributions[gd.grade_label]
        expected_theta, expected_se = _binary_eap_oracle(
            result.calibrated_models[gd.grade_label],
            gd.responses,
            distribution.mean[0],
            distribution.cov[0, 0],
            31,
        )
        assert_allclose(result.scores[gd.grade_label].theta, expected_theta, atol=1e-10)
        assert_allclose(
            result.scores[gd.grade_label].standard_error, expected_se, atol=1e-10
        )
        assert_array_equal(gd.responses, original)


@pytest.mark.parametrize("method", ["fixed_anchor", "floating_anchor"])
def test_failed_joint_calibration_reports_iteration_exhaustion(method):
    grades, models, _, _ = _grade_fixture(n_persons=50)
    with pytest.raises(RuntimeError, match="failed to converge after 1 iterations"):
        vertical_scale(
            grades, models=models, method=method, enforce_monotonicity=False, max_iter=1
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_quadpts": 4}, "n_quadpts"),
        ({"n_quadpts": True}, "n_quadpts"),
        ({"max_iter": 0}, "max_iter"),
        ({"max_iter": 1.5}, "max_iter"),
        ({"tol": 0.0}, "tol"),
        ({"tol": np.nan}, "tol"),
        ({"enforce_monotonicity": "yes"}, "enforce_monotonicity"),
    ],
)
def test_joint_calibration_controls_are_validated_before_fitting(kwargs, message):
    grades, models, _, _ = _grade_fixture(n_persons=10)
    with pytest.raises(ValueError, match=message):
        vertical_scale(grades, models=models, method="floating_anchor", **kwargs)


def test_unobserved_anchor_connection_is_rejected():
    grades, models, _, _ = _grade_fixture(n_persons=10)
    grades[1].responses[:, [1, 2, 3, 4]] = -1
    with pytest.raises(
        ValueError, match="anchors must have observed responses in both grades"
    ):
        vertical_scale(grades, models=models, method="floating_anchor")


def test_globally_unobserved_item_is_rejected():
    grades, models, _, _ = _grade_fixture(n_persons=10)
    grades[0].responses[:, 0] = -1
    with pytest.raises(ValueError, match="contain no observed responses"):
        vertical_scale(grades, models=models, method="floating_anchor")


def test_mixed_calibration_model_families_are_rejected():
    grades, models, _, _ = _grade_fixture(n_persons=10)
    models[1] = ThreeParameterLogistic(9)
    models[1]._is_fitted = True
    with pytest.raises(ValueError, match="one supported built-in model family"):
        vertical_scale(grades, models=models, method="floating_anchor")


def test_shared_polytomous_items_require_matching_category_counts():
    grades = [
        GradeData("g0", np.zeros((10, 2), dtype=int), anchor_items_above=[0, 1]),
        GradeData("g1", np.zeros((10, 3), dtype=int), anchor_items_below=[1, 2]),
    ]
    models = [GradedResponseModel(2, [3, 4]), GradedResponseModel(3, [3, 3, 3])]
    for model in models:
        model._is_fitted = True
    with pytest.raises(ValueError, match="matching category counts"):
        vertical_scale(grades, models=models, method="floating_anchor")


@pytest.mark.parametrize("method", ["fixed_anchor", "floating_anchor"])
def test_anchor_calibration_without_supplied_models_fits_actual_response_data(method):
    grades, _, _, _ = _grade_fixture(n_persons=400)

    result = vertical_scale(grades, method=method, enforce_monotonicity=False)

    assert result.calibration_result.converged
    assert result.calibration_result.model.n_items == 13
    assert result.grade_means["g0"] == 0.0
    assert result.grade_sds["g0"] == 1.0
    assert result.calibrated_models["g0"].n_items == 8
    assert result.calibrated_models["g1"].n_items == 9
    assert result.grade_means["g1"] > 0.2


def test_fixed_anchor_calibration_handles_forms_containing_only_anchors():
    rng = np.random.default_rng(259)
    slopes = np.array([0.65, 0.9, 1.2, 1.5])
    locations = np.array([-1.2, -0.4, 0.4, 1.2])
    grades, models = [], []
    for grade, mean in enumerate([0.0, 0.75]):
        theta = rng.normal(mean, 1.0, 600)
        probability = expit(slopes * (theta[:, None] - locations))
        responses = (rng.random(probability.shape) < probability).astype(int)
        model = TwoParameterLogistic(4)
        model.set_parameters(discrimination=slopes, difficulty=locations - mean)
        model._is_fitted = True
        models.append(model)
        grades.append(
            GradeData(
                f"g{grade}",
                responses,
                anchor_items_below=[0, 1, 2, 3] if grade else None,
                anchor_items_above=[0, 1, 2, 3] if not grade else None,
            )
        )

    result = vertical_scale(
        grades, models=models, method="fixed_anchor", enforce_monotonicity=False
    )

    assert result.calibration_result.n_parameters == 2
    assert result.grade_means["g1"] == pytest.approx(0.75, abs=0.15)
    for model in result.calibrated_models.values():
        assert_array_equal(model.discrimination, slopes)
        assert_array_equal(model.difficulty, locations)
    assert all(
        not np.any(mask)
        for masks in result.free_parameter_masks.values()
        for mask in masks.values()
    )
    calibrated = result.calibrated_models["g0"].copy()
    assert uses_builtin_model_hooks(calibrated, likelihood=True)
    assert calibrated.n_parameters == 0
    assert compute_m2(calibrated, grades[0].responses)["df"] == 10
    fixed_fit = compute_itemfit(calibrated, grades[0].responses, statistics=["S_X2"])
    calibrated.set_free_parameter_masks(None)
    free_fit = compute_itemfit(calibrated, grades[0].responses, statistics=["S_X2"])
    assert_array_equal(fixed_fit["df"], free_fit["df"] + 2)
    assert_array_equal(fixed_fit["S_X2"], free_fit["S_X2"])


def test_fixed_anchor_polytomous_calibration_preserves_category_padding_and_permuted_items():
    rng = np.random.default_rng(348)
    categories = [3, 4, 2]
    slopes = np.array([0.8, 1.1, 1.5])
    reference = GradedResponseModel(3, categories)
    reference.set_parameters(discrimination=slopes)
    reference._is_fitted = True
    thresholds = reference.thresholds.copy()
    permutation = np.array([2, 0, 1])
    upper = GradedResponseModel(3, [categories[item] for item in permutation])
    upper.set_parameters(
        discrimination=slopes[permutation], thresholds=(thresholds - 0.5)[permutation]
    )
    upper._is_fitted = True
    grades = []
    for grade, mean in enumerate([0.0, 0.5]):
        theta = rng.normal(mean, 1.0, 500)
        probability = reference.probability(theta)
        draw = rng.random(probability.shape[:2])
        responses = np.sum(draw[:, :, None] > np.cumsum(probability, axis=2), axis=2)
        if grade:
            responses = responses[:, permutation]
        grades.append(
            GradeData(
                f"g{grade}",
                responses,
                anchor_items_below=np.argsort(permutation).tolist() if grade else None,
                anchor_items_above=[0, 1, 2] if not grade else None,
            )
        )

    result = vertical_scale(
        grades,
        models=[reference, upper],
        method="fixed_anchor",
        enforce_monotonicity=False,
    )

    assert result.calibration_result.n_parameters == 2
    assert result.grade_means["g1"] == pytest.approx(0.5, abs=0.15)
    assert_array_equal(result.calibrated_models["g0"].thresholds, thresholds)
    assert_array_equal(
        result.calibrated_models["g1"].thresholds, thresholds[permutation]
    )
    assert result.calibrated_models["g1"].n_categories == [2, 3, 4]
    assert result.calibrated_models["g0"].n_parameters == 0
    assert result.calibrated_models["g1"].n_parameters == 0
    assert all(
        not np.any(mask)
        for masks in result.free_parameter_masks.values()
        for mask in masks.values()
    )


def test_floating_polytomous_calibration_estimates_unequal_forms_and_variable_categories():
    rng = np.random.default_rng(1826)
    categories = [2, 3, 4, 3, 2, 3]
    bank = GradedResponseModel(6, categories)
    bank.set_parameters(discrimination=np.array([0.9, 1.1, 1.4, 1.0, 0.8, 1.2]))
    forms = [np.array([0, 1, 2, 3]), np.array([4, 3, 1, 2, 5])]
    grades, models = [], []
    for grade, columns in enumerate(forms):
        mean = [0.0, 0.55][grade]
        theta = rng.normal(mean, 1.0, 500)
        probability = bank.probability(theta)
        draw = rng.random(probability.shape[:2])
        responses = np.sum(draw[:, :, None] > np.cumsum(probability, axis=2), axis=2)
        model = GradedResponseModel(
            len(columns), [categories[item] for item in columns]
        )
        model.set_parameters(
            discrimination=bank.discrimination[columns],
            thresholds=(bank.thresholds - mean)[columns],
        )
        model._is_fitted = True
        models.append(model)
        grades.append(
            GradeData(
                f"g{grade}",
                responses[:, columns],
                anchor_items_below=[2, 3, 1] if grade else None,
                anchor_items_above=[1, 2, 3] if not grade else None,
            )
        )

    result = vertical_scale(
        grades, models=models, method="floating_anchor", enforce_monotonicity=False
    )

    assert result.calibration_result.converged
    assert result.calibration_result.n_parameters == 19
    assert result.calibration_result.model.n_items == 6
    assert result.grade_means["g1"] == pytest.approx(0.55, abs=0.2)
    for grade, columns in enumerate(forms):
        model = result.calibrated_models[f"g{grade}"]
        assert model.n_categories == [categories[item] for item in columns]
        assert_array_equal(result.item_maps[f"g{grade}"], columns)
        for item, n_categories in enumerate(model.n_categories):
            assert np.all(np.diff(model.thresholds[item, : n_categories - 1]) > 0)
            assert not np.any(model.thresholds[item, n_categories - 1 :])
            assert not np.any(
                model.free_parameter_masks["thresholds"][item, n_categories - 1 :]
            )
    for name in ["discrimination", "thresholds"]:
        assert_array_equal(
            result.calibrated_models["g0"].parameters[name][[1, 2, 3]],
            result.calibrated_models["g1"].parameters[name][[2, 3, 1]],
        )


@pytest.mark.parametrize(
    "constructor",
    [
        OneParameterLogistic,
        ThreeParameterLogistic,
        FourParameterLogistic,
        GeneralizedPartialCredit,
        PartialCreditModel,
        NominalResponseModel,
    ],
)
def test_fixed_anchor_calibration_supports_identified_binary_and_category_families(
    constructor,
):
    rng = np.random.default_rng(646)
    permutation = np.array([2, 0, 3, 1])
    polytomous = constructor in {
        GeneralizedPartialCredit,
        PartialCreditModel,
        NominalResponseModel,
    }
    reference = constructor(4, **({"n_categories": [2, 3, 4, 3]} if polytomous else {}))
    if not polytomous:
        reference.set_parameters(difficulty=np.array([-1.0, -0.3, 0.4, 1.1]))
    reference._is_fitted = True
    upper = constructor(4, **({"n_categories": [4, 2, 3, 3]} if polytomous else {}))
    upper.set_parameters(
        **{
            name: values[permutation]
            for name, values in reference.parameters.items()
            if np.any(upper.free_parameter_masks[name])
        }
    )
    upper._is_fitted = True
    grades = []
    for grade, mean in enumerate([0.0, 0.5]):
        theta = rng.normal(mean, 1.0, 650)
        probability = reference.probability(theta)
        if polytomous:
            draw = rng.random(probability.shape[:2])
            responses = np.sum(
                draw[:, :, None] > np.cumsum(probability, axis=2), axis=2
            )
        else:
            responses = (rng.random(probability.shape) < probability).astype(int)
        grades.append(
            GradeData(
                f"g{grade}",
                responses[:, permutation] if grade else responses,
                anchor_items_below=np.argsort(permutation).tolist() if grade else None,
                anchor_items_above=list(range(4)) if not grade else None,
            )
        )

    result = vertical_scale(
        grades,
        models=[reference, upper],
        method="fixed_anchor",
        enforce_monotonicity=False,
    )

    assert result.calibration_result.converged
    assert result.calibration_result.n_parameters == 2
    assert result.grade_means["g1"] == pytest.approx(0.5, abs=0.2)
    for grade, columns in enumerate([np.arange(4), permutation]):
        model = result.calibrated_models[f"g{grade}"]
        assert type(model) is constructor
        assert model.n_parameters == 0
        for name, values in reference.parameters.items():
            assert_array_equal(model.parameters[name], values[columns])
        assert uses_builtin_model_hooks(model, likelihood=True)


def test_ordered_calibration_constrains_density_without_shifting_shared_item_models():
    rng = np.random.default_rng(942)
    slopes = np.array([0.65, 0.9, 1.2, 1.5])
    locations = np.array([-1.2, -0.4, 0.4, 1.2])
    grades, models = [], []
    for grade, mean in enumerate([0.0, -0.75]):
        theta = rng.normal(mean, 1.0, 300)
        probability = expit(slopes * (theta[:, None] - locations))
        responses = (rng.random(probability.shape) < probability).astype(int)
        model = TwoParameterLogistic(4)
        model.set_parameters(discrimination=slopes, difficulty=locations - mean)
        model._is_fitted = True
        models.append(model)
        grades.append(
            GradeData(
                f"g{grade}",
                responses,
                anchor_items_below=[0, 1, 2, 3] if grade else None,
                anchor_items_above=[0, 1, 2, 3] if not grade else None,
            )
        )

    result = vertical_scale(
        grades, models=models, method="fixed_anchor", enforce_monotonicity=True
    )

    assert result.grade_means["g1"] >= result.grade_means["g0"]
    assert result.grade_transformations == {}
    for model in result.calibrated_models.values():
        assert_array_equal(model.discrimination, slopes)
        assert_array_equal(model.difficulty, locations)
    assert result.calibration_result.log_likelihood == pytest.approx(
        _joint_log_likelihood_oracle(result, grades, 31), abs=1e-7
    )
    distribution = result.latent_distributions["g1"]
    theta, _ = _binary_eap_oracle(
        result.calibrated_models["g1"],
        grades[1].responses,
        distribution.mean[0],
        distribution.cov[0, 0],
        31,
    )
    assert_allclose(result.scores["g1"].theta, theta, atol=1e-10)
