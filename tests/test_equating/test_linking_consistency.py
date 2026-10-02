"""Affine recovery and statistical oracles for linking workflows."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import least_squares
from scipy.special import expit

from mirt.equating.chain import (
    chain_link,
    concurrent_link,
    transform_theta_to_reference,
    transform_to_reference,
)
from mirt.equating.drift import purify_anchors
from mirt.equating.linking import link
from mirt.equating.vertical import GradeData, vertical_scale
from mirt.models.dichotomous import ThreeParameterLogistic, TwoParameterLogistic
from mirt.scoring import fscores


@pytest.mark.parametrize("reference_index", range(4))
@pytest.mark.parametrize(
    "method",
    [
        "mean_sigma",
        "mean_mean",
        "stocking_lord",
        "haebara",
        "tcc",
        "bisector",
        "orthogonal",
    ],
)
def test_chain_recovers_calibrations_and_response_probabilities(
    method, reference_index
):
    """Each independently constructed metric must map onto any reference."""
    slopes = np.array([0.6, 0.8, 1.0, 1.2, 1.35, 1.5, 1.7, 2.0])
    difficulties = np.array([-1.8, -1.0, -0.4, 0.0, 0.2, 0.7, 1.0, 1.8])
    guessing = np.linspace(0.05, 0.25, 8)
    scales = np.array([1.0, 1.3, 0.72, 1.65])
    shifts = np.array([0.0, 0.4, -0.8, 0.25])
    permutations = [
        np.arange(8),
        np.array([3, 7, 2, 1, 0, 6, 4, 5]),
        np.array([7, 6, 5, 4, 3, 2, 1, 0]),
        np.array([1, 2, 3, 4, 5, 6, 7, 0]),
    ]
    models = []
    for scale, shift, permutation in zip(scales, shifts, permutations, strict=True):
        model = ThreeParameterLogistic(8)
        model.set_parameters(
            discrimination=slopes[permutation] / scale,
            difficulty=scale * difficulties[permutation] + shift,
            guessing=guessing[permutation],
        )
        models.append(model)
    anchors = [np.argsort(permutation)[:6].tolist() for permutation in permutations]
    pairs = list(zip(anchors[:-1], anchors[1:], strict=True))

    result = chain_link(models, pairs, method=method, reference_index=reference_index)

    expected_scales = scales[reference_index] / scales
    expected_shifts = shifts[reference_index] - expected_scales * shifts
    assert_allclose(result.cumulative_A, expected_scales, atol=1e-7)
    assert_allclose(result.cumulative_B, expected_shifts, atol=1e-7)
    canonical_theta = np.array([-2.5, -0.6, 0.0, 1.0, 2.7])
    reference_theta = (
        scales[reference_index] * canonical_theta + shifts[reference_index]
    )
    for index, (model, permutation) in enumerate(
        zip(models, permutations, strict=True)
    ):
        theta = scales[index] * canonical_theta + shifts[index]
        transformed_theta = transform_theta_to_reference(theta, result, index)
        assert_allclose(transformed_theta, reference_theta, atol=1e-7)
        linked_model = transform_to_reference(model, result, index)
        canonical_probabilities = guessing[permutation] + (
            1 - guessing[permutation]
        ) * expit(
            slopes[permutation] * (canonical_theta[:, None] - difficulties[permutation])
        )
        assert_allclose(
            linked_model.probability(reference_theta),
            canonical_probabilities,
            atol=1e-8,
        )


@pytest.mark.parametrize("method", ["stocking_lord", "haebara", "tcc"])
def test_weighted_bootstrap_matches_independent_least_squares_oracle(method):
    """SEs must bootstrap the same population-weighted estimator as the point fit."""
    old_a = np.array([0.65, 0.8, 1.0, 1.3, 1.6, 1.8])
    old_b = np.array([-1.6, -0.9, -0.2, 0.3, 0.9, 1.5])
    new_a = old_a * 1.25 + np.array([0.05, -0.06, 0.01, 0.11, -0.07, 0.08])
    new_b = (old_b - 0.3) / 1.25 + np.array([0.18, -0.12, 0.05, -0.1, 0.2, -0.06])
    theta = np.linspace(-4.0, 4.0, 41)
    weights = np.exp(-0.5 * ((theta - 1.8) / 0.7) ** 2)
    weights /= weights.sum()
    models = [TwoParameterLogistic(6), TwoParameterLogistic(6)]
    models[0].set_parameters(discrimination=old_a, difficulty=old_b)
    models[1].set_parameters(discrimination=new_a, difficulty=new_b)
    seed, n_bootstrap = 713, 12

    fitted = link(
        *models,
        list(range(6)),
        list(range(6)),
        method=method,
        weights=weights,
        n_theta=len(theta),
        compute_se=True,
        n_bootstrap=n_bootstrap,
        random_state=seed,
    )

    def fit_sample(indices):
        old_curves = expit(old_a[indices] * (theta[:, None] - old_b[indices]))

        def residual(parameters):
            scale, shift = np.exp(parameters[0]), parameters[1]
            new_theta = (theta - shift) / scale
            new_curves = expit(new_a[indices] * (new_theta[:, None] - new_b[indices]))
            difference = old_curves - new_curves
            if method != "haebara":
                return np.sqrt(weights) * difference.sum(axis=1)
            return (np.sqrt(weights[:, None]) * difference).ravel()

        result = least_squares(residual, [0.0, 0.0], xtol=1e-12, ftol=1e-12, gtol=1e-12)
        assert result.success
        return np.array([np.exp(result.x[0]), result.x[1]])

    assert_allclose(
        [fitted.constants.A, fitted.constants.B], fit_sample(np.arange(6)), atol=1e-7
    )
    rng = np.random.default_rng(seed)
    samples = np.array(
        [fit_sample(rng.choice(6, size=6, replace=True)) for _ in range(n_bootstrap)]
    )
    assert_allclose(
        [fitted.constants.A_se, fitted.constants.B_se],
        samples.std(axis=0, ddof=1),
        rtol=1e-5,
        atol=1e-8,
    )


@pytest.mark.parametrize("standalone", [False, True])
def test_robust_purification_retains_anchors_stable_under_the_requested_estimator(
    standalone,
):
    """A robust final fit must not retain anchors flagged by that same fit."""
    old_a = np.array([1.9006, 1.7088, 0.5195, 1.9017, 1.3161, 0.7820, 0.7516, 1.0328])
    old_b = np.array(
        [1.0441, -1.8506, -0.1436, 0.3899, 0.4413, 0.5001, -0.7387, -2.8926]
    )
    new_a = np.array([3.0864, 1.7886, 0.6315, 2.3420, 1.4931, 0.9558, 0.8110, 1.2516])
    new_b = np.array(
        [0.9254, -1.9508, -0.4126, -0.0595, 0.1852, 0.2234, -0.8729, -2.7361]
    )
    models = [TwoParameterLogistic(8), TwoParameterLogistic(8)]
    models[0].set_parameters(discrimination=old_a, difficulty=old_b)
    models[1].set_parameters(discrimination=new_a, difficulty=new_b)

    if standalone:
        retained_old, retained_new, _ = purify_anchors(
            *models, list(range(8)), list(range(8)), method="mean_mean", robust=True
        )
        fitted = link(
            *models, retained_old, retained_new, method="mean_mean", robust=True
        )
    else:
        fitted = link(
            *models,
            list(range(8)),
            list(range(8)),
            method="mean_mean",
            robust=True,
            purify_anchors=True,
        )
    retained = fitted.anchor_items
    assert len(retained) > 3
    stable = link(*models, retained, retained, method="mean_mean", robust=True)
    assert not stable.anchor_diagnostics.flagged.any()
    assert_allclose(
        [fitted.constants.A, fitted.constants.B],
        [stable.constants.A, stable.constants.B],
        atol=1e-14,
    )


def test_concurrent_link_reports_iteration_exhaustion():
    old = TwoParameterLogistic(6)
    old.set_parameters(
        discrimination=np.linspace(0.6, 1.8, 6), difficulty=np.linspace(-1.5, 1.5, 6)
    )
    new = TwoParameterLogistic(6)
    new.set_parameters(
        discrimination=np.linspace(0.6, 1.8, 6) / 1.8,
        difficulty=1.8 * np.linspace(-1.5, 1.5, 6) + 0.6,
    )

    with pytest.raises(RuntimeError, match="failed to converge"):
        concurrent_link(
            [old, new],
            [[[(item, item) for item in range(6)]]],
            max_iter=1,
            tol=1e-12,
        )


@pytest.mark.parametrize("reference_index", [0, 1])
def test_concurrent_reference_probabilities_are_evaluated_once(
    monkeypatch, reference_index
):
    models = [TwoParameterLogistic(6), TwoParameterLogistic(6)]
    models[0].set_parameters(
        discrimination=np.linspace(0.6, 1.8, 6), difficulty=np.linspace(-1.5, 1.5, 6)
    )
    models[1].set_parameters(
        discrimination=models[0].discrimination / 1.4,
        difficulty=1.4 * models[0].difficulty + 0.6,
    )
    counts = [0, 0]
    for index, model in enumerate(models):
        original = model.probability

        def counted(theta, item_idx=None, *, _index=index, _original=original):
            counts[_index] += 1
            return _original(theta, item_idx)

        monkeypatch.setattr(model, "probability", counted)

    constants = concurrent_link(
        models,
        [[[(item, item) for item in range(6)]]],
        max_iter=100,
        tol=1e-10,
        reference_index=reference_index,
    )

    expected = (
        [[1.0, 0.0], [1.0 / 1.4, -0.6 / 1.4]]
        if reference_index == 0
        else [[1.4, 0.6], [1.0, 0.0]]
    )
    assert_allclose(constants, expected, atol=2e-5)
    assert counts[reference_index] == 1
    assert counts[1 - reference_index] > 5


@pytest.mark.parametrize("reference_grade", range(3))
@pytest.mark.parametrize("method", ["stocking_lord", "haebara"])
def test_concurrent_vertical_scale_matches_joint_curve_and_score_oracles(
    method, reference_grade
):
    """A noisy three-form design distinguishes joint fitting from pairwise linking."""
    rng = np.random.default_rng(1803)
    slopes = np.array([0.7, 0.9, 1.1, 1.3, 1.5, 1.8])
    difficulties = np.array([-1.5, -0.8, -0.1, 0.3, 0.8, 1.6])
    scales = [1.0, 1.4, 0.85]
    shifts = [0.0, 0.6, -0.35]
    slope_drift = [
        np.zeros(6),
        np.array([0.08, -0.08, 0.03, 0.02, -0.1, 0.06]),
        np.array([-0.06, 0.1, -0.04, 0.12, 0.02, -0.06]),
    ]
    location_drift = [
        np.zeros(6),
        np.array([0.16, -0.05, 0.18, -0.08, 0.02, -0.2]),
        np.array([-0.12, 0.2, 0.05, -0.15, 0.12, 0.08]),
    ]
    models, grade_data = [], []
    anchors = list(range(6))
    for index in range(3):
        model = TwoParameterLogistic(6)
        model.set_parameters(
            discrimination=slopes / scales[index] + slope_drift[index],
            difficulty=scales[index] * difficulties
            + shifts[index]
            + location_drift[index],
        )
        model._is_fitted = True
        models.append(model)
        grade_data.append(
            GradeData(
                f"g{index}",
                rng.integers(0, 2, (20, 6)),
                anchor_items_below=anchors if index else None,
                anchor_items_above=anchors if index < 2 else None,
            )
        )
    theta = np.linspace(-4.0, 4.0, 61)
    weights = np.exp(-0.5 * theta**2)
    weights /= weights.sum()
    free_indices = [index for index in range(3) if index != reference_grade]

    def residual(parameters):
        common_A = np.ones(3)
        common_B = np.zeros(3)
        common_A[free_indices] = np.exp(parameters[:2])
        common_B[free_indices] = parameters[2:]
        curves = [
            expit(model.discrimination * ((theta[:, None] - B) / A - model.difficulty))
            for model, A, B in zip(models, common_A, common_B, strict=True)
        ]
        differences = [curves[index] - curves[index + 1] for index in range(2)]
        if method == "stocking_lord":
            return np.concatenate(
                [
                    np.sqrt(weights) * difference.sum(axis=1)
                    for difference in differences
                ]
            )
        return np.concatenate(
            [
                (np.sqrt(weights[:, None]) * difference).ravel()
                for difference in differences
            ]
        )

    oracle = least_squares(residual, np.zeros(4), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    assert oracle.success
    common_A = np.ones(3)
    common_B = np.zeros(3)
    common_A[free_indices] = np.exp(oracle.x[:2])
    common_B[free_indices] = oracle.x[2:]
    expected_A = common_A / common_A[reference_grade]
    expected_B = (common_B - common_B[reference_grade]) / common_A[reference_grade]

    result = vertical_scale(
        grade_data,
        models=models,
        method="concurrent",
        linking_method=method,
        reference_grade=reference_grade,
        enforce_monotonicity=False,
    )

    for index, (model, gd) in enumerate(zip(models, grade_data, strict=True)):
        assert_allclose(
            result.grade_transformations[gd.grade_label],
            [expected_A[index], expected_B[index]],
            atol=3e-6,
        )
        native_theta = fscores(model, gd.responses, method="EAP").theta
        expected_theta = expected_A[index] * native_theta + expected_B[index]
        assert result.grade_means[gd.grade_label] == pytest.approx(
            expected_theta.mean(), abs=3e-6
        )
        assert result.grade_sds[gd.grade_label] == pytest.approx(
            expected_theta.std(ddof=1), abs=3e-6
        )

    for index, fitted in enumerate(result.linking_results):
        A = expected_A[index + 1] / expected_A[index]
        B = (expected_B[index + 1] - expected_B[index]) / expected_A[index]
        assert_allclose([fitted.constants.A, fitted.constants.B], [A, B], atol=3e-6)
        old, new = models[index : index + 2]
        rmse_a = np.sqrt(np.mean((old.discrimination - new.discrimination / A) ** 2))
        rmse_b = np.sqrt(np.mean((old.difficulty - (A * new.difficulty + B)) ** 2))
        assert fitted.fit_statistics.weighted_rmse == pytest.approx(
            np.hypot(rmse_a, rmse_b), abs=3e-6
        )


def test_concurrent_vertical_rejects_moment_matching_method():
    grades = [
        GradeData("g0", np.zeros((2, 2), dtype=int), anchor_items_above=[0, 1]),
        GradeData("g1", np.zeros((2, 2), dtype=int)),
    ]
    with pytest.raises(ValueError, match="curve-matching linker"):
        vertical_scale(grades, method="concurrent", linking_method="mean_sigma")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "invalid"}, "Unknown linking method"),
        ({"min_anchors": 1}, "min_anchors"),
        ({"min_anchors": 2.5}, "min_anchors"),
        ({"max_iterations": 0}, "max_iterations"),
        ({"max_iterations": True}, "max_iterations"),
        ({"threshold": np.nan}, "threshold"),
        ({"threshold": 0}, "threshold"),
        ({"weights": np.zeros(61)}, "positive sum"),
    ],
)
def test_purification_validates_configuration_even_when_no_removal_is_needed(
    kwargs, message
):
    models = [TwoParameterLogistic(2), TwoParameterLogistic(2)]
    with pytest.raises(ValueError, match=message):
        purify_anchors(*models, [0, 1], [0, 1], **kwargs)
