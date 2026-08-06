"""Regression tests for multidimensional linking and target rotation."""

from __future__ import annotations

import numpy as np
import pytest

import mirt
from mirt.equating.multidimensional import (
    ProcrustesResult,
    compute_mirt_linking_fit,
    factor_congruence_coefficient,
    link_mirt,
    match_factors,
    mirt_linking_summary,
    oblique_procrustes_rotation,
    orthogonal_procrustes_rotation,
    target_rotation,
    transform_mirt_parameters,
    transform_mirt_theta,
)
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.multidimensional import MultidimensionalModel

SOURCE_SLOPES = np.array(
    [
        [1.0, 0.2],
        [0.3, 1.1],
        [0.8, -0.4],
        [-0.2, 0.9],
        [1.2, 0.7],
        [0.4, -1.0],
    ]
)
SOURCE_INTERCEPTS = np.array([0.3, -0.2, 0.8, -0.7, 0.1, 0.5])
KNOWN_ROTATION = np.array([[0.0, 1.0], [1.0, 0.0]])
KNOWN_SCALE = 1.35
KNOWN_SHIFT = np.array([0.25, -0.4])


def make_model(
    slopes: np.ndarray = SOURCE_SLOPES,
    intercepts: np.ndarray = SOURCE_INTERCEPTS,
) -> MultidimensionalModel:
    model = MultidimensionalModel(slopes.shape[0], slopes.shape[1])
    model.set_parameters(slopes=slopes, intercepts=intercepts)
    return model


def make_exact_pair() -> tuple[MultidimensionalModel, MultidimensionalModel]:
    source = make_model()
    target_slopes = KNOWN_SCALE * SOURCE_SLOPES @ KNOWN_ROTATION
    target_intercepts = (
        SOURCE_INTERCEPTS - (SOURCE_SLOPES @ KNOWN_ROTATION) @ KNOWN_SHIFT
    )
    return make_model(target_slopes, target_intercepts), source


class TestOrthogonalProcrustes:
    def test_recovers_reflection_and_scaling(self):
        target = KNOWN_SCALE * SOURCE_SLOPES @ KNOWN_ROTATION

        rotation, scale, rmse = orthogonal_procrustes_rotation(target, SOURCE_SLOPES)

        np.testing.assert_allclose(rotation, KNOWN_ROTATION, atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(-1.0)
        assert scale == pytest.approx(KNOWN_SCALE)
        assert rmse == pytest.approx(0.0, abs=1e-12)

    def test_can_disable_scaling(self):
        target = SOURCE_SLOPES @ KNOWN_ROTATION

        rotation, scale, rmse = orthogonal_procrustes_rotation(
            target, SOURCE_SLOPES, scaling=False
        )

        np.testing.assert_allclose(rotation, KNOWN_ROTATION, atol=1e-12)
        assert scale == 1.0
        assert rmse == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize(
        ("target", "source", "match"),
        [
            (np.ones(3), np.ones((3, 1)), "two-dimensional"),
            (np.ones((3, 2)), np.ones((4, 2)), "same shape"),
            (
                np.array([[1.0, np.nan], [0.0, 1.0]]),
                np.eye(2),
                "finite",
            ),
            (np.ones((1, 2)), np.ones((1, 2)), "at least 2 rows"),
            (np.eye(2), np.ones((2, 2)), "A_source must have full"),
            (np.ones((2, 2)), np.eye(2), "A_target must have full"),
        ],
    )
    def test_validates_matrices(self, target, source, match):
        with pytest.raises(ValueError, match=match):
            orthogonal_procrustes_rotation(target, source)

    def test_validates_scaling_flag(self):
        with pytest.raises(ValueError, match="scaling must be boolean"):
            orthogonal_procrustes_rotation(np.eye(2), np.eye(2), scaling=1)


class TestObliqueProcrustes:
    def test_recovers_general_linear_transformation(self):
        transform = np.array([[1.2, 0.35], [-0.15, 0.75]])
        target = SOURCE_SLOPES @ transform

        rotation, scale, rmse = oblique_procrustes_rotation(
            target, SOURCE_SLOPES, gamma=1.0
        )

        np.testing.assert_allclose(scale * rotation, transform, atol=1e-12)
        assert rmse == pytest.approx(0.0, abs=1e-12)

    def test_can_return_combined_transform_without_scaling(self):
        transform = np.array([[1.2, 0.35], [-0.15, 0.75]])
        target = SOURCE_SLOPES @ transform

        rotation, scale, rmse = oblique_procrustes_rotation(
            target, SOURCE_SLOPES, gamma=1.0, scaling=False
        )

        np.testing.assert_allclose(rotation, transform, atol=1e-12)
        assert scale == 1.0
        assert rmse == pytest.approx(0.0, abs=1e-12)

    def test_gamma_zero_matches_orthogonal_solution(self):
        target = SOURCE_SLOPES @ np.array([[1.2, 0.3], [0.0, 0.8]])
        expected = orthogonal_procrustes_rotation(target, SOURCE_SLOPES)
        actual = oblique_procrustes_rotation(target, SOURCE_SLOPES, gamma=0.0)

        for actual_value, expected_value in zip(actual, expected, strict=True):
            np.testing.assert_allclose(actual_value, expected_value)

    def test_regularization_interpolates_fit(self):
        target = SOURCE_SLOPES @ np.array([[1.2, 0.3], [0.0, 0.8]])
        _, _, orthogonal_rmse = oblique_procrustes_rotation(
            target, SOURCE_SLOPES, gamma=0.0
        )
        _, _, middle_rmse = oblique_procrustes_rotation(
            target, SOURCE_SLOPES, gamma=0.5
        )
        _, _, general_rmse = oblique_procrustes_rotation(
            target, SOURCE_SLOPES, gamma=1.0
        )

        assert general_rmse < middle_rmse < orthogonal_rmse

    @pytest.mark.parametrize("gamma", [-0.1, 1.1, np.nan, np.inf])
    def test_validates_gamma(self, gamma):
        with pytest.raises(ValueError, match="gamma"):
            oblique_procrustes_rotation(np.eye(2), np.eye(2), gamma=gamma)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"max_iter": 0}, "max_iter"),
            ({"max_iter": 1.5}, "max_iter"),
            ({"tol": 0.0}, "tol"),
            ({"tol": np.nan}, "tol"),
            ({"scaling": 1}, "scaling"),
        ],
    )
    def test_validates_controls(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            oblique_procrustes_rotation(np.eye(2), np.eye(2), **kwargs)


class TestLinkMirt:
    def test_documented_model_is_supported(self):
        target, source = make_exact_pair()

        result = link_mirt(target, source, list(range(6)), list(range(6)))

        np.testing.assert_allclose(result.rotation_matrix, KNOWN_ROTATION, atol=1e-12)
        np.testing.assert_allclose(result.translation, KNOWN_SHIFT, atol=1e-12)
        assert result.scaling == pytest.approx(KNOWN_SCALE)
        assert result.rmse == pytest.approx(0.0, abs=1e-12)
        assert result.intercept_rmse == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(result.transformed_loadings, target.slopes)
        np.testing.assert_allclose(
            result.transformed_intercepts, target.intercepts, atol=1e-12
        )
        assert result.anchor_items_old == tuple(range(6))
        assert result.anchor_items_new == tuple(range(6))

    def test_anchor_mappings_can_differ(self):
        target, source = make_exact_pair()
        permutation = [3, 0, 5, 1, 4, 2]
        permuted_source = make_model(
            SOURCE_SLOPES[permutation], SOURCE_INTERCEPTS[permutation]
        )

        result = link_mirt(
            target,
            permuted_source,
            list(range(6)),
            list(np.argsort(permutation)),
        )

        np.testing.assert_allclose(
            result.transformed_loadings[np.argsort(permutation)], target.slopes
        )

    def test_translation_can_be_disabled(self):
        target_slopes = KNOWN_SCALE * SOURCE_SLOPES @ KNOWN_ROTATION
        target = make_model(target_slopes, SOURCE_INTERCEPTS)
        source = make_model()

        result = link_mirt(
            target,
            source,
            list(range(6)),
            list(range(6)),
            translation=False,
        )

        np.testing.assert_array_equal(result.translation, np.zeros(2))
        np.testing.assert_allclose(result.transformed_intercepts, SOURCE_INTERCEPTS)

    def test_oblique_link_recovers_general_transform(self):
        transform = np.array([[1.2, 0.35], [-0.15, 0.75]])
        scale = np.linalg.norm(transform) / np.sqrt(2)
        rotation = transform / scale
        target = make_model(
            SOURCE_SLOPES @ transform,
            SOURCE_INTERCEPTS - (SOURCE_SLOPES @ rotation) @ KNOWN_SHIFT,
        )

        result = link_mirt(
            target,
            make_model(),
            list(range(6)),
            list(range(6)),
            rotation="oblique",
            gamma=1.0,
        )

        np.testing.assert_allclose(
            result.scaling * result.rotation_matrix, transform, atol=1e-12
        )
        np.testing.assert_allclose(result.translation, KNOWN_SHIFT, atol=1e-12)

    def test_supports_multidimensional_2pl_parameterization(self):
        source = TwoParameterLogistic(n_items=6, n_factors=2)
        source.set_parameters(
            discrimination=SOURCE_SLOPES,
            difficulty=np.array([-0.3, 0.2, -0.1, 0.6, -0.4, 0.15]),
        )
        target = transform_mirt_parameters(
            source, KNOWN_ROTATION, KNOWN_SHIFT, KNOWN_SCALE
        )

        result = link_mirt(target, source, list(range(6)), list(range(6)))

        np.testing.assert_allclose(result.rotation_matrix, KNOWN_ROTATION, atol=1e-12)
        np.testing.assert_allclose(result.translation, KNOWN_SHIFT, atol=1e-12)
        assert result.scaling == pytest.approx(KNOWN_SCALE)

    @pytest.mark.parametrize("rotation", ["", "general", "ORTHOGONAL"])
    def test_rejects_unknown_rotation(self, rotation):
        target, source = make_exact_pair()
        with pytest.raises(ValueError, match="rotation must"):
            link_mirt(target, source, [0, 1], [0, 1], rotation=rotation)

    @pytest.mark.parametrize(
        ("old_indices", "new_indices", "error", "match"),
        [
            ([], [], ValueError, "at least one"),
            ([0], [0], ValueError, "at least 2"),
            ([0, 1, 2], [0, 1], ValueError, "same length"),
            ([0.0, 1.0], [0, 1], ValueError, "integer"),
            ([True, False], [0, 1], ValueError, "integer"),
            ([0, 0], [0, 1], ValueError, "duplicate"),
            ([-1, 0], [0, 1], IndexError, "outside"),
            ([0, 6], [0, 1], IndexError, "outside"),
        ],
    )
    def test_validates_anchors(self, old_indices, new_indices, error, match):
        target, source = make_exact_pair()
        with pytest.raises(error, match=match):
            link_mirt(target, source, old_indices, new_indices)

    def test_rejects_rank_deficient_anchor_set(self):
        source_slopes = SOURCE_SLOPES.copy()
        source_slopes[1] = 2.0 * source_slopes[0]
        target = make_model(KNOWN_SCALE * source_slopes @ KNOWN_ROTATION)
        source = make_model(source_slopes)

        with pytest.raises(ValueError, match="full column rank"):
            link_mirt(target, source, [0, 1], [0, 1])

    def test_rejects_factor_count_mismatch(self):
        target = MultidimensionalModel(6, 3)
        source = make_model()
        with pytest.raises(ValueError, match="same number of factors"):
            link_mirt(target, source, [0, 1, 2], [0, 1, 2])

    def test_rejects_unsupported_parameterization(self):
        class Unsupported:
            parameters = {"values": np.ones((4, 2))}
            n_items = 4
            n_factors = 2

        with pytest.raises(ValueError, match="must use either"):
            link_mirt(Unsupported(), Unsupported(), [0, 1], [0, 1])

    @pytest.mark.parametrize(
        ("keyword", "value", "match"),
        [("scaling", 1, "scaling"), ("translation", 0, "translation")],
    )
    def test_validates_boolean_flags(self, keyword, value, match):
        target, source = make_exact_pair()
        kwargs = {keyword: value}
        with pytest.raises(ValueError, match=match):
            link_mirt(target, source, [0, 1], [0, 1], **kwargs)


class TestParameterAndThetaTransforms:
    def test_parameter_transform_returns_independent_copy(self):
        source = make_model()

        transformed = transform_mirt_parameters(
            source, KNOWN_ROTATION, KNOWN_SHIFT, KNOWN_SCALE
        )

        assert transformed is not source
        np.testing.assert_allclose(source.slopes, SOURCE_SLOPES)
        np.testing.assert_allclose(source.intercepts, SOURCE_INTERCEPTS)
        np.testing.assert_allclose(
            transformed.slopes, KNOWN_SCALE * SOURCE_SLOPES @ KNOWN_ROTATION
        )
        np.testing.assert_allclose(
            transformed.intercepts,
            SOURCE_INTERCEPTS - (SOURCE_SLOPES @ KNOWN_ROTATION) @ KNOWN_SHIFT,
        )

    def test_parameter_transform_can_modify_in_place(self):
        source = make_model()

        result = transform_mirt_parameters(source, KNOWN_ROTATION, in_place=True)

        assert result is source
        np.testing.assert_allclose(source.slopes, SOURCE_SLOPES @ KNOWN_ROTATION)

    def test_item_and_theta_transforms_preserve_probabilities(self):
        source = make_model()
        source_theta = np.array([[-1.0, 0.5], [0.2, -0.7], [1.5, 1.1]])
        linked_model = transform_mirt_parameters(
            source, KNOWN_ROTATION, KNOWN_SHIFT, KNOWN_SCALE
        )
        linked_theta = transform_mirt_theta(
            source_theta, KNOWN_ROTATION, KNOWN_SHIFT, KNOWN_SCALE
        )

        np.testing.assert_allclose(
            linked_model.probability(linked_theta),
            source.probability(source_theta),
            atol=1e-13,
        )

    def test_two_parameter_logistic_transform_preserves_probabilities(self):
        source = TwoParameterLogistic(n_items=6, n_factors=2)
        source.set_parameters(
            discrimination=SOURCE_SLOPES,
            difficulty=np.array([-0.3, 0.2, -0.1, 0.6, -0.4, 0.15]),
        )
        theta = np.array([[-0.5, 0.2], [1.0, -0.4]])

        linked = transform_mirt_parameters(
            source, KNOWN_ROTATION, KNOWN_SHIFT, KNOWN_SCALE
        )
        linked_theta = transform_mirt_theta(
            theta, KNOWN_ROTATION, KNOWN_SHIFT, KNOWN_SCALE
        )

        np.testing.assert_allclose(
            linked.probability(linked_theta), source.probability(theta), atol=1e-13
        )

    def test_theta_vector_preserves_dimensionality(self):
        theta = np.array([0.2, -0.7])
        transformed = transform_mirt_theta(
            theta, KNOWN_ROTATION, KNOWN_SHIFT, KNOWN_SCALE
        )
        assert transformed.shape == (2,)

    @pytest.mark.parametrize(
        ("R", "t", "s", "match"),
        [
            (np.eye(3), None, 1.0, "shape"),
            (np.ones((2, 2)), None, 1.0, "nonsingular"),
            (np.eye(2), np.ones(3), 1.0, "t must have shape"),
            (np.eye(2), np.array([0.0, np.nan]), 1.0, "finite"),
            (np.eye(2), None, 0.0, "positive"),
        ],
    )
    def test_parameter_transform_validation(self, R, t, s, match):
        with pytest.raises(ValueError, match=match):
            transform_mirt_parameters(make_model(), R, t=t, s=s)

    @pytest.mark.parametrize(
        ("theta", "R", "t", "s", "match"),
        [
            (np.ones(3), np.eye(2), None, 1.0, "theta must have shape"),
            (np.ones((2, 3)), np.eye(2), None, 1.0, "theta must have shape"),
            (np.array([0.0, np.nan]), np.eye(2), None, 1.0, "finite"),
            (np.ones(2), np.ones((2, 2)), None, 1.0, "nonsingular"),
            (np.ones(2), np.eye(2), np.ones(3), 1.0, "t must have shape"),
            (np.ones(2), np.eye(2), None, -1.0, "positive"),
        ],
    )
    def test_theta_transform_validation(self, theta, R, t, s, match):
        with pytest.raises(ValueError, match=match):
            transform_mirt_theta(theta, R, t=t, s=s)

    def test_confirmatory_pattern_rejects_incompatible_rotation(self):
        pattern = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        model = MultidimensionalModel(
            4, 2, model_type="confirmatory", loading_pattern=pattern
        )

        with pytest.raises(ValueError, match="fixed loading pattern"):
            transform_mirt_parameters(model, KNOWN_ROTATION)

    def test_new_theta_transform_is_available_from_top_level(self):
        assert mirt.transform_mirt_theta is transform_mirt_theta


class TestFactorCongruenceAndMatching:
    def test_congruence_supports_rectangular_factor_sets(self):
        first = SOURCE_SLOPES[:, :1]
        second = np.column_stack((SOURCE_SLOPES, SOURCE_SLOPES[:, 0] * 2.0))

        result = factor_congruence_coefficient(first, second)

        assert result.shape == (1, 3)
        assert result[0, 0] == pytest.approx(1.0)
        assert result[0, 2] == pytest.approx(1.0)

    def test_congruence_tracks_factor_sign(self):
        result = factor_congruence_coefficient(
            SOURCE_SLOPES, SOURCE_SLOPES * np.array([-2.0, 3.0])
        )
        assert result[0, 0] == pytest.approx(-1.0)
        assert result[1, 1] == pytest.approx(1.0)

    def test_congruence_matches_manual_formula(self):
        first = SOURCE_SLOPES
        second = SOURCE_SLOPES @ np.array([[0.8, 0.3], [-0.2, 1.1]])
        manual = np.empty((2, 2))
        for i in range(2):
            for j in range(2):
                manual[i, j] = np.dot(first[:, i], second[:, j]) / (
                    np.linalg.norm(first[:, i]) * np.linalg.norm(second[:, j])
                )
        np.testing.assert_allclose(factor_congruence_coefficient(first, second), manual)

    @pytest.mark.parametrize(
        ("first", "second", "match"),
        [
            (np.ones((3, 2)), np.ones((4, 2)), "same number of rows"),
            (np.column_stack((np.ones(3), np.zeros(3))), np.ones((3, 2)), "zero"),
            (np.ones((3, 2)), np.array([[1.0, np.nan]] * 3), "finite"),
        ],
    )
    def test_congruence_validation(self, first, second, match):
        with pytest.raises(ValueError, match=match):
            factor_congruence_coefficient(first, second)

    def test_match_factors_permutes_and_sign_aligns(self):
        target = np.column_stack(
            (SOURCE_SLOPES, SOURCE_SLOPES[:, 0] - SOURCE_SLOPES[:, 1])
        )
        source = target[:, [2, 0, 1]] * np.array([-1.0, 1.0, -1.0])
        original = source.copy()

        matched, order = match_factors(target, source)

        assert order == [1, 2, 0]
        np.testing.assert_allclose(matched, target)
        np.testing.assert_array_equal(source, original)

    @pytest.mark.parametrize(
        ("target", "source", "match"),
        [
            (np.ones((3, 2)), np.ones((4, 2)), "same number of rows"),
            (np.ones((3, 2)), np.ones((3, 3)), "same number of factors"),
        ],
    )
    def test_match_validation(self, target, source, match):
        with pytest.raises(ValueError, match=match):
            match_factors(target, source)


class TestTargetRotation:
    def test_orthogonal_rotation_uses_target_zeros(self):
        loadings = np.array([[1.0, 1.0], [1.0, -1.0]])
        expected_rotation = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
        target = loadings @ expected_rotation

        rotated, rotation = target_rotation(loadings, target)

        np.testing.assert_allclose(rotation, expected_rotation, atol=1e-12)
        np.testing.assert_allclose(rotated, target, atol=1e-12)
        assert target[0, 1] == pytest.approx(0.0, abs=1e-12)
        assert target[1, 0] == pytest.approx(0.0, abs=1e-12)

    def test_oblique_rotation_recovers_general_transform(self):
        transform = np.array([[1.1, 0.3], [-0.2, 0.8]])
        target = SOURCE_SLOPES @ transform

        rotated, rotation = target_rotation(
            SOURCE_SLOPES, target, rotation_type="oblique"
        )

        np.testing.assert_allclose(rotation, transform, atol=1e-12)
        np.testing.assert_allclose(rotated, target, atol=1e-12)

    @pytest.mark.parametrize(
        ("A", "T", "rotation_type", "match"),
        [
            (np.ones((4, 2)), np.ones((4, 2)), "orthogonal", "full column rank"),
            (np.eye(2), np.ones((2, 2)), "oblique", "singular"),
            (np.eye(2), np.ones((3, 2)), "orthogonal", "same shape"),
            (np.eye(2), np.eye(2), "invalid", "rotation_type"),
        ],
    )
    def test_validation(self, A, T, rotation_type, match):
        with pytest.raises(ValueError, match=match):
            target_rotation(A, T, rotation_type=rotation_type)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [({"max_iter": 0}, "max_iter"), ({"tol": 0}, "tol")],
    )
    def test_control_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            target_rotation(np.eye(2), np.eye(2), **kwargs)


class TestFitAndSummary:
    def test_exact_fit_statistics(self):
        target, source = make_exact_pair()
        result = link_mirt(target, source, list(range(6)), list(range(6)))

        fit = compute_mirt_linking_fit(
            target, source, list(range(6)), list(range(6)), result
        )

        assert fit["rmse"] == pytest.approx(0.0, abs=1e-12)
        assert fit["intercept_rmse"] == pytest.approx(0.0, abs=1e-12)
        assert fit["mean_congruence"] == pytest.approx(1.0)
        assert fit["r_squared"] == pytest.approx(1.0)
        assert fit["scaling"] == pytest.approx(KNOWN_SCALE)

    def test_constant_target_has_finite_r_squared(self):
        slopes = np.tile(np.array([1.0, 2.0]), (4, 1))
        target = make_model(slopes, np.zeros(4))
        source = make_model(slopes, np.zeros(4))
        result = ProcrustesResult(np.eye(2), np.zeros(2), 1.0, 0.0, slopes.copy())

        fit = compute_mirt_linking_fit(
            target, source, list(range(4)), list(range(4)), result
        )

        assert fit["r_squared"] == 1.0

    def test_fit_validates_transformed_shapes(self):
        target, source = make_exact_pair()
        result = ProcrustesResult(np.eye(2), np.zeros(2), 1.0, 0.0, np.ones((3, 2)))
        with pytest.raises(ValueError, match="shape must match"):
            compute_mirt_linking_fit(
                target, source, list(range(6)), list(range(6)), result
            )

    def test_fit_validates_transformed_intercepts(self):
        target, source = make_exact_pair()
        result = link_mirt(target, source, list(range(6)), list(range(6)))
        result.transformed_intercepts = np.ones(3)
        with pytest.raises(ValueError, match="intercepts shape"):
            compute_mirt_linking_fit(
                target, source, list(range(6)), list(range(6)), result
            )

    def test_summary_uses_retained_anchor_mapping_for_different_forms(self):
        target, source = make_exact_pair()
        larger_source = make_model(
            np.vstack((SOURCE_SLOPES, [[0.7, 0.6]])),
            np.append(SOURCE_INTERCEPTS, 0.2),
        )
        result = link_mirt(target, larger_source, list(range(6)), list(range(6)))

        summary = mirt_linking_summary(result, target)

        assert "Slope RMSE: 0.0000" in summary
        assert "Intercept RMSE: 0.0000" in summary
        assert "theta_linked" in summary
        assert "Factor 1:" in summary

    def test_summary_handles_legacy_result_without_anchor_mapping(self):
        model = make_model()
        result = ProcrustesResult(
            np.eye(2), np.zeros(2), 1.0, 0.0, SOURCE_SLOPES.copy()
        )
        assert "Factor 1:" in mirt_linking_summary(result, model)
