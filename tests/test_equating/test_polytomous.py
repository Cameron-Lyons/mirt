"""Tests for polytomous model linking."""

import numpy as np
import pytest

from mirt.equating import (
    LinkingResult,
    link_gpcm,
    link_grm,
    link_nrm,
    transform_polytomous_parameters,
)
from mirt.equating.polytomous import (
    _extract_steps,
    _gpcm_category_probs,
    _grm_category_probs,
    _nrm_category_probs,
)
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
)


@pytest.fixture
def grm_model_pair():
    """Create a pair of GRM models for linking."""
    model_old = GradedResponseModel(n_items=5, n_categories=4)
    disc_old = np.array([1.0, 1.2, 0.8, 1.5, 1.1])
    thresholds_old = np.array(
        [
            [-1.5, -0.5, 0.5],
            [-1.0, 0.0, 1.0],
            [-0.5, 0.5, 1.5],
            [-2.0, -1.0, 0.0],
            [-1.0, 0.0, 1.0],
        ]
    )
    model_old.set_parameters(discrimination=disc_old, thresholds=thresholds_old)
    model_old._is_fitted = True

    A_true = 1.2
    B_true = 0.5
    model_new = GradedResponseModel(n_items=5, n_categories=4)
    disc_new = disc_old / A_true
    thresholds_new = A_true * thresholds_old + B_true
    model_new.set_parameters(discrimination=disc_new, thresholds=thresholds_new)
    model_new._is_fitted = True

    return model_old, model_new, A_true, B_true


@pytest.fixture
def gpcm_model_pair():
    """Create a pair of GPCM models for linking."""
    model_old = GeneralizedPartialCredit(n_items=5, n_categories=4)
    disc_old = np.array([1.0, 1.2, 0.8, 1.5, 1.1])
    steps_old = np.array(
        [
            [-0.5, 0.0, 0.5],
            [-0.3, 0.0, 0.3],
            [-0.4, 0.1, 0.3],
            [-0.6, -0.1, 0.7],
            [-0.2, 0.1, 0.1],
        ]
    )
    model_old.set_parameters(discrimination=disc_old, steps=steps_old)
    model_old._is_fitted = True

    A_true = 1.3
    B_true = -0.2
    model_new = GeneralizedPartialCredit(n_items=5, n_categories=4)
    disc_new = disc_old / A_true
    steps_new = A_true * steps_old + B_true
    model_new.set_parameters(discrimination=disc_new, steps=steps_new)
    model_new._is_fitted = True

    return model_old, model_new, A_true, B_true


@pytest.fixture
def nrm_model_pair():
    """Create an exact affine pair of NRM calibrations."""
    model_old = NominalResponseModel(n_items=5, n_categories=4)
    slopes_old = np.array(
        [
            [0.0, 0.6, 1.1, 1.7],
            [0.0, 0.8, 1.4, 1.9],
            [0.0, -0.4, 0.9, 1.5],
            [0.0, 0.5, 1.5, 2.2],
            [0.0, -0.2, 0.7, 1.8],
        ]
    )
    intercepts_old = np.array(
        [
            [0.0, -0.8, 0.3, -0.2],
            [0.0, -0.4, 0.5, -0.7],
            [0.0, 0.2, -0.6, 0.4],
            [0.0, -0.3, 0.8, -0.5],
            [0.0, 0.6, -0.4, 0.1],
        ]
    )
    model_old.set_parameters(slopes=slopes_old, intercepts=intercepts_old)

    A_true, B_true = 1.6, -0.35
    model_new = transform_polytomous_parameters(
        model_old, A_true, B_true, model_type="nrm"
    )
    return model_old, model_new, A_true, B_true


class TestLinkGRM:
    """Tests for GRM linking."""

    def test_link_grm_returns_result(self, grm_model_pair):
        """Test that link_grm returns LinkingResult."""
        model_old, model_new, _, _ = grm_model_pair
        anchors = list(range(5))

        result = link_grm(model_old, model_new, anchors, anchors)

        assert isinstance(result, LinkingResult)

    def test_link_grm_recovers_constants(self, grm_model_pair):
        """Test that GRM linking recovers transformation constants."""
        model_old, model_new, A_true, B_true = grm_model_pair
        anchors = list(range(5))

        result = link_grm(
            model_new, model_old, anchors, anchors, method="stocking_lord"
        )

        assert result.constants.A == pytest.approx(A_true, abs=1e-10)
        assert result.constants.B == pytest.approx(B_true, abs=1e-10)

    @pytest.mark.parametrize(
        "method", ["mean_sigma", "mean_mean", "stocking_lord", "haebara"]
    )
    def test_all_grm_methods_recover_exact_constants(self, grm_model_pair, method):
        """Every GRM method follows the documented transformation direction."""
        model_old, model_new, A_true, B_true = grm_model_pair
        anchors = list(range(5))

        result = link_grm(
            model_new,
            model_old,
            anchors,
            anchors,
            method=method,
            compute_diagnostics=False,
        )

        assert result.constants.A == pytest.approx(A_true, abs=1e-10)
        assert result.constants.B == pytest.approx(B_true, abs=1e-10)
        assert result.constants.A > 0.0
        if method in {"stocking_lord", "haebara"}:
            assert result.convergence_info == {
                "method": method,
                "success": True,
                "fun": pytest.approx(0.0, abs=1e-14),
                "nit": 0,
            }

    def test_link_grm_methods(self, grm_model_pair):
        """Test all GRM linking methods."""
        model_old, model_new, _, _ = grm_model_pair
        anchors = list(range(5))

        for method in ["mean_sigma", "mean_mean", "stocking_lord", "haebara"]:
            result = link_grm(model_old, model_new, anchors, anchors, method=method)
            assert isinstance(result.constants.A, float)
            assert isinstance(result.constants.B, float)

    def test_link_grm_with_diagnostics(self, grm_model_pair):
        """Test GRM linking with diagnostics."""
        model_old, model_new, _, _ = grm_model_pair
        anchors = list(range(5))

        result = link_grm(
            model_old, model_new, anchors, anchors, compute_diagnostics=True
        )

        assert result.fit_statistics is not None
        assert result.anchor_diagnostics is not None


class TestLinkGPCM:
    """Tests for GPCM linking."""

    def test_link_gpcm_returns_result(self, gpcm_model_pair):
        """Test that link_gpcm returns LinkingResult."""
        model_old, model_new, _, _ = gpcm_model_pair
        anchors = list(range(5))

        result = link_gpcm(model_old, model_new, anchors, anchors)

        assert isinstance(result, LinkingResult)

    def test_link_gpcm_methods(self, gpcm_model_pair):
        """Test GPCM linking methods."""
        model_old, model_new, _, _ = gpcm_model_pair
        anchors = list(range(5))

        for method in ["mean_sigma", "mean_mean", "stocking_lord", "haebara"]:
            result = link_gpcm(model_old, model_new, anchors, anchors, method=method)
            assert isinstance(result.constants.A, float)
            assert isinstance(result.constants.B, float)

    @pytest.mark.parametrize(
        "method", ["mean_sigma", "mean_mean", "stocking_lord", "haebara"]
    )
    def test_all_gpcm_methods_recover_exact_constants(self, gpcm_model_pair, method):
        """GPCM moment and curve methods recover both scale and shift."""
        model_old, model_new, A_true, B_true = gpcm_model_pair
        anchors = list(range(5))

        result = link_gpcm(
            model_new,
            model_old,
            anchors,
            anchors,
            method=method,
            compute_diagnostics=False,
        )

        assert result.constants.A == pytest.approx(A_true, abs=1e-10)
        assert result.constants.B == pytest.approx(B_true, abs=1e-10)
        assert result.constants.A > 0.0

    def test_link_gpcm_returns_anchor_diagnostics(self, gpcm_model_pair):
        """GPCM diagnostics include the same item-level detail as GRM."""
        model_old, model_new, _, _ = gpcm_model_pair

        result = link_gpcm(model_new, model_old, list(range(5)), list(range(5)))

        assert result.fit_statistics is not None
        assert result.anchor_diagnostics is not None


class TestLinkNRM:
    """Tests for NRM linking."""

    @pytest.mark.parametrize("method", ["stocking_lord", "haebara"])
    def test_nrm_methods_recover_exact_constants(self, nrm_model_pair, method):
        """NRM linking uses the correct intercept sign and positive scale."""
        model_old, model_new, A_true, B_true = nrm_model_pair
        anchors = list(range(5))

        result = link_nrm(model_new, model_old, anchors, anchors, method=method)

        assert result.constants.A == pytest.approx(A_true, abs=1e-10)
        assert result.constants.B == pytest.approx(B_true, abs=1e-10)
        assert result.constants.A > 0.0
        assert result.fit_statistics is not None
        assert result.fit_statistics.tcc_rmse == pytest.approx(0.0, abs=1e-12)
        assert result.anchor_diagnostics is not None
        assert not np.any(result.anchor_diagnostics.flagged)


class TestTransformPolytomous:
    """Tests for polytomous parameter transformation."""

    def test_transform_grm_creates_copy(self, grm_model_pair):
        """Test that transformation creates copy by default."""
        model_old, _, A_true, B_true = grm_model_pair

        transformed = transform_polytomous_parameters(
            model_old, A_true, B_true, model_type="grm", in_place=False
        )

        assert transformed is not model_old

    def test_transform_grm_in_place(self, grm_model_pair):
        """Test in-place transformation."""
        model_old, _, A_true, B_true = grm_model_pair
        model_copy = model_old.copy()

        transformed = transform_polytomous_parameters(
            model_copy, A_true, B_true, model_type="grm", in_place=True
        )

        assert transformed is model_copy

    def test_transform_grm_formula(self, grm_model_pair):
        """Test GRM transformation formulas."""
        model_old, _, _, _ = grm_model_pair
        A, B = 1.5, 0.3

        original_disc = np.asarray(model_old.discrimination).copy()
        original_thresh = model_old.parameters["thresholds"].copy()

        transformed = transform_polytomous_parameters(
            model_old, A, B, model_type="grm", in_place=False
        )

        expected_disc = original_disc / A
        expected_thresh = A * original_thresh + B

        np.testing.assert_allclose(
            np.asarray(transformed.discrimination), expected_disc, rtol=1e-10
        )
        np.testing.assert_allclose(
            transformed.parameters["thresholds"], expected_thresh, rtol=1e-10
        )

    @pytest.mark.parametrize(
        ("fixture_name", "model_type"),
        [("grm_model_pair", "grm"), ("gpcm_model_pair", "gpcm")],
    )
    def test_ordered_transform_preserves_response_curves(
        self, request, fixture_name, model_type
    ):
        """Affine parameter changes preserve ordered-model response curves."""
        model_old, model_new, A, B = request.getfixturevalue(fixture_name)
        theta_old = np.linspace(-3.0, 3.0, 31)
        theta_new = A * theta_old + B

        for item in range(model_old.n_items):
            np.testing.assert_allclose(
                model_new.probability(theta_new, item),
                model_old.probability(theta_old, item),
                atol=1e-12,
            )

    def test_transform_gpcm_formula(self, gpcm_model_pair):
        """GPCM steps are affine locations rather than inverse-scale offsets."""
        model_old, _, _, _ = gpcm_model_pair
        A, B = 1.5, 0.3

        transformed = transform_polytomous_parameters(
            model_old, A, B, model_type="gpcm"
        )

        np.testing.assert_allclose(
            transformed.discrimination, model_old.discrimination / A
        )
        np.testing.assert_allclose(transformed.steps, A * model_old.steps + B)

    def test_transform_nrm_formula(self, nrm_model_pair):
        """NRM slopes and intercepts preserve category response curves."""
        model_old, model_new, A, B = nrm_model_pair
        theta_old = np.linspace(-3.0, 3.0, 31)
        theta_new = A * theta_old + B

        for item in range(model_old.n_items):
            np.testing.assert_allclose(
                model_new.probability(theta_new, item),
                model_old.probability(theta_old, item),
                atol=1e-12,
            )

    def test_transform_preserves_unused_padding(self):
        """Variable-category models keep inactive storage columns untouched."""
        model = GeneralizedPartialCredit(n_items=3, n_categories=[2, 3, 4])
        steps = np.array(
            [
                [-0.5, 17.0, 18.0],
                [-0.4, 0.6, 19.0],
                [-0.7, 0.1, 0.8],
            ]
        )
        model.set_parameters(steps=steps)

        transformed = transform_polytomous_parameters(
            model, 1.4, -0.2, model_type="gpcm"
        )

        assert transformed.steps[0, 1:].tolist() == [17.0, 18.0]
        assert transformed.steps[1, 2] == 19.0
        np.testing.assert_allclose(transformed.steps[2], 1.4 * steps[2] - 0.2)

    @pytest.mark.parametrize(
        ("A", "B", "message"),
        [
            (0.0, 0.0, "A must"),
            (-1.0, 0.0, "A must"),
            (np.inf, 0.0, "A must"),
            (1.0, np.nan, "B must"),
        ],
    )
    def test_transform_rejects_invalid_constants(self, grm_model_pair, A, B, message):
        """Invalid affine constants fail before copying or mutating a model."""
        model_old, _, _, _ = grm_model_pair

        with pytest.raises(ValueError, match=message):
            transform_polytomous_parameters(model_old, A, B)

    def test_transform_rejects_unknown_model_type(self, grm_model_pair):
        """An unknown transformation cannot silently leave a model unchanged."""
        model_old, _, _, _ = grm_model_pair

        with pytest.raises(ValueError, match="Unknown polytomous model type"):
            transform_polytomous_parameters(model_old, 1.0, 0.0, model_type="pcm")


class TestPolytomousValidation:
    """Validation tests for polytomous linking."""

    def test_mismatched_anchors(self, grm_model_pair):
        """Test error on mismatched anchor lengths."""
        model_old, model_new, _, _ = grm_model_pair

        with pytest.raises(ValueError, match="same length"):
            link_grm(model_old, model_new, [0, 1, 2], [0, 1])

    def test_invalid_method(self, grm_model_pair):
        """Test error on invalid method."""
        model_old, model_new, _, _ = grm_model_pair

        with pytest.raises(ValueError, match="Unknown method"):
            link_grm(model_old, model_new, [0, 1], [0, 1], method="invalid")

    @pytest.mark.parametrize(
        ("anchors", "message"),
        [
            ([-1, 1], "out of range"),
            ([0, 5], "out of range"),
            ([1, 1], "unique"),
            ([0, 1.5], "integers"),
        ],
    )
    def test_anchor_indices_are_validated(self, grm_model_pair, anchors, message):
        """Invalid indices cannot select unintended or duplicate items."""
        model_old, model_new, _, _ = grm_model_pair

        with pytest.raises(ValueError, match=message):
            link_grm(model_old, model_new, anchors, [0, 1])

    def test_at_least_two_anchors_are_required(self, grm_model_pair):
        """A single anchor cannot identify a stable affine transformation."""
        model_old, model_new, _, _ = grm_model_pair

        with pytest.raises(ValueError, match="At least 2"):
            link_grm(model_old, model_new, [0], [0])

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"theta_range": (1.0, -1.0)}, "theta_range"),
            ({"n_theta": 1}, "n_theta"),
            ({"weights": np.ones(3)}, "weights must have shape"),
            ({"weights": np.zeros(61)}, "positive sum"),
            ({"weights": np.r_[np.ones(60), -1.0]}, "non-negative"),
        ],
    )
    def test_curve_grid_is_validated(self, grm_model_pair, kwargs, message):
        """Malformed integration settings fail with clear errors."""
        model_old, model_new, _, _ = grm_model_pair

        with pytest.raises(ValueError, match=message):
            link_grm(model_old, model_new, [0, 1], [0, 1], **kwargs)

    def test_corresponding_categories_must_match(self):
        """Paired anchors need compatible category response curves."""
        model_old = GradedResponseModel(n_items=2, n_categories=[3, 4])
        model_new = GradedResponseModel(n_items=2, n_categories=[4, 4])

        with pytest.raises(ValueError, match="same number of categories"):
            link_grm(model_old, model_new, [0, 1], [0, 1])

    def test_multidimensional_models_are_rejected(self):
        """Scalar constants cannot silently discard extra dimensions."""
        model = GradedResponseModel(n_items=3, n_categories=4, n_factors=2)

        with pytest.raises(ValueError, match="unidimensional"):
            link_grm(model, model, [0, 1], [0, 1])

    def test_non_increasing_grm_thresholds_are_rejected(self):
        """Invalid threshold ordering cannot create negative probabilities."""
        model = GradedResponseModel(n_items=3, n_categories=4)
        thresholds = model.thresholds.copy()
        thresholds[0] = np.array([0.0, -1.0, 1.0])
        model.set_parameters(thresholds=thresholds)

        with pytest.raises(ValueError, match="strictly increasing"):
            link_grm(model, model, [0, 1], [0, 1])

    def test_unidentified_nrm_slopes_are_rejected(self):
        """Constant category slopes do not identify an NRM scale."""
        model = NominalResponseModel(n_items=3, n_categories=4)
        model.set_parameters(slopes=np.ones((3, 4)))

        with pytest.raises(ValueError, match="do not identify a scale"):
            link_nrm(model, model, [0, 1], [0, 1])

    def test_invalid_nrm_method(self, nrm_model_pair):
        """NRM exposes only implemented curve objectives."""
        model_old, model_new, _, _ = nrm_model_pair

        with pytest.raises(ValueError, match="Unknown method"):
            link_nrm(model_old, model_new, [0, 1], [0, 1], method="mean_mean")


class TestPolytomousEdgeCases:
    """Regression tests for category storage, stability, and diagnostics."""

    def test_variable_category_gpcm_recovers_exact_constants(self):
        """Only active step columns contribute to mixed-category linking."""
        categories = [2, 3, 4, 5]
        model_old = GeneralizedPartialCredit(4, categories)
        model_old.set_parameters(
            discrimination=np.array([0.8, 1.1, 1.4, 1.7]),
            steps=np.array(
                [
                    [-0.7, 0.0, 0.0, 0.0],
                    [-0.8, 0.5, 0.0, 0.0],
                    [-1.0, -0.1, 0.8, 0.0],
                    [-1.2, -0.4, 0.3, 1.1],
                ]
            ),
        )
        A, B = 1.45, -0.25
        model_new = transform_polytomous_parameters(model_old, A, B, model_type="gpcm")

        result = link_gpcm(model_new, model_old, list(range(4)), list(range(4)))

        assert result.constants.A == pytest.approx(A, abs=1e-10)
        assert result.constants.B == pytest.approx(B, abs=1e-10)
        assert [
            len(values) for values in _extract_steps(model_old, list(range(4)))
        ] == [
            1,
            2,
            3,
            4,
        ]

    @pytest.mark.parametrize(
        ("probability", "parameters"),
        [
            (_grm_category_probs, (10.0, np.array([-2.0, 0.0, 2.0]))),
            (_gpcm_category_probs, (10.0, np.array([-2.0, 0.0, 2.0]))),
            (
                _nrm_category_probs,
                (np.array([-2.0, 0.0, 2.0]), np.array([1.0, 0.0, -1.0])),
            ),
        ],
    )
    def test_extreme_category_probabilities_are_finite_and_normalized(
        self, probability, parameters
    ):
        """Extreme logits remain finite and sum to one."""
        theta = np.array([-1e6, 1e6])

        probabilities = probability(theta, *parameters)

        assert np.all(np.isfinite(probabilities))
        assert np.all(probabilities >= 0.0)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-15)

    def test_linking_kernels_match_model_probabilities(
        self, grm_model_pair, gpcm_model_pair, nrm_model_pair
    ):
        """Optimized kernels retain parity with the public model equations."""
        theta = np.linspace(-2.5, 2.5, 17)
        grm, _, _, _ = grm_model_pair
        gpcm, _, _, _ = gpcm_model_pair
        nrm, _, _, _ = nrm_model_pair

        np.testing.assert_allclose(
            _grm_category_probs(theta, grm.discrimination[0], grm.thresholds[0]),
            grm.probability(theta, 0),
            atol=1e-15,
        )
        np.testing.assert_allclose(
            _gpcm_category_probs(theta, gpcm.discrimination[0], gpcm.steps[0]),
            gpcm.probability(theta, 0),
            atol=1e-15,
        )
        np.testing.assert_allclose(
            _nrm_category_probs(theta, nrm.slopes[0], nrm.intercepts[0]),
            nrm.probability(theta, 0),
            atol=1e-15,
        )

    def test_gpcm_step_drift_is_flagged(self):
        """A lone category-location drift appears in item diagnostics."""
        model_old = GeneralizedPartialCredit(5, 4)
        model_new = model_old.copy()
        steps = model_new.steps.copy()
        steps[0, 1] += 2.0
        model_new.set_parameters(steps=steps)

        result = link_gpcm(model_old, model_new, list(range(5)), list(range(5)))

        diagnostics = result.anchor_diagnostics
        assert diagnostics is not None
        assert diagnostics.area_diff[0] > diagnostics.area_diff[1:].max()
        assert diagnostics.flagged.tolist() == [True, False, False, False, False]
