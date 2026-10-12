"""Tests for polytomous IRT models."""

import numpy as np
import pytest

from mirt.exceptions import MirtValidationError
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedRatingScaleModel,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
    RatingScaleModel,
)
from mirt.utils.information import testinfo as total_information


class TestGradedResponseModel:
    """Tests for GRM."""

    def test_initialization(self):
        """Test model initialization."""
        model = GradedResponseModel(n_items=10, n_categories=5)

        assert model.n_items == 10
        assert model.n_factors == 1
        assert model.model_name == "GRM"
        assert model.n_categories == [5] * 10
        assert model.max_categories == 5
        assert not model.is_fitted

    def test_varying_categories(self):
        """Test model with varying categories per item."""
        n_cats = [3, 4, 5, 3, 4]
        model = GradedResponseModel(n_items=5, n_categories=n_cats)

        assert model.n_categories == n_cats
        assert model.max_categories == 5

    def test_probability_shape(self):
        """Test probability output shape."""
        model = GradedResponseModel(n_items=5, n_categories=4)

        theta = np.linspace(-3, 3, 100)
        probs = model.probability(theta, item_idx=0)

        assert probs.shape == (100, 4)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_probability_sum_to_one(self):
        """Test that category probabilities sum to 1."""
        model = GradedResponseModel(n_items=5, n_categories=5)

        theta = np.linspace(-3, 3, 50)

        for item_idx in range(model.n_items):
            probs = model.probability(theta, item_idx=item_idx)
            row_sums = probs.sum(axis=1)
            np.testing.assert_array_almost_equal(row_sums, np.ones(50), decimal=5)

    def test_cumulative_probability(self):
        """Test cumulative probability computation."""
        model = GradedResponseModel(n_items=3, n_categories=4)

        theta = np.array([0.0])

        for item_idx in range(model.n_items):
            cum_probs = [
                model.cumulative_probability(theta, item_idx, k)[0]
                for k in range(model.n_categories[item_idx] - 1)
            ]

            for i in range(len(cum_probs) - 1):
                assert cum_probs[i] >= cum_probs[i + 1]

    def test_information(self):
        """Test Fisher information computation."""
        model = GradedResponseModel(n_items=5, n_categories=5)

        theta = np.linspace(-3, 3, 50)
        info = model.information(theta)

        assert info.shape == (50,)
        assert np.all(info >= 0)

    def test_expected_score(self):
        """Test expected score computation."""
        model = GradedResponseModel(n_items=3, n_categories=5)

        theta = np.linspace(-3, 3, 20)
        expected = model.expected_score(theta)

        assert expected.shape == (20,)
        assert np.all(expected >= 0)
        assert np.all(expected <= model.n_items * (model.max_categories - 1))

    def test_multidimensional(self):
        """Test multidimensional GRM."""
        model = GradedResponseModel(n_items=6, n_categories=4, n_factors=2)

        assert model.n_factors == 2
        assert model.discrimination.shape == (6, 2)

        theta = np.random.randn(20, 2)
        probs = model.probability(theta, item_idx=0)

        assert probs.shape == (20, 4)


class TestGeneralizedPartialCredit:
    """Tests for GPCM."""

    def test_initialization(self):
        """Test model initialization."""
        model = GeneralizedPartialCredit(n_items=8, n_categories=5)

        assert model.n_items == 8
        assert model.model_name == "GPCM"
        assert model.max_categories == 5

    def test_probability_sum_to_one(self):
        """Test that category probabilities sum to 1."""
        model = GeneralizedPartialCredit(n_items=5, n_categories=4)

        theta = np.linspace(-3, 3, 50)

        for item_idx in range(model.n_items):
            probs = model.probability(theta, item_idx=item_idx)
            row_sums = probs.sum(axis=1)
            np.testing.assert_array_almost_equal(row_sums, np.ones(50), decimal=5)

    def test_steps_property(self):
        """Test step parameters exist."""
        model = GeneralizedPartialCredit(n_items=5, n_categories=4)

        steps = model.steps
        assert steps.shape == (5, 3)

    def test_information(self):
        """Test Fisher information computation."""
        model = GeneralizedPartialCredit(n_items=5, n_categories=5)

        theta = np.linspace(-3, 3, 50)
        info = model.information(theta)

        assert info.shape == (50,)
        assert np.all(info >= 0)


class TestPartialCreditModel:
    """Tests for PCM (Rasch-like GPCM)."""

    def test_discrimination_fixed(self):
        """Test that discrimination is fixed to 1."""
        model = PartialCreditModel(n_items=5, n_categories=4)

        np.testing.assert_array_equal(model.discrimination, np.ones(5))

    def test_cannot_set_discrimination(self):
        """Test that discrimination cannot be modified."""
        model = PartialCreditModel(n_items=5, n_categories=4)

        with pytest.raises(ValueError, match="Cannot set discrimination"):
            model.set_parameters(discrimination=np.array([2.0] * 5))

    def test_no_multidimensional(self):
        """Test that PCM doesn't support multidimensional."""
        with pytest.raises(ValueError, match="unidimensional"):
            PartialCreditModel(n_items=5, n_categories=4, n_factors=2)


class TestNominalResponseModel:
    """Tests for NRM."""

    def test_initialization(self):
        """Test model initialization."""
        model = NominalResponseModel(n_items=6, n_categories=4)

        assert model.n_items == 6
        assert model.model_name == "NRM"

    def test_probability_sum_to_one(self):
        """Test that category probabilities sum to 1."""
        model = NominalResponseModel(n_items=5, n_categories=4)

        theta = np.linspace(-3, 3, 50)

        for item_idx in range(model.n_items):
            probs = model.probability(theta, item_idx=item_idx)
            row_sums = probs.sum(axis=1)
            np.testing.assert_array_almost_equal(row_sums, np.ones(50), decimal=5)

    def test_slopes_and_intercepts(self):
        """Test slopes and intercepts properties."""
        model = NominalResponseModel(n_items=5, n_categories=4)

        slopes = model.slopes
        intercepts = model.intercepts

        assert slopes.shape[0] == 5
        assert intercepts.shape == (5, 4)

    def test_information(self):
        """Test Fisher information computation."""
        model = NominalResponseModel(n_items=5, n_categories=5)

        theta = np.linspace(-3, 3, 50)
        info = model.information(theta)

        assert info.shape == (50,)
        assert np.all(info >= 0)

    def test_multidimensional(self):
        """Test multidimensional NRM."""
        model = NominalResponseModel(n_items=6, n_categories=4, n_factors=2)

        assert model.n_factors == 2

        theta = np.random.randn(20, 2)
        probs = model.probability(theta, item_idx=0)

        assert probs.shape == (20, 4)


class TestPolytomousLogLikelihood:
    """Tests for polytomous log-likelihood computation."""

    def test_grm_log_likelihood(self, polytomous_responses):
        """Test GRM log-likelihood computation."""
        responses = polytomous_responses["responses"]
        theta = polytomous_responses["theta"]
        n_items = polytomous_responses["n_items"]
        n_categories = polytomous_responses["n_categories"]

        model = GradedResponseModel(n_items=n_items, n_categories=n_categories)

        theta_2d = theta.reshape(-1, 1)
        ll = model.log_likelihood(responses, theta_2d)

        assert ll.shape == (len(theta),)
        assert np.all(ll <= 0)

    def test_gpcm_log_likelihood(self, polytomous_responses):
        """Test GPCM log-likelihood computation."""
        responses = polytomous_responses["responses"]
        theta = polytomous_responses["theta"]
        n_items = polytomous_responses["n_items"]
        n_categories = polytomous_responses["n_categories"]

        model = GeneralizedPartialCredit(n_items=n_items, n_categories=n_categories)

        theta_2d = theta.reshape(-1, 1)
        ll = model.log_likelihood(responses, theta_2d)

        assert ll.shape == (len(theta),)
        assert np.all(ll <= 0)


class TestCategoryProbabilityBounds:
    """Test that category probabilities handle edge cases."""

    @pytest.mark.parametrize(
        "ModelClass",
        [GradedResponseModel, GeneralizedPartialCredit, NominalResponseModel],
    )
    def test_extreme_theta(self, ModelClass):
        """Test probabilities at extreme theta values."""
        model = ModelClass(n_items=3, n_categories=4)

        theta_extreme = np.array([-10.0, 0.0, 10.0])

        for item_idx in range(model.n_items):
            probs = model.probability(theta_extreme, item_idx=item_idx)

            assert np.all(probs >= 0)
            assert np.all(probs <= 1)

            np.testing.assert_array_almost_equal(
                probs.sum(axis=1), np.ones(3), decimal=5
            )

    @pytest.mark.parametrize(
        "ModelClass",
        [GradedResponseModel, GeneralizedPartialCredit, NominalResponseModel],
    )
    def test_invalid_category(self, ModelClass):
        """Test that invalid category raises error."""
        model = ModelClass(n_items=3, n_categories=4)

        theta = np.array([0.0])

        with pytest.raises(ValueError, match="out of range"):
            model.category_probability(theta, item_idx=0, category=5)

        with pytest.raises(ValueError, match="out of range"):
            model.category_probability(theta, item_idx=0, category=-1)


class TestParameterDomains:
    """Setters reject states whose category curves are undefined."""

    @pytest.mark.parametrize("value", [np.nan, np.inf])
    @pytest.mark.parametrize(
        "ModelClass", [GradedResponseModel, GeneralizedPartialCredit]
    )
    def test_nonfinite_discrimination_is_rejected_atomically(self, ModelClass, value):
        model = ModelClass(2, n_categories=4)
        before = model.parameters
        locations = "thresholds" if ModelClass is GradedResponseModel else "steps"

        with pytest.raises(MirtValidationError, match="finite"):
            model.set_parameters(
                discrimination=np.array([value, 1.0]),
                **{locations: np.zeros((2, 3))},
            )

        for name, values in before.items():
            np.testing.assert_array_equal(model.parameters[name], values)

    def test_gpcm_rejects_nonfinite_steps_but_accepts_disordered_ones(self):
        model = GeneralizedPartialCredit(2, n_categories=3)

        with pytest.raises(MirtValidationError, match="steps"):
            model.set_parameters(steps=np.array([[np.nan, 0.0], [0.0, 1.0]]))
        model.set_parameters(steps=np.array([[1.0, -1.0], [0.0, 0.0]]))

        assert np.all(model.probability(np.array([0.0])) >= 0.0)

    def test_grm_padding_is_not_validated(self):
        model = GradedResponseModel(2, n_categories=[3, 4])

        model.set_parameters(thresholds=np.array([[-1.0, 1.0, np.nan], [-1, 0, 1]]))

        np.testing.assert_allclose(model.probability(np.array([0.0]), 0).sum(), 1.0)


class TestCategoryCounts:
    """Category counts accept NumPy integers and reject other numbers."""

    @pytest.mark.parametrize(
        "ModelClass",
        [
            GradedResponseModel,
            GeneralizedPartialCredit,
            PartialCreditModel,
            NominalResponseModel,
            RatingScaleModel,
            GradedRatingScaleModel,
        ],
    )
    def test_numpy_integer_counts_are_accepted(self, ModelClass):
        responses = np.array([[0, 3, 1], [2, 1, 0]])

        scalar = ModelClass(3, n_categories=responses.max() + 1)
        sequence = ModelClass(3, n_categories=list(np.full(3, 4, dtype=np.int32)))

        assert scalar.n_categories == sequence.n_categories == [4, 4, 4]
        assert all(type(count) is int for count in scalar.n_categories)

    @pytest.mark.parametrize("n_categories", [4.0, True, [4, 4.0], [4, True], "4"])
    @pytest.mark.parametrize(
        "ModelClass", [GradedResponseModel, NominalResponseModel, RatingScaleModel]
    )
    def test_non_integer_counts_are_rejected(self, ModelClass, n_categories):
        with pytest.raises(MirtValidationError, match="n_categories"):
            ModelClass(2, n_categories=n_categories)


def test_nominal_information_is_never_negative_in_the_tails():
    rng = np.random.default_rng(1)
    theta = np.linspace(-40.0, 40.0, 801)
    for n_factors in (1, 2):
        shape = (6, 6) if n_factors == 1 else (6, 6, n_factors)
        model = NominalResponseModel(6, n_categories=6, n_factors=n_factors)
        model.set_parameters(
            slopes=rng.normal(0.0, 2.0, shape), intercepts=rng.normal(0.0, 2.0, (6, 6))
        )
        points = theta[:, None] * np.ones(n_factors)

        by_item = np.column_stack([model.information(points, j) for j in range(6)])
        vectorized = model.information(points[:200])
        trace = np.einsum(
            "jnff->nj",
            np.stack([model.item_information_matrix(points, j) for j in range(6)]),
        )

        # E[a^2] - E[a]^2 gave values down to -7e-15 here.
        assert np.all(by_item >= 0.0)
        np.testing.assert_allclose(by_item, trace, rtol=1e-12, atol=1e-300)
        np.testing.assert_allclose(vectorized, by_item[:200].sum(axis=1), rtol=1e-12)
        assert np.all(total_information(model, points) >= 0.0)
