"""Tests for multidimensional IRT models."""

import numpy as np
import pytest

from mirt.models.bifactor import BifactorModel
from mirt.models.multidimensional import MultidimensionalModel


class TestMultidimensionalModel:
    """Tests for MIRT model."""

    def test_initialization(self):
        """Test model initialization."""
        model = MultidimensionalModel(n_items=10, n_factors=3)

        assert model.n_items == 10
        assert model.n_factors == 3
        assert model.model_name == "MIRT"
        assert model.model_type == "exploratory"
        assert not model.is_fitted

    def test_requires_multiple_factors(self):
        """Test that model requires at least 2 factors."""
        with pytest.raises(ValueError, match="n_factors >= 2"):
            MultidimensionalModel(n_items=10, n_factors=1)

    def test_confirmatory_requires_pattern(self):
        """Test that confirmatory model requires loading pattern."""
        with pytest.raises(ValueError, match="loading_pattern required"):
            MultidimensionalModel(n_items=10, n_factors=2, model_type="confirmatory")

    def test_confirmatory_pattern_shape(self):
        """Test loading pattern shape validation."""
        pattern = np.ones((10, 3))  # Wrong shape

        with pytest.raises(ValueError, match="doesn't match"):
            MultidimensionalModel(
                n_items=10,
                n_factors=2,
                model_type="confirmatory",
                loading_pattern=pattern,
            )

    def test_confirmatory_model(self):
        """Test confirmatory model initialization."""
        pattern = np.array(
            [
                [1, 0],
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
                [0, 1],
            ]
        )

        model = MultidimensionalModel(
            n_items=6,
            n_factors=2,
            model_type="confirmatory",
            loading_pattern=pattern,
        )

        assert model.model_type == "confirmatory"
        np.testing.assert_array_equal(model.loading_pattern, pattern)

    def test_probability_shape(self):
        """Test probability output shape."""
        model = MultidimensionalModel(n_items=5, n_factors=2)

        theta = np.random.randn(100, 2)
        probs = model.probability(theta)

        assert probs.shape == (100, 5)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_probability_single_item(self):
        """Test probability for single item."""
        model = MultidimensionalModel(n_items=5, n_factors=2)

        theta = np.array([[0.0, 0.0]])
        prob = model.probability(theta, item_idx=0)

        assert prob.shape == (1,)
        assert 0 <= prob[0] <= 1

    def test_log_likelihood(self):
        """Test log-likelihood computation."""
        model = MultidimensionalModel(n_items=5, n_factors=2)

        responses = np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]])
        theta = np.array([[0.0, 0.0], [1.0, -1.0]])

        ll = model.log_likelihood(responses, theta)

        assert ll.shape == (2,)
        assert np.all(ll <= 0)

    def test_information(self):
        """Test Fisher information computation."""
        model = MultidimensionalModel(n_items=5, n_factors=2)

        theta = np.random.randn(50, 2)
        info = model.information(theta)

        assert info.shape == (50, 5)
        assert np.all(info >= 0)

    def test_to_irt_parameterization(self):
        """Test conversion to IRT parameterization."""
        model = MultidimensionalModel(n_items=5, n_factors=2)

        params = model.to_irt_parameterization()

        assert "discrimination" in params
        assert "difficulty" in params
        assert params["discrimination"].shape == (5, 2)
        assert params["difficulty"].shape == (5,)

    def test_get_factor_loadings(self):
        """Test factor loading extraction."""
        model = MultidimensionalModel(n_items=5, n_factors=2)

        # Unstandardized
        loadings = model.get_factor_loadings(standardized=False)
        assert loadings.shape == (5, 2)

        # Standardized
        loadings_std = model.get_factor_loadings(standardized=True)
        assert loadings_std.shape == (5, 2)

        # Standardized loadings should be smaller in magnitude
        assert np.all(np.abs(loadings_std) <= np.abs(loadings) + 1e-10)

    def test_communalities(self):
        """Test communality computation."""
        model = MultidimensionalModel(n_items=5, n_factors=2)

        comm = model.communalities()

        assert comm.shape == (5,)
        assert np.all(comm >= 0)
        assert np.all(comm <= 1)


class TestBifactorModel:
    """Tests for Bifactor model."""

    def test_initialization(self):
        """Test model initialization."""
        specific_factors = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        model = BifactorModel(n_items=9, specific_factors=specific_factors)

        assert model.n_items == 9
        assert model.n_factors == 4  # 1 general + 3 specific
        assert model.n_specific_factors == 3
        assert model.model_name == "Bifactor"
        assert not model.is_fitted

    def test_specific_factors_validation(self):
        """Test specific factors length validation."""
        with pytest.raises(ValueError, match="must match n_items"):
            BifactorModel(n_items=5, specific_factors=[0, 0, 1])

    def test_negative_specific_factors(self):
        """Test negative specific factors raise error."""
        with pytest.raises(ValueError, match="non-negative"):
            BifactorModel(n_items=3, specific_factors=[-1, 0, 1])

    def test_probability_shape(self):
        """Test probability output shape."""
        specific_factors = [0, 0, 1, 1]
        model = BifactorModel(n_items=4, specific_factors=specific_factors)

        theta = np.random.randn(100, 3)  # 1 general + 2 specific
        probs = model.probability(theta)

        assert probs.shape == (100, 4)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_probability_single_item(self):
        """Test probability for single item."""
        specific_factors = [0, 0, 1, 1]
        model = BifactorModel(n_items=4, specific_factors=specific_factors)

        theta = np.array([[0.0, 0.0, 0.0]])
        prob = model.probability(theta, item_idx=0)

        assert prob.shape == (1,)
        assert 0 <= prob[0] <= 1

    def test_get_factor_structure(self):
        """Test factor structure extraction."""
        specific_factors = [0, 0, 1, 1, 2]
        model = BifactorModel(n_items=5, specific_factors=specific_factors)

        structure = model.get_factor_structure()

        assert structure == {0: [0, 1], 1: [2, 3], 2: [4]}

    def test_get_loading_matrix(self):
        """Test loading matrix extraction."""
        specific_factors = [0, 0, 1, 1]
        model = BifactorModel(n_items=4, specific_factors=specific_factors)

        loadings = model.get_loading_matrix()

        assert loadings.shape == (4, 3)  # items x (1 general + 2 specific)

        # General loadings in first column
        assert np.all(loadings[:, 0] > 0)

        # Specific loadings in appropriate columns
        assert loadings[0, 1] > 0  # Item 0 loads on specific factor 0
        assert loadings[0, 2] == 0  # Item 0 doesn't load on specific factor 1
        assert loadings[2, 2] > 0  # Item 2 loads on specific factor 1

    def test_omega_hierarchical(self):
        """Test omega hierarchical computation."""
        specific_factors = [0, 0, 0, 1, 1, 1]
        model = BifactorModel(n_items=6, specific_factors=specific_factors)

        omega_h = model.omega_hierarchical()

        assert 0 <= omega_h <= 1

    def test_omega_subscale(self):
        """Test omega subscale computation."""
        specific_factors = [0, 0, 0, 1, 1, 1]
        model = BifactorModel(n_items=6, specific_factors=specific_factors)

        omega_0 = model.omega_subscale(0)
        omega_1 = model.omega_subscale(1)
        omega_invalid = model.omega_subscale(5)  # Non-existent factor

        assert 0 <= omega_0 <= 1
        assert 0 <= omega_1 <= 1
        assert np.isnan(omega_invalid)

    def test_explained_common_variance(self):
        """Test ECV computation."""
        specific_factors = [0, 0, 0, 1, 1, 1]
        model = BifactorModel(n_items=6, specific_factors=specific_factors)

        ecv = model.explained_common_variance()

        assert "general" in ecv
        assert "specific_0" in ecv
        assert "specific_1" in ecv

        # All values should be positive
        assert all(v > 0 for v in ecv.values())

        # Should sum to approximately 1
        total = sum(ecv.values())
        np.testing.assert_almost_equal(total, 1.0, decimal=5)

    def test_information(self):
        """Test Fisher information computation."""
        specific_factors = [0, 0, 1, 1]
        model = BifactorModel(n_items=4, specific_factors=specific_factors)

        theta = np.random.randn(50, 3)
        info = model.information(theta)

        assert info.shape == (50, 4)
        assert np.all(info >= 0)

    def test_log_likelihood(self):
        """Test log-likelihood computation."""
        specific_factors = [0, 0, 1, 1]
        model = BifactorModel(n_items=4, specific_factors=specific_factors)

        responses = np.array([[1, 0, 1, 0], [0, 1, 0, 1]])
        theta = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.5]])

        ll = model.log_likelihood(responses, theta)

        assert ll.shape == (2,)
        assert np.all(ll <= 0)


class TestMultidimensionalParameterSetting:
    """Tests for parameter setting in multidimensional models."""

    def test_set_slopes_respects_pattern(self):
        """Test that set_parameters respects loading pattern."""
        pattern = np.array(
            [
                [1, 0],
                [1, 0],
                [0, 1],
                [0, 1],
            ]
        )

        model = MultidimensionalModel(
            n_items=4,
            n_factors=2,
            model_type="confirmatory",
            loading_pattern=pattern,
        )

        new_slopes = np.ones((4, 2)) * 2.0
        model.set_parameters(slopes=new_slopes)

        # Slopes should be masked by pattern
        expected = np.array(
            [
                [2.0, 0.0],
                [2.0, 0.0],
                [0.0, 2.0],
                [0.0, 2.0],
            ]
        )

        np.testing.assert_array_equal(model.slopes, expected)

    def test_bifactor_set_parameters(self):
        """Test parameter setting for bifactor model."""
        specific_factors = [0, 0, 1, 1]
        model = BifactorModel(n_items=4, specific_factors=specific_factors)

        new_general = np.array([0.8, 0.7, 0.9, 0.6])
        new_specific = np.array([0.5, 0.4, 0.6, 0.3])

        model.set_parameters(
            general_loadings=new_general,
            specific_loadings=new_specific,
        )

        np.testing.assert_array_equal(model.general_loadings, new_general)
        np.testing.assert_array_equal(model.specific_loadings, new_specific)
