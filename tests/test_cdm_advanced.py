"""Tests for Advanced Cognitive Diagnosis Models."""

import numpy as np
import pytest

from mirt.models.cdm_advanced import (
    GDINA,
    AttributeHierarchy,
    HigherOrderCDM,
    fit_gdina,
)


class TestAttributeHierarchy:
    """Tests for AttributeHierarchy."""

    def test_init_basic(self):
        """Test basic initialization."""
        adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        hierarchy = AttributeHierarchy(adjacency=adjacency)

        assert hierarchy.n_attributes == 3
        assert hierarchy.attribute_names == ["A0", "A1", "A2"]

    def test_init_with_names(self):
        """Test initialization with custom names."""
        adjacency = np.array([[0, 1], [0, 0]])
        names = ["Reading", "Comprehension"]
        hierarchy = AttributeHierarchy(adjacency=adjacency, attribute_names=names)

        assert hierarchy.attribute_names == names

    def test_init_name_mismatch(self):
        """Test error on name count mismatch."""
        adjacency = np.array([[0, 1], [0, 0]])
        with pytest.raises(ValueError, match="attribute_names length"):
            AttributeHierarchy(adjacency=adjacency, attribute_names=["A"])

    def test_prerequisites(self):
        """Test direct prerequisites."""
        adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        hierarchy = AttributeHierarchy(adjacency=adjacency)

        assert hierarchy.prerequisites(1) == [0]
        assert hierarchy.prerequisites(2) == [1]
        assert hierarchy.prerequisites(0) == []

    def test_all_prerequisites(self):
        """Test all prerequisites (transitive)."""
        adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        hierarchy = AttributeHierarchy(adjacency=adjacency)

        assert hierarchy.all_prerequisites(2) == {0, 1}
        assert hierarchy.all_prerequisites(1) == {0}
        assert hierarchy.all_prerequisites(0) == set()

    def test_is_valid_pattern(self):
        """Test pattern validity checking."""
        adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        hierarchy = AttributeHierarchy(adjacency=adjacency)

        assert hierarchy.is_valid_pattern(np.array([1, 1, 1]))
        assert hierarchy.is_valid_pattern(np.array([1, 1, 0]))
        assert hierarchy.is_valid_pattern(np.array([1, 0, 0]))
        assert hierarchy.is_valid_pattern(np.array([0, 0, 0]))

        assert not hierarchy.is_valid_pattern(np.array([0, 1, 0]))
        assert not hierarchy.is_valid_pattern(np.array([0, 0, 1]))
        assert not hierarchy.is_valid_pattern(np.array([1, 0, 1]))

    def test_valid_patterns(self):
        """Test generating valid patterns."""
        adjacency = np.array([[0, 1], [0, 0]])
        hierarchy = AttributeHierarchy(adjacency=adjacency)

        valid = hierarchy.valid_patterns()
        assert len(valid) == 3
        valid_list = [tuple(p) for p in valid]
        assert (0, 0) in valid_list
        assert (1, 0) in valid_list
        assert (1, 1) in valid_list

    def test_topological_order(self):
        """Test topological ordering."""
        adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        hierarchy = AttributeHierarchy(adjacency=adjacency)

        order = hierarchy.topological_order()
        assert order.index(0) < order.index(1)
        assert order.index(1) < order.index(2)


class TestGDINA:
    """Tests for Generalized DINA model."""

    def test_init_basic(self):
        """Test basic initialization."""
        q_matrix = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
        model = GDINA(n_items=4, n_attributes=3, q_matrix=q_matrix)

        assert model.n_items == 4
        assert model.n_attributes == 3
        assert len(model.reduced_models) == 4
        assert all(m == "saturated" for m in model.reduced_models)

    def test_init_with_reduced_models(self):
        """Test initialization with specified reduced models."""
        q_matrix = np.array([[1, 0], [0, 1], [1, 1]])
        reduced = ["DINA", "DINO", "ACDM"]
        model = GDINA(
            n_items=3, n_attributes=2, q_matrix=q_matrix, reduced_models=reduced
        )

        assert model.reduced_models == reduced

    def test_init_reduced_model_mismatch(self):
        """Test error on reduced model count mismatch."""
        q_matrix = np.array([[1, 0], [0, 1]])
        with pytest.raises(ValueError, match="reduced_models length"):
            GDINA(n_items=2, n_attributes=2, q_matrix=q_matrix, reduced_models=["DINA"])

    def test_probability_saturated(self):
        """Test probability with saturated model."""
        q_matrix = np.array([[1, 0], [1, 1]])
        model = GDINA(n_items=2, n_attributes=2, q_matrix=q_matrix)

        alpha = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        probs = model.probability(alpha, item_idx=0)

        assert probs.shape == (4,)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_probability_dina(self):
        """Test probability with DINA reduced model."""
        q_matrix = np.array([[1, 1]])
        model = GDINA(
            n_items=1, n_attributes=2, q_matrix=q_matrix, reduced_models=["DINA"]
        )
        model.set_delta_parameters(0, np.array([0.2, 0.8]))

        alpha = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        probs = model.probability(alpha, item_idx=0)

        np.testing.assert_almost_equal(probs[0], 0.2)
        np.testing.assert_almost_equal(probs[1], 0.2)
        np.testing.assert_almost_equal(probs[2], 0.2)
        np.testing.assert_almost_equal(probs[3], 0.8)

    def test_probability_dino(self):
        """Test probability with DINO reduced model."""
        q_matrix = np.array([[1, 1]])
        model = GDINA(
            n_items=1, n_attributes=2, q_matrix=q_matrix, reduced_models=["DINO"]
        )
        model.set_delta_parameters(0, np.array([0.2, 0.8]))

        alpha = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        probs = model.probability(alpha, item_idx=0)

        np.testing.assert_almost_equal(probs[0], 0.2)
        np.testing.assert_almost_equal(probs[1], 0.8)
        np.testing.assert_almost_equal(probs[2], 0.8)
        np.testing.assert_almost_equal(probs[3], 0.8)

    def test_probability_acdm(self):
        """Test probability with additive CDM."""
        q_matrix = np.array([[1, 1]])
        model = GDINA(
            n_items=1, n_attributes=2, q_matrix=q_matrix, reduced_models=["ACDM"]
        )
        model.set_delta_parameters(0, np.array([0.2, 0.3, 0.3]))

        alpha = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        probs = model.probability(alpha, item_idx=0)

        np.testing.assert_almost_equal(probs[0], 0.2)
        np.testing.assert_almost_equal(probs[1], 0.5)
        np.testing.assert_almost_equal(probs[2], 0.5)
        np.testing.assert_almost_equal(probs[3], 0.8)

    def test_probability_all_items(self):
        """Test probability for all items."""
        q_matrix = np.array([[1, 0], [0, 1], [1, 1]])
        model = GDINA(n_items=3, n_attributes=2, q_matrix=q_matrix)

        alpha = np.array([[0, 0], [1, 1]])
        probs = model.probability(alpha)

        assert probs.shape == (2, 3)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_log_likelihood(self):
        """Test log-likelihood computation."""
        q_matrix = np.array([[1, 0], [0, 1]])
        model = GDINA(n_items=2, n_attributes=2, q_matrix=q_matrix)

        responses = np.array([[1, 0], [0, 1], [1, 1]])
        alpha = np.array([[1, 0], [0, 1], [1, 1]])
        ll = model.log_likelihood(responses, alpha)

        assert ll.shape == (3,)
        assert np.all(ll <= 0)

    def test_set_delta_parameters(self):
        """Test setting delta parameters."""
        q_matrix = np.array([[1, 1]])
        model = GDINA(
            n_items=1, n_attributes=2, q_matrix=q_matrix, reduced_models=["DINA"]
        )

        model.set_delta_parameters(0, np.array([0.1, 0.9]))
        delta = model.delta_parameters[0]

        np.testing.assert_array_almost_equal(delta, np.array([0.1, 0.9]))

    def test_set_delta_parameters_wrong_length(self):
        """Test error on wrong delta length."""
        q_matrix = np.array([[1, 1]])
        model = GDINA(
            n_items=1, n_attributes=2, q_matrix=q_matrix, reduced_models=["DINA"]
        )

        with pytest.raises(ValueError, match="delta length"):
            model.set_delta_parameters(0, np.array([0.1, 0.5, 0.9]))

    def test_copy(self):
        """Test model copying."""
        q_matrix = np.array([[1, 0], [1, 1]])
        model = GDINA(
            n_items=2,
            n_attributes=2,
            q_matrix=q_matrix,
            reduced_models=["DINA", "ACDM"],
        )

        model_copy = model.copy()

        assert model_copy is not model
        assert model_copy.reduced_models == model.reduced_models
        np.testing.assert_array_equal(model_copy.q_matrix, model.q_matrix)


class TestHigherOrderCDM:
    """Tests for Higher-Order CDM."""

    def test_init_basic(self):
        """Test basic initialization."""
        q_matrix = np.array([[1, 0], [0, 1], [1, 1]])
        model = HigherOrderCDM(n_items=3, n_attributes=2, q_matrix=q_matrix)

        assert model.n_items == 3
        assert model.n_attributes == 2
        assert model.loadings.shape == (2,)
        assert model.thresholds.shape == (2,)

    def test_init_with_hierarchy(self):
        """Test initialization with attribute hierarchy."""
        q_matrix = np.array([[1, 0], [0, 1], [1, 1]])
        adjacency = np.array([[0, 1], [0, 0]])
        hierarchy = AttributeHierarchy(adjacency=adjacency)

        model = HigherOrderCDM(
            n_items=3, n_attributes=2, q_matrix=q_matrix, hierarchy=hierarchy
        )

        assert model.hierarchy is hierarchy
        assert len(model.valid_patterns) == 3

    def test_set_higher_order_params(self):
        """Test setting higher-order parameters."""
        q_matrix = np.array([[1, 0], [0, 1]])
        model = HigherOrderCDM(n_items=2, n_attributes=2, q_matrix=q_matrix)

        loadings = np.array([1.5, 0.8])
        thresholds = np.array([0.0, -0.5])
        model.set_higher_order_params(loadings, thresholds)

        np.testing.assert_array_equal(model.loadings, loadings)
        np.testing.assert_array_equal(model.thresholds, thresholds)

    def test_set_higher_order_params_wrong_shape(self):
        """Test error on wrong parameter shape."""
        q_matrix = np.array([[1, 0], [0, 1]])
        model = HigherOrderCDM(n_items=2, n_attributes=2, q_matrix=q_matrix)

        with pytest.raises(ValueError, match="loadings shape"):
            model.set_higher_order_params(np.array([1.0]), np.array([0.0, 0.0]))

        with pytest.raises(ValueError, match="thresholds shape"):
            model.set_higher_order_params(np.array([1.0, 1.0]), np.array([0.0]))

    def test_attribute_probability(self):
        """Test attribute mastery probabilities."""
        q_matrix = np.array([[1, 0], [0, 1]])
        model = HigherOrderCDM(n_items=2, n_attributes=2, q_matrix=q_matrix)
        model.set_higher_order_params(np.array([1.0, 1.0]), np.array([0.0, 0.0]))

        theta = np.array([0.0, 1.0, -1.0])
        attr_prob = model.attribute_probability(theta)

        assert attr_prob.shape == (3, 2)
        assert np.all(attr_prob >= 0)
        assert np.all(attr_prob <= 1)

        np.testing.assert_almost_equal(attr_prob[0, 0], 0.5)

    def test_pattern_probability(self):
        """Test pattern probability computation."""
        q_matrix = np.array([[1, 0], [0, 1]])
        model = HigherOrderCDM(n_items=2, n_attributes=2, q_matrix=q_matrix)

        theta = np.array([0.0, 2.0])
        pattern_prob = model.pattern_probability(theta)

        assert pattern_prob.shape == (2, 4)
        np.testing.assert_array_almost_equal(
            pattern_prob.sum(axis=1), np.ones(2), decimal=5
        )

    def test_probability(self):
        """Test response probability computation."""
        q_matrix = np.array([[1, 0], [0, 1], [1, 1]])
        model = HigherOrderCDM(n_items=3, n_attributes=2, q_matrix=q_matrix)

        theta = np.array([0.0, 1.0, -1.0])
        probs = model.probability(theta)

        assert probs.shape == (3, 3)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_probability_single_item(self):
        """Test probability for single item."""
        q_matrix = np.array([[1, 0], [0, 1]])
        model = HigherOrderCDM(n_items=2, n_attributes=2, q_matrix=q_matrix)

        theta = np.array([0.0, 1.0])
        probs = model.probability(theta, item_idx=0)

        assert probs.shape == (2,)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_probability_monotonic(self):
        """Test probability is generally increasing in theta."""
        q_matrix = np.array([[1, 1]])
        model = HigherOrderCDM(n_items=1, n_attributes=2, q_matrix=q_matrix)

        theta = np.linspace(-3, 3, 20)
        probs = model.probability(theta, item_idx=0)

        assert probs[-1] > probs[0]
        assert np.corrcoef(theta, probs)[0, 1] > 0.9

    def test_log_likelihood(self):
        """Test log-likelihood computation."""
        q_matrix = np.array([[1, 0], [0, 1]])
        model = HigherOrderCDM(n_items=2, n_attributes=2, q_matrix=q_matrix)

        responses = np.array([[1, 0], [0, 1], [1, 1]])
        theta = np.array([0.5, -0.5, 1.0])
        ll = model.log_likelihood(responses, theta)

        assert ll.shape == (3,)
        assert np.all(ll <= 0)

    def test_estimate_theta_eap(self):
        """Test EAP theta estimation."""
        q_matrix = np.array([[1, 0], [0, 1], [1, 1]])
        model = HigherOrderCDM(n_items=3, n_attributes=2, q_matrix=q_matrix)
        model.set_higher_order_params(np.array([1.5, 1.5]), np.array([0.0, 0.0]))

        responses = np.array([[1, 1, 1], [0, 0, 0], [1, 0, 1]])
        theta_eap = model.estimate_theta(responses, method="EAP")

        assert theta_eap.shape == (3,)
        assert theta_eap[0] > theta_eap[1]

    def test_estimate_theta_mle(self):
        """Test MLE theta estimation."""
        q_matrix = np.array([[1, 0], [0, 1]])
        model = HigherOrderCDM(n_items=2, n_attributes=2, q_matrix=q_matrix)

        responses = np.array([[1, 1], [0, 0]])
        theta_mle = model.estimate_theta(responses, method="MLE")

        assert theta_mle.shape == (2,)
        assert theta_mle[0] > theta_mle[1]

    def test_copy(self):
        """Test model copying."""
        q_matrix = np.array([[1, 0], [0, 1]])
        model = HigherOrderCDM(n_items=2, n_attributes=2, q_matrix=q_matrix)
        model.set_higher_order_params(np.array([1.5, 0.8]), np.array([0.1, -0.2]))

        model_copy = model.copy()

        assert model_copy is not model
        np.testing.assert_array_equal(model_copy.loadings, model.loadings)
        np.testing.assert_array_equal(model_copy.thresholds, model.thresholds)


class TestFitGDINA:
    """Tests for G-DINA fitting function."""

    def test_fit_basic(self):
        """Test basic G-DINA fitting."""
        np.random.seed(42)
        n_persons = 100
        n_items = 5
        n_attributes = 2

        q_matrix = np.array([[1, 0], [0, 1], [1, 1], [1, 0], [0, 1]])

        alpha_true = np.random.binomial(1, 0.5, (n_persons, n_attributes))

        true_model = GDINA(
            n_items=n_items, n_attributes=n_attributes, q_matrix=q_matrix
        )

        probs = true_model.probability(alpha_true)
        responses = (np.random.random((n_persons, n_items)) < probs).astype(int)

        fitted_model, class_probs = fit_gdina(
            responses, q_matrix, max_iter=20, verbose=False
        )

        assert fitted_model._is_fitted
        assert class_probs.shape == (2**n_attributes,)
        np.testing.assert_almost_equal(class_probs.sum(), 1.0)

    def test_fit_with_reduced_models(self):
        """Test G-DINA fitting with reduced models."""
        np.random.seed(123)
        n_persons = 80
        n_items = 3
        n_attributes = 2

        q_matrix = np.array([[1, 0], [0, 1], [1, 1]])
        reduced = ["DINA", "DINA", "saturated"]

        alpha_true = np.random.binomial(1, 0.6, (n_persons, n_attributes))

        true_model = GDINA(
            n_items=n_items,
            n_attributes=n_attributes,
            q_matrix=q_matrix,
            reduced_models=reduced,
        )

        probs = true_model.probability(alpha_true)
        responses = (np.random.random((n_persons, n_items)) < probs).astype(int)

        fitted_model, class_probs = fit_gdina(
            responses, q_matrix, reduced_models=reduced, max_iter=20, verbose=False
        )

        assert fitted_model._is_fitted
        assert fitted_model.reduced_models == reduced


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
