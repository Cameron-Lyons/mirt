"""Tests for Advanced Cognitive Diagnosis Models."""

import time

import numpy as np
import pytest

from mirt.exceptions import MirtDataError, MirtIndexError, MirtValidationError
from mirt.models.cdm_advanced import (
    GDINA,
    AttributeHierarchy,
    HigherOrderCDM,
    fit_gdina,
    fit_higher_order_cdm,
)


def _strong_higher_order_model() -> HigherOrderCDM:
    """Six items on two attributes with success rates .1 and .9."""
    q_matrix = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [1, 1]])
    model = HigherOrderCDM(6, 2, q_matrix)
    model.set_parameters(loadings=np.array([1.5, 1.5]), thresholds=np.array([0, 0.5]))
    for item_idx, row in enumerate(q_matrix):
        n_required = int(row.sum())
        groups = (np.arange(2**n_required)[:, None] >> np.arange(n_required)) & 1
        model.set_delta_parameters(
            item_idx, np.where(groups.sum(axis=1) == n_required, 0.9, 0.1)
        )
    return model


def _brute_force_likelihood(
    model: HigherOrderCDM, responses: np.ndarray, theta: np.ndarray
) -> np.ndarray:
    """Return sum_alpha P(alpha | theta) prod_j P(X_j | alpha) per person and theta.

    The logistic attribute curves are written out instead of calling the model,
    so the reference also checks the documented higher-order link.
    """
    patterns = model.valid_patterns
    conditional = model.base_model.probability(patterns)
    item_terms = np.where(
        responses[:, None, :] == 1, conditional[None], 1.0 - conditional[None]
    )
    item_terms = np.where(responses[:, None, :] >= 0, item_terms, 1.0)
    pattern_likelihood = item_terms.prod(axis=2)
    return pattern_likelihood @ _brute_force_pattern_prior(model, theta).T


def _brute_force_pattern_prior(model: HigherOrderCDM, theta: np.ndarray) -> np.ndarray:
    """Return P(alpha | theta) for every valid pattern from the logistic link."""
    patterns = model.valid_patterns
    mastery = 1.0 / (
        1.0 + np.exp(-model.loadings * (theta[:, None] - model.thresholds))
    )
    prior = np.prod(
        np.where(patterns[None] == 1, mastery[:, None, :], 1.0 - mastery[:, None, :]),
        axis=2,
    )
    return prior / prior.sum(axis=1, keepdims=True)


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

    @pytest.mark.parametrize(
        "adjacency",
        [
            np.array([0, 1]),
            np.zeros((2, 3)),
            np.array([[0.0, 0.5], [0.0, 0.0]]),
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 1], [1, 0]]),
        ],
    )
    def test_init_rejects_invalid_graphs(self, adjacency):
        """Reject malformed, non-binary, reflexive, and cyclic graphs."""
        with pytest.raises(ValueError):
            AttributeHierarchy(adjacency=adjacency)

    def test_init_owns_immutable_adjacency(self):
        """Prevent caller mutation from invalidating the hierarchy."""
        adjacency = np.array([[0, 1], [0, 0]])
        hierarchy = AttributeHierarchy(adjacency=adjacency)

        adjacency[0, 1] = 0

        assert hierarchy.prerequisites(1) == [0]
        with pytest.raises(ValueError, match="read-only"):
            hierarchy.adjacency[0, 1] = 0

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

    @pytest.mark.parametrize(
        "pattern",
        [np.array([1]), np.array([[1, 1]]), np.array([0, 2]), np.array([0, np.nan])],
    )
    def test_is_valid_pattern_rejects_invalid_inputs(self, pattern):
        """Require one finite binary value per attribute."""
        hierarchy = AttributeHierarchy(adjacency=np.array([[0, 1], [0, 0]]))

        with pytest.raises(ValueError):
            hierarchy.is_valid_pattern(pattern)

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

    def test_set_delta_parameters_validates_index_and_values(self):
        """Reject aliasing negative indices and coefficients outside [0, 1]."""
        q_matrix = np.array([[1, 0], [0, 1], [1, 1]])
        model = GDINA(3, 2, q_matrix, reduced_models=["saturated", "DINA", "saturated"])
        before = model.delta_parameters

        with pytest.raises(MirtIndexError, match="item_idx"):
            model.set_delta_parameters(-1, np.array([0.9, 0.8, 0.7, 0.1]))
        with pytest.raises(MirtValidationError, match="finite"):
            model.set_delta_parameters(0, np.array([0.3, np.nan]))
        with pytest.raises(MirtValidationError, match="outside"):
            model.set_delta_parameters(0, np.array([1.7, 0.5]))
        with pytest.raises(MirtValidationError, match="outside"):
            model.set_delta_parameters(1, np.array([-0.1, 0.9]))
        with pytest.raises(MirtValidationError, match="delta length"):
            model.set_delta_parameters(2, np.array([0.1, 0.9]))
        for actual, expected in zip(model.delta_parameters, before, strict=True):
            np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize(
        ("family", "delta", "message"),
        [
            ("ACDM", [0.5, 0.4, 0.3], "outside"),
            ("ACDM", [0.2, -0.3, 0.1], "outside"),
            ("RRUM", [0.9, -0.5, 0.5], "non-negative"),
            ("RRUM", [1.2, 0.5, 0.5], "outside"),
            ("LLM", [0.0, np.inf, 1.0], "finite"),
        ],
    )
    def test_reduced_families_validate_implied_probabilities(
        self, family, delta, message
    ):
        model = GDINA(1, 2, np.array([[1, 1]]), reduced_models=[family])
        with pytest.raises(MirtValidationError, match=message):
            model.set_delta_parameters(0, np.array(delta))

    def test_default_coefficients_are_valid_probabilities(self):
        """Every family starts from coefficients its own setter accepts."""
        families = ["saturated", "DINA", "DINO", "ACDM", "LLM", "RRUM"]
        model = GDINA(6, 3, np.ones((6, 3), dtype=int), reduced_models=families)
        model.set_parameters(**model.parameters)
        probabilities = model.probability(model.attribute_patterns)
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
        np.testing.assert_allclose(model.item_probabilities(3)[[0, -1]], [0.2, 0.8])

    def test_set_parameters_validates_rows_atomically(self):
        """Validate active coefficients before changing any item."""
        model = GDINA(3, 2, np.array([[1, 0], [0, 1], [1, 1]]))
        before = model.parameters
        delta = model.parameters["delta"]
        delta[0, :2] = [0.1, 0.9]
        delta[2, :4] = [1.7, -0.3, np.nan, 2.0]

        with pytest.raises(MirtValidationError, match="item 2"):
            model.set_parameters(delta=delta)
        with pytest.raises(MirtValidationError, match="delta parameters"):
            model.set_parameters(reduced_model_code=np.array([0.0, 0.0, 2.0]))
        for name, values in before.items():
            np.testing.assert_array_equal(model.parameters[name], values)

        delta[2, :4] = [0.1, 0.2, 0.3, 0.9]
        model.set_parameters(delta=delta)
        np.testing.assert_allclose(model.delta_parameters[0], [0.1, 0.9])

    def test_log_likelihood_validates_responses_and_rows(self):
        """Reject response codes other than 0, 1 and missing, as DINA does."""
        model = GDINA(3, 2, np.array([[1, 0], [0, 1], [1, 1]]))
        alpha = np.array([[1, 1], [0, 0]])

        with pytest.raises(MirtDataError, match="only 0, 1"):
            model.log_likelihood(np.array([[2, 0, 1], [1, 1, 5]]), alpha)
        with pytest.raises(MirtValidationError, match="compatible row counts"):
            model.log_likelihood(np.zeros((2, 3), dtype=int), np.zeros((3, 2), int))
        values = model.log_likelihood(np.array([[1, -1, 0]]), alpha)
        probabilities = model.probability(alpha)
        np.testing.assert_allclose(
            values, np.log(probabilities[:, 0]) + np.log1p(-probabilities[:, 2])
        )

    def test_scoring_api_matches_direct_evaluation(self):
        """Score respondents against every pattern, as DINA and DINO do."""
        model = GDINA(
            4,
            2,
            np.array([[1, 0], [0, 1], [1, 1], [1, 1]]),
            reduced_models=["saturated", "DINA", "ACDM", "LLM"],
        )
        model.set_delta_parameters(0, [0.1, 0.9])
        model.set_delta_parameters(1, [0.2, 0.8])
        model.set_delta_parameters(2, [0.1, 0.4, 0.4])
        model.set_delta_parameters(3, [-2.0, 1.5, 2.5])
        responses = np.array([[1, 0, 1, 0], [1, 1, 1, 1], [0, -1, 0, 0]])
        prior = np.array([0.4, 0.1, 0.1, 0.4])

        log_likelihoods = model.pattern_log_likelihoods(responses)
        expected = np.column_stack(
            [
                model.log_likelihood(responses, pattern)
                for pattern in model.attribute_patterns
            ]
        )
        np.testing.assert_allclose(log_likelihoods, expected)
        posterior = model.attribute_posteriors(responses, prior)
        expected_posterior = np.exp(expected) * prior
        expected_posterior /= expected_posterior.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(posterior, expected_posterior)
        np.testing.assert_allclose(
            model.attribute_marginals(responses, prior),
            posterior @ model.attribute_patterns,
        )
        np.testing.assert_array_equal(
            model.classify_respondents(responses, method="MAP", pattern_prior=prior),
            model.attribute_patterns[np.argmax(expected_posterior, axis=1)],
        )
        np.testing.assert_array_equal(
            model.classify_respondents(responses),
            model.attribute_patterns[np.argmax(expected, axis=1)],
        )
        np.testing.assert_array_equal(
            model.classify_respondents(responses, method="EAP", pattern_prior=prior),
            (posterior @ model.attribute_patterns > 0.5).astype(int),
        )

    def test_gdina_deltas_follow_de_la_torre_parameterization(self):
        """Convert stored coefficients into identity, logit or log-link deltas."""
        q_matrix = np.array([[1, 1], [1, 1], [1, 1], [1, 1], [1, 0]])
        model = GDINA(
            5,
            2,
            q_matrix,
            reduced_models=["saturated", "DINA", "DINO", "RRUM", "ACDM"],
        )
        # Identity-link deltas (0.1, 0.3, 0.3, 0.2) give P(00, 10, 01, 11).
        model.set_delta_parameters(0, [0.1, 0.4, 0.4, 0.9])
        model.set_delta_parameters(1, [0.2, 0.9])
        model.set_delta_parameters(2, [0.2, 0.9])
        model.set_delta_parameters(3, [0.9, 0.5, 0.25])
        model.set_delta_parameters(4, [0.2, 0.6])

        np.testing.assert_allclose(model.gdina_deltas(0), [0.1, 0.3, 0.3, 0.2])
        np.testing.assert_allclose(model.gdina_deltas(1), [0.2, 0.7])
        np.testing.assert_allclose(model.gdina_deltas(2), [0.2, 0.7])
        np.testing.assert_allclose(model.gdina_deltas(4), [0.2, 0.6])
        rrum = model.gdina_deltas(3)
        groups = model.attribute_patterns
        np.testing.assert_allclose(
            np.exp(rrum[0] + groups @ rrum[1:]), model.item_probabilities(3)
        )
        np.testing.assert_allclose(model.item_probabilities(0), [0.1, 0.4, 0.4, 0.9])

    def test_saturated_deltas_order_interactions_by_order(self):
        """Order deltas as intercept, main effects, then two- and three-way terms."""
        model = GDINA(1, 3, np.ones((1, 3), dtype=int))
        effects = np.array([0.05, 0.1, 0.15, 0.2, 0.04, 0.06, 0.08, 0.12])
        groups = model.attribute_patterns
        subsets = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
        probabilities = np.array(
            [
                sum(
                    effect
                    for effect, subset in zip(effects, subsets, strict=True)
                    if all(group[attribute] for attribute in subset)
                )
                for group in groups
            ]
        )
        model.set_delta_parameters(0, probabilities)

        np.testing.assert_allclose(model.gdina_deltas(0), effects)


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

    def test_set_parameters_is_finite_atomic_and_owned(self):
        """Validate both arrays before changing state and retain owned copies."""
        q_matrix = np.array([[1, 0], [0, 1]])
        model = HigherOrderCDM(n_items=2, n_attributes=2, q_matrix=q_matrix)
        original_loadings = model.loadings
        original_thresholds = model.thresholds

        with pytest.raises(ValueError, match="thresholds must contain"):
            model.set_parameters(
                loadings=np.array([2.0, 3.0]),
                thresholds=np.array([0.0, np.nan]),
            )

        np.testing.assert_array_equal(model.loadings, original_loadings)
        np.testing.assert_array_equal(model.thresholds, original_thresholds)

        loadings = np.array([1.2, 1.4])
        thresholds = np.array([-0.2, 0.3])
        model.set_higher_order_params(loadings, thresholds)
        loadings[:] = 99.0
        thresholds[:] = 99.0
        np.testing.assert_array_equal(model.loadings, np.array([1.2, 1.4]))
        np.testing.assert_array_equal(model.thresholds, np.array([-0.2, 0.3]))

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

    def test_extreme_theta_is_stable(self):
        """Keep mastery and hierarchy-conditioned pattern probabilities finite."""
        hierarchy = AttributeHierarchy(adjacency=np.array([[0, 1], [0, 0]]))
        q_matrix = np.array([[1, 0], [0, 1]])
        model = HigherOrderCDM(
            n_items=2,
            n_attributes=2,
            q_matrix=q_matrix,
            hierarchy=hierarchy,
        )
        model.set_higher_order_params(
            np.ones(2),
            np.array([1e308, -1e308]),
        )

        attribute_probability = model.attribute_probability(np.array([-1e308, 1e308]))
        pattern_probability = model.pattern_probability(np.array([0.0]))

        assert np.all(np.isfinite(attribute_probability))
        assert np.all(np.isfinite(pattern_probability))
        np.testing.assert_allclose(pattern_probability.sum(axis=1), 1.0)
        np.testing.assert_allclose(pattern_probability[0], np.array([0.5, 0.0, 0.5]))

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

    def test_probability_matches_pattern_marginalization(self):
        """Match explicit pattern marginalization for all and single items."""
        q_matrix = np.array([[1, 0], [0, 1], [1, 1]])
        model = HigherOrderCDM(n_items=3, n_attributes=2, q_matrix=q_matrix)
        theta = np.array([-1.2, 0.3, 1.7])
        pattern_probability = model.pattern_probability(theta)
        conditional = model._base_cdm.probability(model.valid_patterns)
        expected = pattern_probability @ conditional

        np.testing.assert_allclose(model.probability(theta), expected)
        np.testing.assert_allclose(model.probability(theta, 1), expected[:, 1])

    @pytest.mark.parametrize("item_idx", [-1, 2, True])
    def test_probability_rejects_invalid_item_index(self, item_idx):
        """Reject negative, out-of-range, and boolean item indices."""
        model = HigherOrderCDM(
            n_items=2,
            n_attributes=2,
            q_matrix=np.array([[1, 0], [0, 1]]),
        )

        with pytest.raises(IndexError, match="item_idx"):
            model.probability(np.array([0.0]), item_idx=item_idx)

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

    @pytest.mark.parametrize(
        ("responses", "theta"),
        [
            (np.array([1, 0]), np.array([0.0])),
            (np.array([[1, 0, 1]]), np.array([0.0])),
            (np.array([[1, 2]]), np.array([0.0])),
            (np.array([[1, 0], [0, 1]]), np.array([0.0, 1.0, 2.0])),
        ],
    )
    def test_log_likelihood_rejects_invalid_inputs(self, responses, theta):
        """Validate response matrices, response codes, and paired row counts."""
        model = HigherOrderCDM(
            n_items=2,
            n_attributes=2,
            q_matrix=np.array([[1, 0], [0, 1]]),
        )

        with pytest.raises(ValueError):
            model.log_likelihood(responses, theta)

    def test_log_likelihood_supports_scalar_theta_and_missing_responses(self):
        """Broadcast a scalar trait and ignore negative missing responses."""
        model = HigherOrderCDM(
            n_items=2,
            n_attributes=2,
            q_matrix=np.array([[1, 0], [0, 1]]),
        )
        responses = np.array([[1, -1], [0, -1]])

        result = model.log_likelihood(responses, np.array(0.0))

        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

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

    @pytest.mark.parametrize("method", ["MAP", "eap", ""])
    def test_estimate_theta_rejects_unknown_method(self, method):
        """Reject estimation method typos instead of silently using EAP."""
        model = HigherOrderCDM(
            n_items=2,
            n_attributes=2,
            q_matrix=np.array([[1, 0], [0, 1]]),
        )

        with pytest.raises(ValueError, match="method"):
            model.estimate_theta(np.array([[1, 0]]), method=method)

    @pytest.mark.parametrize("n_quad", [0, -1, 2.5, True])
    def test_estimate_theta_rejects_invalid_quadrature_count(self, n_quad):
        """Require a positive integer quadrature count."""
        model = HigherOrderCDM(
            n_items=2,
            n_attributes=2,
            q_matrix=np.array([[1, 0], [0, 1]]),
        )

        with pytest.raises(ValueError, match="n_quad"):
            model.estimate_theta(np.array([[1, 0]]), n_quad=n_quad)

    def test_estimate_theta_matches_grid_reference(self):
        """Match brute-force EAP and MLE over the shared mastery patterns."""
        model = _strong_higher_order_model()
        responses = np.array(
            [
                [1, 1, 1, 0, 0, 1],
                [1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 1, -1, 1, 1, 1],
            ]
        )
        quad_points = np.linspace(-4, 4, 15)
        likelihood = _brute_force_likelihood(model, responses, quad_points)
        posterior = likelihood * np.exp(-0.5 * quad_points**2)
        posterior /= posterior.sum(axis=1, keepdims=True)
        expected_eap = posterior @ quad_points
        expected_mle = quad_points[np.argmax(likelihood, axis=1)]

        np.testing.assert_allclose(
            model.estimate_theta(responses, method="EAP", n_quad=15), expected_eap
        )
        np.testing.assert_array_equal(
            model.estimate_theta(responses, method="MLE", n_quad=15), expected_mle
        )

    @pytest.mark.parametrize("hierarchical", [False, True])
    def test_log_likelihood_marginalizes_shared_mastery_patterns(self, hierarchical):
        """Items share alpha, so P(X | theta) sums over patterns, not items."""
        model = _strong_higher_order_model()
        if hierarchical:
            hierarchy = AttributeHierarchy(adjacency=np.array([[0, 1], [0, 0]]))
            restricted = HigherOrderCDM(6, 2, model.q_matrix, hierarchy=hierarchy)
            restricted.set_parameters(**model.parameters)
            model = restricted
        responses = np.array(
            [
                [1, 1, 1, 0, 0, 1],
                [1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
            ]
        )
        theta = np.full(4, 0.3)

        expected = np.log(np.diag(_brute_force_likelihood(model, responses, theta)))

        np.testing.assert_allclose(model.log_likelihood(responses, theta), expected)
        np.testing.assert_allclose(
            model.log_likelihood(responses, np.array(0.3)), expected
        )
        if not hierarchical:
            # The former item-independent product gave [-3.93, -3.72, -4.19, -4.41].
            np.testing.assert_allclose(
                expected, [-3.7971, -5.8872, -2.1185, -1.9778], atol=1e-4
            )

    def test_information_is_fisher_information_of_marginal_curves(self):
        """Use (dP/dtheta)**2 / (P (1 - P)), not the Bernoulli variance."""
        model = _strong_higher_order_model()
        theta = np.array([0.0, 1.0])
        step = 1e-5
        probability = model.probability(theta)
        derivative = (
            model.probability(theta + step) - model.probability(theta - step)
        ) / (2 * step)
        expected = derivative**2 / (probability * (1.0 - probability))

        np.testing.assert_allclose(model.information(theta), expected, rtol=1e-6)
        np.testing.assert_allclose(
            model.information(theta, 5), expected[:, 5], rtol=1e-6
        )
        np.testing.assert_allclose(expected[0, :3], 0.36, atol=1e-4)

    def test_attribute_probability_uses_documented_logistic_link(self):
        """P(alpha_k = 1 | theta) = 1 / (1 + exp(-lambda_k (theta - tau_k)))."""
        model = HigherOrderCDM(2, 2, np.eye(2, dtype=int))
        model.set_higher_order_params(np.array([1.0, 2.0]), np.array([0.0, 0.5]))

        np.testing.assert_allclose(
            model.attribute_probability(np.array([1.0])),
            [[1.0 / (1.0 + np.exp(-1.0)), 1.0 / (1.0 + np.exp(-1.0))]],
        )

    def test_scoring_methods_use_valid_patterns_and_implied_prior(self):
        """Classify over hierarchy-valid patterns with the theta-integrated prior."""
        hierarchy = AttributeHierarchy(adjacency=np.array([[0, 1], [0, 0]]))
        base = _strong_higher_order_model()
        model = HigherOrderCDM(6, 2, base.q_matrix, hierarchy=hierarchy)
        model.set_parameters(**base.parameters)
        responses = np.array([[1, 1, 1, 0, 0, 1], [0, 0, 0, 1, 1, 0]])

        points = np.linspace(-4, 4, 31)
        weights = np.exp(-0.5 * points**2)
        weights /= weights.sum()
        likelihood = _brute_force_likelihood(model, responses, points)
        expected_prior = weights @ _brute_force_pattern_prior(model, points)
        np.testing.assert_allclose(model.class_probabilities, expected_prior)
        np.testing.assert_allclose(model.class_probabilities.sum(), 1.0)

        posterior = model.attribute_posteriors(responses)
        log_likelihoods = model.pattern_log_likelihoods(responses)
        expected_posterior = np.exp(log_likelihoods) * expected_prior
        expected_posterior /= expected_posterior.sum(axis=1, keepdims=True)
        assert posterior.shape == (2, 3)
        np.testing.assert_allclose(posterior, expected_posterior)
        np.testing.assert_allclose(
            np.exp(model.log_likelihood(responses, np.zeros(2))),
            likelihood[:, 15],
        )
        np.testing.assert_allclose(
            model.attribute_marginals(responses), posterior @ model.valid_patterns
        )
        for method in ("MLE", "MAP", "EAP"):
            classified = model.classify_respondents(responses, method=method)
            assert classified.shape == (2, 2)
        np.testing.assert_array_equal(
            model.classify_respondents(responses, method="MAP"), [[1, 0], [0, 0]]
        )
        assert not any(
            np.array_equal(row, [0, 1]) for row in model.classify_respondents(responses)
        )

    def test_item_parameters_are_public_and_counted(self):
        """Set G-DINA item coefficients through the higher-order model."""
        model = HigherOrderCDM(3, 2, np.array([[1, 0], [0, 1], [1, 1]]))
        assert model.n_parameters == 2 + 2 + 4 + 2 * 2
        delta = model.parameters["delta"]
        delta[2, :4] = [0.05, 0.3, 0.3, 0.95]
        model.set_parameters(delta=delta, loadings=np.array([1.2, 0.8]))

        np.testing.assert_allclose(model.delta_parameters[2], [0.05, 0.3, 0.3, 0.95])
        np.testing.assert_allclose(model.loadings, [1.2, 0.8])
        np.testing.assert_allclose(
            model.probability(np.array([0.0])),
            model.pattern_probability(np.array([0.0]))
            @ model.base_model.probability(model.valid_patterns),
        )
        model.set_delta_parameters(0, [0.1, 0.8])
        np.testing.assert_allclose(model.parameters["delta"][0, :2], [0.1, 0.8])

        owned = model.base_model
        owned.set_delta_parameters(0, [0.3, 0.4])
        np.testing.assert_allclose(model.delta_parameters[0], [0.1, 0.8])

        before = model.parameters
        bad = model.parameters["delta"]
        bad[1, 0] = 1.5
        with pytest.raises(MirtValidationError, match="outside"):
            model.set_parameters(delta=bad, thresholds=np.array([1.0, 1.0]))
        for name, values in before.items():
            np.testing.assert_array_equal(model.parameters[name], values)

        copied = model.copy()
        np.testing.assert_array_equal(copied.parameters["delta"], before["delta"])
        assert copied.reduced_models == model.reduced_models

    def test_rejects_hierarchy_with_other_attribute_count(self):
        hierarchy = AttributeHierarchy(adjacency=np.zeros((3, 3), dtype=int))
        with pytest.raises(MirtValidationError, match="hierarchy has 3"):
            HigherOrderCDM(2, 2, np.eye(2, dtype=int), hierarchy=hierarchy)

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

    def test_fit_returns_result_with_likelihood_and_criteria(self):
        """Report the marginal log-likelihood at the returned estimates."""
        rng = np.random.default_rng(11)
        q_matrix = np.array([[1, 0], [0, 1], [1, 1], [1, 0], [0, 1], [1, 1]])
        model = GDINA(6, 2, q_matrix, reduced_models=["DINA"] * 6)
        for item_idx in range(6):
            model.set_delta_parameters(item_idx, [0.15, 0.85])
        alpha = model.sample_attribute_patterns(
            400, seed=12, class_probabilities=[0.4, 0.1, 0.1, 0.4]
        )
        responses = model.simulate(alpha, seed=13)
        responses[rng.random(responses.shape) < 0.05] = -1

        result = fit_gdina(responses, q_matrix, max_iter=200, tol=1e-6)
        fitted, class_probabilities = result

        assert fitted is result.model and result[0] is fitted and len(result) == 2
        np.testing.assert_array_equal(class_probabilities, result.class_probabilities)
        expected_ll = np.sum(
            np.log(
                np.exp(fitted.pattern_log_likelihoods(responses)) @ class_probabilities
            )
        )
        np.testing.assert_allclose(result.log_likelihood, expected_ll)
        assert (
            result.n_parameters == fitted.n_parameters + 3 == 2 + 2 + 4 + 2 + 2 + 4 + 3
        )
        assert result.n_observations == 400
        np.testing.assert_allclose(result.aic, -2 * expected_ll + 2 * 19)
        np.testing.assert_allclose(result.bic, -2 * expected_ll + 19 * np.log(400))
        assert result.converged and 0 < result.n_iterations < 200
        assert "Log-Likelihood" in result.summary()

        stopped = fit_gdina(responses, q_matrix, max_iter=2, tol=1e-12)
        assert not stopped.converged and stopped.n_iterations == 2

    def test_fitted_model_scores_with_estimated_class_probabilities(self):
        """Default MAP classification uses the fitted, not a uniform, prior."""
        q_matrix = np.vstack((np.eye(3, dtype=int), np.ones((3, 3), dtype=int)))
        model = GDINA(6, 3, q_matrix, reduced_models=["DINA"] * 6)
        for item_idx in range(6):
            model.set_delta_parameters(item_idx, [0.25, 0.75])
        prior = np.array([0.45, 0.02, 0.02, 0.03, 0.02, 0.03, 0.03, 0.4])
        responses = model.simulate(model.sample_attribute_patterns(600, 2, prior), 3)

        fitted, class_probabilities = fit_gdina(
            responses, q_matrix, reduced_models=["DINA"] * 6
        )

        np.testing.assert_array_equal(fitted.class_probabilities, class_probabilities)
        np.testing.assert_allclose(
            fitted.attribute_posteriors(responses),
            fitted.attribute_posteriors(responses, class_probabilities),
        )
        assert not np.allclose(
            fitted.attribute_marginals(responses),
            fitted.attribute_marginals(responses, np.ones(8)),
        )
        np.testing.assert_array_equal(
            fitted.copy().class_probabilities, class_probabilities
        )
        assert GDINA(6, 3, q_matrix).class_probabilities is None

    def test_fit_imposes_attribute_hierarchy(self):
        """Keep zero class probability on patterns the hierarchy forbids."""
        hierarchy = AttributeHierarchy(adjacency=np.array([[0, 1], [0, 0]]))
        q_matrix = np.array([[1, 0], [0, 1], [1, 1], [1, 0], [0, 1], [1, 1]])
        responses = np.random.default_rng(5).integers(0, 2, size=(200, 6))

        result = fit_gdina(responses, q_matrix, hierarchy=hierarchy, max_iter=30)

        # Pattern (0, 1) masters attribute 1 without its prerequisite 0.
        assert result.class_probabilities[2] == 0.0
        np.testing.assert_allclose(result.class_probabilities.sum(), 1.0)
        assert result.n_parameters == result.model.n_parameters + 2
        for method in ("MLE", "MAP"):
            classified = result.model.classify_respondents(responses, method=method)
            assert not np.any((classified[:, 0] == 0) & (classified[:, 1] == 1))
        with pytest.raises(MirtValidationError, match="hierarchy"):
            fit_gdina(responses, q_matrix, hierarchy=np.array([[0, 1], [0, 0]]))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"verbose": 1}, "verbose"),
            ({"tol": True}, "tol"),
        ],
    )
    def test_fit_controls_match_fit_cdm(self, kwargs, message):
        with pytest.raises(MirtValidationError, match=message):
            fit_gdina(np.array([[1, 0], [0, 1]]), np.eye(2, dtype=int), **kwargs)

    def test_fit_reuses_gemm_counts_and_warm_starts(self):
        """Keep the per-item count totals of the former E-step implementation."""
        rng = np.random.default_rng(9)
        q_matrix = np.array([[1, 0], [0, 1], [1, 1], [1, 1]])
        responses = rng.integers(0, 2, size=(300, 4))
        responses[rng.random(responses.shape) < 0.1] = -1
        families = ["saturated", "ACDM", "LLM", "RRUM"]

        fitted, class_probabilities = fit_gdina(
            responses, q_matrix, reduced_models=families, max_iter=1
        )

        reference = GDINA(4, 2, q_matrix, reduced_models=families)
        patterns = reference.attribute_patterns
        probabilities = reference.probability(patterns)
        log_like = np.zeros((300, 4))
        for item_idx in range(4):
            observed = responses[:, item_idx] >= 0
            values = responses[:, item_idx, None]
            log_like[observed] += np.where(
                values[observed] == 1,
                np.log(probabilities[:, item_idx]),
                np.log1p(-probabilities[:, item_idx]),
            )
        posterior = np.exp(log_like) / np.exp(log_like).sum(axis=1, keepdims=True)
        np.testing.assert_allclose(
            class_probabilities, posterior.mean(axis=0), rtol=1e-12
        )
        for item_idx in range(4):
            observed = responses[:, item_idx] >= 0
            totals = posterior[observed].sum(axis=0)
            successes = (posterior[observed] * responses[observed, item_idx, None]).sum(
                axis=0
            )
            groups = reference._latent_group_idx(patterns, item_idx)
            n_groups = 2 ** int(q_matrix[item_idx].sum())
            expected, _, _ = reference._fit_reduced_parameters(
                families[item_idx],
                reference._required_group_patterns(item_idx),
                np.bincount(groups, weights=successes, minlength=n_groups),
                np.bincount(groups, weights=totals, minlength=n_groups),
                reference.delta_parameters[item_idx],
            )
            np.testing.assert_allclose(
                fitted.delta_parameters[item_idx], expected, rtol=1e-7, atol=1e-9
            )


class TestFitHigherOrderCDM:
    """Tests for higher-order CDM estimation."""

    @staticmethod
    def _simulate(seed: int, n_persons: int = 2000):
        rng = np.random.default_rng(seed)
        n_attributes = 4
        q_matrix = np.zeros((24, n_attributes), dtype=int)
        for item_idx in range(24):
            q_matrix[item_idx, item_idx % n_attributes] = 1
            if item_idx >= 12:
                q_matrix[item_idx, (item_idx + 1) % n_attributes] = 1
        model = HigherOrderCDM(24, n_attributes, q_matrix, reduced_models=["DINA"] * 24)
        loadings = np.array([1.5, 1.2, 1.8, 1.0])
        thresholds = np.array([-0.5, 0.0, 0.5, 0.2])
        model.set_parameters(loadings=loadings, thresholds=thresholds)
        for item_idx in range(24):
            model.set_delta_parameters(item_idx, [0.15, 0.85])
        theta = rng.standard_normal(n_persons)
        cumulative = np.cumsum(model.pattern_probability(theta), axis=1)
        indices = np.argmax(rng.random((n_persons, 1)) < cumulative, axis=1)
        alpha = model.valid_patterns[indices]
        return model.simulate(alpha, seed=seed + 1), q_matrix, loadings, thresholds

    def test_recovers_item_and_higher_order_parameters(self):
        responses, q_matrix, loadings, thresholds = self._simulate(5)
        start = time.perf_counter()

        result = fit_higher_order_cdm(
            responses, q_matrix, reduced_models=["DINA"] * 24, max_iter=300, tol=1e-6
        )

        assert time.perf_counter() - start < 5.0
        fitted = result.model
        assert result.converged and fitted.is_fitted
        np.testing.assert_allclose(fitted.loadings, loadings, atol=0.35)
        np.testing.assert_allclose(fitted.thresholds, thresholds, atol=0.2)
        np.testing.assert_allclose(
            np.array(fitted.delta_parameters), [[0.15, 0.85]] * 24, atol=0.05
        )
        assert result.n_parameters == fitted.n_parameters == 24 * 2 + 8
        np.testing.assert_allclose(
            result.class_probabilities, fitted.class_probabilities
        )
        expected_ll = np.sum(
            np.log(
                np.exp(fitted.pattern_log_likelihoods(responses))
                @ result.class_probabilities
            )
        )
        np.testing.assert_allclose(result.log_likelihood, expected_ll)

    def test_em_increases_likelihood_under_hierarchy(self, capsys):
        responses, q_matrix, _, _ = self._simulate(8, n_persons=500)
        hierarchy = AttributeHierarchy(
            adjacency=np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        )

        result = fit_higher_order_cdm(
            responses, q_matrix, hierarchy=hierarchy, max_iter=25, verbose=True
        )

        lines = capsys.readouterr().out.splitlines()
        log_likelihoods = np.array([float(line.split("=")[1]) for line in lines])
        assert np.all(np.diff(log_likelihoods) > -1e-6)
        assert result.class_probabilities.shape == (9,)
        assert result.model.valid_patterns.shape == (9, 4)
        assert result.n_iterations <= 25

    def test_validates_controls(self):
        responses = np.array([[1, 0], [0, 1]])
        with pytest.raises(MirtValidationError, match="n_quad"):
            fit_higher_order_cdm(responses, np.eye(2, dtype=int), n_quad=0)
        with pytest.raises(MirtDataError, match="observed"):
            fit_higher_order_cdm(np.array([[1, -1], [0, -1]]), np.eye(2, dtype=int))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
