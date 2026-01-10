"""Tests for Cognitive Diagnosis Models (DINA, DINO)."""

import numpy as np
import pytest

from mirt import DINA, DINO, fit_cdm


class TestDINA:
    """Tests for DINA model."""

    def test_init(self, q_matrix):
        """Test DINA initialization."""
        model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)
        assert model.n_items == 6
        assert model.n_attributes == 3
        assert model.model_name == "DINA"
        assert model.q_matrix.shape == (6, 3)

    def test_init_with_item_names(self, q_matrix):
        """Test DINA initialization with item names."""
        item_names = [f"item_{i}" for i in range(6)]
        model = DINA(
            n_items=6, n_attributes=3, q_matrix=q_matrix, item_names=item_names
        )
        assert model.item_names == item_names

    def test_invalid_q_matrix(self):
        """Test that invalid Q-matrix raises error."""
        with pytest.raises(ValueError):
            DINA(n_items=5, n_attributes=3, q_matrix=np.ones((6, 3)))

    def test_invalid_q_matrix_wrong_attributes(self):
        """Test that Q-matrix with wrong number of attributes raises error."""
        with pytest.raises(ValueError):
            DINA(n_items=6, n_attributes=4, q_matrix=np.ones((6, 3)))

    def test_probability(self, q_matrix):
        """Test probability computation."""
        model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()

        alpha = np.array([[1, 1, 1]])
        probs = model.probability(alpha)
        assert probs.shape == (1, 6)
        assert np.all(probs > 0.5)

        alpha = np.array([[0, 0, 0]])
        probs = model.probability(alpha)
        assert np.all(probs < 0.5)

    def test_probability_single_item(self, q_matrix):
        """Test probability for a single item."""
        model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()

        alpha = np.array([[1, 1, 1]])
        prob = model.probability(alpha, item_idx=0)
        assert prob.shape == (1,)
        assert prob[0] > 0.5

    def test_probability_bounds(self, q_matrix):
        """Test that probabilities are within [0, 1]."""
        model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()

        patterns = model.attribute_patterns
        probs = model.probability(patterns)

        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_eta_and_rule(self, q_matrix):
        """Test that DINA uses AND rule (all required attributes needed)."""
        model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)

        alpha_both = np.array([[1, 1, 0]])
        alpha_one = np.array([[1, 0, 0]])
        alpha_other = np.array([[0, 1, 0]])

        assert model.eta(alpha_both, 3) == 1
        assert model.eta(alpha_one, 3) == 0
        assert model.eta(alpha_other, 3) == 0

    def test_classify_respondents(self, cdm_responses):
        """Test attribute classification."""
        q_matrix = cdm_responses["q_matrix"]
        model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()

        responses = cdm_responses["responses"][:10]
        alphas = model.classify_respondents(responses)

        assert alphas.shape == (10, 3)
        assert np.all((alphas == 0) | (alphas == 1))

    def test_classify_respondents_map(self, cdm_responses):
        """Test MAP classification."""
        q_matrix = cdm_responses["q_matrix"]
        model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()

        responses = cdm_responses["responses"][:10]
        alphas = model.classify_respondents(responses, method="MAP")

        assert alphas.shape == (10, 3)
        assert np.all((alphas == 0) | (alphas == 1))

    def test_log_likelihood(self, q_matrix):
        """Test log-likelihood computation."""
        model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()

        responses = np.array([[1, 1, 1, 1, 1, 1]])
        alpha = np.array([[1, 1, 1]])

        ll = model.log_likelihood(responses, alpha)

        assert ll.shape == (1,)
        assert np.isfinite(ll[0])
        assert ll[0] < 0  # Log-likelihood should be negative

    def test_log_likelihood_with_missing(self, q_matrix):
        """Test log-likelihood with missing data."""
        model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()

        responses = np.array([[1, -1, 1, 1, -1, 1]])  # -1 indicates missing
        alpha = np.array([[1, 1, 1]])

        ll = model.log_likelihood(responses, alpha)

        assert np.isfinite(ll[0])

    def test_information(self, q_matrix):
        """Test information function."""
        model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()

        alpha = np.array([[1, 1, 1]])
        info = model.information(alpha)

        assert info.shape == (1, 6)
        assert np.all(info >= 0)

    def test_information_single_item(self, q_matrix):
        """Test information for single item."""
        model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()

        alpha = np.array([[1, 1, 1]])
        info = model.information(alpha, item_idx=0)

        assert info.shape == (1,)
        assert info[0] >= 0

    def test_attribute_patterns(self, q_matrix):
        """Test attribute patterns generation."""
        model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)

        patterns = model.attribute_patterns
        assert patterns.shape == (8, 3)  # 2^3 = 8 patterns
        assert np.all((patterns == 0) | (patterns == 1))

    def test_slip_guess_properties(self, q_matrix):
        """Test slip and guess parameter properties."""
        model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()

        assert model.slip.shape == (6,)
        assert model.guess.shape == (6,)
        assert np.all(model.slip >= 0) and np.all(model.slip <= 1)
        assert np.all(model.guess >= 0) and np.all(model.guess <= 1)

    def test_copy(self, q_matrix):
        """Test model copying."""
        model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()
        model._parameters["slip"][0] = 0.15

        copied = model.copy()

        assert copied.n_items == model.n_items
        assert copied.n_attributes == model.n_attributes
        assert np.array_equal(copied.q_matrix, model.q_matrix)
        assert copied.parameters["slip"][0] == 0.15

        copied._parameters["slip"][0] = 0.25
        assert model.parameters["slip"][0] == 0.15

    def test_1d_alpha_input(self, q_matrix):
        """Test that 1D alpha input is handled correctly."""
        model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()

        alpha_1d = np.array([1, 1, 1])
        probs = model.probability(alpha_1d)

        assert probs.shape == (1, 6)


class TestDINO:
    """Tests for DINO model."""

    def test_init(self, q_matrix):
        """Test DINO initialization."""
        model = DINO(n_items=6, n_attributes=3, q_matrix=q_matrix)
        assert model.model_name == "DINO"
        assert model.n_items == 6
        assert model.n_attributes == 3

    def test_probability_or_rule(self, q_matrix):
        """Test that DINO uses OR rule."""
        model = DINO(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()

        alpha_partial = np.array([[1, 0, 0]])
        alpha_none = np.array([[0, 0, 0]])

        prob_partial = model.probability(alpha_partial)[0, 3]
        prob_none = model.probability(alpha_none)[0, 3]

        assert prob_partial > prob_none

    def test_eta_or_rule(self, q_matrix):
        """Test that DINO uses OR rule for eta computation."""
        model = DINO(n_items=6, n_attributes=3, q_matrix=q_matrix)

        alpha_both = np.array([[1, 1, 0]])
        alpha_first = np.array([[1, 0, 0]])
        alpha_second = np.array([[0, 1, 0]])
        alpha_neither = np.array([[0, 0, 1]])

        assert model.eta(alpha_both, 3) == 1
        assert model.eta(alpha_first, 3) == 1
        assert model.eta(alpha_second, 3) == 1
        assert model.eta(alpha_neither, 3) == 0

    def test_probability_single_item(self, q_matrix):
        """Test probability for a single item."""
        model = DINO(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()

        alpha = np.array([[1, 1, 1]])
        prob = model.probability(alpha, item_idx=0)
        assert prob.shape == (1,)

    def test_log_likelihood(self, q_matrix):
        """Test log-likelihood computation."""
        model = DINO(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()

        responses = np.array([[1, 1, 1, 1, 1, 1]])
        alpha = np.array([[1, 1, 1]])

        ll = model.log_likelihood(responses, alpha)

        assert ll.shape == (1,)
        assert np.isfinite(ll[0])

    def test_information(self, q_matrix):
        """Test information function."""
        model = DINO(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()

        alpha = np.array([[1, 1, 1]])
        info = model.information(alpha)

        assert info.shape == (1, 6)
        assert np.all(info >= 0)

    def test_classify_respondents(self, cdm_responses):
        """Test attribute classification."""
        q_matrix = cdm_responses["q_matrix"]
        model = DINO(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()

        responses = cdm_responses["responses"][:10]
        alphas = model.classify_respondents(responses)

        assert alphas.shape == (10, 3)
        assert np.all((alphas == 0) | (alphas == 1))

    def test_classify_respondents_map(self, cdm_responses):
        """Test MAP classification for DINO."""
        q_matrix = cdm_responses["q_matrix"]
        model = DINO(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()

        responses = cdm_responses["responses"][:10]
        alphas = model.classify_respondents(responses, method="MAP")

        assert alphas.shape == (10, 3)
        assert np.all((alphas == 0) | (alphas == 1))

    def test_copy(self, q_matrix):
        """Test model copying."""
        model = DINO(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()
        model._parameters["slip"][0] = 0.15

        copied = model.copy()

        assert copied.n_items == model.n_items
        assert copied.model_name == "DINO"
        assert np.array_equal(copied.q_matrix, model.q_matrix)

        copied._parameters["slip"][0] = 0.25
        assert model.parameters["slip"][0] == 0.15

    def test_slip_guess_properties(self, q_matrix):
        """Test slip and guess parameter properties."""
        model = DINO(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()

        assert model.slip.shape == (6,)
        assert model.guess.shape == (6,)


class TestFitCDM:
    """Tests for CDM fitting."""

    def test_fit_dina(self, cdm_responses):
        """Test fitting DINA model."""
        model, class_probs = fit_cdm(
            responses=cdm_responses["responses"],
            q_matrix=cdm_responses["q_matrix"],
            model="DINA",
            max_iter=20,
        )

        assert model._is_fitted
        n_patterns = 2 ** cdm_responses["n_attrs"]
        assert class_probs.shape[0] == n_patterns

        slip = model._parameters["slip"]
        guess = model._parameters["guess"]
        assert np.all((slip >= 0) & (slip <= 1))
        assert np.all((guess >= 0) & (guess <= 1))

    def test_fit_dino(self, cdm_responses):
        """Test fitting DINO model."""
        model, class_probs = fit_cdm(
            responses=cdm_responses["responses"],
            q_matrix=cdm_responses["q_matrix"],
            model="DINO",
            max_iter=20,
        )

        assert model._is_fitted
        assert model.model_name == "DINO"

    def test_fit_cdm_unknown_model(self, cdm_responses):
        """Test fitting with unknown model type."""
        with pytest.raises(ValueError, match="Unknown CDM model"):
            fit_cdm(
                responses=cdm_responses["responses"],
                q_matrix=cdm_responses["q_matrix"],
                model="UNKNOWN",
            )

    def test_fit_cdm_class_probabilities_sum_to_one(self, cdm_responses):
        """Test that class probabilities sum to 1."""
        model, class_probs = fit_cdm(
            responses=cdm_responses["responses"],
            q_matrix=cdm_responses["q_matrix"],
            model="DINA",
            max_iter=30,
        )

        assert np.isclose(np.sum(class_probs), 1.0, atol=1e-6)

    def test_fit_cdm_convergence(self, cdm_responses):
        """Test CDM fitting converges."""
        model, class_probs = fit_cdm(
            responses=cdm_responses["responses"],
            q_matrix=cdm_responses["q_matrix"],
            model="DINA",
            max_iter=50,
            tol=1e-4,
        )

        assert model._is_fitted
        assert np.all(model.slip >= 0.001) and np.all(model.slip <= 0.999)
        assert np.all(model.guess >= 0.001) and np.all(model.guess <= 0.999)

    def test_fit_cdm_case_insensitive(self, cdm_responses):
        """Test that model parameter is case insensitive."""
        model1, _ = fit_cdm(
            responses=cdm_responses["responses"],
            q_matrix=cdm_responses["q_matrix"],
            model="dina",
            max_iter=10,
        )
        model2, _ = fit_cdm(
            responses=cdm_responses["responses"],
            q_matrix=cdm_responses["q_matrix"],
            model="DINA",
            max_iter=10,
        )

        assert model1.model_name == model2.model_name

    def test_fit_cdm_parameter_recovery(self, cdm_responses):
        """Test that CDM can recover parameters reasonably."""
        true_slip = cdm_responses["slip"]
        true_guess = cdm_responses["guess"]

        model, _ = fit_cdm(
            responses=cdm_responses["responses"],
            q_matrix=cdm_responses["q_matrix"],
            model="DINA",
            max_iter=50,
        )

        assert np.corrcoef(model.slip, true_slip)[0, 1] > 0 or np.std(true_slip) < 0.01
        assert np.mean(np.abs(model.slip - true_slip)) < 0.3
        assert np.mean(np.abs(model.guess - true_guess)) < 0.3


class TestCDMEdgeCases:
    """Tests for CDM edge cases."""

    def test_single_attribute(self):
        """Test CDM with single attribute."""
        n_items = 4
        q_matrix = np.ones((n_items, 1), dtype=int)

        model = DINA(n_items=n_items, n_attributes=1, q_matrix=q_matrix)
        model._initialize_parameters()

        alpha = np.array([[1], [0]])
        probs = model.probability(alpha)

        assert probs.shape == (2, n_items)
        assert np.all(probs[0] > probs[1])  # Mastery should have higher prob

    def test_many_attributes(self):
        """Test CDM with many attributes."""
        rng = np.random.default_rng(42)
        n_items = 10
        n_attrs = 5
        q_matrix = rng.integers(0, 2, (n_items, n_attrs))
        for i in range(n_items):
            if q_matrix[i].sum() == 0:
                q_matrix[i, rng.integers(0, n_attrs)] = 1

        model = DINA(n_items=n_items, n_attributes=n_attrs, q_matrix=q_matrix)
        model._initialize_parameters()

        patterns = model.attribute_patterns
        assert patterns.shape == (2**n_attrs, n_attrs)

        probs = model.probability(patterns)
        assert probs.shape == (2**n_attrs, n_items)

    def test_sparse_q_matrix(self):
        """Test CDM with sparse Q-matrix (each item requires only one attribute)."""
        n_items = 6
        n_attrs = 3
        q_matrix = np.zeros((n_items, n_attrs), dtype=int)
        for i in range(n_items):
            q_matrix[i, i % n_attrs] = 1

        model = DINA(n_items=n_items, n_attributes=n_attrs, q_matrix=q_matrix)
        model._initialize_parameters()

        alpha = np.array([[1, 0, 0]])
        probs = model.probability(alpha)

        assert probs[0, 0] > 0.5
        assert probs[0, 3] > 0.5

    def test_dense_q_matrix(self):
        """Test CDM with dense Q-matrix (items require multiple attributes)."""
        n_items = 4
        n_attrs = 3
        q_matrix = np.ones((n_items, n_attrs), dtype=int)  # All items need all attrs

        model = DINA(n_items=n_items, n_attributes=n_attrs, q_matrix=q_matrix)
        model._initialize_parameters()

        alpha_full = np.array([[1, 1, 1]])
        alpha_partial = np.array([[1, 1, 0]])

        probs_full = model.probability(alpha_full)
        probs_partial = model.probability(alpha_partial)

        assert np.all(probs_full > probs_partial)

    def test_item_with_no_required_attributes(self):
        """Test DINO with item requiring no attributes (edge case)."""
        n_items = 4
        n_attrs = 2
        q_matrix = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])  # Last item requires none

        model = DINO(n_items=n_items, n_attributes=n_attrs, q_matrix=q_matrix)
        model._initialize_parameters()

        alpha_none = np.array([[0, 0]])
        eta = model.eta(alpha_none, 3)

        assert eta == 1
