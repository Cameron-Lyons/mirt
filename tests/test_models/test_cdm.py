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

    def test_invalid_q_matrix(self):
        """Test that invalid Q-matrix raises error."""
        with pytest.raises(ValueError):
            DINA(n_items=5, n_attributes=3, q_matrix=np.ones((6, 3)))

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

    def test_classify_respondents(self, cdm_responses):
        """Test attribute classification."""
        q_matrix = cdm_responses["q_matrix"]
        model = DINA(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()

        responses = cdm_responses["responses"][:10]
        alphas = model.classify_respondents(responses)

        assert alphas.shape == (10, 3)
        assert np.all((alphas == 0) | (alphas == 1))


class TestDINO:
    """Tests for DINO model."""

    def test_init(self, q_matrix):
        """Test DINO initialization."""
        model = DINO(n_items=6, n_attributes=3, q_matrix=q_matrix)
        assert model.model_name == "DINO"

    def test_probability_or_rule(self, q_matrix):
        """Test that DINO uses OR rule."""
        model = DINO(n_items=6, n_attributes=3, q_matrix=q_matrix)
        model._initialize_parameters()

        alpha_partial = np.array([[1, 0, 0]])
        alpha_none = np.array([[0, 0, 0]])

        prob_partial = model.probability(alpha_partial)[0, 3]
        prob_none = model.probability(alpha_none)[0, 3]

        assert prob_partial > prob_none


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
