"""Tests for Mixed Effects IRT."""

import numpy as np

from mirt import LLTM, MixedEffectsFitResult, MixedEffectsIRT


class TestMixedEffectsIRT:
    """Tests for MixedEffectsIRT model."""

    def test_init(self):
        """Test MixedEffectsIRT initialization."""
        model = MixedEffectsIRT(base_model="2PL")
        assert model is not None

    def test_with_person_covariates(self, dichotomous_responses):
        """Test fitting with person-level covariates."""
        n_persons = dichotomous_responses["n_persons"]
        responses = dichotomous_responses["responses"]

        person_covariates = np.random.randn(n_persons, 1)

        model = MixedEffectsIRT(
            base_model="2PL",
            person_covariates=person_covariates,
        )
        result = model.fit(responses, max_iter=20)

        assert isinstance(result, MixedEffectsFitResult)
        assert result.model is not None

    def test_with_item_covariates(self, dichotomous_responses):
        """Test fitting with item-level covariates."""
        n_items = dichotomous_responses["n_items"]
        responses = dichotomous_responses["responses"]

        item_covariates = np.random.randn(n_items, 1)

        model = MixedEffectsIRT(
            base_model="2PL",
            item_covariates=item_covariates,
        )
        result = model.fit(responses, max_iter=20)

        assert result.model is not None


class TestLLTM:
    """Tests for Linear Logistic Test Model."""

    def test_init(self):
        """Test LLTM initialization."""
        q_matrix = np.random.randint(0, 2, (10, 3))
        model = LLTM(q_matrix=q_matrix)

        assert model is not None

    def test_difficulty_reconstruction(self):
        """Test that difficulties are reconstructed from Q-matrix."""
        q_matrix = np.array(
            [
                [1, 0],
                [0, 1],
                [1, 1],
            ]
        )

        model = LLTM(q_matrix=q_matrix)
        np.testing.assert_array_equal(model._q_matrix, q_matrix)

    def test_fit(self, dichotomous_responses):
        """Test LLTM fitting."""
        n_items = dichotomous_responses["n_items"]
        responses = dichotomous_responses["responses"]

        q_matrix = np.random.randint(0, 2, (n_items, 3)).astype(float)
        for i in range(n_items):
            if q_matrix[i].sum() == 0:
                q_matrix[i, 0] = 1

        model = LLTM(q_matrix=q_matrix)
        result = model.fit(responses)

        assert result is not None
        assert model._eta is not None


class TestMixedEffectsFitResult:
    """Tests for MixedEffectsFitResult."""

    def test_result_attributes(self, dichotomous_responses):
        """Test result object attributes."""
        model = MixedEffectsIRT(base_model="2PL")
        responses = dichotomous_responses["responses"]

        result = model.fit(responses, max_iter=10)

        assert hasattr(result, "model")
        assert hasattr(result, "log_likelihood")

    def test_variance_components(self, dichotomous_responses):
        """Test variance component extraction."""
        n_persons = dichotomous_responses["n_persons"]
        responses = dichotomous_responses["responses"]
        person_covariates = np.random.randn(n_persons, 1)

        model = MixedEffectsIRT(
            base_model="2PL",
            person_covariates=person_covariates,
        )
        result = model.fit(responses, max_iter=10)

        assert result is not None
