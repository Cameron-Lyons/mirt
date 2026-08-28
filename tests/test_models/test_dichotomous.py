"""Tests for dichotomous IRT models."""

import numpy as np
import pytest

from mirt.exceptions import MirtValidationError
from mirt.models.dichotomous import (
    ComplementaryLogLog,
    FiveParameterLogistic,
    FourParameterLogistic,
    NegativeLogLog,
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
    UnipolarLogLogistic,
)


class TestTwoParameterLogistic:
    """Tests for 2PL model."""

    def test_initialization(self):
        """Test model initialization."""
        model = TwoParameterLogistic(n_items=10)

        assert model.n_items == 10
        assert model.n_factors == 1
        assert model.model_name == "2PL"
        assert not model.is_fitted
        assert len(model.item_names) == 10

    def test_probability_shape(self):
        """Test probability output shape."""
        model = TwoParameterLogistic(n_items=5)

        theta = np.linspace(-3, 3, 100)
        probs = model.probability(theta)

        assert probs.shape == (100, 5)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_probability_single_item(self):
        """Test probability for single item."""
        model = TwoParameterLogistic(n_items=5)

        theta = np.array([0.0])
        prob = model.probability(theta, item_idx=0)

        assert prob.shape == (1,)
        assert 0 <= prob[0] <= 1

    def test_icc_at_difficulty(self):
        """Test ICC equals 0.5 at difficulty parameter."""
        model = TwoParameterLogistic(n_items=1)
        model.set_parameters(difficulty=np.array([1.0]))

        theta = np.array([1.0])
        prob = model.icc(theta, item_idx=0)

        np.testing.assert_almost_equal(prob[0], 0.5, decimal=5)

    def test_log_likelihood(self):
        """Test log-likelihood computation."""
        model = TwoParameterLogistic(n_items=3)

        responses = np.array([[1, 0, 1], [0, 1, 0]])
        theta = np.array([[0.0], [0.0]])

        ll = model.log_likelihood(responses, theta)

        assert ll.shape == (2,)
        assert np.all(ll <= 0)

    def test_log_likelihood_batch_matches_pointwise_with_missing_data(self):
        """Batch likelihoods match independently evaluated theta points."""
        model = TwoParameterLogistic(n_items=4, n_factors=2)
        model.set_parameters(
            discrimination=np.array([[0.8, 0.2], [0.1, 1.4], [1.1, 0.7], [0.6, 1.0]]),
            difficulty=np.array([-1.0, -0.2, 0.5, 1.3]),
        )
        responses = np.array([[1, 0, -1, 1], [0, 1, 1, -1], [1, 1, 0, 0]])
        theta = np.array([[-1.5, 0.2], [0.0, 0.0], [1.2, -0.7]])

        actual = model.log_likelihood_batch(responses, theta)
        expected = np.column_stack(
            [
                model.log_likelihood(
                    responses,
                    np.broadcast_to(point, (responses.shape[0], model.n_factors)),
                )
                for point in theta
            ]
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_information(self):
        """Test Fisher information computation."""
        model = TwoParameterLogistic(n_items=5)

        theta = np.linspace(-3, 3, 50)
        info = model.information(theta)

        assert info.shape == (50, 5)
        assert np.all(info >= 0)

    def test_multidimensional(self):
        """Test multidimensional 2PL."""
        model = TwoParameterLogistic(n_items=10, n_factors=2)

        assert model.n_factors == 2
        assert model.discrimination.shape == (10, 2)

        theta = np.random.randn(20, 2)
        probs = model.probability(theta)

        assert probs.shape == (20, 10)


class TestOneParameterLogistic:
    """Tests for 1PL/Rasch model."""

    def test_discrimination_fixed(self):
        """Test that discrimination is fixed to 1."""
        model = OneParameterLogistic(n_items=5)

        np.testing.assert_array_equal(model.discrimination, np.ones(5))

    def test_cannot_set_discrimination(self):
        """Test that discrimination cannot be modified."""
        model = OneParameterLogistic(n_items=5)

        with pytest.raises(ValueError, match="Cannot set discrimination"):
            model.set_parameters(discrimination=np.array([2.0] * 5))

    def test_no_multidimensional(self):
        """Test that 1PL doesn't support multidimensional."""
        with pytest.raises(ValueError, match="unidimensional"):
            OneParameterLogistic(n_items=5, n_factors=2)


class TestThreeParameterLogistic:
    """Tests for 3PL model."""

    def test_guessing_parameter(self):
        """Test guessing parameter effect."""
        model = ThreeParameterLogistic(n_items=1)
        model.set_parameters(
            guessing=np.array([0.25]),
            difficulty=np.array([10.0]),
        )

        theta = np.array([-10.0])
        prob = model.probability(theta, item_idx=0)

        np.testing.assert_almost_equal(prob[0], 0.25, decimal=2)

    def test_probability_bounds(self):
        """Test probabilities are bounded by guessing and 1."""
        model = ThreeParameterLogistic(n_items=5)

        theta = np.linspace(-5, 5, 100)
        probs = model.probability(theta)

        guessing = model.guessing

        for i in range(5):
            assert np.all(probs[:, i] >= guessing[i] - 1e-6)


class TestFourParameterLogistic:
    """Tests for 4PL model."""

    def test_upper_asymptote(self):
        """Test upper asymptote effect."""
        model = FourParameterLogistic(n_items=1)
        model.set_parameters(
            upper=np.array([0.9]),
            difficulty=np.array([-10.0]),
        )

        theta = np.array([10.0])
        prob = model.probability(theta, item_idx=0)

        np.testing.assert_almost_equal(prob[0], 0.9, decimal=2)

    def test_probability_bounds(self):
        """Test probabilities are bounded by guessing and upper."""
        model = FourParameterLogistic(n_items=3)
        model.set_parameters(
            guessing=np.array([0.1, 0.2, 0.15]),
            upper=np.array([0.95, 0.9, 0.85]),
        )

        theta = np.linspace(-5, 5, 100)
        probs = model.probability(theta)

        for i in range(3):
            assert np.all(probs[:, i] >= model.guessing[i] - 1e-6)
            assert np.all(probs[:, i] <= model.upper[i] + 1e-6)


@pytest.mark.parametrize(
    ("model", "parameter", "values", "message"),
    [
        (
            TwoParameterLogistic(2),
            "discrimination",
            np.array([1.0, np.nan]),
            "finite",
        ),
        (
            TwoParameterLogistic(2),
            "difficulty",
            np.array([0.0, np.nan]),
            "finite",
        ),
        (
            ThreeParameterLogistic(2),
            "guessing",
            np.array([-0.01, 0.2]),
            r"\[0, 1\)",
        ),
        (
            ThreeParameterLogistic(2),
            "guessing",
            np.array([0.2, 1.0]),
            r"\[0, 1\)",
        ),
        (
            FourParameterLogistic(2),
            "upper",
            np.array([-0.01, 0.9]),
            r"\[0, 1\]",
        ),
        (
            FourParameterLogistic(2),
            "upper",
            np.array([0.9, 1.01]),
            r"\[0, 1\]",
        ),
        (
            FiveParameterLogistic(2),
            "asymmetry",
            np.array([1.0, 0.0]),
            "strictly positive",
        ),
        (
            UnipolarLogLogistic(2),
            "discrimination",
            np.array([1.0, 0.0]),
            "strictly positive",
        ),
        (
            ComplementaryLogLog(2),
            "discrimination",
            np.array([np.nan, 1.0]),
            "finite",
        ),
        (
            NegativeLogLog(2),
            "difficulty",
            np.array([0.0, -np.inf]),
            "finite",
        ),
    ],
)
def test_parameter_domains_are_validated(model, parameter, values, message):
    before = model.parameters

    with pytest.raises(MirtValidationError, match=message):
        model.set_parameters(**{parameter: values})

    for name, expected in before.items():
        np.testing.assert_array_equal(model.parameters[name], expected)


def test_parameter_updates_are_atomic_and_detached():
    model = FourParameterLogistic(2)
    difficulty = np.array([0.2, 0.4])

    with pytest.raises(MirtValidationError, match="guessing"):
        model.set_parameters(
            difficulty=difficulty,
            guessing=np.array([0.1, 1.2]),
        )
    np.testing.assert_array_equal(model.difficulty, [0.0, 0.0])

    model.set_parameters(difficulty=difficulty)
    difficulty[0] = 99.0
    np.testing.assert_array_equal(model.difficulty, [0.2, 0.4])


def test_asymptotes_are_validated_against_joint_proposed_state():
    model = FourParameterLogistic(2)

    model.set_parameters(
        guessing=np.array([0.8, 0.7]),
        upper=np.array([0.9, 0.95]),
    )
    np.testing.assert_array_equal(model.guessing, [0.8, 0.7])
    np.testing.assert_array_equal(model.upper, [0.9, 0.95])

    with pytest.raises(MirtValidationError, match="cannot exceed"):
        model.set_parameters(guessing=np.array([0.91, 0.7]))
    np.testing.assert_array_equal(model.guessing, [0.8, 0.7])


def test_logistic_discrimination_allows_signed_loadings():
    unidimensional = TwoParameterLogistic(2)
    unidimensional.set_parameters(discrimination=np.array([-1.0, 0.0]))
    np.testing.assert_array_equal(unidimensional.discrimination, [-1.0, 0.0])

    model = TwoParameterLogistic(2, n_factors=2)
    loadings = np.array([[1.0, -0.2], [-0.4, 0.8]])

    model.set_parameters(discrimination=loadings)
    np.testing.assert_array_equal(model.discrimination, loadings)


def test_item_parameter_updates_preserve_validation():
    model = FourParameterLogistic(2)
    model.set_item_parameter(1, "guessing", 0.4)
    assert model.guessing[1] == pytest.approx(0.4)

    with pytest.raises(MirtValidationError, match="cannot exceed"):
        model.set_item_parameter(1, "upper", 0.3)
    assert model.upper[1] == pytest.approx(1.0)

    with pytest.raises(MirtValidationError, match="scalar"):
        model.set_item_parameter(1, "guessing", np.array([0.2]))


def test_one_parameter_item_update_allows_fixed_value_only():
    model = OneParameterLogistic(2)
    model.set_item_parameter(0, "discrimination", 1.0)

    with pytest.raises(ValueError, match="Cannot set discrimination"):
        model.set_item_parameter(0, "discrimination", 1.1)


@pytest.mark.parametrize("item_idx", [-1, 2, 1.5, True])
def test_item_parameter_updates_reject_invalid_indices(item_idx):
    with pytest.raises((IndexError, MirtValidationError), match="item_idx|Item index"):
        TwoParameterLogistic(2).set_item_parameter(item_idx, "difficulty", 0.0)
