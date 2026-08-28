"""Tests for Mixture IRT model."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import expit, logsumexp, roots_hermite

from mirt import MixtureIRT, fit_mixture_irt
from mirt.models.mixture import _item_expected_objective


@pytest.fixture
def separated_mixture():
    """Create a two-class model with strongly separated item curves."""
    model = MixtureIRT(n_items=3, n_classes=2, base_model="2PL")
    model.set_parameters(
        class_proportions=np.array([0.35, 0.65]),
        discrimination_class0=np.array([0.7, 1.0, 1.3]),
        difficulty_class0=np.array([-1.5, -1.0, -0.5]),
        discrimination_class1=np.array([1.8, 1.4, 1.1]),
        difficulty_class1=np.array([0.5, 1.0, 1.5]),
    )
    return model


class TestMixtureIRT:
    """Tests for MixtureIRT model."""

    def test_init(self):
        """Test MixtureIRT initialization."""
        model = MixtureIRT(n_items=10, n_classes=2)
        assert model.n_items == 10
        assert model.n_classes == 2
        assert model.base_model == "2PL"
        assert model.model_name == "MixtureIRT"

    def test_invalid_n_classes(self):
        """Test that n_classes < 2 raises error."""
        with pytest.raises(ValueError):
            MixtureIRT(n_items=10, n_classes=1)

    @pytest.mark.parametrize("n_classes", [True, 2.5, "2"])
    def test_rejects_non_integer_class_counts(self, n_classes):
        """Reject ambiguous class-count inputs."""
        with pytest.raises(ValueError, match="n_classes must be an integer"):
            MixtureIRT(n_items=4, n_classes=n_classes)

    def test_rejects_unknown_base_model(self):
        """Reject model names that would otherwise silently act like a 2PL."""
        with pytest.raises(ValueError, match="Unknown base_model"):
            MixtureIRT(n_items=4, base_model="4PL")

    def test_class_proportions(self):
        """Test class proportion initialization."""
        model = MixtureIRT(n_items=10, n_classes=3)
        model._initialize_parameters()

        props = model.class_proportions
        assert len(props) == 3
        assert np.isclose(props.sum(), 1.0)

    def test_class_parameters(self):
        """Test getting class-specific parameters."""
        model = MixtureIRT(n_items=10, n_classes=2, base_model="3PL")
        model._initialize_parameters()

        params = model.get_class_parameters(0)
        assert "discrimination" in params
        assert "difficulty" in params
        assert "guessing" in params
        assert len(params["difficulty"]) == 10

    def test_public_parameter_accessors_return_detached_arrays(self):
        """Public accessors cannot mutate stored class parameters."""
        model = MixtureIRT(n_items=2, n_classes=2, base_model="3PL")
        before = model.parameters

        model.class_proportions[:] = [1.0, 0.0]
        class_parameters = model.get_class_parameters(0)
        for values in class_parameters.values():
            values[:] = 99.0

        for name, values in before.items():
            assert_allclose(model.parameters[name], values)

    def test_item_updates_preserve_class_parameter_domains(self):
        """Per-item edits run the mixture model's domain validation."""
        model = MixtureIRT(n_items=2, n_classes=2, base_model="3PL")
        before = model.parameters

        with pytest.raises(ValueError, match="Discriminations must be positive"):
            model.set_item_parameter(0, "discrimination_class0", 0.0)
        with pytest.raises(ValueError, match="Guessing parameters"):
            model.set_item_parameter(1, "guessing_class1", 1.0)

        for name, values in before.items():
            assert_allclose(model.parameters[name], values)

    def test_probability(self):
        """Test marginal probability computation."""
        model = MixtureIRT(n_items=10, n_classes=2)
        model._initialize_parameters()

        theta = np.array([[0.0], [1.0], [-1.0]])
        probs = model.probability(theta)

        assert probs.shape == (3, 10)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_probability_matches_weighted_class_curves(self, separated_mixture):
        """Marginal item curves equal the weighted component curves."""
        theta = np.array([-1.0, 0.0, 1.0])
        expected = sum(
            separated_mixture.class_proportions[class_idx]
            * separated_mixture.class_probability(theta, class_idx)
            for class_idx in range(separated_mixture.n_classes)
        )

        assert_allclose(separated_mixture.probability(theta), expected)

    def test_joint_log_likelihood_preserves_shared_class(self):
        """Mix complete response-pattern likelihoods rather than each item."""
        model = MixtureIRT(n_items=2, n_classes=2, base_model="2PL")
        model.set_parameters(
            class_proportions=np.array([0.5, 0.5]),
            discrimination_class0=np.ones(2),
            difficulty_class0=np.full(2, -2.0),
            discrimination_class1=np.ones(2),
            difficulty_class1=np.full(2, 2.0),
        )
        responses = np.array([[1, 1]])
        theta = np.array([0.0])
        log_joint = np.array(
            [
                np.log(model.class_proportions[class_idx])
                + np.log(model.class_probability(theta, class_idx)).sum()
                for class_idx in range(model.n_classes)
            ]
        )

        actual = model.log_likelihood(responses, theta)

        assert actual[0] == pytest.approx(logsumexp(log_joint))
        itemwise_mixture = np.log(model.probability(theta)).sum()
        assert actual[0] != pytest.approx(itemwise_mixture)

    def test_log_likelihood_batch_matches_scalar_evaluation(self, separated_mixture):
        """The batched theta grid preserves scalar likelihood semantics."""
        responses = np.array([[1, 1, 0], [0, -1, 1], [1, 0, 1]])
        theta = np.linspace(-2.0, 2.0, 7)

        batched = separated_mixture.log_likelihood_batch(responses, theta)
        scalar = np.column_stack(
            [
                separated_mixture.log_likelihood(responses, np.array([value]))
                for value in theta
            ]
        )

        assert batched.shape == (3, 7)
        assert_allclose(batched, scalar, rtol=1e-13, atol=1e-13)

    def test_paired_likelihood_broadcasting(self, separated_mixture):
        """Allow one common theta or one response pattern to broadcast."""
        responses = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 1]])

        by_person = separated_mixture.log_likelihood(responses, np.array([0.0]))
        by_theta = separated_mixture.log_likelihood(
            responses[:1], np.array([-1.0, 0.0, 1.0])
        )

        assert by_person.shape == (3,)
        assert by_theta.shape == (3,)
        with pytest.raises(ValueError, match="equal row counts"):
            separated_mixture.log_likelihood(responses, np.array([-1.0, 0.0]))

    def test_class_posterior(self, dichotomous_responses):
        """Test class posterior computation."""
        responses = dichotomous_responses["responses"]
        n_items = dichotomous_responses["n_items"]
        model = MixtureIRT(n_items=n_items, n_classes=2)
        model._initialize_parameters()

        theta = np.zeros((len(responses), 1))
        posterior = model.class_posterior(responses, theta)

        assert posterior.shape == (len(responses), 2)
        assert np.allclose(posterior.sum(axis=1), 1.0)

    def test_class_posterior_matches_manual_joint(self, separated_mixture):
        """Normalize class priors times complete-pattern likelihoods."""
        responses = np.array([[1, 1, 0], [0, 0, 1]])
        theta = np.array([0.2, -0.4])
        log_joint = np.empty((2, 2))
        for class_idx in range(2):
            probability = separated_mixture.class_probability(theta, class_idx)
            log_joint[:, class_idx] = np.log(
                separated_mixture.class_proportions[class_idx]
            ) + (
                responses * np.log(probability)
                + (1 - responses) * np.log1p(-probability)
            ).sum(axis=1)
        expected = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))

        assert_allclose(separated_mixture.class_posterior(responses, theta), expected)

    def test_all_missing_posterior_equals_prior(self, separated_mixture):
        """An empty response pattern contributes no class evidence."""
        posterior = separated_mixture.class_posterior(np.full((4, 3), -1), np.zeros(4))

        assert_allclose(
            posterior,
            np.broadcast_to(separated_mixture.class_proportions, (4, 2)),
        )

    def test_classify_persons(self, dichotomous_responses):
        """Test person classification."""
        responses = dichotomous_responses["responses"]
        n_items = dichotomous_responses["n_items"]
        model = MixtureIRT(n_items=n_items, n_classes=2)
        model._initialize_parameters()

        theta = np.zeros((len(responses), 1))
        classes = model.classify_persons(responses, theta)

        assert classes.shape == (len(responses),)
        assert set(classes).issubset({0, 1})

    def test_information_matches_probability_derivative(self, separated_mixture):
        """Use the derivative of the marginal mixture curve."""
        theta = np.array([-0.7, 0.2, 1.1])
        step = 1e-5
        probability = separated_mixture.probability(theta)
        derivative = (
            separated_mixture.probability(theta + step)
            - separated_mixture.probability(theta - step)
        ) / (2.0 * step)
        expected = derivative**2 / (probability * (1.0 - probability))

        assert_allclose(
            separated_mixture.information(theta), expected, rtol=2e-9, atol=1e-10
        )
        assert_allclose(
            separated_mixture.information(theta, item_idx=1),
            expected[:, 1],
            rtol=2e-9,
            atol=1e-10,
        )

    @pytest.mark.parametrize("class_idx", [-1, 2, 1.5, True])
    def test_rejects_invalid_class_index(self, separated_mixture, class_idx):
        """Validate class selection before parameter access."""
        with pytest.raises(IndexError, match="class_idx"):
            separated_mixture.get_class_parameters(class_idx)

    @pytest.mark.parametrize("item_idx", [-1, 3, 1.5, True])
    def test_rejects_invalid_item_index(self, separated_mixture, item_idx):
        """Validate item selection before curve evaluation."""
        with pytest.raises(IndexError, match="item_idx"):
            separated_mixture.probability(np.array([0.0]), item_idx=item_idx)

    @pytest.mark.parametrize(
        ("name", "value", "message"),
        [
            ("class_proportions", np.array([0.7, 0.7]), "sum to 1"),
            ("class_proportions", np.array([-0.1, 1.1]), "non-negative"),
            ("discrimination_class0", np.array([1.0, 0.0, 1.0]), "positive"),
            ("difficulty_class0", np.array([0.0, np.nan, 1.0]), "finite"),
        ],
    )
    def test_rejects_invalid_parameters(self, separated_mixture, name, value, message):
        """Reject invalid class probabilities and item parameters."""
        with pytest.raises(ValueError, match=message):
            separated_mixture.set_parameters(**{name: value})

    @pytest.mark.parametrize(
        "responses",
        [
            np.array([[0, 1, 2]]),
            np.array([[0, 1]]),
            np.array([0, 1, 0]),
        ],
    )
    def test_rejects_invalid_responses(self, separated_mixture, responses):
        """Enforce a binary response matrix with the configured item count."""
        with pytest.raises(ValueError):
            separated_mixture.log_likelihood(responses, np.array([0.0]))


class TestFitMixtureIRT:
    """Tests for mixture IRT fitting."""

    def test_fit_mixture(self, dichotomous_responses):
        """Test fitting mixture IRT model."""
        responses = dichotomous_responses["responses"]

        model, posteriors = fit_mixture_irt(
            responses=responses,
            n_classes=2,
            base_model="2PL",
            max_iter=20,
        )

        assert model._is_fitted
        assert posteriors.shape == (len(responses), 2)
        assert np.allclose(posteriors.sum(axis=1), 1.0)
        history = model.convergence_info["log_likelihood_history"]
        assert np.all(np.diff(history) >= -1e-8)
        assert model.convergence_info["log_likelihood"] == history[-1]
        assert not np.allclose(model.get_class_parameters(0)["discrimination"], 1.0)

    def test_returned_posteriors_match_final_parameters(self):
        """Return E-step weights aligned with the final M-step parameters."""
        responses = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]] * 20)
        model, posterior = fit_mixture_irt(
            responses, base_model="1PL", max_iter=4, n_quadpts=11
        )
        nodes, weights = roots_hermite(11)
        nodes *= np.sqrt(2.0)
        weights /= np.sqrt(np.pi)
        evidence = np.empty_like(posterior)
        for class_idx in range(model.n_classes):
            class_evidence = np.zeros(responses.shape[0])
            for node, weight in zip(nodes, weights, strict=True):
                probability = model.class_probability(np.array([node]), class_idx)[0]
                log_likelihood = (
                    responses * np.log(probability)
                    + (1 - responses) * np.log1p(-probability)
                ).sum(axis=1)
                class_evidence += weight * np.exp(log_likelihood)
            evidence[:, class_idx] = model.class_proportions[class_idx] * class_evidence
        evidence /= evidence.sum(axis=1, keepdims=True)

        assert_allclose(posterior, evidence, rtol=1e-12, atol=1e-12)

    def test_fit_with_3pl(self, dichotomous_responses):
        """Test fitting mixture 3PL model."""
        responses = dichotomous_responses["responses"]

        model, _ = fit_mixture_irt(
            responses=responses,
            n_classes=2,
            base_model="3PL",
            max_iter=10,
        )

        assert model._is_fitted
        params = model.get_class_parameters(0)
        assert "guessing" in params
        assert np.all((params["guessing"] >= 0.0) & (params["guessing"] <= 0.5))
        assert not np.allclose(params["guessing"], 0.2)

    def test_recovers_separated_one_parameter_classes(self):
        """Recover class ordering and useful posterior assignments."""
        rng = np.random.default_rng(11)
        n_persons = 1200
        true_classes = rng.choice(2, size=n_persons, p=[0.35, 0.65])
        theta = rng.normal(size=n_persons)
        difficulty = np.array(
            [
                [-1.2, -0.9, -0.7, -1.0, -0.5, -1.4],
                [0.8, 1.1, 0.6, 1.3, 0.9, 0.5],
            ]
        )
        probability = expit(theta[:, None] - difficulty[true_classes])
        responses = rng.binomial(1, probability)

        model, posterior = fit_mixture_irt(
            responses,
            n_classes=2,
            base_model="1PL",
            max_iter=50,
            tol=1e-5,
        )

        fitted_locations = np.array(
            [
                model.get_class_parameters(class_idx)["difficulty"].mean()
                for class_idx in range(2)
            ]
        )
        assignments = posterior.argmax(axis=1)
        accuracy = max(
            np.mean(assignments == true_classes),
            np.mean((1 - assignments) == true_classes),
        )
        assert fitted_locations[1] - fitted_locations[0] > 1.0
        assert accuracy > 0.7
        assert model.convergence_info["converged"]

    def test_all_missing_data_is_stable(self):
        """Return the prior class distribution when no items are observed."""
        responses = np.full((20, 4), -1)

        model, posterior = fit_mixture_irt(responses, max_iter=5)

        assert_allclose(posterior, np.full((20, 2), 0.5))
        assert model.convergence_info["converged"]
        assert model.convergence_info["log_likelihood"] == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"max_iter": 0}, "max_iter"),
            ({"max_iter": 1.5}, "max_iter"),
            ({"n_quadpts": 2}, "n_quadpts"),
            ({"n_quadpts": True}, "n_quadpts"),
            ({"tol": 0.0}, "tol"),
            ({"tol": np.nan}, "tol"),
        ],
    )
    def test_rejects_invalid_fit_controls(self, kwargs, message):
        """Validate controls before allocating EM work arrays."""
        responses = np.array([[0, 1], [1, 0]])
        with pytest.raises(ValueError, match=message):
            fit_mixture_irt(responses, **kwargs)

    def test_copy_preserves_independent_convergence_history(self):
        """Copy fitted state without sharing mutable diagnostics."""
        responses = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 10)
        model, _ = fit_mixture_irt(responses, max_iter=3)

        copied = model.copy()
        copied_history = copied.convergence_info["log_likelihood_history"]
        copied_history[0] = 123.0

        assert copied.is_fitted
        assert copied.convergence_info["log_likelihood_history"][0] != 123.0
        assert model.convergence_info["log_likelihood_history"][0] != 123.0


@pytest.mark.parametrize(
    ("base_model", "values"),
    [
        ("1PL", np.array([0.3])),
        ("2PL", np.array([np.log(1.2), 0.3])),
        ("3PL", np.array([np.log(1.2), 0.3, 0.15])),
    ],
)
def test_item_objective_analytic_gradient(base_model, values):
    """Match each item optimizer gradient to finite differences."""
    nodes = np.array([-1.5, -0.4, 0.2, 1.1, 2.0])
    expected_total = np.array([4.0, 7.0, 9.0, 6.0, 3.0])
    expected_correct = np.array([0.3, 2.1, 4.8, 4.7, 2.8])
    _, gradient = _item_expected_objective(
        values, base_model, nodes, expected_correct, expected_total
    )
    numerical = np.empty_like(values)
    step = 1e-6
    for index in range(values.size):
        upper = values.copy()
        lower = values.copy()
        upper[index] += step
        lower[index] -= step
        upper_value, _ = _item_expected_objective(
            upper, base_model, nodes, expected_correct, expected_total
        )
        lower_value, _ = _item_expected_objective(
            lower, base_model, nodes, expected_correct, expected_total
        )
        numerical[index] = (upper_value - lower_value) / (2.0 * step)

    assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-7)
