"""Tests for continuous state-space IRT models."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mirt.models.state_space import (
    StateSpaceBatchPredictiveResult,
    StateSpaceBatchStepResult,
    StateSpaceIRT,
    StateSpacePredictiveResult,
    StateSpaceStepResult,
)


def test_dynamic_module_reexports_state_space_model():
    from mirt.models.dynamic import StateSpaceIRT as DynamicStateSpaceIRT

    assert DynamicStateSpaceIRT is StateSpaceIRT


def test_models_module_exports_state_space_results():
    from mirt.models import (
        StateSpaceBatchPredictiveResult as PublicBatchPredictiveResult,
    )
    from mirt.models import (
        StateSpaceBatchStepResult as PublicBatchStepResult,
    )
    from mirt.models import StateSpacePredictiveResult as PublicPredictiveResult
    from mirt.models import StateSpaceStepResult as PublicStepResult

    assert PublicStepResult is StateSpaceStepResult
    assert PublicBatchStepResult is StateSpaceBatchStepResult
    assert PublicPredictiveResult is StateSpacePredictiveResult
    assert PublicBatchPredictiveResult is StateSpaceBatchPredictiveResult


class TestStateSpaceIRT:
    @staticmethod
    def _reference_simulation(model, n_persons, seed):
        rng = np.random.default_rng(seed)
        transition = model.transition_matrix[0, 0]
        process_noise = model.process_noise[0, 0]
        theta = np.zeros((n_persons, model.n_timepoints))
        responses = np.zeros(
            (n_persons, model.n_timepoints, model.n_items),
            dtype=np.int32,
        )
        theta[:, 0] = rng.normal(
            model.initial_mean,
            np.sqrt(model.initial_var),
            n_persons,
        )
        for time in range(1, model.n_timepoints):
            theta[:, time] = transition * theta[:, time - 1] + rng.normal(
                0,
                np.sqrt(process_noise),
                n_persons,
            )
        for person in range(n_persons):
            for time in range(model.n_timepoints):
                logits = model.discrimination * (theta[person, time] - model.difficulty)
                probabilities = 1.0 / (1.0 + np.exp(-logits))
                if model.base_model == "3PL":
                    probabilities = (
                        model.guessing + (1.0 - model.guessing) * probabilities
                    )
                responses[person, time] = rng.random(model.n_items) < probabilities
        return responses, theta

    @staticmethod
    def _reference_filter(model, responses):
        filtered_means = np.empty(model.n_timepoints)
        filtered_variances = np.empty(model.n_timepoints)
        predicted_mean = model.initial_mean
        predicted_variance = model.initial_var
        transition = model.transition_matrix[0, 0]
        process_variance = model.process_noise[0, 0]

        for time_index in range(model.n_timepoints):
            observed = responses[time_index] >= 0
            if np.any(observed):
                candidate_mean = predicted_mean
                for _ in range(5):
                    logits = model.discrimination * (candidate_mean - model.difficulty)
                    base_probability = 1.0 / (1.0 + np.exp(-logits))
                    if model.base_model == "3PL":
                        guessing_scale = 1.0 - model.guessing
                        probability = model.guessing + guessing_scale * base_probability
                        derivative = (
                            model.discrimination
                            * guessing_scale
                            * base_probability
                            * (1.0 - base_probability)
                        )
                    else:
                        probability = base_probability
                        derivative = (
                            model.discrimination
                            * base_probability
                            * (1.0 - base_probability)
                        )
                    probability = np.clip(
                        probability,
                        1e-10,
                        1.0 - 1e-10,
                    )
                    response_variance = (
                        probability * (1.0 - probability) + model.observation_noise
                    )
                    score = (
                        np.sum(
                            derivative[observed]
                            * (responses[time_index, observed] - probability[observed])
                            / response_variance[observed]
                        )
                        - (candidate_mean - predicted_mean) / predicted_variance
                    )
                    information = 1.0 / predicted_variance + np.sum(
                        derivative[observed] ** 2 / response_variance[observed]
                    )
                    candidate_mean += score / information

                logits = model.discrimination * (candidate_mean - model.difficulty)
                base_probability = 1.0 / (1.0 + np.exp(-logits))
                if model.base_model == "3PL":
                    guessing_scale = 1.0 - model.guessing
                    probability = model.guessing + guessing_scale * base_probability
                    derivative = (
                        model.discrimination
                        * guessing_scale
                        * base_probability
                        * (1.0 - base_probability)
                    )
                else:
                    probability = base_probability
                    derivative = (
                        model.discrimination
                        * base_probability
                        * (1.0 - base_probability)
                    )
                probability = np.clip(probability, 1e-10, 1.0 - 1e-10)
                response_variance = (
                    probability * (1.0 - probability) + model.observation_noise
                )
                information = 1.0 / predicted_variance + np.sum(
                    derivative[observed] ** 2 / response_variance[observed]
                )
                updated_mean = candidate_mean
                updated_variance = 1.0 / information
            else:
                updated_mean = predicted_mean
                updated_variance = predicted_variance

            filtered_means[time_index] = updated_mean
            filtered_variances[time_index] = updated_variance
            if time_index < model.n_timepoints - 1:
                predicted_mean = transition * updated_mean
                predicted_variance = transition**2 * updated_variance + process_variance

        return filtered_means, filtered_variances

    @staticmethod
    def _reference_smoother(model, responses):
        filtered_means, filtered_variances = TestStateSpaceIRT._reference_filter(
            model, responses
        )
        smoothed_means = filtered_means.copy()
        smoothed_variances = filtered_variances.copy()
        transition = model.transition_matrix[0, 0]
        process_variance = model.process_noise[0, 0]

        for time_index in range(model.n_timepoints - 2, -1, -1):
            predicted_mean = transition * filtered_means[time_index]
            predicted_variance = (
                transition**2 * filtered_variances[time_index] + process_variance
            )
            smoothing_gain = (
                filtered_variances[time_index] * transition / predicted_variance
            )
            smoothed_means[time_index] = filtered_means[time_index] + (
                smoothing_gain * (smoothed_means[time_index + 1] - predicted_mean)
            )
            smoothed_variances[time_index] = max(
                filtered_variances[time_index]
                + smoothing_gain**2
                * (smoothed_variances[time_index + 1] - predicted_variance),
                0.0,
            )

        return smoothed_means, smoothed_variances

    @staticmethod
    def _reference_forecast(model, responses, n_steps):
        filtered_means, filtered_variances = TestStateSpaceIRT._reference_filter(
            model, responses
        )
        forecast_means = np.empty(n_steps)
        forecast_variances = np.empty(n_steps)
        current_mean = filtered_means[-1]
        current_variance = filtered_variances[-1]
        transition = model.transition_matrix[0, 0]
        process_variance = model.process_noise[0, 0]

        for step_index in range(n_steps):
            current_mean = transition * current_mean
            current_variance = transition**2 * current_variance + process_variance
            forecast_means[step_index] = current_mean
            forecast_variances[step_index] = current_variance

        return forecast_means, forecast_variances

    @staticmethod
    def _reference_predicted_state_moments(model, responses):
        filtered_means, filtered_variances = TestStateSpaceIRT._reference_filter(
            model, responses
        )
        predicted_means = np.empty(model.n_timepoints)
        predicted_variances = np.empty(model.n_timepoints)
        predicted_means[0] = model.initial_mean
        predicted_variances[0] = model.initial_var
        transition = model.transition_matrix[0, 0]
        process_variance = model.process_noise[0, 0]
        predicted_means[1:] = transition * filtered_means[:-1]
        predicted_variances[1:] = (
            transition**2 * filtered_variances[:-1] + process_variance
        )
        return predicted_means, predicted_variances

    @staticmethod
    def _reference_state_probabilities(
        model,
        state_mean,
        state_variance,
        n_quadpts=101,
    ):
        nodes, weights = np.polynomial.hermite.hermgauss(n_quadpts)
        nodes = nodes * np.sqrt(2.0)
        weights = weights / np.sqrt(np.pi)
        states = state_mean + np.sqrt(state_variance) * nodes
        logits = model.discrimination[None, :] * (
            states[:, None] - model.difficulty[None, :]
        )
        base_probability = 1.0 / (1.0 + np.exp(-logits))
        conditional = (
            model.guessing[None, :] + (1.0 - model.guessing[None, :]) * base_probability
            if model.base_model == "3PL"
            else base_probability
        )
        return weights @ conditional

    @staticmethod
    def _reference_state_log_score(
        model,
        responses,
        state_mean,
        state_variance,
        n_quadpts=101,
    ):
        observed = responses >= 0
        if not np.any(observed):
            return 0.0
        nodes, weights = np.polynomial.hermite.hermgauss(n_quadpts)
        nodes = nodes * np.sqrt(2.0)
        weights = weights / np.sqrt(np.pi)
        states = state_mean + np.sqrt(state_variance) * nodes
        logits = model.discrimination[None, :] * (
            states[:, None] - model.difficulty[None, :]
        )
        base_probability = 1.0 / (1.0 + np.exp(-logits))
        probability = (
            model.guessing[None, :] + (1.0 - model.guessing[None, :]) * base_probability
            if model.base_model == "3PL"
            else base_probability
        )
        probability = np.clip(probability, 1e-10, 1.0 - 1e-10)
        conditional_scores = np.sum(
            np.where(
                responses[observed][None, :] == 1,
                np.log(probability[:, observed]),
                np.log1p(-probability[:, observed]),
            ),
            axis=1,
        )
        log_integrand = np.log(weights) + conditional_scores
        maximum = np.max(log_integrand)
        return maximum + np.log(np.sum(np.exp(log_integrand - maximum)))

    @staticmethod
    def _reference_predictive_probabilities(model, responses, n_quadpts=101):
        predicted_means, predicted_variances = (
            TestStateSpaceIRT._reference_predicted_state_moments(model, responses)
        )
        return np.vstack(
            [
                TestStateSpaceIRT._reference_state_probabilities(
                    model,
                    predicted_mean,
                    predicted_variance,
                    n_quadpts,
                )
                for predicted_mean, predicted_variance in zip(
                    predicted_means,
                    predicted_variances,
                    strict=True,
                )
            ]
        )

    @staticmethod
    def _reference_predictive_log_scores(model, responses, n_quadpts=101):
        predicted_means, predicted_variances = (
            TestStateSpaceIRT._reference_predicted_state_moments(model, responses)
        )
        return np.array(
            [
                TestStateSpaceIRT._reference_state_log_score(
                    model,
                    time_responses,
                    predicted_mean,
                    predicted_variance,
                    n_quadpts,
                )
                for time_responses, predicted_mean, predicted_variance in zip(
                    responses,
                    predicted_means,
                    predicted_variances,
                    strict=True,
                )
            ]
        )

    def test_default_initialization(self):
        model = StateSpaceIRT(n_items=5, n_timepoints=4)
        assert model.n_items == 5
        assert model.n_timepoints == 4
        assert model.transition_matrix.shape == (1, 1)
        assert model.process_noise.shape == (1, 1)
        assert model.guessing is None

    def test_3pl_initialization(self):
        model = StateSpaceIRT(n_items=5, n_timepoints=4, base_model="3PL")
        assert model.guessing is not None
        assert model.guessing.shape == (5,)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"n_items": 0}, "n_items"),
            ({"n_items": True}, "n_items"),
            ({"n_timepoints": 0}, "n_timepoints"),
            ({"n_timepoints": 2.5}, "n_timepoints"),
            ({"base_model": "4PL"}, "base_model"),
            ({"transition_matrix": np.eye(2)}, "transition_matrix"),
            ({"transition_matrix": np.array([[np.nan]])}, "transition_matrix"),
            ({"process_noise": np.array([[-0.1]])}, "process_noise"),
            ({"process_noise": np.ones(2)}, "process_noise"),
            (
                {
                    "transition_matrix": np.zeros((1, 1)),
                    "process_noise": np.zeros((1, 1)),
                },
                "zero variance propagation",
            ),
            ({"discrimination": np.array([1.0, 1.0])}, "discrimination"),
            ({"discrimination": np.array([1.0, 0.0, 1.0])}, "discrimination"),
            ({"difficulty": np.array([0.0, np.inf, 0.0])}, "difficulty"),
            ({"guessing": np.full(3, 0.2)}, "guessing"),
            ({"observation_noise": -0.1}, "observation_noise"),
            ({"observation_noise": True}, "observation_noise"),
            ({"initial_mean": np.nan}, "initial_mean"),
            ({"initial_var": 0.0}, "initial_var"),
        ],
    )
    def test_initialization_validates_parameters(self, kwargs, message):
        options = {"n_items": 3, "n_timepoints": 2}
        options.update(kwargs)

        with pytest.raises(ValueError, match=message):
            StateSpaceIRT(**options)

    @pytest.mark.parametrize(
        "guessing",
        [np.array([0.2, 0.3]), np.array([0.2, -0.1, 0.3]), np.ones(3)],
    )
    def test_3pl_initialization_validates_guessing(self, guessing):
        with pytest.raises(ValueError, match="guessing"):
            StateSpaceIRT(
                n_items=3,
                n_timepoints=2,
                base_model="3PL",
                guessing=guessing,
            )

    def test_extended_kalman_filter(self):
        model = StateSpaceIRT(n_items=5, n_timepoints=4)
        rng = np.random.default_rng(42)
        responses = rng.integers(0, 2, (4, 5))
        means, vars = model.extended_kalman_filter(responses)
        assert means.shape == (4,)
        assert vars.shape == (4,)
        assert np.all(vars > 0)

    @pytest.mark.parametrize("base_model", ["2PL", "3PL"])
    @pytest.mark.parametrize("observation_noise", [0.0, 0.15])
    def test_batch_filter_matches_independent_scalar_reference(
        self,
        base_model,
        observation_noise,
    ):
        model = StateSpaceIRT(
            n_items=6,
            n_timepoints=5,
            base_model=base_model,
            transition_matrix=np.array([[0.85]]),
            process_noise=np.array([[0.07]]),
            observation_noise=observation_noise,
            discrimination=np.linspace(0.6, 1.7, 6),
            difficulty=np.linspace(-1.1, 1.2, 6),
            guessing=np.linspace(0.1, 0.25, 6) if base_model == "3PL" else None,
            initial_mean=-0.2,
            initial_var=0.8,
        )
        responses = np.random.default_rng(31).integers(0, 2, size=(9, 5, 6))
        responses[np.random.default_rng(32).random(responses.shape) < 0.2] = -1
        expected_means = np.empty((len(responses), model.n_timepoints))
        expected_variances = np.empty_like(expected_means)
        for person_index, person_responses in enumerate(responses):
            expected_means[person_index], expected_variances[person_index] = (
                self._reference_filter(model, person_responses)
            )

        means, variances = model.extended_kalman_filter_batch(responses)

        assert_allclose(means, expected_means, rtol=1e-12, atol=1e-12)
        assert_allclose(variances, expected_variances, rtol=1e-12, atol=1e-12)
        single_mean, single_variance = model.extended_kalman_filter(responses[0])
        assert_allclose(single_mean, means[0])
        assert_allclose(single_variance, variances[0])

    @pytest.mark.parametrize("base_model", ["2PL", "3PL"])
    @pytest.mark.parametrize("transition", [0.85, -0.65])
    def test_online_updates_match_full_history_filter(self, base_model, transition):
        model = StateSpaceIRT(
            n_items=7,
            n_timepoints=6,
            base_model=base_model,
            transition_matrix=np.array([[transition]]),
            process_noise=np.array([[0.09]]),
            observation_noise=0.12,
            discrimination=np.linspace(0.6, 1.8, 7),
            difficulty=np.linspace(-1.2, 1.3, 7),
            guessing=np.linspace(0.08, 0.24, 7) if base_model == "3PL" else None,
            initial_mean=-0.15,
            initial_var=0.75,
        )
        responses = np.random.default_rng(83).integers(0, 2, size=(11, 6, 7))
        responses[np.random.default_rng(84).random(responses.shape) < 0.25] = -1
        responses[0, 3] = -1
        expected_means, expected_variances = model.extended_kalman_filter_batch(
            responses
        )
        online_means = np.empty_like(expected_means)
        online_variances = np.empty_like(expected_variances)
        prior_means = np.full(len(responses), model.initial_mean)
        prior_variances = np.full(len(responses), model.initial_var)

        for time_index in range(model.n_timepoints):
            updated_means, updated_variances = model.extended_kalman_update_batch(
                responses[:, time_index],
                prior_means=prior_means,
                prior_variances=prior_variances,
            )
            online_means[:, time_index] = updated_means
            online_variances[:, time_index] = updated_variances
            if time_index < model.n_timepoints - 1:
                prior_means, prior_variances = model.propagate_state_batch(
                    updated_means,
                    updated_variances,
                )

        assert_allclose(online_means, expected_means)
        assert_allclose(online_variances, expected_variances)

        scalar_means = np.empty(model.n_timepoints)
        scalar_variances = np.empty(model.n_timepoints)
        prior_mean = model.initial_mean
        prior_variance = model.initial_var
        for time_index in range(model.n_timepoints):
            prior_mean, prior_variance = model.extended_kalman_update(
                responses[0, time_index],
                prior_mean=prior_mean,
                prior_variance=prior_variance,
            )
            scalar_means[time_index] = prior_mean
            scalar_variances[time_index] = prior_variance
            if time_index < model.n_timepoints - 1:
                prior_mean, prior_variance = model.propagate_state(
                    prior_mean,
                    prior_variance,
                )

        assert_allclose(scalar_means, expected_means[0])
        assert_allclose(scalar_variances, expected_variances[0])

    def test_online_batch_update_matches_scalar_updates_and_preserves_missing(self):
        model = StateSpaceIRT(
            n_items=5,
            n_timepoints=3,
            base_model="3PL",
            discrimination=np.linspace(0.7, 1.5, 5),
            difficulty=np.linspace(-0.8, 0.9, 5),
            guessing=np.linspace(0.1, 0.25, 5),
        )
        responses = np.array(
            [
                [1, 0, 1, -1, 0],
                [-1, -1, -1, -1, -1],
                [0, 1, 1, 0, 1],
            ],
            dtype=np.int32,
        )
        prior_means = np.array([-0.4, 0.2, 1.1])
        prior_variances = np.array([0.5, 0.8, 1.4])

        batch_means, batch_variances = model.extended_kalman_update_batch(
            responses,
            prior_means=prior_means,
            prior_variances=prior_variances,
        )
        scalar = [
            model.extended_kalman_update(
                person_responses,
                prior_mean=person_mean,
                prior_variance=person_variance,
            )
            for person_responses, person_mean, person_variance in zip(
                responses,
                prior_means,
                prior_variances,
                strict=True,
            )
        ]

        assert_allclose(batch_means, [result[0] for result in scalar])
        assert_allclose(batch_variances, [result[1] for result in scalar])
        assert batch_means[1] == prior_means[1]
        assert batch_variances[1] == prior_variances[1]

        default_mean, default_variance = model.extended_kalman_update(responses[0])
        expected_default = model.extended_kalman_update_batch(responses[:1])
        assert default_mean == expected_default[0][0]
        assert default_variance == expected_default[1][0]

    def test_state_propagation_matches_iterative_reference(self):
        model = StateSpaceIRT(
            n_items=4,
            n_timepoints=3,
            transition_matrix=np.array([[-0.7]]),
            process_noise=np.array([[0.08]]),
        )
        state_means = np.array([-1.0, 0.2, 2.0])
        state_variances = np.array([0.3, 1.0, 1.5])
        expected_means = state_means.copy()
        expected_variances = state_variances.copy()
        for _ in range(5):
            expected_means *= -0.7
            expected_variances = 0.49 * expected_variances + 0.08

        propagated_means, propagated_variances = model.propagate_state_batch(
            state_means,
            state_variances,
            n_steps=5,
        )

        assert_allclose(propagated_means, expected_means)
        assert_allclose(propagated_variances, expected_variances)
        shared_variance = model.propagate_state_batch(
            state_means,
            0.5,
        )[1]
        assert_allclose(shared_variance, np.full(3, 0.49 * 0.5 + 0.08))
        for person_index in range(len(state_means)):
            scalar_mean, scalar_variance = model.propagate_state(
                state_means[person_index],
                state_variances[person_index],
                n_steps=5,
            )
            assert_allclose(scalar_mean, propagated_means[person_index])
            assert_allclose(scalar_variance, propagated_variances[person_index])

    @pytest.mark.parametrize(
        ("method_name", "responses", "message"),
        [
            ("extended_kalman_update", np.zeros((1, 3), dtype=int), "shape"),
            ("extended_kalman_update", np.zeros(3), "integer"),
            ("extended_kalman_update", np.array([0, 1, 2]), "only"),
            ("extended_kalman_update_batch", np.zeros((0, 3), dtype=int), "shape"),
            ("extended_kalman_update_batch", np.zeros((2, 2), dtype=int), "shape"),
            ("extended_kalman_update_batch", np.zeros((2, 3)), "integer"),
            (
                "extended_kalman_update_batch",
                np.array([[0, 1, -2], [1, 0, 1]]),
                "only",
            ),
        ],
    )
    def test_online_updates_validate_responses(self, method_name, responses, message):
        model = StateSpaceIRT(n_items=3, n_timepoints=2)

        with pytest.raises(ValueError, match=message):
            getattr(model, method_name)(responses)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"prior_means": np.zeros(2)}, "prior_means"),
            ({"prior_means": np.array([0.0, np.nan, 0.0])}, "finite"),
            ({"prior_means": True}, "finite"),
            ({"prior_means": 1.0 + 0.5j}, "finite"),
            ({"prior_variances": np.ones(2)}, "prior_variances"),
            ({"prior_variances": 0.0}, "positive"),
            ({"prior_variances": np.array([1.0, np.inf, 1.0])}, "positive"),
        ],
    )
    def test_online_batch_update_validates_priors(self, kwargs, message):
        model = StateSpaceIRT(n_items=3, n_timepoints=2)
        responses = np.zeros((3, 3), dtype=np.int32)

        with pytest.raises(ValueError, match=message):
            model.extended_kalman_update_batch(responses, **kwargs)

    @pytest.mark.parametrize(
        ("method_name", "args", "kwargs", "message"),
        [
            ("propagate_state", (np.nan, 1.0), {}, "state_mean"),
            ("propagate_state", (0.0, -0.1), {}, "state_variance"),
            ("propagate_state", (0.0, 1.0), {"n_steps": 0}, "n_steps"),
            (
                "propagate_state_batch",
                (np.empty(0), np.empty(0)),
                {},
                "state_means",
            ),
            (
                "propagate_state_batch",
                (np.zeros((2, 1)), np.ones(2)),
                {},
                "state_means",
            ),
            (
                "propagate_state_batch",
                (np.zeros(3), np.ones(2)),
                {},
                "state_variances",
            ),
            (
                "propagate_state_batch",
                (np.zeros(3), np.array([1.0, -0.1, 1.0])),
                {},
                "nonnegative",
            ),
        ],
    )
    def test_state_propagation_validates_inputs(
        self,
        method_name,
        args,
        kwargs,
        message,
    ):
        model = StateSpaceIRT(n_items=3, n_timepoints=2)

        with pytest.raises(ValueError, match=message):
            getattr(model, method_name)(*args, **kwargs)

    @pytest.mark.parametrize("base_model", ["2PL", "3PL"])
    def test_state_response_predictions_match_high_resolution_reference(
        self,
        base_model,
    ):
        model = StateSpaceIRT(
            n_items=6,
            n_timepoints=4,
            base_model=base_model,
            discrimination=np.linspace(0.6, 1.7, 6),
            difficulty=np.linspace(-1.1, 1.2, 6),
            guessing=np.linspace(0.1, 0.25, 6) if base_model == "3PL" else None,
        )
        responses = np.array(
            [
                [1, 0, 1, -1, 0, 1],
                [0, 1, -1, 1, 1, 0],
                [-1, -1, -1, -1, -1, -1],
            ],
            dtype=np.int32,
        )
        state_means = np.array([-1.2, 0.4, 1.5])
        state_variances = np.array([0.0, 0.75, 1.3])
        expected_probabilities = np.vstack(
            [
                self._reference_state_probabilities(model, mean, variance)
                for mean, variance in zip(
                    state_means,
                    state_variances,
                    strict=True,
                )
            ]
        )
        expected_scores = np.array(
            [
                self._reference_state_log_score(
                    model,
                    person_responses,
                    mean,
                    variance,
                )
                for person_responses, mean, variance in zip(
                    responses,
                    state_means,
                    state_variances,
                    strict=True,
                )
            ]
        )

        probabilities = model.state_response_probabilities_batch(
            state_means,
            state_variances,
            n_quadpts=41,
        )
        scores = model.state_response_log_likelihood_batch(
            responses,
            state_means,
            state_variances,
            n_quadpts=41,
        )

        assert_allclose(probabilities, expected_probabilities, rtol=5e-9, atol=1e-9)
        assert_allclose(scores, expected_scores, rtol=5e-9, atol=2e-8)
        assert scores[2] == 0.0
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
        for person_index in range(len(state_means)):
            scalar_probabilities = model.state_response_probabilities(
                state_means[person_index],
                state_variances[person_index],
                n_quadpts=41,
            )
            scalar_score = model.state_response_log_likelihood(
                responses[person_index],
                state_means[person_index],
                state_variances[person_index],
                n_quadpts=41,
            )
            assert_allclose(scalar_probabilities, probabilities[person_index])
            assert scalar_score == pytest.approx(scores[person_index])

    @pytest.mark.parametrize("base_model", ["2PL", "3PL"])
    def test_online_step_matches_individual_operations(self, base_model):
        model = StateSpaceIRT(
            n_items=6,
            n_timepoints=4,
            base_model=base_model,
            transition_matrix=np.array([[0.85]]),
            process_noise=np.array([[0.09]]),
            observation_noise=0.1,
            discrimination=np.linspace(0.6, 1.7, 6),
            difficulty=np.linspace(-1.1, 1.2, 6),
            guessing=np.linspace(0.1, 0.25, 6) if base_model == "3PL" else None,
        )
        responses = np.array(
            [
                [1, 0, 1, -1, 0, 1],
                [0, 1, -1, 1, 1, 0],
                [-1, -1, -1, -1, -1, -1],
                [1, 1, 0, 0, 1, 0],
            ],
            dtype=np.int32,
        )
        prior_means = np.array([-1.2, 0.4, 1.5, -0.3])
        prior_variances = np.array([0.4, 0.75, 1.3, 0.9])
        expected_probabilities = model.state_response_probabilities_batch(
            prior_means,
            prior_variances,
            n_quadpts=31,
        )
        expected_scores = model.state_response_log_likelihood_batch(
            responses,
            prior_means,
            prior_variances,
            n_quadpts=31,
        )
        expected_updated = model.extended_kalman_update_batch(
            responses,
            prior_means=prior_means,
            prior_variances=prior_variances,
        )
        expected_next = model.propagate_state_batch(*expected_updated)
        observed = responses >= 0
        expected_item_scores = np.where(
            responses == 1,
            np.log(expected_probabilities),
            np.where(responses == 0, np.log1p(-expected_probabilities), np.nan),
        )
        expected_residuals = np.where(
            observed,
            responses - expected_probabilities,
            np.nan,
        )
        expected_standardized = expected_residuals / np.sqrt(
            expected_probabilities * (1.0 - expected_probabilities)
        )

        result = model.online_step_batch(
            responses,
            prior_means=prior_means,
            prior_variances=prior_variances,
            n_quadpts=31,
        )

        assert isinstance(result, StateSpaceBatchStepResult)
        assert result.n_persons == len(responses)
        assert_allclose(result.response_probabilities, expected_probabilities)
        assert_allclose(result.response_log_likelihoods, expected_scores)
        assert_allclose(
            result.item_log_likelihoods,
            expected_item_scores,
            equal_nan=True,
        )
        assert_allclose(result.residuals, expected_residuals, equal_nan=True)
        assert_allclose(
            result.standardized_residuals,
            expected_standardized,
            equal_nan=True,
        )
        assert_allclose(result.updated_means, expected_updated[0])
        assert_allclose(result.updated_variances, expected_updated[1])
        assert_allclose(result.next_means, expected_next[0])
        assert_allclose(result.next_variances, expected_next[1])
        assert result.updated_means[2] == prior_means[2]
        assert result.updated_variances[2] == prior_variances[2]
        assert np.all(np.isnan(result.item_log_likelihoods[2]))
        assert np.all(np.isnan(result.residuals[2]))
        assert np.all(np.isnan(result.standardized_residuals[2]))

        for person_index in range(len(responses)):
            scalar = model.online_step(
                responses[person_index],
                prior_mean=prior_means[person_index],
                prior_variance=prior_variances[person_index],
                n_quadpts=31,
            )
            assert isinstance(scalar, StateSpaceStepResult)
            assert_allclose(
                scalar.response_probabilities,
                result.response_probabilities[person_index],
            )
            assert scalar.response_log_likelihood == pytest.approx(
                result.response_log_likelihoods[person_index]
            )
            assert_allclose(
                scalar.item_log_likelihoods,
                result.item_log_likelihoods[person_index],
                equal_nan=True,
            )
            assert_allclose(
                scalar.residuals,
                result.residuals[person_index],
                equal_nan=True,
            )
            assert_allclose(
                scalar.standardized_residuals,
                result.standardized_residuals[person_index],
                equal_nan=True,
            )
            assert scalar.updated_mean == pytest.approx(
                result.updated_means[person_index]
            )
            assert scalar.updated_variance == pytest.approx(
                result.updated_variances[person_index]
            )
            assert scalar.next_mean == pytest.approx(result.next_means[person_index])
            assert scalar.next_variance == pytest.approx(
                result.next_variances[person_index]
            )

        with pytest.raises(FrozenInstanceError):
            scalar.updated_mean = 0.0

    def test_streaming_predictions_and_scores_match_history_methods(self):
        model = StateSpaceIRT(
            n_items=8,
            n_timepoints=6,
            base_model="3PL",
            transition_matrix=np.array([[0.9]]),
            process_noise=np.array([[0.07]]),
            observation_noise=0.1,
            discrimination=np.linspace(0.7, 1.8, 8),
            difficulty=np.linspace(-1.3, 1.4, 8),
            guessing=np.linspace(0.08, 0.24, 8),
            initial_mean=-0.1,
            initial_var=0.9,
        )
        responses = np.random.default_rng(87).integers(0, 2, size=(13, 6, 8))
        responses[np.random.default_rng(88).random(responses.shape) < 0.2] = -1
        history_probabilities = model.predictive_response_probabilities_batch(
            responses,
            n_quadpts=31,
        )
        history_scores = model.predictive_log_likelihood_batch(
            responses,
            n_quadpts=31,
            pointwise=True,
        )
        history_residuals = model.predictive_residuals_batch(
            responses,
            n_quadpts=31,
        )
        history_standardized_residuals = model.predictive_residuals_batch(
            responses,
            n_quadpts=31,
            standardized=True,
        )
        history_means, history_variances = model.extended_kalman_filter_batch(responses)
        prior_means = np.full(len(responses), model.initial_mean)
        prior_variances = np.full(len(responses), model.initial_var)

        for time_index in range(model.n_timepoints):
            result = model.online_step_batch(
                responses[:, time_index],
                prior_means=prior_means,
                prior_variances=prior_variances,
                n_quadpts=31,
            )
            assert_allclose(
                result.response_probabilities,
                history_probabilities[:, time_index],
            )
            assert_allclose(
                result.response_log_likelihoods,
                history_scores[:, time_index],
            )
            assert_allclose(
                result.residuals,
                history_residuals[:, time_index],
                equal_nan=True,
            )
            assert_allclose(
                result.standardized_residuals,
                history_standardized_residuals[:, time_index],
                equal_nan=True,
            )
            time_responses = responses[:, time_index]
            expected_item_scores = np.where(
                time_responses == 1,
                np.log(result.response_probabilities),
                np.where(
                    time_responses == 0,
                    np.log1p(-result.response_probabilities),
                    np.nan,
                ),
            )
            assert_allclose(
                result.item_log_likelihoods,
                expected_item_scores,
                equal_nan=True,
            )
            assert_allclose(result.updated_means, history_means[:, time_index])
            assert_allclose(result.updated_variances, history_variances[:, time_index])
            prior_means = result.next_means
            prior_variances = result.next_variances

    def test_online_step_defaults_to_initial_state(self):
        model = StateSpaceIRT(
            n_items=4,
            n_timepoints=3,
            initial_mean=-0.3,
            initial_var=0.8,
        )
        responses = np.array([1, 0, -1, 1], dtype=np.int32)

        result = model.online_step(responses)
        expected = model.online_step_batch(responses[None, :])

        assert_allclose(
            result.response_probabilities, expected.response_probabilities[0]
        )
        assert result.response_log_likelihood == pytest.approx(
            expected.response_log_likelihoods[0]
        )
        assert_allclose(
            result.item_log_likelihoods,
            expected.item_log_likelihoods[0],
            equal_nan=True,
        )
        assert_allclose(result.residuals, expected.residuals[0], equal_nan=True)
        assert_allclose(
            result.standardized_residuals,
            expected.standardized_residuals[0],
            equal_nan=True,
        )
        assert result.updated_mean == pytest.approx(expected.updated_means[0])
        assert result.updated_variance == pytest.approx(expected.updated_variances[0])
        assert result.next_mean == pytest.approx(expected.next_means[0])
        assert result.next_variance == pytest.approx(expected.next_variances[0])

    def test_online_item_diagnostics_remain_finite_for_extreme_items(self):
        model = StateSpaceIRT(
            n_items=4,
            n_timepoints=2,
            discrimination=np.array([50.0, 100.0, 150.0, 200.0]),
            difficulty=np.array([-100.0, -50.0, 50.0, 100.0]),
        )
        responses = np.array([0, 1, 0, 1], dtype=np.int32)

        result = model.online_step(responses, prior_mean=0.0, prior_variance=0.5)

        assert np.all(np.isfinite(result.item_log_likelihoods))
        assert np.all(np.isfinite(result.residuals))
        assert np.all(np.isfinite(result.standardized_residuals))

    @pytest.mark.parametrize("n_quadpts", [0, -1, True, 1.5])
    def test_online_steps_require_positive_quadrature_count(self, n_quadpts):
        model = StateSpaceIRT(n_items=3, n_timepoints=2)
        responses = np.zeros((2, 3), dtype=np.int32)

        with pytest.raises(ValueError, match="n_quadpts"):
            model.online_step_batch(responses, n_quadpts=n_quadpts)

    def test_online_steps_require_positive_prior_variance(self):
        model = StateSpaceIRT(n_items=3, n_timepoints=2)
        responses = np.zeros(3, dtype=np.int32)

        with pytest.raises(ValueError, match="prior_variance"):
            model.online_step(responses, prior_variance=0.0)

    @pytest.mark.parametrize("n_quadpts", [0, -1, True, 1.5])
    @pytest.mark.parametrize(
        "method_name",
        [
            "state_response_probabilities_batch",
            "state_response_log_likelihood_batch",
        ],
    )
    def test_state_response_predictions_require_positive_quadrature_count(
        self,
        n_quadpts,
        method_name,
    ):
        model = StateSpaceIRT(n_items=3, n_timepoints=2)
        state_means = np.zeros(2)
        state_variances = np.ones(2)
        args = (
            (np.zeros((2, 3), dtype=np.int32), state_means, state_variances)
            if method_name.endswith("log_likelihood_batch")
            else (state_means, state_variances)
        )

        with pytest.raises(ValueError, match="n_quadpts"):
            getattr(model, method_name)(*args, n_quadpts=n_quadpts)

    def test_state_response_log_scores_require_matching_batch_sizes(self):
        model = StateSpaceIRT(n_items=3, n_timepoints=2)
        responses = np.zeros((3, 3), dtype=np.int32)

        with pytest.raises(ValueError, match="same number"):
            model.state_response_log_likelihood_batch(
                responses,
                np.zeros(2),
                np.ones(2),
            )

    def test_deterministic_state_propagation_preserves_zero_variance(self):
        model = StateSpaceIRT(
            n_items=3,
            n_timepoints=2,
            transition_matrix=np.array([[0.5]]),
            process_noise=np.zeros((1, 1)),
        )

        mean, variance = model.propagate_state(2.0, 0.0, n_steps=3)

        assert mean == pytest.approx(0.25)
        assert variance == 0.0

    @pytest.mark.parametrize("base_model", ["2PL", "3PL"])
    @pytest.mark.parametrize("transition", [0.85, -0.65])
    def test_batch_smoother_matches_independent_scalar_reference(
        self,
        base_model,
        transition,
    ):
        model = StateSpaceIRT(
            n_items=6,
            n_timepoints=5,
            base_model=base_model,
            transition_matrix=np.array([[transition]]),
            process_noise=np.array([[0.07]]),
            observation_noise=0.12,
            discrimination=np.linspace(0.6, 1.7, 6),
            difficulty=np.linspace(-1.1, 1.2, 6),
            guessing=np.linspace(0.1, 0.25, 6) if base_model == "3PL" else None,
            initial_mean=-0.2,
            initial_var=0.8,
        )
        responses = np.random.default_rng(41).integers(0, 2, size=(9, 5, 6))
        responses[np.random.default_rng(42).random(responses.shape) < 0.2] = -1
        expected_means = np.empty((len(responses), model.n_timepoints))
        expected_variances = np.empty_like(expected_means)
        for person_index, person_responses in enumerate(responses):
            expected_means[person_index], expected_variances[person_index] = (
                self._reference_smoother(model, person_responses)
            )

        means, variances = model.extended_kalman_smoother_batch(responses)

        assert_allclose(means, expected_means, rtol=1e-12, atol=1e-12)
        assert_allclose(variances, expected_variances, rtol=1e-12, atol=1e-12)
        single_mean, single_variance = model.extended_kalman_smoother(responses[0])
        assert_allclose(single_mean, means[0])
        assert_allclose(single_variance, variances[0])

    def test_smoother_uses_future_evidence_and_reduces_uncertainty(self):
        model = StateSpaceIRT(
            n_items=8,
            n_timepoints=4,
            transition_matrix=np.array([[0.9]]),
            process_noise=np.array([[0.05]]),
        )
        responses = np.ones((3, 4, 8), dtype=np.int32)
        responses[:, 0] = -1

        filtered_means, filtered_variances = model.extended_kalman_filter_batch(
            responses
        )
        smoothed_means, smoothed_variances = model.extended_kalman_smoother_batch(
            responses
        )

        assert np.all(smoothed_means[:, 0] > filtered_means[:, 0])
        assert np.all(smoothed_variances[:, :-1] < filtered_variances[:, :-1])
        assert_allclose(smoothed_means[:, -1], filtered_means[:, -1])
        assert_allclose(smoothed_variances[:, -1], filtered_variances[:, -1])
        assert np.all(smoothed_variances >= 0.0)

    def test_smoother_preserves_all_missing_state_marginals(self):
        model = StateSpaceIRT(
            n_items=4,
            n_timepoints=4,
            transition_matrix=np.array([[0.8]]),
            process_noise=np.array([[0.2]]),
            initial_mean=1.0,
            initial_var=0.5,
        )
        responses = np.full((3, 4, 4), -1, dtype=np.int32)

        filtered = model.extended_kalman_filter_batch(responses)
        smoothed = model.extended_kalman_smoother_batch(responses)

        assert_allclose(smoothed[0], filtered[0])
        assert_allclose(smoothed[1], filtered[1])

    def test_smoother_improves_simulated_state_recovery(self):
        model = StateSpaceIRT(
            n_items=20,
            n_timepoints=12,
            base_model="3PL",
            transition_matrix=np.array([[0.95]]),
            process_noise=np.array([[0.08]]),
            discrimination=np.linspace(0.7, 1.8, 20),
            difficulty=np.linspace(-2.0, 2.0, 20),
            guessing=np.linspace(0.1, 0.25, 20),
        )
        responses, true_states = model.simulate(500, seed=42)

        filtered_means, _ = model.extended_kalman_filter_batch(responses)
        smoothed_means, _ = model.extended_kalman_smoother_batch(responses)
        filtered_rmse = np.sqrt(np.mean((filtered_means - true_states) ** 2))
        smoothed_rmse = np.sqrt(np.mean((smoothed_means - true_states) ** 2))

        assert smoothed_rmse < 0.9 * filtered_rmse

    @pytest.mark.parametrize("transition", [0.9, -0.65])
    def test_batch_forecast_matches_independent_scalar_reference(self, transition):
        model = StateSpaceIRT(
            n_items=6,
            n_timepoints=5,
            base_model="3PL",
            transition_matrix=np.array([[transition]]),
            process_noise=np.array([[0.07]]),
            observation_noise=0.12,
            discrimination=np.linspace(0.6, 1.7, 6),
            difficulty=np.linspace(-1.1, 1.2, 6),
            guessing=np.linspace(0.1, 0.25, 6),
            initial_mean=-0.2,
            initial_var=0.8,
        )
        responses = np.random.default_rng(51).integers(0, 2, size=(9, 5, 6))
        responses[np.random.default_rng(52).random(responses.shape) < 0.2] = -1
        expected_means = np.empty((len(responses), 4))
        expected_variances = np.empty_like(expected_means)
        for person_index, person_responses in enumerate(responses):
            expected_means[person_index], expected_variances[person_index] = (
                self._reference_forecast(model, person_responses, 4)
            )

        means, variances = model.forecast_batch(responses, 4)

        assert_allclose(means, expected_means, rtol=1e-12, atol=1e-12)
        assert_allclose(variances, expected_variances, rtol=1e-12, atol=1e-12)
        single_mean, single_variance = model.forecast(responses[0], 4)
        assert_allclose(single_mean, means[0])
        assert_allclose(single_variance, variances[0])

    @pytest.mark.parametrize("base_model", ["2PL", "3PL"])
    @pytest.mark.parametrize("transition", [0.9, -0.65])
    def test_state_forecasts_match_history_forecasts(self, base_model, transition):
        model = StateSpaceIRT(
            n_items=6,
            n_timepoints=5,
            base_model=base_model,
            transition_matrix=np.array([[transition]]),
            process_noise=np.array([[0.07]]),
            observation_noise=0.12,
            discrimination=np.linspace(0.6, 1.7, 6),
            difficulty=np.linspace(-1.1, 1.2, 6),
            guessing=np.linspace(0.1, 0.25, 6) if base_model == "3PL" else None,
            initial_mean=-0.2,
            initial_var=0.8,
        )
        responses = np.random.default_rng(53).integers(0, 2, size=(9, 5, 6))
        responses[np.random.default_rng(54).random(responses.shape) < 0.2] = -1
        filtered_means, filtered_variances = model.extended_kalman_filter_batch(
            responses
        )
        history_forecast = model.forecast_batch(responses, 4)
        history_probabilities = model.forecast_response_probabilities_batch(
            responses,
            4,
            n_quadpts=31,
        )

        state_forecast = model.forecast_from_state_batch(
            filtered_means[:, -1],
            filtered_variances[:, -1],
            4,
        )
        state_probabilities = model.forecast_response_probabilities_from_state_batch(
            filtered_means[:, -1],
            filtered_variances[:, -1],
            4,
            n_quadpts=31,
        )

        assert_allclose(state_forecast[0], history_forecast[0])
        assert_allclose(state_forecast[1], history_forecast[1])
        assert_allclose(state_probabilities, history_probabilities)
        scalar_forecast = model.forecast_from_state(
            filtered_means[0, -1],
            filtered_variances[0, -1],
            4,
        )
        scalar_probabilities = model.forecast_response_probabilities_from_state(
            filtered_means[0, -1],
            filtered_variances[0, -1],
            4,
            n_quadpts=31,
        )
        assert_allclose(scalar_forecast[0], state_forecast[0][0])
        assert_allclose(scalar_forecast[1], state_forecast[1][0])
        assert_allclose(scalar_probabilities, state_probabilities[0])

    def test_state_forecast_continues_a_stream_without_history_replay(self):
        model = StateSpaceIRT(
            n_items=8,
            n_timepoints=6,
            base_model="3PL",
            transition_matrix=np.array([[0.92]]),
            process_noise=np.array([[0.06]]),
            discrimination=np.linspace(0.7, 1.8, 8),
            difficulty=np.linspace(-1.3, 1.3, 8),
            guessing=np.linspace(0.1, 0.24, 8),
        )
        responses, _ = model.simulate(1, seed=55)
        prior_mean = model.initial_mean
        prior_variance = model.initial_var

        for time_responses in responses[0]:
            step = model.online_step(
                time_responses,
                prior_mean=prior_mean,
                prior_variance=prior_variance,
            )
            prior_mean = step.next_mean
            prior_variance = step.next_variance

        expected_states = model.forecast(responses[0], 5)
        expected_probabilities = model.forecast_response_probabilities(
            responses[0],
            5,
        )
        forecast_states = model.forecast_from_state(
            step.updated_mean,
            step.updated_variance,
            5,
        )
        forecast_probabilities = model.forecast_response_probabilities_from_state(
            step.updated_mean,
            step.updated_variance,
            5,
        )

        assert_allclose(forecast_states[0], expected_states[0])
        assert_allclose(forecast_states[1], expected_states[1])
        assert_allclose(forecast_probabilities, expected_probabilities)

    @pytest.mark.parametrize("base_model", ["2PL", "3PL"])
    def test_response_forecast_integrates_symmetric_state_distribution(
        self,
        base_model,
    ):
        guessing = np.array([0.1, 0.2, 0.3])
        model = StateSpaceIRT(
            n_items=3,
            n_timepoints=2,
            base_model=base_model,
            transition_matrix=np.zeros((1, 1)),
            process_noise=np.array([[0.4]]),
            discrimination=np.array([0.7, 1.2, 2.0]),
            difficulty=np.zeros(3),
            guessing=guessing if base_model == "3PL" else None,
        )
        responses = np.zeros((4, 2, 3), dtype=np.int32)
        expected = (
            guessing + (1.0 - guessing) * 0.5
            if base_model == "3PL"
            else np.full(3, 0.5)
        )

        probabilities = model.forecast_response_probabilities_batch(
            responses,
            3,
            n_quadpts=31,
        )

        assert probabilities.shape == (4, 3, 3)
        assert_allclose(
            probabilities,
            np.broadcast_to(expected, probabilities.shape),
            atol=1e-14,
        )
        single = model.forecast_response_probabilities(
            responses[0],
            3,
            n_quadpts=31,
        )
        assert_allclose(single, probabilities[0])

    def test_response_forecast_matches_high_resolution_direct_quadrature(self):
        model = StateSpaceIRT(
            n_items=4,
            n_timepoints=3,
            base_model="3PL",
            transition_matrix=np.array([[0.8]]),
            process_noise=np.array([[0.15]]),
            discrimination=np.array([0.6, 1.0, 1.5, 2.0]),
            difficulty=np.array([-1.0, -0.2, 0.5, 1.3]),
            guessing=np.array([0.1, 0.15, 0.2, 0.25]),
        )
        responses = np.random.default_rng(61).integers(0, 2, size=(2, 3, 4))
        forecast_means, forecast_variances = model.forecast_batch(responses, 2)
        nodes, weights = np.polynomial.hermite.hermgauss(101)
        nodes = nodes * np.sqrt(2.0)
        weights = weights / np.sqrt(np.pi)
        expected = np.zeros((2, 2, 4))
        for node, weight in zip(nodes, weights, strict=True):
            states = forecast_means + np.sqrt(forecast_variances) * node
            logits = model.discrimination[None, None, :] * (
                states[:, :, None] - model.difficulty[None, None, :]
            )
            base_probability = 1.0 / (1.0 + np.exp(-logits))
            expected += weight * (
                model.guessing[None, None, :]
                + (1.0 - model.guessing[None, None, :]) * base_probability
            )

        probabilities = model.forecast_response_probabilities_batch(
            responses,
            2,
            n_quadpts=31,
        )

        assert_allclose(probabilities, expected, rtol=2e-11, atol=1e-11)
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))

    @pytest.mark.parametrize("n_steps", [0, -1, True, 1.5])
    @pytest.mark.parametrize(
        "method_name",
        ["forecast_batch", "forecast_response_probabilities_batch"],
    )
    def test_forecasts_require_positive_step_count(self, n_steps, method_name):
        model = StateSpaceIRT(n_items=3, n_timepoints=2)
        responses = np.zeros((2, 2, 3), dtype=np.int32)

        with pytest.raises(ValueError, match="n_steps"):
            getattr(model, method_name)(responses, n_steps)

    @pytest.mark.parametrize("n_steps", [0, -1, True, 1.5])
    @pytest.mark.parametrize(
        ("method_name", "state_means", "state_variances"),
        [
            ("forecast_from_state", 0.0, 1.0),
            ("forecast_from_state_batch", np.zeros(2), np.ones(2)),
            ("forecast_response_probabilities_from_state", 0.0, 1.0),
            (
                "forecast_response_probabilities_from_state_batch",
                np.zeros(2),
                np.ones(2),
            ),
        ],
    )
    def test_state_forecasts_require_positive_step_count(
        self,
        n_steps,
        method_name,
        state_means,
        state_variances,
    ):
        model = StateSpaceIRT(n_items=3, n_timepoints=2)

        with pytest.raises(ValueError, match="n_steps"):
            getattr(model, method_name)(state_means, state_variances, n_steps)

    @pytest.mark.parametrize("n_quadpts", [0, -1, True, 1.5])
    def test_response_forecast_requires_positive_quadrature_count(self, n_quadpts):
        model = StateSpaceIRT(n_items=3, n_timepoints=2)
        responses = np.zeros((2, 2, 3), dtype=np.int32)

        with pytest.raises(ValueError, match="n_quadpts"):
            model.forecast_response_probabilities_batch(
                responses,
                2,
                n_quadpts=n_quadpts,
            )

    @pytest.mark.parametrize("n_quadpts", [0, -1, True, 1.5])
    @pytest.mark.parametrize(
        ("method_name", "state_means", "state_variances"),
        [
            ("forecast_response_probabilities_from_state", 0.0, 1.0),
            (
                "forecast_response_probabilities_from_state_batch",
                np.zeros(2),
                np.ones(2),
            ),
        ],
    )
    def test_state_response_forecasts_require_positive_quadrature_count(
        self,
        n_quadpts,
        method_name,
        state_means,
        state_variances,
    ):
        model = StateSpaceIRT(n_items=3, n_timepoints=2)

        with pytest.raises(ValueError, match="n_quadpts"):
            getattr(model, method_name)(
                state_means,
                state_variances,
                2,
                n_quadpts=n_quadpts,
            )

    def test_state_forecasts_validate_current_moments(self):
        model = StateSpaceIRT(n_items=3, n_timepoints=2)

        with pytest.raises(ValueError, match="state_means"):
            model.forecast_from_state_batch(np.zeros((2, 1)), np.ones(2), 2)
        with pytest.raises(ValueError, match="state_variances"):
            model.forecast_from_state_batch(np.zeros(2), np.array([1.0, -0.1]), 2)

    def test_predictive_diagnostics_match_individual_history_operations(self):
        model = StateSpaceIRT(
            n_items=6,
            n_timepoints=5,
            base_model="3PL",
            transition_matrix=np.array([[0.88]]),
            process_noise=np.array([[0.07]]),
            observation_noise=0.1,
            discrimination=np.linspace(0.7, 1.8, 6),
            difficulty=np.linspace(-1.2, 1.1, 6),
            guessing=np.linspace(0.08, 0.24, 6),
            initial_mean=-0.15,
            initial_var=0.9,
        )
        responses = np.random.default_rng(64).integers(0, 2, size=(7, 5, 6))
        responses[np.random.default_rng(65).random(responses.shape) < 0.2] = -1

        result = model.predictive_diagnostics_batch(responses, n_quadpts=31)

        assert isinstance(result, StateSpaceBatchPredictiveResult)
        assert result.n_persons == len(responses)
        predicted = tuple(
            np.vstack(
                [
                    self._reference_predicted_state_moments(
                        model,
                        person_responses,
                    )[moment_index]
                    for person_responses in responses
                ]
            )
            for moment_index in range(2)
        )
        filtered = model.extended_kalman_filter_batch(responses)
        probabilities = model.predictive_response_probabilities_batch(
            responses,
            n_quadpts=31,
        )
        scores = model.predictive_log_likelihood_batch(
            responses,
            n_quadpts=31,
            pointwise=True,
        )
        residuals = model.predictive_residuals_batch(responses, n_quadpts=31)
        standardized = model.predictive_residuals_batch(
            responses,
            n_quadpts=31,
            standardized=True,
        )
        expected_item_scores = np.where(
            responses == 1,
            np.log(probabilities),
            np.where(responses == 0, np.log1p(-probabilities), np.nan),
        )
        assert_allclose(result.predicted_means, predicted[0])
        assert_allclose(result.predicted_variances, predicted[1])
        assert_allclose(result.filtered_means, filtered[0])
        assert_allclose(result.filtered_variances, filtered[1])
        assert_allclose(result.response_probabilities, probabilities)
        assert_allclose(result.response_log_likelihoods, scores)
        assert_allclose(
            result.item_log_likelihoods,
            expected_item_scores,
            equal_nan=True,
        )
        assert_allclose(result.residuals, residuals, equal_nan=True)
        assert_allclose(
            result.standardized_residuals,
            standardized,
            equal_nan=True,
        )
        assert_allclose(result.total_log_likelihoods, np.sum(scores, axis=1))

        scalar = model.predictive_diagnostics(responses[0], n_quadpts=31)
        assert isinstance(scalar, StateSpacePredictiveResult)
        for field_name in (
            "predicted_means",
            "predicted_variances",
            "filtered_means",
            "filtered_variances",
            "response_probabilities",
            "response_log_likelihoods",
            "item_log_likelihoods",
            "residuals",
            "standardized_residuals",
        ):
            assert_allclose(
                getattr(scalar, field_name),
                getattr(result, field_name)[0],
                equal_nan=True,
            )
        assert scalar.total_log_likelihood == pytest.approx(
            result.total_log_likelihoods[0]
        )
        with pytest.raises(FrozenInstanceError):
            scalar.predicted_means = np.zeros(model.n_timepoints)

    def test_predictive_diagnostics_treat_missing_histories_consistently(self):
        model = StateSpaceIRT(
            n_items=4,
            n_timepoints=3,
            transition_matrix=np.array([[0.8]]),
            process_noise=np.array([[0.12]]),
            initial_mean=0.5,
            initial_var=0.7,
        )
        responses = np.full((5, 3, 4), -1, dtype=np.int32)

        result = model.predictive_diagnostics_batch(responses)

        assert_allclose(result.filtered_means, result.predicted_means)
        assert_allclose(result.filtered_variances, result.predicted_variances)
        assert_allclose(result.response_log_likelihoods, 0.0)
        assert_allclose(result.total_log_likelihoods, 0.0)
        assert np.all(np.isnan(result.item_log_likelihoods))
        assert np.all(np.isnan(result.residuals))
        assert np.all(np.isnan(result.standardized_residuals))

    def test_predictive_diagnostics_are_causal(self):
        model = StateSpaceIRT(
            n_items=8,
            n_timepoints=5,
            transition_matrix=np.array([[0.9]]),
            process_noise=np.array([[0.05]]),
            difficulty=np.linspace(-1.0, 1.0, 8),
        )
        baseline = np.zeros((3, 5, 8), dtype=np.int32)
        changed = baseline.copy()
        changed[:, 2:] = 1

        baseline_result = model.predictive_diagnostics_batch(baseline)
        changed_result = model.predictive_diagnostics_batch(changed)

        assert_allclose(
            baseline_result.predicted_means[:, :3],
            changed_result.predicted_means[:, :3],
        )
        assert_allclose(
            baseline_result.predicted_variances[:, :3],
            changed_result.predicted_variances[:, :3],
        )
        assert_allclose(
            baseline_result.response_probabilities[:, :3],
            changed_result.response_probabilities[:, :3],
        )
        assert (
            np.max(
                np.abs(
                    baseline_result.response_probabilities[:, 3:]
                    - changed_result.response_probabilities[:, 3:]
                )
            )
            > 0.05
        )

    @pytest.mark.parametrize("n_quadpts", [0, -1, True, 1.5])
    @pytest.mark.parametrize(
        "method_name",
        ["predictive_diagnostics", "predictive_diagnostics_batch"],
    )
    def test_predictive_diagnostics_require_positive_quadrature_count_for_results(
        self,
        n_quadpts,
        method_name,
    ):
        model = StateSpaceIRT(n_items=3, n_timepoints=2)
        responses = np.zeros((2, 3), dtype=np.int32)
        if method_name.endswith("_batch"):
            responses = responses[None, :, :]

        with pytest.raises(ValueError, match="n_quadpts"):
            getattr(model, method_name)(responses, n_quadpts=n_quadpts)

    @pytest.mark.parametrize("base_model", ["2PL", "3PL"])
    @pytest.mark.parametrize("transition", [0.85, -0.6])
    def test_predictive_probabilities_match_high_resolution_scalar_reference(
        self,
        base_model,
        transition,
    ):
        model = StateSpaceIRT(
            n_items=6,
            n_timepoints=5,
            base_model=base_model,
            transition_matrix=np.array([[transition]]),
            process_noise=np.array([[0.08]]),
            observation_noise=0.1,
            discrimination=np.linspace(0.6, 1.7, 6),
            difficulty=np.linspace(-1.1, 1.2, 6),
            guessing=np.linspace(0.1, 0.25, 6) if base_model == "3PL" else None,
            initial_mean=-0.2,
            initial_var=0.8,
        )
        responses = np.random.default_rng(66).integers(0, 2, size=(7, 5, 6))
        responses[np.random.default_rng(67).random(responses.shape) < 0.2] = -1
        expected = np.stack(
            [
                self._reference_predictive_probabilities(model, person_responses)
                for person_responses in responses
            ]
        )

        probabilities = model.predictive_response_probabilities_batch(
            responses,
            n_quadpts=31,
        )

        assert_allclose(probabilities, expected, rtol=5e-9, atol=1e-9)
        single = model.predictive_response_probabilities(
            responses[0],
            n_quadpts=31,
        )
        assert_allclose(single, probabilities[0])
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))

    def test_predictive_probabilities_are_causal(self):
        model = StateSpaceIRT(
            n_items=8,
            n_timepoints=5,
            transition_matrix=np.array([[0.9]]),
            process_noise=np.array([[0.05]]),
            difficulty=np.linspace(-1.0, 1.0, 8),
        )
        baseline = np.zeros((3, 5, 8), dtype=np.int32)
        changed = baseline.copy()
        changed[:, 2:] = 1

        baseline_probabilities = model.predictive_response_probabilities_batch(baseline)
        changed_probabilities = model.predictive_response_probabilities_batch(changed)

        assert_allclose(
            baseline_probabilities[:, :3],
            changed_probabilities[:, :3],
        )
        assert (
            np.max(np.abs(baseline_probabilities[:, 3:] - changed_probabilities[:, 3:]))
            > 0.05
        )

    def test_predictive_residuals_match_raw_and_pearson_definitions(self):
        model = StateSpaceIRT(n_items=4, n_timepoints=3)
        responses = np.array(
            [
                [[1, 0, -1, 1], [0, 1, 1, -1], [1, 1, 0, 0]],
                [[0, -1, 1, 0], [1, 0, -1, 1], [-1, 0, 1, 1]],
            ],
            dtype=np.int32,
        )
        probabilities = model.predictive_response_probabilities_batch(responses)

        raw = model.predictive_residuals_batch(responses)
        pearson = model.predictive_residuals_batch(
            responses,
            standardized=True,
        )

        observed = responses >= 0
        assert np.array_equal(np.isnan(raw), ~observed)
        assert np.array_equal(np.isnan(pearson), ~observed)
        assert_allclose(raw[observed], (responses - probabilities)[observed])
        expected_pearson = raw / np.sqrt(probabilities * (1.0 - probabilities))
        assert_allclose(pearson[observed], expected_pearson[observed])
        single_raw = model.predictive_residuals(responses[0])
        assert_allclose(single_raw, raw[0], equal_nan=True)

    def test_predictive_residuals_are_calibrated_under_generating_model(self):
        model = StateSpaceIRT(
            n_items=12,
            n_timepoints=8,
            transition_matrix=np.array([[0.9]]),
            process_noise=np.array([[0.1]]),
            discrimination=np.linspace(0.7, 1.8, 12),
            difficulty=np.linspace(-1.5, 1.5, 12),
        )
        responses, _ = model.simulate(1_000, seed=68)

        residuals = model.predictive_residuals_batch(responses)

        assert abs(np.mean(residuals)) < 0.01

    @pytest.mark.parametrize("n_quadpts", [0, -1, True, 1.5])
    @pytest.mark.parametrize(
        "method_name",
        ["predictive_response_probabilities_batch", "predictive_residuals_batch"],
    )
    def test_predictive_diagnostics_require_positive_quadrature_count(
        self,
        n_quadpts,
        method_name,
    ):
        model = StateSpaceIRT(n_items=3, n_timepoints=2)
        responses = np.zeros((2, 2, 3), dtype=np.int32)

        with pytest.raises(ValueError, match="n_quadpts"):
            getattr(model, method_name)(responses, n_quadpts=n_quadpts)

    @pytest.mark.parametrize("standardized", [0, 1, "yes", None])
    def test_predictive_residuals_require_boolean_standardized(self, standardized):
        model = StateSpaceIRT(n_items=3, n_timepoints=2)
        responses = np.zeros((2, 2, 3), dtype=np.int32)

        with pytest.raises(ValueError, match="standardized"):
            model.predictive_residuals_batch(
                responses,
                standardized=standardized,
            )

    @pytest.mark.parametrize("base_model", ["2PL", "3PL"])
    @pytest.mark.parametrize("transition", [0.85, -0.6])
    def test_predictive_scores_match_high_resolution_scalar_reference(
        self,
        base_model,
        transition,
    ):
        model = StateSpaceIRT(
            n_items=6,
            n_timepoints=5,
            base_model=base_model,
            transition_matrix=np.array([[transition]]),
            process_noise=np.array([[0.08]]),
            observation_noise=0.1,
            discrimination=np.linspace(0.6, 1.7, 6),
            difficulty=np.linspace(-1.1, 1.2, 6),
            guessing=np.linspace(0.1, 0.25, 6) if base_model == "3PL" else None,
            initial_mean=-0.2,
            initial_var=0.8,
        )
        responses = np.random.default_rng(71).integers(0, 2, size=(7, 5, 6))
        responses[np.random.default_rng(72).random(responses.shape) < 0.2] = -1
        expected = np.vstack(
            [
                self._reference_predictive_log_scores(model, person_responses)
                for person_responses in responses
            ]
        )

        pointwise = model.predictive_log_likelihood_batch(
            responses,
            n_quadpts=31,
            pointwise=True,
        )
        totals = model.predictive_log_likelihood_batch(responses, n_quadpts=31)

        assert_allclose(pointwise, expected, rtol=5e-9, atol=2e-8)
        assert_allclose(totals, np.sum(pointwise, axis=1))
        single_pointwise = model.predictive_log_likelihood(
            responses[0],
            n_quadpts=31,
            pointwise=True,
        )
        single_total = model.predictive_log_likelihood(responses[0], n_quadpts=31)
        assert_allclose(single_pointwise, pointwise[0])
        assert single_total == pytest.approx(totals[0])

    def test_predictive_scores_treat_fully_missing_occasions_as_zero(self):
        model = StateSpaceIRT(n_items=4, n_timepoints=3)
        responses = np.full((5, 3, 4), -1, dtype=np.int32)

        pointwise = model.predictive_log_likelihood_batch(
            responses,
            pointwise=True,
        )
        totals = model.predictive_log_likelihood_batch(responses)

        assert_allclose(pointwise, 0.0)
        assert_allclose(totals, 0.0)

    def test_predictive_scores_remain_finite_for_extreme_items(self):
        model = StateSpaceIRT(
            n_items=4,
            n_timepoints=3,
            discrimination=np.array([50.0, 100.0, 150.0, 200.0]),
            difficulty=np.array([-100.0, -50.0, 50.0, 100.0]),
        )
        responses = np.array(
            [
                [[0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]],
                [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]],
            ],
            dtype=np.int32,
        )

        scores = model.predictive_log_likelihood_batch(
            responses,
            pointwise=True,
        )

        assert np.all(np.isfinite(scores))
        assert np.all(scores <= 0.0)

    def test_predictive_scores_favor_generating_configuration(self):
        true_model = StateSpaceIRT(
            n_items=12,
            n_timepoints=8,
            transition_matrix=np.array([[0.9]]),
            process_noise=np.array([[0.1]]),
            discrimination=np.linspace(0.7, 1.8, 12),
            difficulty=np.linspace(-1.5, 1.5, 12),
        )
        wrong_model = StateSpaceIRT(
            n_items=12,
            n_timepoints=8,
            transition_matrix=np.array([[0.9]]),
            process_noise=np.array([[0.1]]),
            discrimination=np.linspace(0.7, 1.8, 12),
            difficulty=np.linspace(-1.5, 1.5, 12) + 3.0,
        )
        responses, _ = true_model.simulate(500, seed=81)

        true_scores = true_model.predictive_log_likelihood_batch(responses)
        wrong_scores = wrong_model.predictive_log_likelihood_batch(responses)

        assert np.mean(true_scores) > np.mean(wrong_scores) + 5.0

    @pytest.mark.parametrize("n_quadpts", [0, -1, True, 1.5])
    def test_predictive_scores_require_positive_quadrature_count(self, n_quadpts):
        model = StateSpaceIRT(n_items=3, n_timepoints=2)
        responses = np.zeros((2, 2, 3), dtype=np.int32)

        with pytest.raises(ValueError, match="n_quadpts"):
            model.predictive_log_likelihood_batch(
                responses,
                n_quadpts=n_quadpts,
            )

    @pytest.mark.parametrize("pointwise", [0, 1, "yes", None])
    def test_predictive_scores_require_boolean_pointwise(self, pointwise):
        model = StateSpaceIRT(n_items=3, n_timepoints=2)
        responses = np.zeros((2, 2, 3), dtype=np.int32)

        with pytest.raises(ValueError, match="pointwise"):
            model.predictive_log_likelihood_batch(
                responses,
                pointwise=pointwise,
            )

    def test_3pl_observation_derivative_matches_finite_difference(self):
        model = StateSpaceIRT(
            n_items=4,
            n_timepoints=2,
            base_model="3PL",
            discrimination=np.array([0.6, 1.0, 1.4, 2.0]),
            difficulty=np.array([-1.0, -0.2, 0.4, 1.2]),
            guessing=np.array([0.1, 0.15, 0.2, 0.25]),
        )
        theta = np.array([-1.5, 0.0, 1.8])
        step = 1e-6

        _, derivative = model._observation_probability_and_derivative(theta)
        plus, _ = model._observation_probability_and_derivative(theta + step)
        minus, _ = model._observation_probability_and_derivative(theta - step)

        assert_allclose(derivative, (plus - minus) / (2.0 * step), rtol=1e-9)

    def test_observation_noise_reduces_update_information(self):
        responses = np.ones((1, 5), dtype=np.int32)
        precise = StateSpaceIRT(n_items=5, n_timepoints=1, observation_noise=0.0)
        noisy = StateSpaceIRT(n_items=5, n_timepoints=1, observation_noise=2.0)

        precise_mean, precise_variance = precise.extended_kalman_filter(responses)
        noisy_mean, noisy_variance = noisy.extended_kalman_filter(responses)

        assert 0.0 < noisy_mean[0] < precise_mean[0]
        assert precise_variance[0] < noisy_variance[0] < noisy.initial_var

    def test_all_missing_batch_propagates_state_prior(self):
        model = StateSpaceIRT(
            n_items=4,
            n_timepoints=4,
            transition_matrix=np.array([[0.8]]),
            process_noise=np.array([[0.2]]),
            initial_mean=1.0,
            initial_var=0.5,
        )
        responses = np.full((3, 4, 4), -1, dtype=np.int32)
        expected_means = np.empty(4)
        expected_variances = np.empty(4)
        expected_means[0] = model.initial_mean
        expected_variances[0] = model.initial_var
        for time_index in range(1, model.n_timepoints):
            expected_means[time_index] = 0.8 * expected_means[time_index - 1]
            expected_variances[time_index] = (
                0.8**2 * expected_variances[time_index - 1] + 0.2
            )

        means, variances = model.extended_kalman_filter_batch(responses)

        assert_allclose(means, np.broadcast_to(expected_means, means.shape))
        assert_allclose(
            variances,
            np.broadcast_to(expected_variances, variances.shape),
        )

    @pytest.mark.parametrize(
        "responses",
        [
            np.zeros((2, 3), dtype=np.int32),
            np.zeros((2, 3, 4), dtype=np.int32),
            np.zeros((2, 4), dtype=np.float64),
            np.full((2, 4), 2, dtype=np.int32),
        ],
    )
    @pytest.mark.parametrize(
        "method_name",
        ["extended_kalman_filter", "extended_kalman_smoother"],
    )
    def test_single_state_estimator_validates_responses(self, responses, method_name):
        model = StateSpaceIRT(n_items=4, n_timepoints=3)

        with pytest.raises(ValueError, match="responses"):
            getattr(model, method_name)(responses)

    @pytest.mark.parametrize(
        "responses",
        [
            np.zeros((3, 4), dtype=np.int32),
            np.empty((0, 3, 4), dtype=np.int32),
            np.zeros((2, 2, 4), dtype=np.int32),
            np.zeros((2, 3, 4), dtype=np.float64),
            np.full((2, 3, 4), 2, dtype=np.int32),
        ],
    )
    @pytest.mark.parametrize(
        "method_name",
        ["extended_kalman_filter_batch", "extended_kalman_smoother_batch"],
    )
    def test_batch_state_estimator_validates_responses(self, responses, method_name):
        model = StateSpaceIRT(n_items=4, n_timepoints=3)

        with pytest.raises(ValueError, match="responses"):
            getattr(model, method_name)(responses)

    def test_ekf_with_missing_data(self):
        model = StateSpaceIRT(n_items=5, n_timepoints=3)
        responses = np.array(
            [
                [1, 0, -1, 1, 0],
                [-1, -1, -1, -1, -1],
                [1, 1, 0, 1, 1],
            ]
        )
        means, vars = model.extended_kalman_filter(responses)
        assert means.shape == (3,)
        assert np.all(np.isfinite(means))

    def test_simulate(self):
        model = StateSpaceIRT(n_items=5, n_timepoints=6)
        n_persons = 20
        responses, theta = model.simulate(n_persons, seed=42)
        assert responses.shape == (n_persons, model.n_timepoints, model.n_items)
        assert theta.shape == (n_persons, model.n_timepoints)
        assert set(np.unique(responses)).issubset({0, 1})

    @pytest.mark.parametrize("base_model", ["2PL", "3PL"])
    def test_vectorized_simulation_preserves_seeded_draws(
        self,
        base_model,
        monkeypatch,
    ):
        """Chunked broadcasting preserves the prior seeded response stream."""
        from mirt.models import state_space as state_space_module

        model = StateSpaceIRT(
            n_items=7,
            n_timepoints=5,
            base_model=base_model,
            discrimination=np.linspace(0.6, 1.8, 7),
            difficulty=np.linspace(-1.0, 1.0, 7),
        )
        monkeypatch.setattr(
            state_space_module,
            "_STATE_SPACE_MAX_PROBABILITY_VALUES",
            17,
        )
        expected_responses, expected_theta = self._reference_simulation(model, 13, 42)

        responses, theta = model.simulate(13, seed=42)

        np.testing.assert_array_equal(responses, expected_responses)
        np.testing.assert_array_equal(theta, expected_theta)

    @pytest.mark.parametrize("n_persons", [0, -1, True, 1.5])
    def test_simulate_requires_positive_person_count(self, n_persons):
        model = StateSpaceIRT(n_items=3, n_timepoints=2)

        with pytest.raises(ValueError, match="n_persons"):
            model.simulate(n_persons)

    def test_simulate_theta_autocorrelation(self):
        model = StateSpaceIRT(
            n_items=5,
            n_timepoints=10,
            process_noise=np.array([[0.01]]),
        )
        _, theta = model.simulate(100, seed=42)
        corr = np.corrcoef(theta[:, :-1].ravel(), theta[:, 1:].ravel())[0, 1]
        assert corr > 0.5

    def test_summary(self):
        model = StateSpaceIRT(n_items=5, n_timepoints=4, observation_noise=0.2)
        summary = model.summary()
        assert "State-Space IRT" in summary
        assert "Observation Noise:  0.2000" in summary
