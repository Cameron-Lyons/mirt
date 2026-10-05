"""Tests for missing data imputation."""

from types import SimpleNamespace

import numpy as np
import pytest

import mirt
import mirt.utils.imputation as imputation_module
from mirt import analyze_missing, averageMI, impute_responses, listwise_deletion
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.utils.imputation import (
    LARGE_DF,
    MIResult,
    _draw_categorical,
    pairwise_available,
)


class TestImputeResponses:
    """Tests for response imputation."""

    def test_impute_mean(self, responses_with_missing):
        """Test mean imputation."""
        responses = responses_with_missing["responses"]

        imputed = impute_responses(responses, method="mean")

        assert np.all(imputed >= 0)

        assert imputed.shape == responses.shape

    def test_impute_mean_uses_rounded_item_values(self):
        responses = np.array([[0, -1], [1, 1], [2, 2], [3, 2]])

        imputed = impute_responses(responses, method="mean")

        np.testing.assert_array_equal(
            imputed,
            np.array([[0, 2], [1, 1], [2, 2], [3, 2]]),
        )

    def test_impute_median_supports_ordered_categories(self):
        responses = np.array(
            [
                [0, -1, 2],
                [1, 1, -1],
                [4, 2, 4],
                [4, 3, 4],
            ]
        )

        imputed = impute_responses(responses, method="median")

        np.testing.assert_array_equal(
            imputed,
            np.array(
                [
                    [0, 2, 2],
                    [1, 1, 4],
                    [4, 2, 4],
                    [4, 3, 4],
                ]
            ),
        )

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            ("median", np.array([[0, 3], [1000, 5], [1000, 5], [1000, 7]])),
            ("mode", np.array([[0, 3], [1000, 3], [1000, 5], [1000, 7]])),
        ],
    )
    def test_simple_imputation_supports_sparse_large_category_codes(
        self, method, expected
    ):
        responses = np.array([[0, 3], [1000, -1], [-1, 5], [1000, 7]])

        imputed = impute_responses(responses, method=method)

        np.testing.assert_array_equal(imputed, expected)

    def test_impute_mode(self, responses_with_missing):
        """Test mode imputation."""
        responses = responses_with_missing["responses"]

        imputed = impute_responses(responses, method="mode")

        assert np.all(imputed >= 0)

        assert set(imputed.flatten()).issubset({0, 1})

    def test_impute_random(self, responses_with_missing, rng):
        """Test random imputation."""
        responses = responses_with_missing["responses"]

        imputed = impute_responses(responses, method="random", seed=42)

        assert np.all(imputed >= 0)

        assert set(imputed.flatten()).issubset({0, 1})

    def test_impute_em(self, responses_with_missing):
        """Test EM imputation."""
        responses = responses_with_missing["responses"]

        imputed = impute_responses(responses, method="EM")

        assert np.all(imputed >= 0)

    def test_impute_multiple(self, responses_with_missing):
        """Test multiple imputation."""
        responses = responses_with_missing["responses"]

        imputed = impute_responses(
            responses,
            method="multiple",
            n_imputations=3,
            seed=42,
        )

        assert isinstance(imputed, list)
        assert len(imputed) == 3

        for imp in imputed:
            assert np.all(imp >= 0)

    def test_no_missing_data(self, dichotomous_responses):
        """Test imputation when no data is missing."""
        responses = dichotomous_responses["responses"]

        imputed = impute_responses(responses, method="mean")

        np.testing.assert_array_equal(imputed, responses)

    def test_complete_multiple_returns_independent_copies(self):
        responses = np.array([[0, 1], [1, 0]])

        imputations = impute_responses(
            responses,
            method="multiple",
            n_imputations=2,
        )

        assert len(imputations) == 2
        np.testing.assert_array_equal(imputations[0], responses)
        assert imputations[0] is not imputations[1]

    @pytest.mark.parametrize(
        "missing_code",
        [1.5, True, np.iinfo(np.int_).max + 1],
    )
    def test_missing_code_must_be_a_supported_integer(self, missing_code):
        with pytest.raises(MirtValidationError, match="missing_code"):
            impute_responses(
                np.array([[0, -1], [1, 1]]),
                method="mean",
                missing_code=missing_code,
            )

    def test_invalid_method_is_rejected_without_missing_data(self):
        responses = np.array([[0, 1], [1, 0]])

        with pytest.raises(MirtValidationError, match="Unknown imputation method"):
            impute_responses(responses, method="unknown")

    @pytest.mark.parametrize("n_imputations", [0, -1, 1.5, True])
    def test_multiple_requires_positive_integer_count(self, n_imputations):
        responses = np.array([[0, 1], [1, 0]])

        with pytest.raises(MirtValidationError, match="positive integer"):
            impute_responses(
                responses,
                method="multiple",
                n_imputations=n_imputations,
            )

    @pytest.mark.parametrize(
        "responses",
        [np.array([0, -1]), np.empty((0, 2)), np.empty((2, 0))],
    )
    def test_invalid_response_shapes_are_rejected(self, responses):
        with pytest.raises(MirtDataError):
            impute_responses(responses, method="mean")

    @pytest.mark.parametrize(
        "method", ["mean", "median", "mode", "random", "EM", "multiple"]
    )
    def test_fully_missing_item_is_rejected(self, method):
        responses = np.array([[-1, 0], [-1, 1]])

        with pytest.raises(MirtDataError, match="no observed responses"):
            impute_responses(responses, method=method, n_imputations=2)

    def test_custom_missing_code_is_imputed(self):
        responses = np.array([[0, 99], [1, 1]])

        imputed = impute_responses(
            responses,
            method="mean",
            missing_code=99,
        )

        np.testing.assert_array_equal(imputed, np.array([[0, 1], [1, 1]]))

    def test_imputation_does_not_mutate_source(self):
        responses = np.array([[0, -1], [1, 1]])
        original = responses.copy()

        impute_responses(responses, method="mode")

        np.testing.assert_array_equal(responses, original)

    def test_multiple_imputation_reuses_fit_and_batches_each_copy(self, monkeypatch):
        responses = np.array([[-1, 0], [1, -1], [0, 1]])
        calls = {"fit": 0, "posterior": 0, "probability": 0}

        class FakeModel:
            is_polytomous = False
            n_factors = 1

            def probability_pairs(self, theta, item_indices):
                calls["probability"] += 1
                return np.full(len(theta), 0.5)

        fake_model = FakeModel()

        def fake_fit(*args, **kwargs):
            calls["fit"] += 1
            np.testing.assert_array_equal(args[0], responses)
            assert kwargs["compute_standard_errors"] is False
            return SimpleNamespace(model=fake_model)

        def fake_posterior(model, observed, n_imputations, n_quadpts, rng):
            calls["posterior"] += 1
            np.testing.assert_array_equal(observed, responses[:2])
            return np.zeros((len(observed), 1, n_imputations))

        monkeypatch.setattr(mirt, "fit_mirt", fake_fit)
        monkeypatch.setattr(
            imputation_module, "_posterior_ability_draws", fake_posterior
        )

        imputations = impute_responses(
            responses,
            method="multiple",
            n_imputations=4,
            seed=42,
        )

        assert calls == {"fit": 1, "posterior": 1, "probability": 4}
        assert len(imputations) == 4
        assert all(np.all(imputation >= 0) for imputation in imputations)

    def test_multiple_imputation_supports_models_without_paired_api(self, monkeypatch):
        responses = np.zeros((20, 12), dtype=int)
        responses[0, 1] = -1
        responses[1, 9] = -1
        calls: list[int | None] = []

        class FakeModel:
            is_polytomous = False
            n_factors = 1

            def probability(self, theta, item=None):
                calls.append(item)
                if item is None:
                    return np.full((len(theta), responses.shape[1]), 0.5)
                return np.full(len(theta), 0.5)

        monkeypatch.setattr(
            mirt,
            "fit_mirt",
            lambda *args, **kwargs: SimpleNamespace(model=FakeModel()),
        )
        monkeypatch.setattr(
            imputation_module,
            "_posterior_ability_draws",
            lambda model, observed, n_imputations, n_quadpts, rng: np.zeros(
                (len(observed), 1, n_imputations)
            ),
        )

        imputations = impute_responses(
            responses,
            method="multiple",
            n_imputations=3,
            seed=42,
        )

        assert calls == [1, 9] * 3
        assert all(np.all(imputation >= 0) for imputation in imputations)

    def test_paired_model_batches_respect_probability_memory_limit(self, monkeypatch):
        responses = np.zeros((10, 4), dtype=int)
        rows, columns = np.indices(responses.shape)
        responses[(rows + columns) % 2 == 0] = -1
        batch_sizes: list[int] = []

        class FakeModel:
            is_polytomous = False
            n_factors = 1

            def probability_pairs(self, theta, item_indices):
                batch_sizes.append(len(theta))
                return np.full(len(theta), 0.5)

        monkeypatch.setattr(imputation_module, "_MODEL_DRAW_TARGET_ELEMENTS", 8)
        monkeypatch.setattr(
            mirt,
            "fit_mirt",
            lambda *args, **kwargs: SimpleNamespace(model=FakeModel()),
        )
        monkeypatch.setattr(
            imputation_module,
            "_posterior_ability_draws",
            lambda model, observed, n_imputations, n_quadpts, rng: np.zeros(
                (len(observed), 1, n_imputations)
            ),
        )

        impute_responses(
            responses,
            method="multiple",
            n_imputations=2,
            seed=42,
        )

        assert batch_sizes == [4] * 10

    def test_dense_polytomous_imputation_batches_all_items(self, monkeypatch):
        responses = np.array(
            [
                [-1, 0, -1],
                [1, -1, 2],
                [-1, 2, 0],
                [2, -1, -1],
            ]
        )
        calls = 0

        class FakeModel:
            is_polytomous = True
            n_categories = [3, 3, 3]
            n_factors = 1

            def probability_pairs(self, theta, item_indices):
                nonlocal calls
                calls += 1
                probabilities = np.zeros((len(theta), 3))
                probabilities[:, 2] = 1.0
                return probabilities

        monkeypatch.setattr(
            mirt,
            "fit_mirt",
            lambda *args, **kwargs: SimpleNamespace(model=FakeModel()),
        )
        monkeypatch.setattr(
            imputation_module,
            "_posterior_ability_draws",
            lambda model, observed, n_imputations, n_quadpts, rng: np.zeros(
                (len(observed), 1, n_imputations)
            ),
        )

        imputations = impute_responses(
            responses,
            method="multiple",
            model="GRM",
            n_imputations=3,
            seed=42,
        )

        assert calls == 3
        for imputed in imputations:
            np.testing.assert_array_equal(imputed[responses == -1], 2)

    def test_multiple_imputation_warns_and_falls_back_when_posterior_fails(
        self, monkeypatch
    ):
        responses = np.array([[-1, 0], [1, -1], [0, 1]])

        monkeypatch.setattr(
            mirt,
            "fit_mirt",
            lambda *args, **kwargs: SimpleNamespace(model=object()),
        )

        def fail_posterior(*args, **kwargs):
            raise RuntimeError("posterior failed")

        monkeypatch.setattr(
            imputation_module, "_posterior_ability_draws", fail_posterior
        )

        with pytest.warns(RuntimeWarning, match="empirical item distributions"):
            imputations = impute_responses(
                responses,
                method="multiple",
                n_imputations=3,
                seed=42,
            )

        assert len(imputations) == 3
        assert all(np.all(imputation >= 0) for imputation in imputations)

    def test_named_model_calibration_and_posterior_receive_only_observed_responses(
        self, monkeypatch
    ):
        from mirt.models import TwoParameterLogistic

        responses = np.array([[1, 99], [0, 1], [99, 0]])
        observed = np.array([[1, -1], [0, 1], [-1, 0]])
        model = TwoParameterLogistic(2)
        model._is_fitted = True
        fit_calls = []
        posterior_calls = []
        original_likelihood = model.log_likelihood_batch

        def calibrated(data, **kwargs):
            fit_calls.append(data.copy())
            return SimpleNamespace(model=model)

        def likelihood(data, points):
            posterior_calls.append(data.copy())
            return original_likelihood(data, points)

        monkeypatch.setattr(mirt, "fit_mirt", calibrated)
        monkeypatch.setattr(model, "log_likelihood_batch", likelihood)
        imputations = impute_responses(
            responses, method="multiple", n_imputations=3, missing_code=99, seed=11
        )

        assert len(fit_calls) == 1
        np.testing.assert_array_equal(fit_calls[0], observed)
        np.testing.assert_array_equal(np.concatenate(posterior_calls), observed[[0, 2]])
        for imputed in imputations:
            np.testing.assert_array_equal(
                imputed[responses != 99], responses[responses != 99]
            )
        np.testing.assert_array_equal(responses, [[1, 99], [0, 1], [99, 0]])

    def test_multiple_imputation_matches_independent_conditional_predictive_integrals(
        self,
    ):
        from scipy.integrate import quad
        from scipy.special import expit

        from mirt.models import TwoParameterLogistic

        slopes = np.array([1.4, 1.1, 1.6])
        locations = np.array([-0.5, 0.2, 0.8])
        model = TwoParameterLogistic(3)
        model.set_parameters(discrimination=slopes, difficulty=locations)
        model._is_fitted = True
        responses = np.tile([1, -1, -1], (2000, 1))
        responses.flags.writeable = False

        def posterior_kernel(point):
            return expit(slopes[0] * (point - locations[0])) * np.exp(-0.5 * point**2)

        def predictive(point, items):
            return np.prod(expit(slopes[items] * (point - locations[items])))

        mass = quad(posterior_kernel, -12.0, 12.0)[0]
        expected = [
            quad(
                lambda point: posterior_kernel(point) * predictive(point, items),
                -12.0,
                12.0,
            )[0]
            / mass
            for items in ([1], [2], [1, 2])
        ]
        draws = np.asarray(
            impute_responses(
                responses,
                method="multiple",
                model=model,
                n_imputations=20,
                n_quadpts=81,
                seed=1035,
            )
        )
        samples = draws[:, :, 1:].reshape(-1, 2)
        actual = [
            samples[:, 0].mean(),
            samples[:, 1].mean(),
            np.all(samples == 1, axis=1).mean(),
        ]

        np.testing.assert_allclose(actual, expected, atol=0.015)
        np.testing.assert_array_equal(draws[:, :, 0], 1)
        np.testing.assert_array_equal(responses, np.tile([1, -1, -1], (2000, 1)))

    def test_multiple_imputation_preserves_dependence_between_latent_factors(self):
        from numpy.polynomial.hermite import hermgauss
        from scipy.special import expit

        from mirt.models import TwoParameterLogistic

        slope = 3.0
        model = TwoParameterLogistic(3, n_factors=2)
        model.set_parameters(
            discrimination=np.array([[slope, slope], [slope, 0.0], [0.0, slope]]),
            difficulty=np.zeros(3),
        )
        model._is_fitted = True
        responses = np.tile([1, -1, -1], (2000, 1))
        nodes, weights = hermgauss(61)
        first, second = np.meshgrid(
            nodes * np.sqrt(2.0), nodes * np.sqrt(2.0), indexing="ij"
        )
        posterior = (
            weights[:, None] * weights[None, :] * expit(slope * (first + second))
        )
        posterior /= posterior.sum()
        first_probability = expit(slope * first)
        second_probability = expit(slope * second)
        expected_means = [
            (posterior * probability).sum()
            for probability in (first_probability, second_probability)
        ]
        expected_joint = (posterior * first_probability * second_probability).sum()
        expected_covariance = expected_joint - np.prod(expected_means)

        draws = np.asarray(
            impute_responses(
                responses,
                method="multiple",
                model=model,
                n_imputations=20,
                n_quadpts=61,
                seed=987,
            )
        )
        samples = draws[:, :, 1:].reshape(-1, 2)
        actual_joint = np.all(samples == 1, axis=1).mean()
        actual_covariance = actual_joint - np.prod(samples.mean(axis=0))

        np.testing.assert_allclose(samples.mean(axis=0), expected_means, atol=0.015)
        assert actual_joint == pytest.approx(expected_joint, abs=0.015)
        assert actual_covariance == pytest.approx(expected_covariance, abs=0.012)
        assert actual_covariance < -0.02

    def test_calibrated_model_and_fit_result_reuse_item_parameters(self, monkeypatch):
        from mirt.models import TwoParameterLogistic
        from mirt.results import FitResult

        model = TwoParameterLogistic(2)
        model._is_fitted = True
        result = FitResult(
            model=model,
            log_likelihood=0.0,
            n_iterations=0,
            converged=True,
            standard_errors={},
            aic=0.0,
            bic=0.0,
        )

        def unexpected_fit(*args, **kwargs):
            raise AssertionError("a supplied calibration must not be re-estimated")

        monkeypatch.setattr(mirt, "fit_mirt", unexpected_fit)
        responses = np.array([[0, -1], [1, -1], [-1, -1]])
        before = {name: values.copy() for name, values in model.parameters.items()}
        first = impute_responses(
            responses, method="multiple", model=model, n_imputations=3, seed=813
        )
        second = impute_responses(
            responses, method="multiple", model=result, n_imputations=3, seed=813
        )

        np.testing.assert_array_equal(first, second)
        for name, values in before.items():
            np.testing.assert_array_equal(model.parameters[name], values)
        assert np.isin(first, [0, 1]).all()

    def test_calibrated_ordinal_model_respects_item_categories(self):
        from mirt.models import GradedResponseModel

        model = GradedResponseModel(2, n_categories=[3, 4])
        model._is_fitted = True
        responses = np.array([[0, -1], [2, 3], [-1, -1]])

        draws = np.asarray(
            impute_responses(
                responses, method="multiple", model=model, n_imputations=10, seed=451
            )
        )

        assert np.all((draws >= 0) & (draws < np.array([3, 4])))
        np.testing.assert_array_equal(draws[:, 1], np.tile([2, 3], (10, 1)))

    def test_calibrated_model_supports_single_model_based_imputation(self):
        from mirt.models import TwoParameterLogistic

        model = TwoParameterLogistic(2)
        model._is_fitted = True
        responses = np.array([[0, -1], [1, -1], [-1, -1]])

        first = impute_responses(responses, method="EM", model=model, seed=185)
        second = impute_responses(responses, method="EM", model=model, seed=185)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[:2, 0], responses[:2, 0])
        assert np.isin(first, [0, 1]).all()

    @pytest.mark.parametrize("n_quadpts", [0, -1, 1.5, True])
    def test_multiple_imputation_validates_posterior_resolution(self, n_quadpts):
        with pytest.raises(MirtValidationError, match="n_quadpts"):
            impute_responses(
                np.array([[0, -1], [1, 1]]), method="multiple", n_quadpts=n_quadpts
            )

    def test_model_based_imputation_rejects_unfitted_or_incompatible_calibrations(self):
        from mirt.models import TwoParameterLogistic

        model = TwoParameterLogistic(2)
        with pytest.raises(MirtValidationError, match="fitted model"):
            impute_responses(
                np.array([[0, -1], [1, 1]]), method="multiple", model=model
            )
        model._is_fitted = True
        with pytest.raises(MirtDataError, match="number of response items"):
            impute_responses(
                np.array([[0, -1, 1], [1, 1, 1]]), method="multiple", model=model
            )
        with pytest.raises(MirtValidationError, match="Unknown imputation model"):
            impute_responses(
                np.array([[0, -1], [1, 1]]), method="multiple", model="unknown"
            )

    def test_calibrated_model_does_not_fall_back_when_posterior_fails(
        self, monkeypatch
    ):
        from mirt.models import TwoParameterLogistic

        model = TwoParameterLogistic(2)
        model._is_fitted = True

        def failed_posterior(*args, **kwargs):
            raise RuntimeError("posterior failed")

        monkeypatch.setattr(
            imputation_module, "_posterior_ability_draws", failed_posterior
        )
        with pytest.raises(RuntimeError, match="posterior failed"):
            impute_responses(
                np.array([[0, -1], [1, 1]]), method="multiple", model=model
            )

    def test_calibration_failure_returns_warned_empirical_draws(self, monkeypatch):
        responses = np.array([[-1, 1], [0, -1], [1, 0]])

        def failed_fit(*args, **kwargs):
            raise RuntimeError("calibration failed")

        monkeypatch.setattr(mirt, "fit_mirt", failed_fit)
        with pytest.warns(RuntimeWarning, match="calibration failed"):
            draws = np.asarray(
                impute_responses(responses, method="multiple", n_imputations=4, seed=21)
            )

        assert np.isin(draws, [0, 1]).all()
        for draw in draws:
            np.testing.assert_array_equal(
                draw[responses >= 0], responses[responses >= 0]
            )

    @pytest.mark.parametrize(
        "model_name", ["2PL", "calibrated_binary", "calibrated_ordinal"]
    )
    def test_model_based_imputation_validates_observed_categories(self, model_name):
        from mirt.models import GradedResponseModel, TwoParameterLogistic

        if model_name == "2PL":
            model = model_name
            responses = np.array([[2, -1], [0, 1]])
        elif model_name == "calibrated_binary":
            model = TwoParameterLogistic(2)
            model._is_fitted = True
            responses = np.array([[2, -1], [0, 1]])
        else:
            model = GradedResponseModel(2, n_categories=[3, 4])
            model._is_fitted = True
            responses = np.array([[3, -1], [0, 1]])

        with pytest.raises(MirtDataError):
            impute_responses(responses, method="multiple", model=model)


class TestAnalyzeMissing:
    """Tests for missing data analysis."""

    def test_analyze_missing(self, responses_with_missing):
        """Test missing data analysis."""
        responses = responses_with_missing["responses"]

        analysis = analyze_missing(responses)

        assert (
            "total_missing" in analysis
            or "total_missing_rate" in analysis
            or "n_missing" in analysis
        )

    def test_missing_by_item(self, responses_with_missing):
        """Test missing data by item."""
        responses = responses_with_missing["responses"]

        analysis = analyze_missing(responses)

        if "item_missing_rate" in analysis:
            n_items = responses.shape[1]
            assert len(analysis["item_missing_rate"]) == n_items

    def test_missing_by_person(self, responses_with_missing):
        """Test missing data by person."""
        responses = responses_with_missing["responses"]

        analysis = analyze_missing(responses)

        if "person_missing_rate" in analysis:
            n_persons = responses.shape[0]
            assert len(analysis["person_missing_rate"]) == n_persons

    def test_exact_summary_and_custom_missing_code(self):
        responses = np.array([[0, 99, 1], [1, 0, 99], [1, 1, 1]])

        analysis = analyze_missing(responses, missing_code=99)

        assert analysis["total_missing_rate"] == pytest.approx(2 / 9)
        np.testing.assert_allclose(analysis["item_missing_rate"], [0, 1 / 3, 1 / 3])
        np.testing.assert_allclose(analysis["person_missing_rate"], [1 / 3, 1 / 3, 0])
        assert analysis["n_complete_cases"] == 1
        assert analysis["n_complete_items"] == 1


class TestListwiseDeletion:
    """Tests for listwise deletion."""

    def test_listwise_deletion(self, responses_with_missing):
        """Test listwise deletion."""
        responses = responses_with_missing["responses"]

        clean = listwise_deletion(responses)

        assert np.all(clean >= 0)

        assert clean.shape[0] <= responses.shape[0]

        assert clean.shape[1] == responses.shape[1]

    def test_listwise_preserves_complete(self, dichotomous_responses):
        """Test that listwise preserves complete data."""
        responses = dichotomous_responses["responses"]

        clean = listwise_deletion(responses)

        assert clean.shape[0] == responses.shape[0]


class TestPairwiseAvailable:
    @pytest.mark.parametrize("chunk_elements", [1, 33, 1_000_000])
    @pytest.mark.parametrize("missing_code", [-1, 99])
    def test_counts_match_pairwise_reference(
        self, monkeypatch, chunk_elements, missing_code
    ):
        rng = np.random.default_rng(53)
        responses = rng.integers(-2, 4, size=(701, 12))[:, ::2]
        responses[:30, 0] = missing_code
        responses[:, 1] = missing_code
        responses[:, 2] = 1
        original = responses.copy()
        valid = (responses >= 0) & (responses != missing_code)
        expected = np.array(
            [
                [np.count_nonzero(first & second) for second in valid.T]
                for first in valid.T
            ]
        )
        monkeypatch.setattr(
            imputation_module, "_PAIRWISE_CHUNK_ELEMENTS", chunk_elements
        )

        available, joint = pairwise_available(responses, missing_code=missing_code)

        np.testing.assert_array_equal(joint, expected)
        np.testing.assert_array_equal(available, np.diag(expected))
        np.testing.assert_array_equal(responses, original)
        assert available.dtype == joint.dtype == np.dtype(np.int_)

    def test_counts_available_responses_and_pairs(self):
        responses = np.array([[1, -1, 0], [0, 1, -1], [-1, 1, 1]])

        available, joint = pairwise_available(responses)

        np.testing.assert_array_equal(available, [2, 2, 2])
        np.testing.assert_array_equal(
            joint,
            [[2, 1, 1], [1, 2, 1], [1, 1, 2]],
        )

    def test_blocked_counts_do_not_overflow(self):
        responses = np.ones((600, 3), dtype=int)
        responses[:17, 1] = -1
        responses[20:49, 2] = -1

        available, joint = pairwise_available(responses)

        np.testing.assert_array_equal(available, [600, 583, 571])
        np.testing.assert_array_equal(
            joint,
            [[600, 583, 571], [583, 583, 554], [571, 554, 571]],
        )
        assert joint.dtype == np.dtype(np.int_)


def test_categorical_draws_respect_degenerate_probabilities():
    probabilities = np.array([[1.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 2.0]])

    draws = _draw_categorical(probabilities, np.random.default_rng(42))

    np.testing.assert_array_equal(draws, [0, 1, 2])


def test_categorical_draws_require_a_probability_matrix():
    with pytest.raises(MirtDataError, match="probabilities"):
        _draw_categorical(np.array([0.5, 0.5]), np.random.default_rng(42))


class TestAverageMI:
    def test_scalar_rubins_rules(self):
        result = averageMI([1.0, 3.0], variances=[1.0, 1.0])

        assert isinstance(result, MIResult)
        assert result.estimate == pytest.approx(2.0)
        assert result.within_variance == pytest.approx(1.0)
        assert result.between_variance == pytest.approx(2.0)
        assert result.total_variance == pytest.approx(4.0)
        assert result.standard_error == pytest.approx(2.0)
        assert result.lambda_hat == pytest.approx(0.75)

    def test_array_standard_errors(self):
        result = averageMI(
            [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            standard_errors=[np.array([1.0, 2.0]), np.array([1.0, 2.0])],
        )

        np.testing.assert_allclose(result.estimate, [2.0, 3.0])
        np.testing.assert_allclose(result.within_variance, [1.0, 4.0])
        np.testing.assert_allclose(result.total_variance, [4.0, 7.0])

    def test_zero_between_variance_is_stable(self):
        with np.errstate(all="raise"):
            result = averageMI([1.0, 1.0], variances=[1.0, 1.0])

        assert result.df == LARGE_DF
        assert result.lambda_hat == 0.0
        assert np.isfinite(result.fmi)

    @pytest.mark.parametrize(
        ("variances", "standard_errors"),
        [(None, None), ([1.0, 1.0], [1.0, 1.0])],
    )
    def test_requires_exactly_one_uncertainty_source(self, variances, standard_errors):
        with pytest.raises(MirtValidationError, match="exactly one"):
            averageMI(
                [1.0, 2.0],
                variances=variances,
                standard_errors=standard_errors,
            )

    def test_uncertainty_count_must_match_imputations(self):
        with pytest.raises(MirtValidationError, match="number"):
            averageMI([1.0, 2.0, 3.0], variances=[1.0, 1.0])

    def test_at_least_two_imputations_are_required(self):
        with pytest.raises(MirtValidationError, match="at least 2"):
            averageMI([1.0], variances=[1.0])

    def test_all_shapes_must_match(self):
        with pytest.raises(MirtValidationError, match="same shape"):
            averageMI(
                [np.array([1.0]), np.array([2.0, 3.0])],
                variances=[np.array([1.0]), np.array([1.0])],
            )

        with pytest.raises(MirtValidationError, match="estimate shape"):
            averageMI(
                [np.array([1.0]), np.array([2.0])],
                variances=[np.array([1.0, 2.0]), np.array([1.0, 2.0])],
            )

    @pytest.mark.parametrize("invalid", [-1.0, np.nan, np.inf])
    def test_uncertainty_must_be_finite_and_nonnegative(self, invalid):
        with pytest.raises(MirtValidationError):
            averageMI([1.0, 2.0], variances=[1.0, invalid])

    def test_estimates_must_be_finite(self):
        with pytest.raises(MirtValidationError, match="finite"):
            averageMI([1.0, np.nan], variances=[1.0, 1.0])

    def test_inputs_must_be_numeric(self):
        with pytest.raises(MirtValidationError, match="estimates"):
            averageMI([1.0, "invalid"], variances=[1.0, 1.0])

        with pytest.raises(MirtValidationError, match="Uncertainty"):
            averageMI([1.0, 2.0], variances=[1.0, "invalid"])


class TestEMImputation:
    """EM imputation calibrates once on the observed responses."""

    @staticmethod
    def _missing_responses(model_name, seed, **kwargs):
        from mirt import simdata

        responses = simdata(model_name, n_persons=300, n_items=6, seed=seed, **kwargs)
        missing = np.random.default_rng(seed).random(responses.shape) < 0.2
        return np.where(missing, -1, responses), missing

    def test_named_model_is_fitted_once_on_the_observed_responses(self, monkeypatch):
        responses, missing = self._missing_responses("2PL", 3)
        calls = []
        original_fit = mirt.fit_mirt

        def counting_fit(data, **kwargs):
            calls.append((np.array(data, copy=True), kwargs))
            return original_fit(data, **kwargs)

        monkeypatch.setattr(mirt, "fit_mirt", counting_fit)

        impute_responses(responses, method="EM", seed=4)

        assert len(calls) == 1
        data, options = calls[0]
        np.testing.assert_array_equal(data, responses)
        assert options["compute_standard_errors"] is False

    @pytest.mark.parametrize(
        ("model_name", "options", "n_categories"),
        [("2PL", {}, 2), ("GRM", {"n_categories": 4}, 4)],
    )
    def test_imputations_keep_observed_cells_and_valid_codes(
        self, model_name, options, n_categories
    ):
        responses, missing = self._missing_responses(model_name, 7, **options)

        first = impute_responses(responses, method="EM", model=model_name, seed=8)
        second = impute_responses(responses, method="EM", model=model_name, seed=8)

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[~missing], responses[~missing])
        assert np.all((first >= 0) & (first < n_categories))

    def test_imputations_follow_the_observed_data_calibration(self):
        from mirt import fit_mirt, fscores, simdata

        responses = simdata("2PL", n_persons=4000, n_items=8, seed=12)
        missing = np.random.default_rng(13).random(responses.shape) < 0.25
        observed = np.where(missing, -1, responses)

        imputed = impute_responses(observed, method="EM", seed=14)

        model = fit_mirt(observed, model="2PL", compute_standard_errors=False).model
        theta = fscores(model, observed, method="EAP").theta.reshape(-1, 1)
        expected = model.probability(theta)
        drawn = imputed[missing].mean()
        predicted = expected[missing].mean()
        mc_error = np.sqrt(predicted * (1 - predicted) / missing.sum())
        assert abs(drawn - predicted) < 4 * mc_error

    def test_calibration_failure_warns_and_draws_empirically(self, monkeypatch):
        responses = np.array([[-1, 1], [0, -1], [1, 0]])

        def failed_fit(*args, **kwargs):
            raise RuntimeError("calibration failed")

        monkeypatch.setattr(mirt, "fit_mirt", failed_fit)
        with pytest.warns(RuntimeWarning, match="calibration failed"):
            imputed = impute_responses(responses, method="EM", seed=2)

        np.testing.assert_array_equal(
            imputed[responses >= 0], responses[responses >= 0]
        )
        assert np.isin(imputed, [0, 1]).all()
