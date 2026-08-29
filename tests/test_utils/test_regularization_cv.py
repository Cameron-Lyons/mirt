"""Tests for regularized MIRT model selection."""

from types import SimpleNamespace

import numpy as np
import pytest

import mirt
import mirt.utils.regularization_cv as regularization_cv
from mirt.estimation.quadrature import GaussHermiteQuadrature
from mirt.estimation.regularized import RegularizedMIRTEstimator
from mirt.exceptions import MirtDataError, MirtEstimationError, MirtValidationError
from mirt.utils import (
    RegularizationCVResult,
    cv_select_lambda,
    information_criteria_path,
)
from mirt.utils.regularization_cv import _compute_test_ll, _score_regularized_result


def _result(
    lambda_value,
    loadings,
    *,
    intercepts=None,
    n_parameters=None,
    aic=10.0,
    bic=10.0,
    ebic=10.0,
):
    loadings = np.asarray(loadings, dtype=np.float64)
    if intercepts is None:
        intercepts = np.zeros(loadings.shape[0])
    if n_parameters is None:
        n_parameters = int(np.count_nonzero(loadings)) + loadings.shape[0]
    return SimpleNamespace(
        lambda_val=float(lambda_value),
        loadings=loadings,
        intercepts=np.asarray(intercepts, dtype=np.float64),
        n_parameters=n_parameters,
        n_nonzero=int(np.count_nonzero(loadings)),
        aic=float(aic),
        bic=float(bic),
        ebic=float(ebic),
    )


@pytest.fixture
def concordant_responses():
    return np.vstack(
        [
            np.tile(np.array([1, 1]), (12, 1)),
            np.tile(np.array([0, 0]), (12, 1)),
        ]
    )


class TestHeldOutLikelihood:
    def test_is_sensitive_to_loading_structure(self, concordant_responses):
        flat = _result(1.0, np.zeros((2, 2)))
        aligned = _result(0.1, np.array([[2.0, 0.0], [2.0, 0.0]]))
        opposed = _result(0.5, np.array([[2.0, 0.0], [-2.0, 0.0]]))

        flat_ll = _compute_test_ll(flat, concordant_responses, n_quadpts=9)
        aligned_ll = _compute_test_ll(aligned, concordant_responses, n_quadpts=9)
        opposed_ll = _compute_test_ll(opposed, concordant_responses, n_quadpts=9)

        assert aligned_ll > flat_ll > opposed_ll

    def test_matches_direct_quadrature_with_missing_values(self):
        result = _result(
            0.2,
            np.array([[1.1, -0.2], [0.3, 0.8]]),
            intercepts=np.array([-0.4, 0.6]),
        )
        responses = np.array([[1, 0], [0, -1], [-1, 1]])
        quadrature = GaussHermiteQuadrature(n_points=5, n_dimensions=2)

        expected = 0.0
        for response in responses:
            conditional = []
            for theta in quadrature.nodes:
                probabilities = 1 / (
                    1 + np.exp(-(theta @ result.loadings.T + result.intercepts))
                )
                likelihood = 1.0
                for item_idx, value in enumerate(response):
                    if value == 1:
                        likelihood *= probabilities[item_idx]
                    elif value == 0:
                        likelihood *= 1 - probabilities[item_idx]
                conditional.append(likelihood)
            expected += np.log(np.dot(quadrature.weights, conditional))

        actual = _compute_test_ll(result, responses, n_quadpts=5)

        assert actual == pytest.approx(expected)

    def test_chunked_scoring_matches_single_block(
        self, concordant_responses, monkeypatch
    ):
        result = _result(0.1, np.array([[1.2, 0.4], [0.5, 1.0]]))
        expected = _compute_test_ll(result, concordant_responses, n_quadpts=5)
        monkeypatch.setattr(
            regularization_cv,
            "_MAX_SCORE_MATRIX_ELEMENTS",
            10,
        )

        actual = _compute_test_ll(result, concordant_responses, n_quadpts=5)

        assert actual == pytest.approx(expected)

    def test_information_criteria_use_held_out_likelihood(self):
        result = _result(
            0.1,
            np.array([[1.0, 0.0], [1.0, 0.0]]),
            n_parameters=4,
        )
        responses = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
        test_ll = _compute_test_ll(result, responses, n_quadpts=5)

        bic_score = _score_regularized_result(result, responses, "bic", 5)
        ebic_score = _score_regularized_result(result, responses, "ebic", 5)

        expected_bic = -(-2 * test_ll + 4 * np.log(4)) / 4
        expected_ebic = expected_bic - (2 * 2 * np.log(2)) / 4
        assert bic_score == pytest.approx(expected_bic)
        assert ebic_score == pytest.approx(expected_ebic)

    def test_rejects_non_dichotomous_data(self):
        result = _result(0.1, np.ones((2, 2)))

        with pytest.raises(MirtDataError, match="dichotomous"):
            _compute_test_ll(result, np.array([[0, 2]]), n_quadpts=3)


class TestCrossValidatedSelection:
    def test_parallel_folds_match_serial_results(self, concordant_responses):
        common = {
            "responses": concordant_responses,
            "lambda_values": [0.3, 0.1],
            "n_folds": 3,
            "n_quadpts": 3,
            "max_iter": 3,
            "tol": 1e-3,
            "seed": 42,
        }

        serial = cv_select_lambda(**common, n_jobs=1)
        parallel = cv_select_lambda(**common, n_jobs=2)

        assert parallel.lambda_values == serial.lambda_values
        assert parallel.best_lambda == serial.best_lambda
        np.testing.assert_allclose(parallel.fold_scores, serial.fold_scores)
        np.testing.assert_allclose(parallel.mean_scores, serial.mean_scores)
        np.testing.assert_allclose(parallel.std_scores, serial.std_scores)
        np.testing.assert_allclose(parallel.mean_nonzero, serial.mean_nonzero)

    def test_selects_loading_structure_from_held_out_data(
        self, concordant_responses, monkeypatch
    ):
        fitted_lambdas = []

        def fake_fit(self, responses, lambda_val=None):
            fitted_lambdas.append(lambda_val)
            if lambda_val == 0.1:
                return _result(lambda_val, np.array([[2.0, 0.0], [2.0, 0.0]]))
            return _result(lambda_val, np.array([[2.0, 0.0], [-2.0, 0.0]]))

        monkeypatch.setattr(RegularizedMIRTEstimator, "fit", fake_fit)

        selection = cv_select_lambda(
            concordant_responses,
            lambda_values=[1.0, 0.1],
            n_folds=3,
            n_quadpts=5,
            max_iter=1,
            seed=42,
        )

        assert selection.best_lambda == 0.1
        assert selection.selected_lambda == 0.1
        assert fitted_lambdas[-1] == 0.1
        assert selection.mean_scores[1] > selection.mean_scores[0]

    def test_bic_does_not_reuse_training_criterion(
        self, concordant_responses, monkeypatch
    ):
        def fake_fit(self, responses, lambda_val=None):
            if lambda_val == 0.1:
                return _result(
                    lambda_val,
                    np.array([[2.0, 0.0], [2.0, 0.0]]),
                    bic=1e9,
                )
            return _result(
                lambda_val,
                np.array([[2.0, 0.0], [-2.0, 0.0]]),
                bic=-1e9,
            )

        monkeypatch.setattr(RegularizedMIRTEstimator, "fit", fake_fit)

        selection = cv_select_lambda(
            concordant_responses,
            lambda_values=[1.0, 0.1],
            n_folds=3,
            criterion="bic",
            n_quadpts=5,
            max_iter=1,
            seed=42,
        )

        assert selection.best_lambda == 0.1

    def test_one_se_rule_chooses_strongest_eligible_penalty(
        self, concordant_responses, monkeypatch
    ):
        fit_lambdas = []
        score_sequences = {
            0.1: iter([8.0, 12.0, 8.0, 12.0]),
            1.0: iter([9.0, 9.0, 9.0, 9.0]),
            0.5: iter([9.5, 9.5, 9.5, 9.5]),
        }

        def fake_fit(self, responses, lambda_val=None):
            fit_lambdas.append(lambda_val)
            return _result(lambda_val, np.ones((2, 2)))

        def fake_score(result, test_data, criterion, n_quadpts):
            return next(score_sequences[result.lambda_val])

        monkeypatch.setattr(RegularizedMIRTEstimator, "fit", fake_fit)
        monkeypatch.setattr(
            regularization_cv,
            "_score_regularized_result",
            fake_score,
        )

        selection = cv_select_lambda(
            concordant_responses,
            lambda_values=[0.1, 1.0, 0.5],
            n_folds=4,
            one_se_rule=True,
            n_quadpts=2,
            max_iter=1,
            seed=42,
        )

        assert selection.best_lambda == 0.1
        assert selection.one_se_lambda == 1.0
        assert selection.selected_lambda == 1.0
        assert fit_lambdas[-1] == 1.0

    def test_zero_signal_generates_single_zero_lambda(
        self, concordant_responses, monkeypatch
    ):
        monkeypatch.setattr(
            RegularizedMIRTEstimator,
            "_compute_lambda_max",
            lambda self, responses: 0.0,
        )
        monkeypatch.setattr(
            RegularizedMIRTEstimator,
            "fit",
            lambda self, responses, lambda_val=None: _result(
                lambda_val, np.zeros((2, 2))
            ),
        )

        selection = cv_select_lambda(
            concordant_responses,
            n_lambda=10,
            n_folds=2,
            n_quadpts=2,
            max_iter=1,
        )

        assert selection.lambda_values == [0.0]
        assert selection.best_lambda == 0.0

    def test_all_failed_fits_raise_clear_error(self, concordant_responses, monkeypatch):
        def fail_fit(self, responses, lambda_val=None):
            raise RuntimeError("fit failed")

        monkeypatch.setattr(RegularizedMIRTEstimator, "fit", fail_fit)

        with pytest.raises(MirtEstimationError, match="Every regularization fit"):
            cv_select_lambda(
                concordant_responses,
                lambda_values=[0.1, 1.0],
                n_folds=2,
                max_iter=1,
            )

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [
            ({"penalty": "none"}, MirtValidationError),
            ({"alpha": -0.1}, MirtValidationError),
            ({"n_factors": 1}, MirtValidationError),
            ({"n_folds": 1}, MirtValidationError),
            ({"n_folds": 25}, MirtValidationError),
            ({"criterion": "aic"}, MirtValidationError),
            ({"one_se_rule": 1}, MirtValidationError),
            ({"n_jobs": 0}, MirtValidationError),
            ({"n_jobs": -2}, MirtValidationError),
            ({"n_jobs": True}, MirtValidationError),
            ({"n_jobs": 1.5}, MirtValidationError),
            ({"lambda_values": []}, MirtValidationError),
            ({"lambda_values": [0.1, np.nan]}, MirtValidationError),
            ({"n_lambda": 0}, MirtValidationError),
            ({"tol": 0}, MirtValidationError),
        ],
    )
    def test_validates_configuration(
        self, concordant_responses, kwargs, error, monkeypatch
    ):
        monkeypatch.setattr(
            RegularizedMIRTEstimator,
            "_compute_lambda_max",
            lambda self, responses: 1.0,
        )

        with pytest.raises(error):
            cv_select_lambda(concordant_responses, max_iter=1, **kwargs)

    def test_rejects_polytomous_responses(self):
        with pytest.raises(MirtDataError, match="dichotomous"):
            cv_select_lambda(
                np.array([[0, 2], [1, 0]]),
                lambda_values=[0.1],
                n_folds=2,
                max_iter=1,
            )


class TestPathAndResult:
    def test_information_path_selects_requested_criterion(self, monkeypatch):
        responses = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        results = [
            _result(1.0, np.ones((2, 2)), aic=4, bic=8, ebic=12),
            _result(0.1, np.ones((2, 2)), aic=6, bic=5, ebic=10),
        ]

        def fake_path(self, responses, lambda_values=None, n_lambda=20, **kwargs):
            assert lambda_values == [1.0, 0.1]
            return results

        monkeypatch.setattr(RegularizedMIRTEstimator, "fit_path", fake_path)

        best_lambda, best_result, path = information_criteria_path(
            responses,
            lambda_values=[1.0, 0.1],
            criterion="bic",
            max_iter=1,
        )

        assert best_lambda == 0.1
        assert best_result is results[1]
        assert path is results

    def test_summary_lists_full_path_and_selected_lambda(self):
        result = RegularizationCVResult(
            lambda_values=[1.0, 0.1],
            mean_scores=[-2.0, -1.0],
            std_scores=[0.2, 0.1],
            best_lambda=0.1,
            best_score=-1.0,
            best_result=_result(0.1, np.ones((2, 2))),
            fold_scores=[[-2.1, -1.9], [-1.1, -0.9]],
            criterion="log_likelihood",
            one_se_lambda=1.0,
            mean_nonzero=[1.0, 4.0],
        )

        summary = result.summary()

        assert "1.000000" in summary
        assert "0.100000" in summary
        assert "* selected for the final fit" in summary
        assert result.selected_lambda == 1.0

    def test_selection_api_is_public(self):
        assert mirt.cv_select_lambda is cv_select_lambda
        assert mirt.information_criteria_path is information_criteria_path
        assert mirt.RegularizationCVResult is RegularizationCVResult
