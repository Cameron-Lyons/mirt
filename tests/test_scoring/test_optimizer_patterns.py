"""Contracts shared by optimizer-based ability scorers."""

from collections.abc import Callable

import numpy as np
import pytest

from mirt.models import GradedResponseModel, TwoParameterLogistic
from mirt.scoring import fscores
from mirt.scoring.map import MAPScorer
from mirt.scoring.ml import MLScorer
from mirt.scoring.wle import WLEScorer

OptimizerScorer = MLScorer | MAPScorer | WLEScorer


@pytest.mark.parametrize(
    ("factory", "method_name"),
    [
        (MLScorer, "_score_unidimensional"),
        (MAPScorer, "_score_unidimensional"),
        (WLEScorer, "_estimate_person"),
    ],
)
def test_optimizer_scorers_reuse_unique_response_patterns(
    fitted_2pl_model,
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[], OptimizerScorer],
    method_name: str,
) -> None:
    model = fitted_2pl_model.model
    first = np.resize(np.array([0, 1]), model.n_items)
    second = 1 - first
    responses = np.vstack([first, second, first, first, second])
    scorer = factory()
    original = getattr(scorer, method_name)
    calls = 0

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(scorer, method_name, counted)

    result = scorer.score(model, responses)

    assert calls == 2
    np.testing.assert_allclose(result.theta[[0, 2, 3]], result.theta[0])
    np.testing.assert_allclose(
        result.standard_error[[0, 2, 3]], result.standard_error[0]
    )
    np.testing.assert_allclose(result.theta[[1, 4]], result.theta[1])


def test_missing_codes_share_one_optimizer_result(
    fitted_2pl_model,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = fitted_2pl_model.model
    responses = np.zeros((2, model.n_items), dtype=int)
    responses[0, 0] = -1
    responses[1, 0] = -999
    scorer = WLEScorer()
    original = scorer._estimate_person
    calls = 0

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(scorer, "_estimate_person", counted)

    result = scorer.score(model, responses)

    assert calls == 1
    np.testing.assert_allclose(result.theta[0], result.theta[1])
    np.testing.assert_allclose(result.standard_error[0], result.standard_error[1])


@pytest.mark.parametrize("scorer", [MLScorer(), MAPScorer(), WLEScorer()])
@pytest.mark.parametrize(
    ("responses", "message"),
    [
        ([0, 1], "2D"),
        ([[0, 1, 0]], "items, expected"),
        ([[0.0, 0.5]], "integer-valued"),
        ([[0.0, np.nan]], "finite"),
        ([[0, 2]], "only 0, 1"),
        ([["yes", "no"]], "numeric"),
    ],
)
def test_optimizer_scorers_validate_response_contracts(
    scorer: OptimizerScorer,
    responses: object,
    message: str,
) -> None:
    model = TwoParameterLogistic(n_items=2)
    model._is_fitted = True

    with pytest.raises(ValueError, match=message):
        scorer.score(model, responses)  # type: ignore[arg-type]


def test_optimizer_scorers_validate_polytomous_categories() -> None:
    model = GradedResponseModel(n_items=2, n_categories=[3, 4])
    model._is_fitted = True

    with pytest.raises(ValueError, match="category range"):
        WLEScorer().score(model, np.array([[3, 0]]))


@pytest.mark.parametrize("scorer", [MLScorer(), MAPScorer(), WLEScorer()])
def test_optimizer_scorers_support_empty_batches(scorer: OptimizerScorer) -> None:
    model = TwoParameterLogistic(n_items=2)
    model._is_fitted = True

    result = scorer.score(model, np.empty((0, 2), dtype=int))

    assert result.theta.shape == (0,)
    assert result.standard_error.shape == (0,)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bounds": (1.0, -1.0)}, "lower < upper"),
        ({"bounds": (0.0,)}, "exactly two"),
        ({"bounds": (0.0, np.inf)}, "finite"),
        ({"tol": 0.0}, "positive"),
        ({"tol": np.nan}, "finite"),
        ({"n_jobs": 0}, "positive integer"),
        ({"n_jobs": -2}, "positive integer"),
    ],
)
def test_wle_validates_configuration(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        WLEScorer(**kwargs)  # type: ignore[arg-type]


def test_wle_reports_factor_specific_multidimensional_uncertainty() -> None:
    model = TwoParameterLogistic(n_items=8, n_factors=2)
    model.set_parameters(
        discrimination=np.array(
            [
                [1.8, 0.1],
                [1.6, 0.2],
                [1.4, 0.1],
                [1.2, 0.2],
                [0.2, 0.9],
                [0.1, 0.8],
                [0.2, 0.7],
                [0.1, 0.6],
            ]
        ),
        difficulty=np.linspace(-1.0, 1.0, 8),
    )
    model._is_fitted = True
    responses = np.array([[1, 0, 1, 0, 1, 0, 1, 0]])

    result = WLEScorer().score(model, responses)

    assert result.standard_error.shape == (1, 2)
    assert np.all(np.isfinite(result.standard_error))
    assert not np.isclose(result.standard_error[0, 0], result.standard_error[0, 1])


def test_wle_parallel_results_match_serial(fitted_2pl_model) -> None:
    model = fitted_2pl_model.model
    responses = np.vstack(
        [
            np.resize(np.array([0, 1]), model.n_items),
            np.resize(np.array([1, 0, 0]), model.n_items),
            np.resize(np.array([1, 1, 0]), model.n_items),
        ]
    )

    serial = WLEScorer(n_jobs=1).score(model, responses)
    parallel = WLEScorer(n_jobs=2).score(model, responses)

    np.testing.assert_allclose(parallel.theta, serial.theta)
    np.testing.assert_allclose(parallel.standard_error, serial.standard_error)


@pytest.mark.parametrize("method", ["MAP", "ML", "WLE"])
def test_fscores_exposes_parallel_optimizer_scoring(
    fitted_2pl_model,
    method: str,
) -> None:
    model = fitted_2pl_model.model
    responses = np.vstack(
        [
            np.resize(np.array([0, 1]), model.n_items),
            np.resize(np.array([1, 0, 0]), model.n_items),
        ]
    )

    serial = fscores(model, responses, method=method, n_jobs=1)  # type: ignore[arg-type]
    parallel = fscores(model, responses, method=method, n_jobs=2)  # type: ignore[arg-type]

    np.testing.assert_allclose(parallel.theta, serial.theta)
    np.testing.assert_allclose(parallel.standard_error, serial.standard_error)
