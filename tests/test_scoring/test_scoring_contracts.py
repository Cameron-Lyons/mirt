"""fscores identifier contracts and dimension-aware EAP quadrature defaults."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import mirt.scoring as scoring
import mirt.scoring.eap as eap_module
from mirt.exceptions import MirtValidationError
from mirt.models import TwoParameterLogistic
from mirt.results.score_result import ScoreResult
from mirt.scoring import EAPScorer, EAPSumScorer, ability_posterior, fscores


def _model(n_factors: int = 1, n_items: int = 6) -> TwoParameterLogistic:
    rng = np.random.default_rng(n_factors)
    model = TwoParameterLogistic(n_items=n_items, n_factors=n_factors)
    discrimination = rng.uniform(0.6, 1.6, (n_items, n_factors))
    model.set_parameters(
        discrimination=discrimination[:, 0] if n_factors == 1 else discrimination,
        difficulty=rng.normal(size=n_items),
    )
    model._is_fitted = True
    return model


def _responses(n_persons: int = 7, n_items: int = 6) -> np.ndarray:
    responses = np.random.default_rng(5).integers(0, 2, (n_persons, n_items))
    responses[0, 0] = -1
    return responses


@pytest.mark.parametrize("method", ["EAP", "MAP", "ML", "WLE", "EAPsum"])
def test_fscores_rejects_mismatched_person_ids_before_scoring(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("scoring should not start")

    for scorer in ("EAPScorer", "EAPSumScorer", "MAPScorer", "MLScorer", "WLEScorer"):
        monkeypatch.setattr(getattr(scoring, scorer), "score", fail)

    with pytest.raises(MirtValidationError, match="one identifier per score row"):
        fscores(_model(), _responses(), method=method, person_ids=["a", "b"])
    with pytest.raises(MirtValidationError, match="one-dimensional"):
        fscores(
            _model(),
            _responses(),
            method=method,
            person_ids=np.arange(14).reshape(7, 2),
        )


def test_fscores_normalizes_array_person_ids_for_round_trips() -> None:
    ids = np.array([f"p{index}" for index in range(7)])

    result = fscores(_model(), _responses(), method="EAP", person_ids=ids)

    assert result.person_ids == ids.tolist()
    assert type(result.person_ids) is list
    restored = ScoreResult.from_json(result.to_json())
    assert restored.person_ids == ids.tolist()
    assert_array_equal(restored.theta, result.theta)


def test_fscores_without_person_ids_keeps_unlabelled_rows() -> None:
    result = fscores(_model(), _responses(), method="MAP")

    assert result.person_ids is None


def test_default_quadrature_sizes_shrink_with_dimension() -> None:
    assert [eap_module._default_n_quadpts(n) for n in range(1, 9)] == [
        49,
        49,
        21,
        9,
        7,
        5,
        5,
        5,
    ]


@pytest.mark.parametrize("n_factors", [1, 2])
def test_low_dimensional_defaults_keep_49_point_results(n_factors: int) -> None:
    model = _model(n_factors)
    responses = _responses()

    default = fscores(model, responses)
    explicit = fscores(model, responses, n_quadpts=49)
    posterior = ability_posterior(model, responses)

    assert_array_equal(default.theta, explicit.theta)
    assert_array_equal(default.standard_error, explicit.standard_error)
    assert posterior.n_points == 49**n_factors


@pytest.mark.parametrize(("n_factors", "n_quadpts"), [(3, 21), (4, 9), (6, 5)])
def test_high_dimensional_defaults_use_feasible_grids(
    n_factors: int, n_quadpts: int
) -> None:
    model = _model(n_factors, n_items=8)
    responses = _responses(n_items=8)

    posterior = ability_posterior(model, responses)
    default = EAPScorer().score(model, responses)
    explicit = EAPScorer(n_quadpts=n_quadpts).score(model, responses)

    assert posterior.n_points == n_quadpts**n_factors
    assert_array_equal(default.theta, explicit.theta)
    assert_array_equal(default.standard_error, explicit.standard_error)


def test_explicit_grids_above_the_node_limit_warn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(eap_module, "_LARGE_GRID_NODES", 100)
    model = _model(2)
    responses = _responses()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        EAPScorer(n_quadpts=10).score(model, responses)
    with pytest.warns(RuntimeWarning, match="121 grid nodes"):
        EAPScorer(n_quadpts=11).score(model, responses)
    with pytest.warns(RuntimeWarning, match="121 grid nodes"):
        ability_posterior(model, responses, n_quadpts=11)
    with pytest.warns(RuntimeWarning, match="omit n_quadpts to use 49 points"):
        EAPScorer(n_quadpts=51).score(model, responses)
    # The automatic grid can only be shrunk by switching methods.
    with pytest.warns(RuntimeWarning, match="2401 grid nodes.*MAP scoring"):
        EAPScorer().score(model, responses)


def test_eap_scorer_reports_automatic_grid() -> None:
    assert repr(EAPScorer()) == "EAPScorer(n_quadpts=None)"
    assert repr(EAPScorer(n_quadpts=21)) == "EAPScorer(n_quadpts=21)"
    with pytest.raises(ValueError, match="at least 5"):
        EAPScorer(n_quadpts=4)


def test_fscores_eapsum_default_keeps_49_points() -> None:
    model = _model()
    responses = _responses()

    actual = fscores(model, responses, method="EAPsum")
    expected = EAPSumScorer(n_quadpts=49).score(model, responses)

    assert_array_equal(actual.theta, expected.theta)
    assert_array_equal(actual.standard_error, expected.standard_error)
