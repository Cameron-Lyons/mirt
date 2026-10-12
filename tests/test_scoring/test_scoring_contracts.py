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


def test_person_id_checks_share_one_validator() -> None:
    import mirt.results.ability_posterior as posterior_module
    import mirt.results.score_result as score_module
    from mirt.results._common import validate_person_ids

    assert not hasattr(scoring, "_validated_person_ids")
    assert score_module.validate_person_ids is validate_person_ids
    assert posterior_module.validate_person_ids is validate_person_ids
    ids = np.array(["a", "b"])
    assert validate_person_ids(ids, 2, row_label="score") == ["a", "b"]
    assert validate_person_ids(None, 2, row_label="score") is None
    with pytest.raises(MirtValidationError, match="one identifier per posterior row"):
        posterior_module.AbilityPosteriorResult._validated_person_ids(["a"], 2)
    with pytest.raises(MirtValidationError, match="one identifier per score row"):
        ScoreResult(np.zeros(2), np.ones(2), "EAP", person_ids=["a"])
    with pytest.raises(MirtValidationError, match="one-dimensional"):
        ScoreResult(np.zeros(2), np.ones(2), "EAP", person_ids=np.zeros((2, 1)))


def _named_model() -> TwoParameterLogistic:
    model = _model(n_items=6)
    model.item_names = [f"Q{index}" for index in range(6)]
    return model


@pytest.mark.parametrize("library", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["EAP", "MAP", "ML", "WLE", "EAPsum"])
def test_fscores_matches_data_frame_columns_to_item_names(
    library: str, method: str
) -> None:
    frames = pytest.importorskip(library)
    model = _named_model()
    responses = _responses(n_persons=12)
    columns = {name: responses[:, index] for index, name in enumerate(model.item_names)}
    reversed_columns = dict(reversed(list(columns.items())))
    reversed_columns["respondent"] = np.arange(12)

    expected = fscores(model, responses, method=method)
    actual = fscores(model, frames.DataFrame(reversed_columns), method=method)

    assert_array_equal(actual.theta, expected.theta)
    assert_array_equal(actual.standard_error, expected.standard_error)


def test_scoring_entry_points_share_data_frame_alignment() -> None:
    pd = pytest.importorskip("pandas")
    from mirt.utils.data import align_response_columns
    from mirt.utils.plausible import generate_plausible_values

    model = _named_model()
    responses = _responses(n_persons=9)
    frame = pd.DataFrame(responses, columns=model.item_names)
    shuffled = frame[["Q3", "Q5", "Q0", "Q2", "Q4", "Q1"]]

    assert_array_equal(
        align_response_columns(shuffled, model.item_names).to_numpy(), responses
    )
    assert_array_equal(
        ability_posterior(model, shuffled).weights,
        ability_posterior(model, responses).weights,
    )
    assert_array_equal(
        scoring.eapsum(model, shuffled).theta, scoring.eapsum(model, responses).theta
    )
    assert_array_equal(
        generate_plausible_values(model, shuffled, seed=3),
        generate_plausible_values(model, responses, seed=3),
    )


def test_data_frames_missing_named_items_are_rejected() -> None:
    pd = pytest.importorskip("pandas")
    from mirt.exceptions import MirtDataError

    model = _named_model()
    responses = _responses()
    frame = pd.DataFrame(responses, columns=model.item_names)

    with pytest.raises(MirtDataError, match="model's items Q4, Q5"):
        fscores(model, frame[["Q0", "Q1", "Q2", "Q3"]])
    renamed = frame.rename(columns={"Q4": "X4", "Q5": "X5"})
    with pytest.raises(MirtDataError, match="Q4, Q5"):
        ability_posterior(model, renamed)
    # Arrays keep positional semantics.
    assert_array_equal(
        fscores(model, renamed.to_numpy()).theta, fscores(model, responses).theta
    )


def test_unnamed_models_and_default_labels_read_frames_by_position() -> None:
    pd = pytest.importorskip("pandas")
    from mirt.utils.data import align_response_columns

    model = _model()
    responses = _responses()
    expected = fscores(model, responses).theta

    labelled = pd.DataFrame(responses, columns=[f"V{index}" for index in range(6)])
    assert_array_equal(fscores(model, labelled).theta, expected)
    assert_array_equal(fscores(model, pd.DataFrame(responses)).theta, expected)
    # Default model labels still select named columns when present.
    shuffled = pd.DataFrame(responses, columns=model.item_names).iloc[:, ::-1]
    assert_array_equal(fscores(model, shuffled).theta, expected)
    assert align_response_columns(responses, model.item_names) is responses
    assert align_response_columns(labelled, None) is labelled


def test_scoring_validators_share_messages_and_error_type() -> None:
    from mirt.exceptions import MirtDataError
    from mirt.models import GradedResponseModel

    model = _model()
    for method in ("EAP", "ML", "EAPsum"):
        with pytest.raises(MirtDataError, match="must be coded as 0 or 1"):
            fscores(model, np.full((1, 6), 2), method=method)
        with pytest.raises(MirtDataError, match="responses must be 2D"):
            fscores(model, np.zeros(6, dtype=int), method=method)
        with pytest.raises(MirtDataError, match="has 3 items, expected 6"):
            fscores(model, np.zeros((1, 3), dtype=int), method=method)
        # Missing codes must be integers, as in fit_mirt.
        with pytest.raises(MirtDataError, match="integer-valued"):
            fscores(model, np.array([[-0.5, 0, 1, 0, 1, 0]]), method=method)

    graded = GradedResponseModel(n_items=2, n_categories=[3, 4])
    graded._is_fitted = True
    for scorer in (EAPScorer(), EAPSumScorer()):
        with pytest.raises(MirtDataError, match="item 0 must be below 3"):
            scorer.score(graded, np.array([[3, 0]]))
    # EAPsum now normalizes every missing code, like the other scorers.
    assert_array_equal(
        EAPSumScorer._validate_responses(graded, np.array([[-9, 1]])), [[-1, 1]]
    )
