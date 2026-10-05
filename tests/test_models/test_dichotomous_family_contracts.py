"""Shared setters and curve dispatch across the parameterized model families."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mirt.models.dichotomous as dichotomous
from mirt import fscores
from mirt._model_defaults import uses_builtin_model_hooks
from mirt.exceptions import MirtModelError, MirtValidationError
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
from mirt.models.sequential import (
    AdjacentCategoryModel,
    ContinuationRatioModel,
    SequentialResponseModel,
)
from mirt.models.zeroinflated import HurdleIRT, ZeroInflated2PL, ZeroInflated3PL

_REGISTERED = [
    TwoParameterLogistic,
    OneParameterLogistic,
    ThreeParameterLogistic,
    FourParameterLogistic,
]
_CURVES = [
    TwoParameterLogistic,
    OneParameterLogistic,
    ThreeParameterLogistic,
    FourParameterLogistic,
    FiveParameterLogistic,
    UnipolarLogLogistic,
    ComplementaryLogLog,
    NegativeLogLog,
]
_ATOMIC = [
    TwoParameterLogistic,
    FiveParameterLogistic,
    ZeroInflated2PL,
    ZeroInflated3PL,
    HurdleIRT,
    SequentialResponseModel,
    ContinuationRatioModel,
    AdjacentCategoryModel,
]


def _build(model_class: type) -> object:
    if model_class in (
        SequentialResponseModel,
        ContinuationRatioModel,
        AdjacentCategoryModel,
    ):
        return model_class(3, [3, 4, 2])
    return model_class(3)


def _constant_curve(self, theta, item_idx=None, **kwargs):
    shape = (len(theta),) if item_idx is not None else (len(theta), self.n_items)
    return np.full(shape, 0.3)


@pytest.mark.parametrize("model_class", _REGISTERED)
def test_registered_dichotomous_models_use_builtin_hooks(model_class: type) -> None:
    model = model_class(3)

    assert uses_builtin_model_hooks(model)
    assert uses_builtin_model_hooks(model, likelihood=True)


@pytest.mark.parametrize("hook", ["probability", "_evaluate_logistic"])
@pytest.mark.parametrize("target", ["instance", "class", "family", "subclass"])
@pytest.mark.parametrize("model_class", _REGISTERED)
def test_replaced_curve_hooks_disable_builtin_shortcuts(
    monkeypatch: pytest.MonkeyPatch, model_class: type, target: str, hook: str
) -> None:
    if target == "subclass":
        custom = type("Custom", (model_class,), {hook: _constant_curve})
        model = custom(3)
    else:
        model = model_class(3)
        if target == "instance":
            monkeypatch.setattr(model, hook, _constant_curve.__get__(model))
        elif target == "class":
            monkeypatch.setattr(model_class, hook, _constant_curve)
        else:
            monkeypatch.setattr(
                dichotomous._ParameterizedDichotomousModel, hook, _constant_curve
            )

    assert not uses_builtin_model_hooks(model)
    assert_array_equal(model.probability(np.zeros((2, 1)), 1), [0.3, 0.3])


@pytest.mark.parametrize(
    ("model_class", "hook"),
    [
        (TwoParameterLogistic, "_evaluate_logistic"),
        (FiveParameterLogistic, "_evaluate_curve"),
        (UnipolarLogLogistic, "_evaluate_curve"),
    ],
)
def test_curve_methods_resolve_the_evaluator_at_call_time(
    monkeypatch: pytest.MonkeyPatch, model_class: type, hook: str
) -> None:
    model = model_class(3)
    calls = []

    def evaluator(theta, item_idx, *, information=False, item_indices=None):
        calls.append((item_idx, information, item_indices is not None))
        return np.zeros(len(theta))

    monkeypatch.setattr(model, hook, evaluator)
    theta = np.zeros((2, 1))
    model.probability(theta, 1)
    model.information(theta, 2)
    model.probability_pairs(theta, np.array([0, 2]))

    assert calls == [(1, False, False), (2, True, False), (None, False, True)]


@pytest.mark.parametrize("n_items", [1, 3])
@pytest.mark.parametrize("model_class", _CURVES)
def test_row_blocks_match_unblocked_evaluation(
    monkeypatch: pytest.MonkeyPatch, model_class: type, n_items: int
) -> None:
    model = model_class(n_items)
    rng = np.random.default_rng(7)
    model.set_parameters(difficulty=rng.normal(size=n_items))
    if model_class not in (OneParameterLogistic,):
        model.set_parameters(discrimination=rng.uniform(0.5, 2.0, n_items))
    theta = np.linspace(-3.0, 3.0, 23)[:, None]
    indices = rng.integers(0, n_items, theta.shape[0])
    expected = {
        "probability": model.probability(theta),
        "information": model.information(theta),
        "single": model.information(theta, n_items - 1),
        "pairs": model.probability_pairs(theta, indices),
    }

    monkeypatch.setattr(dichotomous, "_LOGISTIC_CURVE_CHUNK_ELEMENTS", 5)
    monkeypatch.setattr(dichotomous, "_UNIDIMENSIONAL_CURVE_CHUNK_ELEMENTS", 5)
    actual = {
        "probability": model.probability(theta),
        "information": model.information(theta),
        "single": model.information(theta, n_items - 1),
        "pairs": model.probability_pairs(theta, indices),
    }

    for key, values in expected.items():
        assert actual[key].shape == values.shape
        assert_allclose(actual[key], values, rtol=1e-15, atol=0.0)
    assert expected["probability"].shape == (theta.shape[0], n_items)


def test_five_pl_shares_the_curve_kernel_without_changing_values() -> None:
    model = FiveParameterLogistic(3).set_parameters(
        discrimination=np.array([0.7, 1.4, 2.1]),
        difficulty=np.array([-1.0, 0.0, 1.5]),
        guessing=np.array([0.1, 0.2, 0.0]),
        upper=np.array([0.95, 1.0, 0.9]),
        asymmetry=np.array([0.5, 1.0, 2.5]),
    )
    theta = np.linspace(-4.0, 4.0, 9)
    parameters = [model.parameters[name] for name in model._curve_parameter_names]

    for information in (False, True):
        method = model.information if information else model.probability
        assert_array_equal(
            method(theta[:, None]),
            dichotomous._five_pl_curve(
                theta[:, None], *parameters, information=information
            ),
        )
        assert_array_equal(
            method(theta[:, None], 1),
            dichotomous._five_pl_curve(
                theta, *[values[1] for values in parameters], information=information
            ),
        )


@pytest.mark.parametrize(
    "model_class",
    [
        OneParameterLogistic,
        ThreeParameterLogistic,
        FourParameterLogistic,
        FiveParameterLogistic,
        UnipolarLogLogistic,
        ComplementaryLogLog,
    ],
)
def test_unidimensional_families_reject_extra_factors(model_class: type) -> None:
    with pytest.raises(MirtModelError, match="only supports unidimensional"):
        model_class(3, n_factors=2)
    with pytest.raises(MirtValidationError, match="n_factors must be positive"):
        model_class(3, n_factors=0)


def test_family_properties_are_not_added_to_simpler_models() -> None:
    two_pl = TwoParameterLogistic(2)

    assert_array_equal(two_pl.discrimination, [1.0, 1.0])
    assert_array_equal(UnipolarLogLogistic(2).difficulty, [0.0, 0.0])
    assert not hasattr(two_pl, "guessing")
    assert not hasattr(ThreeParameterLogistic(2), "upper")
    assert not hasattr(SequentialResponseModel(2, 3), "difficulty")


@pytest.mark.parametrize("model_class", _ATOMIC)
def test_non_numeric_updates_raise_validation_errors(model_class: type) -> None:
    model = _build(model_class)
    before = model.parameters

    with pytest.raises(MirtValidationError, match="must contain numeric values"):
        model.set_parameters(discrimination=["a", "b", "c"])
    with pytest.raises(MirtValidationError, match="must contain numeric values"):
        model.set_parameters(discrimination=object())
    with pytest.raises(MirtValidationError, match="Invalid per-item value"):
        model.set_item_parameter(0, "discrimination", "high")

    for name, values in before.items():
        assert_array_equal(model.parameters[name], values)


@pytest.mark.parametrize("model_class", _ATOMIC)
def test_rejected_updates_leave_every_parameter_unchanged(model_class: type) -> None:
    model = _build(model_class)
    before = model.parameters
    names = list(before)

    with pytest.raises(MirtValidationError):
        model.set_parameters(
            **{names[-1]: before[names[-1]] + 0.25},
            discrimination=np.array([1.0, np.nan, 1.0]),
        )
    with pytest.raises(MirtValidationError, match="Unknown parameter"):
        model.set_item_parameter(0, "slope", 1.0)
    with pytest.raises(MirtValidationError, match="must be a scalar for one item"):
        model.set_item_parameter(0, "discrimination", [1.0, 2.0])

    for name, values in before.items():
        assert_array_equal(model.parameters[name], values)


def test_updates_own_their_arrays_and_keep_unchanged_values() -> None:
    model = ZeroInflated2PL(2)
    difficulty = np.array([-0.5, 0.5])

    model.set_parameters(difficulty=difficulty)
    difficulty[0] = 9.0
    model.set_item_parameter(np.int64(1), "zero_inflation", 0.25)

    assert_array_equal(model.difficulty, [-0.5, 0.5])
    assert_array_equal(model.zero_inflation, [0.1, 0.25])
    assert_array_equal(model.discrimination, [1.0, 1.0])


def test_two_dimensional_item_updates_require_one_slope_row() -> None:
    model = TwoParameterLogistic(2, n_factors=2)
    model.set_item_parameter(1, "discrimination", [0.5, 1.5])

    assert_array_equal(model.discrimination, [[1.0, 1.0], [0.5, 1.5]])
    with pytest.raises(MirtValidationError, match=r"must have shape \(2,\)"):
        model.set_item_parameter(1, "discrimination", 1.0)


def test_ordinal_item_updates_accept_active_or_padded_thresholds() -> None:
    model = SequentialResponseModel(2, [3, 4])

    model.set_item_parameter(0, "thresholds", [-0.5, 0.5])
    assert_array_equal(model.thresholds[0], [-0.5, 0.5, 0.0])
    model.set_item_parameter(0, "thresholds", [-1.0, 1.0, 0.0])
    assert_array_equal(model.thresholds[0], [-1.0, 1.0, 0.0])
    with pytest.raises(MirtValidationError, match="must have 2 active values"):
        model.set_item_parameter(0, "thresholds", [0.0])
    with pytest.raises(MirtValidationError, match="finite values"):
        model.set_item_parameter(1, "thresholds", [0.0, np.inf, 1.0])
    assert_array_equal(model.thresholds_for_item(0), [-1.0, 1.0])


@pytest.mark.parametrize(
    "model_class",
    [
        OneParameterLogistic,
        ThreeParameterLogistic,
        FourParameterLogistic,
        FiveParameterLogistic,
        UnipolarLogLogistic,
        NegativeLogLog,
        ZeroInflated2PL,
    ],
)
@pytest.mark.parametrize(
    "n_factors", [1.0, np.int64(1), np.float64(1.0), True], ids=repr
)
def test_unidimensional_families_store_an_integer_factor_count(
    model_class: type, n_factors: object
) -> None:
    model = model_class(3, n_factors=n_factors)

    assert model.n_factors == 1
    assert type(model.n_factors) is int
    assert model.copy().n_factors == 1


@pytest.mark.parametrize("n_factors", [0.5, 1.5, None, "1"], ids=repr)
def test_unidimensional_families_reject_non_unit_factor_counts(
    n_factors: object,
) -> None:
    with pytest.raises(ValueError):
        ThreeParameterLogistic(3, n_factors=n_factors)  # type: ignore[arg-type]


def test_float_unit_factor_count_still_scores() -> None:
    model = ThreeParameterLogistic(4, n_factors=1.0)
    model._is_fitted = True
    responses = np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 0, 1]])

    scores = fscores(model, responses, method="EAP")

    assert np.all(np.isfinite(np.asarray(scores.theta)))


def test_sequential_threshold_updates_reject_non_numeric_values() -> None:
    model = SequentialResponseModel(2, [3, 4])
    before = model.thresholds.copy()

    with pytest.raises(MirtValidationError, match="Invalid per-item value"):
        model.set_item_parameter(0, "thresholds", ["low", "high"])

    assert_array_equal(model.thresholds, before)
