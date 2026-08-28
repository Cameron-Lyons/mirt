"""Regression tests for mixed-effects extraction and prediction utilities."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import mirt
from mirt import (
    FixedEffects,
    MixedEffectsFitResult,
    MixedEffectsIRT,
    RandomEffects,
    conditional_effects,
    fixef,
    predict_mixed,
    randef,
    shrinkage_estimates,
)
from mirt.models import GradedResponseModel, TwoParameterLogistic


@pytest.fixture
def mixed_result() -> MixedEffectsFitResult:
    model = TwoParameterLogistic(n_items=2)
    model.set_parameters(
        discrimination=np.array([1.0, 1.5]),
        difficulty=np.array([-0.5, 0.75]),
    )
    return MixedEffectsFitResult(
        model=model,
        person_effects=np.array([2.0, -0.5]),
        item_effects=np.array([0.5]),
        log_likelihood=-10.0,
        aic=30.0,
        bic=35.0,
        converged=True,
        person_effect_se=np.array([0.1, 0.2]),
        item_effect_se=np.array([0.05]),
        residual_variance=0.4,
        theta=np.array([-1.0, 0.0, 1.0]),
        theta_se=np.full(3, 0.2),
        person_intercept=0.25,
        item_intercept=-0.75,
        person_intercept_se=0.08,
        item_intercept_se=0.06,
        person_covariate_names=("x", "z"),
        item_covariate_names=("complexity",),
    )


def test_prediction_utilities_are_available_from_top_level() -> None:
    expected = {
        "FixedEffects": FixedEffects,
        "RandomEffects": RandomEffects,
        "conditional_effects": conditional_effects,
        "fixef": fixef,
        "predict_mixed": predict_mixed,
        "randef": randef,
        "shrinkage_estimates": shrinkage_estimates,
    }
    for name, value in expected.items():
        assert getattr(mirt, name) is value


def test_predict_mixed_uses_stored_and_explicit_abilities(
    mixed_result: MixedEffectsFitResult,
) -> None:
    expected = mixed_result.model.probability(mixed_result.theta[:, None])
    np.testing.assert_allclose(predict_mixed(mixed_result), expected)

    explicit = np.array([-2.0, 2.0])
    expected_explicit = mixed_result.model.probability(explicit[:, None])
    np.testing.assert_allclose(
        predict_mixed(mixed_result, new_theta=explicit), expected_explicit
    )


def test_predict_mixed_transforms_person_covariates(
    mixed_result: MixedEffectsFitResult,
) -> None:
    covariates = np.array([[1.0, 2.0], [3.0, -1.0]])
    theta = mixed_result.person_intercept + covariates @ mixed_result.person_effects
    expected = mixed_result.model.probability(theta[:, None])

    actual = predict_mixed(mixed_result, new_covariates=covariates)

    np.testing.assert_allclose(actual, expected)
    assert actual.shape == (2, 2)


def test_predict_mixed_can_evaluate_one_item_without_building_full_matrix(
    mixed_result: MixedEffectsFitResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = predict_mixed(mixed_result)
    calls: list[int | None] = []
    probability = mixed_result.model.probability

    def tracked_probability(theta, item_idx=None):
        calls.append(item_idx)
        return probability(theta, item_idx=item_idx)

    monkeypatch.setattr(mixed_result.model, "probability", tracked_probability)

    selected = predict_mixed(mixed_result, item_idx=1)

    assert calls == [1]
    assert selected.shape == (3,)
    np.testing.assert_allclose(selected, full[:, 1])


def test_predict_mixed_documents_polytomous_output_shapes() -> None:
    model = GradedResponseModel(n_items=2, n_categories=[3, 4])
    theta = np.array([-1.0, 0.5])
    result = SimpleNamespace(model=model, theta=theta)

    full = predict_mixed(result)
    selected = predict_mixed(result, item_idx=0)

    assert full.shape == (2, 2, 4)
    assert selected.shape == (2, 3)
    np.testing.assert_allclose(selected, full[:, 0, :3])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"new_theta": np.array([0.0]), "new_covariates": np.array([0.0, 1.0])},
            "either",
        ),
        ({"new_covariates": np.ones((2, 3))}, "2 columns"),
        ({"new_theta": np.ones((2, 2))}, "1 columns"),
        ({"new_theta": np.array([np.nan])}, "finite"),
    ],
)
def test_predict_mixed_validates_inputs(
    mixed_result: MixedEffectsFitResult,
    kwargs: dict[str, np.ndarray],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        predict_mixed(mixed_result, **kwargs)


@pytest.mark.parametrize("item_idx", [-1, 2, True])
def test_predict_mixed_rejects_invalid_item_indices(
    mixed_result: MixedEffectsFitResult,
    item_idx: int,
) -> None:
    with pytest.raises(IndexError, match="item_idx"):
        predict_mixed(mixed_result, item_idx=item_idx)


def test_random_effect_extraction_returns_independent_arrays(
    mixed_result: MixedEffectsFitResult,
) -> None:
    effects = randef(mixed_result)

    np.testing.assert_array_equal(effects.theta, mixed_result.theta)
    np.testing.assert_array_equal(effects.theta_se, mixed_result.theta_se)
    effects.theta[0] = 99.0
    assert mixed_result.theta[0] == -1.0

    with pytest.raises(ValueError, match="abilities"):
        randef(replace(mixed_result, theta=None))
    with pytest.raises(ValueError, match="group-level"):
        randef(mixed_result, level="group")
    with pytest.raises(ValueError, match="Unknown level"):
        randef(mixed_result, level="school")


def test_random_effect_extraction_validates_uncertainty_and_copies_groups(
    mixed_result: MixedEffectsFitResult,
) -> None:
    with pytest.raises(ValueError, match="non-empty and finite"):
        randef(replace(mixed_result, theta=np.array([np.nan])))
    with pytest.raises(ValueError, match="non-negative"):
        randef(replace(mixed_result, theta_se=np.array([-0.1, 0.2, 0.3])))

    source = SimpleNamespace(
        theta=np.array([0.0]),
        theta_se=np.array([0.1]),
        group_effects={"school": np.array([1.0])},
    )
    extracted = randef(source, level="group")
    extracted.group_effects["school"][0] = 9.0
    assert source.group_effects["school"][0] == pytest.approx(1.0)


def test_fixed_effect_extraction_preserves_names_and_uncertainty(
    mixed_result: MixedEffectsFitResult,
) -> None:
    effects = fixef(mixed_result)

    assert effects.person_intercept == pytest.approx(0.25)
    assert effects.item_intercept == pytest.approx(-0.75)
    assert effects.covariate_effects == {
        "person:x": 2.0,
        "person:z": -0.5,
        "item:complexity": 0.5,
    }
    assert effects.covariate_standard_errors == {
        "person:x": 0.1,
        "person:z": 0.2,
        "item:complexity": 0.05,
    }
    assert effects.person_intercept_standard_error == pytest.approx(0.08)
    assert effects.item_intercept_standard_error == pytest.approx(0.06)

    effects.item_parameters["difficulty"][0] = 99.0
    assert mixed_result.model.parameters["difficulty"][0] == pytest.approx(-0.5)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"person_effects": np.ones((1, 2))}, "non-empty vector"),
        ({"person_effects": np.array([np.inf, 0.0])}, "finite"),
        ({"person_covariate_names": ("x", "x")}, "unique"),
        ({"person_effect_se": np.array([-0.1, 0.2])}, "non-negative"),
        ({"person_intercept": np.inf}, "intercept must be finite"),
    ],
)
def test_fixed_effect_extraction_rejects_malformed_fitted_state(
    mixed_result: MixedEffectsFitResult,
    changes: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        fixef(replace(mixed_result, **changes))


def test_conditional_effects_support_named_person_and_item_terms(
    mixed_result: MixedEffectsFitResult,
) -> None:
    values = np.array([-1, 0, 2])

    person = conditional_effects(mixed_result, "x", values)
    np.testing.assert_array_equal(person["values"], values.astype(float))
    np.testing.assert_allclose(person["effects"], [-2.0, 0.0, 4.0])
    np.testing.assert_allclose(person["se"], [0.1, 0.0, 0.2])

    item = conditional_effects(mixed_result, "item:complexity", values)
    np.testing.assert_allclose(item["effects"], [-0.5, 0.0, 1.0])
    np.testing.assert_allclose(item["se"], [0.05, 0.0, 0.1])


def test_conditional_effects_reject_unknown_and_ambiguous_names(
    mixed_result: MixedEffectsFitResult,
) -> None:
    with pytest.raises(KeyError, match="unknown"):
        conditional_effects(mixed_result, "missing", [0.0])

    ambiguous = replace(mixed_result, item_covariate_names=("x",))
    with pytest.raises(ValueError, match="ambiguous"):
        conditional_effects(ambiguous, "x", [0.0])
    np.testing.assert_allclose(
        conditional_effects(ambiguous, "person:x", [2.0])["effects"], [4.0]
    )
    with pytest.raises(ValueError, match="non-empty"):
        conditional_effects(mixed_result, "x", [])


def test_shrinkage_uses_fitted_ability_uncertainty(
    mixed_result: MixedEffectsFitResult,
) -> None:
    statistics = shrinkage_estimates(mixed_result)

    assert statistics["reliability"] == pytest.approx(0.94)
    assert statistics["shrinkage"] == pytest.approx(0.06)
    assert statistics["icc"] is None

    with pytest.raises(ValueError, match="abilities"):
        shrinkage_estimates(replace(mixed_result, theta_se=None))


def test_shrinkage_validates_variance_components(
    mixed_result: MixedEffectsFitResult,
) -> None:
    result = SimpleNamespace(
        theta=mixed_result.theta,
        theta_se=mixed_result.theta_se,
        variance_components={"between_group": -0.1, "within_group": 1.0},
    )

    with pytest.raises(ValueError, match="non-negative"):
        shrinkage_estimates(result)


def test_fit_retains_effect_metadata_and_honors_controls(monkeypatch) -> None:
    responses = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    person_covariates = np.arange(4.0)
    item_covariates = np.arange(3.0)
    theta = 2.0 + 3.0 * person_covariates
    fitted_model = TwoParameterLogistic(n_items=3)
    fitted_model.set_parameters(
        discrimination=np.ones(3),
        difficulty=-1.0 + 0.5 * item_covariates,
    )
    received: dict[str, float | int] = {}

    def fake_fit_mirt(response_data, **kwargs):
        np.testing.assert_array_equal(response_data, responses)
        received.update(kwargs)
        return SimpleNamespace(
            model=fitted_model,
            log_likelihood=-10.0,
            converged=True,
        )

    def fake_fscores(model, response_data, method):
        assert model is fitted_model
        np.testing.assert_array_equal(response_data, responses)
        assert method == "EAP"
        return SimpleNamespace(
            theta=theta[:, None],
            standard_error=np.full((4, 1), 0.2),
        )

    monkeypatch.setattr(mirt, "fit_mirt", fake_fit_mirt)
    monkeypatch.setattr("mirt.scoring.fscores", fake_fscores)

    estimator = MixedEffectsIRT(
        person_covariates=person_covariates,
        item_covariates=item_covariates,
        person_covariate_names=["experience"],
        item_covariate_names=["complexity"],
    )
    result = estimator.fit(responses, max_iter=7, tol=1e-6)

    assert received["model"] == "2PL"
    assert received["max_iter"] == 7
    assert received["tol"] == pytest.approx(1e-6)
    np.testing.assert_allclose(result.theta, theta)
    np.testing.assert_allclose(result.theta_se, 0.2)
    assert result.person_intercept == pytest.approx(2.0)
    np.testing.assert_allclose(result.person_effects, [3.0])
    assert result.item_intercept == pytest.approx(-1.0)
    np.testing.assert_allclose(result.item_effects, [0.5])
    np.testing.assert_allclose(estimator.predict_theta([[4.0]]), [14.0])
    np.testing.assert_allclose(estimator.predict_difficulty([[4.0]]), [1.0])

    summary = result.summary()
    assert "experience: 3.0000" in summary
    assert "complexity: 0.5000" in summary


@pytest.mark.parametrize(
    "kwargs",
    [
        {"person_covariates": np.empty((3, 0))},
        {
            "person_covariates": np.ones((3, 2)),
            "person_covariate_names": ["duplicate", "duplicate"],
        },
        {
            "person_covariates": np.ones((3, 1)),
            "person_covariate_names": ["person:age"],
        },
        {
            "person_covariates": np.ones((3, 1)),
            "person_covariate_names": ["age", "extra"],
        },
    ],
)
def test_mixed_effect_covariate_metadata_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        MixedEffectsIRT(**kwargs)


def test_linear_effect_fit_handles_rank_deficiency() -> None:
    covariates = np.column_stack([np.arange(4.0), np.arange(4.0)])
    outcomes = np.array([1.0, 2.0, 3.0, 4.0])

    intercept, effects, intercept_se, effect_se, variance = (
        MixedEffectsIRT._fit_linear_effects(covariates, outcomes)
    )

    assert np.isfinite(intercept)
    assert np.all(np.isfinite(effects))
    assert np.isfinite(intercept_se)
    assert np.all(np.isfinite(effect_se))
    assert variance > 0.0
