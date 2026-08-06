"""Regression coverage for parameter sampling and score propagation."""

import numpy as np
import pytest

import mirt
from mirt.models.dichotomous import (
    FiveParameterLogistic,
    FourParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.polytomous import GeneralizedPartialCredit
from mirt.utils.sampling import (
    ParameterSamples,
    draw_parameters,
    posterior_summary,
    sample_expected_scores,
)


def _direct_expected_scores(model, theta, samples):
    expected = np.empty((samples.discrimination.shape[0], len(theta)))
    for sample_idx in range(samples.discrimination.shape[0]):
        sampled_model = model.copy()
        parameters = {
            "discrimination": samples.discrimination[sample_idx],
            "difficulty": samples.difficulty[sample_idx],
        }
        for name in ("guessing", "upper", "asymmetry"):
            values = getattr(samples, name)
            if values is not None:
                parameters[name] = values[sample_idx]
        sampled_model.set_parameters(**parameters)
        expected[sample_idx] = sampled_model.probability(theta).sum(axis=1)
    return expected


def test_sample_expected_scores_matches_multidimensional_2pl_models():
    model = TwoParameterLogistic(n_items=2, n_factors=2)
    samples = ParameterSamples(
        discrimination=np.array(
            [
                [[1.0, 2.0], [0.5, -1.0]],
                [[0.75, 0.25], [1.25, 0.5]],
            ]
        ),
        difficulty=np.array([[0.0, 0.25], [-0.5, 0.75]]),
    )
    theta = np.array([[0.0, 1.0], [1.0, 0.0], [-0.5, 0.5]])

    actual = sample_expected_scores(model, theta, samples)

    np.testing.assert_allclose(actual, _direct_expected_scores(model, theta, samples))
    np.testing.assert_allclose(
        sample_expected_scores(model, theta, samples, chunk_size=1), actual
    )


@pytest.mark.parametrize(
    ("model", "samples"),
    [
        (
            FourParameterLogistic(n_items=2),
            ParameterSamples(
                discrimination=np.array([[1.0, 1.5], [0.75, 1.25]]),
                difficulty=np.array([[0.0, 0.5], [-0.5, 0.25]]),
                guessing=np.array([[0.1, 0.2], [0.15, 0.05]]),
                upper=np.array([[0.9, 0.95], [0.85, 0.8]]),
            ),
        ),
        (
            FiveParameterLogistic(n_items=2),
            ParameterSamples(
                discrimination=np.array([[1.0, 1.5], [0.75, 1.25]]),
                difficulty=np.array([[0.0, 0.5], [-0.5, 0.25]]),
                guessing=np.array([[0.1, 0.2], [0.15, 0.05]]),
                upper=np.array([[0.9, 0.95], [0.85, 0.8]]),
                asymmetry=np.array([[0.8, 1.2], [1.5, 0.6]]),
            ),
        ),
    ],
)
def test_sample_expected_scores_matches_bounded_logistic_models(model, samples):
    theta = np.array([[-2.0], [0.0], [2.0]])

    actual = sample_expected_scores(model, theta, samples)

    np.testing.assert_allclose(actual, _direct_expected_scores(model, theta, samples))


def test_sample_expected_scores_is_stable_at_extreme_abilities():
    model = TwoParameterLogistic(n_items=1)
    samples = ParameterSamples(
        discrimination=np.ones((1, 1)),
        difficulty=np.zeros((1, 1)),
    )

    with np.errstate(over="raise", invalid="raise"):
        actual = sample_expected_scores(model, np.array([-1000.0, 1000.0]), samples)

    np.testing.assert_array_equal(actual, np.array([[0.0, 1.0]]))


def test_slipping_samples_define_the_upper_success_probability():
    model = TwoParameterLogistic(n_items=2)
    common = {
        "discrimination": np.ones((1, 2)),
        "difficulty": np.zeros((1, 2)),
    }
    slipping = ParameterSamples(**common, slipping=np.array([[0.1, 0.2]]))
    upper = ParameterSamples(**common, upper=np.array([[0.9, 0.8]]))

    np.testing.assert_allclose(
        sample_expected_scores(model, np.array([-1.0, 1.0]), slipping),
        sample_expected_scores(model, np.array([-1.0, 1.0]), upper),
    )


def test_missing_optional_samples_use_fixed_model_parameters():
    model = FourParameterLogistic(n_items=2)
    model.set_parameters(
        guessing=np.array([0.1, 0.2]),
        upper=np.array([0.9, 0.8]),
    )
    samples = ParameterSamples(
        discrimination=np.array([[1.25, 0.75], [0.5, 1.5]]),
        difficulty=np.array([[0.0, 0.5], [-0.5, 0.25]]),
    )
    theta = np.array([[-1.0], [0.0], [1.0]])

    np.testing.assert_allclose(
        sample_expected_scores(model, theta, samples),
        _direct_expected_scores(model, theta, samples),
    )


def test_draw_parameters_supports_bounded_and_asymmetric_models():
    four_pl = draw_parameters(FourParameterLogistic(3), n_samples=20, seed=42)
    five_pl = draw_parameters(FiveParameterLogistic(3), n_samples=20, seed=42)

    assert four_pl.guessing is not None
    assert four_pl.upper is not None
    assert four_pl.guessing.shape == (20, 3)
    assert four_pl.upper.shape == (20, 3)
    assert np.all(four_pl.guessing <= four_pl.upper)
    assert five_pl.asymmetry is not None
    assert five_pl.asymmetry.shape == (20, 3)
    assert np.all(five_pl.asymmetry > 0.0)


def test_draw_parameters_is_reproducible_for_multidimensional_models():
    model = TwoParameterLogistic(n_items=3, n_factors=2)

    first = draw_parameters(model, n_samples=8, seed=123)
    second = draw_parameters(model, n_samples=8, seed=123)

    assert first.discrimination.shape == (8, 3, 2)
    assert first.difficulty.shape == (8, 3)
    np.testing.assert_array_equal(first.discrimination, second.discrimination)
    np.testing.assert_array_equal(first.difficulty, second.difficulty)


def test_posterior_summary_includes_new_optional_parameters():
    samples = draw_parameters(FiveParameterLogistic(2), n_samples=20, seed=5)

    summary = posterior_summary(samples, credible_level=0.8)

    assert set(summary) == {
        "discrimination",
        "difficulty",
        "guessing",
        "upper",
        "asymmetry",
    }
    assert summary["upper"]["mean"].shape == (2,)
    assert summary["asymmetry"]["ci_lower"].shape == (2,)

    with pytest.raises(ValueError, match="credible_level"):
        posterior_summary(samples, credible_level=1.0)


def test_sampling_rejects_invalid_inputs():
    model = TwoParameterLogistic(n_items=2)
    valid = ParameterSamples(np.ones((2, 2)), np.zeros((2, 2)))

    with pytest.raises(ValueError, match="positive integer"):
        draw_parameters(model, n_samples=0)
    with pytest.raises(ValueError, match="method must be 'mvn'"):
        draw_parameters(model, method="bootstrap")
    with pytest.raises(ValueError, match="vcov must have shape"):
        draw_parameters(model, vcov=np.eye(3))
    with pytest.raises(ValueError, match="chunk_size"):
        sample_expected_scores(model, np.array([0.0]), valid, chunk_size=0)
    with pytest.raises(ValueError, match="logistic item model"):
        sample_expected_scores(GeneralizedPartialCredit(2, 3), np.array([0.0]), valid)

    conflicting = ParameterSamples(
        np.ones((1, 2)),
        np.zeros((1, 2)),
        slipping=np.full((1, 2), 0.1),
        upper=np.full((1, 2), 0.9),
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        sample_expected_scores(model, np.array([0.0]), conflicting)


def test_sampling_utilities_are_available_from_the_top_level_api():
    assert mirt.ParameterSamples is ParameterSamples
    assert mirt.posterior_summary is posterior_summary
    assert mirt.sample_expected_scores is sample_expected_scores
