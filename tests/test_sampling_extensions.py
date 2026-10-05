"""Regression coverage for parameter sampling and score propagation."""

import numpy as np
import pytest

import mirt
import mirt.utils.sampling as sampling_utils
from mirt.exceptions import MirtValidationError
from mirt.models.dichotomous import (
    FiveParameterLogistic,
    FourParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.polytomous import GeneralizedPartialCredit
from mirt.results import FitResult
from mirt.utils.sampling import (
    ParameterSamples,
    draw_parameters,
    posterior_summary,
    sample_expected_scores,
)


def _direct_expected_scores(model, theta, samples, item_idx=None):
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
        probabilities = sampled_model.probability(theta, item_idx=item_idx)
        expected[sample_idx] = (
            probabilities.sum(axis=1) if item_idx is None else probabilities
        )
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
    np.testing.assert_allclose(
        sample_expected_scores(model, theta, samples, item_idx=1),
        _direct_expected_scores(model, theta, samples, item_idx=1),
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


def test_sample_expected_scores_can_target_one_item_without_full_bank_logits(
    monkeypatch,
):
    model = FiveParameterLogistic(n_items=3)
    samples = ParameterSamples(
        discrimination=np.array([[1.0, 1.5, 0.5], [0.75, 1.25, 2.0]]),
        difficulty=np.array([[0.0, 0.5, -0.5], [-0.5, 0.25, 1.0]]),
        guessing=np.array([[0.1, 0.2, 0.05], [0.15, 0.05, 0.1]]),
        upper=np.array([[0.9, 0.95, 0.8], [0.85, 0.8, 0.9]]),
        asymmetry=np.array([[0.8, 1.2, 1.0], [1.5, 0.6, 0.9]]),
    )
    theta = np.array([[-2.0], [0.0], [2.0]])
    logit_shapes = []
    sigmoid = sampling_utils._sigmoid_inplace

    def tracked_sigmoid(values):
        logit_shapes.append(values.shape)
        return sigmoid(values)

    monkeypatch.setattr(sampling_utils, "_sigmoid_inplace", tracked_sigmoid)

    actual = sample_expected_scores(model, theta, samples, item_idx=1)

    np.testing.assert_allclose(
        actual,
        _direct_expected_scores(model, theta, samples, item_idx=1),
    )
    assert logit_shapes
    assert all(shape[2] == 1 for shape in logit_shapes)


def test_sample_expected_scores_can_target_a_subtest():
    model = TwoParameterLogistic(n_items=5)
    samples = ParameterSamples(
        discrimination=np.array(
            [[1.0, 1.5, 0.5, 0.8, 1.2], [0.75, 1.25, 2.0, 1.1, 0.6]]
        ),
        difficulty=np.array(
            [[0.0, 0.5, -0.5, 1.0, -1.0], [-0.5, 0.25, 1.0, 0.2, -0.8]]
        ),
    )
    theta = np.array([[-2.0], [0.0], [2.0]])
    selected = np.array([4, 1, 3])

    actual = sample_expected_scores(model, theta, samples, item_indices=selected)
    expected = sum(
        sample_expected_scores(model, theta, samples, item_idx=int(index))
        for index in selected
    )

    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(
        sample_expected_scores(
            model,
            theta,
            samples,
            item_indices=selected,
            chunk_size=1,
        ),
        expected,
    )


def test_sample_expected_scores_subtest_supports_bounded_models():
    model = FiveParameterLogistic(n_items=4)
    samples = ParameterSamples(
        discrimination=np.array(
            [
                [1.0, 0.5, 0.8, 1.3],
                [0.7, 0.9, 1.1, 0.6],
            ]
        ),
        difficulty=np.array([[0.0, 0.5, -0.5, 1.0], [-0.5, 0.25, 1.0, -0.2]]),
        guessing=np.array([[0.1, 0.2, 0.05, 0.15], [0.15, 0.05, 0.1, 0.2]]),
        upper=np.array([[0.9, 0.95, 0.8, 0.85], [0.85, 0.8, 0.9, 0.95]]),
        asymmetry=np.array([[0.8, 1.2, 1.0, 0.7], [1.5, 0.6, 0.9, 1.1]]),
    )
    theta = np.array([[-1.0], [0.0], [1.0]])

    actual = sample_expected_scores(model, theta, samples, item_indices=[0, 2])
    expected = sample_expected_scores(model, theta, samples, item_idx=0)
    expected += sample_expected_scores(model, theta, samples, item_idx=2)

    np.testing.assert_allclose(actual, expected)


def test_sample_expected_scores_subtest_uses_fixed_asymptotes():
    model = FourParameterLogistic(n_items=4)
    model.set_parameters(
        guessing=np.array([0.1, 0.2, 0.05, 0.15]),
        upper=np.array([0.9, 0.95, 0.8, 0.85]),
    )
    samples = ParameterSamples(
        discrimination=np.array([[1.0, 0.5, 0.8, 1.3], [0.7, 0.9, 1.1, 0.6]]),
        difficulty=np.array([[0.0, 0.5, -0.5, 1.0], [-0.5, 0.25, 1.0, -0.2]]),
    )
    theta = np.array([[-1.0], [0.0], [1.0]])

    actual = sample_expected_scores(model, theta, samples, item_indices=[3, 0])
    expected = sample_expected_scores(model, theta, samples, item_idx=3)
    expected += sample_expected_scores(model, theta, samples, item_idx=0)

    np.testing.assert_allclose(actual, expected)


def test_sample_expected_scores_subtest_supports_multidimensional_models():
    model = TwoParameterLogistic(n_items=4, n_factors=2)
    samples = ParameterSamples(
        discrimination=np.array(
            [
                [[1.0, 0.5], [0.2, 1.2], [0.8, 0.7], [1.3, 0.1]],
                [[0.7, 0.9], [1.1, 0.3], [0.4, 1.4], [0.6, 0.8]],
            ]
        ),
        difficulty=np.array([[0.0, 0.5, -0.5, 1.0], [-0.5, 0.25, 1.0, -0.2]]),
    )
    theta = np.array([[-1.0, 0.5], [0.0, 0.0], [1.0, -0.5]])

    actual = sample_expected_scores(model, theta, samples, item_indices=[1, 3])
    expected = sample_expected_scores(model, theta, samples, item_idx=1)
    expected += sample_expected_scores(model, theta, samples, item_idx=3)

    np.testing.assert_allclose(actual, expected)


def test_targeted_scores_do_not_allocate_full_bank_asymptote_arrays(monkeypatch):
    model = TwoParameterLogistic(n_items=40)
    samples = ParameterSamples(np.ones((6, 40)), np.zeros((6, 40)))
    observed_shapes = []
    original_zeros = sampling_utils.np.zeros
    original_ones = sampling_utils.np.ones

    def tracked_zeros(shape, *args, **kwargs):
        observed_shapes.append(tuple(shape))
        return original_zeros(shape, *args, **kwargs)

    def tracked_ones(shape, *args, **kwargs):
        observed_shapes.append(tuple(shape))
        return original_ones(shape, *args, **kwargs)

    monkeypatch.setattr(sampling_utils.np, "zeros", tracked_zeros)
    monkeypatch.setattr(sampling_utils.np, "ones", tracked_ones)

    sample_expected_scores(
        model,
        np.array([-1.0, 0.0, 1.0]),
        samples,
        item_idx=2,
    )

    assert (6, 1) in observed_shapes
    assert (6, 40) not in observed_shapes
    observed_shapes.clear()

    sample_expected_scores(
        model,
        np.array([-1.0, 0.0, 1.0]),
        samples,
        item_indices=[2, 8, 21],
    )

    assert (6, 3) in observed_shapes
    assert (6, 40) not in observed_shapes


@pytest.mark.parametrize("item_idx", [-1, 3, True, 1.5])
def test_sample_expected_scores_rejects_invalid_item_indices(item_idx):
    model = TwoParameterLogistic(n_items=3)
    samples = ParameterSamples(np.ones((2, 3)), np.zeros((2, 3)))

    with pytest.raises(IndexError, match="item_idx"):
        sample_expected_scores(model, np.array([0.0]), samples, item_idx=item_idx)


@pytest.mark.parametrize(
    ("item_indices", "error", "message"),
    [
        ([], ValueError, "non-empty"),
        ([[0, 1]], ValueError, "one-dimensional"),
        ([0.0, 1.0], ValueError, "integers"),
        ([True, False], ValueError, "integers"),
        ([0, 0], ValueError, "duplicates"),
        ([-1, 0], IndexError, "values"),
        ([0, 3], IndexError, "values"),
    ],
)
def test_sample_expected_scores_rejects_invalid_subtests(item_indices, error, message):
    model = TwoParameterLogistic(n_items=3)
    samples = ParameterSamples(np.ones((2, 3)), np.zeros((2, 3)))

    with pytest.raises(error, match=message):
        sample_expected_scores(
            model,
            np.array([0.0]),
            samples,
            item_indices=item_indices,
        )


def test_sample_expected_scores_rejects_conflicting_item_selections():
    model = TwoParameterLogistic(n_items=3)
    samples = ParameterSamples(np.ones((2, 3)), np.zeros((2, 3)))

    with pytest.raises(ValueError, match="mutually exclusive"):
        sample_expected_scores(
            model,
            np.array([0.0]),
            samples,
            item_idx=0,
            item_indices=[1, 2],
        )


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
    with pytest.warns(FutureWarning, match="without a covariance"):
        four_pl = draw_parameters(FourParameterLogistic(3), n_samples=20, seed=42)
    with pytest.warns(FutureWarning, match="without a covariance"):
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

    vcov = np.eye(9) * 0.02
    first = draw_parameters(model, n_samples=8, vcov=vcov, seed=123)
    second = draw_parameters(model, n_samples=8, vcov=vcov, seed=123)

    assert first.discrimination.shape == (8, 3, 2)
    assert first.difficulty.shape == (8, 3)
    np.testing.assert_array_equal(first.discrimination, second.discrimination)
    np.testing.assert_array_equal(first.difficulty, second.difficulty)


def test_posterior_summary_includes_new_optional_parameters():
    with pytest.warns(FutureWarning):
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


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        (
            ParameterSamples(np.empty((0, 2)), np.empty((0, 2))),
            "at least one draw",
        ),
        (
            ParameterSamples(np.array([[1.0, np.nan]]), np.zeros((1, 2))),
            "discrimination must contain only finite",
        ),
        (
            ParameterSamples(np.ones((3, 2)), np.zeros((3, 1))),
            "difficulty must have shape",
        ),
        (
            ParameterSamples(
                np.ones((3, 2)),
                np.zeros((3, 2)),
                guessing=np.ones((1, 7)),
            ),
            "guessing must have shape",
        ),
        (
            ParameterSamples(
                np.ones((3, 2)),
                np.zeros((3, 2)),
                upper=np.array([[1.0, np.inf], [1.0, 1.0], [1.0, 1.0]]),
            ),
            "upper must contain only finite",
        ),
    ],
)
def test_posterior_summary_rejects_malformed_sample_containers(samples, message):
    with pytest.raises(ValueError, match=message):
        posterior_summary(samples)


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


@pytest.fixture(scope="module")
def two_pl_result():
    data = mirt.simdata(model="2PL", n_persons=800, n_items=4, seed=21)
    return mirt.fit_mirt(data, model="2PL", tol=1e-7)


def test_draw_parameters_uses_the_fit_covariance(two_pl_result):
    result = two_pl_result
    samples = draw_parameters(result, n_samples=40_000, seed=3)

    np.testing.assert_allclose(
        samples.discrimination.std(axis=0),
        result.standard_errors["discrimination"],
        rtol=0.05,
    )
    np.testing.assert_allclose(
        samples.difficulty.std(axis=0), result.standard_errors["difficulty"], rtol=0.05
    )
    sampled = np.corrcoef(samples.discrimination[:, 0], samples.difficulty[:, 0])[0, 1]
    vcov = result.vcov
    expected = vcov[0, 4] / np.sqrt(vcov[0, 0] * vcov[4, 4])
    assert abs(expected) > 0.1
    assert sampled == pytest.approx(expected, abs=0.03)


def test_draw_parameters_warns_when_only_standard_errors_exist(two_pl_result):
    result = FitResult(
        model=two_pl_result.model,
        log_likelihood=two_pl_result.log_likelihood,
        n_iterations=1,
        converged=True,
        standard_errors=two_pl_result.standard_errors,
        aic=0.0,
        bic=0.0,
    )
    with pytest.warns(UserWarning, match="independently"):
        samples = draw_parameters(result, n_samples=20_000, seed=9)
    np.testing.assert_allclose(
        samples.difficulty.std(axis=0), result.standard_errors["difficulty"], rtol=0.05
    )
    assert (
        abs(np.corrcoef(samples.discrimination[:, 0], samples.difficulty[:, 0])[0, 1])
        < 0.03
    )

    empty = FitResult(result.model, -1.0, 1, True, {}, 0.0, 0.0)
    with pytest.raises(MirtValidationError, match="no parameter covariance"):
        draw_parameters(empty)


def test_draw_parameters_samples_asymptotes_from_their_uncertainty():
    model = ThreeParameterLogistic(3, item_names=["a", "b", "c"]).set_parameters(
        discrimination=np.array([1.0, 1.4, 0.8]),
        difficulty=np.array([-0.5, 0.2, 0.9]),
        guessing=np.array([0.0, 0.2, 0.15]),
    )
    # The first guessing coordinate sits on its lower bound.
    variances = np.array([0.04, 0.05, 0.03, 0.02, 0.03, 0.04, np.nan, 0.004, 0.009])
    covariance = np.diag(variances)
    covariance[6, :] = covariance[:, 6] = np.nan
    result = FitResult(
        model, -1.0, 1, True, {}, 0.0, 0.0, se_method="oakes", vcov=covariance
    )

    samples = draw_parameters(result, n_samples=20_000, seed=4)

    np.testing.assert_array_equal(samples.guessing[:, 0], 0.0)
    np.testing.assert_allclose(samples.guessing[:, 1].std(), np.sqrt(0.004), rtol=0.05)
    np.testing.assert_allclose(samples.guessing[:, 2].std(), np.sqrt(0.009), rtol=0.06)
    np.testing.assert_allclose(
        samples.difficulty.std(axis=0), np.sqrt(variances[3:6]), rtol=0.05
    )

    core = draw_parameters(model, n_samples=50, vcov=np.eye(6) * 0.01, seed=1)
    np.testing.assert_array_equal(core.guessing, np.tile(model.guessing, (50, 1)))
    joint = draw_parameters(model, n_samples=2_000, vcov=np.eye(9) * 0.0004, seed=1)
    assert joint.guessing[:, 1].std() == pytest.approx(0.02, rel=0.1)
    with pytest.raises(ValueError, match=r"\(6, 6\) or \(9, 9\)"):
        draw_parameters(model, vcov=np.eye(7))


def test_explicit_vcov_holds_all_nan_rows_like_a_fit_result():
    # Regression: draw_parameters(model, vcov=result.vcov) rejected the NaN
    # rows that mark parameters on an optimizer bound.
    model = ThreeParameterLogistic(3).set_parameters(
        discrimination=np.array([1.0, 1.4, 0.8]),
        difficulty=np.array([-0.5, 0.2, 0.9]),
        guessing=np.array([0.0, 0.2, 0.15]),
    )
    covariance = np.diag([0.04, 0.05, 0.03, 0.02, 0.03, 0.04, np.nan, 0.004, 0.009])
    covariance[0, 3] = covariance[3, 0] = 0.01
    covariance[6, :] = covariance[:, 6] = np.nan
    result = FitResult(model, -1.0, 1, True, {}, 0.0, 0.0, vcov=covariance)

    explicit = draw_parameters(model, n_samples=200, vcov=covariance, seed=5)
    fitted = draw_parameters(result, n_samples=200, seed=5)
    for name in ("discrimination", "difficulty", "guessing"):
        np.testing.assert_array_equal(getattr(explicit, name), getattr(fitted, name))
    np.testing.assert_array_equal(explicit.guessing[:, 0], 0.0)

    covariance[0, 1] = covariance[1, 0] = np.nan
    with pytest.raises(ValueError, match="only finite values"):
        draw_parameters(model, vcov=covariance)


def test_fit_result_draws_require_stored_item_parameters():
    from mirt.models.explanatory import LLTM

    # Regression: an LLTM derives difficulty from its feature weights, and
    # drawing from its fit result raised a bare KeyError.
    features = np.column_stack([np.ones(4), np.linspace(-1.0, 1.0, 4)])
    result = FitResult(LLTM(4, features), -1.0, 1, True, {}, 0.0, 0.0)
    with pytest.raises(MirtValidationError, match="stored discrimination and diff"):
        draw_parameters(result)


def test_sampling_utilities_are_available_from_the_top_level_api():
    assert mirt.ParameterSamples is ParameterSamples
    assert mirt.posterior_summary is posterior_summary
    assert mirt.sample_expected_scores is sample_expected_scores
