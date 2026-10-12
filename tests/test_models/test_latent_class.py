"""Tests for unconstrained latent class analysis (R mirt's ``mdirt``)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

import mirt
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models.latent_class import LatentClassModel, LCAResult, fit_lca, mdirt
from mirt.utils.datasets import load_dataset

TIGHT = {"tol": 1e-10, "max_iter": 50_000}


@pytest.fixture(scope="module")
def lsat7() -> np.ndarray:
    return load_dataset("LSAT7")["data"]


@pytest.fixture(scope="module")
def bock_missing() -> np.ndarray:
    """Bock1997 nominal responses (0-3) with a deterministic 5% missing mask."""
    data = load_dataset("Bock1997")["data"].astype(np.float64)
    rows, items = np.indices(data.shape)
    data[(7 * rows + 3 * items) % 19 == 0] = np.nan
    return data


def _two_class_model() -> LatentClassModel:
    return LatentClassModel(
        [0.65, 0.35],
        [
            np.array([[0.1, 0.9], [0.8, 0.2]]),
            np.array([[0.2, 0.8], [0.7, 0.3]]),
            np.array([[0.1, 0.3, 0.6], [0.6, 0.3, 0.1]]),
            np.array([[0.15, 0.85], [0.9, 0.1]]),
            np.array([[0.05, 0.15, 0.3, 0.5], [0.5, 0.3, 0.15, 0.05]]),
        ],
    )


# R mirt 1.48 references:
#   mdirt(dat, K, TOL = 1e-10, technical = list(NCYCLES = 50000))
# with fscores() for the posteriors and the delta method on vcov(SE = TRUE)
# (Oakes) for the standard errors of the probabilities. R numbers its
# classes by its starting values; fit_lca orders them by size.


def test_lsat7_two_classes_match_r_mdirt(lsat7):
    result = fit_lca(lsat7, 2, **TIGHT)

    assert result.converged
    assert result.n_parameters == 11
    assert result.n_observations == 1000
    assert result.log_likelihood == pytest.approx(-2660.296827, abs=1e-5)
    assert result.aic == pytest.approx(5342.5937, abs=1e-3)
    assert result.bic == pytest.approx(5396.5790, abs=1e-3)
    assert result.sabic == pytest.approx(5361.6423, abs=1e-3)
    assert_allclose(result.class_proportions, [0.682206, 0.317794], atol=1e-4)
    r_correct = np.array(
        [
            [0.912937, 0.799582, 0.924304, 0.711053, 0.905569],
            [0.645667, 0.354068, 0.445050, 0.380483, 0.708683],
        ]
    )
    correct = np.column_stack([table[:, 1] for table in result.item_probabilities])
    assert_allclose(correct, r_correct, atol=1e-4)
    assert_allclose(result.class_posteriors[0], [0.003364, 0.996636], atol=1e-4)
    assert_allclose(result.class_posteriors[-1], [0.971425, 0.028575], atol=1e-4)

    assert_allclose(result.class_proportions_se, [0.058146, 0.058146], atol=1e-5)
    r_correct_se = np.array(
        [
            [0.018610, 0.028436, 0.023673, 0.024741, 0.016762],
            [0.041990, 0.052628, 0.064136, 0.048781, 0.036106],
        ]
    )
    assert result.item_probabilities_se is not None
    for item, table_se in enumerate(result.item_probabilities_se):
        # Both categories of a binary item share one standard error.
        assert_allclose(table_se[:, 0], r_correct_se[:, item], atol=1e-5)
        assert_allclose(table_se[:, 1], r_correct_se[:, item], atol=1e-5)


def test_lsat7_three_classes_reach_r_default_start_solution(lsat7):
    # The first start is R's deterministic mdirt start; on this nearly
    # saturated model it reaches R's local maximum, with two probabilities of
    # the smallest class on the boundary.
    result = fit_lca(lsat7, 3, **TIGHT)

    assert result.n_parameters == 17
    assert result.log_likelihood == pytest.approx(-2652.169543, abs=1e-3)
    assert_allclose(result.class_proportions, [0.587429, 0.381537, 0.031034], atol=1e-3)
    smallest_correct = [table[2, 1] for table in result.item_probabilities]
    assert_allclose(
        smallest_correct, [0.000021, 0.433733, 0.202182, 0.000019, 0.314194], atol=2e-3
    )
    assert result.item_probabilities_se is not None
    # Boundary estimates have no Wald standard error.
    assert np.isnan(result.item_probabilities_se[0][2, 1])
    assert np.all(np.isfinite(result.item_probabilities_se[1]))


def test_one_class_is_the_independence_model(lsat7):
    result = fit_lca(lsat7, 1)

    assert result.n_parameters == 5
    assert result.n_iterations <= 2
    assert result.log_likelihood == pytest.approx(-2743.41, abs=5e-3)  # R mdirt
    assert_allclose(
        [table[0, 1] for table in result.item_probabilities], lsat7.mean(axis=0)
    )
    assert np.isnan(result.entropy)
    assert_allclose(result.class_posteriors, 1.0)
    assert_allclose(result.class_proportions_se, [0.0])


def test_polytomous_items_with_missing_responses_match_r_mdirt(bock_missing):
    result = fit_lca(bock_missing, 2, **TIGHT)

    assert result.n_parameters == 31
    assert result.log_likelihood == pytest.approx(-6184.485811, abs=1e-5)
    assert result.aic == pytest.approx(12430.9716, abs=1e-3)
    assert result.bic == pytest.approx(12583.1120, abs=1e-3)
    assert result.sabic == pytest.approx(12484.6543, abs=1e-3)
    assert_allclose(result.class_proportions, [0.563196, 0.436804], atol=1e-4)
    r_first_item = np.array(
        [
            [0.544922, 0.300885, 0.105583, 0.048610],
            [0.194863, 0.216561, 0.296282, 0.292294],
        ]
    )
    r_last_item = np.array(
        [
            [0.556103, 0.241601, 0.132523, 0.069773],
            [0.200739, 0.230056, 0.278157, 0.291049],
        ]
    )
    assert_allclose(result.item_probabilities[0], r_first_item, atol=1e-4)
    assert_allclose(result.item_probabilities[-1], r_last_item, atol=1e-4)
    assert_allclose(
        result.class_posteriors[[0, 1, 2, -1]],
        [
            [0.367638, 0.632362],
            [0.918482, 0.081518],
            [0.333514, 0.666486],
            [0.069134, 0.930866],
        ],
        atol=2e-4,
    )

    assert_allclose(result.class_proportions_se, [0.055341, 0.055341], atol=2e-5)
    assert result.item_probabilities_se is not None
    assert_allclose(
        result.item_probabilities_se[0],
        [
            [0.031768, 0.023882, 0.022840, 0.015734],
            [0.032269, 0.026835, 0.028084, 0.031984],
        ],
        atol=2e-5,
    )
    assert_allclose(
        result.item_probabilities_se[-1],
        [
            [0.033922, 0.022253, 0.020694, 0.020092],
            [0.029817, 0.026073, 0.027494, 0.028616],
        ],
        atol=2e-5,
    )


def test_polytomous_three_classes_match_r_mdirt(bock_missing):
    result = fit_lca(bock_missing, 3, **TIGHT)

    assert result.n_parameters == 47
    assert result.log_likelihood == pytest.approx(-6163.868523, abs=1e-3)
    assert_allclose(result.class_proportions, [0.613949, 0.239362, 0.146689], atol=1e-3)
    assert_allclose(
        result.item_probabilities[0],
        [
            [0.341254, 0.284985, 0.248118, 0.125643],
            [0.697767, 0.280823, 0.004924, 0.016485],
            [0.103144, 0.148707, 0.242581, 0.505568],
        ],
        atol=1e-3,
    )


def test_smoothing_standard_errors_match_numerical_log_posterior_curvature(lsat7):
    smoothing = 1.0
    result = fit_lca(lsat7, 2, smoothing=smoothing, **TIGHT)
    proportions = result.class_proportions
    correct = np.array([table[:, 1] for table in result.item_probabilities])

    def unpack(theta: np.ndarray) -> LatentClassModel:
        class_logits = np.array([0.0, theta[0]])
        weights = np.exp(class_logits) / np.exp(class_logits).sum()
        probabilities = 1.0 / (1.0 + np.exp(-theta[1:].reshape(5, 2)))
        return LatentClassModel(
            weights, [np.column_stack([1.0 - p, p]) for p in probabilities]
        )

    def objective(theta: np.ndarray) -> float:
        model = unpack(theta)
        flat = np.concatenate(model.item_probabilities, axis=1)
        return model.log_likelihood(lsat7) + smoothing * np.sum(np.log(flat))

    estimate = np.concatenate(
        [
            [np.log(proportions[1] / proportions[0])],
            np.log(correct / (1 - correct)).ravel(),
        ]
    )
    step = 1e-4
    size = estimate.size
    hessian = np.empty((size, size))
    basis = np.eye(size) * step
    for row in range(size):
        for column in range(row, size):
            value = (
                objective(estimate + basis[row] + basis[column])
                - objective(estimate + basis[row] - basis[column])
                - objective(estimate - basis[row] + basis[column])
                + objective(estimate - basis[row] - basis[column])
            ) / (4 * step**2)
            hessian[row, column] = hessian[column, row] = value
    covariance = np.linalg.inv(-hessian)

    proportion_se = proportions[0] * proportions[1] * np.sqrt(covariance[0, 0])
    correct_se = (
        correct.ravel() * (1 - correct.ravel()) * np.sqrt(np.diag(covariance)[1:])
    )
    assert_allclose(result.class_proportions_se, proportion_se, rtol=1e-3)
    assert result.item_probabilities_se is not None
    fitted_se = np.array([table_se[:, 1] for table_se in result.item_probabilities_se])
    assert_allclose(fitted_se.ravel(), correct_se, rtol=1e-3)


def test_smoothing_gives_posterior_modes_away_from_the_boundary(lsat7):
    maximum_likelihood = fit_lca(lsat7, 3, **TIGHT)
    smoothed = fit_lca(lsat7, 3, smoothing=0.5, **TIGHT)

    flat = np.concatenate(smoothed.item_probabilities, axis=1)
    assert flat.min() > 1e-3
    assert smoothed.log_likelihood < maximum_likelihood.log_likelihood
    assert smoothed.item_probabilities_se is not None
    assert all(np.all(np.isfinite(se)) for se in smoothed.item_probabilities_se)

    # At the fixed point each estimate is (count + smoothing) / (total + C * smoothing).
    weights = smoothed.class_posteriors
    for item, table in enumerate(smoothed.item_probabilities):
        counts = np.stack([weights.T @ (lsat7[:, item] == c) for c in range(2)], axis=1)
        expected = (counts + 0.5) / (counts.sum(axis=1, keepdims=True) + 1.0)
        assert_allclose(table, expected, atol=1e-6)


def test_classes_are_ordered_by_decreasing_proportion_and_recovered():
    truth = LatentClassModel(
        [0.55, 0.3, 0.15],
        [
            np.array([[0.9, 0.1], [0.2, 0.8], [0.5, 0.5]]),
            np.array([[0.85, 0.15], [0.25, 0.75], [0.1, 0.9]]),
            np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
            np.array([[0.9, 0.1], [0.7, 0.3], [0.1, 0.9]]),
            np.array([[0.7, 0.2, 0.1], [0.2, 0.2, 0.6], [0.1, 0.8, 0.1]]),
            np.array([[0.8, 0.2], [0.15, 0.85], [0.85, 0.15]]),
        ],
    )
    responses, classes = truth.simulate(4000, seed=11)

    result = fit_lca(responses, 3, n_starts=5, seed=3)

    assert np.all(np.diff(result.class_proportions) <= 0.0)
    assert_allclose(result.class_proportions, truth.class_proportions, atol=0.03)
    assert result.item_probabilities_se is not None
    for fitted, true, se in zip(
        result.item_probabilities,
        truth.item_probabilities,
        result.item_probabilities_se,
        strict=True,
    ):
        # Within sampling error: every estimate lies within 4 standard errors.
        assert np.all(np.abs(fitted - true) < 4.0 * se)
    assert np.mean(result.classify() == classes) > 0.9
    assert 0.0 < result.entropy < 1.0


def test_relabelling_items_leaves_the_fit_unchanged(lsat7):
    result = fit_lca(lsat7, 2, **TIGHT)
    permuted = fit_lca(lsat7[:, ::-1], 2, **TIGHT)

    assert permuted.log_likelihood == pytest.approx(result.log_likelihood, abs=1e-6)
    assert_allclose(permuted.class_proportions, result.class_proportions, atol=1e-5)


def test_entropy_is_the_relative_classification_entropy(lsat7):
    result = fit_lca(lsat7, 2)
    posteriors = result.class_posteriors

    expected = 1.0 - np.sum(-posteriors * np.log(posteriors)) / (1000 * np.log(2))
    assert result.entropy == pytest.approx(expected)
    assert 0.0 < result.entropy < 1.0


def test_random_starts_select_the_best_log_likelihood_reproducibly(lsat7):
    single = fit_lca(lsat7, 3, max_iter=200)
    multi = fit_lca(lsat7, 3, max_iter=200, n_starts=6, seed=5)
    again = fit_lca(lsat7, 3, max_iter=200, n_starts=6, seed=5)

    assert multi.start_log_likelihoods.shape == (6,)
    assert multi.start_log_likelihoods[0] == pytest.approx(single.log_likelihood)
    assert multi.log_likelihood == pytest.approx(multi.start_log_likelihoods.max())
    assert multi.log_likelihood >= single.log_likelihood
    assert_allclose(again.start_log_likelihoods, multi.start_log_likelihoods)
    assert_allclose(again.class_posteriors, multi.class_posteriors)


def test_converged_is_false_when_max_iter_is_reached(lsat7):
    result = fit_lca(lsat7, 2, max_iter=3)

    assert not result.converged
    assert result.n_iterations == 3


def test_nan_and_negative_codes_are_equivalent_missing_responses():
    model = _two_class_model()
    responses, _ = model.simulate(600, seed=4)
    with_nan = responses.astype(np.float64)
    mask = np.random.default_rng(9).random(responses.shape) < 0.1
    with_nan[mask] = np.nan
    with_negative = responses.copy()
    with_negative[mask] = -9

    from_nan = fit_lca(with_nan, 2)
    from_negative = fit_lca(with_negative, 2)

    assert from_nan.log_likelihood == pytest.approx(from_negative.log_likelihood)
    assert_allclose(from_nan.class_posteriors, from_negative.class_posteriors)


def test_rows_without_responses_carry_no_information(lsat7):
    blank = np.vstack([lsat7, np.full((1, 5), -1)])

    result = fit_lca(lsat7, 2)
    padded = fit_lca(blank, 2)

    assert padded.n_observations == 1000  # excluded from n, as in R mdirt
    assert padded.log_likelihood == pytest.approx(result.log_likelihood)
    assert padded.bic == pytest.approx(result.bic)
    assert padded.entropy == pytest.approx(result.entropy)
    assert_allclose(padded.class_posteriors[-1], padded.class_proportions)


def test_posterior_and_classify_score_new_responses():
    responses, _ = _two_class_model().simulate(800, seed=2)
    result = fit_lca(responses, 2, compute_standard_errors=False)

    assert_allclose(result.posterior(responses), result.class_posteriors)
    assert_allclose(result.posterior(), result.class_posteriors)
    assert_allclose(result.class_posteriors.sum(axis=1), 1.0)
    assert np.array_equal(result.classify(), np.argmax(result.class_posteriors, axis=1))

    new = np.array([[1, 1, 2, 1, 3], [0, 0, 0, 0, 0], [-1, -1, -1, -1, -1]])
    posterior = result.posterior(new)
    assert np.array_equal(result.classify(new), np.argmax(posterior, axis=1))
    assert_allclose(posterior[2], result.class_proportions)
    model = result.model
    joint = model.class_log_likelihoods(new) + np.log(model.class_proportions)
    expected = np.exp(joint - joint.max(axis=1, keepdims=True))
    assert_allclose(posterior, expected / expected.sum(axis=1, keepdims=True))

    with pytest.raises(MirtDataError, match="categories"):
        result.posterior(np.array([[1, 1, 3, 1, 3]]))


def test_dataframe_columns_name_items_and_align_new_data():
    pd = pytest.importorskip("pandas")
    responses, _ = _two_class_model().simulate(500, seed=8)
    columns = ["a", "b", "c", "d", "e"]
    frame = pd.DataFrame(responses, columns=columns)

    result = fit_lca(frame, 2, compute_standard_errors=False)

    assert result.model.item_names == columns
    assert result.log_likelihood == pytest.approx(
        fit_lca(responses, 2, compute_standard_errors=False).log_likelihood
    )
    reordered = frame[columns[::-1]].assign(id=np.arange(500))
    assert_allclose(result.posterior(reordered), result.class_posteriors)
    assert "a = 0" in result.summary()


def test_parameter_count_follows_the_category_counts():
    responses, _ = _two_class_model().simulate(300, seed=1)

    result = fit_lca(responses, 3, max_iter=50, compute_standard_errors=False)

    # Categories 2, 2, 3, 2 and 4: K - 1 + K * sum(C_j - 1) = 2 + 3 * 8.
    assert result.n_parameters == 26
    assert result.model.n_parameters == 26
    assert list(result.model.n_categories) == [2, 2, 3, 2, 4]


def test_declared_unused_categories_get_zero_probability():
    responses, _ = _two_class_model().simulate(300, seed=5)

    result = fit_lca(responses, 2, n_categories=[2, 2, 3, 3, 4])

    assert result.n_parameters == 1 + 2 * 9
    assert_allclose(result.item_probabilities[3][:, 2], 0.0, atol=1e-12)
    assert result.item_probabilities_se is not None
    assert np.all(np.isnan(result.item_probabilities_se[3][:, 2]))
    with pytest.raises(MirtDataError, match="categories"):
        fit_lca(responses, 2, n_categories=3)


def test_unidentified_model_warns_and_reports_nan_standard_errors(lsat7):
    # Three binary items have 7 pattern degrees of freedom; three classes
    # need 11 parameters.
    with pytest.warns(RuntimeWarning, match="singular"):
        result = fit_lca(lsat7[:, :3], 3, n_starts=3, seed=0, **TIGHT)

    assert result.class_proportions_se is not None
    assert np.all(np.isnan(result.class_proportions_se))


def test_standard_errors_can_be_skipped(lsat7):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = fit_lca(lsat7[:, :3], 3, compute_standard_errors=False)

    assert result.class_proportions_se is None
    assert result.item_probabilities_se is None


def test_summary_reports_fit_and_profiles(lsat7):
    text = fit_lca(lsat7, 2).summary()

    for label in ("Latent Class Analysis", "SABIC", "Entropy", "Class 1", "Item_5 = 1"):
        assert label in text


def test_compare_models_ranks_class_solutions(lsat7):
    pytest.importorskip("pandas")
    results = [fit_lca(lsat7, k, compute_standard_errors=False) for k in (1, 2, 3)]

    table = mirt.compare_models(results, criteria=["AIC", "BIC", "SABIC"])

    assert_allclose(np.asarray(table["BIC"]), [r.bic for r in results])
    assert_allclose(np.asarray(table["SABIC"]), [r.sabic for r in results])
    assert list(table["Model"])[1] == "Model 2: LCA(2)"


def test_mdirt_alias_and_public_exports(lsat7):
    assert mirt.fit_lca is fit_lca
    assert mirt.mdirt is mdirt
    assert mirt.models.LCAResult is LCAResult
    assert mirt.LatentClassModel is LatentClassModel

    result = mdirt(lsat7, 2, n_starts=2, seed=1)
    reference = fit_lca(lsat7, 2, n_starts=2, seed=1)
    assert isinstance(result, LCAResult)
    assert result.log_likelihood == reference.log_likelihood


def test_simulate_is_reproducible_and_follows_the_probabilities():
    model = _two_class_model()

    responses, classes = model.simulate(20_000, seed=3)
    again, _ = model.simulate(20_000, seed=3)

    assert np.array_equal(responses, again)
    assert np.mean(classes == 0) == pytest.approx(0.65, abs=0.02)
    observed = np.bincount(responses[classes == 1, 4], minlength=4) / np.sum(
        classes == 1
    )
    assert_allclose(observed, model.item_probabilities[4][1], atol=0.02)


@pytest.mark.parametrize(
    ("kwargs", "parameter"),
    [
        ({"n_classes": 0}, "n_classes"),
        ({"n_classes": True}, "n_classes"),
        ({"n_classes": 2.0}, "n_classes"),
        ({"n_classes": 2, "tol": 0.0}, "tol"),
        ({"n_classes": 2, "max_iter": 0}, "max_iter"),
        ({"n_classes": 2, "n_starts": 0}, "n_starts"),
        ({"n_classes": 2, "seed": -1}, "seed"),
        ({"n_classes": 2, "smoothing": -0.5}, "smoothing"),
        ({"n_classes": 2, "smoothing": float("nan")}, "smoothing"),
        ({"n_classes": 2, "verbose": 1}, "verbose"),
        ({"n_classes": 2, "n_categories": 1}, "n_categories"),
        ({"n_classes": 2, "item_names": ["a", "a", "b", "c", "d"]}, "item_names"),
    ],
)
def test_invalid_controls_raise_validation_errors(lsat7, kwargs, parameter):
    with pytest.raises(MirtValidationError) as error:
        fit_lca(lsat7, **kwargs)
    assert parameter in str(error.value)


def test_invalid_data_raise_data_errors(lsat7):
    unobserved_item = lsat7.astype(np.float64)
    unobserved_item[:, 2] = np.nan
    with pytest.raises(MirtDataError, match="Item_3"):
        fit_lca(unobserved_item, 2)
    with pytest.raises(MirtDataError):
        fit_lca(lsat7[:, 0], 2)
    with pytest.raises(MirtDataError):
        fit_lca(lsat7 + 0.5, 2)


def test_model_rejects_inconsistent_parameters():
    with pytest.raises(MirtValidationError, match="sum to one"):
        LatentClassModel([0.5, 0.6], [np.array([[0.5, 0.5], [0.5, 0.5]])])
    with pytest.raises(MirtValidationError, match="shape"):
        LatentClassModel([0.5, 0.5], [np.array([[0.5, 0.5]])])
    with pytest.raises(MirtValidationError, match="sum to one"):
        LatentClassModel([1.0], [np.array([[0.5, 0.6]])])
    with pytest.raises(MirtValidationError, match="item_names"):
        LatentClassModel([1.0], [np.array([[0.5, 0.5]])], item_names=["a", "b"])
