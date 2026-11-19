"""Held-out likelihoods use the fit's population and integration rule."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import logsumexp

from mirt.estimation.bifactor_em import BifactorEMEstimator
from mirt.estimation.quadrature import GaussHermiteQuadrature, build_quadrature
from mirt.models.bifactor import BifactorModel, GradedBifactorModel
from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.testlet import RandomTestletEffectsModel, TestletModel
from mirt.results.fit_result import FitResult
from mirt.utils.cv import LogLikelihoodScorer, PluginLogLikelihoodScorer


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint16, np.uint64])
def test_compressed_numeric_rows_preserve_exact_values_and_frequencies(dtype):
    from mirt.estimation._patterns import compress_responses

    if np.issubdtype(dtype, np.floating):
        codes = [0.25, 0.75, -3.0, np.nan]
    else:
        codes = [0, 1, np.iinfo(dtype).max - 1, np.iinfo(dtype).max]
    unique = np.array([codes, codes[::-1]], dtype=dtype)
    repeated = np.repeat(unique, [190, 110], axis=0)[:, ::-1]
    patterns, counts = compress_responses(repeated)

    assert patterns.dtype == dtype
    np.testing.assert_array_equal(patterns, unique[:, ::-1])
    np.testing.assert_array_equal(counts, [190, 110])


@pytest.mark.parametrize("rule", ["rectangular", "gauss-hermite"])
@pytest.mark.parametrize("family", ["binary", "graded", "testlet", "random-testlet"])
def test_bifactor_scores_equal_full_product_quadrature(family, rule):
    groups = [2, 0, 2, 0, 2, 0]
    if family == "graded":
        model = GradedBifactorModel(6, groups, [2, 4, 3, 2, 3, 4])
    elif family == "testlet":
        model = TestletModel(6, [-1, 0, 2, 0, 2, 0])
    elif family == "random-testlet":
        model = RandomTestletEffectsModel(6, [-1, 0, 2, 0, 2, 0])
    else:
        model = BifactorModel(6, groups)
    rng = np.random.default_rng(140)
    responses = model.simulate(rng.normal(size=(90, 3)), seed=141)
    result = BifactorEMEstimator(
        n_quadpts=7, quadrature=rule, max_iter=2, compute_standard_errors=False
    ).fit(model, responses[:60])

    test = responses[60:].astype(float)
    test[::3, 1] = np.nan
    test[0] = -1
    if family in ("binary", "testlet"):
        # These models accept fractional responses in likelihood evaluation.
        test[1, 0] = 0.25
    grid = build_quadrature(7, 3, rule)
    expected = logsumexp(
        model.log_likelihood_batch(test, grid.nodes) + np.log(grid.weights), axis=1
    ).sum()
    scorer = LogLikelihoodScorer()
    assert scorer(result, responses[:60], test) == pytest.approx(expected, abs=1e-10)
    assert scorer(result, responses[:60], responses[:60]) == pytest.approx(
        result.log_likelihood, abs=1e-10
    )


def test_many_specific_factors_score_without_product_grid(monkeypatch):
    from mirt.estimation import quadrature

    groups = np.repeat(np.arange(7), 2)
    model = BifactorModel(len(groups), groups)
    responses = np.random.default_rng(3).integers(0, 2, (20, len(groups)))
    result = BifactorEMEstimator(
        n_quadpts=21, max_iter=1, compute_standard_errors=False
    ).fit(model, responses)

    def no_product_grid(*args, **kwargs):
        pytest.fail("built a full product grid for a reduced fit")

    monkeypatch.setattr(quadrature, "build_quadrature", no_product_grid)
    scorer = LogLikelihoodScorer()
    restored = FitResult.from_dict(result.to_dict())
    for fit in (
        result,
        restored,
        replace(restored, latent_mean=np.zeros(8), latent_covariance=np.eye(8)),
    ):
        score = scorer(fit, responses, np.tile(responses, (20, 1)))
        assert score == pytest.approx(20 * result.log_likelihood, rel=1e-12)


@pytest.mark.parametrize("model_class", [BifactorModel, TestletModel])
def test_replaced_model_likelihood_is_used_after_fitting(monkeypatch, model_class):
    responses = np.random.default_rng(4).integers(0, 2, (40, 4))
    model = model_class(4, [0, 0, 1, 1])
    result = BifactorEMEstimator(
        n_quadpts=5, max_iter=1, compute_standard_errors=False
    ).fit(model, responses)
    monkeypatch.setattr(
        model_class,
        "log_likelihood_batch",
        lambda self, values, nodes: np.full((len(values), len(nodes)), -3.5),
    )
    assert LogLikelihoodScorer()(result, responses, responses) == pytest.approx(-140)


def test_high_dimensional_recipe_free_score_is_reproducible():
    model = TwoParameterLogistic(1, n_factors=7)
    model.set_parameters(discrimination=np.ones((1, 7)), difficulty=np.zeros(1))
    fit = FitResult(model, 0.0, 1, True, {}, 0.0, 0.0)
    responses = np.array([[1], [0], [-1]])
    scorer = LogLikelihoodScorer()
    score = scorer(fit, responses, responses)
    assert score == scorer(fit, responses, responses)
    # Symmetry gives each observed category probability 1/2.
    assert score == pytest.approx(-2 * np.log(2), abs=1e-4)


def test_recipe_free_gaussian_grid_follows_stored_population():
    model = TwoParameterLogistic(2)
    mean = np.array([28.0])
    covariance = np.array([[0.16]])
    model.set_parameters(
        discrimination=np.array([1.0, 1.2]), difficulty=mean + [-0.5, 0.5]
    )
    fit = FitResult(model, 0.0, 1, True, {}, 0.0, 0.0)
    fit = replace(fit, latent_mean=mean, latent_covariance=covariance)
    responses = np.array([[1, 0], [1, 1], [0, 0]])
    grid = GaussHermiteQuadrature(41, 1, mean=mean, cov=covariance)
    expected = logsumexp(
        model.log_likelihood_batch(responses, grid.nodes) + np.log(grid.weights), axis=1
    ).sum()
    assert LogLikelihoodScorer()(fit, responses, responses.tolist()) == pytest.approx(
        expected, abs=1e-10
    )


def test_recipe_free_score_accepts_fit_covariance_with_roundoff():
    model = TwoParameterLogistic(3, n_factors=2)
    model._is_fitted = True
    fit = FitResult(model, 0.0, 1, True, {}, 0.0, 0.0)
    covariance = np.array([[1.0, 1e-14], [0.0, 1.0]])
    rounded = replace(fit, latent_covariance=covariance)
    responses = np.array([[1, 0, 1], [0, 1, 0]])
    for scorer in (LogLikelihoodScorer(), PluginLogLikelihoodScorer()):
        assert scorer(rounded, responses, responses) == pytest.approx(
            scorer(fit, responses, responses), abs=1e-12
        )


def test_named_nullable_held_out_responses_follow_model_item_order():
    pd = pytest.importorskip("pandas")
    model = TwoParameterLogistic(2, item_names=["alpha", "beta"])
    model.set_parameters(
        discrimination=np.array([1.0, 2.0]), difficulty=np.array([-1.0, 1.0])
    )
    fit = FitResult(model, 0.0, 1, True, {}, 0.0, 0.0)
    responses = np.array([[1, 0], [1, -1], [0, 1]])
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "beta": pd.array([0, None, 1], dtype="Int64"),
            "alpha": [1, 1, 0],
        }
    )
    scorer = LogLikelihoodScorer()
    assert scorer(fit, responses, frame) == pytest.approx(
        scorer(fit, responses, responses)
    )
