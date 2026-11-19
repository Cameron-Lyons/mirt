from dataclasses import replace

import numpy as np
import pytest

import mirt
from mirt.results.score_result import ScoreResult
from mirt.utils.cv import (
    AbilityRMSEScorer,
    KFold,
    PluginLogLikelihoodScorer,
    cross_validate,
)


def _score_rmse(truth, estimates, indices=None):
    scores = ScoreResult(estimates, np.ones_like(estimates), method="EAP")
    return AbilityRMSEScorer(truth).score_from_ability_result(
        None, None, None, scores, indices
    )


@pytest.mark.parametrize("column_estimate", [False, True])
def test_column_vector_truth_compares_matching_persons(column_estimate):
    truth = np.array([[1.5], [0.0], [4.5], [-2.0]])
    estimates = np.array([0.5, 2.0, 1.5])
    if column_estimate:
        estimates = estimates[:, None]
    assert _score_rmse(truth, estimates) == pytest.approx(-np.sqrt(14 / 3))


def test_multidimensional_rmse_pairs_factors_within_selected_persons():
    truth = np.array([[1.5, 2.0], [2.5, -1.0], [3.0, -0.5], [4.5, 1.5]])
    indices = np.array([3, 0, 2])
    estimates = truth[indices] + [[1, 2], [0, 0], [-1, 3]]
    assert _score_rmse(truth, estimates, indices) == pytest.approx(-np.sqrt(15 / 6))


@pytest.mark.parametrize("truth", [np.ones(3), np.ones((1, 2)), np.ones((3, 1, 2))])
def test_incompatible_truth_is_rejected_before_subtraction(truth):
    with pytest.raises(ValueError, match="true_theta"):
        _score_rmse(truth, np.ones((3, 2)))


def test_standalone_and_shared_eap_scores_use_the_fitted_population(monkeypatch):
    responses = np.tile([[1, 0], [0, 1], [0, 0], [1, 1]], (3, 1))
    fit = replace(
        mirt.fit_mirt(responses, model="1PL", n_quadpts=9, max_iter=2),
        latent_mean=np.array([1.5]),
        latent_covariance=np.array([[0.36]]),
    )
    theta = mirt.fscores(fit, responses).theta
    rmse = AbilityRMSEScorer(theta)
    plugin = PluginLogLikelihoodScorer()
    likelihood = fit.model.log_likelihood(responses, theta)
    assert rmse(fit, responses, responses) == pytest.approx(0.0, abs=1e-12)
    assert plugin(fit, responses, responses) == pytest.approx(likelihood.sum())

    monkeypatch.setattr(mirt, "fit_mirt", lambda *args, **kwargs: fit)
    splitter = KFold(n_splits=3, random_state=2)
    result = cross_validate("1PL", responses, splitter=splitter, scorers=[plugin, rmse])
    assert result.scores[rmse.name] == pytest.approx([0.0] * 3, abs=1e-12)
    assert result.scores[plugin.name] == pytest.approx(
        [likelihood[test].sum() for _, test in splitter.split(responses)]
    )


def test_plugin_likelihood_matches_ability_column_order_and_nullable_responses():
    pd = pytest.importorskip("pandas")
    train = pd.DataFrame(
        np.tile([[1, 0], [0, 1], [1, 0], [1, 1]], (8, 1)), columns=["easy", "hard"]
    )
    fit = mirt.fit_mirt(train, model="1PL", n_quadpts=9, max_iter=3)
    test = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "hard": pd.array([0, pd.NA, 1], dtype="Int64"),
            "easy": [1, 0, 0],
        }
    )
    responses = np.array([[1, 0], [0, -1], [0, 1]])
    expected = fit.model.log_likelihood(responses, mirt.fscores(fit, responses).theta)
    assert PluginLogLikelihoodScorer()(fit, train, test) == pytest.approx(
        expected.sum()
    )
