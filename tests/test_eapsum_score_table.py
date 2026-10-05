"""EAPsum probability-space recursion, score tables and sum-score contracts."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mirt
import mirt.scoring._eapsum as eapsum_module
from mirt.exceptions import MirtValidationError
from mirt.models import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.base import BaseItemModel
from mirt.results.fit_result import FitResult
from mirt.scoring import (
    EAPSumScorer,
    SumScoreTable,
    eapsum,
    eapsum_table,
    fscores,
    sum_score_to_theta,
)
from mirt.scoring._common import build_quadrature
from mirt.utils.numeric import logsumexp


def _fitted(model: BaseItemModel) -> BaseItemModel:
    model._is_fitted = True
    return model


def _two_pl(n_items: int = 5) -> TwoParameterLogistic:
    model = TwoParameterLogistic(n_items=n_items)
    model.set_parameters(
        discrimination=np.linspace(0.7, 1.6, n_items),
        difficulty=np.linspace(-1.2, 1.4, n_items),
    )
    return _fitted(model)


def _model(kind: str, n_items: int = 12) -> BaseItemModel:
    rng = np.random.default_rng(len(kind))
    if kind == "3PL":
        model: BaseItemModel = ThreeParameterLogistic(n_items=n_items)
        model.set_parameters(
            discrimination=rng.uniform(0.6, 2.0, n_items),
            difficulty=rng.normal(size=n_items),
            guessing=rng.uniform(0.05, 0.25, n_items),
        )
    elif kind == "GRM":
        model = GradedResponseModel(n_items=n_items, n_categories=[2, 3, 4] * 4)
        thresholds = np.sort(rng.normal(size=model.parameters["thresholds"].shape), 1)
        model.set_parameters(
            discrimination=rng.uniform(0.6, 2.0, n_items), thresholds=thresholds
        )
    else:
        model = GeneralizedPartialCredit(n_items=n_items, n_categories=4)
        model.set_parameters(
            discrimination=rng.uniform(0.6, 2.0, n_items),
            steps=rng.normal(size=model.parameters["steps"].shape),
        )
    return _fitted(model)


def _responses(model: BaseItemModel, n_persons: int, missing: float) -> np.ndarray:
    rng = np.random.default_rng(17)
    categories = np.asarray(model.n_categories) if model.is_polytomous else 2
    responses = rng.integers(0, 1 << 30, size=(n_persons, model.n_items)) % categories
    responses[rng.random(responses.shape) < missing] = -1
    return responses


def _reference_scores(
    model: BaseItemModel, responses: np.ndarray, n_quadpts: int
) -> tuple[np.ndarray, np.ndarray]:
    """Score each row with the original per-mask log-space algorithm."""
    points, weights = build_quadrature(
        n_quadpts=n_quadpts, n_factors=1, prior_mean=None, prior_cov=None
    )
    log_prior = np.log(weights + 1e-300)
    theta = np.empty(responses.shape[0])
    standard_error = np.empty(responses.shape[0])
    for row, pattern in enumerate(responses):
        items = np.flatnonzero(pattern >= 0)
        log_dist = np.zeros((1, len(points)))
        for item in items:
            probabilities = model.probability(points, int(item))
            if probabilities.ndim == 1:
                probabilities = np.column_stack((1.0 - probabilities, probabilities))
            log_probabilities = np.log(probabilities + 1e-300)
            updated = np.full(
                (log_dist.shape[0] + log_probabilities.shape[1] - 1, len(points)),
                -np.inf,
            )
            for category in range(log_probabilities.shape[1]):
                target = updated[category : category + log_dist.shape[0]]
                np.logaddexp(
                    target, log_dist + log_probabilities[:, category], out=target
                )
            log_dist = updated
        log_posterior = log_dist[int(pattern[items].sum())] + log_prior
        posterior = np.exp(log_posterior - logsumexp(log_posterior))
        theta[row] = posterior @ points[:, 0]
        standard_error[row] = np.sqrt(posterior @ (points[:, 0] - theta[row]) ** 2)
    return theta, standard_error


@pytest.mark.parametrize("kind", ["3PL", "GRM", "GPCM"])
def test_missing_data_scores_match_per_mask_log_space_reference(kind: str) -> None:
    model = _model(kind)
    responses = _responses(model, 120, missing=0.15)
    responses[0] = -1

    actual = EAPSumScorer(n_quadpts=31).score(model, responses)
    expected = _reference_scores(model, responses, 31)

    assert_allclose(actual.theta, expected[0], rtol=0.0, atol=1e-12)
    assert_allclose(actual.standard_error, expected[1], rtol=0.0, atol=1e-12)


def test_item_probabilities_are_evaluated_once_per_model_state() -> None:
    model = _model("GRM")
    responses = _responses(model, 80, missing=0.2)
    original = model.probability
    evaluated: list[int] = []

    def counted(theta: np.ndarray, item_idx: int | None = None) -> np.ndarray:
        evaluated.append(item_idx)
        return original(theta, item_idx)

    model.probability = counted  # type: ignore[method-assign]
    scorer = EAPSumScorer(n_quadpts=21)

    scorer.score(model, responses)

    assert len(scorer._lookup_values) > 10
    assert sorted(evaluated) == list(range(model.n_items))

    model.set_item_parameter(0, "discrimination", 1.9)
    scorer.score(model, responses)

    assert len(evaluated) == 2 * model.n_items


def test_extreme_sum_scores_fall_back_to_log_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Very hard, very discriminating items make high sum scores so unlikely
    # that probability-space terms underflow at every quadrature node.
    model = ThreeParameterLogistic(n_items=100)
    model.set_parameters(
        discrimination=np.full(100, 3.0),
        difficulty=np.full(100, 20.0),
        guessing=np.zeros(100),
    )
    _fitted(model)
    points, _ = build_quadrature(
        n_quadpts=49, n_factors=1, prior_mean=None, prior_cov=None
    )
    scorer = EAPSumScorer()
    tables = scorer._item_probability_tables(model, points, tuple(range(100)))
    probability_space = eapsum_module._probability_sum_score_distribution(
        tables, len(points)
    )
    assert np.any(np.all(np.isneginf(probability_space), axis=1))
    log_space = eapsum_module._log_space_sum_score_distribution(tables, len(points))
    # Without a prior, the conditional table alone triggers the fallback.
    assert_array_equal(
        scorer._compute_sum_score_distribution(model, points, 100, tuple(range(100))),
        log_space,
    )

    calls = 0
    original = eapsum_module._log_space_sum_score_distribution

    def counted(*args: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(eapsum_module, "_log_space_sum_score_distribution", counted)
    table = scorer.score_table(model)

    assert calls == 1
    assert np.all(np.isfinite(table.theta))
    assert np.all(np.isfinite(table.standard_error))
    responses = np.tril(np.ones((101, 100), dtype=int), k=-1)
    reference = _reference_scores(model, responses, 49)
    assert_allclose(table.theta, reference[0], rtol=0.0, atol=1e-10)
    assert_allclose(table.standard_error, reference[1], rtol=0.0, atol=1e-10)


def test_long_forms_keep_normalized_conditional_distributions() -> None:
    model = GradedResponseModel(n_items=200, n_categories=4)
    rng = np.random.default_rng(3)
    model.set_parameters(
        discrimination=rng.uniform(0.5, 2.5, 200),
        thresholds=np.sort(rng.normal(size=(200, 3)), axis=1),
    )
    points, _ = build_quadrature(
        n_quadpts=49, n_factors=1, prior_mean=None, prior_cov=None
    )
    distribution = EAPSumScorer()._compute_sum_score_distribution(
        model, points, 600, tuple(range(200))
    )

    assert distribution.shape == (601, 49)
    assert_allclose(np.exp(distribution).sum(axis=0), 1.0, rtol=1e-12)


def test_customized_two_pl_subclass_does_not_use_native_recursion() -> None:
    class SteeperTwoPL(TwoParameterLogistic):
        def probability(self, theta, item_idx=None):
            return super().probability(2.0 * np.asarray(theta), item_idx)

    base = _two_pl()
    custom = _fitted(SteeperTwoPL(n_items=base.n_items))
    custom.set_parameters(
        discrimination=base.discrimination, difficulty=base.difficulty
    )
    equivalent = _fitted(TwoParameterLogistic(n_items=base.n_items))
    equivalent.set_parameters(
        discrimination=2.0 * base.discrimination, difficulty=base.difficulty / 2.0
    )
    responses = np.array([[0, 1, 1, 0, 1], [1, 1, 1, 1, 0]])

    actual = EAPSumScorer(n_quadpts=31).score(custom, responses)
    expected = EAPSumScorer(n_quadpts=31).score(equivalent, responses)

    assert_allclose(actual.theta, expected.theta, atol=1e-12)
    assert_allclose(actual.standard_error, expected.standard_error, atol=1e-12)


def test_score_table_matches_lookup_and_brute_force_marginal() -> None:
    model = _two_pl(3)
    scorer = EAPSumScorer(n_quadpts=21)

    table = scorer.score_table(model)
    lookup = scorer.get_lookup_table(model)

    assert isinstance(table, SumScoreTable)
    assert table.n_scores == 4
    assert_array_equal(table.sum_score, np.arange(4))
    assert_array_equal(table.theta, [lookup[score]["theta"] for score in range(4)])
    assert_array_equal(
        table.standard_error, [lookup[score]["se"] for score in range(4)]
    )
    assert table.observed is None
    assert table.expected is None
    assert table.standardized_residual is None

    points, weights = build_quadrature(
        n_quadpts=21, n_factors=1, prior_mean=None, prior_cov=None
    )
    probabilities = np.column_stack(
        [model.probability(points, item) for item in range(3)]
    )
    brute_force = np.zeros(4)
    for pattern in itertools.product((0, 1), repeat=3):
        likelihood = np.prod(
            np.where(pattern, probabilities, 1.0 - probabilities), axis=1
        )
        brute_force[sum(pattern)] += weights @ likelihood
    assert_allclose(table.expected_proportion, brute_force, rtol=1e-12)
    assert_allclose(table.expected_proportion.sum(), 1.0, rtol=1e-14)


def test_score_table_marginal_matches_public_observed_score_distribution() -> None:
    model = _model("GRM")
    points, weights = build_quadrature(
        n_quadpts=49, n_factors=1, prior_mean=None, prior_cov=None
    )

    table = eapsum_table(model)

    assert_allclose(
        table.expected_proportion,
        mirt.lord_wingersky_recursion(model, points[:, 0], weights),
        rtol=1e-10,
        atol=1e-15,
    )


def test_score_table_reports_observed_frequencies_and_residuals() -> None:
    model = _two_pl()
    responses = _responses(model, 300, missing=0.0)
    fit = FitResult(
        model=model,
        log_likelihood=-10.0,
        n_iterations=1,
        converged=True,
        standard_errors={},
        aic=20.0,
        bic=21.0,
    )

    table = eapsum_table(fit, responses)

    observed = np.bincount(responses.sum(axis=1), minlength=6)
    assert_array_equal(table.observed, observed)
    assert_allclose(table.expected, 300 * table.expected_proportion)
    assert_allclose(
        table.standardized_residual,
        (observed - table.expected) / np.sqrt(table.expected),
    )
    payload = table.to_dict()
    assert list(payload) == [
        "sum_score",
        "theta",
        "standard_error",
        "expected_proportion",
        "observed",
        "expected",
        "standardized_residual",
    ]
    assert payload["observed"] == observed.tolist()
    frame = table.to_dataframe()
    assert list(frame.columns) == list(payload)
    assert len(frame) == table.n_scores == 6
    scores = fscores(fit, responses, method="EAPsum")
    assert_array_equal(scores.theta, table.theta[responses.sum(axis=1)])


def test_score_table_rejects_missing_responses() -> None:
    model = _two_pl()
    responses = np.array([[0, 1, -1, 0, 1]])

    with pytest.raises(MirtValidationError, match="complete responses"):
        eapsum_table(model, responses)


@pytest.mark.parametrize(
    ("sum_scores", "message"),
    [
        ([-1], "between 0 and 5"),
        ([6], "between 0 and 5"),
        ([2.5], "finite integer"),
        ([np.nan], "finite integer"),
        ([np.inf], "finite integer"),
        ([True], "integer sum scores"),
        (["2"], "integer sum scores"),
    ],
)
def test_sum_score_to_theta_rejects_invalid_scores(
    sum_scores: list[object], message: str
) -> None:
    with pytest.raises(MirtValidationError, match=message):
        sum_score_to_theta(_two_pl(), sum_scores)


def test_sum_score_to_theta_indexes_the_lookup_and_accepts_fit_results() -> None:
    model = _two_pl()
    fit = FitResult(
        model=model,
        log_likelihood=-10.0,
        n_iterations=1,
        converged=True,
        standard_errors={},
        aic=20.0,
        bic=21.0,
    )
    lookup = EAPSumScorer().get_lookup_table(model)

    scores = np.array([[5.0, 0.0], [3.0, 3.0]])

    theta, standard_error = sum_score_to_theta(fit, scores)

    assert theta.shape == standard_error.shape == scores.shape
    assert_array_equal(
        theta.ravel(), [lookup[score]["theta"] for score in (5, 0, 3, 3)]
    )
    assert_array_equal(
        standard_error.ravel(), [lookup[score]["se"] for score in (5, 0, 3, 3)]
    )
    empty_theta, empty_se = sum_score_to_theta(model, [])
    assert empty_theta.shape == empty_se.shape == (0,)


def test_sum_score_helpers_keep_their_model_keyword() -> None:
    model = _two_pl()
    responses = np.array([[0, 1, 1, 0, 1]])

    by_keyword = eapsum(model=model, responses=responses)
    theta, standard_error = sum_score_to_theta(model=model, sum_scores=[3])

    assert_array_equal(by_keyword.theta, eapsum(model, responses).theta)
    assert_array_equal(theta, sum_score_to_theta(model, [3])[0])
    assert_array_equal(standard_error, sum_score_to_theta(model, [3])[1])


def test_sum_score_helpers_forward_scoring_priors() -> None:
    model = _two_pl()
    prior = {"prior_mean": np.array([0.4]), "prior_cov": np.array([[1.7]])}
    scorer = EAPSumScorer(n_quadpts=31, **prior)
    responses = np.array([[0, 1, 1, 0, 1]])

    theta, standard_error = sum_score_to_theta(model, [3], n_quadpts=31, **prior)
    table = eapsum_table(model, n_quadpts=31, **prior)
    scores = eapsum(model, responses, n_quadpts=31, **prior)

    lookup = scorer.get_lookup_table(model)
    assert theta[0] == lookup[3]["theta"]
    assert standard_error[0] == lookup[3]["se"]
    assert table.theta[3] == lookup[3]["theta"]
    assert scores.theta[0] == lookup[3]["theta"]
    assert theta[0] != sum_score_to_theta(model, [3], n_quadpts=31)[0][0]
