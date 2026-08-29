"""Contracts for shared-simulation posterior predictive suites."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mirt.diagnostics import posterior_predictive_checks
from mirt.diagnostics.bayesian import posterior_predictive_check
from mirt.estimation.mcmc import MCMCResult
from mirt.models.dichotomous import TwoParameterLogistic


def _case() -> tuple[
    TwoParameterLogistic,
    np.ndarray,
    dict[str, np.ndarray],
    MCMCResult,
]:
    model = TwoParameterLogistic(4).set_parameters(
        discrimination=np.array([0.8, 1.0, 1.2, 1.4]),
        difficulty=np.array([-0.8, -0.2, 0.4, 1.0]),
    )
    responses = np.array(
        [
            [1, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, -1, 1, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ]
    )
    n_samples = 6
    chains = {
        "discrimination": np.tile(model.discrimination, (n_samples, 1)),
        "difficulty": np.tile(model.difficulty, (n_samples, 1)),
        "theta": np.linspace(-1.5, 1.5, n_samples * len(responses)).reshape(
            n_samples,
            len(responses),
            1,
        ),
    }
    result = MCMCResult(
        model=model,
        chains=chains,
        log_likelihood=0.0,
        dic=0.0,
        waic=0.0,
        rhat={},
        ess={},
        n_iterations=n_samples,
        burnin=0,
        thin=1,
    )
    return model, responses, chains, result


def test_suite_matches_individual_checks_from_the_same_draws() -> None:
    model, responses, _, result = _case()
    names = ["item_mean", "person_score", "chi_square"]

    suite = posterior_predictive_checks(
        result,
        responses,
        model,
        names,
        n_rep=9,
        seed=42,
    )

    assert list(suite) == names
    for name in names:
        individual = posterior_predictive_check(
            result,
            responses,
            model,
            name,
            n_rep=9,
            seed=42,
        )
        assert suite[name].test_statistic_observed == (
            individual.test_statistic_observed
        )
        assert_array_equal(
            suite[name].test_statistic_replicated,
            individual.test_statistic_replicated,
        )
        assert suite[name].p_value == individual.p_value
        assert suite[name].summary_stats == individual.summary_stats


def test_suite_simulates_once_per_replication(monkeypatch: pytest.MonkeyPatch) -> None:
    model, responses, _, result = _case()
    original_probability = model.probability
    call_count = 0

    def counted_probability(theta: np.ndarray) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        return original_probability(theta)

    monkeypatch.setattr(model, "probability", counted_probability)

    posterior_predictive_checks(
        result,
        responses,
        model,
        ["item_mean", "person_score", "chi_square"],
        n_rep=7,
        seed=10,
    )

    assert call_count == 7


def test_suite_supports_labeled_custom_statistics() -> None:
    model, responses, _, result = _case()

    def observed_total(values: np.ndarray) -> float:
        return float(np.sum(values[values >= 0]))

    suite = posterior_predictive_checks(
        result,
        responses,
        model,
        {
            "total": observed_total,
            "double_total": lambda values: 2.0 * observed_total(values),
            "average": "item_mean",
        },
        n_rep=5,
        seed=3,
    )

    assert list(suite) == ["total", "double_total", "average"]
    assert suite["double_total"].test_statistic_observed == (
        2.0 * suite["total"].test_statistic_observed
    )
    assert_allclose(
        suite["double_total"].test_statistic_replicated,
        2.0 * suite["total"].test_statistic_replicated,
    )


def test_suite_restores_model_when_a_statistic_raises() -> None:
    model, responses, chains, result = _case()
    original = model.parameters
    calls = 0

    def failing_statistic(values: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("suite statistic failed")
        return float(np.mean(values[values >= 0]))

    shifted = {**chains, "difficulty": chains["difficulty"] + 2.0}
    result.chains = shifted

    with pytest.raises(RuntimeError, match="suite statistic failed"):
        posterior_predictive_checks(
            result,
            responses,
            model,
            {"mean": "item_mean", "failure": failing_statistic},
            n_rep=3,
            seed=5,
        )

    for name, values in original.items():
        assert_array_equal(model.parameters[name], values)


@pytest.mark.parametrize(
    ("statistics", "message"),
    [
        ([], "at least one"),
        ("item_mean", "mapping or an iterable"),
        (["item_mean", "item_mean"], "unique"),
        ([object()], "contain only names"),
        ({"": "item_mean"}, "non-empty strings"),
        ({"value": 3}, "built-in names or callable"),
        ({"value": "unknown"}, "Unknown test statistic"),
    ],
)
def test_suite_validates_statistic_collections(
    statistics: object,
    message: str,
) -> None:
    model, responses, _, result = _case()

    with pytest.raises(ValueError, match=message):
        posterior_predictive_checks(
            result,
            responses,
            model,
            statistics,  # type: ignore[arg-type]
        )


def test_suite_rejects_a_noniterable_collection() -> None:
    model, responses, _, result = _case()

    def invalid(values: np.ndarray) -> float:
        return float(values.size)

    with pytest.raises(ValueError, match="mapping or an iterable"):
        posterior_predictive_checks(
            result,
            responses,
            model,
            invalid,  # type: ignore[arg-type]
        )
