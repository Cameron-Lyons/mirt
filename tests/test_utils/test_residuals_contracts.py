from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

import mirt._local_dependence as local_dependence
from mirt.models.dichotomous import (
    OneParameterLogistic,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.polytomous import GeneralizedPartialCredit, GradedResponseModel
from mirt.utils.residuals import LD_X2, Q3, _compute_ld_matrix, residuals


def _scalar_q3(values: np.ndarray) -> np.ndarray:
    n_items = values.shape[1]
    result = np.full((n_items, n_items), np.nan)
    np.fill_diagonal(result, 1.0)
    for first in range(n_items):
        for second in range(first + 1, n_items):
            valid = np.isfinite(values[:, first]) & np.isfinite(values[:, second])
            if np.count_nonzero(valid) <= 2:
                continue
            first_values = values[valid, first]
            second_values = values[valid, second]
            if np.var(first_values) == 0.0 or np.var(second_values) == 0.0:
                continue
            correlation = np.corrcoef(first_values, second_values)[0, 1]
            result[first, second] = result[second, first] = correlation
    return result


def _scalar_marginal_ld(
    model: Any,
    responses: np.ndarray,
    n_categories: list[int],
    n_quadpts: int = 49,
) -> tuple[np.ndarray, np.ndarray]:
    """Chen-Thissen reference: tables integrated node by node over N(0, 1)."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(n_quadpts)
    weights = weights / weights.sum()
    probabilities = np.asarray(model.probability(nodes[:, None]), dtype=np.float64)
    if probabilities.ndim == 2:
        probabilities = np.stack((1.0 - probabilities, probabilities), axis=2)
    n_items = responses.shape[1]
    statistics = np.full((n_items, n_items), np.nan)
    p_values = np.full((n_items, n_items), np.nan)
    for first in range(n_items):
        for second in range(first + 1, n_items):
            valid = (
                np.isfinite(responses[:, first])
                & (responses[:, first] >= 0)
                & np.isfinite(responses[:, second])
                & (responses[:, second] >= 0)
            )
            n_joint = np.count_nonzero(valid)
            if n_joint < 10:
                continue
            first_categories = n_categories[first]
            second_categories = n_categories[second]
            observed = np.zeros((first_categories, second_categories))
            for first_response, second_response in zip(
                responses[valid, first].astype(np.int64),
                responses[valid, second].astype(np.int64),
                strict=True,
            ):
                observed[first_response, second_response] += 1
            expected = np.zeros_like(observed)
            for node, weight in enumerate(weights):
                expected += weight * np.outer(
                    probabilities[node, first, :first_categories],
                    probabilities[node, second, :second_categories],
                )
            expected *= n_joint
            statistic = np.sum((observed - expected) ** 2 / expected)
            degrees = (first_categories - 1) * (second_categories - 1)
            statistics[first, second] = statistics[second, first] = statistic
            probability = stats.chi2.sf(statistic, df=degrees)
            p_values[first, second] = p_values[second, first] = probability
    return statistics, p_values


def _polytomous_responses(
    model: GradedResponseModel | GeneralizedPartialCredit,
    theta: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probabilities = model.probability(theta)
    responses = np.empty((len(theta), model.n_items), dtype=np.float64)
    for person in range(len(theta)):
        for item, count in enumerate(model.n_categories):
            responses[person, item] = rng.choice(
                count,
                p=probabilities[person, item, :count],
            )
    return responses


def test_vectorized_q3_matches_pairwise_complete_scalar_reference() -> None:
    rng = np.random.default_rng(619)
    values = rng.normal(size=(137, 23))
    values[rng.random(values.shape) < 0.19] = np.nan
    values[:, 0] = 2.0
    values[:, 1] = np.nan
    values[:2, 1] = [0.0, 1.0]

    actual = _compute_ld_matrix(values)
    expected = _scalar_q3(values)

    assert_allclose(actual, expected, rtol=3e-13, atol=3e-13, equal_nan=True)


def test_q3_suppression_preserves_missing_pairs_and_diagonal() -> None:
    values = np.array(
        [
            [0.0, 0.0, np.nan],
            [1.0, 0.9, np.nan],
            [2.0, 2.1, 0.0],
            [3.0, 2.9, 1.0],
        ]
    )

    result = _compute_ld_matrix(values, suppress_abs=1.0)

    assert result[0, 1] == 0.0
    assert np.isnan(result[0, 2])
    assert_array_equal(np.diag(result), np.ones(3))


def test_binary_ld_matches_independent_scalar_reference() -> None:
    rng = np.random.default_rng(701)
    theta = rng.normal(size=(181, 1))
    model = ThreeParameterLogistic(13)
    model.set_parameters(
        discrimination=np.linspace(0.6, 1.8, model.n_items),
        difficulty=np.linspace(-1.5, 1.5, model.n_items),
        guessing=np.linspace(0.05, 0.3, model.n_items),
    )
    probabilities = model.probability(theta)
    responses = (rng.random(probabilities.shape) < probabilities).astype(np.float64)
    responses[rng.random(responses.shape) < 0.11] = np.nan
    responses[rng.random(responses.shape) < 0.07] = -1

    actual = LD_X2(model, responses, use_rust=False)
    expected = _scalar_marginal_ld(model, responses, [2] * model.n_items)

    assert_allclose(actual[0], expected[0], rtol=1e-11, equal_nan=True)
    assert_allclose(actual[1], expected[1], rtol=1e-10, equal_nan=True)


@pytest.mark.parametrize(
    "residual_type", ["raw", "standardized", "pearson", "deviance"]
)
def test_binary_residual_types_match_closed_form_values(residual_type: str) -> None:
    theta = np.linspace(-1.5, 1.5, 9)[:, None]
    model = TwoParameterLogistic(3)
    model.set_parameters(
        discrimination=np.array([0.7, 1.1, 1.6]),
        difficulty=np.array([-0.4, 0.2, 0.8]),
    )
    responses = np.tile([0.0, 1.0, 0.0], (len(theta), 1))
    responses[0, 0] = -1
    probabilities = model.probability(theta)
    observed = responses >= 0
    raw = np.where(observed, responses - probabilities, np.nan)
    if residual_type == "raw":
        expected = raw
    elif residual_type in ("standardized", "pearson"):
        expected = raw / np.sqrt(np.maximum(probabilities * (1 - probabilities), 1e-12))
    else:
        response_probability = np.where(
            responses == 1,
            probabilities,
            1 - probabilities,
        )
        expected = np.where(
            observed,
            np.sign(raw)
            * np.sqrt(-2 * np.log(np.clip(response_probability, 1e-12, 1))),
            np.nan,
        )

    result = residuals(
        model,
        responses,
        theta,
        type=residual_type,  # type: ignore[arg-type]
        use_rust=False,
    )

    assert_allclose(result.raw, raw, equal_nan=True)
    assert_allclose(result.standardized, expected, equal_nan=True)


def test_binary_standardized_residuals_floor_the_bernoulli_variance() -> None:
    model = TwoParameterLogistic(4)
    model.set_parameters(
        discrimination=np.array([0.8, 1.4, 2.0, 2.4]),
        difficulty=np.array([-1.0, 0.0, 0.5, 1.0]),
    )
    theta = np.linspace(-12.0, 12.0, 41)[:, None]
    rng = np.random.default_rng(37)
    responses = rng.integers(0, 2, size=(len(theta), model.n_items)).astype(float)
    responses[::5, 1] = np.nan
    probabilities = model.probability(theta)
    variance = probabilities * (1.0 - probabilities)
    expected = np.where(
        np.isfinite(responses),
        (responses - probabilities) / np.sqrt(np.maximum(variance, 1e-10)),
        np.nan,
    )

    result = residuals(model, responses, theta)

    # p(1 - p) falls below the floor at the extreme abilities.
    assert np.any(variance < 1e-10)
    assert_allclose(result.standardized, expected, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("model_type", [GradedResponseModel, GeneralizedPartialCredit])
def test_polytomous_residuals_use_score_moments_and_category_likelihoods(
    model_type: type[GradedResponseModel] | type[GeneralizedPartialCredit],
) -> None:
    theta = np.linspace(-2.5, 2.5, 71)[:, None]
    model = model_type(4, n_categories=[3, 4, 5, 3])
    responses = _polytomous_responses(model, theta, seed=913)
    responses[0, 0] = -1
    responses[1, 1] = np.nan
    probabilities = model.probability(theta)
    categories = np.arange(probabilities.shape[2])
    expected = probabilities @ categories
    variance = probabilities @ (categories**2) - expected**2
    observed = np.isfinite(responses) & (responses >= 0)
    raw = np.where(observed, responses - expected, np.nan)
    standardized = raw / np.sqrt(np.maximum(variance, 1e-12))
    response_indices = np.where(observed, responses, 0).astype(np.int64)
    response_probability = np.take_along_axis(
        probabilities,
        response_indices[:, :, None],
        axis=2,
    )[:, :, 0]
    deviance = np.where(
        observed,
        np.sign(raw) * np.sqrt(-2 * np.log(np.clip(response_probability, 1e-12, 1))),
        np.nan,
    )

    score_result = residuals(model, responses, theta, use_rust=True)
    deviance_result = residuals(
        model,
        responses,
        theta,
        type="deviance",
        use_rust=True,
    )

    assert_allclose(score_result.raw, raw, equal_nan=True)
    assert_allclose(score_result.standardized, standardized, equal_nan=True)
    assert_allclose(deviance_result.standardized, deviance, equal_nan=True)
    assert score_result.ld_matrix is not None
    assert score_result.ld_matrix.shape == (model.n_items, model.n_items)


def test_polytomous_ld_matches_generalized_scalar_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Small blocks exercise the bounded person and node accumulation.
    monkeypatch.setattr(local_dependence, "_LD_CHUNK_ELEMENTS", 300)
    theta = np.linspace(-3.0, 3.0, 109)[:, None]
    model = GradedResponseModel(8, n_categories=[3, 5, 4, 3, 6, 4, 5, 3])
    responses = _polytomous_responses(model, theta, seed=1007)
    responses[::9, 0] = -1
    responses[::11, 1] = np.nan
    responses[4:, 4] = -1
    responses[:, 5] = np.where(responses[:, 5] == 3, 0, responses[:, 5])

    actual = LD_X2(model, responses)
    expected = _scalar_marginal_ld(model, responses, model.n_categories)

    assert_allclose(actual[0], expected[0], rtol=1e-11, equal_nan=True)
    assert_allclose(actual[1], expected[1], rtol=1e-10, atol=1e-14, equal_nan=True)
    # Item 4 has four responses: its pairs and the diagonal are undefined.
    assert np.all(np.isnan(actual[0][4])) and np.all(np.isnan(actual[1][4]))
    assert np.all(np.isnan(np.diag(actual[1])))


def test_polytomous_ld_small_high_category_case_matches_scalar_reference() -> None:
    theta = np.linspace(-2.0, 2.0, 47)[:, None]
    model = GradedResponseModel(3, n_categories=[8, 7, 6])
    responses = _polytomous_responses(model, theta, seed=1013)
    responses[::7, 0] = -1
    responses[::8, 1] = np.nan

    actual = LD_X2(model, responses)
    expected = _scalar_marginal_ld(model, responses, model.n_categories)

    assert_allclose(actual[0], expected[0], rtol=1e-11, equal_nan=True)
    assert_allclose(actual[1], expected[1], rtol=1e-10, atol=1e-14, equal_nan=True)


def test_ld_x2_agrees_with_diagnostics_and_ignores_theta() -> None:
    from mirt.diagnostics.ld import compute_ld_chi2

    theta = np.linspace(-2.5, 2.5, 120)[:, None]
    model = GeneralizedPartialCredit(5, n_categories=[3, 4, 3, 2, 5])
    responses = _polytomous_responses(model, theta, seed=17)
    responses[::5, 2] = np.nan

    statistics, p_values = LD_X2(model, responses)
    diagnostics_statistics, diagnostics_p = compute_ld_chi2(model, responses)
    with pytest.warns(DeprecationWarning, match="theta"):
        with_theta = LD_X2(model, responses, theta)

    assert_allclose(statistics, diagnostics_statistics, rtol=1e-12, equal_nan=True)
    assert_allclose(p_values, diagnostics_p, rtol=1e-12, equal_nan=True)
    assert_allclose(with_theta[0], statistics, equal_nan=True)


@pytest.mark.parametrize("model_type", [ThreeParameterLogistic, GradedResponseModel])
def test_q3_correlates_raw_residuals(model_type: Any) -> None:
    theta = np.linspace(-2.0, 2.0, 90)[:, None]
    if model_type is GradedResponseModel:
        model = GradedResponseModel(4, n_categories=[3, 4, 2, 5])
        responses = _polytomous_responses(model, theta, seed=29)
        probabilities = model.probability(theta)
        expected = probabilities @ np.arange(probabilities.shape[2])
    else:
        model = ThreeParameterLogistic(4)
        model.set_parameters(guessing=np.full(4, 0.2))
        expected = model.probability(theta)
        rng = np.random.default_rng(29)
        responses = (rng.random(expected.shape) < expected).astype(np.float64)
    responses[::6, 1] = np.nan

    raw = np.where(np.isfinite(responses), responses - expected, np.nan)

    assert_allclose(Q3(model, responses, theta), _scalar_q3(raw), atol=1e-13)


@pytest.mark.parametrize("model_type", [OneParameterLogistic, TwoParameterLogistic])
def test_binary_q3_conventions_agree_across_apis(model_type: Any) -> None:
    from mirt.diagnostics import compute_ld_statistics, compute_q3

    rng = np.random.default_rng(31)
    model = model_type(12)
    parameters = {"difficulty": rng.normal(0.0, 1.2, model.n_items)}
    if model_type is TwoParameterLogistic:
        parameters["discrimination"] = rng.uniform(0.6, 2.4, model.n_items)
    model.set_parameters(**parameters)
    model._is_fitted = True
    theta = rng.normal(0.0, 1.3, (700, 1))
    responses = model.simulate(theta, seed=4).astype(np.float64)
    responses[rng.random(responses.shape) < 0.15] = np.nan
    responses[:698, -1] = -1  # a sparse item: two responses
    observed = np.isfinite(responses) & (responses >= 0)
    raw = np.where(observed, responses - model.probability(theta), np.nan)

    q3 = Q3(model, responses, theta)
    diagnostic = compute_q3(model, responses, theta)

    # Q3 leaves sparse pairs undefined and has a unit diagonal.
    assert_allclose(q3, _scalar_q3(raw), atol=1e-12, equal_nan=True)
    assert np.all(np.isnan(q3[-1, :-1])) and np.all(np.diag(q3) == 1.0)
    assert_array_equal(residuals(model, responses, theta, "raw").ld_matrix, q3)
    # compute_q3 zeroes sparse pairs and the diagonal.
    expected = np.nan_to_num(q3, nan=0.0)
    np.fill_diagonal(expected, 0.0)
    assert_allclose(diagnostic, expected, rtol=0, atol=1e-12)
    assert_allclose(
        compute_ld_statistics(model, responses, theta).q3_matrix,
        diagnostic,
        rtol=0,
        atol=1e-12,
    )


def test_use_rust_does_not_change_binary_results() -> None:
    rng = np.random.default_rng(1103)
    theta = rng.normal(size=(211, 1))
    model = TwoParameterLogistic(9)
    model.set_parameters(
        discrimination=np.linspace(0.6, 1.7, model.n_items),
        difficulty=np.linspace(-1.2, 1.2, model.n_items),
    )
    probabilities = model.probability(theta)
    responses = (rng.random(probabilities.shape) < probabilities).astype(np.float64)
    responses[rng.random(responses.shape) < 0.09] = -1

    for residual_type in ("raw", "standardized"):
        enabled = residuals(model, responses, theta, residual_type, use_rust=True)
        disabled = residuals(model, responses, theta, residual_type, use_rust=False)
        assert_array_equal(enabled.standardized, disabled.standardized)
        assert_array_equal(enabled.ld_matrix, disabled.ld_matrix)
    assert_array_equal(
        Q3(model, responses, theta, use_rust=True),
        Q3(model, responses, theta, use_rust=False),
    )
    for enabled_values, disabled_values in zip(
        LD_X2(model, responses, use_rust=True),
        LD_X2(model, responses, use_rust=False),
        strict=True,
    ):
        assert_array_equal(enabled_values, disabled_values)


@pytest.mark.parametrize("model_type", [ThreeParameterLogistic, TwoParameterLogistic])
def test_negative_and_nan_missing_codes_are_equivalent(model_type: Any) -> None:
    rng = np.random.default_rng(1201)
    theta = rng.normal(size=(83, 1))
    model = model_type(7)
    probabilities = model.probability(theta)
    responses = (rng.random(probabilities.shape) < probabilities).astype(np.float64)
    missing = rng.random(responses.shape) < 0.17
    negative_missing = responses.copy()
    negative_missing[missing] = -1
    nan_missing = responses.copy()
    nan_missing[missing] = np.nan

    negative_result = residuals(model, negative_missing, theta)
    nan_result = residuals(model, nan_missing, theta)

    assert np.all(np.isnan(negative_result.raw[missing]))
    assert np.all(np.isnan(negative_result.standardized[missing]))
    assert_allclose(negative_result.raw, nan_result.raw, equal_nan=True)
    assert_allclose(
        negative_result.standardized,
        nan_result.standardized,
        equal_nan=True,
    )
    assert_allclose(Q3(model, negative_missing, theta), Q3(model, nan_missing, theta))
    for negative_values, nan_values in zip(
        LD_X2(model, negative_missing),
        LD_X2(model, nan_missing),
        strict=True,
    ):
        assert_allclose(negative_values, nan_values, equal_nan=True)


def test_entirely_missing_item_has_defined_summary_and_pair_outputs() -> None:
    model = ThreeParameterLogistic(3)
    responses = np.zeros((9, 3))
    responses[:, 0] = -1
    theta = np.zeros((9, 1))

    result = residuals(model, responses, theta)

    assert np.isnan(result.summary["mean_raw"][0])
    assert np.isnan(result.summary["std_standardized"][0])
    assert np.isnan(result.summary["max_abs_standardized"][0])
    assert result.summary["n_large"][0] == 0
    assert np.isnan(result.ld_matrix[0, 1])


def test_each_public_fallback_evaluates_probabilities_once() -> None:
    class CountingModel:
        n_items = 4
        n_factors = 1

        def __init__(self) -> None:
            self.calls = 0

        def probability(self, theta: np.ndarray) -> np.ndarray:
            self.calls += 1
            return np.full((len(theta), self.n_items), 0.5)

    model = CountingModel()
    responses = np.zeros((7, 4))
    theta = np.zeros((7, 1))

    residuals(model, responses, theta, use_rust=False)  # type: ignore[arg-type]
    assert model.calls == 1
    Q3(model, responses, theta, use_rust=False)  # type: ignore[arg-type]
    assert model.calls == 2
    LD_X2(model, responses, use_rust=False)  # type: ignore[arg-type]
    assert model.calls == 3


@pytest.mark.parametrize(
    ("responses", "theta", "message"),
    [
        (np.zeros(4), np.zeros(4), "two-dimensional"),
        (np.empty((0, 2)), np.empty(0), "nonempty"),
        (np.zeros((4, 2)), np.zeros(3), "one row"),
        (np.zeros((4, 2)), np.zeros((4, 2)), "model.n_factors"),
        (np.zeros((4, 2)), np.full(4, np.inf), "finite"),
        (np.zeros((4, 3)), np.zeros(4), "model.n_items"),
        (np.full((4, 2), np.inf), np.zeros(4), "finite category"),
        (np.full((4, 2), -2.0), np.zeros(4), "only supported negative"),
        (np.full((4, 2), 0.5), np.zeros(4), "integer category"),
        (np.full((4, 2), 2.0), np.zeros(4), "maximum category"),
    ],
)
def test_residual_inputs_fail_with_clear_contract_errors(
    responses: np.ndarray,
    theta: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        residuals(TwoParameterLogistic(2), responses, theta, use_rust=False)


@pytest.mark.parametrize(
    ("probabilities", "n_categories", "message"),
    [
        (np.zeros(4), 2, "shape"),
        (np.zeros((4, 3)), 2, "shape"),
        (np.full((4, 2), np.nan), 2, "finite"),
        (np.full((4, 2), 1.5), 2, r"\[0, 1\]"),
        (np.zeros((4, 2, 2, 1)), 2, "shape"),
        (np.full((4, 2, 3), 1 / 4), 3, "sum to one"),
        (
            np.concatenate(
                [np.full((4, 2, 2), 0.5), np.full((4, 2, 1), 0.1)],
                axis=2,
            ),
            [2, 2],
            "beyond",
        ),
        (np.full((4, 2, 3), 1 / 3), [3], "at least two per item"),
    ],
)
def test_invalid_model_probability_outputs_are_rejected(
    probabilities: np.ndarray,
    n_categories: int | list[int],
    message: str,
) -> None:
    class InvalidModel:
        n_items = 2
        n_factors = 1

        def __init__(self) -> None:
            self.n_categories = n_categories

        def probability(self, theta: np.ndarray) -> np.ndarray:
            return probabilities

    with pytest.raises(ValueError, match=message):
        residuals(
            InvalidModel(),  # type: ignore[arg-type]
            np.zeros((4, 2)),
            np.zeros(4),
            use_rust=False,
        )


@pytest.mark.parametrize(
    ("function", "kwargs", "message"),
    [
        (residuals, {"type": "unknown"}, "Unknown residual type"),
        (residuals, {"use_rust": "yes"}, "boolean"),
        (residuals, {"suppress_abs": -1.0}, "nonnegative"),
        (Q3, {"use_rust": "yes"}, "boolean"),
        (LD_X2, {"use_rust": "yes"}, "boolean"),
    ],
)
def test_public_options_are_validated_before_numerical_work(
    function: Any,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        function(
            TwoParameterLogistic(2),
            np.zeros((4, 2)),
            np.zeros(4),
            **kwargs,
        )
