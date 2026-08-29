from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

from mirt.models.dichotomous import ThreeParameterLogistic, TwoParameterLogistic
from mirt.models.polytomous import GeneralizedPartialCredit, GradedResponseModel
from mirt.utils.residuals import LD_X2, Q3, _compute_ld_matrix, residuals

residuals_module = importlib.import_module("mirt.utils.residuals")


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


def _scalar_binary_ld(
    responses: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_items = responses.shape[1]
    statistics = np.zeros((n_items, n_items))
    p_values = np.ones((n_items, n_items))
    for first in range(n_items):
        for second in range(first + 1, n_items):
            valid = (
                np.isfinite(responses[:, first])
                & (responses[:, first] >= 0)
                & np.isfinite(responses[:, second])
                & (responses[:, second] >= 0)
            )
            if np.count_nonzero(valid) < 5:
                continue
            first_response = responses[valid, first]
            second_response = responses[valid, second]
            first_probability = probabilities[valid, first]
            second_probability = probabilities[valid, second]
            observed = np.array(
                [
                    np.count_nonzero((first_response == 0) & (second_response == 0)),
                    np.count_nonzero((first_response == 0) & (second_response == 1)),
                    np.count_nonzero((first_response == 1) & (second_response == 0)),
                    np.count_nonzero((first_response == 1) & (second_response == 1)),
                ]
            )
            expected = np.array(
                [
                    np.sum((1 - first_probability) * (1 - second_probability)),
                    np.sum((1 - first_probability) * second_probability),
                    np.sum(first_probability * (1 - second_probability)),
                    np.sum(first_probability * second_probability),
                ]
            )
            statistic = np.sum((observed - expected) ** 2 / np.maximum(expected, 0.5))
            statistics[first, second] = statistics[second, first] = statistic
            probability = stats.chi2.sf(statistic, df=1)
            p_values[first, second] = p_values[second, first] = probability
    return statistics, p_values


def _scalar_polytomous_ld(
    responses: np.ndarray,
    probabilities: np.ndarray,
    n_categories: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    n_items = responses.shape[1]
    statistics = np.zeros((n_items, n_items))
    p_values = np.ones((n_items, n_items))
    for first in range(n_items):
        for second in range(first + 1, n_items):
            valid = (
                np.isfinite(responses[:, first])
                & (responses[:, first] >= 0)
                & np.isfinite(responses[:, second])
                & (responses[:, second] >= 0)
            )
            if np.count_nonzero(valid) < 5:
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
            expected = (
                probabilities[valid, first, :first_categories].T
                @ probabilities[valid, second, :second_categories]
            )
            statistic = np.sum((observed - expected) ** 2 / np.maximum(expected, 0.5))
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

    actual = LD_X2(model, responses, theta, use_rust=False)
    expected = _scalar_binary_ld(responses, probabilities)

    assert_allclose(actual[0], expected[0], rtol=2e-13, atol=2e-13)
    assert_allclose(actual[1], expected[1], rtol=2e-13, atol=2e-13)


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
    monkeypatch.setattr(residuals_module, "_LD_CATEGORY_CHUNK_ELEMENTS", 30)

    def unexpected_pairwise(*args: Any, **kwargs: Any) -> None:
        pytest.fail("category blocks should handle wider item sets")

    monkeypatch.setattr(
        residuals_module,
        "_pairwise_polytomous_ld_x2",
        unexpected_pairwise,
    )
    theta = np.linspace(-3.0, 3.0, 109)[:, None]
    model = GradedResponseModel(8, n_categories=[3, 5, 4, 3, 6, 4, 5, 3])
    responses = _polytomous_responses(model, theta, seed=1007)
    responses[::9, 0] = -1
    responses[::11, 1] = np.nan
    responses[4:, 4] = -1
    probabilities = model.probability(theta)

    actual = LD_X2(model, responses, theta)
    expected = _scalar_polytomous_ld(
        responses,
        probabilities,
        model.n_categories,
    )

    assert_allclose(actual[0], expected[0], rtol=2e-13, atol=2e-13)
    assert_allclose(actual[1], expected[1], rtol=2e-13, atol=2e-13)


def test_polytomous_ld_small_high_category_case_matches_scalar_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairwise_calls = 0
    original_pairwise = residuals_module._pairwise_polytomous_ld_x2

    def tracked_pairwise(*args: Any, **kwargs: Any) -> Any:
        nonlocal pairwise_calls
        pairwise_calls += 1
        return original_pairwise(*args, **kwargs)

    monkeypatch.setattr(
        residuals_module,
        "_pairwise_polytomous_ld_x2",
        tracked_pairwise,
    )
    theta = np.linspace(-2.0, 2.0, 47)[:, None]
    model = GradedResponseModel(3, n_categories=[8, 7, 6])
    responses = _polytomous_responses(model, theta, seed=1013)
    responses[::7, 0] = -1
    responses[::8, 1] = np.nan
    probabilities = model.probability(theta)

    actual = LD_X2(model, responses, theta)
    expected = _scalar_polytomous_ld(
        responses,
        probabilities,
        model.n_categories,
    )

    assert_allclose(actual[0], expected[0], rtol=2e-13, atol=2e-13)
    assert_allclose(actual[1], expected[1], rtol=2e-13, atol=2e-13)
    assert pairwise_calls == 1


def test_native_and_vectorized_binary_paths_agree() -> None:
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

    native_residuals = residuals(model, responses, theta, use_rust=True)
    vectorized_residuals = residuals(model, responses, theta, use_rust=False)
    assert_allclose(
        native_residuals.standardized,
        vectorized_residuals.standardized,
        rtol=2e-7,
        atol=2e-7,
        equal_nan=True,
    )
    assert_allclose(
        Q3(model, responses, theta, use_rust=True),
        Q3(model, responses, theta, use_rust=False),
        rtol=2e-8,
        atol=2e-8,
        equal_nan=True,
    )
    for native_values, vectorized_values in zip(
        LD_X2(model, responses, theta, use_rust=True),
        LD_X2(model, responses, theta, use_rust=False),
        strict=True,
    ):
        assert_allclose(native_values, vectorized_values, rtol=2e-13, atol=2e-13)


def test_negative_and_nan_missing_codes_are_equivalent() -> None:
    rng = np.random.default_rng(1201)
    theta = rng.normal(size=(83, 1))
    model = ThreeParameterLogistic(7)
    probabilities = model.probability(theta)
    responses = (rng.random(probabilities.shape) < probabilities).astype(np.float64)
    missing = rng.random(responses.shape) < 0.17
    negative_missing = responses.copy()
    negative_missing[missing] = -1
    nan_missing = responses.copy()
    nan_missing[missing] = np.nan

    negative_result = residuals(model, negative_missing, theta)
    nan_result = residuals(model, nan_missing, theta)

    assert_allclose(negative_result.raw, nan_result.raw, equal_nan=True)
    assert_allclose(
        negative_result.standardized,
        nan_result.standardized,
        equal_nan=True,
    )
    assert_allclose(Q3(model, negative_missing, theta), Q3(model, nan_missing, theta))
    for negative_values, nan_values in zip(
        LD_X2(model, negative_missing, theta),
        LD_X2(model, nan_missing, theta),
        strict=True,
    ):
        assert_allclose(negative_values, nan_values)


@pytest.mark.parametrize(
    "model",
    [
        ThreeParameterLogistic(3),
        TwoParameterLogistic(3, n_factors=2),
    ],
)
def test_richer_models_never_enter_binary_2pl_native_formulas(
    monkeypatch: pytest.MonkeyPatch,
    model: ThreeParameterLogistic | TwoParameterLogistic,
) -> None:
    theta = np.zeros((8, model.n_factors))
    responses = np.zeros((8, model.n_items))

    def unexpected_native(*args: Any, **kwargs: Any) -> None:
        pytest.fail("native 2PL-only formula should not run")

    monkeypatch.setattr(residuals_module, "should_use_rust", lambda value: True)
    monkeypatch.setattr(
        residuals_module, "compute_standardized_residuals", unexpected_native
    )
    monkeypatch.setattr(residuals_module, "compute_q3_matrix", unexpected_native)
    monkeypatch.setattr(residuals_module, "compute_ld_chi2_matrix", unexpected_native)

    result = residuals(model, responses, theta, use_rust=True)
    q3 = Q3(model, responses, theta, use_rust=True)
    ld_x2, p_values = LD_X2(model, responses, theta, use_rust=True)

    assert result.raw.shape == responses.shape
    assert q3.shape == (model.n_items, model.n_items)
    assert ld_x2.shape == p_values.shape == (model.n_items, model.n_items)


def test_native_binary_payload_masks_both_missing_encodings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TwoParameterLogistic(3)
    theta = np.linspace(-1.0, 1.0, 6)[:, None]
    responses = np.zeros((6, 3))
    responses[0, 0] = -1
    responses[1, 1] = np.nan
    payloads: list[np.ndarray] = []

    def fake_standardized(
        response_values: np.ndarray,
        *args: Any,
    ) -> np.ndarray:
        payloads.append(response_values.copy())
        return np.where(response_values >= 0, 0.0, np.nan)

    def fake_q3(response_values: np.ndarray, *args: Any) -> np.ndarray:
        payloads.append(response_values.copy())
        return np.zeros((3, 3))

    def fake_ld(response_values: np.ndarray, *args: Any) -> np.ndarray:
        payloads.append(response_values.copy())
        result = np.full((3, 3), 4.0)
        np.fill_diagonal(result, 0.0)
        return result

    monkeypatch.setattr(residuals_module, "should_use_rust", lambda value: True)
    monkeypatch.setattr(
        residuals_module, "compute_standardized_residuals", fake_standardized
    )
    monkeypatch.setattr(residuals_module, "compute_q3_matrix", fake_q3)
    monkeypatch.setattr(residuals_module, "compute_ld_chi2_matrix", fake_ld)

    result = residuals(model, responses, theta)
    q3 = Q3(model, responses, theta)
    ld_x2, p_values = LD_X2(model, responses, theta)

    assert len(payloads) == 4
    for payload in payloads:
        assert payload[0, 0] == -1
        assert payload[1, 1] == -1
    assert np.isnan(result.raw[0, 0])
    assert_array_equal(np.diag(q3), np.ones(3))
    assert ld_x2[0, 1] == 0.0
    assert p_values[0, 1] == 1.0
    assert ld_x2[0, 2] == 4.0
    assert_allclose(p_values[0, 2], stats.chi2.sf(4.0, df=1))


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
    LD_X2(model, responses, theta, use_rust=False)  # type: ignore[arg-type]
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
