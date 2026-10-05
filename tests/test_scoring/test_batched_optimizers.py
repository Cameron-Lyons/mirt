"""Row-batched optimizer scoring must reproduce the per-pattern SciPy searches."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import minimize_scalar

from mirt.models import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
    NoncompensatoryModel,
    PartialCreditModel,
    ThreeParameterLogistic,
    TwoParameterLogistic,
)
from mirt.models.base import BaseItemModel
from mirt.scoring._common import (
    observed_test_information,
    resolve_prior_distribution,
    score_pattern_chunks,
    score_responses_parallel,
    unique_response_patterns,
    validate_scoring_responses,
)
from mirt.scoring._optimization import (
    batched_hessian_se,
    batched_newton_minimize,
    bounded_scalar_minimize,
)
from mirt.scoring.map import MAPScorer
from mirt.scoring.ml import MLScorer
from mirt.scoring.wle import WLEScorer
from mirt.utils.numeric import compute_hessian_se


def _random_model(kind: str, n_items: int = 10, n_factors: int = 1) -> BaseItemModel:
    rng = np.random.default_rng(sum(map(ord, kind)) + n_factors)
    if kind == "2PL":
        model: BaseItemModel = TwoParameterLogistic(n_items, n_factors=n_factors)
    elif kind == "3PL":
        model = ThreeParameterLogistic(n_items, n_factors=n_factors)
    elif kind == "GRM":
        model = GradedResponseModel(n_items, n_categories=4, n_factors=n_factors)
    elif kind == "GPCM":
        model = GeneralizedPartialCredit(n_items, n_categories=[3, 4] * (n_items // 2))
    elif kind == "PCM":
        model = PartialCreditModel(n_items, n_categories=3)
    else:
        model = NominalResponseModel(n_items, n_categories=3)
    updates = {}
    for name, values in model.parameters.items():
        if kind == "PCM" and name == "discrimination":
            continue
        if name in ("discrimination", "slopes"):
            updates[name] = rng.uniform(0.6, 2.0, values.shape)
        elif name == "guessing":
            updates[name] = rng.uniform(0.05, 0.25, values.shape)
        elif name == "thresholds":
            updates[name] = np.sort(rng.normal(0.0, 1.0, values.shape), axis=-1)
        else:
            updates[name] = rng.normal(0.0, 0.8, values.shape)
    model.set_parameters(**updates)
    model._is_fitted = True
    return model


def _responses(model: BaseItemModel, n_persons: int = 60) -> np.ndarray:
    """Simulate responses with missing cells and every extreme pattern."""
    rng = np.random.default_rng(7)
    theta = 1.5 * rng.standard_normal((n_persons, model.n_factors))
    responses = np.empty((n_persons, model.n_items), dtype=np.int_)
    for item in range(model.n_items):
        probabilities = model.probability(theta, item)
        if probabilities.ndim == 1:
            responses[:, item] = rng.random(n_persons) < probabilities
        else:
            cumulative = probabilities.cumsum(axis=1)
            responses[:, item] = (rng.random((n_persons, 1)) > cumulative).sum(axis=1)
    responses[rng.random(responses.shape) < 0.15] = -1
    top = np.asarray(model.n_categories) - 1 if model.is_polytomous else 1
    responses[0] = -1
    responses[1] = 0
    responses[2] = top
    responses[3] = -1
    responses[3, 0] = 0
    return responses


def _per_pattern(
    model: BaseItemModel,
    responses: np.ndarray,
    score_pattern: Callable[[np.ndarray], tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    patterns, inverse = unique_response_patterns(
        validate_scoring_responses(model, responses)
    )
    theta, standard_error = score_responses_parallel(
        model=model,
        responses=patterns,
        n_jobs=1,
        score_person=lambda index: score_pattern(patterns[index : index + 1]),
    )
    return theta[inverse], standard_error[inverse]


def _assert_same_uncertainty_masks(actual: np.ndarray, expected: np.ndarray) -> None:
    assert_array_equal(np.isnan(actual), np.isnan(expected))
    assert_array_equal(np.isposinf(actual), np.isposinf(expected))


def _row_function(
    offsets: np.ndarray, scales: np.ndarray, shapes: np.ndarray
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def objective(rows: np.ndarray, x: np.ndarray) -> np.ndarray:
        center, scale, shape = offsets[rows], scales[rows], shapes[rows]
        quadratic = scale * (x - center) ** 2 + 0.1 * scale * x**3
        quartic = 0.1 * (x - center) ** 4 + 0.2 * np.sin(3.0 * x)
        linear = scale * x
        return np.where(shape == 0, quadratic, np.where(shape == 1, quartic, linear))

    return objective


@pytest.mark.parametrize("xatol", [1e-5, 1e-8])
@pytest.mark.parametrize("maxiter", [500, 1, 2, 7])
def test_bounded_scalar_minimize_matches_scipy_per_row(
    xatol: float, maxiter: int
) -> None:
    rng = np.random.default_rng(1)
    n_rows = 90
    offsets = rng.normal(0.0, 3.0, n_rows)
    scales = rng.uniform(-1.0, 5.0, n_rows)
    shapes = np.arange(n_rows) % 3
    objective = _row_function(offsets, scales, shapes)
    lower, upper = -4.0, 3.0

    x, fun = bounded_scalar_minimize(
        objective, n_rows, lower, upper, xatol=xatol, maxiter=maxiter
    )

    for row in range(n_rows):
        expected = minimize_scalar(
            lambda value, row=row: float(
                objective(np.array([row]), np.array([value]))[0]
            ),
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": xatol, "maxiter": maxiter},
        )
        assert x[row] == expected.x
        assert fun[row] == expected.fun
    # Linear rows have their optimum at a bound.
    linear = (shapes == 2) & (maxiter == 500)
    assert np.all((x[linear] - lower < 1e-3) | (upper - x[linear] < 1e-3))


def test_bounded_scalar_minimize_handles_empty_batches() -> None:
    def fail(rows: np.ndarray, x: np.ndarray) -> np.ndarray:
        raise AssertionError("no rows should be evaluated")

    x, fun = bounded_scalar_minimize(fail, 0, -1.0, 1.0)

    assert x.shape == fun.shape == (0,)


def test_bounded_scalar_minimize_evaluates_only_active_rows() -> None:
    evaluated: list[int] = []

    def objective(rows: np.ndarray, x: np.ndarray) -> np.ndarray:
        evaluated.append(rows.size)
        # A parabola converges in a few steps; a linear row needs many golden
        # sections to reach its bound.
        return np.where(rows == 0, (x - 1.0) ** 2, x)

    x, _ = bounded_scalar_minimize(objective, 2, -1.0, 5.0)

    assert evaluated[0] == 2
    assert evaluated[-1] == 1
    assert_allclose(x, [1.0, -1.0], atol=1e-4)


def test_row_observed_information_matches_single_mask_information() -> None:
    for kind in ("2PL", "GRM"):
        model = _random_model(kind)
        rng = np.random.default_rng(3)
        theta = rng.normal(size=(25, 1))
        masks = rng.random((25, model.n_items)) < 0.7
        masks[0] = False

        actual = observed_test_information(model, theta, masks)
        expected = np.array(
            [
                observed_test_information(model, theta[row : row + 1], masks[row])[0]
                for row in range(theta.shape[0])
            ]
        )

        assert_allclose(actual, expected, rtol=1e-14, atol=0.0)
        assert actual[0] == 0.0
    with pytest.raises(ValueError, match="observed_mask"):
        observed_test_information(model, theta, masks[:3])


def test_score_pattern_chunks_concatenates_bounded_chunks() -> None:
    patterns = np.arange(14).reshape(7, 2)
    sizes: list[int] = []

    def score_chunk(chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sizes.append(chunk.shape[0])
        return chunk[:, 0].astype(float), chunk[:, 1].astype(float)

    theta, standard_error = score_pattern_chunks(patterns, score_chunk, chunk_size=3)

    assert sizes == [3, 3, 1]
    assert_array_equal(theta, patterns[:, 0])
    assert_array_equal(standard_error, patterns[:, 1])


@pytest.mark.parametrize("kind", ["GRM", "GPCM", "PCM", "NRM", "3PL"])
@pytest.mark.parametrize("bounds", [(-6.0, 6.0), (-4.0, 3.0)])
def test_batched_unidimensional_scorers_match_per_pattern_search(
    kind: str, bounds: tuple[float, float]
) -> None:
    model = _random_model(kind)
    responses = _responses(model)
    patterns, inverse = unique_response_patterns(
        validate_scoring_responses(model, responses)
    )

    ml = MLScorer(theta_bounds=bounds)
    ml_theta, ml_se = ml._score_unidimensional_batch(model, patterns)
    ml_ref_theta, ml_ref_se = _per_pattern(
        model, responses, lambda pattern: ml._score_unidimensional(model, pattern)
    )
    assert_allclose(ml_theta[inverse], ml_ref_theta, rtol=0.0, atol=1e-8)
    assert_allclose(ml_se[inverse], ml_ref_se, rtol=1e-8)
    _assert_same_uncertainty_masks(ml_se[inverse], ml_ref_se)

    map_scorer = MAPScorer(theta_bounds=bounds)
    map_theta, map_se = map_scorer._score_unidimensional_batch(
        model, patterns, 0.3, 1.5
    )
    map_ref_theta, map_ref_se = _per_pattern(
        model,
        responses,
        lambda pattern: map_scorer._score_unidimensional(model, pattern, 0.3, 1.5),
    )
    assert_allclose(map_theta[inverse], map_ref_theta, rtol=0.0, atol=1e-8)
    # MAP SEs are second finite differences with a 1e-5 step; last-bit changes
    # in the likelihood move them by roughly 1e-5 relative.
    assert_allclose(map_se[inverse], map_ref_se, rtol=1e-4)
    _assert_same_uncertainty_masks(map_se[inverse], map_ref_se)

    wle = WLEScorer(bounds=bounds)
    wle_theta, wle_se = wle._estimate_unidimensional_batch(model, patterns)
    wle_ref_theta, wle_ref_se = _per_pattern(
        model, responses, lambda pattern: wle._estimate_person(model, pattern[0])
    )
    assert_allclose(wle_theta[inverse], wle_ref_theta, rtol=0.0, atol=1e-8)
    assert_allclose(wle_se[inverse], wle_ref_se, rtol=1e-8)
    _assert_same_uncertainty_masks(wle_se[inverse], wle_ref_se)


@pytest.mark.parametrize(
    ("scorer", "method_name"),
    [
        (MLScorer(), "_score_unidimensional"),
        (MAPScorer(), "_score_unidimensional"),
        (WLEScorer(), "_estimate_person"),
    ],
)
def test_builtin_polytomous_scoring_uses_one_batched_search(
    monkeypatch: pytest.MonkeyPatch,
    scorer: MLScorer | MAPScorer | WLEScorer,
    method_name: str,
) -> None:
    model = _random_model("GRM")
    responses = _responses(model)

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("built-in models should not optimize per pattern")

    monkeypatch.setattr(type(scorer), method_name, fail)

    result = scorer.score(model, responses)

    assert result.theta.shape == (responses.shape[0],)
    assert np.all(np.isfinite(result.theta))


def test_batched_scorers_score_through_public_api_like_per_pattern_path() -> None:
    model = _random_model("GPCM")
    responses = np.vstack([_responses(model)] * 2)

    for scorer in (MLScorer(), MAPScorer(), WLEScorer()):
        result = scorer.score(model, responses)
        if isinstance(scorer, WLEScorer):
            reference = _per_pattern(
                model,
                responses,
                lambda pattern, scorer=scorer: scorer._estimate_person(
                    model, pattern[0]
                ),
            )
        elif isinstance(scorer, MAPScorer):
            reference = _per_pattern(
                model,
                responses,
                lambda pattern, scorer=scorer: scorer._score_unidimensional(
                    model, pattern, 0.0, 1.0
                ),
            )
        else:
            reference = _per_pattern(
                model,
                responses,
                lambda pattern, scorer=scorer: scorer._score_unidimensional(
                    model, pattern
                ),
            )
        assert_allclose(result.theta, reference[0], rtol=0.0, atol=1e-8)
        assert_allclose(result.standard_error, reference[1], rtol=1e-4)


def test_wle_information_override_keeps_per_pattern_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _random_model("GRM")
    responses = _responses(model)[:12]
    scorer = WLEScorer()
    calls = 0

    def doubled_information(model, theta, valid_mask):
        nonlocal calls
        calls += 1
        return 2.0 * observed_test_information(model, theta, valid_mask)

    monkeypatch.setattr(scorer, "_test_information", doubled_information)

    result = scorer.score(model, responses)
    default = WLEScorer().score(model, responses)

    assert calls > 0
    observed = np.any(responses >= 0, axis=1)
    assert_allclose(
        result.standard_error[observed],
        default.standard_error[observed] / np.sqrt(2.0),
        rtol=1e-6,
    )


@pytest.mark.parametrize("scorer_type", [MLScorer, MAPScorer, WLEScorer])
def test_scorer_subclasses_keep_per_pattern_hooks(scorer_type: type) -> None:
    calls = 0

    class Counting(scorer_type):  # type: ignore[misc, valid-type]
        def _score_unidimensional(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            return super()._score_unidimensional(*args, **kwargs)

        def _estimate_person(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            return super()._estimate_person(*args, **kwargs)

    model = _random_model("GRM")
    responses = _responses(model)[:10]

    Counting().score(model, responses)

    assert calls == unique_response_patterns(responses)[0].shape[0]


def test_instance_likelihood_override_keeps_per_pattern_path() -> None:
    model = _random_model("GRM")
    responses = _responses(model)[:8]
    original = model.log_likelihood
    rows_seen: list[int] = []

    def counted(responses: np.ndarray, theta: np.ndarray) -> np.ndarray:
        rows_seen.append(theta.shape[0])
        return original(responses, theta)

    model.log_likelihood = counted  # type: ignore[method-assign]

    MLScorer().score(model, responses)

    assert set(rows_seen) == {1}


def test_batched_hessian_se_matches_single_row_reference() -> None:
    rng = np.random.default_rng(11)
    n_rows = 12
    curvature = rng.uniform(0.5, 3.0, (n_rows, 3))
    coupling = rng.uniform(-0.3, 0.3, n_rows)
    curvature[0, 2] = 0.0  # A flat direction yields a singular Hessian.

    def row_objective(row: int, x: np.ndarray) -> float:
        return float(
            0.5 * np.sum(curvature[row] * x**2)
            + coupling[row] * x[0] * x[1]
            + 0.05 * x[0] ** 4
        )

    def batched(rows: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.array(
            [row_objective(int(row), point) for row, point in zip(rows, x, strict=True)]
        )

    x = rng.normal(size=(n_rows, 3))

    actual = batched_hessian_se(batched, x)
    expected = np.array(
        [
            compute_hessian_se(lambda point, row=row: row_objective(row, point), x[row])
            for row in range(n_rows)
        ]
    )

    assert_array_equal(actual, expected)
    assert np.all(np.isnan(actual[0]))
    assert np.all(np.isfinite(actual[1:]))


def test_batched_hessian_se_rejects_non_finite_objectives() -> None:
    def objective(rows: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.where(x[:, 0] > 0.0, np.inf, np.sum(x**2, axis=1))

    with pytest.raises(ValueError, match="finite"):
        batched_hessian_se(objective, np.zeros((2, 2)))


def test_batched_newton_minimize_solves_box_constrained_quadratics() -> None:
    rng = np.random.default_rng(5)
    n_rows = 40
    centers = rng.normal(0.0, 2.0, (n_rows, 2))
    factors = rng.normal(size=(n_rows, 2, 2))
    hessians = factors @ factors.transpose(0, 2, 1) + 0.5 * np.eye(2)

    def objective(rows: np.ndarray, x: np.ndarray) -> np.ndarray:
        diff = x - centers[rows]
        return 0.5 * np.einsum("ni,nij,nj->n", diff, hessians[rows], diff)

    x, fun, converged = batched_newton_minimize(
        objective, np.zeros((n_rows, 2)), -1.5, 1.5
    )

    assert np.all(converged)
    assert np.all((x >= -1.5) & (x <= 1.5))
    assert_allclose(fun, objective(np.arange(n_rows), x))
    for row in range(n_rows):
        # Projected-gradient optimality: free coordinates have zero gradient
        # and bound coordinates have gradients pointing out of the box.
        gradient = hessians[row] @ (x[row] - centers[row])
        at_lower = x[row] <= -1.5
        at_upper = x[row] >= 1.5
        free = ~(at_lower | at_upper)
        assert_allclose(gradient[free], 0.0, atol=1e-6)
        assert np.all(gradient[at_lower] >= -1e-6)
        assert np.all(gradient[at_upper] <= 1e-6)
    unconstrained = np.all(np.abs(centers) < 1.4, axis=1)
    assert_allclose(x[unconstrained], centers[unconstrained], atol=1e-7)


def test_batched_newton_minimize_reports_unusable_rows() -> None:
    def objective(rows: np.ndarray, x: np.ndarray) -> np.ndarray:
        values = np.sum((x - 0.5) ** 2, axis=1)
        return np.where(rows == 1, np.nan, values)

    x, _, converged = batched_newton_minimize(objective, np.zeros((3, 2)), -2.0, 2.0)

    assert_array_equal(converged, [True, False, True])
    assert_allclose(x[[0, 2]], 0.5, atol=1e-8)


def _map_objective(
    model: BaseItemModel,
    prior_mean: np.ndarray,
    prior_precision: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray], float]:
    def objective(theta: np.ndarray, pattern: np.ndarray) -> float:
        ll = model.log_likelihood(pattern[None, :], theta[None, :])[0]
        diff = theta - prior_mean
        return float(-(ll - 0.5 * diff @ prior_precision @ diff))

    return objective


@pytest.mark.parametrize(
    ("kind", "n_factors", "prior_cov", "bounds"),
    [
        ("2PL", 2, None, (-6.0, 6.0)),
        (
            "2PL",
            3,
            np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]]),
            (-6.0, 6.0),
        ),
        ("GRM", 2, np.array([[1.0, 0.6], [0.6, 1.5]]), (-6.0, 6.0)),
        ("GRM", 3, None, (-6.0, 6.0)),
        # Tight bounds leave many posterior modes on a face of the box.
        ("2PL", 2, np.array([[1.0, 0.7], [0.7, 1.0]]), (-0.5, 0.4)),
    ],
)
def test_batched_multidimensional_map_matches_lbfgsb(
    kind: str,
    n_factors: int,
    prior_cov: np.ndarray | None,
    bounds: tuple[float, float],
) -> None:
    model = _random_model(kind, n_items=12, n_factors=n_factors)
    responses = _responses(model, n_persons=50)
    prior_mean = np.linspace(-0.2, 0.2, n_factors)
    scorer = MAPScorer(prior_mean=prior_mean, prior_cov=prior_cov, theta_bounds=bounds)

    result = scorer.score(model, responses)

    mean, cov = resolve_prior_distribution(
        n_factors=n_factors, prior_mean=prior_mean, prior_cov=prior_cov
    )
    precision = np.linalg.inv(cov)
    reference_theta, reference_se = _per_pattern(
        model,
        responses,
        lambda pattern: scorer._score_multidimensional(model, pattern, mean, precision),
    )
    objective = _map_objective(model, mean, precision)
    batched_objective = np.array(
        [
            objective(theta, row)
            for theta, row in zip(result.theta, responses, strict=True)
        ]
    )
    reference_objective = np.array(
        [
            objective(theta, row)
            for theta, row in zip(reference_theta, responses, strict=True)
        ]
    )

    assert np.all(batched_objective <= reference_objective + 1e-9)
    assert_allclose(result.theta, reference_theta, atol=5e-4)
    assert_allclose(result.standard_error, reference_se, atol=2e-4)
    assert_array_equal(np.isnan(result.standard_error), np.isnan(reference_se))


def test_batched_multidimensional_map_resolves_stalled_rows_with_lbfgsb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mirt.scoring.map as map_module

    model = _random_model("2PL", n_items=8, n_factors=2)
    responses = _responses(model, n_persons=20)
    original = map_module.batched_newton_minimize

    def stall_first_row(*args: object, **kwargs: object):
        x, fun, converged = original(*args, **kwargs)
        converged[0] = False
        x[0] = 5.0
        return x, fun, converged

    monkeypatch.setattr(map_module, "batched_newton_minimize", stall_first_row)
    calls = 0
    reference = MAPScorer._score_multidimensional

    def counted(self, *args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return reference(self, *args, **kwargs)

    monkeypatch.setattr(MAPScorer, "_score_multidimensional", counted)

    result = MAPScorer().score(model, responses)
    patterns, inverse = unique_response_patterns(responses)
    expected = reference(MAPScorer(), model, patterns[:1], np.zeros(2), np.eye(2))

    assert calls == 1
    first = np.flatnonzero(inverse == 0)
    assert_allclose(result.theta[first], np.broadcast_to(expected[0], (first.size, 2)))


def test_multidimensional_map_override_keeps_per_pattern_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _random_model("2PL", n_items=6, n_factors=2)
    responses = _responses(model, n_persons=6)
    scorer = MAPScorer()
    calls = 0
    original = scorer._score_multidimensional

    def counted(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(scorer, "_score_multidimensional", counted)

    scorer.score(model, responses)

    assert calls == unique_response_patterns(responses)[0].shape[0]


def test_multimodal_noncompensatory_map_keeps_lbfgsb_search() -> None:
    # Noncompensatory posteriors can be multimodal. Newton from the prior mean
    # settles here on a mode whose objective is about 0.85 worse.
    model = NoncompensatoryModel(6, n_factors=2)
    model.set_parameters(
        discrimination=np.array(
            [[1.9, 2.1], [1.2, 0.6], [1.6, 0.8], [1.9, 1.2], [1.4, 2.5], [2.1, 2.2]]
        ),
        difficulty=np.array(
            [[1.1, -0.8], [0.3, 0.3], [0.5, 2.1], [0.8, 2.5], [1.0, -0.3], [-2.3, -2.3]]
        ),
    )
    model._is_fitted = True
    responses = np.array([[0, 0, 0, 1, 1, 0]])
    prior_cov = 100.0 * np.array([[1.0, 0.5], [0.5, 1.0]])
    scorer = MAPScorer(prior_cov=prior_cov, theta_bounds=(-4.0, 4.0))
    mean, cov = resolve_prior_distribution(
        n_factors=2, prior_mean=None, prior_cov=prior_cov
    )
    precision = np.linalg.inv(cov)

    result = scorer.score(model, responses)
    expected = scorer._score_multidimensional(model, responses, mean, precision)
    newton = scorer._score_multidimensional_batch(model, responses, mean, precision)

    assert_array_equal(result.theta[0], expected[0])
    assert_array_equal(result.standard_error[0], expected[1])
    objective = _map_objective(model, mean, precision)
    assert objective(newton[0][0], responses[0]) > objective(expected[0], responses[0])


def test_batched_multidimensional_map_supports_empty_batches() -> None:
    model = _random_model("2PL", n_items=4, n_factors=2)

    result = MAPScorer().score(model, np.empty((0, 4), dtype=int))

    assert result.theta.shape == (0, 2)
    assert result.standard_error.shape == (0, 2)
