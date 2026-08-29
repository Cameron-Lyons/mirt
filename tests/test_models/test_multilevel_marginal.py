"""Marginal response-pattern likelihoods for hierarchical models."""

import numpy as np
import pytest

from mirt.models.dichotomous import TwoParameterLogistic
from mirt.models.multilevel import MultilevelIRTModel, ThreeLevelIRTModel
from mirt.models.polytomous import GradedResponseModel
from mirt.utils.numeric import standard_normal_quadrature


def _manual_marginal(
    model: MultilevelIRTModel | ThreeLevelIRTModel,
    responses: np.ndarray,
    n_quadpts: int,
) -> np.ndarray:
    nodes, weights = standard_normal_quadrature(n_quadpts)
    means = model.person_prior_mean()
    variance = (
        model.within_variance
        if isinstance(model, MultilevelIRTModel)
        else model.variance_components["within"]
    )
    expected = np.empty(len(responses), dtype=np.float64)
    for person, mean in enumerate(means):
        theta = (mean + np.sqrt(variance) * nodes).reshape(-1, 1)
        repeated = np.broadcast_to(responses[person], (n_quadpts, model.n_items))
        conditional = model.base_model.log_likelihood(repeated, theta)
        maximum = np.max(conditional)
        expected[person] = maximum + np.log(weights @ np.exp(conditional - maximum))
    return expected


def test_two_level_marginal_likelihood_matches_personwise_quadrature() -> None:
    base_model = TwoParameterLogistic(n_items=3)
    base_model.set_parameters(
        discrimination=np.array([0.8, 1.2, 1.6]),
        difficulty=np.array([-0.4, 0.2, 0.9]),
    )
    model = MultilevelIRTModel(base_model, np.array([9, 2, 9, 2]))
    model.set_group_means(np.array([-0.5, 0.75]))
    model.set_variance_components(between=0.4, within=0.65)
    responses = np.array([[1, 0, 1], [0, 1, -9], [1, 1, 0], [0, 0, 1]])

    expected = _manual_marginal(model, responses, 19)
    actual = model.marginal_log_likelihoods(responses, 19)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert model.marginal_log_likelihood(responses, 19) == pytest.approx(expected.sum())


def test_three_level_marginal_likelihood_supports_polytomous_models() -> None:
    base_model = GradedResponseModel(n_items=2, n_categories=3)
    base_model.set_parameters(
        discrimination=np.array([1.1, 0.7]),
        thresholds=np.array([[-0.6, 0.5], [-0.2, 0.9]]),
    )
    model = ThreeLevelIRTModel(
        base_model,
        level2_membership=np.array([10, 10, 20, 30]),
        level3_membership=np.array([7, 7, 9]),
    )
    model.set_level_effects(
        level2=np.array([-0.4, 0.2, 0.8]),
        level3=np.array([0.3, -0.5]),
    )
    model.set_variance_components(level2=0.2, level3=0.1, within=0.7)
    responses = np.array([[0, 2], [1, -1], [2, 1], [1, 0]])

    expected = _manual_marginal(model, responses, 17)
    actual = model.marginal_log_likelihoods(responses, 17, chunk_size=1)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert model.marginal_log_likelihood(responses, 17, chunk_size=2) == pytest.approx(
        expected.sum()
    )


def test_marginal_likelihood_is_chunk_invariant_and_handles_all_missing() -> None:
    model = MultilevelIRTModel(
        TwoParameterLogistic(n_items=4),
        np.array([0, 0, 1, 1, 2, 2]),
    )
    model.set_group_means(np.array([0.0, 0.0, 0.7]))
    responses = np.array(
        [
            [1, 0, 1, 0],
            [-1, -1, -1, -1],
            [0, 1, 0, 1],
            [1, 1, 0, -1],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ]
    )

    one_at_a_time = model.marginal_log_likelihoods(responses, 13, chunk_size=1)
    batched = model.marginal_log_likelihoods(responses, 13, chunk_size=20)

    np.testing.assert_array_equal(one_at_a_time, batched)
    assert batched[1] == pytest.approx(0.0, abs=1e-14)


def test_equal_prior_means_share_a_quadrature_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MultilevelIRTModel(
        TwoParameterLogistic(n_items=2),
        np.array([0, 1, 2, 3]),
    )
    model.set_group_means(np.array([-0.3, -0.3, 0.6, 0.6]))
    responses = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    original = model.base_model.log_likelihood_batch
    batch_sizes: list[int] = []

    def counted_batch(response_batch: np.ndarray, theta: np.ndarray) -> np.ndarray:
        batch_sizes.append(len(response_batch))
        return original(response_batch, theta)

    monkeypatch.setattr(model.base_model, "log_likelihood_batch", counted_batch)

    model.marginal_log_likelihoods(responses, 11)

    assert batch_sizes == [2, 2]


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        (np.array([0, 1]), "two-dimensional"),
        (np.zeros((2, 3), dtype=int), "shape"),
        (np.array([[0, 2], [1, 0]]), "0, 1"),
        (np.array([[0.5, 1.0], [1.0, 0.0]]), "integer"),
        (np.array([[0.0, np.nan], [1.0, 0.0]]), "finite"),
    ],
)
def test_marginal_likelihood_validates_responses(
    responses: np.ndarray,
    message: str,
) -> None:
    model = MultilevelIRTModel(TwoParameterLogistic(n_items=2), np.array([0, 1]))

    with pytest.raises(ValueError, match=message):
        model.marginal_log_likelihoods(responses)


@pytest.mark.parametrize("n_quadpts", [0, -1, 2.5, True])
def test_marginal_likelihood_validates_quadrature_count(n_quadpts: object) -> None:
    model = MultilevelIRTModel(TwoParameterLogistic(n_items=2), np.array([0, 1]))

    with pytest.raises(ValueError, match="n_quadpts"):
        model.marginal_log_likelihoods(
            np.zeros((2, 2), dtype=int),
            n_quadpts=n_quadpts,  # type: ignore[arg-type]
        )


def test_marginal_likelihood_requires_unidimensional_base_model() -> None:
    model = MultilevelIRTModel(
        TwoParameterLogistic(n_items=2, n_factors=2),
        np.array([0, 1]),
    )

    with pytest.raises(ValueError, match="unidimensional"):
        model.marginal_log_likelihoods(np.zeros((2, 2), dtype=int))


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_marginal_likelihood_validates_chunk_size(chunk_size: object) -> None:
    model = MultilevelIRTModel(TwoParameterLogistic(n_items=2), np.array([0, 1]))

    with pytest.raises(ValueError, match="chunk_size"):
        model.marginal_log_likelihoods(
            np.zeros((2, 2), dtype=int),
            chunk_size=chunk_size,  # type: ignore[arg-type]
        )
