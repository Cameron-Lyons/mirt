"""Tests for conditional and random-effect testlet models."""

import warnings
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from mirt import TestletModel, create_testlet_structure
from mirt.estimation.em import EMEstimator
from mirt.exceptions import MirtDataError, MirtValidationError
from mirt.models.testlet import (
    BifactorTestletModel,
    RandomTestletEffectsModel,
    compute_testlet_q3,
)


def _normal_quadrature(n_quadpts: int) -> tuple[NDArray, NDArray]:
    nodes, weights = np.polynomial.hermite.hermgauss(n_quadpts)
    return nodes * np.sqrt(2.0), weights / np.sqrt(np.pi)


def _manual_marginal_probability(
    theta: NDArray[np.float64],
    *,
    discrimination: float,
    loading: float,
    difficulty: float,
    variance: float,
    n_quadpts: int,
) -> NDArray[np.float64]:
    nodes, weights = _normal_quadrature(n_quadpts)
    linear = (
        discrimination * theta[:, None]
        + loading * np.sqrt(variance) * nodes[None, :]
        - difficulty
    )
    return (1.0 / (1.0 + np.exp(-linear))) @ weights


class TestTestletConfiguration:
    """Structure, labels, parameters, and copy behavior."""

    def test_sparse_labels_map_to_contiguous_factor_positions(self) -> None:
        model = TestletModel(5, [2, 2, -1, 7, 7])

        assert model.n_testlets == 2
        assert model.n_factors == 3
        np.testing.assert_array_equal(model.testlet_labels, np.array([2, 7]))
        np.testing.assert_array_equal(
            model.testlet_membership, np.array([2, 2, -1, 7, 7])
        )

    def test_membership_property_is_independent(self) -> None:
        model = TestletModel(3, [0, 0, -1])
        exposed = model.testlet_membership

        exposed[:] = 9

        np.testing.assert_array_equal(model.testlet_membership, np.array([0, 0, -1]))

    @pytest.mark.parametrize(
        "membership",
        [
            [0, 0],
            [[0, 0, 1]],
            [0.0, 0.5, 1.0],
            [0, -2, 1],
            [True, False, True],
        ],
    )
    def test_invalid_membership_is_rejected(self, membership: object) -> None:
        with pytest.raises(MirtValidationError, match="testlet"):
            TestletModel(3, membership)

    @pytest.mark.parametrize("n_quadpts", [0, -1, 2.5, True])
    def test_invalid_quadrature_configuration_is_rejected(
        self, n_quadpts: object
    ) -> None:
        with pytest.raises(MirtValidationError, match="n_quadpts"):
            TestletModel(2, [0, 0], n_quadpts=n_quadpts)

    def test_variance_update_uses_external_label(self) -> None:
        model = TestletModel(4, [2, 2, 7, 7])

        model.set_testlet_variance(7, 0.35)

        np.testing.assert_allclose(model.testlet_variances, np.array([1.0, 0.35]))
        with pytest.raises(MirtValidationError, match="Unknown"):
            model.set_testlet_variance(1, 0.2)

    def test_parameter_update_is_atomic(self) -> None:
        model = TestletModel(3, [0, 0, -1])
        before = model.parameters
        invalid_loading = model.testlet_loadings.copy()
        invalid_loading[2] = 0.5

        with pytest.raises(MirtValidationError, match="standalone"):
            model.set_parameters(
                discrimination=np.full(3, 2.0),
                testlet_loadings=invalid_loading,
            )

        for name, values in before.items():
            np.testing.assert_array_equal(model.parameters[name], values)

    @pytest.mark.parametrize(
        ("name", "value", "message"),
        [
            ("testlet_variances", np.array([-0.1]), "non-negative"),
            ("difficulty", np.array([0.0, np.nan]), "finite"),
            ("discrimination", np.array([1.0]), "Shape"),
        ],
    )
    def test_parameter_validation(
        self, name: str, value: NDArray[np.float64], message: str
    ) -> None:
        model = TestletModel(2, [0, 0])

        with pytest.raises(MirtValidationError, match=message):
            model.set_parameters(**{name: value})

    def test_reliability_reflects_testlet_variance(self) -> None:
        model = TestletModel(3, [4, 4, 4])
        model.set_testlet_variance(4, 0.0)
        without_effect = model.testlet_reliability()[4]

        model.set_testlet_variance(4, 2.0)
        with_effect = model.testlet_reliability()[4]

        assert with_effect > without_effect

    def test_copy_preserves_quadrature_and_is_independent(self) -> None:
        model = TestletModel(4, [2, 2, 7, 7], n_quadpts=19)
        copied = model.copy()

        copied.set_testlet_variance(2, 0.2)

        assert copied.n_quadpts == 19
        assert model.testlet_variances[0] == 1.0
        assert copied.testlet_variances[0] == 0.2


class TestTestletProbability:
    """Conditional and marginal probability calculations."""

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: TestletModel(5, [2, 2, -1, 7, 7], n_quadpts=17),
            lambda: BifactorTestletModel(5, [2, 2, -1, 7, 7], n_quadpts=17),
            lambda: RandomTestletEffectsModel(
                5,
                [2, 2, -1, 7, 7],
                n_quadpts=17,
            ),
        ],
    )
    @pytest.mark.parametrize("general_only", [True, False])
    def test_probability_pairs_match_itemwise_evaluation_without_dispatch(
        self,
        factory: Callable[[], TestletModel],
        general_only: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = factory().set_parameters(
            discrimination=np.array([0.8, 1.2, 0.7, 1.5, 0.9]),
            testlet_loadings=np.array([0.4, -0.6, 0.0, 0.8, 0.3]),
            difficulty=np.array([-0.7, 0.2, 0.5, -0.1, 1.0]),
            testlet_variances=np.array([0.6, 1.4]),
        )
        theta_general = np.array([-1.2, -0.5, 0.0, 0.4, 0.9, 1.6, 0.2])
        if general_only:
            theta = theta_general
            empty_theta = np.array([])
        else:
            theta = np.column_stack(
                [
                    theta_general,
                    np.array([0.3, -0.8, 0.1, 0.5, -0.2, 1.0, -0.4]),
                    np.array([-0.6, 0.2, 0.7, -0.3, 0.4, 0.8, 0.0]),
                ]
            )
            empty_theta = np.empty((0, model.n_factors))
        item_indices = np.array([4, 0, 2, 3, 1, 4, 2])
        expected = np.array(
            [
                model.probability(theta[row : row + 1], int(item_idx))[0]
                for row, item_idx in enumerate(item_indices)
            ]
        )

        def fail_item_dispatch(*args: object, **kwargs: object) -> None:
            raise AssertionError("paired evaluation dispatched through probability()")

        monkeypatch.setattr(model, "probability", fail_item_dispatch)

        actual = model.probability_pairs(theta, item_indices)
        empty = model.probability_pairs(
            empty_theta,
            np.array([], dtype=np.int_),
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
        assert empty.shape == (0,)

    def test_marginal_probability_pairs_preserve_results_across_chunks(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        model = TestletModel(5, [2, 2, -1, 7, 7], n_quadpts=7)
        theta = np.linspace(-2.0, 2.0, 11)
        item_indices = np.array([4, 0, 2, 3, 1, 4, 2, 0, 3, 1, 4])
        all_probabilities = model.probability(theta)
        expected = all_probabilities[np.arange(theta.size), item_indices]
        monkeypatch.setattr(
            "mirt.models.testlet._TESTLET_PAIR_TARGET_ELEMENTS",
            2 * model.n_quadpts,
        )

        actual = model.probability_pairs(theta, item_indices)

        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)

    def test_public_general_only_probability_matches_quadrature(self) -> None:
        model = TestletModel(2, [4, 4], n_quadpts=31)
        model.set_parameters(
            discrimination=np.array([1.3, 0.8]),
            testlet_loadings=np.array([0.7, -0.4]),
            difficulty=np.array([0.2, -0.6]),
            testlet_variances=np.array([1.5]),
        )
        theta = np.array([-1.2, 0.0, 1.4])

        expected = _manual_marginal_probability(
            theta,
            discrimination=1.3,
            loading=0.7,
            difficulty=0.2,
            variance=1.5,
            n_quadpts=31,
        )
        np.testing.assert_allclose(model.probability(theta, 0), expected)

    def test_noncontiguous_labels_work_in_marginal_probability(self) -> None:
        model = TestletModel(4, [2, 2, 9, 9])

        probability = model.probability(np.array([-1.0, 0.0, 1.0]))

        assert probability.shape == (3, 4)
        assert np.all(np.isfinite(probability))

    def test_zero_variance_reduces_to_logistic_model(self) -> None:
        model = TestletModel(2, [0, 0], n_quadpts=21)
        model.set_parameters(
            discrimination=np.array([1.4, 0.9]),
            testlet_loadings=np.array([0.8, -0.3]),
            difficulty=np.array([0.2, -0.5]),
            testlet_variances=np.array([0.0]),
        )
        theta = np.array([-1.0, 0.0, 1.0])
        expected = 1.0 / (
            1.0
            + np.exp(
                -(
                    theta[:, None] * model.discrimination[None, :]
                    - model.difficulty[None, :]
                )
            )
        )

        np.testing.assert_allclose(model.probability(theta), expected)

    def test_full_probability_matches_conditional_equation(self) -> None:
        model = TestletModel(3, [2, -1, 7])
        model.set_parameters(
            discrimination=np.array([1.2, 0.8, 1.5]),
            testlet_loadings=np.array([0.6, 0.0, -0.4]),
            difficulty=np.array([0.1, -0.3, 0.7]),
        )
        theta = np.array([[-0.5, 0.3, -0.8], [1.0, -0.4, 0.6]])
        linear = np.column_stack(
            [
                1.2 * theta[:, 0] + 0.6 * theta[:, 1] - 0.1,
                0.8 * theta[:, 0] + 0.3,
                1.5 * theta[:, 0] - 0.4 * theta[:, 2] - 0.7,
            ]
        )
        expected = 1.0 / (1.0 + np.exp(-linear))

        np.testing.assert_allclose(model.probability(theta), expected)
        np.testing.assert_allclose(model.probability(theta, 2), expected[:, 2])

    def test_marginal_information_matches_numerical_derivative(self) -> None:
        model = TestletModel(1, [3], n_quadpts=31)
        model.set_parameters(
            discrimination=np.array([1.6]),
            testlet_loadings=np.array([0.9]),
            difficulty=np.array([0.2]),
            testlet_variances=np.array([1.3]),
        )
        theta = np.array([-1.0, 0.3, 1.2])
        step = 1e-6
        probability = model.probability(theta, 0)
        derivative = (
            model.probability(theta + step, 0) - model.probability(theta - step, 0)
        ) / (2.0 * step)
        expected = derivative**2 / (probability * (1.0 - probability))

        np.testing.assert_allclose(model.information(theta, 0), expected, rtol=1e-7)

    def test_conditional_information_includes_both_factor_loadings(self) -> None:
        model = TestletModel(2, [0, -1])
        model.set_parameters(
            discrimination=np.array([1.2, 0.8]),
            testlet_loadings=np.array([0.5, 0.0]),
        )
        theta = np.zeros((1, 2))

        expected = np.array([[(1.2**2 + 0.5**2) * 0.25, 0.8**2 * 0.25]])
        np.testing.assert_allclose(model.information(theta), expected)

    @pytest.mark.parametrize("item_idx", [-1, 3, 1.2, True])
    def test_invalid_item_index_is_rejected(self, item_idx: object) -> None:
        model = TestletModel(3, [0, 0, -1])

        with pytest.raises(IndexError):
            model.probability(np.array([0.0]), item_idx=item_idx)

    @pytest.mark.parametrize(
        "theta",
        [np.zeros((2, 2)), np.zeros((2, 2, 1)), np.array([np.nan])],
    )
    def test_invalid_theta_is_rejected(self, theta: NDArray[np.float64]) -> None:
        model = TestletModel(4, [0, 0, 1, 1])

        with pytest.raises(MirtValidationError, match="theta"):
            model.probability(theta)


class TestRandomTestletLikelihood:
    """Joint random-effect integration and likelihood grids."""

    def test_missing_response_is_skipped_only_for_its_person(self) -> None:
        model = RandomTestletEffectsModel(2, [-1, -1])
        responses = np.array([[1, 1], [-1, 0]])

        actual = model.integrate_out_testlet_effects(responses, np.zeros(2))

        np.testing.assert_allclose(actual, np.array([2 * np.log(0.5), np.log(0.5)]))

    def test_testlet_joint_likelihood_matches_direct_quadrature(self) -> None:
        model = RandomTestletEffectsModel(2, [4, 4], n_quadpts=31)
        model.set_parameters(
            discrimination=np.array([1.2, 0.9]),
            testlet_loadings=np.array([0.7, -0.4]),
            difficulty=np.array([0.1, -0.3]),
            testlet_variances=np.array([1.4]),
        )
        responses = np.array([[1.0, 0.0], [np.nan, 1.0]])
        theta = np.array([-0.6, 0.8])
        nodes, weights = _normal_quadrature(31)
        expected = []
        for person in range(2):
            likelihood = np.ones(31)
            for item in range(2):
                if np.isnan(responses[person, item]):
                    continue
                linear = (
                    model.discrimination[item] * theta[person]
                    + model.testlet_loadings[item]
                    * np.sqrt(model.testlet_variances[0])
                    * nodes
                    - model.difficulty[item]
                )
                probability = 1.0 / (1.0 + np.exp(-linear))
                likelihood *= np.where(
                    responses[person, item] == 1.0,
                    probability,
                    1.0 - probability,
                )
            expected.append(np.log(likelihood @ weights))

        np.testing.assert_allclose(
            model.integrate_out_testlet_effects(responses, theta), expected
        )

    def test_long_test_likelihood_does_not_underflow(self) -> None:
        n_items = 5000
        model = RandomTestletEffectsModel(n_items, [0] * n_items)
        model.set_all_testlet_variances(0.0)
        responses = np.ones((1, n_items), dtype=int)

        likelihood = model.integrate_out_testlet_effects(responses, np.array([0.0]))

        np.testing.assert_allclose(likelihood, n_items * np.log(0.5), rtol=1e-12)

    def test_general_log_likelihood_uses_joint_marginal(self) -> None:
        model = RandomTestletEffectsModel(3, [2, 2, -1])
        responses = np.array([[1, 0, 1], [0, 1, -1]])
        theta = np.array([-0.4, 0.7])

        np.testing.assert_allclose(
            model.log_likelihood(responses, theta),
            model.integrate_out_testlet_effects(responses, theta),
        )

    def test_conditional_log_likelihood_matches_manual_selection(self) -> None:
        model = RandomTestletEffectsModel(2, [4, 4])
        responses = np.array([[1, 0], [0, 1]])
        theta = np.array([[-0.5, 0.3], [0.8, -0.2]])
        probability = model.probability(theta)
        expected = np.array(
            [
                np.log(probability[0, 0]) + np.log1p(-probability[0, 1]),
                np.log1p(-probability[1, 0]) + np.log(probability[1, 1]),
            ]
        )

        np.testing.assert_allclose(model.log_likelihood(responses, theta), expected)

    def test_batch_marginal_likelihood_matches_personwise_values(self) -> None:
        model = RandomTestletEffectsModel(4, [2, 2, 7, -1], n_quadpts=15)
        responses = np.array([[1, 0, 1, 0], [0, 1, -1, 1], [1, 1, 0, 0]])
        grid = np.array([-1.2, 0.0, 1.1])

        expected = np.column_stack(
            [
                model.log_likelihood(responses, np.full(len(responses), point))
                for point in grid
            ]
        )
        np.testing.assert_allclose(
            model.log_likelihood_batch(responses, grid), expected
        )

    @pytest.mark.parametrize(
        "responses",
        [
            np.array([0, 1]),
            np.array([[0, 1, 0]]),
            np.array([[0.5, 1.0]]),
            np.array([[0.0, np.inf]]),
            np.array([[0.0, -np.inf]]),
            np.array([[2, 0]]),
        ],
    )
    def test_invalid_responses_are_rejected(
        self, responses: NDArray[np.float64]
    ) -> None:
        model = RandomTestletEffectsModel(2, [0, 0])

        with pytest.raises(MirtDataError):
            model.log_likelihood(responses, np.zeros(len(responses)))

    def test_person_count_mismatch_is_rejected(self) -> None:
        model = RandomTestletEffectsModel(2, [0, 0])

        with pytest.raises(MirtDataError, match="same number"):
            model.log_likelihood(np.zeros((2, 2), dtype=int), np.zeros(3))

    def test_variance_estimation_handles_itemwise_missing_data(self) -> None:
        model = RandomTestletEffectsModel(3, [4, 4, 4])
        responses = np.array([[1.0, 0.0, 1.0], [0.0, np.nan, 1.0], [1.0, 1.0, -1.0]])

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            variance = model.estimate_testlet_variances(responses, np.zeros(3))

        assert variance.shape == (1,)
        assert np.all(np.isfinite(variance))
        assert np.all(variance >= 0.0)


class TestBifactorTestletValidation:
    """Bifactor setters preserve constraints and caller-owned arrays."""

    def test_testlet_loading_setter_does_not_mutate_input(self) -> None:
        model = BifactorTestletModel(3, [0, 0, -1])
        loadings = np.array([0.3, 0.7, 0.9])
        original = loadings.copy()

        model.set_testlet_loadings(loadings)

        np.testing.assert_array_equal(loadings, original)
        np.testing.assert_array_equal(model.testlet_loadings, np.array([0.3, 0.7, 0.0]))

    def test_nonfinite_loading_is_rejected_atomically(self) -> None:
        model = BifactorTestletModel(2, [0, 0])
        before = model.general_loadings

        with pytest.raises(MirtValidationError, match="finite"):
            model.set_general_loadings(np.array([1.0, np.nan]))

        np.testing.assert_array_equal(model.general_loadings, before)

    def test_copy_preserves_quadrature_configuration(self) -> None:
        model = BifactorTestletModel(2, [0, 0], n_quadpts=17)

        assert model.copy().n_quadpts == 17


class TestTestletDiagnostics:
    """Structure construction and missing-aware Q3 diagnostics."""

    @pytest.mark.parametrize(
        "sizes",
        [[2, 0, 2], [2, -1, 3], [2, 1.5, 2], [2, True, 2], []],
    )
    def test_structure_rejects_invalid_sizes(self, sizes: list[object]) -> None:
        with pytest.raises(MirtValidationError):
            create_testlet_structure(5, sizes)

    def test_q3_uses_pairwise_complete_responses(self) -> None:
        responses = np.array(
            [
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, np.nan, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, -1.0],
            ]
        )
        theta = np.linspace(-0.8, 0.8, 5)
        result = compute_testlet_q3(
            responses,
            theta,
            np.ones(3),
            np.zeros(3),
            np.array([4, 4, 9]),
        )
        expected_probability = 1.0 / (1.0 + np.exp(-theta))
        pair_observed = np.isfinite(responses[:, 0]) & np.isfinite(responses[:, 1])
        first = responses[pair_observed, 0] - expected_probability[pair_observed]
        second = responses[pair_observed, 1] - expected_probability[pair_observed]
        expected = np.corrcoef(first, second)[0, 1]

        q3 = result["q3_matrix"]
        assert isinstance(q3, np.ndarray)
        assert q3[0, 1] == pytest.approx(expected)

    def test_q3_constant_residuals_return_nan_without_warning(self) -> None:
        responses = np.ones((5, 2), dtype=int)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = compute_testlet_q3(
                responses,
                np.zeros(5),
                np.ones(2),
                np.zeros(2),
                np.array([0, 0]),
            )

        q3 = result["q3_matrix"]
        assert isinstance(q3, np.ndarray)
        assert np.isnan(q3[0, 1])

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda values: values.__setitem__("theta", np.zeros(2)),
            lambda values: values.__setitem__("discrimination", np.ones(2)),
            lambda values: values.__setitem__("membership", np.array([0, -2, 1])),
        ],
    )
    def test_q3_validates_dimensions(
        self, mutate: Callable[[dict[str, NDArray]], None]
    ) -> None:
        values = {
            "responses": np.zeros((3, 3), dtype=int),
            "theta": np.zeros(3),
            "discrimination": np.ones(3),
            "difficulty": np.zeros(3),
            "membership": np.array([0, 0, 1]),
        }
        mutate(values)

        with pytest.raises((MirtDataError, MirtValidationError)):
            compute_testlet_q3(
                values["responses"],
                values["theta"],
                values["discrimination"],
                values["difficulty"],
                values["membership"],
            )


@pytest.mark.parametrize(
    ("n_items", "membership"), [(3, [0, 1, 2]), (6, [0, 0, 0, 1, 1, 1])]
)
def test_generic_em_standard_errors_hold_testlet_variances_fixed(
    n_items: int, membership: list[int]
) -> None:
    """Complete-data errors skip model-level variances, as the M-step does.

    Three items in three testlets give a variance vector as long as the item
    count, which must still not be read as one value per item.
    """
    from mirt.estimation.em import EMEstimator

    model = TestletModel(n_items=n_items, testlet_membership=np.array(membership))
    responses = np.random.default_rng(0).integers(0, 2, (300, n_items))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Rectangular quadrature keeps the product grid, whose "auto" errors
        # are itemwise complete-data curvatures.
        result = EMEstimator(n_quadpts=9, max_iter=20, quadrature="rectangular").fit(
            model, responses
        )

    errors = result.standard_errors
    assert result.se_method == "complete_data"
    np.testing.assert_array_equal(errors["testlet_variances"], 0.0)
    for name in ("discrimination", "difficulty", "testlet_loadings"):
        assert errors[name].shape == (n_items,)
        assert np.all(np.isfinite(errors[name])) and np.all(errors[name] > 0)


def _simulated_testlet_responses(
    seed: int,
    n_persons: int,
    membership: list[int],
    loading: float = 0.9,
) -> NDArray[np.int_]:
    """Draw responses from a testlet model with unit testlet variances."""
    rng = np.random.default_rng(seed)
    n_items = len(membership)
    model = TestletModel(n_items, membership)
    in_testlet = np.asarray(membership) >= 0
    model.set_parameters(
        discrimination=rng.uniform(0.8, 1.6, n_items),
        testlet_loadings=np.where(in_testlet, loading, 0.0),
        difficulty=rng.normal(0.0, 0.8, n_items),
    )
    probability = model.probability(rng.standard_normal((n_persons, model.n_factors)))
    return (rng.random(probability.shape) < probability).astype(int)


class TestTestletVarianceIdentification:
    """Loadings and variances enter only as ``d_j * sigma_t``."""

    def test_full_theta_curves_scale_standardized_effects(self) -> None:
        model = TestletModel(4, [0, 0, 1, 1])
        loadings = np.array([0.8, 0.6, 1.0, 0.5])
        model.set_parameters(testlet_loadings=loadings)
        rescaled = model.copy().set_parameters(
            testlet_variances=np.array([4.0, 4.0]), testlet_loadings=loadings / 2
        )
        theta = np.random.default_rng(0).standard_normal((7, 3))

        np.testing.assert_allclose(
            rescaled.probability(theta), model.probability(theta)
        )
        np.testing.assert_allclose(
            rescaled.probability(theta[:, 0]), model.probability(theta[:, 0])
        )
        np.testing.assert_allclose(
            rescaled.information(theta), model.information(theta)
        )

    def test_variance_is_free_only_with_every_loading_fixed(self) -> None:
        model = TestletModel(5, [0, 0, 1, 1, -1])
        assert not model.free_parameter_masks["testlet_variances"].any()
        assert model.n_parameters == 5 + 4 + 5

        fixed = np.array([False, False, True, True, False])
        model.set_free_parameter_masks({"testlet_loadings": fixed})
        np.testing.assert_array_equal(
            model.free_parameter_masks["testlet_variances"], [True, False]
        )
        assert model.n_parameters == 5 + 2 + 5 + 1

        model.set_free_parameter_masks(
            {"testlet_loadings": fixed, "testlet_variances": np.array([False, False])}
        )
        assert not model.free_parameter_masks["testlet_variances"].any()
        with pytest.raises(MirtValidationError, match="model-family fixed"):
            model.set_free_parameter_masks(
                {"testlet_variances": np.array([True, False])}
            )

    def test_em_curves_agree_with_a_fixed_starting_variance(self) -> None:
        membership = [0, 0, 0, 0, 1, 1, 1, 1, -1, -1]
        responses = _simulated_testlet_responses(2, 1000, membership)
        model = RandomTestletEffectsModel(10, membership)
        model.set_all_testlet_variances(2.5)

        result = EMEstimator(n_quadpts=11).fit(model, responses, start="model")

        # The variance stays at its value and the loadings absorb it, so the
        # general-only curves reproduce each item's observed proportion.
        np.testing.assert_array_equal(result.model.testlet_variances, 2.5)
        assert result.n_parameters == 28
        nodes, weights = _normal_quadrature(41)
        implied = weights @ result.model.probability(nodes)
        np.testing.assert_allclose(implied, responses.mean(axis=0), atol=0.01)

    def test_em_estimates_variances_of_testlets_with_fixed_loadings(self) -> None:
        membership = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        responses = _simulated_testlet_responses(4, 2000, membership, loading=1.0)
        model = RandomTestletEffectsModel(10, membership)
        model.set_parameters(testlet_loadings=np.ones(10))
        model.set_free_parameter_masks({"testlet_loadings": np.zeros(10, dtype=bool)})

        result = EMEstimator(n_quadpts=15).fit(model, responses)

        variances = result.model.testlet_variances
        errors = result.standard_errors["testlet_variances"]
        assert result.converged
        assert result.n_parameters == 10 + 10 + 2
        np.testing.assert_allclose(variances, 1.0, atol=3 * errors.max())
        assert np.all(errors > 0.02) and np.all(errors < 0.5)

    def test_method_of_moments_variances_match_the_generating_values(self) -> None:
        rng = np.random.default_rng(9)
        membership = [0] * 5 + [1] * 5
        for variance in (0.25, 1.0, 4.0):
            model = RandomTestletEffectsModel(10, membership)
            model.set_parameters(
                discrimination=np.full(10, 1.2),
                testlet_loadings=np.ones(10),
                difficulty=np.linspace(-1.0, 1.0, 10),
                testlet_variances=np.full(2, variance),
            )
            theta = rng.standard_normal(8000)
            full = np.column_stack([theta, rng.standard_normal((8000, 2))])
            probability = model.probability(full)
            responses = (rng.random(probability.shape) < probability).astype(int)

            estimate = model.estimate_testlet_variances(responses, theta)

            np.testing.assert_allclose(estimate, variance, rtol=0.2)


class TestTestletEstimation:
    """Testlet models are fitted as bifactor models on reduced grids."""

    def test_em_reduction_matches_the_product_grid_likelihood(self) -> None:
        from mirt.estimation.quadrature import GaussHermiteQuadrature

        membership = [0, 0, 0, 1, 1, 1, -1]
        responses = _simulated_testlet_responses(5, 400, membership)
        result = EMEstimator(n_quadpts=9).fit(TestletModel(7, membership), responses)

        quadrature = GaussHermiteQuadrature(n_points=9, n_dimensions=3)
        log_likelihood = result.model.log_likelihood_batch(responses, quadrature.nodes)
        log_mass = np.log(quadrature.weights / quadrature.weights.sum())
        exact = np.sum(np.logaddexp.reduce(log_likelihood + log_mass, axis=1))
        assert result.se_method == "oakes"
        np.testing.assert_allclose(result.log_likelihood, exact, rtol=0, atol=1e-8)

    def test_em_fits_many_testlets_with_default_settings(self) -> None:
        membership = create_testlet_structure(30, [5] * 6)
        responses = _simulated_testlet_responses(3, 1500, list(membership), 0.8)

        result = EMEstimator().fit(RandomTestletEffectsModel(30, membership), responses)

        loadings = result.model.testlet_loadings
        assert result.converged
        assert result.n_parameters == 90
        assert np.sqrt(np.mean((loadings - 0.8) ** 2)) < 0.3

    def test_em_estimates_one_common_loading_per_testlet(self) -> None:
        membership = [0, 0, 0, 0, 1, 1, 1, 1, -1, -1]
        responses = _simulated_testlet_responses(7, 800, membership)
        fits = [
            EMEstimator(n_quadpts=9, tol=1e-6, **options).fit(
                BifactorTestletModel(10, membership, constrain_testlet_loadings=True),
                responses,
            )
            # SQUAREM keeps the product grid of the same Gauss-Hermite rule.
            for options in (
                {},
                {"quadrature": "gauss-hermite", "accelerate": "squarem"},
            )
        ]

        for result in fits:
            loadings = result.model.testlet_loadings
            errors = result.standard_errors["testlet_loadings"]
            assert result.converged
            assert result.n_parameters == 10 + 10 + 2
            for items in (slice(0, 4), slice(4, 8)):
                np.testing.assert_array_equal(loadings[items], loadings[items][0])
                np.testing.assert_array_equal(errors[items], errors[items][0])
            assert np.all(np.abs(loadings[:8] - 0.9) < 3 * errors[:8])
        reduced, product = (result.model.testlet_loadings for result in fits)
        np.testing.assert_allclose(reduced, product, atol=2e-3)
        np.testing.assert_allclose(
            fits[0].log_likelihood, fits[1].log_likelihood, atol=1e-3
        )


def test_reliability_coefficients_match_the_equivalent_bifactor_model() -> None:
    from mirt.models.bifactor import BifactorModel

    membership = [0, 0, 0, 1, 1, -1]
    discrimination = np.array([1.2, 0.8, 1.5, 1.0, 2.0, 0.9])
    loadings = np.array([0.7, 1.1, 0.4, 0.9, 0.6, 0.0])
    variances = np.array([2.0, 0.5])
    testlet = BifactorTestletModel(6, membership)
    testlet.set_parameters(
        discrimination=discrimination,
        testlet_loadings=loadings,
        testlet_variances=variances,
    )
    # The standalone item loads on a specific factor of its own with slope 0.
    bifactor = BifactorModel(6, [0, 0, 0, 1, 1, 2])
    bifactor.set_parameters(
        general_loadings=discrimination,
        specific_loadings=loadings * np.sqrt(np.r_[variances[[0, 0, 0, 1, 1]], 0.0]),
    )
    general, specific, unique = bifactor._standardized_components()

    assert testlet.omega_hierarchical() == pytest.approx(
        bifactor.omega_hierarchical(), rel=1e-12
    )
    reliability = testlet.testlet_reliability()
    assert reliability[0] == pytest.approx(bifactor.omega_subscale(0), rel=1e-12)
    assert reliability[1] == pytest.approx(bifactor.omega_subscale(1), rel=1e-12)
    shares = testlet.explained_variance()
    assert shares["general"] == pytest.approx(np.mean(general**2), rel=1e-12)
    assert shares["testlet"] == pytest.approx(np.mean(specific**2), rel=1e-12)
    assert shares["unique"] == pytest.approx(np.mean(unique), rel=1e-12)
    assert shares["total_common"] == pytest.approx(1.0 - np.mean(unique), rel=1e-12)
