"""Regression tests for multigroup model and result contracts."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from mirt.models.dichotomous import OneParameterLogistic, TwoParameterLogistic
from mirt.models.multidimensional import MultidimensionalModel
from mirt.models.polytomous import (
    GeneralizedPartialCredit,
    GradedResponseModel,
    NominalResponseModel,
    PartialCreditModel,
)
from mirt.multigroup import (
    GroupLatentDistribution,
    MultigroupFitResult,
    MultigroupModel,
    fit_multigroup,
)
from mirt.utils import set_dataframe_backend


@pytest.fixture(params=["pandas", "polars"])
def dataframe_backend(request: pytest.FixtureRequest) -> Iterator[str]:
    """Exercise result tables with each supported dataframe implementation."""
    backend = str(request.param)
    set_dataframe_backend(backend)
    yield backend
    set_dataframe_backend("auto")


def _records(frame: object) -> list[dict[str, object]]:
    if hasattr(frame, "to_dicts"):
        return frame.to_dicts()  # type: ignore[no-any-return, union-attr]
    return frame.to_dict(orient="records")  # type: ignore[no-any-return, union-attr]


def _columns(frame: object) -> list[str]:
    return list(frame.columns)  # type: ignore[union-attr]


def _result(
    model: MultigroupModel | None = None,
) -> MultigroupFitResult:
    if model is None:
        model = MultigroupModel(
            TwoParameterLogistic(2, item_names=["first", "second"]),
            2,
            ("reference", "focal"),
        )
        model.set_group_parameters(
            0,
            discrimination=np.array([1.0, 2.0]),
            difficulty=np.array([-0.5, 0.5]),
        )
        model.set_group_parameters(
            1,
            discrimination=np.array([1.5, 2.5]),
            difficulty=np.array([-1.0, 1.0]),
        )

    latent_distributions = [
        GroupLatentDistribution(
            mean=np.zeros(model.n_factors),
            cov=np.eye(model.n_factors),
            is_reference=True,
        ),
        GroupLatentDistribution(
            mean=np.full(model.n_factors, 0.25),
            cov=np.eye(model.n_factors) * 1.25,
        ),
    ]
    return MultigroupFitResult(
        model=model,
        invariance="configural",
        log_likelihood=-42.0,
        n_iterations=7,
        converged=True,
        group_log_likelihoods=[-20.0, -22.0],
        group_n_observations=[40, 50],
        latent_distributions=latent_distributions,
        aic=96.0,
        bic=111.0,
        n_parameters=6,
        n_observations=90,
    )


@pytest.mark.parametrize("n_groups", [True, np.bool_(False), 1, 2.5])
def test_constructor_rejects_invalid_group_counts(n_groups: object) -> None:
    with pytest.raises(ValueError, match="n_groups"):
        MultigroupModel(TwoParameterLogistic(2), n_groups)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("labels", "match"),
    [
        (["only"], "length"),
        (["same", "same"], "unique"),
        (["valid", ""], "non-empty"),
        (["valid", 2], "non-empty"),
        ("AB", "sequence"),
    ],
)
def test_constructor_validates_group_labels(labels: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MultigroupModel(
            TwoParameterLogistic(2),
            2,
            labels,  # type: ignore[arg-type]
        )


def test_constructor_accepts_label_sequences_and_protects_model_list() -> None:
    model = MultigroupModel(TwoParameterLogistic(2), 2, ("A", "B"))

    exposed_models = model.group_models
    exposed_models.clear()

    assert model.group_labels == ["A", "B"]
    assert len(model.group_models) == 2


@pytest.mark.parametrize("group_idx", [True, 0.5, -1, 2])
def test_group_access_rejects_invalid_indices(group_idx: object) -> None:
    model = MultigroupModel(TwoParameterLogistic(2), 2)
    expected_error = TypeError if isinstance(group_idx, (bool, float)) else IndexError

    with pytest.raises(expected_error):
        model.get_group_model(group_idx)  # type: ignore[arg-type]

    assert model.get_group_model(np.int64(1)) is model.group_models[1]


@pytest.mark.parametrize(
    ("indices", "error"),
    [
        ([], ValueError),
        ([True], TypeError),
        ([0.5], TypeError),
        ([0, 0], ValueError),
        ([-1], IndexError),
        ([3], IndexError),
    ],
)
def test_item_subset_validation(indices: list[object], error: type[Exception]) -> None:
    model = MultigroupModel(TwoParameterLogistic(3), 2)

    with pytest.raises(error):
        model.set_shared_parameter(
            "difficulty",
            indices,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="Unknown parameter"):
        model.set_shared_parameter("missing")


def test_partial_parameter_sharing_tracks_only_requested_items() -> None:
    model = MultigroupModel(TwoParameterLogistic(4), 2)

    model.set_shared_parameter("difficulty", [3, 1])
    assert model.get_shared_items("difficulty") == [1, 3]
    assert model.get_free_items("difficulty") == [0, 2]

    model.set_shared_parameter("difficulty", [0])
    assert model.get_shared_items("difficulty") == [0, 1, 3]

    model.set_group_specific_parameter("difficulty", [1, 3])
    assert model.get_shared_items("difficulty") == [0]
    assert model.get_free_items("difficulty") == [1, 2, 3]

    model.set_group_specific_parameter("difficulty", [0])
    assert not model.is_parameter_shared("difficulty")
    assert model.get_free_items("difficulty") == [0, 1, 2, 3]


def test_full_sharing_can_restore_individually_freed_items() -> None:
    model = MultigroupModel(TwoParameterLogistic(3), 2)
    model.set_shared_parameter("difficulty")
    model.set_group_specific_parameter("difficulty", [0, 2])

    model.set_shared_parameter("difficulty", [2])

    assert model.get_shared_items("difficulty") == [1, 2]
    assert model.get_free_items("difficulty") == [0]


@pytest.mark.parametrize(
    "base_model",
    [
        OneParameterLogistic(3),
        TwoParameterLogistic(3),
        GradedResponseModel(3, [2, 4, 3]),
        GeneralizedPartialCredit(3, [2, 4, 3]),
        PartialCreditModel(3, [2, 4, 3]),
        NominalResponseModel(3, [2, 4, 3]),
    ],
    ids=["1pl", "2pl", "grm", "gpcm", "pcm", "nrm"],
)
def test_parameter_count_uses_statistically_free_components(base_model: object) -> None:
    model = MultigroupModel(base_model, 3)  # type: ignore[arg-type]

    assert model.n_parameters == base_model.n_parameters * 3  # type: ignore[union-attr]

    for param_name in model.parameter_names:
        model.set_shared_parameter(param_name)
    assert model.n_parameters == base_model.n_parameters  # type: ignore[union-attr]


def test_parameter_count_handles_partial_invariance() -> None:
    model = MultigroupModel(TwoParameterLogistic(3), 3)

    model.set_shared_parameter("discrimination", [0, 2])
    assert model.n_parameters == 14

    model.set_group_specific_parameter("discrimination", [2])
    assert model.n_parameters == 16


def test_synchronize_shared_parameters_updates_only_shared_items() -> None:
    model = MultigroupModel(TwoParameterLogistic(3), 3)
    discriminations = [
        np.array([1.0, 2.0, 3.0]),
        np.array([3.0, 4.0, 5.0]),
        np.array([5.0, 6.0, 7.0]),
    ]
    for group_idx, values in enumerate(discriminations):
        model.set_group_parameters(group_idx, discrimination=values)
    model.set_shared_parameter("discrimination", [0, 2])

    model.synchronize_shared_parameters()

    for group_idx, original in enumerate(discriminations):
        actual = model.get_group_parameters(group_idx)["discrimination"]
        np.testing.assert_allclose(actual[[0, 2]], [3.0, 5.0])
        assert actual[1] == original[1]


def test_synchronization_skips_fixed_and_padded_components() -> None:
    fixed_model = MultigroupModel(OneParameterLogistic(2), 2)
    fixed_model.set_shared_parameter("discrimination")
    fixed_model.synchronize_shared_parameters()
    for group_model in fixed_model.group_models:
        np.testing.assert_array_equal(group_model.parameters["discrimination"], 1.0)

    model = MultigroupModel(GradedResponseModel(2, [2, 3]), 2)
    first = np.array([[-1.0, 99.0], [-2.0, 2.0]])
    second = np.array([[1.0, -99.0], [0.0, 4.0]])
    model.set_group_parameters(0, thresholds=first)
    model.set_group_parameters(1, thresholds=second)
    model.set_shared_parameter("thresholds")

    model.synchronize_shared_parameters()

    first_actual = model.get_group_parameters(0)["thresholds"]
    second_actual = model.get_group_parameters(1)["thresholds"]
    np.testing.assert_allclose(first_actual[[0, 1], [0, 0]], [0.0, -1.0])
    np.testing.assert_allclose(second_actual[[0, 1], [0, 0]], [0.0, -1.0])
    assert first_actual[1, 1] == second_actual[1, 1] == 3.0
    assert first_actual[0, 1] == 99.0
    assert second_actual[0, 1] == -99.0


def test_synchronization_batches_updates_per_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MultigroupModel(TwoParameterLogistic(100), 3)
    model.set_shared_parameter("discrimination")
    model.set_shared_parameter("difficulty")
    call_counts = [0, 0, 0]

    for group_idx, group_model in enumerate(model.group_models):
        original = group_model.set_parameters

        def tracked_set_parameters(
            *,
            _group_idx: int = group_idx,
            _original: object = original,
            **params: np.ndarray,
        ) -> object:
            call_counts[_group_idx] += 1
            return _original(**params)  # type: ignore[operator]

        monkeypatch.setattr(group_model, "set_parameters", tracked_set_parameters)

    model.synchronize_shared_parameters()

    assert call_counts == [2, 2, 2]


def test_copy_shared_to_all_updates_only_selected_items() -> None:
    model = MultigroupModel(TwoParameterLogistic(3), 2)
    model.set_group_parameters(0, difficulty=np.array([1.0, 2.0, 3.0]))
    model.set_group_parameters(1, difficulty=np.array([4.0, 5.0, 6.0]))
    model.set_shared_parameter("difficulty", [0, 2])

    model.copy_shared_to_all(source_group=np.int64(1))

    np.testing.assert_allclose(
        model.get_group_parameters(0)["difficulty"], [4.0, 2.0, 6.0]
    )
    with pytest.raises(TypeError):
        model.copy_shared_to_all(source_group=True)
    with pytest.raises(IndexError):
        model.copy_shared_to_all(source_group=2)


def test_end_to_end_fit_reports_correct_information_criteria() -> None:
    responses = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 1],
            [1, 1],
            [0, 0],
            [1, 0],
        ]
    )
    groups = np.array(["A"] * 4 + ["B"] * 4)

    result = fit_multigroup(
        responses,
        groups,
        model="1PL",
        invariance="configural",
        n_quadpts=7,
        max_iter=1,
    )

    assert result.model.n_parameters == 4
    assert result.n_parameters == 6
    assert result.aic == pytest.approx(-2 * result.log_likelihood + 12)
    assert result.bic == pytest.approx(-2 * result.log_likelihood + np.log(8) * 6)


def test_single_and_combined_coefficient_tables(dataframe_backend: str) -> None:
    result = _result()

    by_index = result.coef(0)
    by_label = result.coef("reference")
    combined = result.coef()

    if dataframe_backend == "pandas":
        assert list(by_index.index) == ["first", "second"]
        assert by_index.index.name == "item"
    else:
        assert _columns(by_index)[0] == "item"
        assert [row["item"] for row in _records(by_index)] == ["first", "second"]
    assert _records(by_index) == _records(by_label)
    assert len(_records(combined)) == 4
    assert _records(combined)[0]["group"] == "reference"


@pytest.mark.parametrize("group", [True, 0.5, -1, 2, "missing"])
def test_coefficient_table_validates_group(group: object) -> None:
    result = _result()
    expected = ValueError if group == "missing" else TypeError
    if group in (-1, 2):
        expected = IndexError

    with pytest.raises(expected):
        result.coef(group)  # type: ignore[arg-type]


def test_coefficient_table_can_include_standard_errors(
    dataframe_backend: str,
) -> None:
    result = _result()
    result.standard_errors = {
        "difficulty": {0: np.array([0.1, 0.2])},
    }

    rows = _records(result.coef(0, include_standard_errors=True))

    assert rows[0]["difficulty_se"] == pytest.approx(0.1)
    assert np.isnan(rows[0]["discrimination_se"])
    with pytest.raises(TypeError, match="boolean"):
        result.coef(0, include_standard_errors=1)  # type: ignore[arg-type]

    result.standard_errors["difficulty"][0] = np.array([0.1])
    with pytest.raises(ValueError, match="match the parameter shape"):
        result.coef(0, include_standard_errors=True)


def test_coefficient_table_flattens_higher_dimensional_parameters(
    dataframe_backend: str,
) -> None:
    model = MultigroupModel(
        NominalResponseModel(
            2,
            [2, 3],
            n_factors=2,
            item_names=["first", "second"],
        ),
        2,
        ("reference", "focal"),
    )
    result = _result(model)

    columns = _columns(result.coef(0))

    assert "slopes_0_0" in columns
    assert "slopes_2_1" in columns
    assert "intercepts_2" in columns


def test_multidimensional_latent_table_and_result_metadata(
    dataframe_backend: str,
) -> None:
    model = MultigroupModel(
        MultidimensionalModel(2, 2, item_names=["first", "second"]),
        2,
        ("reference", "focal"),
    )
    result = _result(model)

    columns = _columns(result.latent_pars())
    statistics = result.fit_statistics()
    labels = result.group_labels
    labels[0] = "changed"

    assert {"mean_0", "mean_1", "var_0", "cov_0_1", "var_1"} <= set(columns)
    assert statistics["n_parameters"] == 6
    assert statistics["converged"] is True
    assert result.group_labels == ["reference", "focal"]
    assert "Multigroup IRT Analysis Results" in result.summary()
    assert "n_groups=2" in repr(result)
