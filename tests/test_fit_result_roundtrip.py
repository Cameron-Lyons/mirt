"""Round-trip contracts for FitResult.to_dict/from_dict and JSON export."""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pytest

from mirt import fit_mirt, fscores
from mirt.exceptions import MirtValidationError
from mirt.results import FitResult


@pytest.fixture(scope="module")
def responses() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(31)
    dichotomous = (rng.random((150, 6)) < 0.6).astype(np.int_)
    dichotomous[rng.random(dichotomous.shape) < 0.05] = -1
    polytomous = rng.integers(0, 4, size=(150, 6))
    # Item 0 has two categories, so polytomous thresholds are zero padded.
    polytomous[:, 0] = np.minimum(polytomous[:, 0], 1)
    return {"dichotomous": dichotomous, "polytomous": polytomous}


CASES = [
    ("1PL", 1),
    ("2PL", 1),
    ("2PL", 2),
    ("3PL", 1),
    ("4PL", 1),
    ("GRM", 1),
    ("GPCM", 1),
    ("PCM", 1),
    ("NRM", 1),
]


def _fit(model: str, n_factors: int, responses: dict[str, np.ndarray]) -> FitResult:
    polytomous = model in {"GRM", "GPCM", "PCM", "NRM"}
    data = responses["polytomous" if polytomous else "dichotomous"]
    return fit_mirt(data, model=model, n_factors=n_factors, n_quadpts=9, max_iter=3)  # type: ignore[arg-type]


@pytest.mark.parametrize(("model", "n_factors"), CASES)
def test_json_round_trip_rebuilds_model_and_scores(
    model: str, n_factors: int, responses: dict[str, np.ndarray]
) -> None:
    original = _fit(model, n_factors, responses)

    restored = FitResult.from_json(original.to_json())

    assert type(restored.model) is type(original.model)
    assert restored.model.n_factors == original.model.n_factors
    assert restored.model.item_names == original.model.item_names
    assert restored.model.is_fitted
    for name, values in original.model.parameters.items():
        np.testing.assert_array_equal(restored.model.parameters[name], values)
    assert restored.standard_errors.keys() == original.standard_errors.keys()
    for name, values in original.standard_errors.items():
        np.testing.assert_array_equal(restored.standard_errors[name], values)
    assert restored.fit_statistics() == original.fit_statistics()
    assert restored.se_method == original.se_method
    assert restored.vcov_labels == original.vcov_labels
    if original.vcov is None:
        assert restored.vcov is None
    else:
        np.testing.assert_array_equal(restored.vcov, original.vcov)
    assert restored.to_json() == original.to_json()

    data = responses["polytomous" if original.model.is_polytomous else "dichotomous"]
    for method in ("EAP", "MAP"):
        expected = fscores(original, data, method=method)
        actual = fscores(restored, data, method=method)
        np.testing.assert_array_equal(actual.theta, expected.theta)
        np.testing.assert_array_equal(actual.standard_error, expected.standard_error)


def test_export_records_per_item_category_counts(
    responses: dict[str, np.ndarray],
) -> None:
    polytomous = _fit("GRM", 1, responses).to_dict()["model"]
    dichotomous = _fit("2PL", 1, responses).to_dict()["model"]

    assert polytomous["n_categories"] == [2, 4, 4, 4, 4, 4]
    assert dichotomous["n_categories"] is None
    assert set(dichotomous) == {
        "name",
        "n_items",
        "n_factors",
        "item_names",
        "n_categories",
    }


def test_export_repeats_a_shared_category_count() -> None:
    # Custom polytomous item types report one count for all items; exporting
    # them must not fail while recording per-item counts.
    from mirt.models.custom import CustomItemModel, create_item_type

    def probabilities(theta: np.ndarray, difficulty: float) -> np.ndarray:
        logits = np.column_stack([np.zeros(len(theta)), theta[:, 0] - difficulty])
        logits = np.column_stack([logits, 2.0 * (theta[:, 0] - difficulty)])
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        return weights / weights.sum(axis=1, keepdims=True)

    item_type = create_item_type(
        "Ordinal3",
        probabilities,
        par_defaults={"difficulty": 0.0},
        n_categories=3,
    )
    model = CustomItemModel(n_items=2, item_type=item_type)
    result = FitResult(model, -1.0, 1, True, {}, 4.0, 5.0)

    exported = result.to_dict()["model"]

    assert exported["name"] == "Ordinal3"
    assert exported["n_categories"] == [3, 3]
    with pytest.raises(MirtValidationError, match="only fit_mirt model families"):
        FitResult.from_dict(result.to_dict())


def _assert_round_trip(original: FitResult, data: np.ndarray) -> FitResult:
    restored = FitResult.from_json(original.to_json())

    assert type(restored.model) is type(original.model)
    assert restored.model.is_fitted
    assert restored.model.n_parameters == original.model.n_parameters
    for name, values in original.model.parameters.items():
        np.testing.assert_array_equal(restored.model.parameters[name], values)
        np.testing.assert_array_equal(
            restored.model.free_parameter_masks[name],
            original.model.free_parameter_masks[name],
        )
    assert restored.vcov_labels == original.vcov_labels
    if original.latent_covariance is None:
        assert restored.latent_covariance is None
    else:
        np.testing.assert_array_equal(
            restored.latent_covariance, original.latent_covariance
        )
    assert restored.to_json() == original.to_json()
    expected = fscores(original, data)
    np.testing.assert_array_equal(fscores(restored, data).theta, expected.theta)
    return restored


@pytest.mark.parametrize(
    ("model", "spec", "name"),
    [
        ("2PL", "F1 = 1-3\nF2 = 4-6\nCOV = F1*F2", "MIRT"),
        ("2PL", "F1 = 1-4\nF2 = 3-6", "MIRT"),
        ("GRM", "F1 = 1-3\nF2 = 4-6\nCOV = F1*F2", "GRM"),
        (
            "2PL",
            "F = 1-6\nFIXED = (1, a1), (2, difficulty)\nSTART = (1, a1, 1.0)",
            "2PL",
        ),
    ],
)
def test_json_round_trip_rebuilds_confirmatory_fits(
    model: str, spec: str, name: str, responses: dict[str, np.ndarray]
) -> None:
    # Regression: multi-factor 2PL fits were rejected, and FIXED masks of
    # rebuilt families were lost.
    data = responses["polytomous" if model == "GRM" else "dichotomous"]
    original = fit_mirt(data, model=model, spec=spec, n_quadpts=7, max_iter=3)  # type: ignore[arg-type]
    exported = original.to_dict()["model"]

    assert exported["name"] == name
    assert ("loading_pattern" in exported) == (name == "MIRT")
    restored = _assert_round_trip(original, data)
    if "FIXED" in spec:
        assert restored.model.n_parameters == 6 * 2 - 2


def test_json_round_trip_rebuilds_bifactor_fits() -> None:
    from mirt import bfactor

    rng = np.random.default_rng(4)
    labels = [3, 3, 3, 5, 5, 5]
    theta = rng.standard_normal((200, 3))
    logits = theta[:, :1] + 0.7 * theta[:, [1, 1, 1, 2, 2, 2]] - 0.2
    data = (rng.random(logits.shape) < 1 / (1 + np.exp(-logits))).astype(int)
    original = bfactor(
        data,
        labels,
        n_quadpts=7,
        max_iter=3,
        fixed={"specific_loadings": np.array([True] + [False] * 5)},
    )

    assert original.to_dict()["model"]["specific_factors"] == labels
    restored = _assert_round_trip(original, data)
    np.testing.assert_array_equal(restored.model.specific_factors, labels)


def test_from_dict_validates_structured_models(
    responses: dict[str, np.ndarray],
) -> None:
    data = responses["dichotomous"]
    payload = fit_mirt(
        data, spec="F1 = 1-3\nF2 = 4-6", n_quadpts=7, max_iter=2
    ).to_dict()

    def rejected(change: Any, message: str) -> None:
        changed = copy.deepcopy(payload)
        change(changed)
        with pytest.raises(MirtValidationError, match=message):
            FitResult.from_dict(changed)

    rejected(lambda p: p["model"].pop("loading_pattern"), "is required for 'MIRT'")
    rejected(
        lambda p: p["model"].update(loading_pattern=[[0.5, 1.0]] * 6), "zeros and ones"
    )
    rejected(lambda p: p["model"].update(n_factors=3), "n_factors")
    rejected(lambda p: p["model"].update(specific_factors=[0] * 6), "does not apply")
    rejected(
        lambda p: p["parameters"]["slopes"][0].__setitem__(1, 0.7),
        "zero where model.loading_pattern is zero",
    )

    def short_bifactor(changed: dict[str, Any]) -> None:
        info = changed["model"]
        del info["loading_pattern"]
        info.update(name="Bifactor", specific_factors=[0, 0, 0, 1, 1], n_factors=3)

    rejected(short_bifactor, "specific_factors")


def test_round_trip_keeps_covariance_of_fitted_coordinates(
    responses: dict[str, np.ndarray],
) -> None:
    original = _fit("2PL", 1, responses)
    payload = original.to_dict()

    assert payload["se_method"] == "oakes"
    assert len(payload["vcov"]["labels"]) == original.model.n_parameters
    restored = FitResult.from_dict(payload)
    np.testing.assert_array_equal(restored.vcov, original.vcov)
    compact = FitResult.from_dict(original.to_dict(include_standard_errors=False))
    assert compact.vcov is None
    assert compact.se_method is None


def test_missing_standard_errors_restore_as_unknown(
    responses: dict[str, np.ndarray],
) -> None:
    original = _fit("GRM", 1, responses)

    restored = FitResult.from_dict(original.to_dict(include_standard_errors=False))

    assert restored.standard_errors == {}
    statistics = restored.parameter_statistics()
    assert np.isnan(statistics["discrimination"]["standard_error"]).all()


@pytest.fixture(scope="module")
def payload(responses: dict[str, np.ndarray]) -> dict[str, Any]:
    return _fit("2PL", 1, responses).to_dict()


def _modified(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> Any:
    changed = copy.deepcopy(payload)
    target = changed
    for key in path[:-1]:
        target = target[key]
    if value is _DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return changed


_DELETE = object()


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("extra",), 1, "unknown fields: extra"),
        (("parameters",), _DELETE, "include_parameters=True"),
        (("aic",), _DELETE, "missing required fields: aic"),
        (("aic",), "bad", "aic must be a number"),
        (("n_iterations",), 1.5, "non-negative integer"),
        (("converged",), "yes", "converged must be a boolean"),
        (("model", "name"), "RSM", "only fit_mirt model families"),
        (("model", "name"), "MIRT", "loading_pattern is required"),
        (("model", "loading_pattern"), [[1.0]] * 6, "does not apply to '2PL'"),
        (("model", "free_parameter_masks"), [True], "map parameter names"),
        (("model", "free_parameter_masks"), {"slope": [True] * 6}, "Unknown"),
        (("model", "free_parameter_masks"), {"difficulty": [1] * 6}, "Boolean"),
        (("model", "color"), "red", "unknown fields: color"),
        (("model", "n_items"), 0, "positive integer"),
        (("model", "n_items"), 5, "item_names"),
        (("model", "item_names"), "abc", "list of strings"),
        (("model", "n_factors"), 0, "n_factors"),
        (("model", "n_categories"), [3] * 6, "n_categories"),
        (("parameters", "difficulty"), [0.0] * 5, "Shape mismatch"),
        (("parameters", "slope"), [0.0] * 6, "exactly"),
        (("parameters", "difficulty"), ["a"] * 6, "numeric"),
        (("standard_errors", "difficulty"), [-1.0] * 6, "cannot be negative"),
        (("standard_errors", "dificulty"), [0.1] * 6, "unknown parameters"),
        (("se_method",), 3, "se_method must be a string"),
        (("vcov",), {"labels": []}, "labels and matrix"),
        (("vcov", "labels"), "abc", "labels must be a list"),
        (("vcov", "matrix"), [["a"]], "numeric"),
        (("vcov", "matrix"), [[1.0, 2.0]], "square"),
        (("vcov", "labels"), ["discrimination[Item_1]"] * 12, "unique"),
    ],
)
def test_from_dict_rejects_malformed_payloads(
    payload: dict[str, Any], path: tuple[str, ...], value: Any, message: str
) -> None:
    with pytest.raises(MirtValidationError, match=message):
        FitResult.from_dict(_modified(payload, path, value))


def test_from_dict_rejects_payload_without_parameters(
    responses: dict[str, np.ndarray],
) -> None:
    compact = _fit("2PL", 1, responses).to_dict(include_parameters=False)

    with pytest.raises(MirtValidationError, match="include_parameters=True"):
        FitResult.from_dict(compact)


def test_from_dict_rejects_unsupported_factor_counts(
    responses: dict[str, np.ndarray],
) -> None:
    payload = _fit("3PL", 1, responses).to_dict()
    payload["model"]["n_factors"] = 2

    with pytest.raises(MirtValidationError, match="multidimensional"):
        FitResult.from_dict(payload)


def test_from_dict_checks_family_fixed_parameters(
    responses: dict[str, np.ndarray],
) -> None:
    payload = _fit("1PL", 1, responses).to_dict()
    assert payload["parameters"]["discrimination"] == [1.0] * 6

    payload["parameters"]["discrimination"][0] = 2.0
    with pytest.raises(MirtValidationError, match="fixed by the 1PL family"):
        FitResult.from_dict(payload)


def test_polytomous_payload_requires_category_counts(
    responses: dict[str, np.ndarray],
) -> None:
    payload = _fit("GPCM", 1, responses).to_dict()
    payload["model"]["n_categories"] = None

    with pytest.raises(MirtValidationError, match="n_categories"):
        FitResult.from_dict(payload)


@pytest.mark.parametrize("value", [None, 1, ["not", "a", "mapping"]])
def test_from_dict_requires_a_mapping(value: Any) -> None:
    with pytest.raises(MirtValidationError, match="mapping"):
        FitResult.from_dict(value)


def test_from_json_validates_input() -> None:
    with pytest.raises(MirtValidationError, match="string or bytes"):
        FitResult.from_json(123)  # type: ignore[arg-type]
    with pytest.raises(MirtValidationError, match="valid JSON object"):
        FitResult.from_json("{not json")


def test_from_json_accepts_bytes(payload: dict[str, Any]) -> None:
    restored = FitResult.from_json(json.dumps(payload).encode())

    assert restored.to_json() == json.dumps(payload)
