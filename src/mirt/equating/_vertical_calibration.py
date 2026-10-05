"""Joint anchor calibration for grade forms with different item positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mirt.utils.data import validate_responses

if TYPE_CHECKING:
    from mirt.equating.linking import LinkingResult
    from mirt.equating.vertical import GradeData, VerticalScaleResult
    from mirt.models.base import BaseItemModel


@dataclass
class _UnionItemDesign:
    """A physical-item bank and each grade's sparse administration."""

    responses: list[NDArray[np.int_]]
    item_maps: list[NDArray[np.intp]]
    members: list[list[tuple[int, int]]]

    @property
    def n_items(self) -> int:
        return len(self.members)


def _union_item_design(grade_data: list[GradeData]) -> _UnionItemDesign:
    """Merge only declared anchor identities, retaining every original column."""
    from mirt.equating.vertical import _resolve_anchor_pair

    responses = [validate_responses(gd.responses) for gd in grade_data]
    widths = [response.shape[1] for response in responses]
    offsets = np.r_[0, np.cumsum(widths)]
    parent = np.arange(offsets[-1], dtype=np.intp)

    def representative(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for grade, (lower, upper) in enumerate(
        zip(grade_data[:-1], grade_data[1:], strict=True)
    ):
        anchors_lower, anchors_upper = _resolve_anchor_pair(lower, upper)
        informative = 0
        for item_lower, item_upper in zip(anchors_lower, anchors_upper, strict=True):
            left = representative(int(offsets[grade] + item_lower))
            right = representative(int(offsets[grade + 1] + item_upper))
            parent[right] = left
            informative += bool(np.any(responses[grade][:, item_lower] >= 0)) and bool(
                np.any(responses[grade + 1][:, item_upper] >= 0)
            )
        if informative < 2:
            raise ValueError(
                f"At least 2 anchors must have observed responses in both grades "
                f"'{lower.grade_label}' and '{upper.grade_label}'"
            )

    component_columns: dict[int, int] = {}
    members: list[list[tuple[int, int]]] = []
    item_maps = []
    for grade, width in enumerate(widths):
        mapping = np.empty(width, dtype=np.intp)
        for item in range(width):
            root = representative(int(offsets[grade] + item))
            column = component_columns.get(root)
            if column is None:
                column = len(members)
                component_columns[root] = column
                members.append([])
            mapping[item] = column
            members[column].append((grade, item))
        if np.unique(mapping).size != width:
            raise ValueError("Anchor mappings merge different items within one grade")
        item_maps.append(mapping)

    n_items = len(members)
    union_responses = []
    observed = np.zeros(n_items, dtype=np.bool_)
    for response, mapping in zip(responses, item_maps, strict=True):
        padded = np.full((len(response), n_items), -1, dtype=np.int_)
        padded[:, mapping] = response
        observed[mapping] |= np.any(response >= 0, axis=0)
        union_responses.append(padded)
    if not np.all(observed):
        missing = np.flatnonzero(~observed).tolist()
        raise ValueError(f"Physical items {missing} contain no observed responses")
    return _UnionItemDesign(union_responses, item_maps, members)


def _model_types() -> dict[str, type[BaseItemModel]]:
    from mirt.models._factory import ITEM_MODEL_FAMILIES, item_model_class

    return {name: item_model_class(name) for name in ITEM_MODEL_FAMILIES}


def _make_model(
    model_name: str,
    n_items: int,
    categories: list[int] | None = None,
    item_names: list[str] | None = None,
) -> BaseItemModel:
    from mirt.models._factory import build_item_model

    return build_item_model(
        model_name, n_items, n_categories=categories, item_names=item_names
    )


def _validate_models_and_categories(
    grade_data: list[GradeData],
    models: list[BaseItemModel] | None,
    design: _UnionItemDesign,
) -> tuple[str, list[int] | None]:
    """Validate one built-in family and physical-item response categories."""
    if models is None:
        model_name, categories = "2PL", None
    else:
        model_name = models[0].model_name
        supported = _model_types()
        if model_name not in supported or any(
            type(model) is not supported[model_name] for model in models
        ):
            raise ValueError(
                "Anchor calibration requires one supported built-in model family"
            )
        if any(not model.is_fitted for model in models):
            raise ValueError(
                "Provided grade models must be fitted before anchor calibration"
            )
        categories = None
        if models[0].is_polytomous:
            categories = []
            for members in design.members:
                counts = {
                    int(models[grade].n_categories[item]) for grade, item in members
                }
                if len(counts) != 1:
                    raise ValueError(
                        "Paired anchor items must have matching category counts"
                    )
                categories.append(counts.pop())

    for grade, (gd, mapping) in enumerate(
        zip(grade_data, design.item_maps, strict=True)
    ):
        response = validate_responses(gd.responses)
        maximum = (
            np.ones(len(mapping), dtype=np.int_)
            if categories is None
            else (np.asarray(categories, dtype=np.int_)[mapping] - 1)
        )
        if np.any(response > maximum[None, :]):
            raise ValueError(
                f"Responses for grade '{gd.grade_label}' exceed model categories"
            )
    return model_name, categories


def _pairwise_initial_constants(
    grade_data: list[GradeData], models: list[BaseItemModel]
) -> list[tuple[float, float]]:
    """Use moment links only to initialize the joint response calibration."""
    from mirt.equating.polytomous import _linker_for
    from mirt.equating.vertical import _resolve_anchor_pair

    constants = []
    for grade in range(len(models) - 1):
        anchors_old, anchors_new = _resolve_anchor_pair(
            grade_data[grade], grade_data[grade + 1]
        )
        old, new = models[grade : grade + 2]
        linker = _linker_for(old)
        method = "haebara" if old.model_name == "NRM" else "mean_mean"
        fitted = linker(
            old, new, anchors_old, anchors_new, method=method, compute_diagnostics=False
        )
        constants.append((fitted.constants.A, fitted.constants.B))
    return constants


def _anchor_calibration_diagnostics(
    grade_data: list[GradeData],
    models: list[BaseItemModel],
    method: str,
) -> list[LinkingResult]:
    """Describe adjacent shared anchors on their fitted common metric."""
    from mirt.equating.polytomous import _linker_for
    from mirt.equating.vertical import _resolve_anchor_pair

    results = []
    for grade in range(len(models) - 1):
        anchors_old, anchors_new = _resolve_anchor_pair(
            grade_data[grade], grade_data[grade + 1]
        )
        old, new = models[grade : grade + 2]
        diagnostic = _linker_for(old)(
            old, new, anchors_old, anchors_new, method="stocking_lord"
        )
        diagnostic.constants.method = method
        diagnostic.convergence_info = {"success": True, "joint_calibration": True}
        results.append(diagnostic)
    return results


def _initial_global_parameters(
    global_model: BaseItemModel,
    grade_data: list[GradeData],
    models: list[BaseItemModel],
    design: _UnionItemDesign,
    reference_grade: int,
) -> None:
    """Seed physical items using nearby calibrations on the reference metric."""
    constants = _pairwise_initial_constants(grade_data, models)
    scales = np.ones(len(models))
    shifts = np.zeros(len(models))
    for grade in range(reference_grade + 1, len(models)):
        A, B = constants[grade - 1]
        scales[grade] = scales[grade - 1] * A
        shifts[grade] = shifts[grade - 1] + scales[grade - 1] * B
    for grade in range(reference_grade - 1, -1, -1):
        A, B = constants[grade]
        scales[grade] = scales[grade + 1] / A
        shifts[grade] = shifts[grade + 1] - scales[grade + 1] * B / A

    parameters = global_model.parameters
    source_parameters = [model.parameters for model in models]
    for column, members in enumerate(design.members):
        grade, item = min(members, key=lambda member: abs(member[0] - reference_grade))
        A, B = scales[grade], shifts[grade]
        source = source_parameters[grade]
        for name, destination in parameters.items():
            row = np.asarray(source[name][item], dtype=np.float64).copy()
            if name in {"discrimination", "slopes"}:
                row /= A
            elif name in {"difficulty", "thresholds", "steps"}:
                row = A * row + B
            elif name == "intercepts" and "slopes" in source:
                row -= source["slopes"][item] * B / A
            if destination.ndim == 1:
                destination[column] = row
            else:
                active = row.size
                if models[grade].is_polytomous:
                    active = models[grade].n_categories[item] - (
                        1 if name in {"thresholds", "steps"} else 0
                    )
                destination[column, :active] = row[:active]
    free_masks = global_model.free_parameter_masks
    global_model.set_parameters(
        **{
            name: values
            for name, values in parameters.items()
            if np.any(free_masks[name])
        }
    )
    global_model._is_fitted = True


def _local_calibrated_model(
    global_model: BaseItemModel,
    mapping: NDArray[np.intp],
    categories: list[int] | None,
    item_names: list[str] | None,
) -> BaseItemModel:
    local_categories = (
        None if categories is None else [categories[int(column)] for column in mapping]
    )
    local = _make_model(
        global_model.model_name, len(mapping), local_categories, item_names
    )
    source = global_model.parameters
    local_parameters = local.parameters
    free_masks = local.free_parameter_masks
    for name, values in local_parameters.items():
        selected = source[name][mapping]
        values[...] = selected if values.ndim == 1 else selected[:, : values.shape[1]]
    local.set_parameters(
        **{
            name: values
            for name, values in local_parameters.items()
            if np.any(free_masks[name])
        }
    )
    local._is_fitted = True
    return local


def calibrate_vertical_scale(
    grade_data: list[GradeData],
    models: list[BaseItemModel] | None,
    method: str,
    reference_grade: int,
    *,
    enforce_monotonicity: bool,
    n_quadpts: int,
    max_iter: int,
    tol: float,
) -> VerticalScaleResult:
    """Fit a shared physical-item bank and grade-specific ability densities."""
    from mirt.equating.vertical import VerticalScaleResult
    from mirt.multigroup.estimator import MultigroupEMEstimator
    from mirt.multigroup.model import MultigroupModel
    from mirt.scoring import fscores

    for name, value, minimum in (
        ("n_quadpts", n_quadpts, 5),
        ("max_iter", max_iter, 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    if not isinstance(enforce_monotonicity, (bool, np.bool_)):
        raise ValueError("enforce_monotonicity must be boolean")

    design = _union_item_design(grade_data)
    model_name, categories = _validate_models_and_categories(grade_data, models, design)
    global_model = _make_model(model_name, design.n_items, categories)

    initial_models = models
    if method == "fixed_anchor" and models is None:
        from mirt import fit_mirt

        reference = fit_mirt(
            validate_responses(grade_data[reference_grade].responses),
            model="2PL",
            n_quadpts=n_quadpts,
            max_iter=max_iter,
            tol=tol,
            compute_standard_errors=False,
            verbose=False,
        )
        if not reference.converged:
            raise RuntimeError("Reference grade calibration failed to converge")
        initial_models = [
            reference.model
            if grade == reference_grade
            else _make_model("2PL", len(mapping))
            for grade, mapping in enumerate(design.item_maps)
        ]
    if initial_models is not None:
        _initial_global_parameters(
            global_model, grade_data, initial_models, design, reference_grade
        )

    fixed_parameters: dict[str, dict[int, float | NDArray[np.float64]]] = {}
    if method == "fixed_anchor":
        assert initial_models is not None
        reference_parameters = initial_models[reference_grade].parameters
        global_parameters = global_model.parameters
        for column, members in enumerate(design.members):
            reference_items = [
                item for grade, item in members if grade == reference_grade
            ]
            if len(members) < 2 or not reference_items:
                continue
            item = reference_items[0]
            for name, values in reference_parameters.items():
                row = values[item]
                if np.ndim(row) == 0:
                    value = float(row)
                else:
                    value = global_parameters[name][column].copy()
                    value[: row.size] = row
                fixed_parameters.setdefault(name, {})[column] = value

    joint_model = MultigroupModel(global_model, len(grade_data))
    estimator = MultigroupEMEstimator(n_quadpts=n_quadpts, max_iter=max_iter, tol=tol)
    fit_kwargs: dict[str, Any] = {
        "invariance": "strict",
        "reference_group": reference_grade,
    }
    if fixed_parameters:
        fit_kwargs["fixed_parameters"] = fixed_parameters
    if enforce_monotonicity:
        fit_kwargs["mean_order"] = list(range(len(grade_data)))
    fitted = estimator.fit(joint_model, design.responses, **fit_kwargs)
    if not fitted.converged:
        raise RuntimeError(
            f"{method} vertical calibration failed to converge after {fitted.n_iterations} iterations"
        )

    calibrated_models, scores, distributions, item_maps = {}, {}, {}, {}
    calibration_masks = {}
    grade_means, grade_sds = {}, {}
    for grade, (gd, mapping, distribution) in enumerate(
        zip(grade_data, design.item_maps, fitted.latent_distributions, strict=True)
    ):
        item_names = None if models is None else models[grade].item_names
        local = _local_calibrated_model(
            fitted.model.get_group_model(grade), mapping, categories, item_names
        )
        score = fscores(
            local,
            validate_responses(gd.responses),
            method="EAP",
            n_quadpts=n_quadpts,
            prior_mean=distribution.mean,
            prior_cov=distribution.cov,
        )
        calibrated_models[gd.grade_label] = local
        scores[gd.grade_label] = score
        distributions[gd.grade_label] = distribution.copy()
        item_maps[gd.grade_label] = mapping.copy()
        global_masks = fitted.model.effective_free_parameter_masks(grade)
        calibration_masks[gd.grade_label] = {
            name: (
                global_masks[name][mapping].copy()
                if values.ndim == 1
                else global_masks[name][mapping, : values.shape[1]].copy()
            )
            for name, values in local.parameters.items()
        }
        local.set_free_parameter_masks(calibration_masks[gd.grade_label])
        grade_means[gd.grade_label] = float(distribution.mean[0])
        grade_sds[gd.grade_label] = float(np.sqrt(distribution.cov[0, 0]))

    return VerticalScaleResult(
        grade_transformations={},
        grade_means=grade_means,
        grade_sds=grade_sds,
        linking_results=_anchor_calibration_diagnostics(
            grade_data, list(calibrated_models.values()), method
        ),
        monotonicity_violations=[
            (lower.grade_label, upper.grade_label)
            for lower, upper in zip(grade_data[:-1], grade_data[1:], strict=True)
            if grade_means[upper.grade_label] < grade_means[lower.grade_label]
        ],
        growth_curve=np.array(list(grade_means.values())),
        method=method,
        reference_grade=reference_grade,
        calibrated_models=calibrated_models,
        scores=scores,
        latent_distributions=distributions,
        calibration_result=fitted,
        item_maps=item_maps,
        free_parameter_masks=calibration_masks,
    )
