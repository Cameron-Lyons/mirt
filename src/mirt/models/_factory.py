"""Shared construction of the built-in item-model families.

``fit_mirt``, ``fit_multigroup``, vertical calibration, and
``FitResult.from_dict`` build their models here so that factor counts, category
counts, and item names are validated identically everywhere.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mirt.exceptions import MirtDataError, MirtModelError, MirtValidationError

if TYPE_CHECKING:
    from mirt.models.base import BaseItemModel
    from mirt.models.mixed_format import MixedItemModel

ITEM_MODEL_FAMILIES: dict[str, tuple[str, str]] = {
    "1PL": ("mirt.models.dichotomous", "OneParameterLogistic"),
    "2PL": ("mirt.models.dichotomous", "TwoParameterLogistic"),
    "3PL": ("mirt.models.dichotomous", "ThreeParameterLogistic"),
    "4PL": ("mirt.models.dichotomous", "FourParameterLogistic"),
    "GRM": ("mirt.models.polytomous", "GradedResponseModel"),
    "GPCM": ("mirt.models.polytomous", "GeneralizedPartialCredit"),
    "PCM": ("mirt.models.polytomous", "PartialCreditModel"),
    "NRM": ("mirt.models.polytomous", "NominalResponseModel"),
}
POLYTOMOUS_FAMILIES = frozenset({"GRM", "GPCM", "PCM", "NRM"})


def item_model_class(name: str) -> type[BaseItemModel]:
    """Return the class implementing a built-in model family.

    Raises
    ------
    MirtModelError
        If ``name`` is not one of :data:`ITEM_MODEL_FAMILIES`.
    MirtValidationError
        If ``name`` is a sequence of per-item families.
    """
    if isinstance(name, (list, tuple, np.ndarray)):
        raise MirtValidationError(
            "model must name one family here; per-item family sequences are "
            "supported by fit_mirt and MixedFormatEMEstimator",
            parameter="model",
            value=type(name).__name__,
            expected=", ".join(ITEM_MODEL_FAMILIES),
        )
    try:
        module_name, class_name = ITEM_MODEL_FAMILIES[name]
    except (KeyError, TypeError):
        raise MirtModelError(
            f"Unknown model: {name}",
            model_type=str(name),
            expected=", ".join(ITEM_MODEL_FAMILIES),
        ) from None
    model_class: type[BaseItemModel] = getattr(
        importlib.import_module(module_name), class_name
    )
    return model_class


def validate_n_factors(n_factors: Any) -> int:
    """Return ``n_factors`` as a Python integer of at least one."""
    if (
        isinstance(n_factors, (bool, np.bool_))
        or not isinstance(n_factors, (int, np.integer))
        or n_factors < 1
    ):
        raise MirtValidationError(
            "n_factors must be a positive integer",
            parameter="n_factors",
            value=n_factors,
            expected="integer >= 1",
        )
    return int(n_factors)


def resolve_category_counts(
    n_items: int,
    n_categories: int | Sequence[int] | None,
    responses: NDArray[np.int_] | None = None,
) -> int | list[int]:
    """Validate declared category counts or infer one count per item.

    A declared scalar is preserved; sequences and inferred counts become a list
    of Python integers. Inference uses each item's largest observed code with a
    minimum of two categories.
    """
    if n_categories is None:
        maxima = None if responses is None else responses.max(axis=0)
        if maxima is None or np.any(maxima < 0):
            raise MirtValidationError(
                "n_categories is required for items with no observed responses",
                parameter="n_categories",
                expected="one category count per item, each >= 2",
            )
        n_categories = np.maximum(maxima + 1, 2).tolist()
    try:
        counts = np.asarray(n_categories)
    except (TypeError, ValueError) as exc:
        raise MirtValidationError(
            "n_categories must be an integer or one integer count per item",
            parameter="n_categories",
            value=n_categories,
        ) from exc
    if counts.ndim > 1 or (counts.ndim == 1 and counts.size != n_items):
        raise MirtValidationError(
            f"n_categories must be a scalar or have shape ({n_items},)",
            parameter="n_categories",
            value=n_categories,
        )
    if counts.dtype.kind not in "iu":
        raise MirtValidationError(
            "n_categories must contain integer category counts",
            parameter="n_categories",
            value=n_categories,
        )
    if np.any(counts < 2):
        raise MirtValidationError(
            "n_categories must be at least 2 for each item",
            parameter="n_categories",
            value=n_categories,
            expected=">= 2",
        )
    return int(counts) if counts.ndim == 0 else counts.tolist()


def build_item_model(
    name: str,
    n_items: int,
    *,
    n_factors: Any = 1,
    n_categories: int | Sequence[int] | None = None,
    item_names: Sequence[str] | None = None,
    responses: NDArray[np.int_] | None = None,
) -> BaseItemModel:
    """Construct a built-in item model with validated structure.

    Parameters
    ----------
    name : {"1PL", "2PL", "3PL", "4PL", "GRM", "GPCM", "PCM", "NRM"}
        Model family.
    n_items : int
        Number of items.
    n_factors : int, default=1
        Number of latent factors. Families without multidimensional support
        reject values above one instead of silently fitting one factor.
    n_categories : int or sequence of int, optional
        Polytomous category counts. When omitted they are inferred from
        ``responses``. Ignored for dichotomous families.
    item_names : sequence of str, optional
        Item labels.
    responses : ndarray of shape (n_persons, n_items), optional
        Validated responses with missing values coded as negative numbers.
        When supplied, observed codes are checked against the model family.

    Returns
    -------
    BaseItemModel
        An unfitted model.

    Raises
    ------
    MirtModelError
        If the family is unknown or does not support ``n_factors``.
    MirtValidationError
        If ``n_factors`` or ``n_categories`` is invalid.
    MirtDataError
        If ``responses`` contains codes outside the model's categories.
    """
    model_class = item_model_class(name)
    n_factors = validate_n_factors(n_factors)
    if n_factors > 1 and not model_class.supports_multidimensional:
        raise MirtModelError(
            f"{name} does not support multidimensional models",
            model_type=name,
            n_factors=n_factors,
        )

    kwargs: dict[str, Any] = {
        "n_items": n_items,
        "item_names": None if item_names is None else list(item_names),
    }
    if model_class.supports_multidimensional:
        kwargs["n_factors"] = n_factors

    if name in POLYTOMOUS_FAMILIES:
        counts = resolve_category_counts(n_items, n_categories, responses)
        if responses is not None and np.any(responses >= np.asarray(counts)):
            raise MirtDataError(
                "polytomous response codes must be below n_categories for each item",
                n_persons=responses.shape[0],
                n_items=responses.shape[1],
            )
        kwargs["n_categories"] = counts
    elif responses is not None and np.any(responses > 1):
        raise MirtDataError(
            "dichotomous responses must be coded as 0 or 1",
            n_persons=responses.shape[0],
            n_items=responses.shape[1],
        )

    return model_class(**kwargs)


def validate_item_types(item_types: Any, n_items: int | None = None) -> str | list[str]:
    """Validate one family name or a sequence of per-item family names.

    Parameters
    ----------
    item_types : str or sequence of str
        A built-in family, or one family per item.
    n_items : int, optional
        Number of items a sequence must name.

    Returns
    -------
    str or list of str
        The family name, or the per-item names as a list.

    Raises
    ------
    MirtModelError
        If a family is unknown.
    MirtValidationError
        If the sequence is empty or its length differs from ``n_items``.
    """
    if isinstance(item_types, str):
        item_model_class(item_types)
        return item_types
    if isinstance(item_types, (bytes, Mapping)) or not isinstance(
        item_types, (Sequence, np.ndarray)
    ):
        raise MirtModelError(
            "model must be a family name or a sequence of per-item family names",
            model_type=type(item_types).__name__,
            expected=", ".join(ITEM_MODEL_FAMILIES),
        )
    names = [str(name) if isinstance(name, np.str_) else name for name in item_types]
    if not names:
        raise MirtValidationError(
            "a per-item model sequence must name at least one item",
            parameter="model",
            value=names,
        )
    for name in names:
        if not isinstance(name, str):
            # A nested sequence is not a family, nor a sequence of them.
            raise MirtModelError(
                f"Unknown model: {name!r}",
                model_type=str(name),
                expected=", ".join(ITEM_MODEL_FAMILIES),
            )
        item_model_class(name)
    if n_items is not None and len(names) != n_items:
        raise MirtValidationError(
            f"model names {len(names)} items but the data have {n_items}",
            parameter="model",
            value=len(names),
            expected=str(n_items),
        )
    return names


def single_item_family(item_types: Any, n_items: int, *, operation: str) -> str:
    """Return the one family named by a family name or per-item sequence.

    A sequence that names one family for every item is that family, as in
    ``fit_mirt``.

    Raises
    ------
    MirtModelError
        If a family is unknown.
    MirtValidationError
        If the sequence has the wrong length or names several families,
        which ``operation`` does not support.
    """
    names = validate_item_types(item_types, n_items)
    if isinstance(names, str):
        return names
    families = list(dict.fromkeys(names))
    if len(families) > 1:
        raise MirtValidationError(
            f"{operation} does not support mixed item families "
            f"({', '.join(families)}); fit a mixed-format test with "
            "fit_mirt(data, model=[...]) without it",
            parameter="model",
            value=families,
            expected="one family for every item",
        )
    return families[0]


def build_mixed_item_model(
    item_types: Sequence[str],
    *,
    n_factors: Any = 1,
    n_categories: int | Sequence[int] | None = None,
    item_names: Sequence[str] | None = None,
    responses: NDArray[np.int_] | None = None,
) -> MixedItemModel:
    """Construct a mixed-format model from one built-in family per item.

    Items of one family form one component, in order of first appearance,
    and each component is built by :func:`build_item_model`.

    Parameters
    ----------
    item_types : sequence of str
        Family of every item.
    n_factors : int, default=1
        Number of latent factors; every family must support it.
    n_categories : int or sequence of int, optional
        Category count of every polytomous item, or one count per item in
        which dichotomous items have 2. Inferred from ``responses`` when
        omitted.
    item_names : sequence of str, optional
        Item names in test order.
    responses : ndarray of shape (n_persons, n_items), optional
        Validated responses; observed codes are checked against each family.

    Returns
    -------
    MixedItemModel
        An unfitted model.

    Raises
    ------
    MirtModelError
        If a family is unknown or does not support ``n_factors``.
    MirtValidationError
        If the names, ``n_factors`` or ``n_categories`` are invalid.
    MirtDataError
        If ``responses`` contains codes outside an item's categories.
    """
    from mirt.models.mixed_format import MixedItemModel

    if isinstance(item_types, str):
        raise MirtValidationError(
            "item_types must be a sequence with one family name per item",
            parameter="item_types",
            value=item_types,
        )
    names = validate_item_types(
        item_types, None if responses is None else responses.shape[1]
    )
    assert isinstance(names, list)
    n_items = len(names)
    if item_names is not None and len(item_names) != n_items:
        raise MirtValidationError(
            f"Length of item_names ({len(item_names)}) must match n_items ({n_items})",
            parameter="item_names",
            value=len(item_names),
            expected=str(n_items),
        )
    counts = (
        None if n_categories is None else resolve_category_counts(n_items, n_categories)
    )
    families = np.asarray(names)
    if isinstance(counts, list):
        binary = ~np.isin(families, list(POLYTOMOUS_FAMILIES))
        if np.any(np.asarray(counts)[binary] != 2):
            raise MirtValidationError(
                "n_categories must be 2 for dichotomous items",
                parameter="n_categories",
                value=n_categories,
            )

    components = []
    for family in dict.fromkeys(names):
        items = np.flatnonzero(families == family)
        positions = items.tolist()
        component = build_item_model(
            family,
            len(positions),
            n_factors=n_factors,
            n_categories=(
                [counts[item] for item in positions]
                if isinstance(counts, list)
                else counts
            ),
            item_names=(
                None if item_names is None else [item_names[i] for i in positions]
            ),
            responses=None if responses is None else responses[:, items],
        )
        components.append((component, items))
    return MixedItemModel(
        components, item_names=None if item_names is None else list(item_names)
    )
