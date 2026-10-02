"""Original model hooks used to qualify built-in numerical shortcuts."""

from __future__ import annotations

from typing import Any, TypeVar

_Model = TypeVar("_Model")
_KERNEL_HOOKS = (
    "probability",
    "_category_probabilities",
    "_evaluate_logistic",
    "_curve_parameters",
    "_logits",
    "_ensure_theta_2d",
    "parameters",
    "set_parameters",
    "set_item_parameter",
    "_canonical_parameter_values",
    "_validate_parameter_state",
    "free_parameter_masks",
    "_apply_free_parameter_restrictions",
    "slopes",
    "intercepts",
    "general_loadings",
    "specific_loadings",
)
_LIKELIHOOD_HOOKS = (
    "log_likelihood",
    "log_likelihood_batch",
    "_validate_dichotomous_responses",
    "_validate_polytomous_responses",
)
_DEFAULT_HOOKS: dict[type, tuple[tuple[tuple[str, type, Any], ...], ...]] = {}
_BASE_HOOKS: dict[type, dict[str, Any]] = {}
_MISSING = object()


def record_model_base(cls: type[_Model]) -> type[_Model]:
    """Retain base definitions even if they change before a concrete import."""
    _BASE_HOOKS[cls] = {
        name: vars(cls)[name]
        for name in _KERNEL_HOOKS + _LIKELIHOOD_HOOKS + ("information",)
        if name in vars(cls)
    }
    return cls


def register_builtin_model(cls: type[_Model]) -> type[_Model]:
    """Capture authored hooks before downstream code can replace them.

    This decorator returns the same class and imports no other model modules.
    Inherited hooks are recorded too; subclassing alone does not register a
    new model as eligible for a built-in shortcut.
    """
    record_model_base(cls)
    groups = []
    for names in (_KERNEL_HOOKS, _LIKELIHOOD_HOOKS):
        hooks = []
        for name in names:
            definitions = []
            for owner in cls.__mro__:
                authored = _BASE_HOOKS.get(owner, vars(owner))
                if name in authored:
                    definitions.append((owner, authored[name]))
            if not definitions:
                continue
            # Resolved lookup also detects new attributes on the concrete
            # class, without checking the defining parent twice.
            hooks.append((name, cls, definitions[0][1]))
            # Overridden methods and properties can still call super().
            hooks.extend((name, owner, original) for owner, original in definitions[1:])
        groups.append(tuple(hooks))
    _DEFAULT_HOOKS[cls] = tuple(groups)
    return cls


def original_model_hook(cls: type, name: str) -> Any:
    """Return an authored hook even after a model or base class is changed."""
    for owner in cls.__mro__:
        hooks = _BASE_HOOKS.get(owner, {})
        if name in hooks:
            return hooks[name]
    raise KeyError(f"No original model hook recorded for {cls.__name__}.{name}")


def uses_builtin_model_hooks(model: object, *, likelihood: bool = False) -> bool:
    """Require original class hooks with no overriding instance attributes."""
    cls = type(model)
    defaults = _DEFAULT_HOOKS.get(cls)
    if defaults is None:
        return False
    namespace = vars(model)
    for hooks in defaults[: 2 if likelihood else 1]:
        for name, owner, original in hooks:
            if (owner is cls and name in namespace) or getattr(
                owner, name, _MISSING
            ) is not original:
                return False
    return True


def uses_original_model_hook(model: object, name: str) -> bool:
    """Check a known hook and its recorded ancestors without limiting curves."""
    cls = type(model)
    try:
        original = original_model_hook(cls, name)
    except KeyError:
        return False
    if name in vars(model) or getattr(cls, name, _MISSING) is not original:
        return False
    for owner in cls.__mro__:
        authored = _BASE_HOOKS.get(owner, {})
        if name in authored and getattr(owner, name, _MISSING) is not authored[name]:
            return False
    return True
