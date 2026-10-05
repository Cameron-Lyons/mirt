"""Model syntax in the style of R's ``mirt.model`` and confirmatory fitting.

:func:`mirt_model` parses a specification such as::

    F1 = 1-5
    F2 = 6-10
    COV = F1*F2
    START = (1, a1, 1.5)
    FIXED = (1, a1)
    PRIOR = (1-10, d, norm, 0, 2)

into a :class:`ModelSpec`, which ``fit_mirt(data, spec=...)`` fits as a
confirmatory model with estimated factor correlations.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mirt.exceptions import MirtModelError, MirtValidationError

if TYPE_CHECKING:
    from mirt.estimation.latent_density import FactorCovarianceDensity
    from mirt.estimation.priors import Prior, PriorSpecification
    from mirt.estimation.standard_errors import StandardErrorMethod
    from mirt.models.base import BaseItemModel
    from mirt.results.fit_result import FitResult

_KEYWORDS = ("COV", "FIXED", "START", "PRIOR", "CONSTRAIN")
# R mirt.model keywords outside the supported subset.
_UNSUPPORTED_KEYWORDS = frozenset(
    {"MEAN", "CONSTRAINB", "LBOUND", "UBOUND", "FREE", "NEXPLORE"}
)
_PRIOR_DISTRIBUTIONS = ("norm", "lnorm", "beta")
# Trailing values after the parameter names of each keyword's groups.
_TRAILING_VALUES = {"FIXED": 0, "CONSTRAIN": 0, "START": 1, "PRIOR": 3}
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*\Z")
_PARAMETER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_ITEM_RANGE = re.compile(r"(\d+)\s*(?:[-:]\s*(\d+))?\Z")
_SLOPE_ALIAS = re.compile(r"a(\d*)\Z")
# Short R names for stored parameters; ``a`` and ``a1``, ``a2``, ... select
# slope columns.
_PARAMETER_ALIASES = {"d": "intercepts", "g": "guessing", "u": "upper"}
_ALIAS_TOKEN = re.compile(r"(a\d*|d|g|u)\Z")
_CONFIRMATORY_FAMILIES = ("2PL", "GRM", "GPCM")


def _syntax_error(message: str, line: int) -> MirtValidationError:
    prefix = f"line {line}: " if line else ""
    return MirtValidationError(
        f"{prefix}{message}",
        parameter="syntax",
        **({"line": line} if line else {}),
    )


def _format_items(items: Sequence[int]) -> str:
    """Format zero-based indices as one-based ranges such as ``1-5, 7``."""
    parts: list[str] = []
    ordered = sorted(items)
    start = previous = ordered[0]
    for index in [*ordered[1:], None]:
        if index is not None and index == previous + 1:
            previous = index
            continue
        parts.append(
            str(start + 1) if start == previous else f"{start + 1}-{previous + 1}"
        )
        if index is not None:
            start = previous = index
    return ", ".join(parts)


def _item_tuple(items: Any, what: str) -> tuple[int, ...]:
    try:
        values = tuple(items)
    except TypeError:
        raise MirtValidationError(
            f"{what} must be a sequence of item indices", parameter="spec"
        ) from None
    if not values or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
        for value in values
    ):
        raise MirtValidationError(
            f"{what} must list zero-based item indices",
            parameter="spec",
            value=values,
        )
    if len(set(values)) != len(values):
        raise MirtValidationError(
            f"{what} lists an item more than once", parameter="spec", value=values
        )
    return tuple(sorted(int(value) for value in values))


@dataclass(frozen=True)
class ParameterEntry:
    """Item parameters selected by a ``FIXED``, ``START`` or ``PRIOR`` group.

    ``CONSTRAIN`` groups use the same type.

    Attributes
    ----------
    items : tuple of int
        Zero-based item indices.
    parameters : tuple of str
        Parameter names as written: ``a`` or ``a1``, ``a2``, ... for slopes,
        ``d`` for intercepts, ``g`` for guessing, ``u`` for upper asymptotes,
        or a stored parameter name such as ``"thresholds"``.
    value : float, optional
        Starting value of a ``START`` group.
    prior : tuple of (str, float, float), optional
        Distribution (``"norm"``, ``"lnorm"`` or ``"beta"``) and its two
        arguments for a ``PRIOR`` group.
    line : int
        Source line in the syntax, or 0 for entries built directly. It is not
        part of equality.
    """

    items: tuple[int, ...]
    parameters: tuple[str, ...]
    value: float | None = None
    prior: tuple[str, float, float] | None = None
    line: int = field(default=0, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", _item_tuple(self.items, "entry items"))
        parameters = (
            (self.parameters,)
            if isinstance(self.parameters, str)
            else tuple(self.parameters)
        )
        if not parameters or not all(
            isinstance(name, str) and _PARAMETER.match(name) for name in parameters
        ):
            raise _syntax_error(
                f"invalid parameter names {parameters!r}; use names such as a1 or d",
                self.line,
            )
        object.__setattr__(self, "parameters", parameters)
        if self.value is not None:
            try:
                value = float(self.value)
            except (TypeError, ValueError):
                value = np.nan
            if not np.isfinite(value):
                raise _syntax_error("starting values must be finite numbers", self.line)
            object.__setattr__(self, "value", value)
        if self.prior is not None:
            object.__setattr__(self, "prior", _validate_prior(self.prior, self.line))

    def _syntax(self) -> str:
        parts = [_format_items(self.items), *self.parameters]
        if self.value is not None:
            parts.append(repr(self.value))
        if self.prior is not None:
            distribution, first, second = self.prior
            parts.extend((distribution, repr(first), repr(second)))
        return f"({', '.join(parts)})"


def _validate_prior(prior: Any, line: int) -> tuple[str, float, float]:
    try:
        distribution, first, second = prior
        first, second = float(first), float(second)
    except (TypeError, ValueError):
        raise _syntax_error(
            "a prior must be (distribution, first, second)", line
        ) from None
    if distribution not in _PRIOR_DISTRIBUTIONS:
        raise _syntax_error(
            f"unknown prior distribution {distribution!r}; use "
            + ", ".join(_PRIOR_DISTRIBUTIONS),
            line,
        )
    if not (np.isfinite(first) and np.isfinite(second)):
        raise _syntax_error("prior arguments must be finite", line)
    if second <= 0.0 or (distribution == "beta" and first <= 0.0):
        raise _syntax_error(
            f"{distribution} prior arguments must be positive"
            if distribution == "beta"
            else f"{distribution} prior standard deviation must be positive",
            line,
        )
    return str(distribution), first, second


@dataclass(frozen=True, repr=False)
class ModelSpec:
    """Confirmatory model specification parsed by :func:`mirt_model`.

    Item indices are zero-based; the syntax and :meth:`to_syntax` number
    items from one.

    Attributes
    ----------
    factors : tuple of str
        Factor names in order.
    loadings : tuple of tuple of int
        Items loading on each factor.
    covariances : tuple of (str, str)
        Free factor covariances. A pair of distinct factors frees their
        correlation; a repeated factor frees its variance.
    fixed, start, priors, constraints : tuple of ParameterEntry
        ``FIXED``, ``START``, ``PRIOR`` and ``CONSTRAIN`` groups.
    item_names : tuple of str, optional
        Item names the syntax was resolved against.
    """

    factors: tuple[str, ...]
    loadings: tuple[tuple[int, ...], ...]
    covariances: tuple[tuple[str, str], ...] = ()
    fixed: tuple[ParameterEntry, ...] = ()
    start: tuple[ParameterEntry, ...] = ()
    priors: tuple[ParameterEntry, ...] = ()
    constraints: tuple[ParameterEntry, ...] = ()
    item_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        factors = tuple(self.factors)
        if not factors:
            raise MirtValidationError("a model needs at least one factor")
        for name in factors:
            if not isinstance(name, str) or not _NAME.match(name):
                raise MirtValidationError(
                    f"invalid factor name {name!r}", parameter="factors"
                )
            if name in _KEYWORDS or name in _UNSUPPORTED_KEYWORDS:
                raise MirtValidationError(
                    f"{name} is a keyword, not a factor name", parameter="factors"
                )
        if len(set(factors)) != len(factors):
            raise MirtValidationError(
                "factor names must be unique", parameter="factors", value=factors
            )
        raw_loadings = tuple(self.loadings)
        if len(raw_loadings) != len(factors):
            raise MirtValidationError(
                "loadings must list the items of every factor", parameter="loadings"
            )
        loadings = tuple(
            _item_tuple(items, f"loadings of {name}")
            for name, items in zip(factors, raw_loadings, strict=True)
        )
        position = {name: index for index, name in enumerate(factors)}
        covariances = set()
        for pair in self.covariances:
            try:
                first, second = pair
            except (TypeError, ValueError):
                raise MirtValidationError(
                    "covariances must be pairs of factor names",
                    parameter="covariances",
                    value=pair,
                ) from None
            if first not in position or second not in position:
                raise MirtValidationError(
                    f"covariance {first}*{second} names an unknown factor",
                    parameter="covariances",
                )
            covariances.add(tuple(sorted((first, second), key=position.__getitem__)))
        item_names = None
        if self.item_names is not None:
            item_names = _validate_item_names(self.item_names)
        object.__setattr__(self, "factors", factors)
        object.__setattr__(self, "loadings", loadings)
        object.__setattr__(
            self,
            "covariances",
            tuple(
                sorted(
                    covariances,
                    key=lambda pair: (position[pair[0]], position[pair[1]]),
                )
            ),
        )
        object.__setattr__(self, "item_names", item_names)
        for kind in ("fixed", "start", "priors", "constraints"):
            entries = tuple(getattr(self, kind))
            if not all(isinstance(entry, ParameterEntry) for entry in entries):
                raise MirtValidationError(
                    f"{kind} must contain ParameterEntry objects", parameter=kind
                )
            object.__setattr__(self, kind, entries)
        for entry in self.start:
            if entry.value is None:
                raise _syntax_error("START groups need a value", entry.line)
        for entry in self.priors:
            if entry.prior is None:
                raise _syntax_error("PRIOR groups need a distribution", entry.line)
        if item_names is not None and self._largest_item() >= len(item_names):
            raise MirtValidationError(
                f"the model refers to item {self._largest_item() + 1}, but only "
                f"{len(item_names)} item names were given",
                parameter="item_names",
            )

    @property
    def n_factors(self) -> int:
        """Number of factors."""
        return len(self.factors)

    def _entries(self) -> tuple[ParameterEntry, ...]:
        return (*self.fixed, *self.start, *self.priors, *self.constraints)

    def _largest_item(self) -> int:
        groups = [*self.loadings, *(entry.items for entry in self._entries())]
        return max(max(items) for items in groups)

    def loading_pattern(self, n_items: int | None = None) -> NDArray[np.bool_]:
        """Return the ``(n_items, n_factors)`` Boolean loading pattern.

        Parameters
        ----------
        n_items : int, optional
            Number of items. Defaults to the number of item names, or else to
            the largest item the specification refers to.

        Raises
        ------
        MirtValidationError
            If the specification refers to an item beyond ``n_items``.
        """
        if n_items is None:
            n_items = (
                len(self.item_names)
                if self.item_names is not None
                else self._largest_item() + 1
            )
        if self._largest_item() >= n_items:
            raise MirtValidationError(
                f"the model refers to item {self._largest_item() + 1}, but the "
                f"data have {n_items} items",
                parameter="spec",
            )
        pattern = np.zeros((n_items, self.n_factors), dtype=np.bool_)
        for column, items in enumerate(self.loadings):
            pattern[list(items), column] = True
        return pattern

    def covariance_pattern(self) -> NDArray[np.bool_]:
        """Return the symmetric Boolean mask of free factor (co)variances."""
        position = {name: index for index, name in enumerate(self.factors)}
        pattern = np.zeros((self.n_factors, self.n_factors), dtype=np.bool_)
        for first, second in self.covariances:
            pattern[position[first], position[second]] = True
            pattern[position[second], position[first]] = True
        return pattern

    def to_syntax(self) -> str:
        """Return equivalent syntax with items numbered from one."""
        lines = [
            f"{name} = {_format_items(items)}"
            for name, items in zip(self.factors, self.loadings, strict=True)
        ]
        if self.covariances:
            pairs = ", ".join(f"{first}*{second}" for first, second in self.covariances)
            lines.append(f"COV = {pairs}")
        for keyword, entries in (
            ("FIXED", self.fixed),
            ("START", self.start),
            ("PRIOR", self.priors),
            ("CONSTRAIN", self.constraints),
        ):
            if entries:
                groups = ", ".join(entry._syntax() for entry in entries)
                lines.append(f"{keyword} = {groups}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_syntax()

    def __repr__(self) -> str:
        body = "\n".join(f"    {line}" for line in self.to_syntax().splitlines())
        return f"ModelSpec(\n{body}\n)"


def _validate_item_names(item_names: Any) -> tuple[str, ...]:
    if isinstance(item_names, str):
        raise MirtValidationError(
            "item_names must be a sequence of strings", parameter="item_names"
        )
    try:
        names = tuple(item_names)
    except TypeError:
        raise MirtValidationError(
            "item_names must be a sequence of strings", parameter="item_names"
        ) from None
    if not all(isinstance(name, str) and name for name in names):
        raise MirtValidationError(
            "item_names must be non-empty strings", parameter="item_names"
        )
    if len(set(names)) != len(names):
        raise MirtValidationError("item_names must be unique", parameter="item_names")
    return names


def mirt_model(syntax: str, item_names: Sequence[str] | None = None) -> ModelSpec:
    """Parse ``mirt.model``-style model syntax into a :class:`ModelSpec`.

    Each statement has the form ``NAME = value``. ``#`` starts a comment,
    blank lines and surrounding whitespace are ignored, and a statement whose
    value ends with a comma or an open parenthesis continues on the next
    line. Items are numbered from one, ranges use ``-`` or ``:``, and with
    ``item_names`` items can also be named, including ranges such as
    ``Q1-Q5`` in item order. Numbers always refer to item positions, even
    for items whose names are numbers. Supported statements:

    ``F1 = 1-5, 7``
        A factor and the items loading on it. Every name other than a
        keyword defines a factor.
    ``COV = F1*F2, F1*F3``
        Free factor correlations; ``F1*F2*F3`` frees every pair among them.
        Pairs not listed have zero correlation. ``F1*F1`` frees the variance
        of ``F1``, which is identified only when ``FIXED`` holds a nonzero
        slope on ``F1``. Other variances are one.
    ``FIXED = (1, a1), (2-3, d)``
        Hold parameters at their ``START`` values or the model defaults.
    ``START = (1-5, a1, 1.5)``
        Starting values.
    ``PRIOR = (1-10, a1, lnorm, 0, 0.5)``
        Item priors for Bayes modal estimation: ``norm`` (mean, standard
        deviation), ``lnorm`` (log-scale mean and standard deviation) or
        ``beta`` (two shape parameters). Every free coordinate of a stored
        parameter must receive the same prior.
    ``CONSTRAIN = (1-3, a1)``
        Equality constraints. They are parsed, but :func:`mirt.fit_mirt`
        does not fit them yet.

    Parameter names are resolved against the model when it is fitted:
    ``a`` selects the slopes and ``a1``, ``a2``, ... the slopes on each
    factor, ``d`` the intercepts of slope-intercept models, ``g`` guessing
    and ``u`` upper asymptotes. Stored parameter names such as
    ``difficulty`` or ``thresholds`` select whole item rows.

    Parameters
    ----------
    syntax : str
        Model specification.
    item_names : sequence of str, optional
        Item names, to refer to items by name.

    Returns
    -------
    ModelSpec
        The parsed specification.

    Raises
    ------
    MirtValidationError
        If the syntax is malformed. The message starts with the line number,
        which is also available as ``error.context["line"]``.
    NotImplementedError
        For ``mirt.model`` keywords outside the supported subset, such as
        ``MEAN`` or ``LBOUND``.

    Examples
    --------
    >>> spec = mirt_model('''
    ...     F1 = 1-4
    ...     F2 = 5-8
    ...     COV = F1*F2
    ... ''')
    >>> spec.loading_pattern().sum(axis=0)
    array([4, 4])
    """
    if not isinstance(syntax, str):
        raise MirtValidationError(
            "syntax must be a string",
            parameter="syntax",
            value=type(syntax).__name__,
            expected="str",
        )
    names = None if item_names is None else _validate_item_names(item_names)
    lookup = (
        None if names is None else {name: index for index, name in enumerate(names)}
    )

    factors: dict[str, tuple[int, ...]] = {}
    keyword_statements = []
    for line, name, value in _statements(syntax):
        if name in _KEYWORDS:
            keyword_statements.append((line, name, value))
        elif name in _UNSUPPORTED_KEYWORDS:
            raise NotImplementedError(
                f"line {line}: {name} is not supported; supported keywords are "
                + ", ".join(_KEYWORDS)
            )
        elif name.upper() in _KEYWORDS or name.upper() in _UNSUPPORTED_KEYWORDS:
            raise _syntax_error(f"keywords are upper case; write {name.upper()}", line)
        elif not _NAME.match(name):
            raise _syntax_error(f"invalid factor name {name!r}", line)
        elif name in factors:
            raise _syntax_error(f"factor {name} is defined twice", line)
        else:
            factors[name] = _items(_split(value, line), lookup, line, f"factor {name}")
    if not factors:
        raise MirtValidationError(
            "the model syntax defines no factors", parameter="syntax"
        )

    covariances: list[tuple[str, str]] = []
    entries: dict[str, list[ParameterEntry]] = {
        keyword: [] for keyword in _TRAILING_VALUES
    }
    for line, keyword, value in keyword_statements:
        if keyword == "COV":
            covariances.extend(_covariances(value, factors, line))
        else:
            entries[keyword].extend(
                _parameter_entry(keyword, tokens, lookup, line)
                for tokens in _groups(value, line)
            )
    return ModelSpec(
        factors=tuple(factors),
        loadings=tuple(factors.values()),
        covariances=tuple(covariances),
        fixed=tuple(entries["FIXED"]),
        start=tuple(entries["START"]),
        priors=tuple(entries["PRIOR"]),
        constraints=tuple(entries["CONSTRAIN"]),
        item_names=names,
    )


def _continues(value: str) -> bool:
    return value.endswith(",") or value.count("(") > value.count(")")


def _statements(syntax: str) -> list[tuple[int, str, str]]:
    """Split syntax into ``(line, name, value)`` statements."""
    statements: list[list[Any]] = []
    for line, raw in enumerate(syntax.splitlines(), start=1):
        text = raw.split("#", 1)[0].strip()
        if not text:
            continue
        if statements and _continues(statements[-1][2]):
            statements[-1][2] = f"{statements[-1][2]} {text}"
            continue
        name, separator, value = text.partition("=")
        name, value = name.strip(), value.strip()
        if not separator or not name:
            raise _syntax_error(f"expected 'NAME = value', got {text!r}", line)
        if not value:
            raise _syntax_error(f"{name} has no value", line)
        statements.append([line, name, value])
    for line, name, value in statements:
        if _continues(value):
            raise _syntax_error(f"{name} is incomplete", line)
    return [(line, name, value) for line, name, value in statements]


def _split(value: str, line: int) -> list[str]:
    tokens = [token.strip() for token in value.split(",")]
    if not all(tokens):
        raise _syntax_error(f"empty entry in {value!r}", line)
    return tokens


def _item_token(
    token: str, lookup: Mapping[str, int] | None, line: int
) -> list[int] | None:
    """Return the items of one token, or ``None`` if it names no item.

    Numbers are always one-based item positions, also when an item is named
    by a number, so ``1`` means the first item whatever the item names are.
    """
    match = _ITEM_RANGE.match(token)
    if match:
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if first < 1 or last < first:
            raise _syntax_error(f"invalid item range {token!r}", line)
        if lookup is not None and last > len(lookup):
            raise _syntax_error(
                f"item {last} is beyond the {len(lookup)} named items", line
            )
        return list(range(first - 1, last))
    if lookup is not None and token in lookup:
        return [lookup[token]]
    if lookup is not None:
        for position, character in enumerate(token):
            if character not in "-:":
                continue
            first_name = token[:position].strip()
            last_name = token[position + 1 :].strip()
            if first_name in lookup and last_name in lookup:
                if lookup[last_name] < lookup[first_name]:
                    raise _syntax_error(f"item range {token!r} runs backwards", line)
                return list(range(lookup[first_name], lookup[last_name] + 1))
    return None


def _unknown_item(
    token: str, lookup: Mapping[str, int] | None, line: int
) -> MirtValidationError:
    hint = "" if lookup is not None else "; pass item_names to refer to items by name"
    return _syntax_error(f"unknown item {token!r}{hint}", line)


def _items(
    tokens: Sequence[str], lookup: Mapping[str, int] | None, line: int, what: str
) -> tuple[int, ...]:
    items: list[int] = []
    for token in tokens:
        indices = _item_token(token, lookup, line)
        if indices is None:
            raise _unknown_item(token, lookup, line)
        items.extend(indices)
    if len(set(items)) != len(items):
        raise _syntax_error(f"{what} lists an item more than once", line)
    return tuple(sorted(items))


def _covariances(
    value: str, factors: Mapping[str, Any], line: int
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for token in _split(value, line):
        members = [part.strip() for part in token.split("*")]
        if len(members) < 2 or not all(members):
            raise _syntax_error(f"COV terms look like F1*F2, got {token!r}", line)
        for member in members:
            if member not in factors:
                raise _syntax_error(f"unknown factor {member!r} in COV", line)
        if len(members) == 2 and members[0] == members[1]:
            pairs.append((members[0], members[0]))
        elif len(set(members)) != len(members):
            raise _syntax_error(
                f"COV term {token!r} repeats a factor; write F1*F1 to free a variance",
                line,
            )
        else:
            pairs.extend(combinations(members, 2))
    return pairs


def _groups(value: str, line: int) -> list[list[str]]:
    """Split ``(a, b), (c, d)`` into token lists."""
    groups: list[list[str]] = []
    outside: list[str] = []
    start: int | None = None
    for position, character in enumerate(value):
        if character == "(":
            if start is not None:
                raise _syntax_error("nested parentheses", line)
            start = position + 1
        elif character == ")":
            if start is None:
                raise _syntax_error("unbalanced parentheses", line)
            groups.append(_split(value[start:position], line))
            start = None
        elif start is None:
            outside.append(character)
    if start is not None:
        raise _syntax_error("unbalanced parentheses", line)
    if not groups or "".join(outside).replace(",", "").strip():
        raise _syntax_error(
            "write groups as (items, parameter, ...) separated by commas", line
        )
    return groups


def _number(token: str, line: int) -> float:
    try:
        return float(token)
    except ValueError:
        raise _syntax_error(f"expected a number, got {token!r}", line) from None


def _parameter_entry(
    keyword: str, tokens: Sequence[str], lookup: Mapping[str, int] | None, line: int
) -> ParameterEntry:
    n_values = _TRAILING_VALUES[keyword]
    if len(tokens) < 2 + n_values:
        usage = {
            0: "(items, parameter, ...)",
            1: "(items, parameter, ..., value)",
            3: "(items, parameter, ..., distribution, first, second)",
        }[n_values]
        raise _syntax_error(f"{keyword} groups look like {usage}", line)
    head = tokens[: len(tokens) - n_values]
    values = tokens[len(tokens) - n_values :]
    items: list[int] = []
    parameters: list[str] = []
    for position, token in enumerate(head):
        indices = None
        # Leading tokens are items; a1, d and the like after the first token
        # are parameters even when an item has the same name.
        if not parameters and (position == 0 or not _ALIAS_TOKEN.match(token)):
            indices = _item_token(token, lookup, line)
        if indices is not None:
            items.extend(indices)
        elif position == 0:
            raise _unknown_item(token, lookup, line)
        elif not _PARAMETER.match(token):
            raise _syntax_error(f"invalid parameter name {token!r}", line)
        else:
            parameters.append(token)
    if not parameters:
        raise _syntax_error(f"{keyword} group names no parameter", line)
    if len(set(items)) != len(items):
        raise _syntax_error(f"{keyword} group lists an item more than once", line)
    value = None
    prior = None
    if keyword == "START":
        value = _number(values[0], line)
    elif keyword == "PRIOR":
        prior = (values[0], _number(values[1], line), _number(values[2], line))
    return ParameterEntry(
        tuple(items), tuple(parameters), value=value, prior=prior, line=line
    )


class _ParameterTargets:
    """Resolve syntax parameter names to stored coordinates of one model."""

    def __init__(self, model: BaseItemModel, pattern: NDArray[np.bool_]) -> None:
        from mirt.models.multidimensional import MultidimensionalModel

        self.model = model
        self.pattern = pattern
        parameters = model._parameters
        self.item_parameters = [
            name
            for name, values in parameters.items()
            if values.ndim and values.shape[0] == model.n_items
        ]
        self.slope: str | None = None
        if isinstance(model, MultidimensionalModel):
            self.slope = "slopes"
        elif "discrimination" in parameters:
            self.slope = "discrimination"

    def _target(self, name: str, line: int) -> tuple[str, int | None]:
        if name in self.item_parameters:
            return name, None
        n_factors = self.pattern.shape[1]
        match = _SLOPE_ALIAS.match(name)
        if match and self.slope is not None:
            if not match.group(1):
                return self.slope, None
            factor = int(match.group(1))
            if not 1 <= factor <= n_factors:
                raise _syntax_error(
                    f"{name} refers to factor {factor}, but the model has "
                    f"{n_factors} factor{'s' if n_factors > 1 else ''}",
                    line,
                )
            return self.slope, factor - 1
        alias = _PARAMETER_ALIASES.get(name)
        if alias is not None and alias in self.item_parameters:
            return alias, None
        available = [
            short
            for short, stored in _PARAMETER_ALIASES.items()
            if stored in self.item_parameters
        ]
        if self.slope is not None:
            available = ["a", *(f"a{k + 1}" for k in range(n_factors)), *available]
        available.extend(self.item_parameters)
        raise _syntax_error(
            f"unknown parameter {name!r} for the {self.model.model_name} model; "
            f"use {', '.join(available)}",
            line,
        )

    def masks(self, entry: ParameterEntry) -> dict[str, NDArray[np.bool_]]:
        """Return Boolean masks of the coordinates an entry selects."""
        rows = np.asarray(entry.items, dtype=np.intp)
        masks: dict[str, NDArray[np.bool_]] = {}
        for name in entry.parameters:
            target, column = self._target(name, entry.line)
            shape = self.model._parameters[target].shape
            mask = masks.setdefault(target, np.zeros(shape, dtype=np.bool_))
            if target != self.slope:
                mask[rows] = True
                continue
            # Slope coordinates outside the loading pattern stay at zero.
            selected = np.zeros(self.pattern.shape, dtype=np.bool_)
            if column is None:
                selected[rows] = self.pattern[rows]
            else:
                outside = rows[~self.pattern[rows, column]]
                if outside.size:
                    raise _syntax_error(
                        f"item {outside[0] + 1} does not load on factor {column + 1}",
                        entry.line,
                    )
                selected[rows, column] = True
            mask |= selected.reshape(shape)
        return masks


def _confirmatory_model(
    name: str,
    pattern: NDArray[np.bool_],
    n_categories: int | Sequence[int] | None,
    item_names: list[str],
    responses: NDArray[np.int_],
) -> BaseItemModel:
    """Build an unfitted model whose slopes follow ``pattern``."""
    from mirt.models._factory import build_item_model
    from mirt.models.multidimensional import MultidimensionalModel

    n_items, n_factors = pattern.shape
    if n_factors > 1 and name not in _CONFIRMATORY_FAMILIES:
        raise MirtModelError(
            "multidimensional model syntax supports "
            + ", ".join(_CONFIRMATORY_FAMILIES)
            + " models",
            model_type=name,
            n_factors=n_factors,
        )
    # The factory also checks the response codes against the family.
    model = build_item_model(
        name,
        n_items,
        n_factors=n_factors,
        n_categories=n_categories,
        item_names=item_names,
        responses=responses,
    )
    if n_factors == 1:
        return model
    if name == "2PL":
        return MultidimensionalModel(
            n_items,
            n_factors,
            item_names=list(item_names),
            model_type="confirmatory",
            loading_pattern=pattern.astype(np.float64),
        )
    model.set_parameters(discrimination=model.parameters["discrimination"] * pattern)
    model.set_free_parameter_masks({"discrimination": pattern})
    return model


def _start_values(
    model: BaseItemModel,
    spec: ModelSpec,
    targets: _ParameterTargets,
    start_values: Mapping[str, ArrayLike] | None,
) -> dict[str, NDArray[np.float64]]:
    """Merge ``start_values`` arrays with the coordinates set by ``START``."""
    from mirt.estimation.base import _validate_start

    start: dict[str, NDArray[np.float64]] = {}
    if start_values is not None:
        validated = _validate_start(start_values)
        if isinstance(validated, str):
            raise MirtValidationError(
                "start_values must map parameter names to arrays",
                parameter="start_values",
                value=validated,
            )
        start.update(validated)
    current = model.parameters
    for entry in spec.start:
        for name, mask in targets.masks(entry).items():
            values = start.get(name, current[name])
            if values.shape != current[name].shape:
                raise MirtValidationError(
                    f"starting values for {name!r} must have shape "
                    f"{current[name].shape}",
                    parameter="start_values",
                    value=values.shape,
                )
            values = values.copy()
            values[mask] = entry.value
            start[name] = values
    slope = targets.slope
    if slope in start:
        values = start[slope].reshape(targets.pattern.shape)
        if np.any(values[~targets.pattern] != 0.0):
            raise MirtValidationError(
                f"starting values for {slope!r} must be zero where an item does "
                "not load on a factor",
                parameter="start_values",
            )
    return start


def _spec_priors(
    model: BaseItemModel, spec: ModelSpec, targets: _ParameterTargets
) -> dict[str, Prior]:
    """Translate ``PRIOR`` groups into one prior per stored parameter."""
    from mirt.estimation.priors import BetaPrior, LogNormalPrior, NormalPrior

    distributions: dict[str, type[NormalPrior | LogNormalPrior | BetaPrior]] = {
        "norm": NormalPrior,
        "lnorm": LogNormalPrior,
        "beta": BetaPrior,
    }
    chosen: dict[str, tuple[tuple[str, float, float], int]] = {}
    covered: dict[str, NDArray[np.bool_]] = {}
    for entry in spec.priors:
        assert entry.prior is not None
        for name, mask in targets.masks(entry).items():
            prior, _ = chosen.setdefault(name, (entry.prior, entry.line))
            if prior != entry.prior:
                raise _syntax_error(
                    f"PRIOR gives {name!r} two distributions; item-specific "
                    "priors are not supported",
                    entry.line,
                )
            covered[name] = covered[name] | mask if name in covered else mask
    free = model.free_parameter_masks
    priors: dict[str, Prior] = {}
    for name, ((distribution, first, second), line) in chosen.items():
        if np.any(free[name] & ~covered[name]):
            raise _syntax_error(
                f"PRIOR must cover every free coordinate of {name!r}; "
                "item-specific priors are not supported",
                line,
            )
        priors[name] = distributions[distribution](first, second)
    return priors


def _factor_density(
    model: BaseItemModel,
    spec: ModelSpec,
    targets: _ParameterTargets,
    start: Mapping[str, NDArray[np.float64]],
) -> FactorCovarianceDensity | None:
    """Return the latent density for the free (co)variances, if any."""
    from mirt.estimation.latent_density import FactorCovarianceDensity

    free = spec.covariance_pattern()
    if not np.any(free):
        return None
    pattern = targets.pattern
    anchors = np.zeros(pattern.shape, dtype=np.bool_)
    if targets.slope is not None:
        slope = targets.slope
        values = start.get(slope, model.parameters[slope]).reshape(pattern.shape)
        held = ~model.free_parameter_masks[slope].reshape(pattern.shape)
        anchors = held & pattern & (values != 0.0)
    for factor in np.flatnonzero(np.diag(free) & ~anchors.any(axis=0)):
        name = spec.factors[factor]
        raise MirtValidationError(
            f"COV = {name}*{name} frees the variance of {name}, which is "
            f"identified only when FIXED holds a nonzero slope on {name}",
            parameter="spec",
        )
    return FactorCovarianceDensity(spec.n_factors, free=free)


def _fit_spec(
    responses: NDArray[np.int_],
    spec: ModelSpec | str,
    *,
    model: str,
    n_factors: int,
    n_categories: int | Sequence[int] | None,
    estimation: str,
    n_quadpts: int,
    max_iter: int,
    tol: float,
    verbose: bool,
    item_names: list[str] | None,
    use_rust: bool,
    compute_standard_errors: bool,
    start_values: Mapping[str, ArrayLike] | None,
    fixed: Mapping[str, ArrayLike] | None,
    priors: PriorSpecification | Mapping[str, Prior] | None,
    se_method: StandardErrorMethod,
    accelerate: Literal["none", "squarem"] = "none",
) -> FitResult:
    """Fit a confirmatory model for ``fit_mirt(data, spec=...)``.

    ``fit_mirt`` has validated the responses, the family and ``n_factors``;
    ``item_names`` are the caller's or the data's names, if any.
    """
    from mirt.estimation._item_priors import (
        _SPECIFICATION_FIELDS,
        resolve_item_priors,
    )
    from mirt.estimation.base import _free_masks_from_fixed
    from mirt.estimation.em import EMEstimator
    from mirt.estimation.priors import PriorSpecification

    if estimation != "EM":
        raise MirtValidationError(
            "model syntax is fitted only with estimation='EM'",
            parameter="estimation",
            value=estimation,
            expected="EM",
        )
    n_items = responses.shape[1]
    if item_names is not None and len(item_names) != n_items:
        raise MirtValidationError(
            f"item_names has {len(item_names)} names, but the data have "
            f"{n_items} items",
            parameter="item_names",
        )
    if isinstance(spec, str):
        spec = mirt_model(spec, item_names=item_names)
    elif not isinstance(spec, ModelSpec):
        raise MirtValidationError(
            "spec must be a ModelSpec or a model syntax string",
            parameter="spec",
            value=type(spec).__name__,
        )
    if spec.item_names is not None:
        if len(spec.item_names) != n_items:
            raise MirtValidationError(
                f"spec was parsed with {len(spec.item_names)} item names, but "
                f"the data have {n_items} items",
                parameter="spec",
            )
        if item_names is None:
            item_names = list(spec.item_names)
        elif tuple(item_names) != spec.item_names:
            raise MirtValidationError(
                "spec was parsed with item names that differ from the data's",
                parameter="spec",
            )
    if item_names is None:
        item_names = [f"Item_{index + 1}" for index in range(n_items)]
    if n_factors not in (1, spec.n_factors):
        raise MirtValidationError(
            f"n_factors={n_factors} conflicts with the {spec.n_factors} factors "
            "of spec",
            parameter="n_factors",
            value=n_factors,
            expected=str(spec.n_factors),
        )
    if spec.constraints:
        raise NotImplementedError(
            f"line {spec.constraints[0].line}: CONSTRAIN equality constraints "
            "are not supported by fit_mirt yet"
        )
    pattern = spec.loading_pattern(n_items)
    unloaded = np.flatnonzero(~pattern.any(axis=1))
    if unloaded.size:
        names = ", ".join(item_names[index] for index in unloaded[:5])
        raise MirtValidationError(
            f"every item must load on a factor; {names} load on none",
            parameter="spec",
        )

    irt_model = _confirmatory_model(model, pattern, n_categories, item_names, responses)
    targets = _ParameterTargets(irt_model, pattern)
    free = irt_model.free_parameter_masks
    for entry in spec.fixed:
        for name, mask in targets.masks(entry).items():
            free[name] &= ~mask
    if fixed is not None:
        for name, user_free in _free_masks_from_fixed(irt_model, fixed).items():
            free[name] &= user_free
    irt_model.set_free_parameter_masks(free)

    start = _start_values(irt_model, spec, targets, start_values)
    spec_priors = _spec_priors(irt_model, spec, targets)
    if spec_priors and priors is not None:
        raise MirtValidationError(
            "give priors either in the model syntax or through priors=, not both",
            parameter="priors",
        )
    if (
        isinstance(priors, PriorSpecification)
        and any(getattr(priors, name) is not None for name in _SPECIFICATION_FIELDS)
        and not resolve_item_priors(priors, irt_model)
    ):
        raise MirtValidationError(
            "the PriorSpecification sets no prior on the parameters of the "
            f"{irt_model.model_name} model ({', '.join(irt_model.parameters)}); "
            "pass priors as a mapping of these names or use PRIOR",
            parameter="priors",
        )
    density = _factor_density(irt_model, spec, targets, start)
    estimator = EMEstimator(
        n_quadpts=n_quadpts,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        latent_density=density,
        use_rust=use_rust,
        compute_standard_errors=compute_standard_errors,
        item_priors=spec_priors or priors,
        se_method=se_method,
        accelerate=accelerate,
    )
    result = estimator.fit(irt_model, responses, start=start or "default")
    if density is None:
        return result
    return replace(result, latent_covariance=density.cov.copy())
