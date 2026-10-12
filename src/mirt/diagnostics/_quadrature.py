"""Dimension-aware quadrature defaults shared by the diagnostics.

Diagnostics that integrate over the latent population use tensor-product
Gauss-Hermite grids of ``n_quadpts ** n_factors`` nodes, so a fixed number of
points per dimension becomes intractable with a few factors. These helpers
give every diagnostic the same default grid.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np

from mirt.exceptions import MirtValidationError

# Default grids are shrunk until they have at most this many nodes.
DEFAULT_MAX_GRID_NODES = 100_000


def default_diagnostic_quadpts(
    n_factors: int,
    *,
    max_nodes: int = DEFAULT_MAX_GRID_NODES,
) -> int:
    """Return the default quadrature points per latent dimension.

    The rule is that of EAP scoring in :func:`mirt.fscores`: 49 points for one
    or two factors, 21 for three, 9 for four, 7 for five and 5 for more. The
    count is then reduced, to no fewer than two points, until the product grid
    has at most ``max_nodes`` nodes. With the default limit, six and seven
    factors keep 5 points and eight factors use 4.

    Parameters
    ----------
    n_factors : int
        Number of latent dimensions.
    max_nodes : int, default=100000
        Largest number of grid nodes a default grid may have.

    Returns
    -------
    int
        Quadrature points per dimension.
    """
    from mirt.scoring.eap import _default_n_quadpts

    n_factors = int(n_factors)
    n_points = _default_n_quadpts(n_factors)
    while n_points > 2 and n_points**n_factors > max_nodes:
        n_points -= 1
    return n_points


def resolve_diagnostic_quadpts(
    n_quadpts: int | None,
    n_factors: int,
    *,
    minimum: int = 2,
) -> int:
    """Validate an explicit grid size or return the dimension-aware default.

    Parameters
    ----------
    n_quadpts : int or None
        Requested quadrature points per dimension. ``None`` selects
        :func:`default_diagnostic_quadpts`.
    n_factors : int
        Number of latent dimensions.
    minimum : int, default=2
        Smallest accepted explicit grid size.

    Returns
    -------
    int
        Quadrature points per dimension.

    Raises
    ------
    MirtValidationError
        If ``n_quadpts`` is not an integer of at least ``minimum``.
    """
    if n_quadpts is None:
        return default_diagnostic_quadpts(n_factors)
    if isinstance(n_quadpts, (bool, np.bool_)) or not isinstance(n_quadpts, Integral):
        raise MirtValidationError(
            "n_quadpts must be an integer or None",
            parameter="n_quadpts",
            value=n_quadpts,
        )
    if n_quadpts < minimum:
        raise MirtValidationError(
            f"n_quadpts must be at least {minimum}",
            parameter="n_quadpts",
            value=n_quadpts,
        )
    return int(n_quadpts)
