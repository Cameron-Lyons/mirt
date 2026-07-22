"""Example: multigroup fit with invariance constraints."""

from __future__ import annotations

import numpy as np

import mirt
from mirt.multigroup import fit_multigroup


def main() -> None:
    rng = np.random.default_rng(3)
    n_items = 12
    a = rng.uniform(0.7, 1.7, size=n_items)
    b = rng.normal(0.0, 1.0, size=n_items)

    g0 = mirt.simdata(
        model="2PL",
        discrimination=a,
        difficulty=b,
        theta=rng.normal(0.0, 1.0, size=250),
        seed=20,
    )
    g1 = mirt.simdata(
        model="2PL",
        discrimination=a,
        difficulty=b + 0.15,
        theta=rng.normal(-0.3, 1.0, size=250),
        seed=21,
    )

    data = np.vstack([g0, g1])
    groups = np.array([0] * g0.shape[0] + [1] * g1.shape[0])
    result = fit_multigroup(
        data,
        groups,
        model="2PL",
        invariance="configural",
        n_quadpts=15,
        max_iter=40,
        verbose=False,
    )
    print(result.summary())


if __name__ == "__main__":
    main()
