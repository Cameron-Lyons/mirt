"""Example: DIF analysis on simulated two-group data."""

from __future__ import annotations

import numpy as np

import mirt


def main() -> None:
    rng = np.random.default_rng(42)
    n_persons, n_items = 400, 15
    a = rng.uniform(0.7, 1.8, size=n_items)
    b = rng.normal(0.0, 1.0, size=n_items)
    b_focal = b.copy()
    b_focal[3] += 0.8

    theta_ref = rng.normal(0.0, 1.0, size=n_persons // 2)
    theta_foc = rng.normal(-0.2, 1.0, size=n_persons // 2)

    ref = mirt.simdata(
        model="2PL",
        discrimination=a,
        difficulty=b,
        theta=theta_ref,
        seed=1,
    )
    foc = mirt.simdata(
        model="2PL",
        discrimination=a,
        difficulty=b_focal,
        theta=theta_foc,
        seed=2,
    )
    data = np.vstack([ref, foc])
    group = np.array([0] * ref.shape[0] + [1] * foc.shape[0])

    dif_result = mirt.dif(data, group, model="2PL", method="likelihood_ratio")
    print(dif_result)


if __name__ == "__main__":
    main()
