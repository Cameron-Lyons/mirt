"""Example: fixed-item calibration and Stocking-Lord equating."""

from __future__ import annotations

import numpy as np

import mirt


def main() -> None:
    rng = np.random.default_rng(7)
    n_items = 20
    a = rng.uniform(0.8, 1.6, size=n_items)
    b = rng.normal(0.0, 1.0, size=n_items)

    form_x = mirt.simdata(
        model="2PL", discrimination=a, difficulty=b, n_persons=500, seed=10
    )
    a_y = a.copy()
    b_y = b.copy()
    b_y[8:] += 0.25
    form_y = mirt.simdata(
        model="2PL", discrimination=a_y, difficulty=b_y, n_persons=500, seed=11
    )

    fit_x = mirt.fit_mirt(form_x, model="2PL")
    fit_y = mirt.fit_mirt(form_y, model="2PL")

    anchors = list(range(8))
    equate_result = mirt.equate(
        fit_x.model,
        fit_y.model,
        anchors,
        anchors,
        method="stocking_lord",
    )
    print(equate_result)


if __name__ == "__main__":
    main()
