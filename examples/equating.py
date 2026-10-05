"""Example: Stocking-Lord linking, kernel equating, and fixed-item calibration."""

from __future__ import annotations

import numpy as np

import mirt
from mirt.equating import irt_kernel_equating, link


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

    # Link form Y onto the form X scale: theta_x = A * theta_y + B.
    anchors = list(range(8))
    linking = link(fit_x.model, fit_y.model, anchors, anchors, method="stocking_lord")
    print(linking.constants)

    # IRT observed-score kernel equating: form Y equivalents of form X scores.
    equated = irt_kernel_equating(fit_x.model, fit_y.model, linking_result=linking)
    print("Form Y equivalents:", np.round(equated.new_scores, 2))

    # Fixed-item calibration: hold the anchors at their form X estimates and
    # estimate the other form Y items and the form Y population on that scale.
    params = fit_x.model.parameters
    calibration = mirt.fixed_item_calibration(
        form_y,
        mirt.TwoParameterLogistic(n_items),
        anchors,
        {name: params[name][anchors] for name in ("discrimination", "difficulty")},
    )
    print("Form Y population:", calibration.latent_mean, calibration.latent_cov)


if __name__ == "__main__":
    main()
