"""End-to-end example: fit LSAT7 → score → itemfit → CAT simulation."""

from __future__ import annotations

import mirt
from mirt.cat import CATEngine


def main() -> None:
    dataset = mirt.load_dataset("LSAT7")
    responses = dataset["data"]

    result = mirt.fit_mirt(responses, model="2PL", verbose=False)
    print(result.summary())

    scores = mirt.fscores(result, responses, method="EAP")
    print("EAP mean:", float(scores.theta.mean()))
    print("EAP SE mean:", float(scores.standard_error.mean()))

    item_fit = mirt.itemfit(result, responses)
    print(item_fit)

    engine = CATEngine(
        result.model,
        item_selection="MFI",
        stopping_rule="SE",
        se_threshold=0.35,
        max_items=10,
    )
    sim = engine.run_simulation(true_theta=0.0)
    print("CAT items administered:", sim.n_items_administered)
    print("Final theta / SE:", sim.theta, sim.standard_error)


if __name__ == "__main__":
    main()
