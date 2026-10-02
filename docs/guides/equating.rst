Test Equating and Linking
=========================

Link separately calibrated forms onto a common scale using anchor items.

Legacy helper
-------------

.. code-block:: python

   import mirt

   result = mirt.equate(
       model_old,
       model_new,
       anchor_items_old,
       anchor_items_new,
       method="stocking_lord",
   )

Preferred API
-------------

Prefer :func:`mirt.equating.link` for additional linking methods, diagnostics,
and polytomous support (Stocking–Lord, Haebara, mean/sigma, mean/mean, and more).

Score conversions across calibrations
-------------------------------------

Linking constants map abilities from the new calibration onto the reference
scale: ``theta_old = A * theta_new + B``. Both true-score and observed-score
equating accept this result and evaluate the new form at
``(theta_old - B) / A``.

.. code-block:: python

   import numpy as np
   from mirt.equating import link, observed_score_equating, true_score_equating

   linking = link(model_old, model_new, anchor_items_old, anchor_items_new)
   true_scores = true_score_equating(
       model_old, model_new, linking_result=linking,
   )

   theta = np.linspace(-4, 4, 101)
   population_mass = np.exp(-0.5 * theta**2)
   observed_scores = observed_score_equating(
       model_old, model_new,
       linking_result=linking,
       theta_grid=theta,
       theta_distribution=population_mass,
       batch_size=32,
       smoothing="kernel",
   )

True-score equating inverts the old form's expected score curve over
``theta_range`` and evaluates the new form at the corresponding ability.
Scores outside the curve's finite lookup range use the nearest theta endpoint;
widen ``theta_range`` when more extreme ability values are needed.

Observed-score equating integrates conditional score probabilities with
Lord–Wingersky recursion, then matches score-distribution percentile midpoints.
The default population uses normal-density weights on the reference grid.
For observed equating, ``theta_distribution`` supplies probability masses on
the reference-scale grid. These same masses weight both forms; transforming
the grid does not change them. Custom grids can be unevenly spaced; supply
appropriate integration weights when approximating a continuous distribution.
Dichotomous asymptotes, item-specific ordinal category counts, and item subsets
are supported. Score recursion evaluates bounded theta batches
and retains only the marginal score distribution across batches. Omit
``batch_size`` to choose a memory bound from the form size. Custom probability
curves use the Python path even when Rust is installed.

Use ``smoothing="none"`` for the original score distributions,
``"loglinear"`` for a polynomial fit to log probabilities, or ``"kernel"``
for Gaussian smoothing before percentile matching.

The ``score-equating`` benchmark suite records timing and peak Python/NumPy
allocations for 2PL, 3PL, GRM, and GPCM recursion. For this suite, ``--persons``
sets the number of integration-grid points:

.. code-block:: bash

   uv run --no-sync python benchmarks/run_benchmarks.py \
       --suite score-equating --backend numpy --persons 3000 --items 150 \
       --repeats 5 --warmups 1 --json score-equating.json

Parallel linking uncertainty
----------------------------

Curve-based and response-refit bootstrap replicates can run concurrently with
``n_jobs``. Seeded results are identical across worker counts, so a sequential
analysis can be scaled up without changing its samples.

.. code-block:: python

   from mirt.equating import bootstrap_linking_se

   se_a, se_b, a_samples, b_samples = bootstrap_linking_se(
       model_old,
       model_new,
       responses_old=None,
       responses_new=None,
       anchors_old=anchor_items_old,
       anchors_new=anchor_items_new,
       method="stocking_lord",
       n_bootstrap=500,
       seed=42,
       n_jobs=4,
   )

Closed-form anchor bootstraps are already evaluated as vectorized batches and
do not need multiple workers.

Fixed-item calibration
----------------------

:func:`mirt.fixed_calib` calibrates new items onto an existing scale defined by
anchors.

See ``examples/equating.py``.
