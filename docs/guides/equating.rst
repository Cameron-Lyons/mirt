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

Score conversion between linked forms
-------------------------------------

Linking constants place new-form abilities on the reference scale:
``theta_old = A * theta_new + B``. Pass the linking result to either score
equating method to evaluate both forms for the same abilities. Both methods
support dichotomous and ordered polytomous scores, including item subsets.

.. code-block:: python

   from mirt.equating import link, observed_score_equating, true_score_equating

   linking = link(
       model_old, model_new, anchor_items_old, anchor_items_new,
       method="stocking_lord",
   )
   true_scores = true_score_equating(
       model_old, model_new, linking_result=linking,
   )
   observed_scores = observed_score_equating(
       model_old, model_new, linking_result=linking, smoothing="kernel",
   )

True-score equating inverts the old form's expected score curve over
``theta_range`` and evaluates the new form at the corresponding ability.
Scores outside the curve's finite lookup range use the nearest theta endpoint;
widen ``theta_range`` when more extreme ability values are needed.

Observed-score equating integrates conditional score probabilities with
Lord–Wingersky recursion, then matches score-distribution percentile midpoints.
The default population uses normal-density weights on the reference grid.
For a custom population, supply ``theta_grid`` and ``theta_distribution`` on
the old/reference scale. Distribution entries are probability masses at the
grid points, are normalized automatically, and stay paired with their points
when linking the new form. Custom grids can be unevenly spaced; supply
appropriate integration weights when approximating a continuous distribution.

Use ``smoothing="none"`` for the original score distributions,
``"loglinear"`` for a polynomial fit to log probabilities, or ``"kernel"``
for Gaussian smoothing before percentile matching.

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
