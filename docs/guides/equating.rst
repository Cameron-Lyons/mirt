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
