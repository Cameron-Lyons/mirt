Differential Item Functioning
=============================

DIF detects whether items function differently across groups after matching
on ability.

.. code-block:: python

   import mirt
   import numpy as np

   result = mirt.dif(data, groups, model="2PL", method="likelihood_ratio")
   print(result)

Methods
-------

* ``likelihood_ratio`` — nested model LR test
* ``wald`` — Wald DIF
* ``lord`` — Lord's chi-square
* ``raju`` — Raju area measures

Multiple-testing control
------------------------

Item-wide DIF analyses can control family-wise error or false discoveries
without an optional statistics package. The raw and adjusted p-values are both
returned, and ETS classifications use the adjusted values.

.. code-block:: python

   result = mirt.dif(
       data,
       groups,
       model="2PL",
       method="wald",
       p_adjust="holm",
   )
   print(result[["p_value", "p_value_adjusted", "classification"]])

Available methods are ``"none"``, ``"bonferroni"``, ``"holm"``, and
``"fdr_bh"``. For custom diagnostic arrays, use
:func:`mirt.diagnostics.adjust_p_values`; its ``axis`` argument adjusts many
families independently while preserving missing values.

Test-level impact
-----------------

Differential Test Functioning summarizes the reference-minus-focal expected
score difference in score units. By default, the score curves are averaged
over a standard-normal ability distribution. Use ``weighting="uniform"`` or
provide custom nonnegative grid weights when another target population is
appropriate.

.. code-block:: python

   dtf = mirt.compute_dtf(
       data,
       groups,
       method="unsigned",
       focal_group="focal",
       weighting="normal",
       n_bootstrap=200,
       random_state=42,
       n_jobs=4,
   )
   print(dtf["DTF"], dtf["confidence_interval"])

Set ``n_bootstrap=0`` when only the descriptive score curves and effect size
are needed. The result reports successful and failed bootstrap replicate
counts so uncertainty estimates can be audited. For larger studies,
``n_jobs`` runs independent bootstrap refits in worker processes while
preserving seeded results; the serial default is ``n_jobs=1``, and ``-1``
uses all available CPU cores. The same option is available on
:func:`mirt.reliability_invariance`.

Related utilities
-----------------

* :func:`mirt.sibtest` — SIBTEST
* :func:`mirt.compute_grdif` — multi-group GRDIF with robust scaling and
  itemwise multiplicity control
* :func:`mirt.compute_dtf` / :func:`mirt.compute_drf` — test/response functioning

See ``examples/dif_analysis.py``.
