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

SIBTEST inference
-----------------

:func:`mirt.sibtest` compares complete binary responses within matching-score
strata. Its uniform effect ``beta`` is a pooled-population weighted average of
reference-minus-focal suspect scores. ``beta_se`` uses the within-group
sampling variances in each stratum. ``correction=True`` applies the
Shealy–Stout true-score regression correction using each group's KR-20
matching-subtest reliability and neighboring conditional suspect-score means.

.. code-block:: python

   result = mirt.sibtest(
       data,
       groups,
       suspect_items=[2],
       matching_items=[0, 1, 3, 4, 5, 6, 7],
       focal_group="focal",
       min_cell_size=5,
       method="crossing",
   )
   print(result["beta"], result["chi2"], result["df"], result["p_value"])

The ``"crossing"`` method estimates a crossing location by weighted regression
of conditional score differences. Following `Chalmers (2018)
<https://doi.org/10.1007/s11336-017-9583-8>`_, ``chi2`` sums the squared
standardized effects from the two regions. ``df`` is two when both regions
are estimable and one when a single region is estimable. The reported crossing
effect is an absolute score-area difference; its ``z`` is descriptive, and
``p_value`` comes from the chi-square test. ``crossing_point`` and ``n_strata``
report the estimated location and number of retained score strata.

``min_cell_size`` requires at least that many persons in each group at a
matching score; its default is two, the minimum needed to estimate sampling
variance. Larger minima exclude sparse cells. Corrected analysis also needs
at least two matching items, positive matching reliability in both groups,
and observed neighboring scores for the regression slopes. Missing responses
are rejected. Unestimable tests return ``NaN`` p-values and zero degrees of
freedom, so they are never flagged as significant.

:func:`mirt.sibtest_items` shares matching-score totals and reliability moments
across items and returns the same inference fields as arrays. Its
``p_adjust`` option supports ``"none"``, ``"bonferroni"``, ``"holm"``, and
``"fdr_bh"``. Both APIs accept ``focal_group``; reversing it reverses the
uniform effect while preserving its two-sided p-value. The original correction
and standard-error formulation is described by `Shealy and Stout (1993)
<https://doi.org/10.1007/BF02294572>`_.

Related utilities
-----------------

* :func:`mirt.sibtest` — SIBTEST
* :func:`mirt.compute_grdif` — multi-group GRDIF with robust scaling and
  itemwise multiplicity control
* :func:`mirt.compute_dtf` / :func:`mirt.compute_drf` — test/response functioning

See ``examples/dif_analysis.py``.
