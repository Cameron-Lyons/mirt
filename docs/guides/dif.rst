Differential Item Functioning
=============================

DIF detects whether items function differently across groups after matching
on ability. Every method compares the groups on one latent scale, so a
difference in group ability (impact) is not reported as DIF.

.. code-block:: python

   import mirt
   import numpy as np

   result = mirt.dif(data, groups, model="2PL", method="likelihood_ratio")
   print(result)

The likelihood-ratio default fits one baseline multiple-group model plus one
warm-started refit per tested item. Built-in 1PL and 2PL items take batched
Newton M-steps, so 30 binary items with 1,000 persons per group take one to
two seconds. 3PL and polytomous items use an itemwise optimizer and are
several times slower per item. ``n_jobs=-1`` runs the refits in parallel,
and ``method="wald"`` and :func:`mirt.diagnostics.compute_grdif` are fast
screens that need no refits.

Methods
-------

* ``likelihood_ratio`` — nested multiple-group likelihood-ratio test. Each
  studied item is compared constrained versus free across groups, while the
  focal latent mean and variance are estimated. ``df`` is the number of
  parameters the free model adds. This is the most accurate test, and the
  slowest: it needs one baseline fit plus one refit per tested item. Refits
  are warm-started, and ``n_jobs`` runs them in worker processes.
* ``wald`` — Wald test of item-parameter differences. The groups are
  calibrated separately and the focal calibration is linked onto the
  reference scale by Stocking-Lord. Focal estimates and standard errors are
  rescaled with the linking constants. The statistic uses each parameter's
  standard error from the group fits and ignores parameter covariances and
  linking error, so it can be liberal, particularly in small samples and for
  polytomous items. It is fast; prefer ``likelihood_ratio`` for inference.
  Separate 3PL calibrations estimate guessing poorly, so ``wald`` and
  ``raju`` are unreliable for 3PL; ``likelihood_ratio`` holds guessing equal
  across groups.
* ``lord`` — Lord's chi-square, an alias of ``wald``.
* ``raju`` — Raju's signed and unsigned areas between the linked response
  curves over theta in [-4, 4]. Areas are descriptive: ``p_value`` is
  ``NaN`` and the ETS class uses the signed area alone.

``effect_size`` is the focal-minus-reference item location (difficulty, or
the mean threshold) on the common scale, or the signed area for ``raju``.
Positive values mean the item is harder for the focal group. Results also
report ``df``, ``tested`` and ``converged`` for every item.

Anchors and schemes
-------------------

``anchors`` lists items assumed free of DIF, by index or by name (DataFrame
column names, otherwise ``Item_0``, ``Item_1``, ...). They are not tested.
Likelihood ratio tests hold them equal across groups in every model; ``wald``
and ``raju`` link the groups over them. Without anchors, a likelihood-ratio test
uses every other item as an anchor, and ``wald`` and ``raju`` link on all
items. Both assume that any DIF balances across items.

:func:`mirt.multigroup.multigroup_dif` runs the likelihood-ratio tests for
two or more groups and returns one row per studied item with ``chi2``,
``df``, raw and adjusted p-values, ``delta_aic``, ``delta_bic`` and
``flagged``. Its ``scheme`` follows ``mirt::DIF``:

* ``"drop"`` — start from a fully constrained model and free one studied
  item at a time.
* ``"add"`` — start from a model that constrains only ``anchors`` and
  constrain one studied item at a time.
* ``"drop_sequential"`` — repeat the drop step with previously flagged items
  left free until no new item is flagged.
* ``"add_sequential"`` — repeat the add step, adding items without DIF to the
  anchors, until no new invariant item is found.

``parameters`` selects the tested families (``"discrimination"`` and
``"intercepts"``); other parameters, such as 3PL guessing, stay equal across
groups. A family without free coordinates in any studied item, such as
``"discrimination"`` for a 1PL model, raises ``ValueError``.

:func:`mirt.dif` and :func:`mirt.multigroup.multigroup_dif` run the same
tests with the same defaults: ``p_adjust="none"``, as in ``mirt::DIF``, so
the same data give the same flags. ``multigroup_dif`` names the reference
group by index or label (``reference_group``), ``mirt.dif`` names the focal
group by label (``focal_group``). An integer ``reference_group`` that is also
the label of another group is rejected as ambiguous; pass the label as a
string.

.. code-block:: python

   from mirt.multigroup import multigroup_dif, select_dif_anchors

   anchors = select_dif_anchors(data, groups, model="2PL", n_anchors=4)
   table = multigroup_dif(
       data, groups, model="2PL", scheme="add", anchors=anchors, p_adjust="holm"
   )

:func:`mirt.multigroup.select_dif_anchors` implements the iterative
all-other-as-anchor strategy (``method="aoaa_iterative"``; Kopf, Zeileis and
Strobl, 2015) and a single-pass ranking (``method="rank"``). Both rank
candidates by their likelihood-ratio p-values.

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
families independently while preserving missing values. Sequential
likelihood-ratio schemes adjust the items tested in each round.

Test-level impact
-----------------

Differential Test Functioning summarizes the reference-minus-focal expected
score difference in score units. By default, the score curves are averaged
over a standard-normal ability distribution. Use ``weighting="uniform"`` or
provide custom nonnegative grid weights when another target population is
appropriate.

Each group is calibrated on its own standard-normal scale. Pass
``anchor_items`` to link the focal calibration onto the reference scale over
those items, in the observed fit and in every bootstrap replicate. Without
anchors the curves are compared unlinked, so impact is confounded with DTF,
and a warning is issued. Linking on all items is not offered for DTF: the
Stocking-Lord criterion matches the test characteristic curves that DTF
compares, which would drive DTF towards zero by construction. The result
reports ``linking_constants`` and ``anchor_items``.

.. code-block:: python

   dtf = mirt.compute_dtf(
       data,
       groups,
       method="unsigned",
       focal_group="focal",
       weighting="normal",
       anchor_items=[4, 5, 6, 7],
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

:func:`mirt.compute_drf` and :func:`mirt.compute_item_drf` compare
information curves after linking the focal calibration onto the reference
scale, by default over all items (``anchor_items`` overrides this).
Marginal reliabilities remain within-group quantities.

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
* :func:`mirt.diagnostics.compute_grdif` — multi-group GRDIF with robust
  scaling and itemwise multiplicity control; a fast residual screen that does
  not refit models per item
* :func:`mirt.diagnostics.grdif_effect_size` — spread of the GRDIF residual
  moments across groups, from the calibration and final abilities of
  :func:`mirt.diagnostics.compute_grdif` (no refit)
* :func:`mirt.compute_dtf` / :func:`mirt.compute_drf` — test/response functioning
* :doc:`multigroup` — multiple-group models and invariance testing

See ``examples/dif_analysis.py``.
