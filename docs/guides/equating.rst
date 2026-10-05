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
   theta_on_old_scale = mirt.utils.transform_theta(theta_new, result)

The deprecated ``equate`` delegates to :func:`mirt.equating.link` and reports
the inverse constants, ``theta_new = A * theta_old + B``; ``transform_theta``
maps new-form scores onto the old scale as ``(theta - B) / A``.

Preferred API
-------------

Prefer :func:`mirt.equating.link` for unidimensional dichotomous linking with
Stocking–Lord, Haebara, mean/sigma, mean/mean, and additional regression methods.
Curve matching includes the lower and upper asymptotes of 3PL and 4PL items and
the 5PL asymmetry. Other dichotomous families, such as CLL, NLL, ULL, and
zero-inflated models, are matched through their own response functions.
Use :func:`mirt.equating.link_grm`, :func:`mirt.equating.link_gpcm`, and
:func:`mirt.equating.link_nrm` for polytomous models, or
:func:`mirt.equating.link_mirt` for multidimensional models.

Population-weighted linking
---------------------------

``link`` accepts nonnegative ``weights`` on its reference-scale theta grid.
These weights define the population for curve matching and fit statistics.
Bootstrap uncertainty and anchor purification use the same weights and linking
method as the point estimate. ``robust=True`` also applies to moment-linking
iterations during purification.

.. code-block:: python

   import numpy as np
   from mirt.equating import link

   theta = np.linspace(-4, 4, 61)
   weights = np.exp(-0.5 * ((theta - 0.8) / 1.2) ** 2)
   result = link(
       model_old, model_new, anchor_items_old, anchor_items_new,
       weights=weights, compute_se=True, n_bootstrap=200,
       purify_anchors=True, random_state=42,
   )

Multiple administrations and grade levels
-----------------------------------------

:func:`mirt.equating.chain_link` estimates adjacent-form links and composes
them into transformations from each calibration onto ``reference_index``.
Paired anchor lists may use different item positions on each form.
``pairwise_results[t]`` maps model ``t + 1`` onto model ``t``; the cumulative
constants map each model onto the selected reference. Adjacent GRM, GPCM/PCM,
or NRM forms are linked with the matching polytomous linker, and
``transform_to_reference`` transforms GRM, GPCM, and NRM category parameters
(PCM discriminations are fixed at 1, so PCM models cannot be rescaled). NRM
anchors report robust drift statistics but no difficulty changes, because their
category-intercept contrasts are not on the theta metric.

.. code-block:: python

   from mirt.equating import chain_link, transform_to_reference

   result = chain_link(
       [model_t0, model_t1, model_t2],
       [([0, 1, 2], [4, 5, 6]), ([7, 8, 9], [0, 1, 2])],
       reference_index=1,
   )
   model_t2_on_reference = transform_to_reference(model_t2, result, time_index=2)

The standalone :func:`mirt.equating.accumulate_constants` instead accepts
forward coordinate maps ``theta[t + 1] = A[t] * theta[t] + B[t]``.
Invert pairwise ``link`` constants before passing them to that helper.

:func:`mirt.equating.concurrent_link` fits all form transformations jointly.
Its connected anchor design can include adjacent and nonadjacent forms.
``reference_index`` fixes the chosen calibration at ``(A, B) = (1, 0)`` and
defines the population-weighting grid used during optimization. Stocking–Lord
and TCC match summed expected-score curves; Haebara matches paired item curves
and the full category probabilities of polytomous items. An exhausted or failed
optimization raises an error instead of returning unconverged constants.

``vertical_scale(..., method="concurrent")`` performs the same simultaneous
curve matching for an adjacent-grade design, using ``reference_grade`` as the
fixed calibration. Its adjacent-pair diagnostics describe the transformations
implied by the final joint solution. ``linking_method`` supports
``"stocking_lord"``, ``"tcc"``, and ``"haebara"`` for concurrent scaling.
Set ``enforce_monotonicity=False`` to inspect the calibrated grade means;
enabling it shifts grade locations to enforce increasing mean ability.

Joint vertical anchor calibration
---------------------------------

``vertical_scale`` also calibrates responses jointly with
``method="fixed_anchor"`` or ``method="floating_anchor"``. Explicit adjacent
anchor mappings identify the same physical item in different forms. Each grade
may administer a different number of items in a different order; unadministered
physical items are treated as missing responses in the joint item bank. Every
adjacent pair needs at least two anchors with responses in both grades.

Both methods estimate grade-specific Gaussian ability distributions, fixing the
reference grade at mean zero and variance one. Floating-anchor calibration
estimates all physical item parameters while requiring shared items to have
identical parameters across grades. Fixed-anchor calibration freezes shared
items administered in the reference grade at that model's values. Anchors that
bridge later grades are estimated jointly on the identified scale. It estimates
unique items and nonreference grade distributions from the response likelihood.

.. code-block:: python

   from mirt.equating import GradeData, vertical_scale

   grades = [
       GradeData("G3", responses_g3, anchor_items_above=[4, 5, 6]),
       GradeData("G4", responses_g4, anchor_items_below=[0, 1, 2]),
   ]
   result = vertical_scale(
       grades,
       models=[model_g3, model_g4],
       method="fixed_anchor",
       reference_grade=0,
       enforce_monotonicity=True,
       n_quadpts=31,
       max_iter=500,
       tol=1e-4,
   )
   model_g4_on_scale = result.calibrated_models["G4"]
   theta_g4 = result.scores["G4"].theta
   population_g4 = result.latent_distributions["G4"]
   global_item_columns_g4 = result.item_maps["G4"]

Supplied fitted models provide starting values; fixed-anchor calibration also
uses the reference model's anchor values. The models must share one built-in
unidimensional family: 1PL, 2PL, 3PL, 4PL, GRM, GPCM, PCM, or NRM. Shared
polytomous items must have matching category counts. Without supplied models,
both methods fit a 2PL bank; fixed-anchor calibration first fits the reference
grade to obtain its anchor values. ``linking_method`` does not determine the
response calibration.

``grade_means`` and ``grade_sds`` describe the fitted population distributions.
``scores`` contains EAP estimates and posterior standard errors using those
distributions as priors. ``calibrated_models`` retains each grade's original
item order. ``calibration_result`` provides the full joint model, likelihood,
effective parameter count, and convergence diagnostics. Recalibration changes
item curves, so ``grade_transformations`` is empty; use the returned models and
scores for subsequent analysis.

``free_parameter_masks[grade]`` supplies the effective local item masks for
the joint calibration, including fixed-anchor restrictions and category
padding. The returned built-in models carry these same masks, so diagnostics
that inspect the model's free parameters account for fixed anchors. Copies
preserve the restrictions. Call ``model.set_free_parameter_masks(None)`` to
restore its family masks before an unrestricted subsequent fit. Parameters
shared between grades are counted once in the joint result; local diagnostic
masks describe the items administered in that grade.

With ``enforce_monotonicity=True``, population means are constrained to be
nondecreasing during Gaussian density estimation. The reference distribution
remains fixed, and the returned models and scores use the fitted constrained
distributions. Set it to ``False`` to estimate unrestricted grade means;
``monotonicity_violations`` then records decreasing adjacent fitted means.
Calibration raises an error if the reference fit or joint EM fit does not
converge.

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

True-score equating solves the old form's test characteristic curve for the
ability at each score by bracketed root finding, then evaluates the new form
at that ability. ``theta_range`` sets the reporting grid in ``result.theta``
and seeds the search; abilities outside it are still found. Scores at or
below the sum of the old form's lower asymptotes (for example the 3PL
guessing parameters) have no true-score ability. Following Kolen and Brennan
(2014, sec. 6.5), they map linearly from ``(0, 0)`` to the two forms'
lower-asymptote sums. Scores at or above the old upper-asymptote sum map
linearly to the maximum scores, so a perfect score on the old form equates to a
perfect score on the new form. The asymptote sums are evaluated numerically
from each test characteristic curve far outside ``theta_range``, so custom
models must return valid probabilities at extreme abilities.

Observed-score equating integrates conditional score probabilities with
Lord–Wingersky recursion, then applies equipercentile equating as defined by
Kolen and Brennan (2014, eqs. 2.14–2.18). Each integer score is spread
uniformly over ``[x - 0.5, x + 0.5]``. An old score's percentile rank is
inverted within the new score interval that contains it, skipping new scores
with zero probability. Equated scores therefore range over
``[-0.5, K_Y + 0.5]``, where ``K_Y`` is the new maximum score.
:func:`mirt.equating.equipercentile_equating` exposes the same calculation for
score distributions.
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
for Gaussian smoothing before percentile matching. The ``"kernel"`` option
smooths the discrete distributions; it is not kernel equating, which is
described below.

The ``score-equating`` benchmark suite records timing and peak Python/NumPy
allocations for 2PL, 3PL, GRM, and GPCM recursion. For this suite, ``--persons``
sets the number of integration-grid points:

.. code-block:: bash

   uv run --no-sync python benchmarks/run_benchmarks.py \
       --suite score-equating --backend numpy --persons 3000 --items 150 \
       --repeats 5 --warmups 1 --json score-equating.json

Kernel equating
---------------

:func:`mirt.equating.kernel_equating` implements the Gaussian kernel method of
von Davier, Holland, and Thayer (2004) for an equivalent-groups design. Each
score distribution is continuized with a Gaussian kernel that preserves its
mean and variance, and each old score ``x`` maps to ``G_h^{-1}(F_h(x))``.
By default each bandwidth minimizes the squared difference between the score
probabilities and the continuized density at the score points (``PEN1``).
A positive ``kappa`` adds ``kappa`` times the second penalty, ``PEN2``, which
counts score points around which the density is U-shaped. The default
``kappa=0`` matches the default of R's kequate. ``bandwidth="linear"`` uses
1000 times each score standard deviation, which reproduces linear equating. A
number or an ``(old, new)`` pair fixes the bandwidths.

.. code-block:: python

   from mirt.equating import kernel_equating

   result = kernel_equating(
       frequencies_old, frequencies_new,
       presmoothing=4,
       n_old=int(frequencies_old.sum()), n_new=int(frequencies_new.sum()),
   )
   result.new_scores, result.standard_errors
   result.bandwidth_old, result.bandwidth_new

``presmoothing`` fits a polynomial log-linear model of the given degree by
maximum likelihood before continuization. It preserves that many moments of
each observed distribution and must be less than the number of observed
scores. Pass an ``(old, new)`` pair for different degrees, using None to leave
one form unsmoothed.
With sample sizes, ``standard_errors`` holds delta-method standard errors of
equating from the multinomial or log-linear sampling covariance, with the
bandwidths held fixed.

:func:`mirt.equating.irt_kernel_equating` is the IRT observed-score variant
(Andersson and Wiberg, 2017). It continuizes the model-implied score
distributions from Lord–Wingersky recursion, using the same population,
linking, and item-subset arguments as ``observed_score_equating``. It does not
report standard errors because they depend on the item parameter covariance.

.. code-block:: python

   from mirt.equating import irt_kernel_equating

   result = irt_kernel_equating(model_old, model_new, linking_result=linking)

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

Fixed-item parameter calibration (FIPC) places new items on the scale of
previously calibrated anchor items without a separate linking step.
:func:`mirt.fixed_item_calibration` holds the anchor parameters at their known
values while EM estimates the new items together with the mean and covariance
of the calibration population. It supports every dichotomous and polytomous
item family, and responses to new items given to only part of the sample may be
coded as missing:

.. code-block:: python

   import mirt

   # Columns 0-4 hold anchor items whose parameters are stored in anchor_model.
   calibration = mirt.fixed_item_calibration(
       responses,
       mirt.GradedResponseModel(responses.shape[1], n_categories=4),
       anchor_items=[0, 1, 2, 3, 4],
       anchor_parameters=anchor_model,
   )
   print(calibration.latent_mean, calibration.latent_cov)
   new_items = calibration.new_item_parameters
   scores = mirt.fscores(
       calibration.model,
       responses,
       prior_mean=calibration.latent_mean,
       prior_cov=calibration.latent_cov,
   )

``anchor_parameters`` may be a model or fit result containing exactly the
anchor items, in ``anchor_items`` order, or a mapping from parameter names to
anchor rows. Anchor parameters report zero standard errors, and the new items'
standard errors are conditional on the estimated population. Parameters shared
by all items, such as rating-scale thresholds, belong to the anchored scale and
stay fixed. The legacy :func:`mirt.fixed_calib` fits only new 2PL items under a
standard normal population. :doc:`mixed_format` shows how to calibrate
mixed-format tests against fixed anchors.

See ``examples/equating.py``.
