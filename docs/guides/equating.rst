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

Prefer :func:`mirt.equating.link` for unidimensional dichotomous linking with
Stocking–Lord, Haebara, mean/sigma, mean/mean, and additional regression methods.
Curve matching includes the lower and upper asymptotes of 3PL and 4PL items.
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
constants map each model onto the selected reference.

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
