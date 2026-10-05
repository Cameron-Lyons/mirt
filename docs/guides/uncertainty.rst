Uncertainty and Missing Responses
=================================

Parameter draws can propagate item-calibration uncertainty into score curves.
The workflow uses NumPy arrays throughout and does not require a dataframe or
plotting package.

Item parameter standard errors
------------------------------

EM fits of unidimensional 1PL-4PL, GRM, GPCM and PCM models report
observed-information standard errors by default. The information is the
negative Hessian of the marginal log-likelihood, computed exactly from Louis's
(1982) missing-information identity, which is also the target of the Oakes
(1999) method. The fit also stores the covariance of the free parameters:

.. code-block:: python

   result = mirt.fit_mirt(responses, model="2PL")

   result.se_method      # "oakes"
   result.vcov           # (n_free, n_free) covariance
   result.vcov_labels    # ["discrimination[Item_1]", ..., "difficulty[Item_1]", ...]

Select another estimator with ``se_method`` on ``fit_mirt`` or
``EMEstimator``:

* ``"oakes"``: inverse observed information.
* ``"crossprod"``: inverse cross-product of the marginal person scores.
* ``"sandwich"``: observed information around the score cross-product, robust
  to misspecification of the item response functions.
* ``"complete_data"``: itemwise curvature of the expected complete-data
  likelihood. It omits the information lost to the unobserved latent trait
  and covariances between parameters, so it understates uncertainty, often by
  20-60%. It does not produce a covariance matrix.

The default ``"auto"`` uses ``"oakes"`` where the exact computation applies
and ``"complete_data"`` for nominal, multidimensional and custom item models.
For EM and Bock-Lieberman fits, ``result.se_method`` names the estimator
used, so check it before relying on the errors; it is ``None`` for estimators
that do not record one. An explicit ``"oakes"`` also works for those models:
built-in nominal and multidimensional models difference each item's curve,
and custom models difference the marginal likelihood, which takes O(P^2)
likelihood evaluations for P parameters. Exploratory multidimensional
solutions are rotationally unidentified, so their information matrix is
singular.

The matrix estimators treat the latent density as fixed, which slightly
understates uncertainty when its mean, variance or shape is estimated.
Parameters on an optimizer bound, such as a guessing parameter of 0, are held
fixed: their standard errors and covariance rows are ``NaN`` and the other
errors are conditional on them. The cost grows as N Q P^2 for N distinct
response patterns, Q quadrature nodes and P parameters; dichotomous models
use a factorization that reduces it to N Q J^2 for J items.
``BLEstimator`` inverts the full Hessian of its marginal likelihood, exact for
built-in item models and numerical otherwise, and labels its results
``"hessian"``.

``result.vcov`` serializes with :meth:`FitResult.to_dict`, and Wald tests read
it directly. Parameter indices follow ``result.model.parameters`` with every
stored entry, including fixed ones; hypotheses on fixed parameters or on
parameters at a bound are rejected:

.. code-block:: python

   # H0: the first item's discrimination equals 1.
   test = mirt.wald(result, param_indices=[0], constraint_values=[1.0])

Draw and summarize parameters
-----------------------------

``draw_parameters`` samples item parameters jointly from the asymptotic
multivariate-normal approximation defined by ``result.vcov``. Guessing, upper
and asymmetry parameters of supported logistic models are drawn from the same
covariance; fixed parameters and parameters on a bound stay at their
estimates.

.. code-block:: python

   import mirt

   result = mirt.fit_mirt(responses, model="2PL")
   samples = mirt.draw_parameters(result, n_samples=2_000, seed=42)
   summary = mirt.posterior_summary(samples, credible_level=0.95)

   difficulty_mean = summary["difficulty"]["mean"]
   difficulty_lower = summary["difficulty"]["ci_lower"]
   difficulty_upper = summary["difficulty"]["ci_upper"]

Equal-tail intervals remain the default. For skewed or multimodal parameter
draws, request the narrowest empirical interval containing the target mass:

.. code-block:: python

   density_summary = mirt.posterior_summary(
       samples,
       credible_level=0.95,
       interval_method="highest_density",
   )

   difficulty_hdi_lower = density_summary["difficulty"]["ci_lower"]
   difficulty_hdi_upper = density_summary["difficulty"]["ci_upper"]

A result without ``vcov``, for example an MCMC fit or one using
``se_method="complete_data"``, is sampled from its standard errors with a
warning that cross-parameter covariance is ignored. For a bare model, pass a
covariance through ``vcov``. It lists discrimination (item-major), then
difficulty, and optionally guessing, slipping, upper and asymmetry. Without a
covariance, bare models fall back to fixed sampling scales unrelated to the
data. This fallback emits a ``FutureWarning`` and will be removed.

Propagate uncertainty to scores
-------------------------------

``sample_expected_scores`` evaluates every parameter draw at one or more ability
values. With no item selection, it returns an expected total-test score for each
draw and ability value.

.. code-block:: python

   import numpy as np

   theta = np.linspace(-3.0, 3.0, 61)
   total_draws = mirt.sample_expected_scores(result.model, theta, samples)

   total_mean = total_draws.mean(axis=0)
   total_interval = np.quantile(total_draws, [0.025, 0.975], axis=0)

Use ``item_idx`` for one item or ``item_indices`` for a subtest. Subtest scores
are summed within each draw, so the output retains shape
``(n_samples, n_theta)``.

.. code-block:: python

   item_draws = mirt.sample_expected_scores(
       result.model,
       theta,
       samples,
       item_idx=3,
   )

   subtest_draws = mirt.sample_expected_scores(
       result.model,
       theta,
       samples,
       item_indices=[0, 3, 7, 11],
   )

The subset path evaluates all selected items together and avoids allocating
probability or asymptote arrays for the rest of the bank. Item indices must be
unique and in range. ``item_idx`` and ``item_indices`` cannot be supplied
together.

Memory control
--------------

Parameter draws are processed in memory-aware chunks by default. Set
``chunk_size`` to cap the number of draws evaluated at once when a deployment
has a specific memory budget. Chunking changes only temporary storage, not the
returned score draws.

.. code-block:: python

   subtest_draws = mirt.sample_expected_scores(
       result.model,
       theta,
       samples,
       item_indices=[0, 3, 7, 11],
       chunk_size=250,
   )

Ability uncertainty and missing responses
-----------------------------------------

``generate_plausible_values`` draws from each respondent's joint ability
posterior. With ``method="posterior"``, draws are quadrature nodes selected
according to their posterior mass. Increase ``n_quadpts`` for a finer grid;
the sampler preserves factor dependence and introduces no additional noise.
Normal scoring priors must have finite means and symmetric, positive-definite
covariance matrices.

.. code-block:: python

   plausible = mirt.generate_plausible_values(
       result,
       responses,
       n_plausible=10,
       n_quadpts=49,
       seed=42,
   )

Plausible values reproduce population moments only when their prior matches
the population. On short tests, draws under the default standard normal prior
shrink toward it, so pass the population mean and covariance on the model's
scale. A ``(n_persons, n_factors)`` mean conditions each person's draws, for
example on latent-regression predictions:

.. code-block:: python

   plausible = mirt.generate_plausible_values(
       result,
       responses,
       n_plausible=10,
       prior_mean=population_mean,
       prior_cov=population_cov,
       seed=42,
   )

For multiple imputation, pass a fitted model or fit result to reuse the item
calibration. Missing responses are sampled conditional on the observed cells,
using joint posterior ability draws before drawing response categories. This
also works for multidimensional and heterogeneous ordinal calibrations.

.. code-block:: python

   completed = mirt.impute_responses(
       responses,
       method="multiple",
       model=result,
       n_imputations=10,
       n_quadpts=21,
       seed=42,
   )

Observed cells remain unchanged. These imputations include ability and response
uncertainty conditional on fixed item parameters; calibration uncertainty is
not sampled. When a model name is supplied, its calibration uses the original
observed responses. Failed named-model calibration warns before falling back
to empirical item draws.

Bootstrap confidence intervals
------------------------------

``bootstrap_ci(..., method="BCa")`` uses bias ranks that account for ties and
a complete leave-one-person-out jackknife for acceleration. It requires one
additional model fit per person. Set ``n_jobs`` to distribute those fits across
processes. Jackknife samples are generated lazily and their moments accumulated
without retaining all respondent-level score vectors. A failed jackknife
produces missing intervals and a warning rather than an approximate BCa result.

Bootstrap likelihood-ratio tests
--------------------------------

The chi-square reference of a likelihood-ratio test fails when the reduced
model lies on the boundary of the full model, as for 2PL versus 3PL or ``k``
versus ``k + 1`` factors. ``bootstrap_lr`` simulates the statistic's null
distribution from the fitted reduced model instead. Both models are refitted
to the observed responses and to every replicate with identical estimator
settings, and replicates keep the observed missing-data pattern.

.. code-block:: python

   reduced = mirt.fit_mirt(responses, model="2PL")
   full = mirt.fit_mirt(responses, model="3PL")
   test = mirt.bootstrap_lr(reduced, full, responses, n_bootstrap=200, seed=42)
   test.statistic, test.p_value, test.asymptotic_p_value
