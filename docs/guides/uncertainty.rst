Uncertainty and Missing Responses
=================================

Parameter draws can propagate item-calibration uncertainty into score curves.
The workflow uses NumPy arrays throughout and does not require a dataframe or
plotting package.

Draw and summarize parameters
-----------------------------

``draw_parameters`` samples discrimination and difficulty jointly from a
multivariate-normal approximation. It also carries the asymptote and asymmetry
parameters used by supported logistic models.

.. code-block:: python

   import mirt

   result = mirt.fit_mirt(responses, model="2PL")
   samples = mirt.draw_parameters(result.model, n_samples=2_000, seed=42)
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

Pass a full parameter covariance matrix through ``vcov`` when one is available.
Without it, the sampler uses a documented diagonal approximation.

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
