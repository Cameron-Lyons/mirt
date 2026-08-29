Bayesian Estimation
===================

``GibbsSampler`` retains posterior draws for item parameters, person abilities,
and the log-likelihood. The fitted model uses the posterior mean of each item
parameter, while ``MCMCResult`` keeps the complete draws for uncertainty analysis.

Fitting a model
---------------

Pass a model instance and response matrix directly to the sampler:

.. code-block:: python

   import mirt

   dataset = mirt.load_dataset("LSAT7")
   responses = dataset["data"]
   model = mirt.TwoParameterLogistic(n_items=responses.shape[1])

   result = mirt.GibbsSampler(
       n_iter=2_000,
       burnin=500,
       thin=2,
       seed=42,
   ).fit(model, responses)

   print(result.summary())

Posterior summaries
-------------------

Use ``posterior_summary`` to compute means, standard deviations, medians, and
equal-tailed credible intervals directly from the stored draws:

.. code-block:: python

   item_summary = result.posterior_summary(
       credible_level=0.90,
       parameters=["discrimination", "difficulty"],
   )

   print(item_summary["difficulty"]["mean"])
   print(item_summary["difficulty"]["ci_lower"])
   print(item_summary["difficulty"]["ci_upper"])

Statistics are computed over the leading draw dimension. All remaining dimensions
are preserved. For example, person-by-factor ability summaries have the same shape
as one draw from the ``theta`` chain:

.. code-block:: python

   ability_summary = result.posterior_summary(parameters="theta")
   ability_means = ability_summary["theta"]["mean"]

When only interval bounds are needed, ``credible_intervals`` avoids computing the
other statistics:

.. code-block:: python

   lower, upper = result.credible_intervals(
       credible_level=0.95,
       parameters="discrimination",
   )["discrimination"]

Omitting ``parameters`` summarizes every stored chain, including ``theta`` and
``log_likelihood``. Selected chains must contain finite values and share a common
number of posterior draws.

Plausible-value regression
--------------------------

Plausible values propagate ability uncertainty into secondary analyses. Generate
them from a fitted item model, then combine a regression across all retained draws:

.. code-block:: python

   plausible_values = mirt.generate_plausible_values(
       fitted_model,
       responses,
       n_plausible=20,
       seed=42,
   )
   analysis = mirt.plausible_value_regression(
       plausible_values,
       outcome,
       X=covariates,
       weights=case_weights,
   )

``coefficients``, ``se``, ``pvalues``, and ``df`` contain the combined regression
inference. The calculation projects the invariant intercept and covariates once,
then processes plausible-value draws in memory-bounded batches. Pass
``batch_size`` to set an explicit number of draws per batch; the default chooses a
bound from the number of people and factors.

Diagnostics
-----------

``rhat`` and ``ess`` contain convergence and effective-sample-size diagnostics for
item parameters. Both are included in ``summary()``. Values of R-hat close to one
and larger effective sample sizes indicate more reliable posterior estimates.

Pointwise predictive criteria
-----------------------------

Use ``compute_pointwise_log_lik`` to prepare posterior log-likelihood matrices for
WAIC or PSIS-LOO. Choose the aggregation level according to the predictive unit:

.. code-block:: python

   from mirt.diagnostics import compute_pointwise_log_lik, psis_loo, waic

   person_log_lik = compute_pointwise_log_lik(
       result.model,
       responses,
       result.chains,
       by="person",
   )
   print(waic(person_log_lik).summary())
   print(psis_loo(person_log_lik).summary())

``by="observation"`` retains the flattened person-item layout and assigns zero to
missing cells. ``by="observed"`` omits missing cells, producing the most compact
matrix when each observed response is the predictive unit. Exact unidimensional
2PL models evaluate fixed or sampled item parameters and abilities through a
batched native implementation when available, with a memory-bounded NumPy fallback.
Other model families retain the general probability-based implementation.

Posterior predictive suites
---------------------------

Use ``posterior_predictive_checks`` when several aspects of model fit should be
checked together. Every requested statistic sees the same replicated datasets,
so comparisons are paired and model probabilities are evaluated only once per
replication:

.. code-block:: python

   from mirt.diagnostics import posterior_predictive_checks

   checks = posterior_predictive_checks(
       result,
       responses,
       result.model,
       ["item_mean", "person_score", "correlation"],
       n_rep=500,
       seed=42,
   )
   for name, check in checks.items():
       print(name, check.p_value)

A mapping can provide descriptive labels and custom scalar statistics. Values may
mix built-in names and callables. Use ``posterior_predictive_check`` when only one
statistic is needed; it returns the same ``PPCResult`` as before.
