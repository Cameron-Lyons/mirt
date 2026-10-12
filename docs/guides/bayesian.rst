Bayesian Estimation
===================

``GibbsSampler`` retains posterior draws for item parameters, person abilities,
and the log-likelihood. The fitted model uses the posterior mean of each item
parameter, while ``MCMCResult`` keeps the complete draws for uncertainty analysis.

Each iteration updates every person's abilities by a random-walk Metropolis
step against their standard normal prior, and then every item's free
parameters by its own Metropolis step: given the abilities the items are
independent, so each item is accepted or rejected on its own likelihood.
Bounded parameters move on an unconstrained scale, with the Jacobian in the
acceptance ratio: slopes as the log of their distance to the lower bound,
guessing and upper asymptotes as logits within their boxes, and graded
thresholds as the first threshold and the logs of the gaps between consecutive
ones, so every draw is valid and ordered. Proposal scales and covariances
adapt during burn-in and stay fixed afterwards. Coordinates fixed with
``set_free_parameter_masks`` keep their values.

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

Priors
------

Difficulties have ``N(0, 1)`` priors, positive discriminations
``LogNormal(1, 1)`` priors (density proportional to ``exp(-log(a)**2 / 2)``),
guessing parameters ``Beta(2, 8)`` and upper asymptotes ``Beta(8, 2)``; other
item parameters are uniform on their family's estimation box, which for the
unidimensional 1PL-4PL is ``[0.1, 5]`` for slopes and ``[-6, 6]`` for
difficulties. These are the priors and boxes of the native unidimensional 2PL
kernel, so both backends sample the same posterior. ``priors`` replaces them by
parameter name and runs the NumPy sampler:

.. code-block:: python

   from mirt.estimation.priors import BetaPrior, NormalPrior

   sampler = mirt.GibbsSampler(
       priors={"difficulty": NormalPrior(0, 2), "guessing": BetaPrior(5, 17)},
       seed=42,
   )

A ``PriorSpecification`` sets the priors of the parameters it names, with
``LogNormal(0, 0.5)`` discrimination and ``N(0, 2)`` difficulty for omitted
fields; its ``theta`` prior must remain standard normal. Other values, such as
tuples or dictionaries of moments, raise ``MirtValidationError``.

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

Convergence is assessed for every free item coordinate with the
rank-normalized split R-hat and the bulk and tail effective sample sizes of
Vehtari et al. (2021), computed within the ``n_chains`` chains of the fit.
``rhat`` holds the largest R-hat of each item parameter, ``ess`` its smallest
bulk ESS and ``ess_tail`` its smallest tail ESS, and ``summary()`` prints all
three. ``converged`` is true when every R-hat is below 1.01 and every bulk and
tail ESS reaches 100 per chain, the thresholds the paper recommends.
``convergence_diagnostics()`` returns the per-coordinate values, shaped like
the parameters, with fixed coordinates as ``NaN``:

.. code-block:: python

   result = mirt.GibbsSampler(n_iter=4_000, burnin=1_000, n_chains=4, seed=1).fit(
       mirt.TwoParameterLogistic(n_items=responses.shape[1]), responses
   )
   diagnostics = result.convergence_diagnostics()
   print(result.converged, diagnostics["difficulty"]["rhat"])

Several chains make R-hat sensitive to chains that have not forgotten their
starts: chains start from the model's values jittered on the sampling scale.
The abilities and the item locations and slopes drift slowly as a whole, so a few
thousand iterations are typical; run longer chains when ``converged`` is
false.

Log-likelihood and information criteria
---------------------------------------

``MCMCResult.log_likelihood`` is the marginal log-likelihood at the posterior
means of the item parameters, integrated over the abilities as EM integrates
it: on EM's default grid for up to three factors, so it equals EM's
log-likelihood at the same parameters, and by importance sampling from normal
approximations of each person's posterior beyond. The AIC and BIC of
``fit_mirt(estimation="MCMC")`` are built from it and compare with EM fits.
``dic`` and ``waic`` are Bayesian criteria built from the conditional
log-likelihood of the draws, which ``chains["log_likelihood"]`` stores; its
posterior mean lies far above the marginal log-likelihood and is not a
substitute for it.

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
       batch_size=512,
   )
   print(waic(person_log_lik).summary())
   print(psis_loo(person_log_lik).summary())

``by="observation"`` retains the flattened person-item layout and assigns zero to
missing cells. ``by="observed"`` omits missing cells, producing the most compact
matrix when each observed response is the predictive unit. Exact unidimensional
2PL models evaluate fixed or sampled item parameters and abilities through a
batched native implementation when available, with a memory-bounded NumPy fallback.
Other built-in families evaluate curves in respondent blocks bounded by their
item and category counts. ``batch_size`` sets the maximum respondents per block
for either backend. The returned matrix still scales with posterior draws and
the number of predictive units. Fixed abilities and item parameters use broadcast
views instead of dense copies across posterior draws.

Customized 2PL curves follow their public probability methods. Custom models use
the complete respondent matrix by default to preserve callback context; an explicit
``batch_size`` opts into block evaluation. Model parameters and supplied chains
are preserved, including when a custom curve raises an exception.

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
