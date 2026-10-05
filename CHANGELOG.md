# Changelog

All notable changes to the mirt package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Multidimensional GradedResponseModel and NominalResponseModel now provide
  exact item_information_matrix(theta, item_idx) and
  test_information_matrix(theta). D- and A-optimal MCAT item selection for these
  models previously fell back to a first-category p*q*a a^T approximation, which
  could be about 20x too small (0.028 vs 0.569 for a 2-factor GRM item).
- Native GRM EM fits no longer jitter because the M-step stopped early under a
  loose relative tolerance. The native graded M-step now uses `ftol <= 1e-10`.
  GRM 2000x30x5 needs 48 instead of 176 iterations (about 2.5x faster), and
  estimates are about 10x closer to the converged optimum.
- The native `gibbs_sample_2pl` and `GibbsSampler` no longer panic when
  `(n_iter - burnin)` is not a multiple of `thin`. They keep
  `ceil((n_iter - burnin) / thin)` draws, as the NumPy sampler does. `thin=0`
  and `burnin >= n_iter` now raise `ValueError` (backend) or
  `MirtValidationError` (`GibbsSampler`) instead of a Rust panic. Native Gibbs
  log-likelihood chains are now summed sequentially, so a seed reproduces them
  exactly for any thread count.
- Native MHRM (`fit_mirt(estimation='MHRM')`, `MHRMEstimator` for unidimensional
  2PL) updated item parameters only after burn-in, and then with tiny gains. The
  default call returned the starting values a=1, b=0. It now updates every cycle
  with the NumPy gradient forms, honours `gain_sequence`, and returns the mean
  of the post-burn-in iterates (the final iterate when `burnin >= n_cycles`).
- `MCATEngine` now applies randomesque exposure control. It used to be silently
  ignored. `CATEngine` and `MCATEngine` share one content, exposure and
  selection dispatch, and randomesque breaks ties by item index.
- Progressive exposure control no longer raises for polytomous models (GRM and
  GPCM under CAT, and multidimensional GPCM under MCAT). Candidates are
  evaluated item by item when `information()` returns test totals.
- Randomesque over `RandomSelection` or `RandomMCATSelection` is now a seeded
  random choice across the pool. Before, it always picked among the lowest item
  indices. Both strategies now return seeded uniform scores from
  `get_item_criteria`.
- A-stratified selection now spends `test_length / n_strata` items in each
  stratum (Chang & Ying, 1999). Before, tests of 51 items or fewer on a 300-item
  pool never left the lowest-a stratum. `AStratified` gains `test_length`
  (capped at the pool size) and `within` ('MFI' or 'b-matching'), plus the
  method `current_stratum`. `CATEngine` supplies `max_items` when `test_length`
  is unset. `get_item_criteria` ranks only the current stratum, so randomesque
  stays inside it. Strata are rebuilt when the strategy is used with a different
  model object. This changes item sequences for existing a-stratified users.
- `create_selection_strategy` and `create_mcat_selection_strategy` match names
  regardless of case and surrounding whitespace and accept '_' for '-'. The
  documented 'a_stratified' and 'urry' now work.
- MCAT selection no longer approximates polytomous Fisher information from the
  category-0 probability. Polytomous models without `item_information_matrix`
  raise `MirtModelError`. Multidimensional GRM and NRM now provide exact
  matrices. MCAT item choices change for multidimensional ordinal models.
- diagnostics.compute_q3 and compute_ld_chi2 no longer send 3PL, 4PL and
  multidimensional models to the 2PL-only native kernel. They now match
  compute_ld_statistics on every backend; previously 3PL LD chi2 was off by up
  to about 400 and 2-factor models raised ValueError. The NumPy path is as fast
  or faster (10000x60 2PL: Q3 72 to 55 ms, LD chi2 399 to 68 ms).
- compute_outfit_infit and identify_misfitting_patterns now use the same
  infit/outfit definition as itemfit() and personfit(). Near-deterministic
  responses no longer inflate outfit (previously item outfit could reach about
  2e7), and infit no longer adds an epsilon to its denominator. Misfit flags can
  change for low-variance cells.
- S-X2 p-values for polytomous items are no longer NaN when a one-person score
  row's pooled expected count rounds just below min_expected. This affected many
  or all items on longer 3-5 category forms.
- S-X2 binary score-row pooling breaks ties between equally populated neighbours
  by observed person counts, so ties go to the lower score as documented instead
  of being decided by floating-point rounding. Affected items' S_X2 and df can
  change.
- compute_itemfit treats NaN as a missing response for the mean-square and
  ability-grouped statistics, as S-X2 already did, instead of raising.
- `fscores(..., person_ids=...)` now raises `MirtValidationError` before scoring
  when the identifiers are not one-dimensional with one per response row. Array
  identifiers are normalized to a list, so `to_dict`/`to_json` round-trips and
  `to_dataframe` works.
- `mirt.scoring.eapsum` is always the public scoring function. Previously,
  importing `EAPSumScorer` or `sum_score_to_theta` first replaced it with the
  implementation submodule ('module' object is not callable).
- EAPsum uses the compiled Lord-Wingersky recursion only for unmodified built-in
  1PL/2PL models. Subclasses that override `probability` were previously scored
  with the stock 2PL curve.
- `true_score_equating` now solves each old-form score exactly by root finding
  instead of clamping abilities to `theta_range`. Scores at or below the sum of
  lower asymptotes (e.g. 3PL guessing) follow Kolen & Brennan's line from (0, 0)
  to (L_X, L_Y) instead of collapsing to one flat value, and scores above the
  upper-asymptote sum interpolate to the maximum, so a zero score maps to 0 and
  a perfect score to a perfect score. `theta_range`/`n_theta` now only define
  the reporting grid in `result.theta`. Models are evaluated far outside
  `theta_range` and must return valid probabilities there.
- `link()` (Stocking-Lord, TCC and Haebara criteria, anchor-area diagnostics,
  TCC fit, bootstrap SEs and purification) now uses each model's own response
  function. 5PL asymmetry is included, and CLL, NLL, ULL, zero-inflated, hurdle
  and other non-logistic dichotomous families are matched through
  `model.probability` instead of a 3PL/4PL logistic curve. `link()`,
  `bootstrap_linking_se`, `delta_method_se` and `compute_linking_fit` now share
  one estimator. Chain and vertical results for non-logistic families change
  accordingly.
- Curve linking no longer fails with 'initial linking did not produce a positive
  finite slope' when one form's anchor difficulties are tied. It starts from
  mean/mean instead.
- `delta_method_se` no longer rejects model-native forms whose lower or upper
  asymptote is estimated on its [0, 1] bound. Those parameters are differenced
  one-sidedly.
- The deprecated `equate()` now delegates to `mirt.equating.link`. Its
  mean/sigma and mean/mean constants, its Stocking-Lord criterion (previously a
  duplicate of Haebara) and its `rmse` (now 0 for an exact link) are corrected,
  while it still reports `theta_new = A * theta_old + B`. Curve methods now use
  link's normal-weighted 61-point grid, so constants from noisy anchors shift
  slightly. Multidimensional or otherwise invalid anchor parameters are now
  rejected instead of silently using the first discrimination column.
- `transform_theta` now maps new-form scores onto the old scale as `(theta - B)
  / A`, matching its documentation and `equate()`'s convention. It previously
  applied the old-to-new map. It also rejects non-finite or non-positive A with
  MirtValidationError.
- fit_mirt no longer silently fits a one-factor model when n_factors > 1 is
  requested for 1PL, 3PL, 4PL or PCM; it raises MirtModelError. Invalid
  n_factors values (0, negative, non-integer, bool, strings, None) raise
  MirtValidationError. Before, they were ignored, stored as a bool, or raised a
  raw TypeError.
- fit_multigroup infers polytomous category counts per item from the pooled
  data, as fit_mirt does, instead of using one global maximum. Items with unused
  top categories no longer get a phantom threshold stuck at the optimiser bound,
  and n_parameters, AIC/BIC and the invariance LRT degrees of freedom are
  corrected.
- ItemAnalysisReport and FullDiagnosticReport include every cell of matrix
  parameters (e.g. GRM thresholds, multidimensional discriminations) instead of
  silently dropping them, and show unknown standard errors as NA instead of
  0.0000. Report p-values and intervals now come from
  FitResult.parameter_statistics.
- `assemble_form` and `assemble_parallel_forms` no longer discard the best
  feasible solution when a solver limit (`time_limit`, `node_limit`) is reached.
  The solution is checked against every constraint and returned with
  `is_optimal=False`.
- MCAT stopping rules now validate their parameters like the CAT rules. NaN or
  infinite thresholds, non-integer `max_items`/`n_stable`, and negative or
  non-integer `min_items` raise `ValueError`.
- `CombinedMCATStop.get_reason()` no longer reports a stale rule after a
  non-stopping state. Its `"or"` mode now short-circuits like `CombinedStop`,
  and `MCATStoppingRule` gains a default `reset()`.
- `NonlinearGrowthModel.fit_individual` now actually fits the curve. It runs a
  bounded nonlinear least-squares fit over the asymptote, the (log) rate and,
  for logistic/Gompertz, the inflection. Before, the rate was never estimated,
  the inflection gradient was taken at the model defaults, and the fixed-step
  gradient descent barely moved. Inputs are now validated, and a starting curve
  that overflows raises a clear error. The result has plain-float
  `asymptote`/`rate`/`inflection` plus new `converged` and `sse` entries;
  `converged` is False when the iteration limit is hit or the rate is stuck at
  its bound. The asymptote is no longer clamped to at least 0.1, and exponential
  fits return the model's unused `inflection` unchanged.
- `GrowthMixtureModel.predict_trajectory_moments` computes the conditional
  variance from the posterior random-effect covariance without a subtraction.
  Before, variances could lose most of their digits, or be clamped to zero, when
  the residual variance was small relative to the random effects.
- `bootstrap_se`, `bootstrap_ci` (including BCa jackknife fits) and
  `parametric_bootstrap` now actually warm-start replicate fits from the
  original estimates when `warm_start=True`; previously EM silently
  reinitialized every replicate. Seeded results for models other than the native
  2PL change by convergence noise, and BCa intervals on a GRM run about 2-4x
  faster.
- `warm_start=False` bootstrap replicates now keep user-fixed (masked) parameter
  values and reinitialize only free parameters, and the native 2PL bootstrap is
  skipped for models with free-parameter restrictions.
- Process workers for `bootstrap_se`, `bootstrap_ci`, `parametric_bootstrap`,
  the DTF/DRF bootstraps and `fit_models(parallel_backend='process')` now
  inherit the parent's backend preference, so seeded results are identical for
  every `n_jobs` even after `set_backend('numpy')`; `multi_start_fit(n_jobs>1)`
  now spawns workers instead of forking (scripts need an `if __name__ ==
  '__main__':` guard and picklable models).
- `LongitudinalGibbsSampler` restarted every theta Metropolis-Hastings step at
  the growth-curve prediction instead of the current chain state. That kernel
  was invalid and collapsed `residual_variance` to its 0.01 floor. Theta is now
  carried as chain state with prior N(growth-curve prediction,
  residual_variance). Seeded longitudinal results, including AIC and BIC,
  change.
- `ResponseTimeGibbsSampler` ignored
  `RTModelPriors.time_disc_mean`/`time_disc_var` and reused the covariance
  prior's `sigma_df` as a Gamma prior on time precision. Log time discrimination
  now gets the documented normal prior through vectorized random-walk
  Metropolis: five cheap sufficient-statistic steps per sweep, which keeps
  effective sample size close to the old direct draw. `sigma_df` now only
  affects the ability-speed covariance.
- `ResponseTimeGibbsSampler` added a spurious Jacobian to the accuracy
  discrimination Metropolis step, which shifted the effective log-discrimination
  prior mean from `disc_mean` to `disc_mean + disc_var`.
- WeightedEMEstimator, MCEMEstimator, QMCEMEstimator and StochasticEMEstimator
  no longer reset parameters fixed with set_free_parameter_masks to their
  default starting values; fixed coordinates now keep their values, as in
  EMEstimator.
- MCEMEstimator can now report convergence. Default fits previously always ran
  to max_iter. It uses an adaptation of ascent-based MCEM (Caffo, Jank & Jones
  2005): the log-likelihood change is checked on common draws, and the sample
  grows automatically up to the new max_samples (default 10 x n_samples). It
  adds a sample_size_history property and raises a RuntimeWarning when Monte
  Carlo precision is exhausted.
- The `likelihood_ratio` DIF method was a pseudo-Wald statistic with SE=1 on
  unlinked groups, so DIF went undetected. It is now a real nested
  multiple-group likelihood-ratio test that estimates the focal latent mean and
  variance.
- `wald` and `lord` DIF reported p=1 for every item because standard errors were
  disabled. Groups are now linked by Stocking-Lord over the anchors, and focal
  estimates and standard errors are rescaled to the reference scale. `lord` is a
  documented alias of `wald`. The test uses diagonal standard errors and ignores
  linking error, so it is liberal, especially for polytomous items and 3PL.
- `raju` DIF p-values came from an invented standard error, and group impact
  flagged every item. Areas now use linked curves, `p_value` is NaN, the ETS
  class uses the signed area, and `flag_dif_items` flags Raju results on effect
  size.
- `compute_drf` and `compute_item_drf` compared separately standardized group
  calibrations, so group impact was reported as DRF. The focal calibration is
  now linked onto the reference scale, and marginal reliabilities stay
  within-group.
- `compute_dtf` with `anchor_items` now links the groups, including in every
  bootstrap replicate. Without anchors it warns that impact is confounded with
  DTF.
- `grdif_effect_size` refitted an unrelated 2PL model. It now reuses the
  residual moments from `compute_grdif`'s own calibration and final abilities;
  hand-built results need `model=`.
- Separate group calibrations of polytomous models now share the category counts
  of the pooled data, with NaN-coded responses treated as missing, so linked
  items have matching structures.
- `compute_se(method="fisher")` used theta information and gave every parameter
  of an item the same SE. It now computes the marginal expected information by
  response-pattern enumeration, for at most 2^16 patterns; larger models raise
  an error pointing to "oakes".
- `BLEstimator` standard errors inverted only the Hessian diagonal. They now
  invert the full marginal Hessian, which is exact for built-in item models with
  any optimizer method, and store the covariance with `se_method="hessian"`.
- The native 2PL path of `fit_mirt` and the Python EM path now use the same
  standard-error estimator, so they agree up to their slightly different
  parameter estimates.
- Parameters on an optimizer bound (for example 3PL guessing at 0) get NaN
  standard errors and are held fixed, rather than contaminating the item's other
  SEs. Items with no observed responses get NaN rather than pseudo-inverse zeros
  in the matrix SE methods.
- The `estfun()` and `estfun_summary()` documentation no longer claims that
  plug-in scores sum to zero at the MLE or yield sandwich standard errors.
- The native 3PL EM iteration now reaches the bounded maximum when a guessing
  (or other) parameter sits on, or just inside, one of its bounds. It previously
  stalled at non-stationary points with a lower log-likelihood (0.1-9.9 below
  the NumPy EM optimum on seeded data) and different item estimates. Native and
  NumPy 3PL fits now agree, and the native fit needs far fewer EM iterations.
- Native multigroup E-steps now raise IndexError for out-of-range response
  categories, and ValueError for mismatched group inputs or invalid group
  priors, instead of a Rust PanicException. Native multigroup expected counts
  reject posteriors whose row counts do not match the responses.
- Negative item indices no longer silently select another item's curve. For
  example, `TwoParameterLogistic(3).probability(theta, -1)` returned item 2's
  curve, and so did the same call on 3PL, GRM, GPCM, NRM, RSM,
  MultidimensionalModel, BifactorModel and the partially compensatory,
  noncompensatory and disjunctive models. Boolean and non-integer indices
  (`True`, `np.True_`, `1.0`) are now rejected instead of being treated as item
  1 or raising a `TypeError`. NumPy integer scalars and 0-d integer arrays are
  still accepted.
- `set_parameters` and `set_item_parameter` on the 2PL-5PL, ULL, CLL, NLL,
  zero-inflated, hurdle, sequential, continuation-ratio and adjacent-category
  models now raise `MirtValidationError` for non-numeric values (including
  threshold lists), instead of a bare `ValueError` or `TypeError`. A rejected
  multi-parameter update still leaves every parameter unchanged.
- The `BKTGibbsSampler` docstring now correctly describes forward-filtering
  backward-sampling state draws and conjugate Beta parameter draws instead of
  "Baum-Welch style" updates.
- `EMEstimator` now estimates the shared thresholds of `RatingScaleModel` and
  the shared slope and thresholds of `GradedRatingScaleModel`. Before, fits
  reported `converged=True` with those parameters still at their starting values
  and item locations biased by about 0.35; the default fit with standard errors
  raised `MirtValidationError`.
- A shared parameter whose length equals `n_items`, such as `RatingScaleModel(3,
  4)` thresholds or the GRSM slope with one item, is no longer mistaken for a
  per-item parameter. This applies to `get_item_parameters`, EM fits, item
  priors, `extract_item`, S-X2, Monte Carlo EM standard errors,
  `fixed_item_calibration` and `FitResult` (summary labels, `vcov_labels`,
  `coef`). Before, `fixed_item_calibration` rejected or freed the anchors'
  thresholds, and `coef()` tabulated the threshold vector as one value per item.
- `RatingScaleModel.set_parameters` and `GradedRatingScaleModel.set_parameters`
  no longer mark the model as fitted. As with other families, EM now treats
  values set this way as unfitted; pass `start=` or fix them with masks.
- MHRM (estimation="MHRM" and MHRMEstimator, both the NumPy and native 2PL
  paths) now implements Cai's (2010) MH-RM. A Robbins-Monro step is
  preconditioned by a running estimate of the complete-data information, burn-in
  cycles use unit (Newton) gains, and gains decrease afterwards. Default fits
  now match EM: on seeded 2PL 1000x10 data, difficulties are within about 0.1 of
  EM and the difficulty range is within a few percent, where they were
  previously shrunk to about 0.6x.
- NumPy MHRM now supports every built-in item family. Polytomous models
  previously raised an error. 3PL/4PL guessing and upper parameters and
  multidimensional slopes were previously never updated. The fixed 1PL slope is
  no longer modified.
- GibbsSampler(parallel_chains=True) now runs NumPy chains in spawned worker
  processes that inherit the backend preference, avoiding fork-after-threads
  deadlocks. Seeded parallel and serial runs produce identical draws, and
  unpicklable models raise MirtValidationError.
- Several README examples no longer ran: `MCATEngine` was given a nonexistent
  `selection_method` argument and read a nonexistent `theta_cov` attribute, the
  DINA Q-matrix had the wrong number of rows, `from mirt.utils import residuals`
  imported a module instead of the function, the person-fit filter failed under
  polars, and later sections overwrote `result` with unrelated fits. Apart from
  data the reader must supply, the examples now run from top to bottom.
- The documentation build passes with warnings treated as errors again. The
  `dif` and `compute_drf` docstrings had bullet lists that docutils rejected,
  and ambiguous cross-reference warnings caused by attribute shape descriptions
  are now suppressed. New tests check that every guide is in the table of
  contents, that every API reference entry imports, and that README and guide
  code parses and names only existing `mirt` objects.
- Diagnostics and utilities given a `FitResult` now use its estimated
  `latent_covariance` as the latent population instead of assuming uncorrelated
  standard-normal factors. Examples are the factor correlations of
  `fit_mirt(spec=...)` and a Rasch latent variance. This covers
  `itemfit`/`compute_itemfit`/`compute_s_x2` (S-X2, X2, G2, PV-Q1, infit and
  outfit), `compute_m2`/`compute_fit_indices`, the residual and local-dependence
  statistics, `vuong_test`, `generate_plausible_values`, `impute_responses`,
  `simdata`, `eapsum`/`eapsum_table`/`sum_score_to_theta`,
  `marginal_rxx`/`conditional_rxx` and the HTML reports. A two-factor CFA with
  an estimated correlation of 0.64 (N=1500) now gets M2=32.5 on 34 df (p=0.54)
  instead of 265.7 on 35 df (p<1e-36). Bootstrap refits still assume
  uncorrelated standard-normal factors.
- MH-RM standard errors for unidimensional 1PL-4PL, GRM, GPCM and PCM fits now
  come from the exact observed information of the marginal likelihood at the
  MH-RM estimates, as EM computes with `se_method="oakes"`. They fill
  `FitResult.vcov` and set `se_method="oakes"` on both the NumPy and native 2PL
  paths. For converged 2PL, GRM and GPCM fits they agree with EM's errors within
  about 10%. Other models still report the spread of the post-burn-in iterates,
  now labelled `se_method="mhrm_iterate_sd"` because it is not a sampling
  standard error.
- MH-RM fits report `n_observations` as the number of persons rather than
  persons times items, so `vuong_test` accepts them and `compare_models`
  information criteria use the right sample size.
- The native GRM M-step now keeps thresholds ordered, using the same 1e-6
  minimum gap as the generic optimizer. Before, it applied only box bounds, so
  BFGS steps could cross thresholds (e.g. [0.38, 6, -6]). The resulting negative
  category probabilities were clipped, and the reported log-likelihood was
  inflated above the generating model's. In 2000-person, 30-item, 5-category
  simulations with sorted standard-normal thresholds, 7-11 of the 30 items ended
  disordered. Native fits now match `use_rust=False` within optimizer tolerance.
  When the ordering constraint never binds, native fits are unchanged bit for
  bit. The low-level `mirt_rs.m_step_polytomous` kernel also rejects fixed
  graded thresholds that are disordered or leave no room for free ones, raising
  ValueError.
- The bundled LSAT7 dataset had 980 rows and 13 wrong pattern counts, which gave
  near-zero or negative inter-item correlations. It now holds Bock and
  Lieberman's (1970) 1000 examinees, matches the published item proportions, and
  reproduces R mirt's 2PL estimates (log-likelihood -2658.805).
- fit_mirt(spec=..., accelerate='squarem') now forwards the acceleration option
  to the confirmatory fit instead of silently ignoring it.
- `fscores` (EAP, MAP, ML, WLE, EAPsum), `ability_posterior`, `eapsum`,
  `eapsum_table`, `personfit`/`compute_personfit`, `itemfit`/`compute_itemfit`,
  `vuong_test` and the residual diagnostics (`compute_residuals`,
  `analyze_residuals`, `compute_outfit_infit`, `identify_misfitting_patterns`)
  now read `NaN`, pandas `pd.NA` and polars nulls as missing responses, as
  `fit_mirt` does. Results are identical to those for the `-1`-coded matrix, so
  `fscores(fit_mirt(df), df)` now works for data frames with missing values.
  Infinite values are still rejected.
- `compute_m2`, `compute_fit_indices`, the local-dependence diagnostics
  (`compute_ld_statistics`, `compute_q3`, `compute_ld_chi2`) and `residuals`,
  `Q3` and `LD_X2` now accept nullable pandas data frames. Previously they
  raised `TypeError` or 'must contain numeric values'.
- `analyze_residuals` no longer splits rows containing `NaN` into separate
  response patterns, and GRM deviance residuals no longer raise `IndexError` for
  float-coded responses.
- Mixed-format fits with item priors (`fit_mirt(model=[...], priors=...)`,
  `MixedFormatEMEstimator(item_priors=...)`) now include each prior's curvature
  in standard errors and `FitResult.vcov` for every `se_method` (oakes,
  crossprod, sandwich, complete_data). Previously they were likelihood-only
  errors around posterior-mode estimates, often several times too large.
- `estimate_covariance` raises `MirtValidationError` when `prior_information`
  names a parameter the model lacks, instead of silently dropping that prior.
- `MixedItemModel` exposes its components' shared parameters (rating-scale
  thresholds, the GRSM slope). `FitResult.vcov_labels` and `summary()` label
  them once per component (`RSM.thresholds[1]`) instead of per item.
  `item_parameter_arrays()` and `coef()` raise the documented `MirtModelError`
  instead of returning misaligned rows.
- `BLEstimator` on a `MixedItemModel` uses each component's optimizer bounds
  (e.g. guessing in [0, 0.5]) instead of (-10, 10), which allowed negative
  guessing estimates.
- `compute_se` with the itemwise methods (`"numerical"` default, `"central"`,
  `"forward"`, `"richardson"`) supports `MixedItemModel`, component by component
  under qualified names, instead of raising `IndexError`.
- `fit_mirt(model=[...], spec=...)` and `fit_multigroup(model=[...])` accept a
  per-item family sequence that names one family, as `fit_mirt` does. Mixed
  families raise a clear `MirtValidationError` instead of the misleading
  "Unknown model: [...]". The model factory also rejects family sequences with a
  clear message.
- Itemwise estimators (e.g. `MCEMEstimator`) reject a `MixedItemModel` up front
  with advice to use `MixedFormatEMEstimator`, also when a component has shared
  parameters.
- bootstrap_se, bootstrap_ci (including the BCa jackknife), parametric_bootstrap
  and bootstrap_lr now refit a FitResult with its own estimator, item priors,
  latent density, factor covariance pattern, equality constraints and
  quadrature. Previously every replicate was refitted under N(0, I) without
  priors or constraints, so FIPC intervals excluded the point estimates and
  Bayes-modal fits were bootstrapped as maximum likelihood.
- Bfactor results are bootstrapped and multi-started with BifactorEMEstimator
  instead of the exponential product grid of EMEstimator. bootstrap_se with S=3,
  N=300 and n_bootstrap=2 went from several minutes to 0.6 s; S=4 no longer
  needs about 10 GB.
- parametric_bootstrap and bootstrap_lr draw abilities from the fit's latent
  mean and covariance.
- bootstrap_lr counts free parameters from FitResult.n_parameters for fits
  refitted with their recorded settings, so constrained fits and estimated
  factor correlations are tested with the right degrees of freedom instead of
  being rejected.
- multi_start_fit gives every start its own copy of a latent_density instance,
  so the caller's instance is no longer mutated, and ranks Bayes-modal starts by
  log-posterior.
- fixed_item_calibration(...).fit_result and EMEstimator fits with a
  non-standard Gaussian density carry their latent population. fscores,
  ability_posterior, plausible values, simdata, itemfit, vuong_test,
  marginal_rxx and impute_responses use it by default, and resolve_latent_prior
  now also defaults prior_mean.
- The bootstrap functions warn when a FitResult without refit settings (for
  example one from from_dict) has item priors or a latent population that the
  default refit would drop.
- `mirt.compute_se` matrix methods (`oakes`, `louis`, `sem`, `crossprod`,
  `sandwich`, `fisher`) now hold coordinates on an EM optimizer bound fixed,
  with `NaN` standard errors, as fits do. They now reproduce the standard errors
  of an EM fit without priors or equality constraints. `compute_oakes_se`,
  `compute_crossprod_se`, `compute_sandwich_se` and `compute_sem_se` accept
  `bounds=` and `prior_information=`.
- A `PriorSpecification` that reaches no item parameter (e.g. for NRM) now
  raises `MirtValidationError` from `fit_mirt(priors=...)` and
  `EMEstimator(item_priors=...)`, as `fit_mirt(spec=...)` already did, instead
  of being silently ignored. Mixed-format fits check across all components. An
  explicit non-default discrimination or difficulty prior for a parameter the
  model lacks (e.g. difficulty on GRM) now warns.
- `draw_parameters(result)` now raises a clear `MirtValidationError` instead of
  a bare `KeyError` for models that derive difficulty rather than store it (e.g.
  LLTM).
- `draw_parameters(model, vcov=...)`, `wald(model, vcov=...)` and `lagrange(...,
  vcov=...)` now accept a covariance such as `result.vcov`, whose all-NaN rows
  and columns mark parameters on a bound. Those parameters are held at their
  estimates and cannot be tested.
- EM item M-steps now solve built-in dichotomous items with analytic gradients
  (1PL-4PL and the logistic items of multidimensional and bifactor models) to a
  relative tolerance of at most 1e-10. 4PL fits, and 3PL fits with item priors,
  fixed coordinates, `n_jobs` > 1 or `use_rust=False`, now reach the optimum.
  Before, they stopped up to hundreds of iterations later at estimates up to 0.8
  away. They now run 3-12x faster, and 4PL fits that used to hit `max_iter`
  converge.
- GRM fits with `use_rust=False`, item priors or fixed coordinates no longer
  raise `MirtEstimationError('... did not preserve threshold ordering')` when an
  empty category ties thresholds; 18 of 30 seeded datasets did before. Steps are
  projected onto ordered thresholds with the native kernel's 1e-6 gap and never
  raise the item objective. Fixed thresholds that leave the free thresholds
  between or beyond them no ordered room within the bounds now raise
  `MirtValidationError` naming the item.
- `EMEstimator(latent_density=...)` accepts the advertised names
  'empirical'/'histogram'/'eh', 'ehw', 'davidian' and 'mixture', which
  previously raised TypeError. Unknown names raise `MirtValidationError` at
  construction. Univariate densities ('davidian', 'mixture', 'ehw') are refused
  for multidimensional models. 'gaussian'/'normal' now use the prior mean and
  covariance passed to `fit`.
- `WeightedEMEstimator` results report `se_method='complete_data'`. The
  estimator accepts `compute_standard_errors`, `prob_epsilon`,
  `item_optim_maxiter`, `item_optim_ftol` and `se_step_size`.
- `MHRMEstimator` no longer advises `set_free_parameter_masks`, which it also
  rejects, when refusing RSM/GRSM. Its docstring and the estimation guide now
  say rating scale models need EM or Bock-Lieberman estimation.
- The native 2PL EM (default `fit_mirt(model='2PL')` and cold-start
  `bootstrap_fit_2pl`) starts from logit-scale difficulties ln((1-p)/p), as its
  documentation said. This about halves the EM iterations: 30 to 15 on a 15-item
  test with difficulties in [-3, 3], and 39 to 19 on a default simulated 2PL
  test. Estimates are unchanged.
- The `bootstrap_lr` docstring no longer describes a native 3PL guessing prior
  that does not exist.
- Exploratory multidimensional models now start with slopes staggered across
  factors instead of equal slopes, so EM no longer stops at the symmetric saddle
  point where the factors coincide. This covers models where several factors
  have the same free slope pattern. It applies to fit_mirt(n_factors > 1) and to
  MultigroupModel fits. Configural multigroup fits now reach at least the
  log-likelihood of the nested metric and scalar fits, and exploratory fits
  reach a higher log-likelihood: 4 to 21 units in simulated 2- and 3-factor
  data, about 95 for one 10-item 2PL. The real maximum often lies near a slope
  bound, so these fits usually need several times more EM iterations, and
  3-factor models can reach max_iter.
- z_infit/z_outfit are no longer documented as standard normal on the default
  path: with EAP abilities from the same responses, about half of well-fitting
  items fall below -1.96. compute_itemfit and mirt.itemfit now warn when z
  statistics are requested without theta, and mirt.itemfit accepts theta= for
  external abilities.
- An integer reference_group that is also the label of a different group now
  raises ValueError instead of silently selecting by index, for example labels
  {1, 2} with reference_group=1. compute_dif passes its reference group by
  label. The reference_group of fit_multigroup, compare_invariance,
  test_invariance_hierarchy, multigroup_dif and select_dif_anchors now defaults
  to None (the first group in sorted order), so default calls work for any
  labels.
- multigroup_dif raises ValueError when the tested parameter families have no
  free coordinates, for example parameters=('discrimination',) for 1PL. Before,
  it ran 1+J fits and returned df=0/NaN rows. It now warns when only some
  studied items lack free coordinates.
- The multigroup item optimizer uses the single-group parameter boxes, keeping
  the negative-slope box for multidimensional binary items.
  FiveParameterLogistic and other families with asymmetry, engagement or
  zero-inflation parameters no longer get the (-10, 10) fallback, and a 5PL
  MultigroupModel no longer crashes with 'asymmetry must be strictly positive'.
- Removed unconditional noise from quadrature plausible values so posterior
  variance and joint factor distributions match their probability masses.
  Native sampling preserves exact zero prior masses and seeded reproducibility.
- Validated normal scoring priors consistently, rejecting nonfinite, asymmetric,
  and nonpositive-definite covariance matrices instead of silently changing
  their distributions. Sum-score lookup now validates fitted unidimensional
  models and retains owned prior inputs.
- Made delta-method covariance validation and standard-error propagation
  respect parameter units, including very small variances, singular covariance,
  and mixed magnitudes, without squaring representable errors into overflow.
- Corrected BCa bias ranks for tied statistics and acceleration scaling, and
  used the complete leave-one-person-out jackknife instead of a hidden
  20-person approximation. Unsuccessful jackknives report missing intervals.
- Required release filenames and archive metadata directories to agree with
  the published package/version, preventing stale or mislabeled artifacts
  from passing the publication gate.
- Preserved custom batch information and reused item-callback scratch buffers
  safely during form assembly.
- Conditioned multiple-imputation ability draws on observed responses and
  preserved joint factor dependence instead of scoring artificially filled
  responses. Discrete posterior sampling skips zero-mass nodes at CDF boundaries.
- Counted independent coefficients across cognitive-diagnosis, explanatory,
  ordinal-process, custom, testlet, unfolding, nonparametric, and mixture models,
  excluding design metadata, inactive padding, fixed bounds, and dependent storage.
  Every item-model copy preserves additional fixed-parameter masks.
- Canonicalized rating-scale shared thresholds by storing the first threshold
  as zero and absorbing its raw value into item locations. Nominal and nested
  distractor setters now store reference-category contrasts. Response
  probabilities are preserved, while returned parameter arrays use the identified
  representation. Declared common testlet loadings are enforced by generic setters.
- Replaced M2/M2* and S-X2 approximations with covariance-weighted nuisance
  projection and exact conditional score-category probabilities. Degrees of
  freedom respect parameter masks and unestimable inference is explicit.
- Identified multigroup latent distributions from the remaining invariance
  anchors, retained fixed coordinates through initialization and optimization,
  and enforced graded-response threshold order around fixed neighbors.
- Aligned longitudinal drift by declared physical item identities and used
  signed reference-scale difficulty changes to classify drift direction.
- Inferred ordinal category counts per item and accepted explicit heterogeneous
  category metadata in the public fitting API.
- Corrected chain-link transformation directions and preserved requested
  weighting, robust estimation, and linking methods during anchor purification
  and bootstrap uncertainty. Concurrent linking now rejects failed optimization
  and uses the selected reference metric; oblique multidimensional linking uses
  direct least squares for nearly collinear anchor loadings.
- Corrected SIBTEST sampling uncertainty, pooled score-stratum weighting, and
  true-score regression correction. Crossing inference uses estimated crossing
  regions with chi-square tests instead of an effect-dependent standard error.
  Unestimable strata and degenerate groups report missing inference explicitly.
- Preserved customized and inherited 2PL curves in Bayesian pointwise
  likelihood preparation. Rejected response codes before integer conversion
  can overflow, and protected supplied parameter chains during failed custom
  curve evaluations.
- Corrected score equating to invert the documented new-to-reference linking
  transformation. Equivalent calibrations now yield identical true-score
  tables. Compiled score recursion respects overridden model curves and exact
  model types.
- Corrected multidimensional GPCM adjacent logits to
  `theta @ a - sum(a) * step`, matching centered-threshold simulation and the
  two-category 2PL reduction. Updated fitting gradients and uncertainty
  curvature consistently. Previously fitted multidimensional GPCM parameters
  should be recalibrated. Stabilized GPCM, PCM, and RSM information at saturated
  response probabilities.
- Rejected invalid CAT/MCAT response categories before mutating administration
  state. A corrected response can still answer the pending item. Simulation
  inputs are validated before resetting an active session.
- Counted initial interactive CAT/MCAT exposure sessions correctly, without
  counting unused resets as additional examinees.
- Matched native CAT simulation posteriors to the public EAP scorer's bounded
  binary probabilities, including saturated curves. Native dispatch preserves
  customized model, selection, control, stopping, and engine methods. Native
  results preserve configured stopping-rule priority and distinguish item-pool
  exhaustion from a reached length cap.
- Checked native response-buffer alignment before constructing Rust slices,
  normalized unaligned inputs in the Python grouping wrapper, and copied array
  dimensions before releasing the GIL.
- Preserved finite native equating logits when ability/difficulty subtraction
  overflows before multiplication by a small item discrimination.
- Preserved the selected backend in cross-validation worker processes and used
  portable spawned workers by default, avoiding dependence on forkserver sockets.
- Made binary likelihood validation consistent for numeric response matrices,
  prevented unsigned subtraction overflow, and evaluated public probability
  curves at the original ability points. Batched likelihoods now broadcast
  constant curves correctly and preserve missing-item masking and finite
  per-item cancellation with very large responses. Custom binary validation
  remains authoritative in prepared and Monte Carlo likelihood paths, and
  boolean responses preserve custom float32 curve arithmetic.
- Fixed GPU EM dispatch for GPCM step parameters and ignored inactive padding
  in GRM/GPCM likelihoods with item-specific category counts. Used exact model
  types for dispatch, preserving customized display names.
- Preserved class-level and inherited model hooks in prepared item, BL,
  uncertainty, native fitting, and response-compression paths. Shared original
  hook checks are recorded at model definition time, including overrides made
  before estimator imports. Polytomous likelihoods now respect public
  probability overrides, and custom curves retain numerical fitting.

### Added

- `EMEstimator(accelerate="squarem")` adds SQUAREM (SqS3) EM acceleration with
  step-length adaptation, projection onto the item parameter boxes, a GRM
  threshold-order check, and a fallback to the plain EM step whenever the
  extrapolated point lowers the likelihood. On 2000x30 benchmarks it needs 2-3x
  fewer E-steps (2PL 31 -> 16, GRM 48 -> 23, 2-D 2PL 59 -> 19). It falls back to
  plain EM, with a warning, when the latent density is estimated. The default
  stays `"none"`.
- `EMEstimator` has a class docstring documenting every option.
- `MHRMEstimator` validates `n_cycles`, `burnin`, `proposal_sd` and
  `gain_sequence`, and `GibbsSampler` validates `n_iter`, `burnin` and `thin`;
  both raise `MirtValidationError`. The native `mhrm_fit_2pl` backend accepts
  `gain_sequence`.
- Lock-step batch simulation for unmodified engines using MFI, EAP, SE and/or
  maximum-length stopping (including `MaxItemsStop` alone), no exposure or
  content control, and built-in unidimensional 1PL-4PL, GRM, GPCM or PCM models.
- 'z_infit' and 'z_outfit' Wilson-Hilferty standardized mean squares for itemfit
  and personfit, plus compute_outfit_infit(include_standardized=True).
- Ability-grouped item fit statistics 'X2' (Bock/Yen Q1), 'G2' and 'PV_Q1'
  (Chalmers & Ng plausible-value Q1) for unidimensional models. They come with
  <name>_df and <name>_p columns, optional <name>_p_adjusted columns, and
  n_groups (default 10), n_plausible and seed arguments. X2/G2 p-values are
  approximate and liberal on short tests. n_groups now warns as deprecated only
  when it is used solely for S-X2.
- `mirt.scoring.eapsum_table` and `EAPSumScorer.score_table`, which return a
  `SumScoreTable`: theta, standard error and model-implied proportion for every
  attainable sum score, plus observed/expected counts and standardized residuals
  when complete responses are given. It has `to_dict` and `to_dataframe`. No X2
  fit statistic is reported.
- `kernel_equating` for Gaussian kernel equating (von Davier, Holland & Thayer
  2004), with bandwidth selection by penalty minimization (PEN1 + kappa*PEN2),
  fixed or `"linear"` bandwidths, optional maximum-likelihood log-linear
  presmoothing (degree below the number of observed scores), and delta-method
  standard errors of equating when sample sizes are given.
- `irt_kernel_equating`, the IRT observed-score kernel equating variant, which
  continuizes Lord-Wingersky score distributions and supports linking constants,
  item subsets and custom populations, plus `KernelEquatingResult`, a
  `ScoreEquatingResult` subclass that reports the bandwidths and continuized
  distributions.
- `chain_link` supports GRM, GPCM/PCM and NRM forms through their polytomous
  linkers, and `vertical_scale(method="chain")` inherits this. Adjacent forms
  must share a response family. `transform_to_reference` transforms GRM, GPCM
  and NRM parameters. NRM pairs report robust drift z-statistics but NaN
  difficulty changes.
- fit_multigroup accepts item_names and per-item n_categories, and validates
  responses as fit_mirt does. Non-2D data now raises MirtDataError, which is
  still a ValueError, and dichotomous codes above 1 are rejected.
- fit_mirt and fit_multigroup use unique pandas or polars DataFrame column names
  as item_names when item_names is not given. Default positional labels (pandas
  0..n-1, polars column_0..) keep the Item_1.. names.
- FitResult.from_dict and FitResult.from_json rebuild fitted
  1PL/2PL/3PL/4PL/GRM/GPCM/PCM/NRM results from full to_dict/to_json exports,
  including multidimensional and mixed-category models, so stored calibrations
  can be reloaded and scored. to_dict()['model'] now records per-item
  n_categories (None for dichotomous models).
- plot_icc, plot_category_curves, plot_information, plot_person_item_map,
  plot_expected_score and plot_se accept a FitResult as well as a model.
- `ShadowTestSelection` (mirt.cat) for shadow-test item selection (van der
  Linden & Reese). Each step assembles a full constrained test with
  `assemble_form`, so a completed adaptive test meets the content blueprint,
  enemy pairs, all-or-none item bundles and cost budget exactly. It works with
  randomesque and Sympson-Hetter exposure control, relaxing exposure exclusions
  when the constraints need an excluded item.
- `SEChangeStop`, `MinInformationStop` (catR's minInfo) and
  `PredictedSEReductionStop` (Choi, Grady & Dodd, 2011) stopping rules, plus the
  `"se_change"` stopping-rule name. Paired with an SE rule, they end tests early
  for examinees the pool cannot measure precisely.
- `summarize_cat_simulation` / `CATSimulationReport` (mirt.cat), a catR-style
  simulation report. It covers bias, RMSE, MAE, correlation, test-length and
  stopping-reason statistics, item exposure rates with Wilson bounds, unused
  items, the chi-square exposure index, the test overlap rate, and conditional
  results by ability bin. MCAT results get per-factor accuracy.
- `require_optimal` option and `is_optimal` / `mip_gap` result fields for
  `assemble_form` and `assemble_parallel_forms`.
- `generate_plausible_values(..., prior_mean=, prior_cov=)` draws plausible
  values under a normal population prior, shared or person-specific
  (latent-regression conditioning), for both posterior and MCMC methods;
  defaults are unchanged.
- `PolytomousItemModel.simulate(theta, seed=None, *, chunk_size=None)` for GRM,
  GPCM, PCM, RSM, GRSM, NRM, nested-logit, sequential and GGUM models, with
  chunk-invariant seeded output.
- `simdata` accepts a fitted dichotomous or polytomous model or a `FitResult`
  and simulates a new sample from it; `n_items` now defaults to the length of
  supplied item-parameter arrays (otherwise 20).
- `fixed_item_calibration` and `FixedItemCalibrationResult` for MML fixed-item
  parameter calibration (FIPC) of any item model family, holding anchor items
  fixed while estimating new items and the calibration population's latent mean
  and covariance.
- `bootstrap_lr` and `BootstrapLRResult`, a parametric bootstrap
  likelihood-ratio test for nested models (R mirt `boot.LR`) that works for
  boundary comparisons such as 2PL vs 3PL; seeded results are identical for
  every `n_jobs`.
- `RTModelPriors` is listed in the API reference, and the response-time guide
  documents the item and covariance priors.
- `start` option ("default", "model", or a mapping of parameter arrays) on
  EMEstimator.fit, WeightedEMEstimator.fit,
  MCEMEstimator/QMCEMEstimator/StochasticEMEstimator.fit and GVEMEstimator.fit;
  "default" keeps the previous behavior (reset an unfitted model, warm-start a
  fitted one).
- fit_mirt(start_values=..., fixed=..., priors=...) for user starting values,
  fixed parameters (Boolean masks, True = fixed) and item priors. When any of
  them takes effect, fit_mirt skips the native 2PL EM fast path. start_values
  with MHRM or Gibbs runs the NumPy samplers.
- Bayes modal (MAP) EM via EMEstimator(item_priors=PriorSpecification | {name:
  Prior}). Priors apply to free coordinates in every itemwise M-step path,
  including threaded and constrained GRM, and convergence is judged on the
  log-posterior. FitResult.log_posterior reports it, and it also appears in
  summary(), fit_statistics() and to_dict/from_dict. log_likelihood, AIC, BIC
  and standard errors stay likelihood-based.
- Prior.grad_log_pdf (analytic for the built-in priors, numerical fallback for
  subclasses) and CustomPrior(grad_log_pdf_fn=...).
- Documentation guide docs/guides/estimation.rst (starting values, fixed
  parameters, item priors, MCEM convergence).
- `mirt.multigroup.multigroup_dif` (also `mirt.multigroup_dif`). It runs nested
  multiple-group likelihood-ratio DIF tests with the `drop`, `add`,
  `drop_sequential` and `add_sequential` schemes. Options cover anchor items,
  the tested parameter families (`discrimination`, `intercepts`; other
  parameters such as 3PL guessing stay equal across groups) and multiplicity
  adjustment (Holm by default). Each studied item gets chi2, df, raw and
  adjusted p-values, ΔAIC/ΔBIC, flags and convergence. Refits are warm-started
  and run in parallel with `n_jobs`.
- `mirt.multigroup.select_dif_anchors` selects DIF-free anchor items by
  iterative all-other-as-anchor tests or a single-pass ranking.
- `MultigroupEMEstimator.fit(initial_latent=...)` warm-starts the estimated
  latent means and covariances from a previous fit.
- Keyword-only `anchors`, `scheme` and `n_jobs` arguments for `mirt.dif` and
  `compute_dif`. Results now report per-item `df`, `tested` and `converged`,
  plus `method`, `anchors` and `linking_constants`.
- `anchor_items` for `compute_dtf`, `compute_drf` and `compute_item_drf`. The
  results report `anchor_items` and `linking_constants`.
- `compute_grdif` results include the per-group residual moments `mrr`, `msr`
  and `group_item_counts`.
- `se_method` option on `fit_mirt` and `EMEstimator` ("auto", "oakes",
  "crossprod", "sandwich", "complete_data").
- `FitResult.se_method`, `FitResult.vcov` and `FitResult.vcov_labels` record the
  SE estimator and the labelled free-parameter covariance. They are serialized
  by `to_dict`/`from_dict`/`to_json` when standard errors are included.
- `wald()` and `draw_parameters()` accept a `FitResult` and use its parameter
  covariance. Hypotheses on fixed or boundary parameters are rejected, and
  asymptote and asymmetry parameters are drawn from their own uncertainty.
- `mirt.MirtIndexError` (also in `mirt.exceptions`), which subclasses both
  `MirtValidationError` and `IndexError`. Dichotomous, polytomous, nested,
  sequential, zero-inflated, mixture, CDM, unfolding, testlet, compensatory,
  multidimensional and bifactor item models now raise it for an invalid item
  index, with a consistent message ('item_idx must be an integer' or 'item_idx N
  out of range [0, n)'). Code catching either `ValueError` or `IndexError` keeps
  working.
- `fit_bkt_em` fits Bayesian Knowledge Tracing parameters by maximum likelihood
  (Baum-Welch EM). The M-step is closed form and the log-likelihood never
  decreases. It supports shared and person-specific trial layouts, optional
  per-skill forgetting, multiple seeded starts, and slip/guess bounds (default
  upper bound 0.5, which prevents label-swapped solutions; equal bounds fix the
  parameter and remove it from the AIC/BIC count). It reports `converged`,
  `n_iterations`, the log-likelihood trace and per-start log-likelihoods in the
  new `BKTEMResult`. It takes about 0.1-0.3 s for 2000 learners x 30 trials,
  versus about 25-35 s for the default `BKTGibbsSampler` run, and the estimates
  agree to within 0.002.
- `BKTModel` smoothing can return the smoothed transition probabilities
  (learning and forgetting since the previous same-skill opportunity) from the
  same vectorized backward sweep. EM uses them for the forgetting expectations.
- `BKTGibbsSampler`, `BKTPriors`, `fit_bkt_em` and `BKTEMResult` are now
  exported from `mirt` and `mirt.estimation`.
- `mirt.bfactor(data, specific_factors, ...)`, a full-information bifactor fit
  like R's `mirt::bfactor`, and `BifactorEMEstimator`. Gibbons-Hedeker dimension
  reduction integrates each specific factor jointly with the general factor on
  an `n_quadpts**2` grid instead of the `n_quadpts**(1+S)` product grid. The
  log-likelihood is identical to the product-grid quadrature and the cost grows
  linearly in the number of specific factors, so the default 21 points stay
  practical for any number of specific factors. It is about 13-17x faster than
  product-grid `EMEstimator` at S=3, Q=7. It supports dichotomous
  `BifactorModel` with independent standard-normal factors; an item loads on the
  general factor only when its specific loading is fixed at zero.
- Bifactor standard errors from the exact observed information (Louis identity),
  computed on the reduced posteriors. `se_method` accepts auto/oakes, crossprod,
  sandwich or complete_data, and the covariance is stored in `FitResult.vcov`.
- `EMEstimator` issues a `RuntimeWarning` before building a product quadrature
  grid of more than one million nodes, and points bifactor models to
  `mirt.bfactor`.
- A `bifactor-fit` benchmark suite and a 'Bifactor models' section in the
  estimation guide.
- `mirt.mirt_model` parses R `mirt.model`-style syntax into a frozen
  `mirt.ModelSpec`. It supports factor lines with item ranges, names and name
  ranges, plus `COV`, `FIXED`, `START`, `PRIOR` and `CONSTRAIN` statements, with
  comments, continued lines and line-numbered errors. Numbers always refer to
  item positions, `to_syntax()` round-trips, and the repr is readable.
- `fit_mirt(spec=...)` fits confirmatory models from a `ModelSpec` or a syntax
  string, using EM only. A multi-factor 2PL is fitted as a slope-intercept
  `MultidimensionalModel`; GRM and GPCM hold unloaded discriminations at zero;
  one-factor specs work for every family. `FIXED`, `START` and `PRIOR` go
  through the existing `fixed`, `start_values` and `priors` machinery.
  `CONSTRAIN` is parsed but raises `NotImplementedError`. A `PriorSpecification`
  that matches no parameter of the fitted model is rejected instead of being
  silently ignored.
- `mirt.estimation.FactorCovarianceDensity` estimates factor correlations for
  any `COV` zero pattern, and factor variances anchored by a fixed slope. After
  each item M-step, EM takes a conditional-maximization (ECM) step over the free
  covariance entries, so iterations stay monotone, including with item priors.
- `FitResult.latent_covariance` and `FitResult.factor_correlation`, which
  `summary()` reports and `to_dict`/`from_dict` serialize. `fscores` and
  `ability_posterior` use the estimated covariance as their default prior
  covariance.
- `mirt.fit_mfrm` / `mirt.estimation.fit_mfrm` estimates `ManyFacetRaschModel`
  and `PolytomousMFRM` (rating scale or partial credit) by marginal maximum
  likelihood. Item difficulties, anchored facet levels, centred thresholds and
  the person SD are estimated jointly with an analytic gradient. Standard errors
  for every item, facet level, threshold and sigma come from the exact (Louis)
  observed information.
- `fit_mfrm` integrates person measures on an equally spaced grid over +-6
  sigma. By default the grid is refined (41 up to 401 nodes) until it resolves
  the person posteriors, so sigma and item difficulties stay unbiased when
  persons have many ratings. An explicit `n_quadpts` that is too coarse emits a
  RuntimeWarning, and `MFRMResult.n_quadpts` reports the grid used.
- `fit_mfrm` accepts a long layout through `item_indices`, so several raters can
  score the same person and item; item and facet indices at missing ratings are
  ignored.
- MFRM infit and outfit mean squares for every item and facet level, averaged
  over each person's posterior so they centre on 1 when the model holds, plus
  EAP person measures with posterior SDs.
- `MFRMResult` gains `item_difficulty`, `item_se`, `item_infit`, `item_outfit`,
  `thresholds`, `threshold_se`, `sigma`, `sigma_se`, `theta`, `theta_se`,
  `n_parameters`, `n_observations`, `n_quadpts`, `aic`/`bic` properties and
  `summary()`; existing constructions remain valid.
- `fit_mfrm` rejects unidentified or inestimable designs with clear errors:
  unanchored facets, facets confounded with items or other facets, unused facet
  levels, extreme-only items or levels, unused categories, and an unidentified
  person SD.
- Many-Facet Rasch Models user guide and API reference entries.
- `MixedItemModel` for mixed-format tests whose items come from different
  families, for example 3PL multiple-choice and GRM constructed-response items
  on one latent trait. It has per-item category counts, component-qualified
  parameter names such as `"3PL.guessing"`, `MixedItemModel.from_itemtypes` and
  `parameter_items`, and supports scoring, simulation, information, item fit and
  CAT for calibrated mixed item pools.
- `MixedFormatEMEstimator`, which fits mixed-format models with one joint E-step
  and each component's own M-step (keeping the native polytomous and batched
  Newton paths). It supports per-component item priors, start values and fixed
  masks under qualified names, and observed-information standard errors
  (`oakes`, `crossprod`, `sandwich`) whose `vcov` covers all components jointly.
- `fit_mirt(model=[...])` accepts one family name per item, like R mirt's
  `itemtype` vector. A sequence that names a single family fits that family's
  ordinary model.
- Mixed-format fits work with `FitResult.coef()`, with NaN for parameters an
  item's family lacks, and with `summary()` and `vcov_labels`, which label rows
  by item name. They also work with `bootstrap_se`, `bootstrap_ci`,
  `parametric_bootstrap`, `bootstrap_lr`, `gen_random_pars` and
  `multi_start_fit`. `mod2values`, `extract_item`, linking,
  `transform_parameters`, `MultigroupModel`, `fixed_item_calibration`, and the
  single-family estimators (`EMEstimator`, `MCEMEstimator`,
  `WeightedEMEstimator`) raise `MirtModelError` for a `MixedItemModel`.
- EM M-steps include a conditional-maximization step for parameters shared by
  all items, so the marginal likelihood stays monotone. It has closed-form
  gradients for RSM and GRSM and keeps GRSM thresholds ordered.
  `WeightedEMEstimator` and `accelerate="squarem"` also cover the shared
  coordinates.
- Exact (Louis) observed information covers shared coordinates.
  `se_method="auto"` now gives observed-information standard errors and
  `FitResult.vcov` for RSM and GRSM fits, and `BLEstimator` uses the exact
  information for them too. The complete-data method and the numerical, forward
  and Richardson `compute_se` methods report shared-parameter standard errors.
- `fit_mirt(accelerate="squarem")` enables SQUAREM-accelerated EM. With it, a
  unidimensional 2PL fit skips the native full-EM fast path.
- `Prior.hess_log_pdf`, the second derivative of the log density: analytic for
  the built-in priors, with a central-difference fallback for custom priors.
- The equating guide documents fixed-item calibration (FIPC). The quickstart
  gains sections on DataFrame input and missing values, and on simulating from
  fitted models, plus a Next steps list. The estimation, model-syntax,
  mixed-format, uncertainty, DIF and multigroup guides now link to each other.
- API reference entries for `KernelEquatingResult`, `RatingScaleModel`,
  `GradedRatingScaleModel`, and the CAT tools `MCATEngine`,
  `ShadowTestSelection`, `SEChangeStop`, `MinInformationStop`,
  `PredictedSEReductionStop`, `assemble_form`, `assemble_parallel_forms`,
  `summarize_cat_simulation` and `CATSimulationReport`.
- Keyword-only `prior_mean`/`prior_cov` on `compute_itemfit`, `compute_s_x2`,
  `mirt.itemfit`, `compute_m2` and `compute_fit_indices`. Given a `FitResult`,
  M2 projects out the estimated latent correlations and variances like free item
  parameters, one degree of freedom each. An explicitly passed population is
  treated as known.
- `MHRMEstimator(compute_standard_errors=..., n_quadpts=...)`;
  `fit_mirt(estimation="MHRM")` forwards both.
- `fit_mirt(constraints=...)` and `EMEstimator(constraints=...)` hold a
  parameter equal across items, like `CONSTRAIN` in R's mirt. Each constraint is
  `{'parameter': ..., 'items': [...], 'column': ...}`, a `(parameter, items[,
  column])` tuple or an `EqualityConstraint`; items are zero-based positions or
  item names, and leaving them out ties every item. In each M-step, the items
  linked by a constraint are optimized jointly, with each tied group as one
  coordinate. This keeps EM monotone. SQUAREM also treats a group as one
  coordinate. The batched 2PL and native polytomous M-steps still update the
  untied items. Each group counts as one parameter in `n_parameters`, AIC and
  BIC. Standard errors use the constrained information `J' I J` for every
  `se_method`. Constraints are not available for mixed-format models or for
  estimation methods other than EM.
- `fit_mirt(spec=...)` now fits model-syntax `CONSTRAIN = (1-5, a1), (6-7,
  thresholds)` through `constraints` instead of raising `NotImplementedError`.
  Overlapping groups raise a validation error that gives the line number. Groups
  that equate different parameters, such as `(1, 3, a1, a2)`, still raise
  `NotImplementedError`.
- A regression test and docs show that `MixedFormatEMEstimator` estimates the
  shared thresholds of rating-scale components (RSM and GRSM).
- A `RuntimeWarning` when EAP scoring or `ability_posterior` of a bifactor model
  uses an automatically chosen product grid coarser than 11 points per
  dimension.
- `FitResult.to_dict()`/`to_json()` export the structure of multidimensional
  `fit_mirt(spec=...)` 2PL fits (`"MIRT"` with `loading_pattern`), `bfactor`
  fits (`"Bifactor"` with `specific_factors`) and mixed-format fits (`"Mixed"`
  with each component's family and items). Free-parameter restrictions are
  exported as `free_parameter_masks`. `from_dict()`/`from_json()` rebuild these
  models, restore `fixed`/`FIXED` masks and loading patterns for every family,
  and keep `latent_covariance`.
- FitResult.refit_recipe records the estimator class and settings of a fit:
  latent density, item priors, equality constraints, quadrature, acceleration
  and se_method. Worker threads, verbosity and standard errors are left to each
  refit. EM, mixed-format, bfactor and native 2PL fits record it, and it is not
  serialized.
- FitResult.latent_mean holds the mean of the fit's latent population. It is
  validated, serialized by to_dict/from_dict and printed by summary().
  EMEstimator reports the final mean and covariance of any non-standard Gaussian
  latent density, estimated or fixed, as latent_mean/latent_covariance.
- `lagrange` accepts a `FitResult` of a maximum-likelihood fit and runs a
  marginal score test. It frees the tested fixed parameters in a copy of the
  model and uses the marginal score and observed information at the constrained
  estimates (new keyword `n_quadpts`). `theta` is now optional and used only for
  bare models. Bayes modal fits are rejected. Previously this call raised
  `AttributeError`.
- compute_dif/mirt.dif anchors accept item names (DataFrame columns or Item_0,
  Item_1, ...).
- compute_m2, compute_fit_indices, compute_itemfit, compute_s_x2 and
  mirt.itemfit accept constraints=, the equality constraints of a fit_mirt fit.
  M2 projects one direction per tied group. S-X2/X2/G2/PV_Q1 degrees of freedom
  count a group of k tied coordinates as 1/k parameter per item.
- A multigroup_dif_likelihood_ratio case in the multigroup-fit benchmark suite.
- All-or-none ``item_bundles`` in fixed and parallel form assembly, including
  overlapping bundles, propagated required anchors, excluded-member handling,
  and joint content, security, cost, usage, and overlap constraints. Added
  independent exhaustive-search optimum checks and sparse-pool allocation gates.
- Independent numerical-integration and quadrature reference tests for scoring priors,
  posterior sampling moments, and BCa intervals, alongside archive identity
  failures exercised through the actual release-validator CLI.
- Fitted-model and fit-result inputs plus configurable quadrature resolution for
  imputation, supporting existing multidimensional and heterogeneous ordinal
  calibrations and completely missing items with a supplied calibration.
- True fixed-anchor and floating-anchor vertical calibration with physical-item
  maps, unequal forms, joint item/population estimation, native calibrated
  models, posterior scores, and constrained population-mean ordering.
- Native parameter-mask restrictions, preserved across model copies and used
  by EM estimation and diagnostic nuisance counts. Shared fixed coordinates
  retain their exact values across linked groups.
- Actual simultaneous curve matching for concurrent vertical scaling, with
  selectable reference grade and diagnostics based on the final joint solution.
- Optional respondent ``batch_size`` for Bayesian pointwise likelihoods and a
  ``pointwise`` timing/memory suite for 2PL, 3PL, and heterogeneous GRM models.
- SIBTEST minimum stratum sizes, explicit focal-group selection, crossing
  location, chi-square statistics, degrees of freedom, and retained stratum
  counts. Added analytical and repeated-simulation inference regressions.
- Required CPU PyTorch parity checks and release artifact version validation
  before publication. Documentation checks now include merge queues.
- Linked observed-score equating with reference-population probability masses
  and configurable theta batching. Added exact item and test Fisher information
  matrices for GPCM and PCM, including their use by MCAT selection.
- Independent mathematical and end-to-end regressions for equivalent
  calibrations, exhaustive response-pattern distributions, multidimensional
  parameter recovery, simulation frequencies, Fisher matrices, and CAT session
  recovery after invalid inputs.
- A `score-equating` benchmark suite reporting timing and traced allocation
  peaks for dichotomous and heterogeneous ordinal forms.
- Added optional `mp_context` to cross-validation and regularization selection
  so callers can choose their multiprocessing context without global changes.
- Added an `adaptive-scoring` timing and traced-memory benchmark suite for
  fixed-history CAT/MCAT scoring across one to three dimensions.
- Enabled existing tensor likelihood kernels for 1PL and PCM estimation, and
  added optional per-item `n_categories` metadata to GRM/GPCM GPU kernels.
  Added a `gpu-likelihood` benchmark suite with device and runtime metadata.
- Optional diagonal complete-data standard errors for MCEM, QMCEM, and
  stochastic EM via `compute_standard_errors=True`, with configurable
  `se_step_size`. Used exact polytomous diagonal curvature, bounded analytic
  logistic gradients, and shared-grid counts, with numerical custom-objective
  fallback, protected fitted parameters,
  boundary-aware stencils, and zero errors for fixed coordinates. Added
  uncertainty stages to the Monte Carlo benchmark suites. Defaults retain
  placeholder errors and existing fitting costs.

### Changed

- `item_optim_ftol` is documented as a relative tolerance that is tightened to
  at most 1e-10 for the native GRM M-step and under SQUAREM.
- `fit_mirt(estimation='MHRM')` now uses `burnin=min(500, max(max_iter // 4,
  1))`, mirroring the Gibbs branch. Default fits on both backends therefore
  average post-burn-in iterates. On the NumPy path (non-2PL models) this also
  fills `standard_errors` with chain standard deviations, where it was empty
  before. Native MHRM estimates change for all users; before this they were
  effectively unfitted.
- The native MHRM path now reports the log-likelihood, and so AIC/BIC, at MAP
  abilities, as the NumPy path does. Information criteria no longer depend on
  the backend.
- `BayesianMCAT` is documented as, and implemented as a subclass of,
  `AOptimality`; its criterion was already identical.
  `KullbackLeiblerMCAT(n_integration_points=...)` is deprecated and ignored.
- `CATEngine.run_batch_simulation` and `compute_conditional_mse` take
  `vectorized=True`. Configurations that cannot use Rust but meet the lock-step
  conditions now run in lock step. Lock-step results include full theta, SE and
  information histories. Seeded results equal the sequential loop for
  fixed-length tests and differ in which responses are drawn when tests end
  early. Pass `vectorized=False` for the previous per-session stream.
- MFI and MEI selection break ties by the lowest item index regardless of set
  order.
- Itemfit, personfit, compute_itemfit and compute_personfit raise
  MirtValidationError (a ValueError) for unknown or empty statistic names
  instead of silently ignoring them. A single name string is accepted, and the
  'lz' alias remains.
- The EAPsum implementation module moved from `mirt.scoring.eapsum` to the
  private `mirt.scoring._eapsum`. Deep imports such as `from mirt.scoring.eapsum
  import EAPSumScorer` must become `from mirt.scoring import EAPSumScorer` (also
  `eapsum`, `sum_score_to_theta`).
- `sum_score_to_theta` raises `MirtValidationError` for non-integer, non-finite
  or out-of-range sum scores instead of silently clipping or truncating them
  (integral floats such as 3.0 are still accepted). It now accepts a `FitResult`
  and `prior_mean`/`prior_cov`, and returns arrays with the input's shape.
  `eapsum` also accepts a `FitResult`.
- The default EAP grid (`fscores`, `EAPScorer`, `ability_posterior`) now depends
  on the number of factors: 49 points per dimension for 1-2 factors (results
  unchanged), 21 for 3, 9 for 4, 7 for 5 and 5 for 6 or more. Four or more
  factors were previously impractical (49**4 is 5.7M nodes).
  Three-or-more-factor EAP scores change, including diagnostics that score
  internally with the default. In 3D, theta moved by up to about 1e-3 (2PL) and
  2e-2 (30-item, 4-category GRM) compared with 49 points. Pass `n_quadpts`
  explicitly for a finer grid. Explicit grids above 21**5 nodes emit a
  `RuntimeWarning`.
- `equipercentile_equating` and `observed_score_equating` use the standard Kolen
  & Brennan (2014) percentile-rank inverse instead of interpolating between
  percentile-rank midpoints. Low and high ranks are no longer clamped to [0, K],
  outputs range over [-0.5, K_Y + 0.5], zero-probability new scores are skipped,
  and tail precision is preserved. Observed-score conversion tables change
  numerically.
- `delta_method_se` for Stocking-Lord, TCC and Haebara links between 1PL-5PL
  forms uses implicit-function derivatives of the fitted criterion instead of
  finite differences of Nelder-Mead re-fits. SEs shift by up to about 1e-3
  relative, which was the optimizer noise in the old values. It is about 15-19x
  faster.
- fit_mirt, fit_multigroup and vertical-scale calibration build models through
  one shared internal factory, so factor and category validation is the same
  everywhere.
- validate_responses, and the entry points that use it (fit_mirt,
  fit_multigroup, the estimators and others), treat NaN as a missing response.
  This includes R's NA and pandas nullable Int64/boolean/Float64 pd.NA. Pass
  nan_as_missing=False to restore rejection. Infinite, non-integer and text
  codes are still rejected. fscores, itemfit and personfit still require missing
  responses coded as negative numbers.
- FullDiagnosticReport's item-fit table now also flags the 0.8-1.2 'Check' tier,
  matching ItemAnalysisReport.
- ScoreResult.classify and AbilityPosteriorResult.classify share cut-score and
  confidence validation, with identical error messages ('strictly between 0.5
  and 1').
- mirt.models now exports every model, adding GradedRatingScaleModel,
  ZeroInflated2PL/3PL, HurdleIRT, the unfolding models, MixtureIRT,
  DINA/DINO/BaseCDM, UnipolarLogLogistic, ThreeParameterLogisticUpper, fit_cdm
  and fit_mixture_irt. The top-level mirt namespace now exposes every concrete
  item model. New top-level names include FiveParameterLogistic,
  ComplementaryLogLog, NegativeLogLog, the sequential, compensatory,
  nonparametric and explanatory models, GDINA/HigherOrderCDM and the testlet
  variants. mirt.LLTM is still the mixed-effects LLTM.
- CAT and MCAT stopping rules share one implementation for validation,
  combination and stability tracking; unused internal trigger flags were
  removed.
- `NonlinearGrowthModel.initial_value` is documented as reserved and unused by
  the growth curves.
- All utility `n_jobs` arguments share one validator; `cross_validate` now
  raises `MirtValidationError` (still a `ValueError`) for invalid `n_jobs`.
- Bootstrap, posterior predictive checks and `DichotomousItemModel.simulate`
  share one response simulator (seeded streams unchanged); posterior predictive
  checks now raise `MirtModelError` (a `ValueError`) for invalid model
  probabilities, and model simulation rejects binary probabilities outside [0,
  1].
- `combine_plausible_values` and `averageMI` share one Rubin pooling
  implementation (results unchanged; each keeps its degrees-of-freedom
  convention when the between-imputation variance is zero), and clinical and
  confidence-interval utilities share argument validators; `delta_method` and
  `PLCI` now report empty and non-finite vector inputs with separate messages.
- In `ResponseTimeGibbsSampler`, items with no observed response times now draw
  their time parameters from the prior instead of staying at their initial
  values. Seeded response-time fits change.
- `StateSpaceIRT` predictive probabilities, log-likelihoods and diagnostics
  share a single quadrature kernel, about 1.15-1.25x faster for
  `predictive_log_likelihood_batch` and `predictive_diagnostics_batch`.
- MHRMEstimator, GibbsSampler, GVEMEstimator and SparseBayesianEstimator now
  raise MirtValidationError for models with set_free_parameter_masks
  restrictions. Before, they silently estimated the fixed parameters.
- MCEMEstimator's default tol is now 1e-3 and bounds a 50% confidence interval
  for the log-likelihood change between iterates. QMCEMEstimator and
  StochasticEMEstimator keep their plain rule and defaults. Seeded MCEM results
  and iteration counts change, and memory grows with the sample size.
- multi_start_fit starts each fit from its random values with start="model"
  instead of a private flag. gen_random_pars keeps coordinates fixed with
  set_free_parameter_masks; an item whose random graded thresholds would lose
  their order around a fixed threshold keeps its current thresholds.
- DIF `effect_size` is now the signed focal-minus-reference item location on the
  common scale (positive means harder for the focal group). It was previously an
  unsigned difference across unlinked scales. A missing p-value now gives ETS
  class A, except for the descriptive `raju` method.
- `plot_dif` and `DIFAnalysisReport` accept untested items with NaN effect
  sizes, such as DIF anchors, and draw them at zero height.
- `compute_dif` validates `model` (1PL, 2PL, 3PL, GRM or GPCM) and `scheme`.
  `compute_dtf` now rejects missing float labels in object-dtype groups, like
  DRF. `DIFAnalysisReport` infers its method from the DIF results.
- EM fits of unidimensional 1PL-4PL, GRM, GPCM and PCM models (`fit_mirt` and
  `EMEstimator`) now report observed-information (Oakes/Louis) standard errors
  by default instead of itemwise complete-data curvature. The old default
  understated SEs: on a 2PL, empirical SD/SE was about 1.4 and 95% interval
  coverage about 82%; GRM slope coverage was about 50%. Nominal,
  multidimensional, custom and survey-weighted fits keep complete-data SEs by
  default.
- `draw_parameters()` on a bare model without a covariance emits a
  FutureWarning; its fixed 0.01*I fallback will be removed. An explicit
  discrimination/difficulty covariance no longer adds fabricated noise to
  asymptote parameters.
- The `damping_ab`/`damping_c` arguments of
  `mirt.backends.rust.em_iteration_3pl` are deprecated and ignored; passing them
  emits a DeprecationWarning. Its M-step is now a bounded Fisher-scoring ascent
  (projected Newton with an epsilon-active set) with a backtracking line search.
- Native multigroup 2PL posteriors now use exact log-sigmoid probabilities
  instead of clipping at 1e-10, matching the single-group kernels. Values differ
  only for extreme logits (|z| > 23).
- A unidimensional-only model now raises `MirtModelError` ('<model> only
  supports unidimensional models') for any `n_factors` other than 1, and stores
  `n_factors` as the plain int 1 when given `1.0`, `np.int64(1)` or `True`. A
  non-numeric `n_factors` raises `MirtValidationError` instead of a `TypeError`.
  1PL, 3PL, 4PL, 5PL, ULL, CLL and NLL used to raise a plain `ValueError` with a
  model-specific message. `MirtModelError` is still a `ValueError`.
- `BKTGibbsSampler` input-validation errors are now `MirtValidationError` (still
  a `ValueError`, same messages). Its result summaries (learning curves,
  mastery, AIC/BIC) come from a vectorized helper shared with `fit_bkt_em`, and
  the numbers are unchanged.
- `MCEMEstimator`, `QMCEMEstimator` and `StochasticEMEstimator` raise
  `MirtModelError` for models with free shared parameters, where they used to
  leave them silently unchanged. To use these estimators, fix those parameters
  with `set_free_parameter_masks`.
- Bayes-modal (item-prior) EM standard errors now include the prior's curvature.
  The oakes and sandwich bread, crossprod and complete-data errors describe the
  log-posterior at the mode, so they are smaller than the likelihood-only errors
  reported before.
- The MHRM `burnin` cycles now take unit-gain steps, and the `gain_sequence`
  schedules restart counting at the end of burn-in. Seeded MHRM results differ
  from earlier releases; the native 2PL kernel is about 1.6x slower per cycle.
- GibbsSampler(n_chains=k) now runs and stacks k chains (seeded seed + 1000*i)
  even without parallel_chains, on both the NumPy and the native 2PL paths. The
  native kernel previously ignored n_chains. n_chains must be a positive
  integer, and unseeded multi-chain runs no longer always use seed 0.
- The README feature list, examples and API tables cover the new estimation,
  scoring, diagnostics, DIF, CAT, equating and model features. New examples
  cover: observed-information standard errors and `vcov`; SQUAREM; starting
  values, fixed parameters and priors; `mirt_model` with factor correlations;
  mixed formats; `bfactor`; rating-scale fits; shadow tests, the new stopping
  rules and simulation reports; `multigroup_dif`; kernel equating and FIPC;
  `fit_bkt_em` and `fit_mfrm`; `bootstrap_lr`; plausible values under an
  estimated population; and DataFrame input with `FitResult.from_json`.
- `examples/equating.py` and the README now use `mirt.equating.link`, kernel
  equating and `fixed_item_calibration` instead of the deprecated `equate` and
  the legacy `fixed_calib`.
- `compute_residuals`, `analyze_residuals`, `compute_outfit_infit`,
  `identify_misfitting_patterns`, `compute_q3`, `compute_ld_chi2`,
  `compute_ld_statistics`, `compute_m2`, `compute_fit_indices`,
  `compute_itemfit`, `marginal_rxx` and `conditional_rxx` accept a `FitResult`
  as well as an item model. `conditional_rxx(latent_variance=None)` defaults to
  the fit's estimated variance, or to 1.
- For constrained fits, `FitResult.vcov` repeats each tied group's row and
  column for every tied coordinate, with the usual per-coordinate `vcov_labels`.
  Tied coordinates therefore have equal standard errors and correlation 1.
- `compute_personfit` now validates response codes like `fscores` does, so codes
  outside an item's category range raise `ValueError`.
- `fscores(method="EAP")` scores `BifactorModel`s, including `bfactor()`
  results, by Gibbons-Hedeker dimension reduction (one general-by-specific grid
  per specific factor) when the prior keeps the specific factors conditionally
  independent. In that case the default is 49 points per dimension for any
  number of factors. Each result equals the product-grid EAP at the same
  `n_quadpts` to rounding error. The old coarse defaults erred by up to about
  0.3-0.46 with six factors.
- Binary items of a `MixedItemModel` are simulated by the binary rule (1 when
  the uniform draw is below p), so a one-component pool reproduces its
  component's seeded `simulate()`. Seeded simulations from mixed models with
  dichotomous components differ from earlier releases.
- Documentation updates. The mixed-format guide covers prior curvature in SEs,
  exact observed information for RSM/GRSM components, `coef()` limits for shared
  parameters, the binary simulation rule, JSON round trips, BL/`compute_se`
  support and single-family sequences with spec/multigroup. The results guide
  lists the newly rebuildable models. The model-syntax Limitations note that a
  per-item model sequence must name one family.
- Standard-error docstrings, the `BLEstimator` notes and the uncertainty guide
  now name the model families with exact Louis information, the families that
  `se_method='auto'` and MH-RM treat as closed-form, and the O(P^2) cost for
  custom, testlet and mixture models.
- Graded-threshold ordering (the 1e-6 gap, constraint and projection) is defined
  once in `mirt.estimation._graded_order` and shared by EM, MHRM, SQUAREM and
  the shared-parameter step.
- multigroup_dif and select_dif_anchors default to p_adjust='none', matching
  mirt.dif/compute_dif and R's mirt::DIF, so both entry points flag the same
  items.
- Reused repeated response-pattern likelihoods when retaining ability
  posteriors, with bounded expansion scratch space and respondent-order outputs.
  Sparse built-in assembly pools evaluate only eligible item curves; dense
  pools retain vectorized evaluation and custom callbacks remain authoritative.
- Streamed mergeable BCa jackknife moments instead of retaining every omitted
  person's full score vector. Limited BLAS/OpenMP thread pools during CI and
  release tests to avoid nested parallelism and unstable performance comparisons.
- Bounded respondent/item/category scratch space in Bayesian likelihoods and
  reused fixed or sampled posterior inputs through broadcast and strided views.
  Native SIBTEST shares matching totals and accumulates score strata in one
  pass instead of repeatedly scanning and copying group response matrices.
- Bounded score-recursion memory with theta batches and reused contiguous work
  buffers. Streamed CAT/MCAT conditional-error moments instead of retaining
  every simulated estimate.
- CI builds its tested wheel from the source distribution. Release validation
  tests rebuilt source distributions and every published wheel architecture,
  including Intel macOS and Python 3.14, using locked dependencies and the full
  regular test suite. Local Make targets preserve the installed package with
  `uv run --no-sync`.
- Python security checks audit all locked dependencies across optional extras
  and platform markers, retain their findings, and fail on collection errors.
  Updated locked urllib3 to 2.8.0 and replaced yanked NumPy 2.4.0 with 2.4.6.
- Automatic version bumps synchronize the runtime module, Cargo manifest, and
  versioned package lock entries, refusing inconsistent inputs before edits.
- Reused adaptive EAP quadrature grids and evaluated only administered item
  curves for independent built-in binary models, with bounded probability and
  covariance scratch space. Current parameters are reevaluated on each update;
  custom models retain general scoring. Native simulation now adds each new
  response's likelihood once rather than reevaluating the entire history.
- Removed redundant new-form curve evaluation during linked true-score equating
  and prepared serial cross-validation training folds lazily.
- Reused binary likelihood response coefficients and result buffers to reduce
  temporary NumPy allocations. Exceptional curves and large responses use
  bounded item sums. Added a `binary-likelihood` timing and memory suite for
  matched, shared, and grid evaluations across ten binary model families.
- Reduced 2PL, 3PL, multidimensional, and complete GPU E-step likelihoods with
  shared matrix operations, avoiding person/grid/item intermediate tensors
  while preserving missing and general numeric responses.
- Consolidated built-in model eligibility checks across prepared and native
  estimation paths, and reused prepared gradients for bifactor Monte Carlo
  item uncertainty.
- Reused fresh Monte Carlo E-step evidence for likelihood reporting, avoiding
  repeated draw evaluation and importance normalization before item updates.
  Released previous draws and weights before the next E-step, and transformed
  importance draws in owned storage. Preserved custom estimator and model
  likelihood/validation hooks. Added reported E-steps to `mcem-fit` and
  `qmcem-fit` benchmarks.
- Bounded Gaussian prior solves and reductions during posterior MCEM and
  stochastic EM. Used diagonal scaling or triangular solves for Cholesky
  factors, reused proposal buffers, released temporary normal draws, and
  updated accepted cells without copying accepted draw arrays. Extended the
  `mcem-sampling` benchmarks with prior kernels and correlated priors.
- Bounded person-specific Monte Carlo likelihood evaluation without repeating
  response matrices. Reduced binary item curves against response blocks and
  gathered ordinal categories from bounded item curves, preserving public
  probability callbacks and custom likelihood/validation overrides. Converted
  sample inputs only within bounded blocks. Extended `mcem-fit` with likelihood
  refreshes and added an `mcem-sampling` benchmark for posterior MCEM and
  stochastic EM.
- Aggregated QMCEM M-step counts on its shared ability grid, avoiding repeated
  respondent/sample probability evaluation and sample-array copies during item
  optimization. Reused analytic logistic, affine, and polytomous gradients and
  shared bounded category-count accumulation while preserving custom curves,
  missing responses, and loading constraints. Refreshed likelihoods directly on
  the shared grid without expanding responses or samples. Added `qmcem-mstep`
  and `qmcem-fit` timing and traced-memory benchmark suites.
- Shared analytic item kernels with person-specific MCEM, QMCEM, and stochastic
  sample fitting. Cached small observed sample blocks and streamed bounded
  larger blocks without retaining full per-item sample copies. Preserved custom
  models and item objectives on their numerical path. Added a `mcem-fit`
  timing and traced-memory benchmark covering six model families.
- Reused IRTree response preparation across E/M-steps and uncertainty, bounded
  node/row likelihood scratch space, and normalized owned likelihood buffers
  in place. Released previous posteriors before the next E-step and reused the
  final posterior on convergence. Batched bounded node information matrices,
  avoiding unused correct-count calculations during uncertainty. Shared the
  clipped logistic objective with ordinary item fitting. Added an `irtree-fit`
  timing and traced-memory benchmark suite.
- Reused clipped item-gradient kernels for joint BL marginal optimization of
  built-in 1PL–4PL, GRM, GPCM, PCM, NRM, MIRT, and bifactor models. Shared
  response preparation with diagonal marginal curvature, bounded likelihood
  scratch space, and isolated trial model state. Added a `bl-fit` timing and
  traced-memory benchmark suite. Custom likelihoods retain numerical optimization.
- Shared numerical item curvature across EM and standalone standard errors.
  Prepared bounded posterior counts once per uncertainty call, reused them
  across Richardson step sizes, and used one worker pool for all item fields.
  Fixed and padded coordinates avoid unnecessary probability trials. Added an
  `item-curvature` timing and traced-memory benchmark suite.
- Prepared analytic GRM, GPCM, PCM, and NRM item objectives for Python EM and
  survey-weighted EM, including multidimensional models. Category counts use
  bounded accumulation, and trial evaluation leaves model parameters intact.
  Added a `polytomous-fit` timing and traced-memory benchmark suite.
- Accumulated survey-weighted binary and category counts in bounded blocks,
  eliminating dense weighted posterior copies from optimization and standard
  errors. Weighted EM now shares core item objectives and analytic gradients;
  fixed items and items with no weighted observations retain their parameters.
  Failed numerical trials restore parameter state. Added a `weighted-mstep`
  timing and traced-memory benchmark suite.
- Reused response preparation across regularized E/M-steps, bounded response
  and probability scratch space, and removed unused item masks. Shared in-place
  posterior normalization across ordinary, survey-weighted, and regularized EM.
  Added a `regularized` timing and traced-memory benchmark suite.
- Prepared Python 1PL–4PL item objectives once per optimization and consolidated
  their likelihood and gradient calculations. Added a `logistic-fit` benchmark
  suite covering unidimensional models and multidimensional 2PL.
- Prepared analytic M-step objectives for MIRT and bifactor models, sharing
  quadrature design arrays and expected counts without mutating trial model
  parameters. Added a `multidimensional-fit` benchmark suite including
  confirmatory constraints and missing responses.
- Shared bounded full, single-item, and paired probability evaluation for MIRT
  and bifactor models. Reused owned logit buffers on the ordinary logistic path
  and preserved sparse bifactor loading access. Added a
  `multidimensional-probability` timing and memory benchmark suite.
- Shared stable logistic variances and bounded multidimensional information
  reductions. Test matrices contract only active factor pairs, with a fast path
  for single-person item matrices used by adaptive selection. Added a
  `multidimensional-information` timing and memory benchmark suite.
- Shared bounded full, single-item, and paired probability evaluation across
  1PL–4PL models. Added a `logistic-probability` benchmark suite covering all
  query forms and multidimensional 2PL models.
- Shared bounded query handling across ULL, CLL, and NLL, and reused the symmetric
  sigmoid derivative in unipolar curves and logistic information. Added a
  `unipolar` benchmark suite for probability, information, and paired queries.
- Shared bounded information evaluation across 1PL–4PL models, reusing buffers
  and keeping multidimensional slope norms factored during evaluation. Added a
  `logistic-information` benchmark suite for full and single-item queries.
- Shared CLL/NLL probability and information kernels with bounded evaluation
  and in-place buffers. Extended the `asymmetric` benchmark suite to both links.
- Shared bounded 5PL probability and information evaluation with in-place
  buffers, restricting log-domain reductions and overflow recovery to affected
  cells. Added an `asymmetric` benchmark suite for full and paired probabilities
  and information.
- Shared ability validation and bounded test-information reductions between
  information and reliability utilities. Streamed item-information and expected
  scores, evaluated sparse item selections directly, and reused repeated item
  queries. Added a `curves` timing and traced-memory benchmark suite.
- Bounded information evaluation for reliability and measurement-error summaries
  by cohort blocks. Added a `reliability` benchmark suite with binary and ordinal
  models, including timing and traced-memory measurements.
- Shared binary item and total-score moments in classical statistics, using
  bounded, centered item-total correlation blocks and prepared NumPy deletion
  statistics. Removed redundant response copies and reused variance buffers.
  Added a `classical` benchmark suite for both correlation modes and missing data.
- Streamed empirical RMSEA probability evaluation and binned residual reductions,
  removing the dense person-by-bin matrix and unused bin-mean calculations.
  Sparse histograms retain only occupied bins within each block. Added an
  `empirical` benchmark suite for binary and polytomous fit with 10 or 100 bins.
- Calibrated kernel-smoothed item curves in bounded grid blocks with in-place
  Gaussian buffers, restricting missing-item fallbacks to affected grid points.
  Added a `kernel-smoothing` benchmark suite for weighted complete/missing data
  on standard and dense grids, with timing and traced-memory measurements.
- Shared Gaussian smoothing between calibration and empirical `itemGAM` curves,
  reusing kernels across items and bounding temporary memory for both means and
  uncertainty bands. Extended the smoothing benchmarks to binary and polytomous
  empirical curves with complete and missing responses.
- Batched built-in survey-weighted EM likelihoods over quadrature points and
  retained log marginals through convergence and fit statistics, avoiding
  likelihood underflow on long tests. Reused posterior normalization storage
  and shared item-curvature calculations for weighted standard errors,
  preserving fixed-parameter masks and avoiding repeated finite differences.
  Added a `weighted-em` timing and traced-memory benchmark suite.
- Shared Gaussian variational E-step calculations between GVEM's NumPy
  fallback and sparse Bayesian estimation. Respondent blocks bound temporary
  arrays while preserving missing-data bounds, shifted priors, and each
  estimator's lower bound on local variational parameters. Small samples use
  one block without copying outputs. Added a `variational` timing and memory suite.
- Shared bounded M-step statistics between NumPy GVEM and sparse Bayesian fitting,
  reusing weighted means for intercept updates and batching GVEM item solves.
  Fixed loadings skip covariance calculations. Added a `variational-mstep`
  benchmark suite for free- and fixed-loading updates and traced allocations.
- Shared the NumPy variational objective between GVEM and sparse Bayesian
  estimation, bounding likelihood and Gaussian KL scratch space by respondent
  blocks and reusing temporary arrays. Added a `variational-objective` benchmark
  suite for timing and traced allocations on precomputed variational states.
- Computed built-in GVEM standard errors directly from diagonal variational
  curvature at the default step size, avoiding full-objective finite differences
  and cancellation against parameter-independent terms. Curvature reductions
  use bounded row blocks. Custom objectives and nondefault steps retain a shared
  numerical fallback that restores parameters on failed trial evaluations.
  Added a `gvem-uncertainty` benchmark suite for uncertainty and complete fits.
- Streamed model-fit moments in bounded response and probability blocks, sharing
  pairwise counts and reusing observed moments for the independence baseline.
  Complete blocks avoid missing-data matrix products. Added a `model-fit`
  benchmark suite for empirical/quadrature integration and traced memory peaks.
- Shared bounded-memory LD chi-square/G² calculations between local-dependence
  diagnostics and the NumPy backend fallback, reusing transposed cells and
  complete-block observed margins to reduce matrix products. Extended the
  `diagnostics` benchmark suite with complete/missing-data LD workloads.
- Streamed misfit identification in bounded probability blocks, retaining only
  flagged responses and fit totals instead of a full residual analysis and
  unused response-pattern summaries. Added complete/missing-data 2PL and GRM
  timing and traced-memory workloads to the `misfit` benchmark suite.
- Consolidated pairwise residual correlations across diagnostics, testlet models,
  and NumPy backend fallbacks into one bounded-memory kernel. Complete-data blocks
  use a single matrix product, and column offsets improve numerical stability.
- Added a `diagnostics` benchmark suite for complete/missing-data Q3 timing and
  traced Python/NumPy memory peaks.
- Shared bounded WAIC/PSIS predictive-density reductions, reusing exponential
  buffers and centering variance calculations for large log-likelihood shifts.
  PSIS partitions its tail cutoff, sorts the selected tail once, and bounds
  queued thread tasks. Added Bayesian diagnostic timing and memory workloads.
- Kept EM convergence and final likelihoods in log space to avoid underflow on
  long tests, and reused built-in likelihood buffers for posterior normalization.
- Batched generic EM expected counts with bounded matrix products and cached
  response preparation; reused Python and Rust worker pools within each fit.
- Retained response-pattern compression through native 2PL standard errors,
  weighting posterior contributions by pattern frequencies.
- Made the legacy Rust-backend namespace lazy and routed internal imports to
  their owning backend modules.
- Cached bounded item/category probability tables in native likelihood and E-step
  kernels, with contiguous output storage, in-place posterior normalization,
  zero-copy posterior transfers, and GIL release during native fitting.
- Added batched Rust GRM/GPCM/PCM M-steps with analytic gradients and bounded
  optimization, and native unidimensional 1PL–4PL MAP/ML scoring.
- Compressed repetitive EM data into frequency-weighted response patterns while
  preserving sample counts, latent-density updates, and uncertainty objectives.
- Streamed marginal-information perturbations to remove quadratic person-array
  caching; built-in unidimensional EM models use analytic item curvature at the
  default SE step setting to avoid finite-difference cancellation.
- Added likelihood, optimization, and information benchmark suites, including
  traced information-memory peaks and BLAS/Rayon thread metadata.
- Replaced per-point Gaussian covariance tensors with weighted matrix products
  and reused owned point buffers during updates and log-density evaluation.
  Added a `latent-density` benchmark suite for timing and traced memory.
- Accelerated pairwise availability counts with bounded BLAS matrix products
  and exact integer accumulation.
- Shared memory-bounded category counting between imputation and item statistics.
- Added a `data` benchmark suite for counts, imputation, and item statistics.
- Shared response-pattern grouping between ability scoring and collapse utilities,
  with Rust hashing over compact integer rows and GIL release during grouping.
- Reduced copies and sorting work in the NumPy pattern-grouping fallback while
  preserving first-appearance order and full-width response codes.
- Added a `patterns` benchmark suite for repeated and mostly distinct responses.
- Shared exact row-wise CDF searches across posterior quantiles, intervals, and
  sampling, with Rust acceleration and batched reproducible draws.
- Moved shortest-interval selection to a linear Rust scan with a reusable grid buffer.
- Bounded highest-density interval working memory by the full joint grid and
  removed redundant marginal reductions when all coordinates are unique.

### Performance

- Polytomous log_likelihood_batch now uses one sparse one-hot product instead of
  per-item gathers. This covers the Python path for multidimensional GRM/GPCM,
  NRM, RSM, GRSM, nested logit, GGUM, sequential/adjacent-category and custom
  polytomous models. Micro-benchmarks are roughly 2-12x faster for
  multidimensional and nominal models, with bit-identical results except
  sequential/adjacent models (within 5e-13).
- Structural parameter masks and canonical parameter values for GRM, GPCM, NRM,
  nested logit and GGUM models are vectorized. This removes an O(J^2) cost from
  the generic per-item M-step; at J=200, nested-logit masks and canonical values
  are 15-20x faster.
- Iteminfo/testinfo and model.information(theta) evaluate all items at once for
  small ability batches (at most 256 points) for GRM, GRSM, GPCM, PCM, RSM and
  NRM. For one theta and a 300-item pool this is about 70x faster or more.
  Sequential and adjacent-category models compute each item once in iteminfo,
  and polytomous test information is computed in bounded memory for long
  batches.
- Built-in 1PL/2PL items (any dimension, no parameter restrictions) are fitted
  in one batched Newton M-step instead of one scipy optimization per item.
  EMEstimator 2PL 2000x30 drops from about 0.6 s to about 0.06-0.17 s with the
  same 31 iterations. 2-D 2PL needs 59 instead of 66 iterations and runs roughly
  1.5-2x faster. The fixed point is unchanged. Items whose estimate leaves the
  parameter box still use the bounded optimizer.
- The generic polytomous M-step (multi-D GRM/GPCM, NRM, `use_rust=False`)
  computes all items' category counts with blocked one-hot products, about 3-6x
  faster per M-step, and the results are bit-identical.
- The native likelihood table is stored category-major, so each response adds
  one contiguous grid row. 2PL/3PL/GRM/GPCM and multidimensional log-likelihood
  kernels are about 2-4x faster, and `e_step_complete` about 2-2.5x. Results are
  bitwise identical.
- Native expected counts are gathered from contiguous posterior rows through one
  shared helper, and `em_fit_2pl` and `bootstrap_fit_2pl` share one EM
  implementation. Single-threaded, `em_fit_2pl` (the default 2PL fit) is about
  1.5-2x faster, `bootstrap_fit_2pl` about 1.8x, `em_iteration_3pl` 2.3-2.9x and
  `m_step_polytomous` about 1.3x. Results are bitwise identical and independent
  of thread count.
- `bootstrap_fit_2pl`, `gibbs_sample_2pl`, `mhrm_fit_2pl` and `fixed_calib_em`
  release the GIL while they run, so other Python threads keep running.
- Lock-step batch simulation is roughly 45-90x faster for 3PL and 88-204x faster
  for GRM (N=2000, 300-item pool, up to 30 items).
- MEI selection uses a fixed number of model calls per step for dichotomous
  banks and O(candidates + administered) calls for polytomous banks. A 30-item
  2PL MEI CAT on a 300-item pool is about 21-23x faster per examinee; GRM is
  about 3.5-4.5x faster.
- MCAT computes information matrices for all candidates at once, including the
  default `MultidimensionalModel` through one logit evaluation. D, A and C
  criteria are evaluated over the whole candidate stack. D-optimality MCAT
  engines are roughly 2.5-10x faster for MIRT and 10-15x faster for
  multidimensional 2PL; the candidate criteria themselves are up to about 100x
  faster, including KL.
- S-X2 builds leave-one-out score distributions from prefix and suffix
  recursions with one matrix product per item, and vectorises sparse-cell
  pooling. It is about 3-14x faster, for example GRM5 with 60 items and 5000
  persons 587 to 43 ms, and 120 items 5.7 to 1.3 s.
- M2 and the fit indices whiten well-conditioned moment covariances with a
  guarded Cholesky factor instead of an eigendecomposition. Results are
  unchanged (M2 and df identical, indices within 3e-12 relative). For example, a
  40-item 2PL compute_fit_indices goes from 790 to 220 ms.
- Unidimensional MAP, ML and WLE scoring for models without a compiled scorer
  (GRM, GPCM, PCM, NRM, RSM, 1PL/3PL/4PL WLE, and others with standard
  likelihoods) runs one row-batched bounded Brent search over all response
  patterns instead of a SciPy search per pattern. It follows SciPy's steps
  exactly: theta is identical or within about 1e-10. For 2000x30 polytomous data
  it is about 90-400x faster. `n_jobs` has no effect on this path.
- Multidimensional MAP scoring for built-in models (2PL/MIRT, bifactor, GRM,
  GPCM, PCM, NRM) uses a row-batched projected Newton search with vectorized
  finite-difference standard errors. Unconverged rows fall back to per-pattern
  L-BFGS-B. It is about 30-85x faster for a 2-factor 2PL. Estimates are
  typically within 1e-4 of the previous L-BFGS-B results; where they differ
  more, the new estimate has a higher posterior density. Noncompensatory and
  custom models keep the per-pattern L-BFGS-B search.
- EAPsum caches the quadrature grid and item probability tables per model state
  and runs the sum-score recursion in probability space, with an automatic
  log-space fallback for scores that could underflow. Scoring data with many
  missing-item patterns is about 2-12x faster, with results equal to within
  1e-15.
- Expected-score curves are computed in one vectorized pass over items (about
  1.8x faster for a 150-item 3PL or a mixed-category GRM), and polytomous
  category probabilities are validated in one pass.
- Stocking-Lord, TCC and Haebara linking use Levenberg-Marquardt least squares
  with an analytic Jacobian for 1PL-5PL forms. They fall back to Nelder-Mead
  when LM fails, stalls on a flat plateau or does not improve on the start, and
  agree with the previous optima to within 3e-8. Single links are about 3.5-6x
  faster and `link(compute_se=True)` bootstraps about 1.7-2.6x faster.
  `convergence_info` gains an `optimizer` key ('least_squares', 'nelder_mead',
  or 'none' when the start is already exact).
- `concurrent_link` uses a separable finite-difference gradient that
  re-evaluates only the perturbed form, plus batched curve evaluation for dense
  anchor selections. Results are bit-identical to before, and it is about 15x
  faster at 6 forms and 37-45x at 10 forms (for example, 4.2 s -> 0.11 s).
- `GrowthMixtureModel.class_log_likelihood`, `posterior_probabilities`,
  `fit`/`fit_em`, `predict_trajectories` and `predict_trajectory_moments` handle
  missing occasions with per-person 2x2 Woodbury/Sylvester algebra on the random
  intercept/slope basis. They no longer loop over unique missingness patterns.
  At 2000 persons with 15% missing, `fit` is about 25x faster for 10 occasions
  (1.6 s to 66 ms) and about 49x faster for 20 occasions (7.0 s to 141 ms).
- `BKTModel.forward_backward_batch` smooths person-specific skill layouts in one
  vectorized chronological pass over every learner's per-skill chains. This
  replaces per-learner Python smoothing and per-layout calls for one-off
  layouts. Layouts repeated across at least 32 learners still use the compiled
  batch kernel. On 2000 learners x 30 trials with all-distinct layouts it takes
  about 5 ms instead of about 30 ms (compiled backend) or about 450 ms (NumPy
  backend). The shared-layout NumPy fallback is about 2-3x faster.
  Zero-likelihood (degenerate slip/guess) chains keep their existing semantics.
- `impute_responses(method='EM')` with a model name fits the model once on the
  observed responses instead of up to 10 refits on mode- or random-filled data
  (about 10x faster on 2PL 2000x30); seeded EM imputations change, and a failed
  calibration now warns before falling back to empirical draws.
- `BKTGibbsSampler` no longer runs a forward-backward pass per retained draw for
  a log-likelihood trace that was never read, and it reuses the posterior-mean
  smoothing pass for the final log-likelihood. Seeded results are bitwise
  unchanged. A 100-iteration 2000x30 fit is about 1.25x faster with native
  kernels and 2.4x faster on NumPy.
- `LongitudinalGibbsSampler` draws theta proposals vectorized and drops an
  unused per-draw log-likelihood, about 1.4-1.9x faster per iteration at
  2000x5x30.
- `IRTreeEMEstimator` collapses product-grid expected counts onto each node's
  1-D trait grid before the per-node M-step optimizations and the item standard
  errors. This is exact, and the M-step is 1.3-1.5x faster at 2000x30 with 11 or
  15 quadrature points.
- Likelihood-ratio DIF refits are warm-started from the baseline item estimates
  and latent distributions, about 2x faster than cold refits with identical chi2
  (12 items, 800 per group: 19.8 s to 9.6 s). They can also run in worker
  processes with `n_jobs`. `grdif_effect_size` no longer refits a model.
- Matrix SE methods (`compute_se` with
  "oakes"/"louis"/"sem"/"crossprod"/"sandwich", `compute_observed_information`,
  `compute_oakes_se`, `compute_crossprod_se`, `compute_sandwich_se`) compute the
  exact Louis information and marginal scores for built-in item models instead
  of O(P^2) finite differences. For example, a 2PL 2000x30 takes about 20 ms
  instead of 4.4 s, and a GRM 2000x8 about 15 ms instead of 1.2 s.
- Native multigroup E-steps (2PL/3PL/GRM/GPCM/PCM/NRM) now reuse the shared
  cached likelihood tables and release the GIL. A 2-group x 1000-person x
  30-item GRM E-step drops from about 16 ms to 1.2 ms on one thread, and
  multigroup expected counts reuse the shared count kernel.
- The multigroup EM M-step now fits all free coordinates of each built-in
  1PL-4PL item in one L-BFGS-B solve. Shared coordinates are single variables,
  group-specific coordinates get one variable per group, groups with identical
  built-in curves pool their counts, and fixed or masked coordinates never move.
  Joint solves use a relative tolerance of at most 1e-10, so flat 3PL/4PL
  directions are solved fully. 2PL fits are 1.7-2.3x faster end to end
  (1.55-1.8x per iteration). 3PL fits need far fewer EM iterations (3PL
  configural, 2x1000x30: 30.7 s / 355 iterations -> 1.8 s / 41 iterations) and
  reach equal or higher log-likelihoods.
- Bifactor EAP scoring costs time linear, not exponential, in the number of
  specific factors. For six factors and 600 respondents, default `fscores` drops
  from 0.21 s to 0.06 s while also becoming accurate.
- The exact (Louis) observed information now covers mixed-format models with
  rating-scale (RSM, GRSM) components. Their `se_method="auto"`/`"oakes"`
  standard errors no longer use O(P^2) finite differences of the marginal
  likelihood. The SE step for a 3PL(15)+RSM(6), N=1000 fit drops from 4.2 s to
  under 0.1 s, and the values agree with finite differences to about 4e-8.
  `BLEstimator` on such models also uses the exact information.
- Bootstrap, jackknife, parametric and likelihood-ratio replicates skip standard
  errors (bootstrap_se 3PL N=2000 J=30 n_bootstrap=20: 0.80 s to 0.43 s; GRM
  J=20: 1.2 s to 0.65 s). multi_start_fit computes them once for the best start
  (2PL N=2000 J=30 n_starts=5: 0.35 s to 0.28 s).
- Built-in models whose likelihood is a product of item curves now use exact
  item-local Louis information instead of O(P^2) marginal-likelihood
  differencing. This covers the 5PL, complementary and negative log-log,
  unipolar log-logistic, compensatory-logic, nested-logit, monotone polynomial
  and spline, sequential, continuation-ratio, adjacent-category, GGUM,
  ideal-point, hyperbolic cosine, zero-inflated, hurdle, `MultidimensionalModel`
  and `BifactorModel` models. The Hessian in BL standard errors drops from 4.8 s
  to 0.02 s for a 40-item CLL and from 6.4 s to 0.03 s for a 20-item 5PL.
  `fit_mirt(spec=..., se_method='oakes')` on a 2-factor, 16-item model drops
  from 11.6 s to 1.9 s.
- Items with coordinates fixed by `fixed=` or `set_free_parameter_masks` keep
  their family's analytic M-step objective instead of numerically
  differentiating the item curves. This covers 1PL-4PL, multidimensional and
  bifactor logistic, GRM, GPCM, PCM and NRM items. Masked 2PL, GRM and GPCM fits
  run about 3-5x faster with the same estimates.
- Adaptive MCEM reuses the E-step's draw log-likelihoods in its ascent check and
  evaluates only the previous parameters. That is one fewer likelihood pass per
  iteration (29 to 20 passes on a 10-iteration 2PL fit), with bitwise-identical
  results.
- Multigroup EM computes all expected item counts once per group. It solves
  built-in 1PL/2PL items whose parameters are all shared or all group-specific
  with batched Newton M-steps. The default likelihood-ratio DIF
  (compute_dif/mirt.dif) gives the same statistics and is about 7x faster for 30
  2PL items with 1,000 persons per group (11.0s -> 1.6s on a loaded machine) and
  about 4.5x faster for 1PL. Configural and scalar fit_multigroup 2PL fits are
  7-12x faster. 3PL, polytomous and metric-invariance fits are unchanged.

### Removed

- The unused private `EMEstimator._log_multivariate_normal` and the unused `n_k`
  argument of the private `_optimize_item*` methods. Internally, item-parameter
  boxes now come from one shared table.
- Unused comparison.relative_fit, modelfit.model_fit_summary,
  modelfit._compute_srmsr, sibtest._adjust_p_values and
  utils.numeric.compute_expected_variance. None was exported.
- Dead polytomous wrappers and duplicated validators and kernels across the
  linking, diagnostics, polytomous, drift and vertical modules. Vertical chain
  scaling now delegates to `chain_link`.
- Private helpers `mirt.diagnostics.dtf._bootstrap_dtf_se` and
  `mirt.diagnostics._utils.extract_item_se`.
- 49 native kernels that no Python code called were removed from the internal
  `mirt.mirt_rs` extension, including the whole irtree, mfrm, mirt_models and
  multilevel kernel modules (LLTM/latent-regression, MFRM and multilevel
  likelihoods, PSIS-LOO/WAIC fast paths, Haebara/TCC/robust-z equating helpers,
  MCEM/weighted E-steps, `m_step_3pl_parallel`, `compute_se_from_hessian`,
  `compute_q3_matrix_sparse`, `generate_qmc_samples` and others). Code calling
  these `mirt.mirt_rs` functions directly must use the public Python APIs.
- The unused GPU helper `mirt._gpu_backend.compute_expected_counts_gpu`.
- Duplicated parameter setters, probability/information dispatch, row-blocking
  loops, dimensionality guards, properties and eight copies of the item-index
  validator across the dichotomous, zero-inflated, sequential, nested, mixture,
  CDM, unfolding, testlet, MIRT and bifactor models, for about 318 fewer source
  lines. The 5PL model now shares the unidimensional curve evaluator, and its
  outputs are bit-identical to before.
- The multigroup E-step's duplicated Python implementation.

### Fixed
- Preserved finite Gaussian log kernels near floating-point limits and finite
  Cholesky factors for large prior covariances. Posterior acceptance now keeps
  prior differences under huge common likelihood offsets. Cached and read-only
  custom likelihood/prior outputs remain intact during chain updates, including
  callbacks that share scratch storage.
- Preserved custom MCEM item callbacks when using a shared QMC grid. Normalized
  importance weights after centering log likelihoods so large common offsets
  retain unit posterior mass, with bounded scratch and protected borrowed inputs.
- Preserved IRTree posterior normalization for large common log-likelihood
  offsets and protected borrowed custom likelihood buffers. Explicit node masks
  suppress stored decisions consistently in likelihoods and counts. IRTree
  optimizer gradients now respect clipped tails. Custom theta overrides retain
  numerical EM item objectives for logistic and affine models.
- Restored BL parameters after failed numerical optimization or curvature
  trials, and skipped optimization for models with no free parameters.
- Preserved custom model constructor state and bound instance probability
  methods in parallel numerical uncertainty. Core EM uncertainty falls back
  to numerical curvature for custom curve and parameter overrides.
- Enabled item updates and access for category-by-factor parameter arrays,
  fixing multidimensional NRM fitting. Shared coordinate-wise numerical
  curvature now preserves tensor shapes in EM and standalone standard errors,
  with parameter restoration when probability evaluation fails.
- Kept regularized MIRT convergence, likelihoods, and fit statistics in log space
  so long tests do not underflow into a likelihood floor or falsely converge.
  Posterior weights preserve unit row sums and prior differences under large
  common log-likelihood offsets. Failed adaptive warmstarts restore the penalty
  and adaptive weights, and response preparation is released after each fit.
- Matched Python 1PL–4PL EM gradients to the clipped likelihood, preserved small
  positive-tail derivatives, and avoided NaNs from zero counts at saturated
  probabilities. Custom binary models now retain their own probability functions
  during optimization. Multidimensional 2PL objectives share stable logit recovery.
- Preserved confirmatory loading patterns across model copies and excluded
  fixed zero loadings from optimization and parameter counts. Parallel EM
  workers now isolate numerical trial updates, including custom model state.
  Bifactor copies retain subclass behavior. Affine M-step gradients follow
  probability clipping and handle zero counts at saturated probabilities.
- Recovered finite MIRT and bifactor logits after product overflow, intercept
  cancellation, or cancellation between factor contributions. Probabilities
  and scalar/matrix information now share this recovery, while unrelated
  bifactor columns and genuinely nonfinite inputs retain their prior semantics.
- Retained multidimensional and bifactor information in both logistic tails,
  recovering representable scalar and matrix entries after intermediate slope
  overflow or tail underflow. Standardized MIRT loadings and communalities now
  retain large finite slopes without overflowing their squared norms.
- Logistic probabilities recover finite results from overflowing offsets, and
  4PL curves retain exact asymptotes. Multidimensional 2PL probability and
  information recenter large locations before subtraction, with exact arithmetic
  limited to cells whose severe cancellation survives centered evaluation.
- Unipolar probability and information preserve both tails and small nonzero
  information near the curve peak, including extreme slope rescaling. Corrected
  the docstring to describe the implemented symmetric curve with a 0.25 peak.
- Logistic Fisher information preserves both tails when success probabilities
  round to one, and recovers representable results after extreme slope scaling.
  Multidimensional 2PL information avoids premature overflow in slope norms and
  location offsets; NaN abilities propagate instead of appearing uninformative.
- CLL/NLL information preserves representable tails when probabilities round to
  zero or one, including recovery after slope rescaling. Extreme finite offsets
  no longer overflow before small slopes can rescale them, and probabilities
  retain their exact limits at infinite abilities.
- Strongly asymmetric 5PL curves preserve probabilities and Fisher information
  when the underlying sigmoid rounds to zero or one. Generalized difficulty
  inversion retains interior solutions at extreme asymmetries and targets near
  probability boundaries instead of incorrectly clamping to the search limits.
  Finite abilities and parameters also retain representable results when their
  initial differences or logit products overflow before asymmetry rescaling.
- Information and expected-score queries accept NumPy integer indices and item
  selection arrays, preserve repeated selection order, and interpret a single
  multidimensional ability vector consistently. Invalid indices and integration
  inputs now fail validation before model evaluation.
- Empirical reliability preserves score-unit invariance for tiny and very large
  scores and standard errors. Marginal reliability normalizes large or tiny
  density weights safely, handles normal-density tails, and retains positive
  variances without an arbitrary minimum. Invalid item information is rejected
  before summing it into test information.
- Classical item-total correlations remain accurate and within [-1, 1] on long,
  nearly constant tests, including exactly constant corrected totals.
- Empirical quantile bins handle extreme finite abilities without overflowing
  their edges or means. Empirical RMSEA accepts shared category counts from
  custom ordinal models and rejects malformed category metadata.
- Kernel calibration preserves person weights when observations share very large
  distances, excludes zero-weight neighbors from extreme-distance centering, and
  keeps ordinary grid points independent of overflowing-distance fallbacks.
- `itemGAM` produces finite curves with tiny positive bandwidths and preserves
  uncertainty in grid tails when nearby responses are missing.
- Highest-density intervals no longer depend on row position or batch size at
  adjacent floating-point probability boundaries, or return reversed bounds at
  very small positive probability levels.
- Benchmark comparisons reject different person counts for pattern grouping
  and the new highest-density interval workload.

## [1.1.0] - 2026-08-18

### Added
- Documented Rust fallback contract (`numpy` / `optional` / `required` / `mixed`)
- `should_use_rust()` honors global `set_backend("numpy")` alongside per-call flags
- Rust↔Python numerical parity tests for likelihood, E-step, scoring, and diagnostics
- User guides for CAT, DIF, multigroup, and equating; curated `examples/` scripts
- Benchmark harness under `benchmarks/` for EM fit, scoring, and CAT
- Expanded `@pytest.mark.slow` coverage for MCMC / high-cost paths

### Changed
- Modular Rust wrappers under `mirt.backends.rust` ( `_rust_backend` remains a shim)
- Coverage gate raised from 55% toward Production/Stable expectations (now 59%)
- Stricter mypy checking on additional public modules

### Included from the original 1.1 development milestone

### Added
- Custom exception hierarchy for better error handling
- CI performance smoke-test job and weekly scheduled slow-test job
- Performance regression smoke tests for import time and small-model fit time
- Documentation regression tests to prevent stale API symbol references in docs
- Improved docstring coverage across modules

### Changed
- Minimum Python version lowered from 3.14 to 3.11 for broader compatibility
- Development status updated to Production/Stable
- CI now tests on Python 3.11, 3.12, 3.13, and 3.14
- Top-level API now lazy-loads heavy equating/multigroup/plotting/report symbols
- Dataset constants in `mirt.utils.datasets` are lazily materialized on first access
- Replaced broad `except Exception` handlers with narrower recoverable exception classes
- Sphinx quickstart and API reference examples now match current public symbols
- Integration/scoring tests use backend-tolerant convergence and correlation expectations
- Reduced dead/duplicate implementation noise across EM, CAT/MCAT, scoring, and residuals
- Optimized EM dichotomous item updates with analytic gradients (`1PL`/`2PL`/`3PL`/`4PL`)
- Optimized `fit_ising` interaction-gradient computation

## [1.0.0] - 2026-01-15

### Added
- Stable public API policy for v1.x (see README API Stability)
- Computerized Adaptive Testing (CAT) module with:
  - Multiple item selection strategies (MFI, MEI, KL, Urry, random, a-stratified)
  - Configurable stopping rules (SE threshold, max/min items)
  - Exposure control methods (Sympson-Hetter, randomesque)
  - Content balancing constraints
  - CAT simulation and batch evaluation functions
- DIF analysis (LR, Wald, Lord, Raju), DTF/DRF, SIBTEST, GRDIF
- Multigroup IRT with invariance testing; bifactor models
- Zero-inflated, unfolding, testlet, CDM, and mixture IRT models
- GVEM and sparse Bayesian estimation paths
- HTML report generation, vertical scaling, fixed-item calibration / equating

### Changed
- Declared Production/Stable for core public API
- Refactored Rust backend into modular structure
- Consolidated duplicate code patterns with shared utility functions

## [0.1.11] - 2025-01-08

### Added
- Computerized Adaptive Testing (CAT) module with:
  - Multiple item selection strategies (MFI, MEI, KL, Urry, random, a-stratified)
  - Configurable stopping rules (SE threshold, max/min items)
  - Exposure control methods (Sympson-Hetter, randomesque)
  - Content balancing constraints
  - CAT simulation and batch evaluation functions

### Changed
- Refactored Rust backend into modular structure
- Consolidated duplicate code patterns with shared utility functions

## [0.1.10] - 2025-01-07

### Added
- Zero-inflated IRT models (ZI-2PL, ZI-3PL, Hurdle IRT)
- Unfolding models (GGUM, Ideal Point, Hyperbolic Cosine)
- Testlet response models for local dependence
- Cognitive Diagnostic Models (DINA, DINO)
- Mixture IRT models for latent class analysis

## [0.1.9] - 2025-01-07

### Added
- DIF (Differential Item Functioning) analysis with multiple methods:
  - Likelihood ratio test
  - Wald test
  - Lord's chi-square
  - Raju's area measures
- DTF (Differential Test Functioning) and DRF analysis
- SIBTEST implementation

## [0.1.8] - 2025-01-06

### Added
- Multigroup IRT analysis with invariance testing:
  - Configural, metric, scalar, and strict invariance
  - Likelihood ratio tests for invariance constraints
- Bifactor model support

## [0.1.7] - 2025-01-05

### Added
- MCMC estimation methods (Metropolis-Hastings, Gibbs sampling)
- Mixed effects IRT models
- LLTM (Linear Logistic Test Model)

## [0.1.6] - 2025-01-04

### Added
- Model fit statistics (M2, RMSEA, CFI, TLI, SRMSR)
- Item fit indices (S-X2, infit, outfit)
- Person fit statistics

## [0.1.5] - 2025-01-03

### Added
- Bootstrap standard errors and confidence intervals
- Plausible values generation
- Multiple imputation for missing data

## [0.1.4] - 2025-01-02

### Added
- Polytomous IRT models:
  - Graded Response Model (GRM)
  - Generalized Partial Credit Model (GPCM)
  - Partial Credit Model (PCM)
  - Nominal Response Model (NRM)
- Multidimensional IRT models (exploratory and confirmatory)

## [0.1.3] - 2025-01-01

### Added
- High-performance Rust backend via PyO3
- Parallel E-step computation with Rayon
- EAPsum scoring with Lord-Wingersky algorithm

## [0.1.2] - 2024-12-31

### Added
- EM algorithm estimation
- Multiple scoring methods (EAP, MAP, ML, WLE)
- Standard error estimation

## [0.1.1] - 2024-12-30

### Added
- Core dichotomous models (1PL, 2PL, 3PL, 4PL)
- Basic parameter estimation
- Response simulation

## [0.1.0] - 2024-12-29

### Added
- Initial release
- Basic IRT model infrastructure
- NumPy/SciPy integration
