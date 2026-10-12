Working with Results
====================

Fitting and scoring functions return structured result objects. These objects validate
their shapes and metadata when they are created, copy caller-owned arrays, and provide
portable exports that do not require a dataframe package.

Model-fit results
-----------------

``FitResult`` stores the fitted model, convergence metadata, fit indices, and parameter
uncertainty. Use ``parameter_statistics`` for vectorized normal-approximation inference:

.. code-block:: python

   result = mirt.fit_mirt(responses, model="2PL")
   statistics = result.parameter_statistics(alpha=0.05)

   difficulty = statistics["difficulty"]
   print(difficulty["estimate"])
   print(difficulty["standard_error"])
   print(difficulty["p_value"])
   print(difficulty["ci_lower"], difficulty["ci_upper"])

Missing standard errors are represented by ``NaN``. They are never reported as exact
zeros. Tail probabilities use a numerically stable vectorized normal survival
calculation, so large finite z-values retain meaningful nonzero p-values.

Confidence intervals are also available directly:

.. code-block:: python

   lower, upper = result.confidence_intervals(alpha=0.10)["difficulty"]

When a workflow only consumes fitted parameters, skip repeated inferential work
explicitly:

.. code-block:: python

   parameter_only = mirt.fit_mirt(
       responses,
       model="2PL",
       compute_standard_errors=False,
   )

This is useful for resampling and repeated diagnostic fits. The returned
``FitResult`` retains parameters, convergence metadata, and fit indices, while
its standard-error mapping is empty and inferential summaries report ``NaN``.
Diagnostic group refits use this mode by default.

``summary()`` supports one-, two-, and higher-dimensional parameter arrays. The legacy
``coef()`` and ``coef_with_se()`` methods remain available for item-aligned one- and
two-dimensional parameters through the configured pandas or polars backend.

Rotating exploratory solutions
------------------------------

An exploratory fit, such as ``fit_mirt(responses, model="2PL", n_factors=2)``,
determines its factors only up to a rotation. ``get_rotated_loadings`` rotates the
standardized loadings as R mirt's ``summary(rotate=...)`` does, and
``apply_rotation_to_model`` re-expresses the fit on the rotated factors, like R's
``coef(rotate=...)``:

.. code-block:: python

   from mirt.utils.rotation import apply_rotation_to_model, get_rotated_loadings

   result = mirt.fit_mirt(responses, model="2PL", n_factors=2)
   loadings, phi = get_rotated_loadings(result, "oblimin", standardized=True)

   apply_rotation_to_model(result, "oblimin")
   print(result.factor_correlation)
   scores = mirt.fscores(result, responses)

Both accept a ``FitResult`` or its model: a 2PL, GRM, GPCM, NRM or ``MIRT`` model
whose slopes on two or more factors are all free. ``apply_rotation_to_model`` maps
the slopes ``a`` to ``a @ T`` and the factors to ``inv(T) @ theta``. An oblique
rotation records the factor correlation ``inv(T) @ inv(T).T`` as the result's
``latent_covariance``, so the marginal likelihood, response probabilities and fit
statistics are unchanged, and scoring, simulation and diagnostics use the correlated
factors. A bare model has nowhere to keep that correlation, so it accepts only
orthogonal rotations. The 2PL, GRM and GPCM have multidimensional logits
``a @ theta - sum(a) * b``, so their locations ``b`` are rescaled to keep each
item's intercepts. The standard errors and ``vcov`` entries of the rotated
parameters become ``NaN``, as R drops them.

``rotate_loadings`` provides varimax, quartimax, equamax, orthogonal and oblique
geomin (``geominT``, ``geominQ``), bifactor (``bifactorT``, ``bifactorQ``), target
and partially specified target (``targetT``, ``targetQ``, ``pstT``, ``pstQ``),
oblimin, quartimin and promax rotations. Every criterion except promax is minimized
by gradient projection (Jennrich, 2001, 2002) with a monotone line search, from the
identity and from ``random_starts`` random orthogonal starts. The defaults follow R
mirt. Kaiser normalization is off (``normalize=None``) except in the varimax step of
promax, whose power target is formed from the de-normalized varimax loadings as in
``stats::promax``. Each column is reflected so that its largest loading is positive,
and ``get_rotated_loadings`` defaults to ``"oblimin"``. With these defaults the
loadings and factor correlations match R mirt 1.48 to the optimizer tolerance.

Some differences remain. ``tol`` defaults to ``1e-6``, where GPArotation's ``eps``
is ``1e-5``. GPArotation's Barzilai-Borwein steps can stop at a different point
within that tolerance. For criteria with several local minima, such as
biquartimin, they can also reach a different minimum. Equamax is the orthomax
criterion with weight ``n_factors / 2``, GPArotation's
``cfT(kappa = n_factors / (2 * n_items))``, which R mirt's ``summary()`` does not
offer. R's ``coef(rotate=...)`` multiplies GPCM and nominal slopes by the
``sqrt((K - 1) * 1.702)`` scale of their standardized loadings, whereas
``get_rotated_loadings`` returns the model's own slopes.

Monte Carlo item uncertainty
----------------------------

``MCEMEstimator``, ``QMCEMEstimator``, and ``StochasticEMEstimator`` estimate
standard errors after fitting with ``compute_standard_errors=True``:

.. code-block:: python

   from mirt import TwoParameterLogistic
   from mirt.estimation import MCEMEstimator

   result = MCEMEstimator(
       n_samples=256,
       seed=42,
       compute_standard_errors=True,
       se_step_size=1e-5,
   ).fit(TwoParameterLogistic(responses.shape[1]), responses)

   print(result.se_method, result.standard_errors["difficulty"])

Unidimensional built-in 1PL-4PL, GRM, GPCM and PCM fits report the observed
information of the marginal likelihood at the estimates (Louis, 1982),
integrated over the ability prior on EM's default grid, as
``se_method="oakes"`` with the free-parameter covariance in ``vcov``. They match
EM's default standard errors at the same parameters. The diagonal
complete-data curvature these estimators used to report was 0.4 to 0.8 times
those errors on a 2PL of 1000 persons and 20 items.

Other models, and estimators or models with custom likelihood hooks, report
diagonal complete-data standard errors on the final draws and weights, with
``se_method="complete_data"`` and no ``vcov``. Draws and weights remain fixed
during differentiation. This approximation excludes missing information,
covariance between parameters, and Monte Carlo sampling error, so it
understates the uncertainty. Built-in polytomous items use exact diagonal
curvature, while logistic and affine items use prepared item gradients. Custom
item objectives use numerical likelihood curvature. Fixed coordinates receive
zero errors; unobserved items and undefined curvature receive ``NaN``.

The option defaults to ``False``, retaining placeholder errors and the existing
fitting cost. Enabling it preserves the fitted parameters, likelihood, random
draws, and convergence history. ``se_step_size`` must be finite and positive;
it controls finite differences and does not affect exact derivatives.

Portable model exports
----------------------

Use ``to_dict`` when a dataframe package is unnecessary or when results need to be
serialized:

.. code-block:: python

   import json

   payload = result.to_dict()
   json_text = json.dumps(payload)

   compact = result.to_dict(
       include_parameters=False,
       include_standard_errors=False,
   )

   json_text = result.to_json(indent=2)

The export contains model metadata, including per-item category counts for
polytomous models, and scalar fit statistics. Parameter and uncertainty arrays are
converted to nested Python lists.

Results from the ``fit_mirt`` families (1PL, 2PL, 3PL, 4PL, GRM, GPCM, PCM and NRM)
can be rebuilt from a full export and scored again, as can confirmatory
``fit_mirt(spec=...)`` fits (a multi-factor 2PL is exported as ``"MIRT"`` with its
loading pattern), ``bfactor`` fits (``"Bifactor"`` with the specific-factor labels)
and mixed-format fits of these families (``"Mixed"`` with each component's family and
items):

.. code-block:: python

   restored = mirt.FitResult.from_json(result.to_json())
   scores = mirt.fscores(restored, responses)

Coordinates held fixed by ``fixed``, ``FIXED`` or ``set_free_parameter_masks`` are
exported as ``free_parameter_masks`` and restored, and so is an estimated
``latent_covariance``. ``from_dict()`` and ``from_json()`` reject unknown fields,
exports written with ``include_parameters=False``, and other models, such as custom
item types or rating-scale components. Unknown standard errors are written as ``NaN``,
which Python's ``json`` module accepts but strict JSON parsers may not.

Person-score results
--------------------

``ScoreResult`` supports unidimensional and multidimensional scores:

.. code-block:: python

   scores = mirt.fscores(result, responses, method="EAP")
   lower, upper = scores.confidence_intervals()

   probability_above = scores.classification_probabilities(cut_score=0.0)
   decisions = scores.classify(cut_score=0.0, confidence=0.95)

   values = scores.to_array(include_se=True)
   payload = scores.to_dict()
   json_text = scores.to_json(indent=2)
   restored = mirt.ScoreResult.from_json(json_text)
   print(scores.summary())

Person identifiers may be lists or NumPy arrays and must contain one identifier per
score row. NumPy scalar identifiers are converted to standard Python scalars in
portable exports. ``from_dict()`` and ``from_json()`` validate reconstructed array
shapes and any supplied ``n_persons`` or ``n_factors`` metadata, which also restores
the factor count of an empty multidimensional result. ``to_json()`` writes standard
JSON: infinite and NaN values, such as the infinite standard errors of extreme ML
scores, become the strings ``"Inf"``, ``"-Inf"`` and ``"NaN"``, which
``from_json()`` restores exactly and R's ``jsonlite::fromJSON`` reads as numbers.
``to_dict()`` keeps them as floats. ``to_dataframe()`` preserves identifiers with
either supported dataframe backend. Score and uncertainty arrays must have identical
shapes, and finite standard errors cannot be negative.

``classification_probabilities()`` uses the score and its standard error to calculate
the normal-approximation probability that ability exceeds a cut score. ``classify()``
returns ``"above"`` or ``"below"`` only when that one-sided probability reaches the
requested confidence; otherwise it returns ``"uncertain"``. Both methods accept a
scalar cut or an array broadcastable to the score shape, including one cut per factor
for multidimensional scores. Zero standard error produces a deterministic decision
away from the cut, while infinite or unknown uncertainty does not force a decision.

Response columns and extreme ML scores
--------------------------------------

Every scoring function, including ``fscores``, ``ability_posterior``, the EAPsum
functions, ``generate_plausible_values`` and ``personfit``, reads array columns by
position. A data frame's columns are matched to the model's item names instead, as
when the model was fitted to a data frame: they may come in any order, and other
columns, such as respondent identifiers, are ignored. A data frame without columns
for some of the named items raises ``MirtDataError`` listing them; pass
``frame.to_numpy()`` to score its columns by position. Models fitted to unnamed
arrays, whose items are named ``Item_1``, ``Item_2``, ..., also read other data
frames by position. ``mirt.utils.data.align_response_columns`` applies this rule.
R's ``mirt`` always reads ``response.pattern`` columns by position.

A response pattern whose observed responses are all in the lowest category, or all
in each item's highest category, has no finite maximum likelihood estimate. For every
model family, ``method="ML"`` reports it at the lower or upper ``bounds`` value in
every coordinate with an infinite standard error, where R's ``mirt`` reports
``-Inf`` or ``Inf`` with a missing standard error. In multidimensional models, a
coordinate whose likelihood still increases at a bound, such as a factor whose items
are all answered in their lowest or highest category, is likewise reported at that
bound with an infinite standard error. WLE scores of such patterns are finite.

EAP grids for bifactor models
-----------------------------

EAP integrates over a product grid of ``n_quadpts ** n_factors`` nodes, so its
default points per dimension fall as factors are added (five for six or more
factors). For a bifactor model, such as a ``mirt.bfactor`` result, ``fscores``
instead integrates each specific factor jointly with the general factor on a
two-dimensional grid, as ``bfactor`` estimation does (Gibbons and Hedeker, 1992).
This equals the product-grid EAP at the same ``n_quadpts`` but costs time linear in
the number of specific factors, so bifactor scores default to 49 points per
dimension whatever the factor count. It applies when the prior keeps the specific
factors independent given the general factor, as the default standard-normal
prior does. ``ability_posterior`` must return the joint grid and therefore keeps
the product grid; it warns when an automatic bifactor grid has fewer than 11
points per dimension, which biases the posterior summaries.

WLE performance and missing responses
-------------------------------------

Unidimensional 2PL WLE scoring groups repeated response patterns and evaluates the
unique patterns through a batched backend. It uses the native implementation when
available and a vectorized, memory-bounded NumPy implementation otherwise. Pass
``n_jobs`` to ``fscores`` to control native worker parallelism; ``-1`` uses all
available CPU cores. Other unidimensional families share one row-batched search.

The unidimensional WLE is the global maximum of the weighted likelihood over the
search bounds. A test that mixes weakly and highly discriminating items, with
sparse responses, can give that criterion several local maxima, and a local search
then stops on a lower one. Every search is therefore followed by a scan of each
pattern's criterion on a grid at most 0.25 apart; the three highest grid maxima
away from (or above) the estimate are refined, and the estimate moves to a higher
maximum when one is found. Unimodal patterns keep the local search's estimate, so
typical tests score as before and as R's ``fscores(method="WLE")``. On multimodal
patterns R's local search can stop on the lower maximum.

Multidimensional WLE follows Warm's estimator as R's ``mirt`` does: it maximizes the
log likelihood plus half the log determinant of the observed items' Fisher
information matrix (Wang, 2015), and its standard errors come from the inverse of
that matrix. Built-in multidimensional models are scored, for ML as for MAP and WLE,
by one row-batched projected Newton search over the unique response patterns, with
an L-BFGS-B fallback for rows that do not converge. Factors that no observed item
measures are reported as zero with infinite standard errors.

Negative response values, ``NaN`` and the nulls of nullable data frame columns are
treated as missing by every scoring method, as by ``fit_mirt``. Missing items
contribute neither log likelihood nor test information, so reported WLE standard
errors reflect only the items observed for each person. A person with no observed
items receives theta zero and infinite standard error.

Posterior ability distributions
-------------------------------

Use ``ability_posterior`` when inference needs the complete EAP distribution instead
of only its first two moments:

.. code-block:: python

   posterior = mirt.ability_posterior(
       result,
       responses,
       n_quadpts=49,
       person_ids=person_ids,
   )

   lower, upper = posterior.credible_intervals(level=0.95)
   hdi_lower, hdi_upper = posterior.highest_density_intervals(level=0.95)
   median = posterior.quantile(0.5)
   percentiles = posterior.quantile([0.1, 0.5, 0.9])
   probability_above = posterior.classification_probabilities(cut_score=0.0)
   decisions = posterior.classify(cut_score=0.0, confidence=0.95)
   map_values = posterior.map_estimate
   scores = posterior.to_score_result()
   plausible_values = posterior.sample(n_draws=10, seed=42)

``points`` contains the shared quadrature grid and each row of ``weights`` is a
normalized respondent distribution over that grid. ``log_marginal_likelihood``
contains the integrated response-pattern likelihood under the chosen prior.
``mean``, ``median``, ``standard_error``, ``quantile()``, and ``map_estimate``
support unidimensional and multidimensional models. ``credible_intervals`` returns
equal-tail bounds, while ``highest_density_intervals`` returns the shortest
contiguous marginal interval containing the requested mass. The latter is often
narrower for skewed posteriors and combines duplicate factor coordinates before
searching. Both intervals and threshold probabilities use each factor's exact grid
distribution rather than a normal approximation.

Likelihood evaluation is memory-bounded through the scorer's ``batch_size``. The
highest-density search also batches respondents automatically; pass its own
``batch_size`` when a fixed temporary-memory ceiling is required. The returned
weight matrix necessarily contains ``n_persons * n_points`` values.
Quantiles, equal-tail intervals, entropy, classification
probabilities, and sampling also process respondent batches to limit temporary
memory use.

Pass a sequence to ``quantile()`` to compute several quantiles using the same
cumulative distributions. The result has shape ``(n_probabilities, n_persons)``
for one factor, or ``(n_probabilities, n_persons, n_factors)`` for multiple factors.
Scalar queries retain the usual score shape. Probabilities must be strictly
between zero and one, and their requested order is preserved.

``sample()`` draws from the stored joint posterior without reevaluating the model.
It returns ``(n_persons, n_factors, n_draws)``, matching the layout used by
``generate_plausible_values``. Each draw selects a complete grid point, preserving
dependence between factors. A fixed seed gives reproducible draws. These draws
are conditional on the fitted item parameters and the chosen quadrature grid;
they do not include uncertainty in item calibration.
