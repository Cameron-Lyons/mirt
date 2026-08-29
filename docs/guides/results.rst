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

``summary()`` supports one-, two-, and higher-dimensional parameter arrays. The legacy
``coef()`` and ``coef_with_se()`` methods remain available for item-aligned one- and
two-dimensional parameters through the configured pandas or polars backend.

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

The export contains model metadata and scalar fit statistics. Parameter and uncertainty
arrays are converted to nested Python lists.

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
shapes and any supplied ``n_persons`` or ``n_factors`` metadata. ``to_dataframe()``
preserves identifiers with either supported dataframe backend. Score and uncertainty
arrays must have identical shapes, and finite standard errors cannot be negative.

``classification_probabilities()`` uses the score and its standard error to calculate
the normal-approximation probability that ability exceeds a cut score. ``classify()``
returns ``"above"`` or ``"below"`` only when that one-sided probability reaches the
requested confidence; otherwise it returns ``"uncertain"``. Both methods accept a
scalar cut or an array broadcastable to the score shape, including one cut per factor
for multidimensional scores. Zero standard error produces a deterministic decision
away from the cut, while infinite or unknown uncertainty does not force a decision.

WLE performance and missing responses
-------------------------------------

Unidimensional 2PL WLE scoring groups repeated response patterns and evaluates the
unique patterns through a batched backend. It uses the native implementation when
available and a vectorized, memory-bounded NumPy implementation otherwise. Pass
``n_jobs`` to ``fscores`` to control native worker parallelism; ``-1`` uses all
available CPU cores. Other model families and multidimensional models retain the
general scorer.

Negative response values are treated as missing. Missing items contribute neither
log likelihood nor test information, so reported WLE standard errors reflect only
the items observed for each person. A person with no observed items receives theta
zero and infinite standard error.

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
   median = posterior.quantile(0.5)
   probability_above = posterior.classification_probabilities(cut_score=0.0)
   decisions = posterior.classify(cut_score=0.0, confidence=0.95)
   map_values = posterior.map_estimate
   scores = posterior.to_score_result()

``points`` contains the shared quadrature grid and each row of ``weights`` is a
normalized respondent distribution over that grid. ``log_marginal_likelihood``
contains the integrated response-pattern likelihood under the chosen prior.
``mean``, ``median``, ``standard_error``, ``quantile()``, and ``map_estimate`` support
unidimensional and multidimensional models; credible intervals and threshold
probabilities are computed from each factor's exact marginal grid distribution rather
than a normal approximation. Likelihood evaluation is memory-bounded through
``batch_size``, while the returned weight matrix necessarily contains
``n_persons * n_points`` values.
