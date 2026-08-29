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
   print(scores.summary())

Person identifiers may be lists or NumPy arrays and must contain one identifier per
score row. ``to_dataframe()`` preserves those identifiers with either supported
dataframe backend. Score and uncertainty arrays must have identical shapes, and finite
standard errors cannot be negative.

``classification_probabilities()`` uses the score and its standard error to calculate
the normal-approximation probability that ability exceeds a cut score. ``classify()``
returns ``"above"`` or ``"below"`` only when that one-sided probability reaches the
requested confidence; otherwise it returns ``"uncertain"``. Both methods accept a
scalar cut or an array broadcastable to the score shape, including one cut per factor
for multidimensional scores. Zero standard error produces a deterministic decision
away from the cut, while infinite or unknown uncertainty does not force a decision.
