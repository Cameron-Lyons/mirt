Custom Models
=============

Custom item models let you supply a response-probability callback while using
the same parameter, likelihood, information, and expected-score interface as
the built-in item models. Callbacks may define dichotomous, polytomous, or
multidimensional response functions.

Dichotomous items
-----------------

The first callback argument is theta. Remaining arguments are item parameters
and can be inferred from the function signature.

.. code-block:: python

   import numpy as np
   from mirt import CustomItemModel, create_item_type

   def logistic(theta, slope, location):
       return 1 / (1 + np.exp(-slope * (theta - location)))

   logistic_spec = create_item_type(
       "Logistic",
       logistic,
       par_bounds={"slope": (0.05, 5), "location": (-5, 5)},
       par_defaults={"slope": 1, "location": 0},
   )

   model = CustomItemModel(n_items=3, item_type=logistic_spec)
   model.set_parameters(slope=[0.8, 1.2, 1.5], location=0)

   theta = np.linspace(-3, 3, 61)
   probability = model.probability(theta)       # (61, 3)
   information = model.information(theta)       # (61, 3)
   total_score = model.expected_score(theta)    # (61,)

If ``info_function`` is omitted, information is computed by central numerical
differentiation. An optional ``gradient_function`` can return a mapping from
each parameter name to its probability derivative; otherwise parameter
gradients are also computed numerically.

All-item batch callbacks
------------------------

Models with many items can avoid one Python callback per item by supplying a
``batch_icc_function``. It receives theta plus complete ``(n_items,)``
parameter arrays. Dichotomous callbacks return ``(n_theta, n_items)``;
polytomous callbacks return ``(n_theta, n_items, n_categories)``.

.. code-block:: python

   def logistic_batch(theta, slope, location):
       linear = slope[None, :] * (theta[:, None] - location[None, :])
       return 1 / (1 + np.exp(-linear))

   fast_spec = create_item_type(
       "BatchedLogistic",
       logistic,
       par_bounds={"slope": (0.05, 5), "location": (-5, 5)},
       par_defaults={"slope": 1, "location": 0},
       batch_icc_function=logistic_batch,
   )
   fast_model = CustomItemModel(1_000, fast_spec)
   probabilities = fast_model.probability(theta)  # one batch callback

The per-item callback remains the source for requests with ``item_idx``. When
no analytical information callback is defined, all-item numerical information
automatically reuses ``batch_icc_function``. An optional
``batch_info_function`` can instead return information directly with shape
``(n_theta, n_items)``. Parameter arrays passed to batch callbacks are copies,
so callback-side mutation cannot alter the model.

Polytomous items
----------------

A polytomous callback returns a complete probability trace. Its shape must be
``(n_theta, n_categories)`` and every row must sum to one.

.. code-block:: python

   def ordinal(theta, shift):
       eta = theta - shift
       weights = np.column_stack((
           np.ones_like(eta),
           np.exp(eta),
           np.exp(2 * eta),
       ))
       return weights / weights.sum(axis=1, keepdims=True)

   ordinal_spec = create_item_type(
       "Ordinal",
       ordinal,
       par_bounds={"shift": (-4, 4)},
       par_defaults={"shift": 0},
       n_categories=3,
   )
   ordinal_model = CustomItemModel(5, ordinal_spec)

   traces = ordinal_model.probability(theta)    # (61, 5, 3)
   item_score = ordinal_model.expected_score(theta, item_idx=0)

Likelihood methods use ``-1`` for missing responses. The batch method evaluates
every response pattern at every supplied theta point.

.. code-block:: python

   responses = np.array([[0, 1, 2, -1, 1], [1, 2, 2, 0, 1]])
   grid = np.linspace(-4, 4, 81)
   log_likelihood = ordinal_model.log_likelihood_batch(responses, grid)
   # shape: (2, 81)

Multidimensional callbacks
--------------------------

With more than one factor, the callback receives theta as an
``(n_theta, n_factors)`` matrix. Numerical information is reported as the trace
of the Fisher information matrix.

.. code-block:: python

   def compensatory(theta, slope):
       linear = slope * theta.sum(axis=1)
       return 1 / (1 + np.exp(-linear))

   spec = create_item_type(
       "Compensatory",
       compensatory,
       par_bounds={"slope": (0.05, 5)},
       par_defaults={"slope": 1},
   )
   multidimensional = CustomItemModel(4, spec, n_factors=2)

Custom groups
-------------

Group specifications define a latent mean and covariance. Each callback only
receives the parameters present in its own signature, so mean and covariance
parameters can remain independent.

.. code-block:: python

   from mirt import CustomGroupModel, create_group

   group_spec = create_group(
       "FreeGroup",
       mean_function=lambda mean: mean,
       cov_function=lambda scale: np.array([[scale**2]]),
       par_bounds={"mean": (-4, 4), "scale": (0.05, 4)},
       par_defaults={"mean": 0, "scale": 1},
   )
   group = CustomGroupModel(group_spec)
   group.set_parameters(mean=0.4, scale=1.2)
   draws = group.sample(1_000, rng=np.random.default_rng(42))

The compatibility spelling ``createGroup`` remains available. The snake-case
``create_group`` name is preferred for new Python code.

Validation
----------

Parameter updates are atomic and checked against declared bounds. Probability
callbacks must return finite values in ``[0, 1]``. Group callbacks must return a
finite mean vector and a symmetric positive-semidefinite covariance matrix.
Invalid callbacks therefore fail immediately when evaluated instead of
silently contaminating later estimation or scoring steps.
