Joint Accuracy and Response-Time Models
=======================================

Response-time models combine response accuracy with the time taken on each
item.  :class:`mirt.ResponseTimeModel` implements the Van der Linden
hierarchical model with either 2PL or 3PL accuracy and log-normal response
times.

Model evaluation
----------------

Create a model with one set of accuracy and timing parameters per item:

.. code-block:: python

   import numpy as np
   from mirt import ResponseTimeModel

   model = ResponseTimeModel(
       n_items=3,
       discrimination=np.array([1.2, 0.9, 1.4]),
       difficulty=np.array([-0.4, 0.1, 0.8]),
       time_intensity=np.array([3.0, 3.2, 2.8]),
       time_discrimination=np.array([1.1, 0.8, 1.3]),
   )

   responses = np.array([[1, 0, 1], [0, 1, -1]])
   log_response_times = np.log(
       np.array([[18.0, 27.0, 14.0], [35.0, 22.0, np.nan]])
   )
   theta = np.array([0.5, -0.3])
   speed = np.array([0.2, -0.1])

   person_log_likelihood = model.joint_log_likelihood(
       responses, log_response_times, theta, speed
   )

Negative or ``NaN`` responses are treated as missing.  A ``NaN`` log response
time is missing independently of its accuracy response.

Timing predictions
------------------

Deadline probabilities and response-time quantiles are evaluated directly on
the original time scale. A scalar deadline or quantile is broadcast across all
persons and items, while vectors and matrices support distinct requirements.

.. code-block:: python

   probability_by_30 = model.response_time_cdf(30.0, speed)
   lower = model.response_time_quantile(0.025, speed)
   median = model.response_time_quantile(0.5, speed)
   upper = model.response_time_quantile(0.975, speed)

   assert probability_by_30.shape == (2, 3)
   assert np.all(lower < median)
   assert np.all(median < upper)

Simulation and estimation
-------------------------

The model can simulate correlated ability and speed values together with
responses and response times:

.. code-block:: python

   responses, response_times, theta, speed = model.simulate(500, seed=42)

Fit all item, person, and population parameters with
:class:`mirt.ResponseTimeGibbsSampler`:

.. code-block:: python

   from mirt import ResponseTimeGibbsSampler

   result = ResponseTimeGibbsSampler(
       n_iter=2_000,
       burnin=500,
       seed=42,
   ).fit(responses, response_times)

   print(result.summary())
   print(result.person_summary(5))

Backend selection
-----------------

Joint likelihood evaluation and Gibbs person updates use compiled parallel
kernels when available.  Proposal randomness is generated before dispatch, so
the compiled and NumPy paths make the same accept/reject decisions for the same
inputs.

Disable acceleration for one workflow with ``use_rust=False`` on either
``ResponseTimeModel`` or ``ResponseTimeGibbsSampler``.  To use NumPy globally,
call ``mirt.set_backend("numpy")``.
