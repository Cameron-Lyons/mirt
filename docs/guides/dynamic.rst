Dynamic knowledge models
========================

Bayesian knowledge tracing models a separate hidden mastery state for each
skill. Trials may be blocked by skill or interleaved in any order; inference
links only successive opportunities for the same skill.

Per-skill mastery
-----------------

.. code-block:: python

   import numpy as np
   from mirt.models import BKTModel

   model = BKTModel(
       n_skills=3,
       skill_names=["fractions", "decimals", "ratios"],
   )
   responses = np.array([1, 0, 1, -1, 1, 0])
   skills = np.array([0, 1, 2, 0, 1, 2])

   mastery = model.predict_mastery_by_skill(responses, skills)
   print(dict(zip(model.skill_names, mastery)))

Responses use ``1`` for correct, ``0`` for incorrect, and ``-1`` for missing.
A missing response contributes no response evidence while preserving the trial's
place in its skill sequence.

Batch inference
---------------

For a response matrix, calculate all learners' per-skill mastery estimates in
one call:

.. code-block:: python

   response_matrix = np.array(
       [
           [1, 0, 1, -1, 1, 0],
           [0, 1, 0, 1, 1, 1],
       ]
   )
   mastery = model.predict_mastery_batch(response_matrix, skills)
   print(mastery.shape)  # (2, 3)

   posterior, log_likelihoods = model.forward_backward_batch(
       response_matrix,
       skills,
   )
   print(posterior.shape)  # (2, 6, 2)

``skill_assignments`` may also be a matrix matching the response matrix when
learners receive different trial layouts. Shared layouts use the compiled
parallel implementation when it is available. Set ``use_rust=False`` on
``BKTModel`` or ``BKTGibbsSampler`` to select the NumPy implementation for a
specific workflow; the global ``mirt.set_backend("numpy")`` preference is also
honored.

Simulation
----------

Use ``simulate`` to generate reproducible responses, skill assignments, and
latent mastery states:

.. code-block:: python

   responses, skills, states = model.simulate(
       n_persons=1_000,
       n_trials_per_skill=20,
       seed=42,
   )

State-space filtering
---------------------

Track a continuously evolving ability state with a 2PL or 3PL observation
model. The batch filter updates every learner in one vectorized call:

.. code-block:: python

   import numpy as np
   from mirt.models import StateSpaceIRT

   state_model = StateSpaceIRT(
       n_items=20,
       n_timepoints=12,
       base_model="3PL",
       transition_matrix=np.array([[0.95]]),
       process_noise=np.array([[0.08]]),
       observation_noise=0.05,
   )
   responses, true_states = state_model.simulate(500, seed=42)
   responses[0, 3, 5] = -1

   diagnostics = state_model.predictive_diagnostics_batch(responses)
   means = diagnostics.filtered_means
   variances = diagnostics.filtered_variances
   print(means.shape, variances.shape)  # (500, 12) (500, 12)

   smoothed_means, smoothed_variances = (
       state_model.extended_kalman_smoother_batch(responses)
   )

   future_means, future_variances = state_model.forecast_batch(
       responses,
       n_steps=4,
   )
   future_success = state_model.forecast_response_probabilities_batch(
       responses,
       n_steps=4,
   )
   print(future_success.shape)  # (500, 4, 20)

   causal_state_means = diagnostics.predicted_means
   predictive_success = diagnostics.response_probabilities
   predictive_by_time = diagnostics.response_log_likelihoods
   predictive_total = diagnostics.total_log_likelihoods
   predictive_residuals = diagnostics.standardized_residuals

Filtering conditions each state on responses available through that occasion,
which supports online tracking. Smoothing adds a vectorized
Rauch--Tung--Striebel backward pass so every state uses the complete response
history. Use ``extended_kalman_filter`` or ``extended_kalman_smoother`` for one
response history.

For a live response stream, update only the newest occasion instead of
refiltering the complete history:

.. code-block:: python

   prior_mean = state_model.initial_mean
   prior_variance = state_model.initial_var
   online_means = []

   for response_vector in responses[0]:
       step = state_model.online_step(
           response_vector,
           prior_mean=prior_mean,
           prior_variance=prior_variance,
       )
       predicted_success = step.response_probabilities
       response_log_score = step.response_log_likelihood
       item_log_scores = step.item_log_likelihoods
       predictive_residuals = step.standardized_residuals
       online_means.append(step.updated_mean)
       prior_mean = step.next_mean
       prior_variance = step.next_variance

``online_step`` predicts marginal item success, scores the joint observed
pattern, updates the state, and prepares the next prior in one quadrature pass.
``online_step_batch`` returns the same fields as arrays for many learners and
accepts either shared scalar priors or one prior per learner. Both results also
include item-level marginal log scores, raw residuals, and Pearson residuals;
missing item diagnostics are ``numpy.nan``.

``extended_kalman_update_batch`` performs the same one-occasion update for
many learners, accepting either shared scalar priors or one prior per learner.
``propagate_state_batch`` advances their state distributions by one or more
occasions. Fully missing response rows leave the supplied priors unchanged.
``state_response_probabilities_batch`` and
``state_response_log_likelihood_batch`` vectorize the corresponding prediction
and scoring steps. Probabilities integrate state uncertainty item by item; the
log score integrates the complete observed response pattern over its shared
state. A fully missing pattern has a zero log score.

Forecasts start one step after the final response occasion and propagate both
state means and process uncertainty. Item-success forecasts integrate the 2PL
or 3PL response curve over each Gaussian forecast distribution rather than
evaluating it only at the mean. The default 21-point quadrature can be adjusted
with ``n_quadpts``. Use ``forecast`` and
``forecast_response_probabilities`` for one response history.

Predictive log likelihoods score each observed item pattern against the state
distribution implied by earlier occasions. Items at the same occasion are
integrated jointly over their shared state, and the pointwise form supports
occasion-level diagnostics. Fully missing occasions contribute zero. These
scores use the extended Kalman approximation and are useful for comparing
state-space configurations on the same histories.

Predictive response probabilities are causal: an occasion uses only the
initial distribution and earlier responses, never its own or future outcomes.
Raw residuals are observed minus predicted success; ``standardized=True``
returns Pearson residuals. Missing responses produce ``numpy.nan`` residuals,
while their model probabilities remain available for inspection.
``predictive_diagnostics_batch`` returns the causal state predictions,
filtered states, probabilities, joint and marginal item log scores, and both
residual forms from one filtering and quadrature pass. Use
``predictive_diagnostics`` for one response history; the separate predictive
methods remain convenient when only one output is needed.

State estimation and forecasting accept only ``1`` (correct), ``0``
(incorrect), and ``-1`` (missing); a fully missing occasion propagates the
predicted state without an observation update. ``observation_noise`` adds
response-scale variance to the linearized filter, reducing the information
assigned to each response. It does not change draws from ``simulate``.

Growth-mixture fitting
----------------------

Fit heterogeneous longitudinal trajectories with a structured result that
includes model-selection criteria, classification entropy, class sizes, and
convergence diagnostics:

.. code-block:: python

   import numpy as np
   from mirt.models import GrowthMixtureModel

   time_values = np.arange(6, dtype=float)
   mixture = GrowthMixtureModel(n_classes=3, n_timepoints=6)
   trajectories, _ = mixture.simulate(500, time_values, seed=42)

   result = mixture.fit(trajectories, time_values)
   print(result.aic, result.bic)
   print(result.class_counts)
   print(result.summary())

``fit_em`` remains available when a mapping is preferred. ``fit`` uses the same
estimation path and packages its final state into ``GrowthMixtureResult``.

Incomplete trajectories
-----------------------

Use ``numpy.nan`` when a person missed a measurement occasion. Likelihoods,
posterior probabilities, classification, entropy, and fitting marginalize over
the missing occasions for each person:

.. code-block:: python

   incomplete = trajectories.copy()
   incomplete[::3, 1] = np.nan
   incomplete[1::4, 4] = np.nan

   result = mixture.fit(incomplete, time_values)
   probabilities = mixture.posterior_probabilities(incomplete, time_values)

Every trajectory must retain at least one observed value. Entirely unobserved
time columns are allowed when the remaining occasions still identify the chosen
growth curve. Infinite values remain invalid. Rows with identical observation
patterns share covariance work, while complete inputs continue through the
optimized complete-data path.

Person-level trajectory prediction
----------------------------------

Predict smoothed latent trajectories on the observed grid or extrapolate them
to new times after fitting:

.. code-block:: python

   future_times = np.arange(0.0, 10.0)
   predicted = mixture.predict_trajectories(
       incomplete,
       time_values,
       prediction_times=future_times,
   )
   print(predicted.shape)  # (500, 10)

   mean, latent_variance = mixture.predict_trajectory_moments(
       incomplete,
       time_values,
       prediction_times=future_times,
   )
   _, observation_variance = mixture.predict_trajectory_moments(
       incomplete,
       time_values,
       prediction_times=future_times,
       include_residual=True,
   )

Predictions combine posterior class probabilities with empirical-Bayes random
intercept and slope estimates conditioned on each person's available history.
They are latent trajectory means, so they exclude new occasion-specific
residual error. Missing occasions are handled directly and need not be imputed
first. For piecewise growth with an implicit changepoint, extrapolation keeps
the midpoint resolved from the observed time grid. The moment method returns
exact mixture means and pointwise variances without assuming the class mixture
is normal; ``include_residual=True`` adds new occasion-specific observation
noise.

For continuous two-segment trajectories, use piecewise growth with a shared
changepoint and class-specific slopes on each side:

.. code-block:: python

   piecewise = GrowthMixtureModel(
       n_classes=2,
       growth_type="piecewise",
       changepoint=3.0,
       class_slopes=np.array([0.2, 0.5]),
       class_post_slopes=np.array([0.8, -0.1]),
   )

Omit ``changepoint`` to use the midpoint of the supplied time range. Piecewise
trajectories remain continuous at the changepoint.
