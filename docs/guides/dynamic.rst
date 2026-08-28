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

Online updates
--------------

For a live trial stream, retain one next-opportunity prior per skill instead of
replaying the complete history:

.. code-block:: python

   mastery_priors = model.p_init.copy()
   latest_mastery = model.p_init.copy()

   for response, skill_idx in zip(responses, skills):
       step = model.online_step(
           int(response),
           int(skill_idx),
           prior_mastery=mastery_priors[skill_idx],
       )
       predicted_success = step.response_probability
       response_log_score = step.response_log_likelihood
       predictive_residual = step.standardized_residual
       latest_mastery[skill_idx] = step.updated_mastery
       mastery_priors[skill_idx] = step.next_mastery

``online_step`` applies response evidence and prepares the assigned skill for
its next opportunity in constant history space. ``online_step_batch`` performs
the same update for many learners; skill assignments and prior mastery may be
shared scalars or one value per learner. Missing responses have zero log score
and ``numpy.nan`` residuals while still applying the configured learning and
forgetting transition.

History diagnostics
-------------------

Compute the complete causal record in one pass when a trial history is already
available:

.. code-block:: python

   diagnostics = model.predictive_diagnostics(responses, skills)
   causal_mastery = diagnostics.predicted_mastery
   filtered_mastery = diagnostics.updated_mastery
   predictive_success = diagnostics.response_probabilities
   predictive_log_scores = diagnostics.response_log_likelihoods
   predictive_residuals = diagnostics.standardized_residuals
   total_log_score = diagnostics.total_log_likelihood
   latest_mastery = diagnostics.latest_mastery_by_skill
   next_priors = diagnostics.next_mastery_priors

   future = model.forecast_from_priors(next_priors, n_steps=6)

Predicted mastery and success at a trial use only earlier opportunities for
the assigned skill. Updated mastery additionally conditions on the current
response, and ``next_mastery`` includes the learning and forgetting transition
for that skill's next opportunity. ``predictive_diagnostics_batch`` performs
the same sequential calculation across many learners at once and supports
shared or person-specific skill layouts. Missing trials retain causal mastery,
contribute zero log score, and produce ``numpy.nan`` residuals. Both result
types retain final per-skill posteriors and next-opportunity priors, so a
forecast can continue without replaying the history.

Adaptive skill ranking
----------------------

Rank the next skill opportunity directly from retained mastery priors:

.. code-block:: python

   ranking = model.rank_skills(
       diagnostics.next_mastery_priors,
       criterion="information_gain",
       top_k=2,
   )
   next_skill = ranking.best_skill_index
   next_skill_name = model.skill_names[next_skill]

``information_gain`` measures the exact mutual information, in nats, between
the latent mastery state and the next binary response. It favors opportunities
whose outcome is expected to be most diagnostic. Other objectives prioritize
expected net ``mastery_gain``, ``lowest_mastery`` for remediation, or the
highest ``success_probability``. Pass ``available_skills`` to restrict the
candidate set. Equal scores are ordered by the lower skill index.

``rank_skills_batch`` evaluates an ``(n_persons, n_skills)`` matrix together
and returns one ranked row per learner. The retained priors from
``predictive_diagnostics`` or ``predictive_diagnostics_batch`` can be passed
directly to the ranking methods, without replaying response histories.

Mastery forecasts
-----------------

Project mastery and success probabilities directly from the retained
next-opportunity priors:

.. code-block:: python

   future = model.forecast_from_priors(mastery_priors, n_steps=6)
   future_mastery = future.mastery_probabilities
   future_success = future.response_probabilities
   print(future_mastery.shape)  # (6, 3)

When retained priors are unavailable, forecast from a response history in one
call:

.. code-block:: python

   future = model.forecast(responses, skills, n_steps=6)
   next_priors = model.next_mastery_priors(responses, skills)

``forecast_from_priors_batch`` accepts an
``(n_persons, n_skills)`` prior matrix, while ``forecast_batch`` accepts shared
or person-specific skill layouts. Each forecast step represents one future
opportunity for every modeled skill. Forecasts are unconditional on unknown
future responses and use a closed-form transition, so their runtime does not
include a Python loop over the horizon. Skills absent from a response history
begin at their configured initial mastery probability.

Mastery targets
---------------

Calculate the minimum additional opportunities needed for each skill's
expected mastery probability to reach a target:

.. code-block:: python

   progress = model.opportunities_to_mastery(
       diagnostics.next_mastery_priors,
       target_mastery=0.9,
   )
   practice_counts = progress.opportunities
   reachable = progress.reachable

A count of zero means the retained prior already meets the target.
``numpy.inf`` explicitly marks a target that cannot be reached under the
model's unconditional transition path, including a target equal to a limiting
probability that is approached only asymptotically. This avoids choosing an
arbitrary forecast horizon or mistaking horizon exhaustion for
unreachability.

``opportunities_to_mastery_batch`` evaluates every learner and skill together.
It accepts a shared scalar target, a shared ``(n_skills,)`` vector, or a
person-specific ``(n_persons, n_skills)`` matrix. Both methods solve the
transition recurrence directly, so memory and runtime do not grow with the
number of opportunities returned. The calculation describes expected mastery
before future responses are known; new evidence can be incorporated with an
online update and the target recomputed.

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

Terminal helpers such as ``predict_mastery_batch`` and
``next_mastery_priors_batch`` avoid backward smoothing when a compiled shared
layout is unavailable. They update all learners together in one causal pass
and retain only the current per-skill states.

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

   future = state_model.forecast_summary_batch(
       responses,
       n_steps=4,
   )
   future_means = future.state_means
   future_variances = future.state_variances
   future_success = future.response_probabilities
   future_lower, future_upper = future.state_interval(confidence=0.95)
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

   future = state_model.forecast_summary_from_state(
       step.updated_mean,
       step.updated_variance,
       n_steps=4,
   )
   future_success = future.response_probabilities

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
``forecast_response_probabilities`` for one response history. When the current
posterior state is already available, ``forecast_from_state`` and
``forecast_response_probabilities_from_state`` produce the same multi-step
forecasts without replaying earlier responses. Their ``_batch`` variants
accept one current state per learner. ``forecast_summary`` and
``forecast_summary_from_state`` combine latent moments and response
probabilities in one result, and provide dependency-free Gaussian state
intervals. Use their ``_batch`` variants for multiple learners.

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
