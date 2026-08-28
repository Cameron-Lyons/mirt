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
