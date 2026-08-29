Item and test information
=========================

Information curves show where a fitted model measures ability most precisely.
For a dichotomous item with response probability :math:`P(\theta)`, Fisher
information is computed from the analytic probability derivative:

.. math::

   I(\theta) = \frac{[P'(\theta)]^2}{P(\theta)[1-P(\theta)]}.

The analytic implementation covers logistic, asymmetric logistic, unipolar,
complementary-log-log, and negative-log-log item curves. It remains finite in
extreme tails where a curve has already reached a representable asymptote.

Inspect fitted information
--------------------------

.. code-block:: python

   import numpy as np
   import mirt

   responses = mirt.load_dataset("LSAT7")["data"]
   fit = mirt.fit_mirt(responses, model="3PL")
   theta = np.linspace(-4.0, 4.0, 161)

   item_information = mirt.iteminfo(fit.model, theta)
   test_information = mirt.testinfo(fit.model, theta)
   most_precise_theta = theta[np.argmax(test_information)]

``item_information`` has one column per item. ``test_information`` is their
row-wise sum and can be converted to conditional standard error with
``1 / np.sqrt(test_information)`` wherever information is positive.

Inspect conditional reliability
-------------------------------

``conditional_rxx`` converts total information into a reliability profile
over the supplied ability values. It uses a standard-normal reference
variance by default and accepts another positive latent variance when the
target population is more or less dispersed.

.. code-block:: python

   reliability = mirt.conditional_rxx(fit.model, theta)
   wider_population = mirt.conditional_rxx(
       fit.model,
       theta,
       latent_variance=1.5,
   )

The returned array aligns with ``theta``. Reliability is zero wherever test
information is zero and approaches one as information increases.

Single-item curves
------------------

Pass an item index to avoid allocating the complete item matrix:

.. code-block:: python

   first_item = fit.model.information(theta, item_idx=0)

This path uses the same analytic formula as the all-item calculation.

Find target-precision ranges
----------------------------

``information_intervals`` finds every ability range where an item or complete
test meets a chosen Fisher-information target. The result retains disjoint
regions, which is useful for forms with separated information peaks.

.. code-block:: python

   # Conditional SEM <= 0.5 requires information >= 1 / 0.5**2.
   coverage = mirt.information_intervals(
       fit.model,
       min_information=1 / 0.5**2,
       theta_range=(-4.0, 4.0),
   )

   # Restrict the calculation to the first item.
   item_coverage = mirt.information_intervals(
       fit.model,
       min_information=1.0,
       item_idx=0,
   )

Each row contains the lower and upper theta boundary for one covered interval.
An empty ``(0, 2)`` array means the target is not reached within the requested
range. Increase ``n_points`` when a test may have extremely narrow peaks; the
detected boundaries are refined after the grid search.
