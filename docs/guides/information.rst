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

Single-item curves
------------------

Pass an item index to avoid allocating the complete item matrix:

.. code-block:: python

   first_item = fit.model.information(theta, item_idx=0)

This path uses the same analytic formula as the all-item calculation.
