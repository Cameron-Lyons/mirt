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
The unipolar log-logistic model's ``theta`` is the log of Lucke's unipolar
trait :math:`\tau`, so its information is about :math:`\log \tau`; dividing
by :math:`\tau^2` gives the information about :math:`\tau` that R mirt reports
for ``itemtype = "ULL"``.

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

Every information and reliability helper accepts the ``FitResult`` itself as
well as ``fit.model``.

For multidimensional models, supply one row per ability point and one column
per factor. A vector of length ``model.n_factors`` represents a single point.
``iteminfo`` and ``expected_score`` accept a scalar integer item index or a
one-dimensional selection, including NumPy arrays. Selections preserve their
order and repeated indices; an empty selection returns zero columns.

Multidimensional information
----------------------------

Fisher information about several factors is a matrix :math:`I(\theta)`. By
default ``testinfo`` and ``iteminfo`` return its trace, the sum over factors
of the information along each factor axis. As in R, ``degrees`` gives one
angle per factor between a direction and each axis; the result is the
information along that direction, :math:`u' I(\theta) u` with
:math:`u = \cos(\text{degrees})` scaled to unit length. ``return_matrix=True``
returns the matrices themselves (R's ``multidim_matrix = TRUE``):

.. code-block:: python

   two_factor = mirt.fit_mirt(responses, model="2PL", n_factors=2)
   points = np.array([[0.0, 0.0], [1.0, -0.5]])

   along_first = mirt.testinfo(two_factor, points, degrees=[0, 90])
   along_diagonal = mirt.testinfo(two_factor, points, degrees=[45, 45])
   matrices = mirt.testinfo(two_factor, points, return_matrix=True)
   item_matrices = mirt.iteminfo(two_factor, points, item_idx=0, return_matrix=True)

``degrees=[0, 90]`` gives the diagonal element :math:`I_{11}(\theta)`, the
curve that ``plot_information(..., factor=0)`` draws.

Area under the information curve
--------------------------------

``areainfo`` integrates test or item information over ``theta_range`` with
the trapezoidal rule. ``return_details=True`` adds the quantities of R's
``areainfo``: the information over the whole real line, the proportion of it
inside the range and the number of items integrated.

.. code-block:: python

   details = mirt.areainfo(fit, theta_range=(-2.0, 2.0), return_details=True)
   print(f"{details['proportion']:.1%} of {details['total_info']:.2f}")

Inspect conditional reliability
-------------------------------

``conditional_rxx`` converts total information into a reliability profile
over the supplied ability values. It uses a standard-normal reference
variance by default, or the estimated latent variance of a ``FitResult``
that has one, and accepts another positive latent variance when the target
population is more or less dispersed.

.. code-block:: python

   reliability = mirt.conditional_rxx(fit.model, theta)
   wider_population = mirt.conditional_rxx(
       fit.model,
       theta,
       latent_variance=1.5,
   )

The returned array aligns with ``theta``. Reliability is zero wherever test
information is zero and approaches one as information increases.

Empirical reliability
---------------------

``empirical_rxx`` estimates reliability from scored respondents, as R's
``empirical_rxx``. Pass the ``ScoreResult`` of :func:`mirt.fscores`, which
carries the standard errors and the scoring method:

.. code-block:: python

   eap = mirt.fscores(fit, responses, method="EAP")
   ml = mirt.fscores(fit, responses, method="ML")

   eap_reliability = mirt.empirical_rxx(fit, eap)
   ml_reliability = mirt.empirical_rxx(fit, ml)

EAP, MAP and EAPsum scores are shrunken, so their variance estimates the
true-score variance and their standard errors are posterior standard
deviations; reliability is ``Var(theta_hat) / (Var(theta_hat) + E[SE^2])``,
R's default. The variance of ML and WLE scores already contains the error, so
reliability is ``1 - E[SE^2] / Var(theta_hat)``, R's ``T_as_X = TRUE``; rows
with infinite estimates or errors, such as ML scores of perfect patterns, are
dropped as in R. ``scores_include_error`` overrides the choice.

Without standard errors, ``method="information"`` derives them as
``1 / sqrt(I(theta_hat))``. These are the asymptotic errors of ML estimates,
so this method suits ML and WLE scores only; an EAP array without its
standard errors cannot give the right coefficient, and the call warns.

Classical test statistics
-------------------------

``traditional`` reports Cronbach's alpha, alpha if each item is deleted,
standardized alpha, the standard error of measurement and corrected
item-total correlations, as R's ``itemstats``, for dichotomous and polytomous
items alike. ``difficulty`` holds the mean item score (the proportion correct
for binary items) and ``item_sd`` the item standard deviations.

.. code-block:: python

   ctt = mirt.traditional(responses)
   print(f"alpha = {ctt.alpha:.3f}, SEM = {ctt.sem_alpha:.3f}")

   pairwise = mirt.traditional(responses, missing="pairwise")

Missing responses are never counted as zero. The default
``missing="complete"`` matches R: alpha and the total-score mean and standard
deviation use the respondents who answered every item, and alpha if item
``j`` is deleted uses those who answered every other item. Complete cases can
be few, so ``missing="pairwise"`` instead derives every statistic from the
pairwise-complete item covariance matrix, which uses all responses and suits
planned-missing designs. The two agree on complete data.

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
