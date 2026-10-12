Latent Class Analysis
=====================

A latent class model assumes that every respondent belongs to one of ``K``
unobserved classes and that the item responses are independent within a
class. Class ``k`` has a proportion ``pi_k`` and, for every item ``j``, its
own probabilities of the response categories:

.. math::

   P(\mathbf{x}_i) = \sum_{k=1}^{K} \pi_k \prod_{j} P(X_j = x_{ij} \mid k).

The category probabilities are unconstrained, so the model applies to binary
items (latent class analysis) and to items with any number of unordered
categories alike. It is the model of R mirt's ``mdirt()`` with its default
``itemtype = "lca"``: :func:`~mirt.models.latent_class.fit_lca` estimates it
by expectation maximization, and :func:`~mirt.models.latent_class.mdirt` is
an alias with R's name.

Fitting a model
---------------

Pass the responses and the number of classes:

.. code-block:: python

   import mirt

   responses = mirt.load_dataset("LSAT7")["data"]
   result = mirt.fit_lca(responses, n_classes=2)

   print(result.summary())
   result.log_likelihood, result.aic, result.bic, result.sabic
   result.class_proportions          # largest class first
   result.item_probabilities[0]      # P(X_1 = c | k), shape (2, 2)
   result.class_posteriors[:3]       # posterior class probabilities
   result.classify()[:3]             # most probable (MAP) class
   result.entropy                    # relative entropy, 0 to 1

``result`` is an :class:`~mirt.models.latent_class.LCAResult`. Its ``model``,
a :class:`~mirt.models.latent_class.LatentClassModel`, holds the estimates.
``n_parameters`` is ``K - 1 + K * sum_j (C_j - 1)`` for ``C_j`` categories of
item ``j``. ``entropy`` is the relative entropy
``1 - sum_i sum_k -w_ik log w_ik / (n log K)`` of the posterior class
probabilities ``w``; values near 1 indicate well separated classes, and it is
``NaN`` for one class.

On this LSAT7 example the two-class maximum log-likelihood is -2660.2968,
the value of R's ``mdirt(expand.table(LSAT7), 2)``; at that maximum the
class proportions, item probabilities, posteriors and standard errors agree
with R's to about ``1e-4``.

Class order
-----------

Latent class labels are arbitrary: permuting the classes leaves the
likelihood unchanged. ``fit_lca`` orders the classes by decreasing estimated
proportion, so class 0 is the largest and every array indexed by class
follows that order. R numbers its classes by its starting values, so R's
class numbers may be a permutation of these.

Starts and local maxima
-----------------------

Latent class likelihoods often have several local maxima. The first start
uses R mirt's deterministic ``mdirt`` starting values; ``n_starts`` adds
random starts, like R's ``nruns``, and the start with the largest
log-likelihood is returned:

.. code-block:: python

   result = mirt.fit_lca(responses, n_classes=3, n_starts=20, seed=1)
   result.start_log_likelihoods      # final log-likelihood of every start

A best log-likelihood reached by several starts is more likely the global
maximum. R's starting values give every nonzero category of an item the
same logit, which suits ordered binary items but can leave polytomous fits
in a poor local maximum, so use several starts for polytomous items.

EM stops when the log-likelihood changes by less than ``tol`` (``1e-4``) in
absolute value between iterations, or after ``max_iter`` (500) iterations;
``result.converged`` is ``False`` in the second case. R instead stops when
no parameter changes by more than its ``TOL``, so the two can stop at
slightly different points of a flat likelihood; a smaller ``tol`` brings
both to the maximum.

Choosing the number of classes
------------------------------

Fit a sequence of models and compare their information criteria;
:func:`~mirt.compare_models` accepts the results:

.. code-block:: python

   fits = [
       mirt.fit_lca(responses, n_classes=k, n_starts=10, seed=1)
       for k in range(1, 5)
   ]
   mirt.compare_models(fits, criteria=["AIC", "BIC", "SABIC"])

``bic`` and ``sabic`` use the number of respondents with at least one
observed response; ``sabic`` replaces ``n`` by ``(n + 2) / 24``. A model
with more classes than the data identify has a singular information matrix;
``fit_lca`` then warns and reports ``NaN`` standard errors.

Polytomous items and missing responses
--------------------------------------

Code the responses of item ``j`` as ``0, ..., C_j - 1``. Negative values and
``NaN`` (including pandas and polars nulls) are missing and leave the
likelihood unchanged; rows without any observed response receive the class
proportions as their posterior and, as in R, do not count as observations.
Unique DataFrame column names become the item names:

.. code-block:: python

   import numpy as np

   data = mirt.load_dataset("Bock1997")["data"].astype(float)
   data[::17, 2] = np.nan

   result = mirt.fit_lca(data, n_classes=3, n_starts=10, seed=2)
   result.item_probabilities[2]       # shape (3, 4)

Category counts are inferred from each item's largest code. Pass
``n_categories`` (one count or one per item) to declare more categories; a
category that no respondent uses then has probability 0. R recodes the
observed categories consecutively instead, so recode such items to match
R's parameter count.

Boundary estimates and smoothing
--------------------------------

Maximum-likelihood category probabilities can reach 0 or 1, for example in
a small class. ``smoothing`` adds a pseudo-count to every category count of
every class, which makes the estimates posterior modes under a symmetric
Dirichlet prior with concentration ``1 + smoothing``:

.. code-block:: python

   smoothed = mirt.fit_lca(responses, n_classes=3, smoothing=0.5)

The default ``smoothing=0`` gives maximum-likelihood estimates, as R's
``mdirt``, whose ``lca`` items have no prior. With smoothing, EM maximizes
the log-posterior, while ``log_likelihood`` and the information criteria
still report the log-likelihood.

Standard errors
---------------

``class_proportions_se`` and ``item_probabilities_se`` hold standard errors
on the probability scale. They come from the exact observed information
(Louis' identity) in a baseline-logit parameterization, including the
prior's curvature when ``smoothing > 0``, mapped to probabilities by the
delta method; they match R's ``SE = TRUE`` (Oakes) errors transformed the
same way. Estimates within ``1e-8`` of 0 or 1 have no Wald standard error
and are ``NaN``. Pass ``compute_standard_errors=False`` to skip them.

Scoring new respondents and simulating
--------------------------------------

``posterior`` and ``classify`` score new responses with the fitted class
proportions as the prior; DataFrame columns are matched to the item names:

.. code-block:: python

   new = responses[:5]
   result.posterior(new)
   result.classify(new)

A :class:`~mirt.models.latent_class.LatentClassModel` can also be built from
known parameters, for example to simulate data:

.. code-block:: python

   from mirt import LatentClassModel

   truth = LatentClassModel(
       [0.6, 0.4],
       [np.array([[0.9, 0.1], [0.3, 0.7]])] * 6,
   )
   simulated, classes = truth.simulate(1000, seed=3)
   mirt.fit_lca(simulated, n_classes=2).class_proportions

Relation to R's mdirt
---------------------

=================================  ==============================================
R mirt                             mirt
=================================  ==============================================
``mdirt(data, K)``                 ``fit_lca(data, K)`` or ``mdirt(data, K)``
``nruns``                          ``n_starts`` (plus ``seed``)
``coef(fit)``, ``summary(fit)``    ``item_probabilities``, ``summary()``
``fscores(fit)``                   ``class_posteriors``, ``posterior()``
``SE = TRUE``                      ``compute_standard_errors=True``
=================================  ==============================================

Only the unconstrained model is available: R's ``customTheta``,
``structure`` (log-linear class structures), ``item.Q``, latent regression
(``covdata`` and ``formula``), multiple groups and custom item types are
not supported.
