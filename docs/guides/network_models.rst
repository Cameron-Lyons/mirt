Network Models
==============

The network module estimates conditional relationships between observed
variables. It includes a binary Ising model and a Gaussian graphical model,
using only the project's core numerical dependencies.

Binary Ising networks
---------------------

``fit_ising`` estimates thresholds and one symmetric parameter per edge by
maximizing the joint pseudolikelihood. The L1 penalty controls sparsity and
does not shrink node thresholds.

.. code-block:: python

   import numpy as np
   from mirt.models import fit_ising

   rng = np.random.default_rng(42)
   responses = rng.integers(0, 2, size=(1_000, 12))

   model, log_pseudolikelihood = fit_ising(
       responses,
       regularization=0.02,
       max_iter=500,
       tol=1e-6,
       node_names=[f"Item {index + 1}" for index in range(12)],
   )

   print(model.converged, model.n_iterations)
   print(model.thresholds)
   print(model.interactions)

The optimizer updates all unique symmetric edges together. Its penalized
objective history is available through ``model.objective_history``. Binary
inputs, callback widths, parameter shapes, and finite values are validated
before numerical work begins.

Conditional and exact probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Full-conditional probabilities are evaluated for every observation and node
in one call.

.. code-block:: python

   conditional = model.conditional_probabilities(responses[:20])
   # shape: (20, 12)

For small networks, normalized joint probabilities can be computed by exact
state enumeration.

.. code-block:: python

   from mirt.models import IsingModel

   small = IsingModel(3)
   states = np.array([
       [0, 0, 0],
       [0, 0, 1],
       [0, 1, 0],
       [0, 1, 1],
       [1, 0, 0],
       [1, 0, 1],
       [1, 1, 0],
       [1, 1, 1],
   ])
   probabilities = small.probability(states)
   assert np.isclose(probabilities.sum(), 1)

Exact enumeration is guarded at 16 nodes by default because its cost doubles
with each additional node. Set ``max_nodes`` explicitly only when that cost is
acceptable. The same enumerated distribution can produce independent samples,
which avoids burn-in and autocorrelation for small networks.

.. code-block:: python

   independent_draws = small.sample(
       n_samples=2_000,
       method="exact",
       seed=42,
   )

Gibbs sampling remains the default and supports burn-in and thinning for
larger networks.

.. code-block:: python

   draws = model.sample(
       n_samples=2_000,
       n_burnin=500,
       thin=2,
       seed=42,
   )

Gaussian graphical models
-------------------------

``fit_ggm`` estimates a positive-definite precision matrix. Off-diagonal
precision entries receive an L1 penalty; diagonal entries remain unpenalized.
The resulting partial correlations represent conditional relationships.

.. code-block:: python

   from mirt.models import fit_ggm

   observations = rng.multivariate_normal(
       mean=np.zeros(6),
       cov=np.eye(6),
       size=500,
   )
   gaussian, log_likelihood = fit_ggm(
       observations,
       regularization=0.05,
       max_iter=500,
       tol=1e-6,
   )

   partial_correlations = gaussian.partial_correlations()
   covariance = gaussian.covariance_matrix

Regularized estimation uses an adaptive alternating-direction solver with a
positive-definite precision update. This also supports singular sample
covariances, including settings with more nodes than observations. An
unregularized fit instead raises a clear error when the sample covariance has
no inverse.

Comparing networks
------------------

``compare_networks`` compares only unique edges, excluding the diagonal and
unused lower triangle. Models with the same names in a different order are
aligned automatically.

.. code-block:: python

   from mirt.models import compare_networks

   comparison = compare_networks(model_a, model_b, edge_threshold=1e-6)
   print(comparison["edge_correlation"])
   print(comparison["mean_edge_difference"])
   print(comparison["edge_jaccard"])

References
----------

The Ising estimator follows the maximum-pseudolikelihood formulation described
by `Chatterjee (2007) <https://projecteuclid.org/journals/annals-of-statistics/volume-35/issue-5/Estimation-in-spin-glasses-A-first-step/10.1214/009053607000000109.full>`_.
The sparse precision objective follows `Friedman, Hastie, and Tibshirani (2008)
<https://pmc.ncbi.nlm.nih.gov/articles/PMC3019769/>`_.
