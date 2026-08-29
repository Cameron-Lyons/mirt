Multilevel Models
=================

Hierarchical IRT models attach a normal ability distribution to each stored
group or level combination. The model can evaluate conditional likelihoods at
known abilities and can also integrate those abilities out for model comparison
or parameter-search objectives.

Two-level likelihoods
---------------------

Create a base item model, assign each respondent to a group, and set the
current group effects and variance components:

.. code-block:: python

   import numpy as np

   from mirt import TwoParameterLogistic
   from mirt.models import MultilevelIRTModel

   base = TwoParameterLogistic(n_items=3)
   groups = np.array([10, 10, 25, 25])
   model = MultilevelIRTModel(base, groups)
   model.set_group_means(np.array([-0.4, 0.6]))
   model.set_variance_components(between=0.3, within=0.8)

   responses = np.array(
       [
           [1, 0, 1],
           [0, 1, 1],
           [1, 1, 0],
           [0, -1, 1],
       ]
   )
   person_loglik = model.marginal_log_likelihoods(responses, n_quadpts=21)
   total_loglik = model.marginal_log_likelihood(responses, n_quadpts=21)

Negative response codes are treated as missing. The plural method returns one
integrated response-pattern value per respondent; the singular method returns
their sum.

Three-level likelihoods
-----------------------

``ThreeLevelIRTModel`` uses the sum of the current level-2 and level-3 effects
as each person's prior mean. Its two marginal-likelihood methods have the same
interface and also support unidimensional polytomous base models.

Both implementations reuse a quadrature evaluation for respondents with the
same prior mean. Large groups are processed in bounded chunks; pass
``chunk_size`` to impose a smaller explicit row bound when needed.
