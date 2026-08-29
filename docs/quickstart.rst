Quick Start Guide
=================

This guide walks through a minimal end-to-end IRT workflow with MIRT.

Basic Usage
-----------

1. **Prepare your data**: Response data should be a 2D NumPy array where rows
   are respondents and columns are items.

2. **Fit a model**: Use ``fit_mirt()`` with an appropriate model type.

3. **Score respondents**: Use ``fscores()`` to estimate person abilities.

4. **Analyze fit**: Use item and person fit diagnostics.

Explore the sample datasets without loading response-matrix copies:

.. code-block:: python

   import mirt

   print(mirt.list_datasets())
   print(mirt.describe_dataset("LSAT7"))

For repeated read-only workflows, ``load_dataset(name, copy=False)`` returns
arrays backed by the process-local cache. Those arrays are intentionally
read-only; omit the argument when writable, independent arrays are needed.

Example: 2PL Model
------------------

.. code-block:: python

   import mirt

   dataset = mirt.load_dataset("LSAT7")
   responses = dataset["data"]

   result = mirt.fit_mirt(responses, model="2PL", max_iter=200)
   print(result.summary())

   print("Discrimination:", result.model.parameters["discrimination"])
   print("Difficulty:", result.model.parameters["difficulty"])

   scores = mirt.fscores(result, responses, method="EAP")
   print("Theta estimates:", scores.theta[:5])
   print("SE estimates:", scores.standard_error[:5])

Example: Graded Response Model
------------------------------

For polytomous items (Likert scales, etc.):

.. code-block:: python

   import mirt

   data = mirt.load_dataset("verbal_aggression")
   responses = data["data"]

   result = mirt.fit_mirt(responses, model="GRM", n_categories=3)
   print(result.summary())

Scoring
-------

After fitting a model, you can score new respondents:

.. code-block:: python

   import mirt

   dataset = mirt.load_dataset("LSAT7")
   responses = dataset["data"]
   result = mirt.fit_mirt(responses, model="2PL")

   new_responses = responses[:10]
   score_result = mirt.fscores(result, new_responses, method="MAP")
   print(score_result.theta)

EAP scoring automatically evaluates large respondent sets in memory-bounded
batches. It detects repetition-heavy response matrices and reuses posterior
calculations when pattern compression is expected to improve throughput. Use
``batch_size`` when a deployment needs an explicit upper bound on the number of
response rows evaluated at once:

.. code-block:: python

   score_result = mirt.fscores(
       result,
       new_responses,
       method="EAP",
       batch_size=5_000,
   )

Smaller batches reduce peak working memory; larger batches can improve throughput
on systems with ample memory. Compressed rows are expanded back to their original
respondent order without changing estimates or standard errors.

Model Fit
---------

Evaluate model fit with various diagnostics:

.. note::

   ``itemfit()`` and ``personfit()`` return DataFrame objects. Install either
   pandas or polars (for example, ``pip install mirt[pandas]``) to use these
   outputs.

.. code-block:: python

   import mirt

   dataset = mirt.load_dataset("LSAT7")
   responses = dataset["data"]
   result = mirt.fit_mirt(responses, model="2PL")

   item_stats = mirt.itemfit(result, responses, statistics=["infit", "outfit"])
   person_stats = mirt.personfit(
       result, responses, statistics=["infit", "outfit", "Zh"]
   )
   print(item_stats.head())
   print(person_stats.head())
