Installation
============

Requirements
------------

* Python 3.11 or higher
* NumPy >= 1.24
* SciPy >= 1.9

Install from PyPI
-----------------

The easiest way to install MIRT is via pip:

.. code-block:: bash

   pip install mirt

This will install pre-built wheels for most platforms (Linux, macOS, Windows).

Install from Source
-------------------

To install from source, you'll need the Rust toolchain:

1. Install Rust: https://rustup.rs/

2. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/Cameron-Lyons/mirt.git
      cd mirt

3. Install with pip:

   .. code-block:: bash

      pip install -e .

Optional Dependencies
---------------------

For pandas DataFrame support:

.. code-block:: bash

   pip install mirt[pandas]

For polars DataFrame support:

.. code-block:: bash

   pip install mirt[polars]

For plotting and embedded report visualizations:

.. code-block:: bash

   pip install "mirt[plot]"

For GPU acceleration and tensor likelihood benchmarks:

.. code-block:: bash

   pip install "mirt[gpu]"

GPU EM supports unidimensional 1PL, 2PL, 3PL, GRM, GPCM, and PCM models.
Category models can have different category counts for each item. Tensor
functions also run on CPU when CUDA is unavailable; EM uses its ordinary
CPU path in that case.

EM and GVEM use an available GPU under the default ``mirt.set_backend("auto")``
(and ``"gpu"``). ``mirt.set_backend("rust")`` and ``mirt.set_backend("numpy")``
keep them on the CPU. An explicit ``use_gpu=True`` or ``use_gpu=False``, on
``fit_mirt`` or an estimator, overrides the preference for that fit. A CPU-only
PyTorch build is recognized from its version file, so it is not imported to
look for a GPU.

For development:

.. code-block:: bash

   pip install mirt[dev]
