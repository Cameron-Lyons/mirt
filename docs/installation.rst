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

For development:

.. code-block:: bash

   pip install mirt[dev]
