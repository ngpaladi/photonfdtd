Installation
============

From PyPI
---------

The straightforward way:

.. code-block:: bash

   pip install photonfdtd

The wheel is pure Python. It depends on NumPy, SciPy, Matplotlib and JAX (the
CPU ``jaxlib``, which runs the time loop). Everything that follows on this page
is optional.

From source
-----------

If you plan to poke at the code, install from a checkout:

.. code-block:: bash

   git clone https://github.com/ngpaladi/photonfdtd
   cd photonfdtd
   pip install -e .

Installing straight from GitHub works too:

.. code-block:: bash

   pip install "git+https://github.com/ngpaladi/photonfdtd"

Optional extras
---------------

.. code-block:: bash

   pip install -e ".[test]"             # pytest, for the test suite
   pip install -e ".[docs]"             # sphinx + the Read the Docs theme, to build these docs

photonfdtd needs Python 3.10+. The ``gdsfactory`` adapter is imported lazily,
so it costs you nothing until you actually reach for it.

GPU acceleration
----------------

Install a CUDA-enabled ``jaxlib`` and the forward run *and* the differentiable
and reversible adjoints run on the GPU through XLA. Your code doesn't change:

.. code-block:: bash

   pip install "jax[cuda12]"   # NVIDIA / CUDA 12 GPU

This is validated on an NVIDIA RTX 4080, with both adjoints running on-device.
If JAX quietly falls back to CPU with a "cuSPARSE not found"-style error, the
CUDA libraries just aren't on the loader path yet. Point ``LD_LIBRARY_PATH`` at
the ``nvidia/*/lib`` directories the ``jax[cuda12]`` install dropped in your
site-packages and it'll find them. Devices, precision and memory planning are
covered on the :doc:`backends` page.
